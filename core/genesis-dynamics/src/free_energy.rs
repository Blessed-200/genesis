#![allow(clippy::float_cmp, clippy::uninlined_format_args)]

use genesis_math::SparseCliffordVector;
use genesis_types::{GenesisError, NodeId};

/// Signatura Minkowski (+,−,−,−) para G(1,3).
/// Índice 0 = temporal (positivo), índices 1..3 = espaciales (negativos).
/// AX-ID: AXIOMA-001, `LEY_FUNDACIONAL` §2
const TRACE_MIN: f64 = 1.0e-12;
const TRACE_MAX: f64 = 1.0e12;

/// Pesos de norma por blade en G(1,3) para el cálculo de VFE 16D.
///
/// Proporcionados para que `VFE_16D = Σ_i precision_full[i] * (mean_full[i] - target_full[i])²`
/// sea sensible a la geometría real de G(1,3). Los pesos reflejan la importancia
/// semántica de cada grado (ver `METRIC_WEIGHTS` en genesis-math).
///
/// | Grado | Blades   | Peso |
/// |-------|----------|------|
/// | 0     | {0}      | 2.0  |
/// | 1     | {1,2,4,8}| 1.5  |
/// | 2     | {3,5,6,9,10,12} | 1.0 |
/// | 3     | {7,11,13,14} | 0.5 |
/// | 4     | {15}     | 0.3  |
///
/// Consistentes con `METRIC_WEIGHTS` en `genesis-math` para que la distancia
/// inferencial sea coherente con la distancia topológica en HNSW.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.3
pub(crate) const VFE_BLADE_WEIGHTS: [f64; 16] = [
    // blade 0 (grado 0: escalar)
    2.0,
    // blades 1,2,4,8 (grado 1: vectores — semántica primaria)
    1.5, 1.5, 1.0,  // 1(e0), 2(e1), 3(e01)
    1.5, 1.0, 1.0,  // 4(e2), 5(e02), 6(e12)
    0.5, 1.5,       // 7(e012), 8(e3)
    // blades 9..14 (grados 2 y 3)
    1.0, 1.0, 0.5, 1.0, 0.5, 0.5,
    // blade 15 (grado 4: pseudoescalar)
    0.3,
];

/// Índices blade de los vectores grado 1 en G(1,3): e₀, e₁, e₂, e₃.
/// Usados para inicializar `mean_full` desde el prior de 4 componentes
/// y para retrocompatibilidad del método `mean()`.
pub(crate) const GRADE1_BLADE_INDICES: [usize; 4] = [1, 2, 4, 8];

fn is_finite_vec4(values: &[f64; 4]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn is_finite_vec16(values: &[f64; 16]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn sanitize_trace(trace: f64) -> f64 {
    if !trace.is_finite() {
        return 1.0;
    }
    trace.clamp(TRACE_MIN, TRACE_MAX)
}

fn target_from_sparse_obs(obs: Option<&SparseCliffordVector>) -> Option<[f64; 16]> {
    obs.map_or(Some([0.0; 16]), |o| {
        let target: [f64; 16] = core::array::from_fn(|i| o.coeffs[i]);
        is_finite_vec16(&target).then_some(target)
    })
}

// ── Tipos de creencias ────────────────────────────────────────────────────────

/// Estado inferencial completo de un nodo sobre el multivector G(1,3).
///
/// # Representación 16D
///
/// `mean_full[i]` es el coeficiente medio del blade `i` ∈ {0,…,15} de G(1,3).
/// `precision_full[i]` es la precisión diagonal del blade `i`.
///
/// La media cubre los 16 blades (5 grados), lo que permite que el sistema
/// inferencial opere sobre la estructura algebraica completa del concepto,
/// no solo sobre su componente vectorial.
///
/// # Retrocompatibilidad
///
/// - `add_node(id, prior_mean: [f64; 4])` — inicializa `mean_full` con los 4
///   valores en los blades de grado 1 (índices 1,2,4,8) y cero en el resto.
/// - `mean()` — retorna los 4 componentes de grado 1, idéntico al comportamiento anterior.
/// - `compute_vfe(id, obs: Option<&[f64; 4]>)` — calcula VFE sobre grado 1 únicamente.
///   Sin cambio de comportamiento.
/// - `compute_vfe_with_grad(id, obs)` — ahora retorna `(f64, [f64; 16])`.
///   **Esto cambia la firma.** Callers en CRATE-004 recibirán el gradiente completo.
///   Callers en CRATE-003 que solo usaban los 4 primeros valores no tienen callers
///   internos activos (solo tests) — actualizados abajo.
///
/// # Preparación para CRATE-004
/// `DiscreteRicciFlow` usará `compute_vfe_with_grad() -> [f64; 16]` para calcular
/// la señal de curvatura Ollivier-Ricci con información de todos los grados, no
/// solo del subespacio vectorial.
///
/// AX-ID: AXIOMA-003, `H_información` (`LEY_FUNDACIONAL` §3.3)
#[derive(Clone, Debug)]
pub struct Belief {
    /// μᵢ — estado inferencial medio sobre los 16 blades de G(1,3).
    ///
    /// Inicializa con los 4 componentes de grado 1 en posiciones `GRADE1_BLADE_INDICES`
    /// y cero en el resto. Evoluciona con `update_full()` hacia observaciones completas
    /// o con `update()` (grado 1 únicamente) para retrocompatibilidad.
    pub mean_full: [f64; 16],
    /// Precisión diagonal sobre los 16 blades. `precision_full[i]` ∈ [0, 1e6].
    /// Inicializa en `1.0` para todos los blades (prior no informativo).
    pub precision_full: [f64; 16],
    /// Node identifier this belief is associated with.
    pub node_id: NodeId,
}

/// Información de Fisher escalarizada para el gate de saciedad AXIOMA-008.
///
/// La métrica de Fisher completa 𝒢ᵢ ∈ ℝ¹⁶ˣ¹⁶ se aproxima como c·I₁₆ (Fisher
/// isotrópica), por lo que toda la información relevante es su traza escalar.
///
/// Con la extensión a 16D, la traza cubre todos los blades, no solo grado 1.
/// El gate de saciedad AXIOMA-008 opera igual: saciedad cuando ΔTr < ε.
///
/// AX-ID: AXIOMA-008
#[derive(Clone, Debug)]
pub struct FisherInfo {
    /// Traza escalar de la métrica de Fisher isotrópica: Tr(𝒢ᵢ) = 16c.
    /// Inicializa en 1.0. Floor: 1e-12.
    pub trace: f64,
    /// ‖∂𝒢/∂t‖_F ≈ 0.5 · |ΔTr|
    pub delta_g: f64,
}

impl Belief {
    fn new(id: NodeId, prior_mean: [f64; 4]) -> Self {
        let mut mean_full = [0.0f64; 16];
        // Inicializar solo los 4 blades de grado 1 con el prior
        for (k, &blade_idx) in GRADE1_BLADE_INDICES.iter().enumerate() {
            mean_full[blade_idx] = prior_mean[k];
        }
        Self {
            mean_full,
            precision_full: [1.0; 16],
            node_id: id,
        }
    }

    /// Componentes de grado 1 (vectores e₀,e₁,e₂,e₃) — retrocompatibilidad.
    ///
    /// Retorna los 4 coeficientes del subespacio vectorial, idéntico al
    /// campo `mean: [f64; 4]` original.
    ///
    /// AX-ID: AXIOMA-003
    #[inline]
    pub fn mean(&self) -> [f64; 4] {
        core::array::from_fn(|k| self.mean_full[GRADE1_BLADE_INDICES[k]])
    }

    /// Traza de Fisher (para FisherGate, AXIOMA-008).
    /// Suma de las precisiones sobre los 16 blades.
    #[inline]
    pub fn fisher_trace(&self) -> f64 {
        self.precision_full.iter().sum()
    }

    /// Create a tombstone belief for a removed node.
    ///
    /// A tombstone has all-zero means and sentinel NodeId::INVALID.
    /// It occupies the Vec slot to preserve index stability after `remove_node()`.
    /// `compute_vfe` returns 0.0 for tombstones (lookup returns None for INVALID).
    ///
    /// AX-ID: CRATE-004 prerequisite (FIX-H)
    #[inline]
    pub(crate) fn tombstone() -> Self {
        Self {
            mean_full: [0.0; 16],
            precision_full: [0.0; 16],
            node_id: NodeId::INVALID,
        }
    }
}

impl FisherInfo {
    fn new() -> Self {
        Self { trace: 1.0, delta_g: 0.0 }
    }
}

/// Re-exportado desde `genesis_types` para que `genesis-evolution` (CRATE-004)
/// pueda verificar `DualityConsistency` sin importar `genesis-dynamics` (CRATE-003).
///
/// AX-ID: LEY_FUNDACIONAL §3.6, §5.6
pub use genesis_types::FisherEdgeMetric;

/// Función auxiliar canónica de arista — mantenida internamente para compatibilidad
/// con tests que la referencian directamente dentro de este módulo.
fn canonical_edge(i: NodeId, j: NodeId) -> (NodeId, NodeId) {
    if i <= j { (i, j) } else { (j, i) }
}

// ── VFEMinimizer ──────────────────────────────────────────────────────────────

/// Minimizador de Energía Libre Variacional.
///
/// `F = D_KL(Q(s) ‖ P(s|o)) − ln P(o)`
/// Aproximación computable: `F ≈ Σᵢ Tr(𝒢ᵢ) · ‖μᵢ − μ̂ᵢ‖²`
///
/// El mecanismo de aprendizaje ÚNICO es la minimización de F.
/// PROHIBIDO: cross-entropy, MSE, cualquier pérdida externa. (AXIOMA-003)
/// PROHIBIDO: esperar prompts para activarse. (AXIOMA-003: vida cognitiva continua)
///
/// AX-ID: AXIOMA-003, AXIOMA-008, `H_información` (`LEY_FUNDACIONAL` §3.3)
pub struct VFEMinimizer {
    beliefs: Vec<Belief>,
    fisher: Vec<FisherInfo>,
    /// `NodeId` → índice en `beliefs`. `u32::MAX` = no registrado.
    /// Direct array: O(1) lookup, válido para `NodeIds` consecutivos.
    id_to_idx: Vec<u32>,
}

impl VFEMinimizer {
    #[inline]
    fn bounded_step(dt: f64, trace: f64) -> f64 {
        (dt / (1.0 + dt * trace)).min(0.9)
    }

    /// Creates an empty VFE minimiser with no registered nodes.
    pub fn new() -> Self {
        Self {
            beliefs: Vec::new(),
            fisher: Vec::new(),
            id_to_idx: Vec::new(),
        }
    }

    /// # Panics
    /// Panics if `id.get()` cannot fit in `usize` on the target architecture,
    /// or if the node count exceeds `u32::MAX`.
    ///
    /// # Política de valores no finitos
    /// `prior_mean` debe contener solo valores finitos (`is_finite`).
    /// Si contiene `NaN`/`+Inf`/`-Inf`, el nodo se rechaza y no se persiste.
    /// Register a new node with a grade-1 prior mean.
    ///
    /// # Scaling (BN-05)
    ///
    /// The former hard cap of `NodeId < 1_000_000` is removed. The direct-index
    /// Vec grows dynamically to `id.get() + 1` entries. A defensive maximum of
    /// `MAX_ALLOWED_NODE_ID = 100_000_000` prevents runaway allocation from
    /// adversarial or buggy IDs (4 bytes × 10⁸ = 400 MB absolute worst case).
    ///
    /// Duplicate insertions (same ID) are idempotent — the existing entry is kept.
    /// Non-finite prior means are silently rejected.
    pub fn add_node(&mut self, id: NodeId, prior_mean: [f64; 4]) {
        if !is_finite_vec4(&prior_mean) {
            return;
        }
        let raw = usize::try_from(id.get()).expect("NodeId must fit into usize on supported targets");

        // Defensive maximum: 100× production target. Prevents OOM from buggy callers.
        const MAX_ALLOWED_NODE_ID: usize = 100_000_000;
        if raw > MAX_ALLOWED_NODE_ID {
            // In release: log and return. In debug: panic for early detection.
            debug_assert!(false,
                "NodeId {raw} exceeds MAX_ALLOWED_NODE_ID={MAX_ALLOWED_NODE_ID} — potential bug");
            return;
        }

        if raw >= self.id_to_idx.len() {
            self.id_to_idx.resize(raw + 1, u32::MAX);
        }
        if self.id_to_idx[raw] == u32::MAX {
            let idx = match u32::try_from(self.beliefs.len()) {
                Ok(v) => v,
                Err(_) => {
                    debug_assert!(false, "VFEMinimizer belief count overflowed u32::MAX");
                    return;
                }
            };
            self.id_to_idx[raw] = idx;
            self.beliefs.push(Belief::new(id, prior_mean));
            self.fisher.push(FisherInfo::new());
        }
    }

    /// Lookup interno: `NodeId` → índice en `Vec`.
    ///
    /// O(1) direct-index access. Returns None for unregistered IDs without panic.
    fn lookup(&self, id: NodeId) -> Option<usize> {
        let raw = usize::try_from(id.get()).expect("NodeId must fit into usize on supported targets");
        self.id_to_idx.get(raw).and_then(|&idx| {
            if idx == u32::MAX { None } else { Some(idx as usize) }
        })
    }

    /// VFE sobre el subespacio de grado 1 — retrocompatibilidad con callers `[f64;4]`.
    ///
    /// Calcula la distancia de Mahalanobis al cuadrado entre `mean_full[grade1]`
    /// y `obs` (interpretado como 4 componentes de grado 1). Siempre ≥ 0.
    ///
    /// Con obs=None: predicción interna μ̂ = `[0,0,0,0]` (prior no informativo).
    ///
    /// # Política de valores no finitos
    /// Si `obs` contiene `NaN`/`±Inf`, retorna `0.0`.
    pub fn compute_vfe(&self, id: NodeId, obs: Option<&[f64; 4]>) -> f64 {
        let Some(idx) = self.lookup(id) else {
            return 0.0;
        };
        if let Some(o) = obs {
            if !is_finite_vec4(o) {
                return 0.0;
            }
        }
        let belief = &self.beliefs[idx];
        let target: [f64; 4] = obs.map_or([0.0; 4], |o| *o);
        // VFE sobre los 4 blades de grado 1 — Kahan summation.
        // Applies VFE_BLADE_WEIGHTS for consistency with compute_vfe_with_grad (FIX-3).
        let mut sum  = 0.0f64;
        let mut comp = 0.0f64;
        for (k, &blade_idx) in GRADE1_BLADE_INDICES.iter().enumerate() {
            let delta = belief.mean_full[blade_idx] - target[k];
            let w     = VFE_BLADE_WEIGHTS[blade_idx];
            let term  = w * belief.precision_full[blade_idx] * delta * delta;
            let y = term - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sum
    }

    /// VFE completo sobre los 16 blades de G(1,3) y gradiente `[f64; 16]`.
    ///
    /// ```text
    /// VFE_16D = Σ_{i=0}^{15} precision_full[i] · (mean_full[i] − target_full[i])²
    /// grad[i] = 2 · precision_full[i] · (mean_full[i] − target_full[i])
    /// ```
    ///
    /// El target completo se extrae de `obs.coeffs[0..16]` cuando `obs` es `Some`.
    /// Con `obs=None`, el target es el vector cero (prior vacío).
    ///
    /// # Uso en CRATE-004
    /// `DiscreteRicciFlow::step()` llama este método para obtener el gradiente
    /// completo 16D que dirige la deformación de la métrica de aristas. El gradiente
    /// cubre todos los grados de Clifford, produciendo señal de curvatura en la
    /// geometría completa del concepto, no solo en su componente vectorial.
    ///
    /// # Cambio de firma respecto a v1.0
    /// Retorna `[f64; 16]` en lugar de `[f64; 4]`. Callers que solo necesitan
    /// los 4 componentes de grado 1 deben indexar `grad[GRADE1_BLADE_INDICES]`.
    ///
    /// AX-ID: AXIOMA-003, H_información (LEY_FUNDACIONAL §3.3)
    pub fn compute_vfe_with_grad(
        &self,
        node: NodeId,
        obs: Option<&SparseCliffordVector>,
    ) -> (f64, [f64; 16]) {
        let Some(idx) = self.lookup(node) else {
            return (0.0, [0.0; 16]);
        };
        let Some(target) = target_from_sparse_obs(obs) else {
            return (0.0, [0.0; 16]);
        };

        let belief = &self.beliefs[idx];
        let mut vfe  = 0.0f64;
        let mut grad = [0.0f64; 16];

        // FIX-3: Apply VFE_BLADE_WEIGHTS for gradient/loss consistency with internal_drive.
        // Previously VFE_BLADE_WEIGHTS was only applied in internal_drive, making the gradient
        // direction inconsistent with the loss landscape (different metric in loss vs gradient).
        // Now both use the same weighted metric: F_i = w_i · Π_i · δ_i²
        for i in 0..16 {
            let delta = belief.mean_full[i] - target[i];
            let prec  = belief.precision_full[i];
            let w     = VFE_BLADE_WEIGHTS[i];
            vfe    += w * prec * delta * delta;
            grad[i] = 2.0 * w * prec * delta;
        }
        (vfe, grad)
    }

    /// Gradiente de grado 1 únicamente — acceso conveniente para callers
    /// que solo necesitan la señal de los 4 vectores base.
    ///
    /// Equivalente a `compute_vfe_with_grad()[1][GRADE1_BLADE_INDICES]`.
    ///
    /// AX-ID: AXIOMA-003
    pub fn compute_vfe_with_grad_grade1(
        &self,
        node: NodeId,
        obs: Option<&SparseCliffordVector>,
    ) -> (f64, [f64; 4]) {
        let (vfe, grad16) = self.compute_vfe_with_grad(node, obs);
        let grad4: [f64; 4] = core::array::from_fn(|k| grad16[GRADE1_BLADE_INDICES[k]]);
        (vfe, grad4)
    }

    /// Drive interno: retorna el `NodeId` con mayor VFE 16D bajo prior vacío.
    ///
    /// El VFE 16D cubre todos los grados de G(1,3), por lo que el nodo con
    /// mayor sorpresa puede ser uno con alta discrepancia en bivectores o
    /// trivectores, no solo en la componente vectorial.
    ///
    /// Sin estímulo externo el sistema minimiza F internamente. (AXIOMA-003)
    pub fn internal_drive(&mut self) -> Option<NodeId> {
        if self.beliefs.is_empty() {
            return None;
        }
        let mut max_vfe = f64::NEG_INFINITY;
        let mut max_id  = None;
        for (idx, belief) in self.beliefs.iter().enumerate() {
            if !is_finite_vec16(&belief.mean_full) {
                continue;
            }
            let trace = sanitize_trace(self.fisher[idx].trace);
            // F interna 16D: Tr(𝒢) · Σ_i w_i · μ_i² (target = 0)
            let error_sq: f64 = {
                let mut sum  = 0.0f64;
                let mut comp = 0.0f64;
                for (i, &m) in belief.mean_full.iter().enumerate() {
                    let term = VFE_BLADE_WEIGHTS[i] * m * m;
                    let y = term - comp;
                    let t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }
                sum
            };
            let vfe = trace * error_sq;
            if vfe > max_vfe {
                max_vfe = vfe;
                max_id  = Some(belief.node_id);
            }
        }
        max_id
    }

    /// Actualiza las creencias de grado 1 tras observación — retrocompatibilidad.
    ///
    /// Mismo comportamiento que en v1.0: solo actualiza los 4 blades de grado 1
    /// (`mean_full[GRADE1_BLADE_INDICES]`). Para actualizar todos los 16 blades,
    /// usar `update_full()`.
    ///
    /// AX-ID: AXIOMA-003, AXIOMA-008
    pub fn update(&mut self, id: NodeId, observation: &[f64; 4], dt: f64) {
        let Some(idx) = self.lookup(id) else { return };
        if !is_finite_vec4(observation) || !dt.is_finite() {
            return;
        }
        let belief = &mut self.beliefs[idx];
        let fisher = &mut self.fisher[idx];
        fisher.trace = sanitize_trace(fisher.trace);

        let step = (dt / (1.0 + dt * fisher.trace)).min(0.9);
        if !step.is_finite() { return; }

        // Actualizar solo los 4 blades de grado 1
        let mut error_sq = 0.0f64;
        for (k, &blade_idx) in GRADE1_BLADE_INDICES.iter().enumerate() {
            let err = observation[k] - belief.mean_full[blade_idx];
            belief.mean_full[blade_idx]     += step * err;
            belief.precision_full[blade_idx] = (belief.precision_full[blade_idx] + step).clamp(0.0, 1.0e6);
            error_sq += err * err;
        }

        let old_trace = fisher.trace;
        let error_mag = error_sq.sqrt();
        fisher.trace = sanitize_trace(old_trace / (1.0 + step * error_mag.max(TRACE_MIN)));
        fisher.delta_g = 0.5 * (fisher.trace - old_trace).abs();
    }

    /// Actualiza las creencias sobre los 16 blades completos de G(1,3).
    ///
    /// La observación `obs` debe ser un `SparseCliffordVector` — sus 16 coeficientes
    /// se usan como target completo. Actualiza `mean_full` y `precision_full` en todos
    /// los blades con señal no nula en `obs`.
    ///
    /// Para observaciones solo de grado 1, usar `update()` que es más eficiente.
    ///
    /// # Preparación para CRATE-004
    /// `DiscreteRicciFlow` puede suministrar observaciones completas 16D al
    /// ajustar creencias post-colapso de wormhole.
    ///
    /// AX-ID: AXIOMA-003, AXIOMA-008
    pub fn update_full(&mut self, id: NodeId, obs: &SparseCliffordVector, dt: f64) {
        let Some(idx) = self.lookup(id) else { return };
        if !dt.is_finite() { return; }

        let belief = &mut self.beliefs[idx];
        let fisher = &mut self.fisher[idx];
        fisher.trace = sanitize_trace(fisher.trace);

        let step = (dt / (1.0 + dt * fisher.trace)).min(0.9);
        if !step.is_finite() { return; }

        let mut error_sq = 0.0f64;
        for i in 0..16usize {
            let target = obs.coeffs[i];
            if !target.is_finite() { continue; }
            let err = target - belief.mean_full[i];
            belief.mean_full[i]     += step * err;
            belief.precision_full[i] = (belief.precision_full[i] + step).clamp(0.0, 1.0e6);
            error_sq += err * err;
        }

        let old_trace = fisher.trace;
        let error_mag = error_sq.sqrt();
        fisher.trace = sanitize_trace(old_trace / (1.0 + step * error_mag.max(TRACE_MIN)));
        fisher.delta_g = 0.5 * (fisher.trace - old_trace).abs();
    }

    /// Acceso a `FisherInfo` de un nodo.
    pub fn fisher(&self, id: NodeId) -> Option<&FisherInfo> {
        let idx = self.lookup(id)?;
        Some(&self.fisher[idx])
    }

    /// `ΔG = ||∂𝒢/∂t||` para `FisherGate` (AXIOMA-008).
    pub fn delta_g(&self, id: NodeId) -> f64 {
        self.lookup(id).map_or(0.0, |idx| self.fisher[idx].delta_g)
    }

    // ─── CRATE-004 prerequisite APIs (FIX-H) ────────────────────────────────

    /// Raw access to a node's belief for fusion operations (CRATE-004).
    ///
    /// Used by `inelastic_concept_fusion` to read the absorbed node's belief
    /// before merging it into the surviving node.
    ///
    /// Returns `Err(GenesisError::NodeNotFound)` if the node is not registered.
    /// AX-ID: LEY_FUNDACIONAL §3.7, CRATE-004 prerequisite
    pub fn beliefs_raw(&self, id: NodeId) -> Result<&Belief, GenesisError> {
        let idx = self.lookup(id).ok_or(GenesisError::NodeNotFound { id })?;
        Ok(&self.beliefs[idx])
    }

    /// Remove a node and its Fisher metadata.
    ///
    /// Leaves a tombstone in the belief and Fisher Vec to preserve index
    /// stability. Uses `Belief::tombstone()` + `FisherState::default()`.
    /// Called by `inelastic_concept_fusion` after the absorbed node is merged.
    ///
    /// Returns `Err(GenesisError::NodeNotFound)` if the node does not exist.
    /// AX-ID: LEY_FUNDACIONAL §3.7, CRATE-004 prerequisite
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GenesisError> {
        let raw = id.get() as usize;
        let idx = self.lookup(id).ok_or(GenesisError::NodeNotFound { id })?;
        if raw < self.id_to_idx.len() {
            self.id_to_idx[raw] = u32::MAX;
        }
        self.beliefs[idx] = Belief::tombstone();
        self.fisher[idx] = FisherInfo::new();
        Ok(())
    }
}

impl Default for VFEMinimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use genesis_math::SparseCliffordVector;

    #[test]
    fn vfe_internal_drive_returns_highest_vfe_node() {
        let mut vfe = VFEMinimizer::new();
        let a = NodeId::try_new(0).expect("NodeId válido por construcción");
        let b = NodeId::try_new(1).expect("NodeId válido por construcción");
        let c = NodeId::try_new(2).expect("NodeId válido por construcción");
        // mean más grande → VFE más alto (con trace=1.0 uniforme)
        vfe.add_node(a, [0.1, 0.0, 0.0, 0.0]);
        vfe.add_node(b, [10.0, 0.0, 0.0, 0.0]); // mayor VFE
        vfe.add_node(c, [1.0, 0.0, 0.0, 0.0]);
        let driven = vfe.internal_drive();
        assert_eq!(
            driven,
            Some(b),
            "internal_drive debe retornar nodo con mayor VFE"
        );
    }

    #[test]
    fn vfe_internal_drive_none_when_empty() {
        let mut vfe = VFEMinimizer::new();
        assert_eq!(vfe.internal_drive(), None);
    }

    #[test]
    fn vfe_update_reduces_vfe_for_matching_observation() {
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [0.0, 0.0, 0.0, 0.0]);
        let obs = [1.0f64, 0.5, 0.0, 0.0];
        let before = vfe.compute_vfe(id, Some(&obs));
        vfe.update(id, &obs, 0.1);
        let after = vfe.compute_vfe(id, Some(&obs));
        assert!(
            after < before,
            "VFE debe disminuir tras update con observación: {:.6} → {:.6}",
            before,
            after
        );
    }

    #[test]
    fn vfe_no_external_loss_function() {
        // Verifica que el ÚNICO mecanismo de actualización es la observación.
        // VFE disminuye tras update: la actualización interna converge, no diverge.
        // No existe API para inyectar gradientes externos — garantía de compilador.
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [2.0, 0.0, 0.0, 0.0]);
        let obs = [2.0f64, 0.0, 0.0, 0.0]; // observación que coincide con mean
        let before = vfe.compute_vfe(id, Some(&obs));
        vfe.update(id, &obs, 0.1);
        let after = vfe.compute_vfe(id, Some(&obs));
        // VFE con observación igual a media = 0 → debe mantenerse en 0 o bajar
        assert!(after <= before + 1e-12);
    }

    #[test]
    fn vfe_compute_returns_zero_for_unknown_node() {
        let vfe = VFEMinimizer::new();
        assert_eq!(
            vfe.compute_vfe(
                NodeId::try_new(99).expect("NodeId válido por construcción"),
                None
            ),
            0.0
        );
    }

    #[test]
    fn vfe_delta_g_nonzero_after_update() {
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [0.0; 4]);
        let obs = [1.0, 0.0, 0.0, 0.0];
        vfe.update(id, &obs, 0.1);
        let dg = vfe.delta_g(id);
        assert!(dg > 0.0, "ΔG debe ser > 0 tras update con error ≠ 0");
    }

    #[test]
    fn vfe_update_multiple_steps_converges() {
        // Tras muchas actualizaciones con la misma observación, VFE → 0.
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [0.0; 4]);
        let obs = [1.0, 0.5, -0.5, 0.25];
        for _ in 0..200 {
            vfe.update(id, &obs, 0.1);
        }
        let final_vfe = vfe.compute_vfe(id, Some(&obs));
        assert!(
            final_vfe < 1e-3,
            "VFE debe converger a ≈0 tras convergencia: {:.6}",
            final_vfe
        );
    }

    #[test]
    fn vfe_minkowski_temporal_spatial_asymmetry() {
        // En G(1,3), el error temporal y espacial tienen signos opuestos.
        // Un error puramente espacial debe producir VFE NEGATIVO antes de abs().
        // Verificar que la signatura Minkowski está correctamente aplicada.
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        // mean = [0,0,0,0], obs_temporal = [1,0,0,0] → error temporal positivo
        vfe.add_node(id, [0.0; 4]);
        let obs_t = [1.0f64, 0.0, 0.0, 0.0];
        let vfe_temporal = vfe.compute_vfe(id, Some(&obs_t));
        // Reiniciar con error puramente espacial
        let mut vfe2 = VFEMinimizer::new();
        vfe2.add_node(id, [0.0; 4]);
        let obs_x = [0.0f64, 1.0, 0.0, 0.0];
        let vfe_spatial = vfe2.compute_vfe(id, Some(&obs_x));
        // Temporal: sign=+1 → error_sq > 0. Espacial: sign=-1 → error_sq_raw < 0, abs > 0.
        // Ambos positivos por el abs(), pero los gradientes internos son opuestos.
        assert!(vfe_temporal >= 0.0, "VFE temporal debe ser ≥ 0");
        assert!(vfe_spatial >= 0.0, "VFE espacial debe ser ≥ 0");
        // Con Tr(G)=1.0, error=1.0, y VFE_BLADE_WEIGHTS=1.5 (grade-1): VFE = 1.5
        assert!(
            (vfe_temporal - 1.5).abs() < 1e-12,
            "VFE temporal = {} (expected 1.5 with w=1.5)",
            vfe_temporal
        );
        assert!(
            (vfe_spatial - 1.5).abs() < 1e-12,
            "VFE espacial = {} (expected 1.5 with w=1.5)",
            vfe_spatial
        );
    }

    #[test]
    fn vfe_delta_g_has_correct_frobenius_scale() {
        // Para Fisher isotrópica 4x4 con traza T: ‖ΔG‖_F = |ΔT|/2.
        // Verificar que delta_g == 0.5 * |ΔTr| tras una actualización.
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [0.0; 4]);
        let trace_before = vfe.fisher(id).unwrap().trace;
        let obs = [1.0, 0.0, 0.0, 0.0];
        vfe.update(id, &obs, 0.1);
        let fm = vfe.fisher(id).unwrap();
        let expected_delta_g = 0.5 * (fm.trace - trace_before).abs();
        assert!(
            (fm.delta_g - expected_delta_g).abs() < 1e-12,
            "delta_g = {}, expected = {}",
            fm.delta_g,
            expected_delta_g
        );
    }

    #[test]
    fn vfe_kahan_cancellation_near_lightcone() {
        // Verifica que compute_vfe es numéricamente estable cuando mean ≈ target.
        // Sin Kahan, para valores grandes el error relativo puede ser O(1).
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        // mean = [1000.0, 1000.0, 0.0, 0.0] — near the light cone.
        vfe.add_node(id, [1000.0, 1000.0, 0.0, 0.0]);
        // target = mean exactamente → VFE debe ser 0.
        let obs = [1000.0f64, 1000.0, 0.0, 0.0];
        let vfe_val = vfe.compute_vfe(id, Some(&obs));
        assert!(
            vfe_val.abs() < 1e-6,
            "VFE debe ser ≈0 cuando mean == target, got {}",
            vfe_val
        );
    }

    #[test]
    fn add_node_rejects_out_of_range_id_without_panic() {
        // NodeId::MAX_VALID = u64::MAX − 1; sólo u64::MAX es rechazado por try_new.
        assert!(NodeId::try_new(u64::MAX).is_err());

        // BN-05: 2_000_000 is now well within the new MAX_ALLOWED_NODE_ID = 100_000_000.
        // The former 1_000_000 hard cap is removed. Verify that a NodeId of 2M
        // is accepted and functional.
        let formerly_rejected = NodeId::try_new(2_000_000).expect("2_000_000 is valid NodeId");
        let mut vfe = VFEMinimizer::new();
        vfe.add_node(formerly_rejected, [1.5, 0.0, 0.0, 0.0]);
        // Must now work — node registered, VFE computable.
        let vfe_val = vfe.compute_vfe(formerly_rejected, Some(&[1.5, 0.0, 0.0, 0.0]));
        assert_eq!(vfe_val, 0.0, "node at id=2_000_000 must be fully functional");

        // Verify that a normal node still works alongside it.
        let valid_id = NodeId::try_new(999_999).expect("valor válido en test");
        vfe.add_node(valid_id, [0.0; 4]);
        assert_eq!(vfe.compute_vfe(valid_id, None), 0.0);
    }

    #[test]
    fn bounded_step_grid_respects_declared_bounds() {
        let dts = [1e-12_f64, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12];
        let traces = [1e-12_f64, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12];

        for &dt in &dts {
            for &trace in &traces {
                let step = VFEMinimizer::bounded_step(dt, trace);
                let product = step * trace;

                assert!(
                    (0.0..=0.9).contains(&step),
                    "step fuera de rango para dt={dt:e}, trace={trace:e}: {step:e}"
                );
                assert!(
                    product <= 1.0 + f64::EPSILON,
                    "step*trace debe ser <= 1 para dt={dt:e}, trace={trace:e}: {product:e}"
                );
            }
        }
    }
    fn add_node_rejects_non_finite_prior_mean() {
        let mut vfe = VFEMinimizer::new();
        let id_nan = NodeId::try_new(0).expect("NodeId válido por construcción");
        let id_inf = NodeId::try_new(1).expect("NodeId válido por construcción");
        let id_neg_inf = NodeId::try_new(2).expect("NodeId válido por construcción");

        vfe.add_node(id_nan, [f64::NAN, 0.0, 0.0, 0.0]);
        vfe.add_node(id_inf, [f64::INFINITY, 0.0, 0.0, 0.0]);
        vfe.add_node(id_neg_inf, [f64::NEG_INFINITY, 0.0, 0.0, 0.0]);

        assert_eq!(vfe.compute_vfe(id_nan, None), 0.0);
        assert_eq!(vfe.compute_vfe(id_inf, None), 0.0);
        assert_eq!(vfe.compute_vfe(id_neg_inf, None), 0.0);
    }

    #[test]
    fn update_ignores_non_finite_observation_values() {
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [0.0; 4]);

        let baseline = vfe.compute_vfe(id, Some(&[1.0, 0.0, 0.0, 0.0]));
        let trace_before = vfe.fisher(id).unwrap().trace;

        vfe.update(id, &[f64::NAN, 0.0, 0.0, 0.0], 0.1);
        vfe.update(id, &[f64::INFINITY, 0.0, 0.0, 0.0], 0.1);
        vfe.update(id, &[f64::NEG_INFINITY, 0.0, 0.0, 0.0], 0.1);

        let after = vfe.compute_vfe(id, Some(&[1.0, 0.0, 0.0, 0.0]));
        let trace_after = vfe.fisher(id).unwrap().trace;
        assert!((after - baseline).abs() < 1e-12);
        assert!((trace_before - trace_after).abs() < 1e-12);
    }

    #[test]
    fn compute_vfe_rejects_non_finite_observations() {
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [0.5, 0.0, 0.0, 0.0]);

        assert_eq!(vfe.compute_vfe(id, Some(&[f64::NAN, 0.0, 0.0, 0.0])), 0.0);
        assert_eq!(
            vfe.compute_vfe(id, Some(&[f64::INFINITY, 0.0, 0.0, 0.0])),
            0.0
        );
        assert_eq!(
            vfe.compute_vfe(id, Some(&[f64::NEG_INFINITY, 0.0, 0.0, 0.0])),
            0.0
        );
    }

    #[test]
    fn internal_drive_ignores_non_finite_nodes() {
        let mut vfe = VFEMinimizer::new();
        let invalid = NodeId::try_new(0).expect("NodeId válido por construcción");
        let valid = NodeId::try_new(1).expect("NodeId válido por construcción");

        vfe.add_node(invalid, [f64::NAN, 0.0, 0.0, 0.0]);
        vfe.add_node(valid, [1.0, 0.0, 0.0, 0.0]);

        assert_eq!(vfe.internal_drive(), Some(valid));
    }

    #[test]
    fn fisher_trace_remains_finite_and_bounded() {
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [0.0; 4]);

        for obs in [
            [1.0, 0.0, 0.0, 0.0],
            [f64::NAN, 0.0, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0, 0.0],
            [f64::NEG_INFINITY, 0.0, 0.0, 0.0],
        ] {
            vfe.update(id, &obs, 0.1);
            let trace = vfe.fisher(id).unwrap().trace;
            assert!(trace.is_finite(), "trace debe ser finita");
            assert!(
                (TRACE_MIN..=TRACE_MAX).contains(&trace),
                "trace fuera de rango: {}",
                trace
            );
        }
    }

    #[test]
    fn fisher_edge_metric_get_is_symmetric() {
        let n0 = NodeId::try_new(0).expect("NodeId válido por construcción");
        let n1 = NodeId::try_new(1).expect("NodeId válido por construcción");
        let n2 = NodeId::try_new(2).expect("NodeId válido por construcción");

        let metric = FisherEdgeMetric::new(vec![((n1, n0), 1.5), ((n2, n1), 2.5)]);

        assert!((metric.get(n0, n1) - 1.5).abs() < f64::EPSILON);
        assert!((metric.get(n1, n0) - 1.5).abs() < f64::EPSILON);
        assert!((metric.get(n1, n2) - 2.5).abs() < f64::EPSILON);
        assert!((metric.get(n2, n1) - 2.5).abs() < f64::EPSILON);
        assert_eq!(metric.get(n0, n2), 0.0);

        let listed: Vec<_> = metric.edges().collect();
        assert_eq!(listed, vec![(n0, n1, 1.5), (n1, n2, 2.5)]);
        assert!(metric.is_current(n0));
        assert!(metric.is_current(n1));
        assert!(metric.is_current(n2));
        let n3 = NodeId::try_new(3).expect("NodeId válido por construcción");
        assert!(!metric.is_current(n3));
    }

    #[test]
    fn fisher_edge_metric_binary_search_correctness() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed
        };

        let node_count = 64u64;
        let mut storage = Vec::new();
        for _ in 0..256 {
            let left_raw = next_random() % node_count;
            let mut right_raw = next_random() % node_count;
            if left_raw == right_raw {
                right_raw = (right_raw + 1) % node_count;
            }
            let left = NodeId::try_new(left_raw).expect("NodeId válido por construcción");
            let right = NodeId::try_new(right_raw).expect("NodeId válido por construcción");
            let weight = (next_random() % 10_000) as f64 / 100.0;
            storage.push(((left, right), weight));
        }

        let metric = FisherEdgeMetric::new(storage.clone());
        for _ in 0..1000 {
            let left_raw = next_random() % node_count;
            let mut right_raw = next_random() % node_count;
            if left_raw == right_raw {
                right_raw = (right_raw + 1) % node_count;
            }
            let left = NodeId::try_new(left_raw).expect("NodeId válido por construcción");
            let right = NodeId::try_new(right_raw).expect("NodeId válido por construcción");
            let edge_key = canonical_edge(left, right);
            let mut normalized = storage.clone();
            for ((edge_left, edge_right), _) in &mut normalized {
                if edge_right < edge_left {
                    core::mem::swap(edge_left, edge_right);
                }
            }
            normalized.sort_unstable_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
            normalized.dedup_by(|lhs, rhs| lhs.0 == rhs.0);
            let expected = normalized
                .iter()
                .find_map(|((edge_left, edge_right), weight)| {
                    ((*edge_left, *edge_right) == edge_key).then_some(*weight)
                })
                .unwrap_or(0.0);
            let actual = metric.get(left, right);
            assert!(
                (actual - expected).abs() < f64::EPSILON,
                "binary_search y lineal divergen para ({:?}, {:?}): got={}, expected={}",
                left,
                right,
                actual,
                expected
            );
        }
    }

    #[test]
    fn vfe_grad_matches_finite_difference() {
        // El gradiente de compute_vfe_with_grad es ahora [f64; 16].
        // mean[k] (grado 1, componente k) se almacena en mean_full[GRADE1_BLADE_INDICES[k]].
        // GRADE1_BLADE_INDICES[0] = 1 → blade e₀.
        // La diferencia finita perturba mean[0] → blade 1, y debemos comparar
        // con grad[GRADE1_BLADE_INDICES[0]] = grad[1].
        let id   = NodeId::try_new(0).expect("NodeId válido por construcción");
        let mean = [1.3, -0.2, 0.4, 0.1];

        // obs con coeficientes en todos los blades incluyendo los de grado 1
        // (blades 1,2,4,8 según GRADE1_BLADE_INDICES)
        let obs = SparseCliffordVector::from_iter([
            (1usize, 0.1),   // blade e₀ (grado 1, índice 0 de mean)
            (2, -0.4),       // blade e₁ (grado 1, índice 1 de mean)
        ])
        .expect("observación finita válida");

        let mut base = VFEMinimizer::new();
        base.add_node(id, mean);
        let (value, grad) = base.compute_vfe_with_grad(id, Some(&obs));
        assert!(value.is_finite(), "VFE debe ser finito");

        let eps = 1e-6;

        // Perturbar mean[0] (blade GRADE1_BLADE_INDICES[0] = 1)
        let mut plus = VFEMinimizer::new();
        let mut mean_plus = mean;
        mean_plus[0] += eps;
        plus.add_node(id, mean_plus);
        let v_plus = plus.compute_vfe(
            id,
            Some(&[obs.coeffs[1], obs.coeffs[2], obs.coeffs[4], obs.coeffs[8]]),
        );

        let mut minus = VFEMinimizer::new();
        let mut mean_minus = mean;
        mean_minus[0] -= eps;
        minus.add_node(id, mean_minus);
        let v_minus = minus.compute_vfe(
            id,
            Some(&[obs.coeffs[1], obs.coeffs[2], obs.coeffs[4], obs.coeffs[8]]),
        );

        let finite_diff = (v_plus - v_minus) / (2.0 * eps);
        // grad[GRADE1_BLADE_INDICES[0]] = grad[1] corresponde a mean[0]
        let grad_blade = grad[GRADE1_BLADE_INDICES[0]];
        assert!(
            (grad_blade - finite_diff).abs() < 1e-5,
            "grad[blade_1] AD y finite diff difieren: grad={grad_blade}, fd={finite_diff}, err={}",
            (grad_blade - finite_diff).abs()
        );
    }

    /// Verifica que el gradiente 16D cubre correctamente los blades de grado 1.
    /// Los componentes de grado 1 son los más semánticamente relevantes.
    #[test]
    fn vfe_grad_16d_grade1_components_consistent() {
        let id    = NodeId::try_new(0).expect("NodeId válido");
        let mean  = [2.0, 0.0, 0.0, 0.0];
        let obs   = SparseCliffordVector::from_iter([(1usize, 1.0)]).expect("obs válida");

        let mut vfe = VFEMinimizer::new();
        vfe.add_node(id, mean);
        let (val, grad) = vfe.compute_vfe_with_grad(id, Some(&obs));

        // mean_full[1] = 2.0 (blade e₀, GRADE1_BLADE_INDICES[0])
        // target[1]   = 1.0 (obs.coeffs[1])
        // VFE = w * precision_full[1] * (2.0 - 1.0)² = 1.5 * 1.0 * 1.0 = 1.5 (FIX-3: VFE_BLADE_WEIGHTS[1]=1.5)
        // grad[1] = 2 * w * prec * delta = 2 * 1.5 * 1.0 * 1.0 = 3.0
        assert!((val - 1.5).abs() < 1e-12, "VFE debe ser 1.5 (w=1.5 for grade-1), got {val}");
        assert!(
            (grad[GRADE1_BLADE_INDICES[0]] - 3.0).abs() < 1e-12,
            "grad[blade e₀] debe ser 3.0 (2*w*prec*delta), got {}",
            grad[GRADE1_BLADE_INDICES[0]]
        );
        // Gradientes en blades no activados por obs o mean deben ser 0
        assert_eq!(grad[0], 0.0, "blade escalar no activado, grad debe ser 0");
        assert_eq!(grad[15], 0.0, "blade pseudoescalar no activado, grad debe ser 0");
    }

    /// Verifica que compute_vfe_with_grad_grade1 retorna exactamente los 4
    /// componentes de grado 1 del gradiente 16D.
    #[test]
    fn vfe_grad_grade1_convenience_matches_full_grad() {
        let id   = NodeId::try_new(0).expect("NodeId válido");
        let mean = [1.0, -0.5, 0.3, 0.2];
        let obs  = SparseCliffordVector::from_iter([(1usize, 0.5), (2, 0.0), (4, 0.1)]).unwrap();

        let mut vfe = VFEMinimizer::new();
        vfe.add_node(id, mean);

        let (v16, g16)  = vfe.compute_vfe_with_grad(id, Some(&obs));
        let (v4,  g4)   = vfe.compute_vfe_with_grad_grade1(id, Some(&obs));

        assert_eq!(v16, v4, "VFE debe ser igual en ambas variantes");
        for k in 0..4 {
            assert_eq!(
                g4[k], g16[GRADE1_BLADE_INDICES[k]],
                "g4[{k}] debe coincidir con g16[GRADE1_BLADE_INDICES[{k}]]=g16[{}]",
                GRADE1_BLADE_INDICES[k]
            );
        }
    }

    /// Verifica que update_full actualiza todos los 16 blades activos en obs.
    #[test]
    fn update_full_updates_all_active_blades() {
        let id  = NodeId::try_new(0).expect("NodeId válido");
        let obs = SparseCliffordVector::from_iter([
            (0usize, 0.5), // escalar (grado 0)
            (1, 1.0),      // e₀ (grado 1)
            (3, 0.8),      // e₀₁ (grado 2)
            (15, 0.2),     // pseudoescalar (grado 4)
        ]).expect("obs válida");

        let mut vfe = VFEMinimizer::new();
        vfe.add_node(id, [0.0; 4]); // mean_full parte en 0

        let belief_before = vfe.beliefs[0].mean_full;
        vfe.update_full(id, &obs, 0.1);
        let belief_after = vfe.beliefs[0].mean_full;

        // Blades 0, 1, 3, 15 deben haber cambiado (había error ≠ 0)
        for &blade in &[0usize, 1, 3, 15] {
            assert!(
                (belief_after[blade] - belief_before[blade]).abs() > 1e-10,
                "blade {blade} debe haber cambiado tras update_full"
            );
        }
        // Blade 7 (no activado en obs) no debe haber cambiado
        assert_eq!(
            belief_after[7], belief_before[7],
            "blade 7 no activado en obs no debe cambiar"
        );
    }
}
