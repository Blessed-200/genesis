use crate::oscillator::QuantumOscillator;
use genesis_types::{GenesisError, NodeId};

/// Genera una muestra aproximadamente `N(0,1)` vía Box-Muller con LCG de 64 bits.
///
/// Algoritmo:
///   1. LCG step: `rng = rng * 6364136223846793005 + 1442695040888963407`
///   2. `u1 = ((rng >> 11) / 2^53).clamp(MIN_POSITIVE, 1.0)`
///   3. LCG step adicional para u2 ∈ [0, 1)
///   4. r = sqrt(-2 * ln(u1)),  θ = 2π * u2
///   5. Retorna r * cos(θ) ~ N(0, 1)
///
/// Propiedades:
/// - Distribución: Box-Muller estándar sobre una malla discreta de `u1`/`u2`.
/// - Cota de amplitud por clamp: para `u1_min = f64::MIN_POSITIVE = 2^-1022`,
///   `r_max = sqrt(-2 ln(u1_min)) = sqrt(2 * 1022 * ln 2) ≈ 37.64σ`.
/// - Consistencia estadística: los tests validan `E[η] ≈ 0` y `Var[η] ≈ 1`.
/// - Costo: 2 iteraciones LCG + ln + sqrt + cos ≈ 25-35 ciclos en AVX-512
/// - Heap: cero. Dependencias: cero. Estado: 8 bytes (rng en registro).
///
/// Para N=10⁶ osciladores × 5 grados = 5×10⁶ llamadas/step.
/// A 30 ciclos/llamada × 3.5 GHz = ~43ms/step. Aceptable para dt=0.01s.
/// Si el profiler demuestra regresión, reemplazar por Ziggurat vectorizado.
///
/// AX-ID: AXIOMA-006 — decoherencia térmica requerida en integración Euler-Maruyama.
// Hot path: invocada 5 veces por nodo por step de integración.
#[allow(clippy::inline_always)]
#[inline(always)]
fn gaussian_noise(rng: &mut u64) -> f64 {
    // Implementación Box-Muller determinística para mantener E[η]≈0 y Var[η]≈1
    // (ver test `gaussian_noise_mean_and_variance`).
    *rng = rng
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // `rng >> 11` deja 53 bits; la conversión a `f64` es exacta por mantisa IEEE-754.
    #[allow(clippy::cast_precision_loss)]
    let u1 = ((*rng >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0);

    *rng = rng
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // Mismo razonamiento: 53 bits efectivos en numerador y 2^53 exacto en denominador.
    #[allow(clippy::cast_precision_loss)]
    let u2 = (*rng >> 11) as f64 / (1u64 << 53) as f64;

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * core::f64::consts::PI * u2;
    r * theta.cos()
}

// ── wrap_phase_diff ───────────────────────────────────────────────────────────

/// Diferencia de fase en S¹.
///
/// Rango esperado para la diferencia principal: (−π, π].
/// Con la convención branchless usada aquí, los casos de frontera `d = ±π`
/// (y sus equivalentes periódicos) se representan como `-π`.
///
/// Comportamiento exacto de fronteras:
/// - `d = n·2π + π`  → `-π`
/// - `d = n·2π - π`  → `-π`
///
/// Si `d` es `NaN`, el resultado permanece `NaN` (propagación IEEE-754).
///
/// Invariante a traslación circular: wrap(φ + 2πk) = wrap(φ) ∀k ∈ ℤ.
// Hot path de diferencias angulares usado en métricas y acoplamiento de fase.
#[allow(clippy::inline_always)]
#[inline(always)]
fn wrap_phase_diff(d: f64) -> f64 {
    d - core::f64::consts::TAU * (d * (1.0 / core::f64::consts::TAU) + 0.5).floor()
}

// ── QuantumKuramotoNetwork ────────────────────────────────────────────────────

/// Red de osciladores de Kuramoto cuánticos.
///
/// La ecuación de movimiento NO es postulada — se DERIVA de `H_dinámica`:
///   dφᵢ/dt = ωᵢ + Σⱼ Γᵢⱼ sin(φⱼ - φᵢ) + η√(2kT·dt)
///
/// PROHIBIDO: colapso determinista forzado (AXIOMA-006).
/// PROHIBIDO: `HashMap` en ningún hot path.
/// Sin dependencia de `rand` — LCG determinístico interno.
///
/// AX-ID: AXIOMA-006, `H_dinámica` (`LEY_FUNDACIONAL` §3.2)
pub struct QuantumKuramotoNetwork {
    pub(crate) oscillators: Vec<QuantumOscillator>,

    /// Acoplamiento público (`NodeId`-based) — fuente de verdad para serialización.
    /// PROHIBIDO `HashMap` — `Vec` sparse.
    pub coupling: Vec<(NodeId, NodeId, f64)>,

    /// Índice interno: (`idx_i`, `idx_j`, `gamma`) sorted by `idx_i` para binary search.
    /// Se reconstruye lazy cuando `dirty = true`.
    coupling_idx: Vec<(u32, u32, f64)>,

    /// `NodeId` → índice en `oscillators`. `u32::MAX` = no registrado.
    /// Direct array: O(1) lookup, válido para `NodeIds` consecutivos.
    id_to_idx: Vec<u32>,

    /// kT — controla amplitud de decoherencia térmica.
    pub temperature: f64,

    /// Estado del LCG. Semilla fija → reproducibilidad determinística.
    rng_state: u64,

    /// Segunda muestra Box-Muller pendiente de consumir.
    spare_gaussian: Option<f64>,

    /// Scratch para fases del paso anterior (Euler-Maruyama correcto).
    /// Reutilizado entre `steps`. Un único alloc, resize solo en `add_oscillator`.
    phase_scratch: Vec<[f64; 5]>,

    /// Offsets por nodo origen en `coupling_idx`: `(start, end)` para acceso O(1).
    coupling_offsets: Vec<(usize, usize)>,

    /// Scratch amplitudes per node for adaptive coupling (reused each step).
    amp_scratch: Vec<f64>,

    /// Scratch saturations per node for adaptive coupling (reused each step).
    sat_scratch: Vec<f64>,

    /// Flag: `coupling_idx` necesita rebuild.
    dirty: bool,

    /// Caché del parámetro de orden r_sync.
    /// Invalidado por `step()` vía `sync_dirty = true`.
    sync_cache: f64,
    sync_dirty: bool,
}

impl QuantumKuramotoNetwork {
    /// Creates a new Kuramoto network with the given noise temperature `kT`.
    ///
    /// Higher `temperature` → stronger stochastic exploration (AXIOMA-006).
    pub fn new(temperature: f64) -> Self {
        Self {
            oscillators: Vec::new(),
            coupling: Vec::new(),
            coupling_idx: Vec::new(),
            id_to_idx: Vec::new(),
            temperature,
            rng_state: 0xdead_beef_cafe_babe_u64,
            spare_gaussian: None,
            phase_scratch: Vec::new(),
            coupling_offsets: Vec::new(),
            amp_scratch: Vec::new(),
            sat_scratch: Vec::new(),
            dirty: false,
            sync_cache: 0.0,
            sync_dirty: true,
        }
    }

    /// # Errors
    /// Returns [`GenesisError::NodeIdOutOfRange`] when `osc.node_id` is invalid,
    /// or [`GenesisError::KuramotoDuplicateNodeId`] if the node already exists.
    ///
    /// # Panics
    /// Panics if `osc.node_id.get()` cannot fit in `usize` on the target architecture,
    /// or if the oscillator count exceeds `u32::MAX`.
    pub fn add_oscillator(&mut self, osc: QuantumOscillator) -> Result<NodeId, GenesisError> {
        let node_id = NodeId::try_new(osc.node_id.get())?;
        let raw_u64 = node_id.get();
        let raw =
            usize::try_from(raw_u64).expect("NodeId must fit into usize on supported targets");

        if raw < self.id_to_idx.len() && self.id_to_idx[raw] != u32::MAX {
            return Err(GenesisError::KuramotoDuplicateNodeId { raw: raw_u64 });
        }

        let idx = u32::try_from(self.oscillators.len())
            .expect("oscillator count must stay below u32::MAX");

        if raw >= self.id_to_idx.len() {
            self.id_to_idx.resize(raw + 1, u32::MAX);
        }

        self.id_to_idx[raw] = idx;
        self.oscillators.push(osc);
        self.phase_scratch.push([0.0; 5]);
        self.coupling_offsets.push((0, 0));
        self.amp_scratch.push(1.0);
        self.sat_scratch.push(0.0);
        self.dirty = true;

        Ok(node_id)
    }

    /// Establece acoplamiento Γᵢⱼ. Si gamma == 0.0, elimina la arista.
    ///
    /// COMPLEJIDAD: O(log E) — binary search sobre Vec ordenado.
    /// Mantiene `coupling` sorted por (i.get(), j.get()) para O(log E) lookup.
    pub fn set_coupling(&mut self, i: NodeId, j: NodeId, gamma: f64) {
        let key = (i.get(), j.get());
        let pos = self
            .coupling
            .binary_search_by_key(&key, |&(a, b, _)| (a.get(), b.get()));
        match (pos, gamma != 0.0) {
            (Ok(p), true) => self.coupling[p].2 = gamma,
            (Ok(p), false) => {
                self.coupling.remove(p);
            }
            (Err(p), true) => self.coupling.insert(p, (i, j, gamma)),
            (Err(_), false) => {}
        }
        self.dirty = true;
    }

    /// Inserta N acoplamientos en O(E log E) total en lugar de O(N × E).
    pub fn set_coupling_batch(&mut self, pairs: &[(NodeId, NodeId, f64)]) {
        for &(i, j, gamma) in pairs {
            self.set_coupling(i, j, gamma);
        }
    }

    /// Adaptive Kuramoto step with bivector frustration and semantic habituation.
    ///
    /// Replaces the standard `step(dt)` when multivector geometry should modulate
    /// coupling. The effective coupling per edge (i,j) is:
    ///
    /// ```text
    /// adaptive_Γ = Γ₀ · (Aᵢ · Aⱼ / Ā²) · (α + β · ⟨Bᵢ, Bⱼ⟩₂) · (1 − satᵢ · satⱼ)
    /// ```
    ///
    /// Where:
    /// - `Aᵢ = oscillator.amplitude_norm()` — inferential certainty
    /// - `⟨Bᵢ, Bⱼ⟩₂ = vec_i.dot_bivectors(vec_j)` — orientation alignment ∈ (−|B|², +|B|²)
    /// - `satᵢ = 1 − Aᵢ` — semantic saturation (0=uncertain, 1=settled)
    /// - α = `KURAMOTO_COUPLING_ALPHA` (isotropic baseline)
    /// - β = `KURAMOTO_COUPLING_BETA` (frustration weight)
    ///
    /// **Frustration:** Anti-aligned bivectors produce γ < 0 (repulsive coupling),
    /// preventing global synchronisation. Clusters form by geometric orientation
    /// affinity, not proximity alone. No eigenvalue computation — purely local.
    ///
    /// **Habituation:** When both nodes are settled (`sat ≈ 1`), coupling collapses
    /// to zero. Compute is automatically redirected to uncertain pairs (high VFE).
    /// This is the AXIOMA-006 × AXIOMA-008 coupling made explicit in the integrator.
    ///
    /// **Integrator:** Euler-Maruyama (same as `step`). Semi-implicit Verlet is
    /// applicable for the deterministic path but not yet implemented — the noise
    /// term requires Maruyama correction regardless.
    ///
    /// # Parameters
    /// `vecs` — slice of `SparseCliffordVector` indexed identically to `self.oscillators`.
    ///          Length must equal `node_count()` or the method is a no-op.
    /// `dt` — integration timestep in seconds.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008, H_dinámica (LEY_FUNDACIONAL §3.2)
    pub fn adaptive_step(&mut self, vecs: &[genesis_math::SparseCliffordVector], dt: f64) {
        use genesis_types::constants::{
            KURAMOTO_COUPLING_ALPHA, KURAMOTO_COUPLING_BETA, KURAMOTO_COUPLING_FLOOR,
        };

        self.rebuild_if_dirty();
        let n = self.oscillators.len();
        if n == 0 || vecs.len() != n {
            return;
        }

        // Snapshot phases before update (simultaneous semantics — Euler-Maruyama).
        for i in 0..n {
            self.phase_scratch[i] = self.oscillators[i].phases;
        }

        // Mean amplitude squared — normalises amplitude product.
        let avg_amp_sq: f64 = {
            let s: f64 = self.oscillators.iter().map(|o| o.amplitude_norm()).sum();
            let mean = s / n as f64;
            (mean * mean).max(f64::MIN_POSITIVE)
        };

        self.amp_scratch.resize(n, 0.0);
        self.sat_scratch.resize(n, 0.0);
        for (i, osc) in self.oscillators.iter().enumerate() {
            let amp = osc.amplitude_norm();
            self.amp_scratch[i] = amp;
            self.sat_scratch[i] = 1.0 - amp;
        }

        let sqrt_2k_t_dt = (2.0 * self.temperature * dt).sqrt();

        for i in 0..n {
            if !self.oscillators[i].state.contributes_to_sync() {
                continue;
            }
            let (start, end) = self.coupling_offsets[i];
            let edges = &self.coupling_idx[start..end];
            let phi_i = &self.phase_scratch[i];
            let amp_i = self.amp_scratch[i];
            let sat_i = self.sat_scratch[i];
            let mut coupling_sums = [0.0f64; 5];

            for &(_, j_u32, gamma_0) in edges {
                #[allow(clippy::cast_possible_truncation)]
                let j = j_u32 as usize;
                let phi_j = &self.phase_scratch[j];
                let amp_j = self.amp_scratch[j];
                let sat_j = self.sat_scratch[j];

                // Bivector frustration: dot product of grade-2 components.
                // Positive → aligned orientation → attractive coupling.
                // Negative → opposed orientation → repulsive coupling.
                let dot_biv = vecs[i].dot_bivectors(&vecs[j]);

                // Amplitude modulation: high-certainty pairs contribute more.
                let amp_factor = (amp_i * amp_j) / avg_amp_sq;

                // Orientation modulation: α + β·⟨Bᵢ,Bⱼ⟩₂ ∈ (α−β, α+β) for unit bivectors.
                let orient_factor = KURAMOTO_COUPLING_ALPHA + KURAMOTO_COUPLING_BETA * dot_biv;

                // Habituation: settled pairs withdraw coupling (AXIOMA-008 coupling).
                let habituate = 1.0 - sat_i * sat_j;

                let adaptive_gamma =
                    (gamma_0 * amp_factor * orient_factor * habituate).max(KURAMOTO_COUPLING_FLOOR);

                for g in 0..5usize {
                    coupling_sums[g] += adaptive_gamma * (phi_j[g] - phi_i[g]).sin();
                }
            }

            for (g, &coupling_sum) in coupling_sums.iter().enumerate() {
                let omega = self.oscillators[i].frequencies[g];
                let eta = if self.temperature > 0.0 {
                    self.next_gaussian() * sqrt_2k_t_dt
                } else {
                    0.0
                };
                self.oscillators[i].phases[g] += (omega + coupling_sum) * dt + eta;
            }
        }

        self.sync_dirty = true;
    }

    /// Número de nodos registrados.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.oscillators.len()
    }

    /// Acceso inmutable a todos los osciladores.
    /// Expuesto para `genesis-evolution::compression` (contrato CRATE-004).
    #[inline]
    pub fn phases(&self) -> &[QuantumOscillator] {
        &self.oscillators
    }

    /// Diferencia de fase normalizada (RMS sobre 5 grados) entre nodos i y j.
    ///
    /// Retorna `√(Σ_g wrap(φᵢg - φⱼg)² / 5) ∈ [0, π]`.
    /// Simétrica: `phase_diff_norm(i,j) == phase_diff_norm(j,i)`.
    /// Cero para el mismo nodo. Invariante a traslación circular (módulo 2π).
    ///
    /// Expuesto para `genesis-evolution::compression` (`LEY_FUNDACIONAL` §3.7).
    /// AX-ID: `LEY_FUNDACIONAL` §3.7
    pub fn phase_diff_norm(&self, i: NodeId, j: NodeId) -> f64 {
        let Some(ii) = self.lookup_idx(i) else {
            return f64::NAN;
        };
        let Some(ij) = self.lookup_idx(j) else {
            return f64::NAN;
        };
        if ii == ij {
            return 0.0;
        }
        let oi = &self.oscillators[ii];
        let oj = &self.oscillators[ij];
        let sum_sq: f64 = (0..5)
            .map(|g| {
                let d = wrap_phase_diff(oi.phases[g] - oj.phases[g]);
                d * d
            })
            .sum();
        (sum_sq / 5.0).sqrt()
    }

    // ── Integración ──────────────────────────────────────────────────────────
    /// Un paso de integración Euler-Maruyama, `dt` segundos.
    ///
    /// Para cada oscilador i y grado g:
    ///   φᵢg(t+dt) = φᵢg(t) + [ωᵢg + Σⱼ Γᵢⱼ sin(φⱼg(t) - φᵢg(t))] · dt
    ///              + η · √(2 kT dt)
    ///
    /// `η ~ N(0,1)` vía LCG Box-Muller. Sin heap por `step`.
    /// PROHIBIDO: colapso determinista (AXIOMA-006).
    pub fn step(&mut self, dt: f64) {
        self.rebuild_if_dirty();
        let n = self.oscillators.len();
        if n == 0 {
            return;
        }

        // Snapshot de fases previas en scratch (Euler-Maruyama: usa φ(t), no φ(t+dt)).
        for i in 0..n {
            self.phase_scratch[i] = self.oscillators[i].phases;
        }

        let sqrt_2k_t_dt = (2.0 * self.temperature * dt).sqrt();

        if self.temperature > 0.0 {
            self.step_inner_noisy(dt, sqrt_2k_t_dt);
        } else {
            self.step_inner_deterministic(dt);
        }

        self.sync_dirty = true;
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn next_gaussian(&mut self) -> f64 {
        if let Some(spare) = self.spare_gaussian.take() {
            return spare;
        }

        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let u1 =
            ((self.rng_state >> 11) as f64 / (1u64 << 53) as f64).clamp(f64::MIN_POSITIVE, 1.0);

        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let u2 = (self.rng_state >> 11) as f64 / (1u64 << 53) as f64;

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        let (sin_theta, cos_theta) = theta.sin_cos();
        self.spare_gaussian = Some(r * sin_theta);
        r * cos_theta
    }

    /// Compute the 5-grade coupling sum for oscillator `i`.
    ///
    /// Extracted from both `step_inner_deterministic` and `step_inner_noisy`
    /// to eliminate code duplication (FIX-G). The function is `#[inline]` — no
    /// call overhead. Both callers now share identical coupling logic, making
    /// discrepancies between deterministic and stochastic paths impossible.
    ///
    /// Complexity: O(|edges_i| × G) where G=5 constant grades.
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    fn compute_coupling_sums(
        phase_scratch: &[[f64; 5]],
        coupling_idx: &[(u32, u32, f64)],
        coupling_offsets: &[(usize, usize)],
        oscillators: &[crate::oscillator::QuantumOscillator],
        i: usize,
    ) -> [f64; 5] {
        let mut sums = [0.0f64; 5];
        let (start, end) = coupling_offsets[i];
        let phi_i = &phase_scratch[i];
        for &(_, j_u32, gamma) in &coupling_idx[start..end] {
            if oscillators[j_u32 as usize].state.contributes_to_sync() {
                #[allow(clippy::cast_possible_truncation)]
                let phi_j = &phase_scratch[j_u32 as usize];
                for g in 0..5usize {
                    sums[g] += gamma * (phi_j[g] - phi_i[g]).sin();
                }
            }
        }
        sums
    }

    fn step_inner_deterministic(&mut self, dt: f64) {
        let n = self.oscillators.len();
        for i in 0..n {
            // Skip oscillators that don't contribute to dynamics (Pruned state).
            // AX-ID: AXIOMA-008 (Saturated), AXIOMA-016 (Pruned)
            if !self.oscillators[i].state.contributes_to_sync() {
                continue;
            }
            // FIX-G: coupling_sums extracted to shared inline function.
            let coupling_sums = Self::compute_coupling_sums(
                &self.phase_scratch,
                &self.coupling_idx,
                &self.coupling_offsets,
                &self.oscillators,
                i,
            );
            for (g, coupling_sum) in coupling_sums.iter().enumerate() {
                let omega = self.oscillators[i].frequencies[g];
                self.oscillators[i].phases[g] += (omega + coupling_sum) * dt;
            }
        }
    }

    fn step_inner_noisy(&mut self, dt: f64, sqrt_2k_t_dt: f64) {
        let n = self.oscillators.len();
        for i in 0..n {
            // Skip oscillators that don't contribute to dynamics (Pruned state).
            // AX-ID: AXIOMA-008 (Saturated), AXIOMA-016 (Pruned)
            if !self.oscillators[i].state.contributes_to_sync() {
                continue;
            }
            // FIX-G: coupling_sums extracted to shared inline function.
            let coupling_sums = Self::compute_coupling_sums(
                &self.phase_scratch,
                &self.coupling_idx,
                &self.coupling_offsets,
                &self.oscillators,
                i,
            );
            // FIX-D: Pre-generate all 5 Gaussian samples before the update loop.
            // This reduces calls to next_gaussian() from 5/node to ceil(5/2)=3/node
            // (Box-Muller spare sample), and makes the branch predictor pattern
            // for self access more predictable.
            let noise_buf = {
                let mut buf = [0.0f64; 5];
                for n in buf.iter_mut() {
                    *n = self.next_gaussian();
                }
                buf
            };
            for (g, coupling_sum) in coupling_sums.iter().enumerate() {
                let omega = self.oscillators[i].frequencies[g];
                self.oscillators[i].phases[g] +=
                    (omega + coupling_sum) * dt + sqrt_2k_t_dt * noise_buf[g];
            }
        }
    }

    /// Parámetro de orden de Kuramoto r = |Σ e^{iφ}| / N ∈ `[0,1]`.
    ///
    /// Resultado cacheado: recomputa solo si `step()` fue llamado desde la
    /// última invocación. En bucles de control que llaman este método sin
    /// llamar `step()` entre medias, costo = O(1).
    ///
    /// # Costo
    /// - Primera llamada tras `step()`: O(N)
    /// - Llamadas subsecuentes sin `step()` intermedio: O(1)
    ///
    /// AX-ID: AXIOMA-006
    pub fn synchrony_order_cached(&mut self) -> f64 {
        if self.sync_dirty {
            self.sync_cache = self.compute_synchrony_order_internal();
            self.sync_dirty = false;
        }
        self.sync_cache
    }

    fn compute_synchrony_order_internal(&self) -> f64 {
        let n = self.oscillators.len();
        if n == 0 {
            return 0.0;
        }
        const N_GRADES: usize = 5;
        const EPS: f64 = 1e-30;
        let mut r_total = 0.0f64;
        for g in 0..N_GRADES {
            let (sc, ss, sa) =
                self.oscillators
                    .iter()
                    .fold((0.0f64, 0.0f64, 0.0f64), |(sc, ss, sa), osc| {
                        let a = osc.amplitudes[g];
                        (
                            sc + a * osc.phases[g].cos(),
                            ss + a * osc.phases[g].sin(),
                            sa + a,
                        )
                    });
            // |Σ A·e^{iφ}| / (Σ A) — idéntico al clásico cuando A_i = 1.0 ∀i
            r_total += sc.hypot(ss) / (sa + EPS);
        }
        #[allow(clippy::cast_precision_loss)]
        {
            r_total / N_GRADES as f64
        }
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    fn lookup_idx(&self, id: NodeId) -> Option<usize> {
        let raw =
            usize::try_from(id.get()).expect("NodeId must fit into usize on supported targets");
        if raw >= self.id_to_idx.len() {
            return None;
        }
        let idx = self.id_to_idx[raw];
        if idx == u32::MAX {
            None
        } else {
            Some(idx as usize)
        }
    }

    /// Reconstruye `coupling_idx` sorted by source oscilador index y materializa offsets.
    fn rebuild_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        self.coupling_idx.clear();
        self.coupling_offsets.resize(self.oscillators.len(), (0, 0));

        for &(ni, nj, gamma) in &self.coupling {
            if let (Some(ii), Some(ij)) = (self.lookup_idx(ni), self.lookup_idx(nj)) {
                self.coupling_idx.push((
                    u32::try_from(ii).expect("index must stay below u32::MAX"),
                    u32::try_from(ij).expect("index must stay below u32::MAX"),
                    gamma,
                ));
            }
        }
        self.coupling_idx.sort_unstable_by_key(|&(i, _, _)| i);

        let mut cursor = 0usize;
        for (node_idx, offsets) in self.coupling_offsets.iter_mut().enumerate() {
            let start = cursor;
            while cursor < self.coupling_idx.len() && {
                // SAFETY: coupling_idx almacena índices internos `u32` creados a partir
                // del índice de oscilador. En targets LP64/LLP64, u32 → usize es exacto.
                #[allow(clippy::cast_possible_truncation)]
                let idx = self.coupling_idx[cursor].0 as usize;
                idx == node_idx
            } {
                cursor += 1;
            }
            *offsets = (start, cursor);
        }

        self.dirty = false;
    }

    /// Number of coupling edges currently registered.
    ///
    /// Useful for verifying that `remove_oscillator` correctly purges edges.
    pub fn coupling_count(&self) -> usize {
        self.coupling.len()
    }

    // ─── CRATE-004 prerequisite APIs (FIX-H) ────────────────────────────────

    /// Amplitude norm of the oscillator for `id`. O(1) via direct-index lookup.
    ///
    /// Delegates to `QuantumOscillator::amplitude_norm()` which returns
    /// `Σ_g amplitudes[g] / G` for G=5 grades.
    ///
    /// Returns 0.0 if the node is not registered.
    /// AX-ID: LEY_FUNDACIONAL §3.7, CRATE-004 prerequisite
    pub fn amplitude_norm(&self, id: NodeId) -> f64 {
        let raw = id.get() as usize;
        let idx = self
            .id_to_idx
            .get(raw)
            .copied()
            .filter(|&i| i != u32::MAX)
            .map(|i| i as usize);
        idx.and_then(|i| self.oscillators.get(i))
            .map_or(0.0, crate::oscillator::QuantumOscillator::amplitude_norm)
    }

    /// Set the same amplitude for all 5 grades of oscillator `id`.
    ///
    /// Used by `inelastic_concept_fusion` to normalise the surviving node's
    /// amplitude after absorbing the collapsed node.
    /// Clamps to [0.0, 1.0]. No-op if the node is not registered.
    /// Invalidates sync cache.
    ///
    /// AX-ID: LEY_FUNDACIONAL §3.7, CRATE-004 prerequisite
    pub fn set_amplitude_all_grades(&mut self, id: NodeId, amplitude: f64) {
        let raw = id.get() as usize;
        let idx = self
            .id_to_idx
            .get(raw)
            .copied()
            .filter(|&i| i != u32::MAX)
            .map(|i| i as usize);
        if let Some(i) = idx {
            if let Some(osc) = self.oscillators.get_mut(i) {
                osc.amplitudes = [amplitude.clamp(0.0, 1.0); 5];
                self.sync_dirty = true;
            }
        }
    }

    /// Remove oscillator `id` from all internal structures.
    ///
    /// Steps:
    /// 1. Invalidates `id_to_idx[raw]` → `u32::MAX`.
    /// 2. Removes all coupling edges that reference `id` from `coupling`.
    /// 3. Sets `dirty = true` so `rebuild_offsets()` is triggered on next step.
    /// 4. Marks tombstone in oscillators Vec (preserves index stability).
    ///
    /// Returns `Err(NodeNotFound)` if the node is not registered.
    /// Complexity: O(E) for coupling cleanup.
    ///
    /// AX-ID: LEY_FUNDACIONAL §3.7 (WormholeCollapse), CRATE-004 prerequisite
    pub fn remove_oscillator(&mut self, id: NodeId) -> Result<(), GenesisError> {
        let raw = id.get() as usize;
        let idx = self
            .id_to_idx
            .get(raw)
            .copied()
            .filter(|&i| i != u32::MAX)
            .map(|i| i as usize)
            .ok_or(GenesisError::NodeNotFound { id })?;

        // Step 1: Invalidate direct index.
        if raw < self.id_to_idx.len() {
            self.id_to_idx[raw] = u32::MAX;
        }

        // Step 2: Remove coupling edges referencing this oscillator.
        // Remove coupling edges referencing this node.
        // coupling stores (NodeId, NodeId, f64).
        self.coupling.retain(|&(ni, nj, _)| ni != id && nj != id);

        // Step 3: Mark offsets as dirty — rebuild_offsets() triggered on next step.
        self.dirty = true;
        self.sync_dirty = true;

        // Step 4: Tombstone the oscillator slot (zero amplitudes, phases).
        // Do NOT swap-remove — that would invalidate all coupling indices.
        if let Some(osc) = self.oscillators.get_mut(idx) {
            osc.amplitudes = [0.0; 5];
            osc.phases = [0.0; 5];
            // Mark as pruned so it is skipped in step loops.
            // Tombstone: mark as Pruned at timestamp 0 to skip in step loops.
            osc.state = crate::oscillator::OscillatorState::Pruned { at_ns: 0 };
        }

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::uninlined_format_args,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use crate::synchrony::synchrony_order;

    fn make_osc(id: u64, freq: f64) -> QuantumOscillator {
        QuantumOscillator::new(
            NodeId::try_new(id).expect("NodeId válido por construcción"),
            [freq; 5],
        )
    }

    #[test]
    fn add_oscillator_rejects_duplicate_node_in_release() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let node_id = NodeId::try_new(7).expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::new(node_id, [0.1; 5]))
            .expect("primer insert debe funcionar");

        let err = net
            .add_oscillator(QuantumOscillator::new(node_id, [0.2; 5]))
            .expect_err("NodeId duplicado debe rechazarse en runtime");

        assert_eq!(err, GenesisError::KuramotoDuplicateNodeId { raw: 7 });
        assert_eq!(net.node_count(), 1);
        assert!((net.oscillators[0].frequencies[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn coupling_offsets_cover_all_sorted_edges() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        for i in 0..4_u64 {
            net.add_oscillator(make_osc(i, 0.1))
                .expect("NodeId válido por construcción");
        }
        net.set_coupling(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            0.5,
        );
        net.set_coupling(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            NodeId::try_new(2).expect("NodeId válido por construcción"),
            0.5,
        );
        net.set_coupling(
            NodeId::try_new(2).expect("NodeId válido por construcción"),
            NodeId::try_new(3).expect("NodeId válido por construcción"),
            0.5,
        );

        net.rebuild_if_dirty();
        assert_eq!(net.coupling_offsets.len(), net.node_count());

        let counted: usize = net
            .coupling_offsets
            .iter()
            .map(|(s, e)| e.saturating_sub(*s))
            .sum();
        assert_eq!(counted, net.coupling_idx.len());
    }

    #[test]
    fn kuramoto_step_updates_phases() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        net.add_oscillator(make_osc(0, 1.0))
            .expect("NodeId válido por construcción");

        let before = net.oscillators[0].phases[0];
        net.step(0.1);
        // ω = 1.0, dt = 0.1 → Δφ = 0.1 (sin ruido, sin acoplamiento)
        let after = net.oscillators[0].phases[0];
        assert!(
            (after - before - 0.1).abs() < 1e-12,
            "expected Δφ ≈ 0.1, got {}",
            after - before
        );
    }

    #[test]
    fn kuramoto_noise_is_nonzero_with_nonzero_temperature() {
        let mut net = QuantumKuramotoNetwork::new(1.0);
        // ω = 0, sin acoplamiento → todo el cambio viene del ruido
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        net.step(0.01);
        let phase_after = net.oscillators[0].phases[0];
        assert_ne!(
            phase_after, 0.0,
            "Con T>0 y dt>0 el ruido térmico debe cambiar las fases"
        );
    }

    #[test]
    fn kuramoto_coupling_induces_synchronization_multigrade() {
        let n = 10usize;
        let mut net = QuantumKuramotoNetwork::new(0.05);

        for i in 0..n {
            let phase = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            let freqs = [0.0, 0.05, 0.10, 0.15, 0.20];
            let osc = QuantumOscillator::with_phases(
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                [phase; 5],
                freqs,
            );
            net.add_oscillator(osc)
                .expect("NodeId válido por construcción");
        }

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    net.set_coupling(
                        NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                        NodeId::try_new(j as u64).expect("NodeId válido por construcción"),
                        2.0,
                    );
                }
            }
        }

        for _ in 0..2000 {
            net.step(0.02);
        }

        let r = synchrony_order(&net);
        assert!(
            r > 0.6,
            "r_sync multigrade = {:.4} — acoplamiento K>>Kc debe inducir sincronización global",
            r
        );
    }

    #[test]
    fn step_deterministic_and_noisy_paths_identical_at_zero_temperature() {
        let setup = || {
            let mut net = QuantumKuramotoNetwork::new(0.0);
            for i in 0..5u64 {
                let phase = i as f64 * 0.3;
                net.add_oscillator(QuantumOscillator::with_phases(
                    NodeId::try_new(i).expect("NodeId válido por construcción"),
                    [phase; 5],
                    [0.5; 5],
                ))
                .expect("NodeId válido por construcción");
            }
            for i in 0..5 {
                for j in 0..5 {
                    if i != j {
                        net.set_coupling(
                            NodeId::try_new(i).expect("NodeId válido por construcción"),
                            NodeId::try_new(j).expect("NodeId válido por construcción"),
                            0.8,
                        );
                    }
                }
            }
            net
        };
        let mut net1 = setup();
        let mut net2 = setup();
        net1.step(0.01);
        net2.step(0.01);
        for i in 0..5 {
            for g in 0..5 {
                let p1 = net1.oscillators[i].phases[g];
                let p2 = net2.oscillators[i].phases[g];
                assert!(
                    (p1 - p2).abs() < 1e-14,
                    "paths divergen en nodo {i} grade {g}: {p1:.15} vs {p2:.15}"
                );
            }
        }
    }

    #[test]
    fn set_coupling_sorted_invariant() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        for i in 0..5u64 {
            net.add_oscillator(QuantumOscillator::new(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                [0.0; 5],
            ))
            .expect("NodeId válido por construcción");
        }
        net.set_coupling(
            NodeId::try_new(4).expect("NodeId válido por construcción"),
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            0.5,
        );
        net.set_coupling(
            NodeId::try_new(2).expect("NodeId válido por construcción"),
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            0.3,
        );
        net.set_coupling(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            NodeId::try_new(3).expect("NodeId válido por construcción"),
            0.7,
        );

        let keys: Vec<_> = net
            .coupling
            .iter()
            .map(|&(a, b, _)| (a.get(), b.get()))
            .collect();
        let sorted = {
            let mut k = keys.clone();
            k.sort_unstable();
            k
        };
        assert_eq!(
            keys, sorted,
            "coupling debe mantenerse sorted para O(log E) lookup"
        );
    }

    #[test]
    fn set_coupling_update_existing_no_duplicate() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        net.set_coupling(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            0.5,
        );
        net.set_coupling(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            0.9,
        );
        assert_eq!(net.coupling.len(), 1);
        assert!((net.coupling[0].2 - 0.9).abs() < 1e-12);
    }

    #[test]
    fn set_coupling_gamma_zero_removes_edge() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        net.set_coupling(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            0.5,
        );
        assert_eq!(net.coupling.len(), 1);
        net.set_coupling(
            NodeId::try_new(0).expect("NodeId válido por construcción"),
            NodeId::try_new(1).expect("NodeId válido por construcción"),
            0.0,
        );
        assert_eq!(net.coupling.len(), 0, "gamma=0.0 debe eliminar la arista");
    }

    #[test]
    fn phase_diff_norm_symmetric() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let i = NodeId::try_new(0).expect("NodeId válido por construcción");
        let j = NodeId::try_new(1).expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(
            i,
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(
            j,
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        assert_eq!(net.phase_diff_norm(i, j), net.phase_diff_norm(j, i));
    }

    #[test]
    fn phase_diff_norm_returns_nan_for_unknown_node() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let i = NodeId::try_new(0).expect("NodeId válido por construcción");
        let j = NodeId::try_new(1).expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(i, [0.0; 5], [0.0; 5]))
            .expect("NodeId válido por construcción");

        assert!(net.phase_diff_norm(i, j).is_nan());
        assert!(net.phase_diff_norm(j, i).is_nan());
    }

    #[test]
    fn phase_diff_norm_is_symmetric() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let i = NodeId::try_new(7).expect("NodeId válido por construcción");
        let j = NodeId::try_new(8).expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(
            i,
            [0.25, 0.5, 0.75, 1.0, 1.25],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(
            j,
            [1.25, 1.0, 0.75, 0.5, 0.25],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");

        let a = net.phase_diff_norm(i, j);
        let b = net.phase_diff_norm(j, i);
        assert!((a - b).abs() <= 1e-12, "a={}, b={}", a, b);
    }

    #[test]
    fn phase_diff_norm_zero_for_same_node() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let i = NodeId::try_new(42).expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(
            i,
            [1.5, 2.5, 3.5, 4.5, 5.5],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        assert_eq!(net.phase_diff_norm(i, i), 0.0);
    }

    #[test]
    fn phase_diff_norm_correct_value() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let i = NodeId::try_new(0).expect("NodeId válido por construcción");
        let j = NodeId::try_new(1).expect("NodeId válido por construcción");
        // Diferencia = [1,1,1,1,1] → sum_sq = 5 → RMS = sqrt(5/5) = 1.0
        net.add_oscillator(QuantumOscillator::with_phases(i, [0.0; 5], [0.0; 5]))
            .expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(j, [1.0; 5], [0.0; 5]))
            .expect("NodeId válido por construcción");
        let d = net.phase_diff_norm(i, j);
        assert!((d - 1.0).abs() < 1e-12, "expected 1.0, got {}", d);
    }

    #[test]
    fn phase_diff_norm_circular_invariant() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let i = NodeId::try_new(0).expect("NodeId válido por construcción");
        let j = NodeId::try_new(1).expect("NodeId válido por construcción");
        let two_pi = 2.0 * core::f64::consts::PI;
        // φᵢ = 0.1, φⱼ = 0.1 + 2π → misma posición en S¹ → diferencia debe ser ≈ 0
        net.add_oscillator(QuantumOscillator::with_phases(i, [0.1; 5], [0.0; 5]))
            .expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(
            j,
            [0.1 + two_pi; 5],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        assert!(
            net.phase_diff_norm(i, j) < 1e-10,
            "fases que difieren por 2π deben tener distancia ≈ 0, got {}",
            net.phase_diff_norm(i, j)
        );
    }

    #[test]
    fn phase_diff_norm_max_is_pi() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        let i = NodeId::try_new(0).expect("NodeId válido por construcción");
        let j = NodeId::try_new(1).expect("NodeId válido por construcción");
        // φᵢ = 0, φⱼ = π → distancia máxima en S¹
        net.add_oscillator(QuantumOscillator::with_phases(i, [0.0; 5], [0.0; 5]))
            .expect("NodeId válido por construcción");
        net.add_oscillator(QuantumOscillator::with_phases(
            j,
            [core::f64::consts::PI; 5],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        let d = net.phase_diff_norm(i, j);
        assert!(
            (d - core::f64::consts::PI).abs() < 1e-10,
            "distancia máxima circular debe ser π, got {}",
            d
        );
    }

    #[test]
    fn wrap_phase_diff_property() {
        let two_pi = core::f64::consts::TAU;
        let pi = core::f64::consts::PI;

        // Bordes exactos y equivalentes periódicos.
        let edge_cases = [-pi, pi, -3.0 * pi, 3.0 * pi, -two_pi, two_pi, 0.0, f64::NAN];
        for &d in &edge_cases {
            let w = wrap_phase_diff(d);
            if d.is_nan() {
                assert!(w.is_nan(), "NaN debe propagarse");
            } else {
                assert!(w >= -pi && w < pi, "wrap fuera de rango [-π, π): {}", w);
            }
        }

        // 10k casos pseudoaleatorios con invarianza periódica y rango.
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..10_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            let d = (unit - 0.5) * 2.0 * 1_000.0 * two_pi;
            let w = wrap_phase_diff(d);
            assert!(w >= -pi && w < pi, "wrap fuera de rango [-π, π): {}", w);

            let wp = wrap_phase_diff(d + 37.0 * two_pi);
            assert!(
                (w - wp).abs() <= 1e-12,
                "invarianza periódica rota: d={} wrap={} wrap+2πk={}",
                d,
                w,
                wp
            );
        }
    }

    #[test]
    fn gaussian_noise_mean_and_variance() {
        // Verificar E[η] ≈ 0 y Var[η] ≈ 1 para N=10000 muestras.
        let mut rng = 0xdeadbeef_u64;
        let n = 10_000usize;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for _ in 0..n {
            let x = gaussian_noise(&mut rng);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "E[η] = {:.4} debe ser ≈ 0", mean);
        assert!(
            (variance - 1.0).abs() < 0.05,
            "Var[η] = {:.4} debe ser ≈ 1",
            variance
        );
    }

    #[test]
    fn gaussian_spare_reduces_trig_calls() {
        let mut net = QuantumKuramotoNetwork::new(0.1);
        let first = net.next_gaussian();
        let rng_after_first = net.rng_state;
        assert!(net.spare_gaussian.is_some());

        let second = net.next_gaussian();
        assert_eq!(net.rng_state, rng_after_first);
        assert!(net.spare_gaussian.is_none());

        // Debe seguir entregando muestras válidas sin degenerar.
        assert!(first.is_finite());
        assert!(second.is_finite());
        assert_ne!(first, second);
    }
}

#[cfg(test)]
mod prerequisite_api_tests {
    use super::*;
    use crate::oscillator::{OscillatorState, QuantumOscillator};
    use genesis_types::NodeId;

    fn node(raw: u64) -> NodeId {
        NodeId::try_new(raw).unwrap()
    }

    fn make_osc(id: NodeId) -> QuantumOscillator {
        QuantumOscillator::new(id, [0.1_f64; 5])
    }

    #[test]
    fn amplitude_norm_returns_zero_for_missing_node() {
        let net = QuantumKuramotoNetwork::new(0.01);
        assert_eq!(net.amplitude_norm(node(99)), 0.0);
    }

    #[test]
    fn amplitude_norm_returns_expected_value() {
        let mut net = QuantumKuramotoNetwork::new(0.01);
        let id = node(1);
        let mut osc = make_osc(id);
        osc.amplitudes = [0.5; 5];
        net.add_oscillator(osc).unwrap();
        // amplitude_norm = sum(amplitudes) / 5.0 = 2.5 / 5.0 = 0.5
        assert!((net.amplitude_norm(id) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn set_amplitude_all_grades_clamps_and_applies() {
        let mut net = QuantumKuramotoNetwork::new(0.01);
        let id = node(2);
        net.add_oscillator(make_osc(id)).unwrap();
        // Set beyond 1.0 — should clamp
        net.set_amplitude_all_grades(id, 1.5);
        assert!((net.amplitude_norm(id) - 1.0).abs() < 1e-10);
        // Set to 0.3
        net.set_amplitude_all_grades(id, 0.3);
        assert!((net.amplitude_norm(id) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn set_amplitude_noop_for_missing_node() {
        let mut net = QuantumKuramotoNetwork::new(0.01);
        // Should not panic
        net.set_amplitude_all_grades(node(999), 0.5);
    }

    #[test]
    fn remove_oscillator_returns_err_for_missing_node() {
        let mut net = QuantumKuramotoNetwork::new(0.01);
        assert!(net.remove_oscillator(node(999)).is_err());
    }

    #[test]
    fn remove_oscillator_removes_and_marks_pruned() {
        let mut net = QuantumKuramotoNetwork::new(0.01);
        let id = node(7);
        net.add_oscillator(make_osc(id)).unwrap();
        assert_eq!(net.amplitude_norm(id), 1.0); // active
        net.remove_oscillator(id).unwrap();
        // After removal: amplitude_norm should return 0.0 (Pruned oscillator has zero amplitudes)
        assert_eq!(net.amplitude_norm(id), 0.0);
    }

    #[test]
    fn remove_oscillator_purges_coupling_edges() {
        let mut net = QuantumKuramotoNetwork::new(0.01);
        let a = node(10);
        let b = node(11);
        net.add_oscillator(make_osc(a)).unwrap();
        net.add_oscillator(make_osc(b)).unwrap();
        net.set_coupling(a, b, 0.5);
        assert_eq!(net.coupling_count(), 1);
        net.remove_oscillator(a).unwrap();
        // Coupling referencing a should be removed
        assert_eq!(net.coupling_count(), 0);
    }
}
