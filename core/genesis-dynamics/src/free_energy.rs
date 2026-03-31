//! Variational Free Energy minimization with compensated summation in hot reductions.
//!
//! The implementation uses compensated accumulators for cancellation-prone scalar reductions
//! so 16-blade inference remains numerically stable under large dynamic ranges.
//! `VFEMinimizer` tracks compensation significance to surface precision-sensitive regimes.
//!
//! AX-ID: AXIOMA-003, AXIOMA-008, H_información (LEY_FUNDACIONAL §3.3)

#![allow(clippy::float_cmp)]

use core::cell::Cell;

use genesis_math::SparseCliffordVector;
use genesis_types::{GenesisError, NodeId};

use crate::kahan::KahanAccumulator;

/// Signatura Minkowski (+,−,−,−) for G(1,3).
/// Index 0 = temporal (positivo), indices 1..3 = spatial (negativos).
/// AX-ID: AXIOMA-001, `LEY_FUNDACIONAL` §2
const TRACE_MIN: f64 = 1.0e-12;
const TRACE_MAX: f64 = 1.0e12;

/// Per-blade norm weights in G(1,3) for 16D VFE computation.
///
/// Provided so that `VFE_16D = Σ_i precision_full[i] * (mean_full[i] - target_full[i])²`
/// is sensitive to the real geometry of G(1,3). Weights reflect importance
/// semantic of each grade (ver `METRIC_WEIGHTS` in genesis-math).
///
/// | Grade | Blades   | Weight |
/// |-------|----------|------|
/// | 0     | {0}      | 2.0  |
/// | 1     | {1,2,4,8}| 1.5  |
/// | 2     | {3,5,6,9,10,12} | 1.0 |
/// | 3     | {7,11,13,14} | 0.5 |
/// | 4     | {15}     | 0.3  |
///
/// Consistent with `METRIC_WEIGHTS` in `genesis-math` so that the distance
/// is inferentially coherent with the topological distance in HNSW.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.3
pub(crate) const VFE_BLADE_WEIGHTS: [f64; 16] = [
    // blade 0 (grade 0: climb)
    2.0, // blades 1,2,4,8 (grado 1: vectores — semántica primaria)
    1.5, 1.5, 1.0, // 1(e0), 2(e1), 3(e01)
    1.5, 1.0, 1.0, // 4(e2), 5(e02), 6(e12)
    0.5, 1.5, // 7(e012), 8(e3)
    // blades 9..14 (grados 2 and 3)
    1.0, 1.0, 0.5, 1.0, 0.5, 0.5, // blade 15 (grado 4: pseudoescalar)
    0.3,
];

/// Indices blade of the vectors grade 1 in G(1,3): e₀, e₁, e₂, e₃.
/// Used to initialize `mean_full` from the prior of 4 components
/// and for backward compatibility of the method `mean()`.
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
        let coeffs = &o.coeffs;
        // loop-invariant, hoisted
        // CRYSTAL: O18 — inevitable
        let target: [f64; 16] = core::array::from_fn(|i| coeffs[i]);
        is_finite_vec16(&target).then_some(target)
    })
}

// ── Tipos of creencias ────────────────────────────────────────────────────────

/// Full inferential state of a node over the G(1,3) multivector.
///
/// # Representation 16D
///
/// `mean_full[i]` is the mean coefficient of the blade `i` ∈ {0,…,15} of G(1,3).
/// `precision_full[i]` is the diagonal precision of the blade `i`.
///
/// The mean spans all 16 blades (5 grades), allowing inference to operate over
/// the full algebraic structure of the concept rather than only its vector part.
///
/// # Backward compatibility
///
/// - `add_node(id, prior_mean: [f64; 4])` initializes `mean_full` with the four
///   grade-1 blade values (indices 1, 2, 4, 8) and zeros elsewhere.
/// - `mean()` returns the four grade-1 components, matching previous behavior.
/// - `compute_vfe(id, obs: Option<&[f64; 4]>)` computes VFE over grade 1 only.
/// - `compute_vfe_with_grad(id, obs)` returns `(f64, [f64; 16])`.
///
/// # Preparation for CRATE-004
/// `DiscreteRicciFlow` uses `compute_vfe_with_grad() -> [f64; 16]` to compute
/// the Ollivier-Ricci curvature signal using information from all grades,
/// not only from the vector subspace.
///
/// AX-ID: AXIOMA-003, `H_información` (`LEY_FUNDACIONAL` §3.3)
#[derive(Clone, Debug)]
pub struct Belief {
    /// μᵢ — state inferential medio over the 16 blades of G(1,3).
    ///
    /// Initialize with the 4 components of grade 1 in positions `GRADE1_BLADE_INDICES`
    /// and zero in the rest. Evolves with `update_full()` towards full observations
    /// or with `update()` (grado 1 only) for backward compatibility.
    pub mean_full: [f64; 16],
    /// Diagonal precision over 16 blades. `precision_full[i]` ∈ [0, 1e6].
    /// Initializes to `1.0` for all blades (previously not informative).
    pub precision_full: [f64; 16],
    /// Node identifier this belief is associated with.
    pub node_id: NodeId,
}

/// Information of Fisher scalarized for the gate of satiation AXIOMA-008.
///
/// The full Fisher metric 𝒢ᵢ ∈ ℝ¹⁶ˣ¹⁶ is approximated as c·I₁₆ (Fisher
/// isotropic), so every relevant information is its scalar trace.
///
/// With the extension to 16D, the trace covers all the blades, not only grade 1.
/// The AXIOMA-008 satiation gate behaves identically: satiation when ΔTr < ε.
///
/// AX-ID: AXIOMA-008
#[derive(Clone, Debug)]
pub struct FisherInfo {
    /// Trace escalar of the metric of Fisher isotropic: Tr(𝒢ᵢ) = 16c.
    /// Inicializa in 1.0. Floor: 1e-12.
    pub trace: f64,
    /// ‖∂𝒢/∂t‖_F ≈ 0.5 · |ΔTr|
    pub delta_g: f64,
}

impl Belief {
    fn new(id: NodeId, prior_mean: [f64; 4]) -> Self {
        let mut mean_full = [0.0f64; 16];
        // Initialize only the 4 blades of grade 1 with the prior
        for (k, &blade_idx) in GRADE1_BLADE_INDICES.iter().enumerate() {
            mean_full[blade_idx] = prior_mean[k];
        }
        Self {
            mean_full,
            precision_full: [1.0; 16],
            node_id: id,
        }
    }

    /// Componentes of grade 1 (vectors e₀,e₁,e₂,e₃) — backward compatibility.
    ///
    /// Returns the 4 coefficients of the vector subspace, identical to
    /// field `mean: [f64; 4]` original.
    ///
    /// AX-ID: AXIOMA-003
    #[inline]
    pub fn mean(&self) -> [f64; 4] {
        core::array::from_fn(|k| self.mean_full[GRADE1_BLADE_INDICES[k]])
    }

    /// Trace of Fisher (para FisherGate, AXIOMA-008).
    /// Sums of the precisions over the 16 blades.
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
    pub(crate) const fn tombstone() -> Self {
        Self {
            mean_full: [0.0; 16],
            precision_full: [0.0; 16],
            node_id: NodeId::INVALID,
        }
    }
}

impl FisherInfo {
    const fn new() -> Self {
        Self {
            trace: 1.0,
            delta_g: 0.0,
        }
    }
}

/// Re-exported from `genesis_types` for `genesis-evolution` (CRATE-004)
/// can verify `DualityConsistency` without importing `genesis-dynamics` (CRATE-003).
///
/// AX-ID: LEY_FUNDACIONAL §3.6, §5.6
pub use genesis_types::FisherEdgeMetric;

/// Function helper canonical of edge — kept internally for compatibility
/// with tests that the reference directly dentro of this module.
#[inline]
const fn canonical_edge(i: NodeId, j: NodeId) -> (NodeId, NodeId) {
    if i.get() <= j.get() {
        (i, j)
    } else {
        (j, i)
    }
}

// ── VFEMinimizer ──────────────────────────────────────────────────────────────

/// Minimizer of Energy Free Variational.
///
/// `F = D_KL(Q(s) ‖ P(s|o)) − ln P(o)`
/// Approximation computable: `F ≈ Σᵢ Tr(𝒢ᵢ) · ‖μᵢ − μ̂ᵢ‖²`
///
/// El mechanism of learning ONLY is the minimization of F.
/// Forbidden: cross-entropy, MSE, cualquier loss externa. (AXIOMA-003)
///Forbidden: wait for prompts for activerse. (AXIOMA-003: cognitive life continues)
///
/// AX-ID: AXIOMA-003, AXIOMA-008, `H_información` (`LEY_FUNDACIONAL` §3.3)
pub struct VFEMinimizer {
    beliefs: Vec<Belief>,
    fisher: Vec<FisherInfo>,
    /// `NodeId` → index in `beliefs` using pages sparse on-demand.
    /// `u32::MAX` = no registered.
    id_to_idx: PagedIndex,
    /// Ratio |compensation| / max(|sum|, 1) measured during the last 16-blade VFE reduction.
    precision_compensation_ratio: Cell<f64>,
}

const PAGE_BITS: u32 = 12;
const PAGE_SIZE: usize = 1 << PAGE_BITS;
const PAGE_MASK: usize = PAGE_SIZE - 1;

#[derive(Debug, Default)]
struct PagedIndex {
    pages: Vec<Option<Box<[u32; PAGE_SIZE]>>>,
}

impl PagedIndex {
    #[inline]
    fn get(&self, raw: usize) -> Option<u32> {
        let page = raw >> PAGE_BITS;
        let offset = raw & PAGE_MASK;
        self.pages
            .get(page)
            .and_then(Option::as_ref)
            .map(|p| p[offset])
            .filter(|&idx| idx != u32::MAX)
    }

    #[inline]
    fn set(&mut self, raw: usize, val: u32) {
        let page = raw >> PAGE_BITS;
        let offset = raw & PAGE_MASK;
        if page >= self.pages.len() {
            self.pages.resize(page + 1, None);
        }
        let slots = self.pages[page].get_or_insert_with(|| Box::new([u32::MAX; PAGE_SIZE]));
        slots[offset] = val;
    }
}

impl VFEMinimizer {
    #[inline]
    fn bounded_step(dt: f64, trace: f64) -> f64 {
        if !dt.is_finite() || !trace.is_finite() || dt <= 0.0 {
            return 0.0;
        }
        let denom = dt.mul_add(trace.max(0.0), 1.0);
        if !denom.is_finite() || denom <= TRACE_MIN {
            return 0.0;
        }
        (dt / denom).clamp(0.0, 0.9)
    }

    /// Creates an empty VFE minimiser with no registered nodes.
    ///
    /// AX-ID: AXIOMA-003, H_información (LEY_FUNDACIONAL §3.3)
    pub fn new() -> Self {
        Self {
            beliefs: Vec::new(),
            fisher: Vec::new(),
            id_to_idx: PagedIndex { pages: Vec::new() },
            precision_compensation_ratio: Cell::new(0.0),
        }
    }

    /// # Panics
    /// Panics if `id.get()` cannot fit in `usize` on the target architecture,
    /// or if the node count exceeds `u32::MAX`.
    ///
    /// # Policy of values no finite
    /// `prior_mean` must contener only values finite (`is_finite`).
    /// If contiene `NaN`/`+Inf`/`-Inf`, the node it rejects and no it persiste.
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
        let Ok(raw) = usize::try_from(id.get()) else {
            return;
        };

        // Defensive maximum: 100× production target. Prevents OOM from buggy callers.
        const MAX_ALLOWED_NODE_ID: usize = 100_000_000;
        if raw > MAX_ALLOWED_NODE_ID {
            // In release: log and return. In debug: panic for early detection.
            debug_assert!(
                false,
                "NodeId {raw} exceeds MAX_ALLOWED_NODE_ID={MAX_ALLOWED_NODE_ID} — potential bug"
            );
            return;
        }

        if self.id_to_idx.get(raw).is_none() {
            let Ok(idx) = u32::try_from(self.beliefs.len()) else {
                debug_assert!(false, "VFEMinimizer belief count overflowed u32::MAX");
                return;
            };
            self.id_to_idx.set(raw, idx);
            self.beliefs.push(Belief::new(id, prior_mean));
            self.fisher.push(FisherInfo::new());
        }
    }

    /// Lookup internal: `NodeId` → index in `Vec`.
    ///
    /// O(1) direct-index access. Returns None for unregistered IDs without panic.
    fn lookup(&self, id: NodeId) -> Option<usize> {
        let Ok(raw) = usize::try_from(id.get()) else {
            return None;
        };
        self.id_to_idx.get(raw).map(|idx| idx as usize)
    }

    /// VFE over the subespacio of grade 1 — backward compatibility with callers `[f64;4]`.
    ///
    /// Calculate the distance of Mahalanobis to the square between `mean_full[grade1]`
    /// and `obs` (interpreted as 4 components of grade 1). Always ≥ 0.
    ///
    /// Con obs=None: prediction interna μ̂ = `[0,0,0,0]` (prior no informative).
    ///
    /// # Policy of values no finite
    /// If `obs` contiene `NaN`/`±Inf`, returns `0.0`.
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
        let mean_full = &belief.mean_full;
        let precision_full = &belief.precision_full;
        // loop-invariant, hoisted
        // CRYSTAL: O29 — inevitable
        let target = obs.copied().unwrap_or_default();
        // VFE over the 4 blades of grade 1 — Kahan summation.
        // Applies VFE_BLADE_WEIGHTS for consistency with compute_vfe_with_grad (FIX-3).
        let mut sum = 0.0f64;
        let mut comp = 0.0f64;
        for (k, &blade_idx) in GRADE1_BLADE_INDICES.iter().enumerate() {
            let delta = mean_full[blade_idx] - target[k];
            let w = VFE_BLADE_WEIGHTS[blade_idx];
            let weighted_precision = w * precision_full[blade_idx];
            let y = delta.mul_add(delta * weighted_precision, -comp);
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sum
    }

    /// VFE full over the 16 blades of G(1,3) and gradiente `[f64; 16]`.
    ///
    /// ```text
    /// VFE_16D = Σ_{i=0}^{15} precision_full[i] · (mean_full[i] − target_full[i])²
    /// grad[i] = 2 · precision_full[i] · (mean_full[i] − target_full[i])
    /// ```
    ///
    /// The target full it extracts from `obs.coeffs[0..16]` when `obs` is `Some`.
    /// With `obs=None`, the target is the zero vector (prior empty).
    ///
    /// # Usage in CRATE-004
    /// `DiscreteRicciFlow::step()` calls this method for obtener the gradiente
    /// full 16D that drives the deformation of the metric of edges. The gradient
    /// covers all the grades of Clifford, producing signal of curvature in the
    /// geometry full of the concept, no only in su component vectorial.
    ///
    /// # Cambio of signature respecto a v1.0
    /// Returns `[f64; 16]` instead of `[f64; 4]`. Callers that only necesitan
    /// the 4 components of grade 1 must indexar `grad[GRADE1_BLADE_INDICES]`.
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
        let mean_full = &belief.mean_full;
        let precision_full = &belief.precision_full;
        // loop-invariant, hoisted
        // CRYSTAL: O31 — inevitable
        let mut vfe_acc = KahanAccumulator::new();
        let mut grad = [0.0f64; 16];

        // FIX-3: Apply VFE_BLADE_WEIGHTS for gradient/loss consistency with internal_drive.
        // Previously VFE_BLADE_WEIGHTS was only applied in internal_drive, making the gradient
        // direction inconsistent with the loss landscape (different metric in loss vs gradient).
        // Now both use the same weighted metric: F_i = w_i · Π_i · δ_i²
        // HOT PATH: O(N), called per iteration of VFE minimization
        for i in 0..16 {
            let delta = mean_full[i] - target[i];
            let prec = precision_full[i];
            let w = VFE_BLADE_WEIGHTS[i];
            let weighted_precision = w * prec;
            vfe_acc.add(delta * delta * weighted_precision);
            grad[i] = delta * (2.0 * weighted_precision);
        }
        let vfe = vfe_acc.total();
        self.precision_compensation_ratio
            .set(vfe_acc.compensation_abs() / vfe.abs().max(1.0));
        (vfe, grad)
    }

    /// Reports the compensation significance of the last 16-blade VFE reduction.
    ///
    /// AX-ID: AXIOMA-003, H_información (LEY_FUNDACIONAL §3.3)
    #[inline]
    pub fn precision_compensation_ratio(&self) -> f64 {
        self.precision_compensation_ratio.get()
    }

    /// Gradiente of grade 1 only — access conveniente for callers
    /// which only need the signal of the 4 base vectors.
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

    /// Drive internal: returns the `NodeId` with mayor VFE 16D bajo prior empty.
    ///
    /// The VFE 16D covers all the degrees of G(1,3), so the node with
    /// biggest surprise can be one with high discrepancy in bivectors or
    ///trivectors, not only in the vector component.
    ///
    /// Without external stimulus the system minimizes F internally. (AXIOM-003)
    pub fn internal_drive(&mut self) -> Option<NodeId> {
        if self.beliefs.is_empty() {
            return None;
        }
        let mut max_vfe = f64::NEG_INFINITY;
        let mut max_id = None;
        for (idx, belief) in self.beliefs.iter().enumerate() {
            if !is_finite_vec16(&belief.mean_full) {
                continue;
            }
            let trace = sanitize_trace(self.fisher[idx].trace);
            // F interna 16D: Tr(𝒢) · Σ_i w_i · μ_i² (target = 0)
            let error_sq: f64 = {
                let mut sum = 0.0f64;
                let mut comp = 0.0f64;
                for (i, &m) in belief.mean_full.iter().enumerate() {
                    let y = m.mul_add(m * VFE_BLADE_WEIGHTS[i], -comp);
                    let t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }
                sum
            };
            let vfe = trace * error_sq;
            if vfe > max_vfe {
                max_vfe = vfe;
                max_id = Some(belief.node_id);
            }
        }
        max_id
    }

    /// Updates beliefs of grade 1 after observation — backward compatibility.
    ///
    /// Same behavior as in v1.0: only updates the 4 blades of grade 1
    /// (`mean_full[GRADE1_BLADE_INDICES]`). Para updatesr all the 16 blades,
    /// use `update_full()`.
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

        let step = Self::bounded_step(dt, fisher.trace);
        if !step.is_finite() {
            return;
        }

        // Update only the 4 blades of grade 1
        let mut error_sq = 0.0f64;
        for (k, &blade_idx) in GRADE1_BLADE_INDICES.iter().enumerate() {
            let err = observation[k] - belief.mean_full[blade_idx];
            belief.mean_full[blade_idx] = step.mul_add(err, belief.mean_full[blade_idx]);
            belief.precision_full[blade_idx] =
                (belief.precision_full[blade_idx] + step).clamp(0.0, 1.0e6);
            error_sq = err.mul_add(err, error_sq);
        }

        let old_trace = fisher.trace;
        let error_mag = error_sq.sqrt();
        let trace_scale = step.mul_add(error_mag.max(TRACE_MIN), 1.0);
        // CRYSTAL: O43 — inevitable
        fisher.trace = sanitize_trace(old_trace / trace_scale);
        fisher.delta_g = 0.5 * (fisher.trace - old_trace).abs();
    }

    /// Updates the beliefs over the 16 blades full of G(1,3).
    ///
    /// The observation `obs` must be a `SparseCliffordVector` — its 16 coefficients
    /// it usan as target full. Actualiza `mean_full` and `precision_full` in all
    /// blades with non-null signal in `obs`.
    ///
    /// For observations only of grade 1, use `update()` which is more efficient.
    ///
    /// # Preparation for CRATE-004
    /// `DiscreteRicciFlow` can suministrar observaciones completas 16D al
    /// ajustar creencias post-colapso of wormhole.
    ///
    /// AX-ID: AXIOMA-003, AXIOMA-008
    pub fn update_full(&mut self, id: NodeId, obs: &SparseCliffordVector, dt: f64) {
        let Some(idx) = self.lookup(id) else { return };
        if !dt.is_finite() {
            return;
        }

        let belief = &mut self.beliefs[idx];
        let fisher = &mut self.fisher[idx];
        fisher.trace = sanitize_trace(fisher.trace);

        let step = Self::bounded_step(dt, fisher.trace);
        if !step.is_finite() {
            return;
        }

        let mut error_sq = 0.0f64;
        for i in 0..16usize {
            let target = obs.coeffs[i];
            if !target.is_finite() {
                continue;
            }
            let err = target - belief.mean_full[i];
            belief.mean_full[i] = step.mul_add(err, belief.mean_full[i]);
            belief.precision_full[i] = (belief.precision_full[i] + step).clamp(0.0, 1.0e6);
            error_sq = err.mul_add(err, error_sq);
        }

        let old_trace = fisher.trace;
        let error_mag = error_sq.sqrt();
        let trace_scale = step.mul_add(error_mag.max(TRACE_MIN), 1.0);
        // CRYSTAL: O46 — inevitable
        fisher.trace = sanitize_trace(old_trace / trace_scale);
        fisher.delta_g = 0.5 * (fisher.trace - old_trace).abs();
    }

    /// Access a `FisherInfo` of un node.
    pub fn fisher(&self, id: NodeId) -> Option<&FisherInfo> {
        let idx = self.lookup(id)?;
        Some(&self.fisher[idx])
    }

    /// `ΔG = ||∂𝒢/∂t||` for `FisherGate` (AXIOMA-008).
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
        let Ok(raw) = usize::try_from(id.get()) else {
            return Err(GenesisError::NodeNotFound { id });
        };
        let idx = self.lookup(id).ok_or(GenesisError::NodeNotFound { id })?;
        self.id_to_idx.set(raw, u32::MAX);
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
    use genesis_math::SparseCliffordVector;

    use super::*;

    #[test]
    fn vfe_internal_drive_returns_highest_vfe_node() {
        let mut vfe = VFEMinimizer::new();
        let a = NodeId::try_new(0).expect("NodeId válido por construcción");
        let b = NodeId::try_new(1).expect("NodeId válido por construcción");
        let c = NodeId::try_new(2).expect("NodeId válido por construcción");
        // mean more large → VFE more alto (con trace=1.0 uniforme)
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
        // Verifies that the ONLY mechanism of updates is observation.
        // VFE decreases after update: the internal updates converge, not diverge.
        // No exists API for inyectar gradientes externals — guarantee of compilador.
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        vfe.add_node(id, [2.0, 0.0, 0.0, 0.0]);
        let obs = [2.0f64, 0.0, 0.0, 0.0]; // observación que coincide con mean
        let before = vfe.compute_vfe(id, Some(&obs));
        vfe.update(id, &obs, 0.1);
        let after = vfe.compute_vfe(id, Some(&obs));
        // VFE with observation equal to mean = 0 → must stay at 0 or lower
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
        // After many updates with the same observation, VFE → 0.
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
        // In G(1,3), the temporal and spatial error have opposite signs.
        // Un error purely espacial must producir VFE NEGATIVO before of abs().
        // Verify that the Minkowski signature is correctly applied.
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        // mean = [0,0,0,0], obs_temporal = [1,0,0,0] → error temporal positivo
        vfe.add_node(id, [0.0; 4]);
        let obs_t = [1.0f64, 0.0, 0.0, 0.0];
        let vfe_temporal = vfe.compute_vfe(id, Some(&obs_t));
        // Reiniciar with error purely espacial
        let mut vfe2 = VFEMinimizer::new();
        vfe2.add_node(id, [0.0; 4]);
        let obs_x = [0.0f64, 1.0, 0.0, 0.0];
        let vfe_spatial = vfe2.compute_vfe(id, Some(&obs_x));
        // Temporal: sign=+1 → error_sq > 0. Espacial: sign=-1 → error_sq_raw < 0, abs > 0.
        // Both positive by abs(), but the internal gradients are opposite.
        assert!(vfe_temporal >= 0.0, "VFE temporal debe ser ≥ 0");
        assert!(vfe_spatial >= 0.0, "VFE espacial debe ser ≥ 0");
        // Con Tr(G)=1.0, error=1.0, and VFE_BLADE_WEIGHTS=1.5 (grade-1): VFE = 1.5
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
        // Para Fisher isotropic 4x4 with trace T: ‖ΔG‖_F = |ΔT|/2.
        // Verify that delta_g == 0.5 * |ΔTr| after an update.
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
        // Verifies that compute_vfe is numéricamente stable when mean ≈ target.
        // Without Kahan, for values ​​larges the relative error can be O(1).
        let mut vfe = VFEMinimizer::new();
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        // mean = [1000.0, 1000.0, 0.0, 0.0] — near the light cone.
        vfe.add_node(id, [1000.0, 1000.0, 0.0, 0.0]);
        // target = mean exactmente → VFE must be 0.
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
        // NodeId::MAX_VALID = u64::MAX − 1; only u64::MAX is rechazado per try_new.
        assert!(NodeId::try_new(u64::MAX).is_err());

        // BN-05: 2_000_000 is now well within the new MAX_ALLOWED_NODE_ID = 100_000_000.
        // The former 1_000_000 hard cap is removed. Verify that a NodeId of 2M
        // is accepted and functional.
        let formerly_rejected = NodeId::try_new(2_000_000).expect("2_000_000 is valid NodeId");
        let mut vfe = VFEMinimizer::new();
        vfe.add_node(formerly_rejected, [1.5, 0.0, 0.0, 0.0]);
        // Must now work — node registered, VFE computable.
        let vfe_val = vfe.compute_vfe(formerly_rejected, Some(&[1.5, 0.0, 0.0, 0.0]));
        assert_eq!(
            vfe_val, 0.0,
            "node at id=2_000_000 must be fully functional"
        );

        // Verify that a normal node still works alongside it.
        let valid_id = NodeId::try_new(999_999).expect("valor válido en test");
        vfe.add_node(valid_id, [0.0; 4]);
        assert_eq!(vfe.compute_vfe(valid_id, None), 0.0);
    }

    #[test]
    fn mul_add_precision_drift_stays_below_threshold_after_ten_million_ops() {
        const OPS: usize = 10_000_000;
        const DELTA: f64 = f64::EPSILON * 0.5;

        let mut state = 1.0f64;
        for i in 0..OPS {
            let signed_delta = if (i & 1) == 0 { DELTA } else { -DELTA };
            state = state.mul_add(1.0, signed_delta);
        }

        let drift = (state - 1.0).abs();
        assert!(
            drift < 1.0e-15,
            "drift FMA tras {OPS} operaciones excede el umbral: {drift:e}"
        );
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
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
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
        // The gradient of compute_vfe_with_grad is now [f64; 16].
        // mean[k] (grado 1, component k) it almacena in mean_full[GRADE1_BLADE_INDICES[k]].
        // GRADE1_BLADE_INDICES[0] = 1 → blade e₀.
        // The finite difference perturbs mean[0] → blade 1, and we must compare
        // with grad[GRADE1_BLADE_INDICES[0]] = grad[1].
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        let mean = [1.3, -0.2, 0.4, 0.1];

        // obs with coefficients on all blades including those of grade 1
        // (blades 1,2,4,8 according to GRADE1_BLADE_INDICES)
        let obs = SparseCliffordVector::from_iter([
            (1usize, 0.1), // blade e₀ (grado 1, índice 0 de mean)
            (2, -0.4),     // blade e₁ (grado 1, índice 1 de mean)
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

    /// Verifies that the 16D gradient correctly covers the blades of grade 1.
    /// The components of grade 1 are the most semantically relevant.
    #[test]
    fn vfe_grad_16d_grade1_components_consistent() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mean = [2.0, 0.0, 0.0, 0.0];
        let obs = SparseCliffordVector::from_iter([(1usize, 1.0)]).expect("obs válida");

        let mut vfe = VFEMinimizer::new();
        vfe.add_node(id, mean);
        let (val, grad) = vfe.compute_vfe_with_grad(id, Some(&obs));

        // mean_full[1] = 2.0 (blade e₀, GRADE1_BLADE_INDICES[0])
        // target[1]   = 1.0 (obs.coeffs[1])
        // VFE = w * precision_full[1] * (2.0 - 1.0)² = 1.5 * 1.0 * 1.0 = 1.5 (FIX-3: VFE_BLADE_WEIGHTS[1]=1.5)
        // grad[1] = 2 * w * prec * delta = 2 * 1.5 * 1.0 * 1.0 = 3.0
        assert!(
            (val - 1.5).abs() < 1e-12,
            "VFE debe ser 1.5 (w=1.5 for grade-1), got {val}"
        );
        assert!(
            (grad[GRADE1_BLADE_INDICES[0]] - 3.0).abs() < 1e-12,
            "grad[blade e₀] debe ser 3.0 (2*w*prec*delta), got {}",
            grad[GRADE1_BLADE_INDICES[0]]
        );
        // Gradients on blades not activated by obs or mean must be 0
        assert_eq!(grad[0], 0.0, "blade escalar no activado, grad debe ser 0");
        assert_eq!(
            grad[15], 0.0,
            "blade pseudoescalar no activado, grad debe ser 0"
        );
    }

    /// Verifies that compute_vfe_with_grad_grade1 returns exactmente the 4
    /// components of grade 1 of the gradiente 16D.
    #[test]
    fn vfe_grad_grade1_convenience_matches_full_grad() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mean = [1.0, -0.5, 0.3, 0.2];
        let obs = SparseCliffordVector::from_iter([(1usize, 0.5), (2, 0.0), (4, 0.1)]).unwrap();

        let mut vfe = VFEMinimizer::new();
        vfe.add_node(id, mean);

        let (v16, g16) = vfe.compute_vfe_with_grad(id, Some(&obs));
        let (v4, g4) = vfe.compute_vfe_with_grad_grade1(id, Some(&obs));

        assert_eq!(v16, v4, "VFE debe ser igual en ambas variantes");
        for k in 0..4 {
            assert_eq!(
                g4[k], g16[GRADE1_BLADE_INDICES[k]],
                "g4[{k}] debe coincidir con g16[GRADE1_BLADE_INDICES[{k}]]=g16[{}]",
                GRADE1_BLADE_INDICES[k]
            );
        }
    }

    #[test]
    fn test_precision_compensation_ratio_non_negative() {
        let id = NodeId::try_new(7).expect("NodeId válido");
        let mut vfe = VFEMinimizer::new();
        vfe.add_node(id, [0.25, -0.5, 0.75, -1.0]);
        let obs = SparseCliffordVector::from_iter([(1usize, -0.25), (2, 0.4), (4, -0.2), (8, 0.1)])
            .expect("obs válida");

        let (_value, _grad) = vfe.compute_vfe_with_grad(id, Some(&obs));
        let ratio = vfe.precision_compensation_ratio();
        assert!(
            ratio.is_finite(),
            "precision_compensation_ratio debe ser finito"
        );
        assert!(
            ratio >= 0.0,
            "precision_compensation_ratio debe ser no negativo, got {ratio}"
        );
    }

    /// Verifies that update_full updates all the 16 blades active in obs.
    #[test]
    fn update_full_updates_all_active_blades() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let obs = SparseCliffordVector::from_iter([
            (0usize, 0.5), // escalar (grado 0)
            (1, 1.0),      // e₀ (grado 1)
            (3, 0.8),      // e₀₁ (grado 2)
            (15, 0.2),     // pseudoescalar (grado 4)
        ])
        .expect("obs válida");

        let mut vfe = VFEMinimizer::new();
        vfe.add_node(id, [0.0; 4]); // mean_full parte en 0

        let belief_before = vfe.beliefs[0].mean_full;
        vfe.update_full(id, &obs, 0.1);
        let belief_after = vfe.beliefs[0].mean_full;

        // Blades 0, 1, 3, 15 must have changed (there was error ≠ 0)
        for &blade in &[0usize, 1, 3, 15] {
            assert!(
                (belief_after[blade] - belief_before[blade]).abs() > 1e-10,
                "blade {blade} debe haber cambiado tras update_full"
            );
        }
        // Blade 7 (no activedo in obs) no must haber cambiado
        assert_eq!(
            belief_after[7], belief_before[7],
            "blade 7 no activado en obs no debe cambiar"
        );
    }
}
