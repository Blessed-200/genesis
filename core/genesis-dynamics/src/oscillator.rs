#![allow(clippy::float_cmp, clippy::must_use_candidate)]

use core::cell::{Cell, UnsafeCell};

use genesis_types::NodeId;

/// Per-node quantum oscillator with complex amplitudes per Clifford grade.
///
/// # State structure
///
/// Each node `i` carries 5 Clifford grades (`0 = scalar` … `4 = pseudoscalar`).
/// For each grade `g`, the complex quantum state is:
///
/// ```text
/// ψ_{i,g} = amplitude[g] · e^{iφ_{i,g}}
/// ```
///
/// where:
/// - `phases[g]` — phase angle in radians, accumulated without wrapping
/// - `amplitudes[g]` — modulus in `[0.0, 1.0]`, oscillator inferential certainty
///
/// # Amplitudes: semantics and Fisher coupling
///
/// `amplitudes[g]` initializes to `1.0` for all grades.
/// With this initialization, Kuramoto behavior is **identical to the previous version**:
/// the `r_sync` order parameter reproduces prior values exactly.
///
/// When `VFEMinimizer` updates a node, the external CRATE-003 pipeline
/// couples amplitude to normalized `FisherInfo::trace`:
///
/// ```text
/// amplitude[g] = (FisherInfo::trace / TRACE_INITIAL).clamp(0.0, 1.0)
/// ```
///
/// - Node with **high VFE** (high surprise, not yet learned):
///   `FisherInfo::trace ≈ 1.0` → `amplitude ≈ 1.0` → full `r_sync` contribution
/// - Node with **low VFE** (saturated domain, AXIOMA-008):
///   `FisherInfo::trace → 0` → `amplitude → 0` → suppressed `r_sync` contribution
///
/// This makes `r_sync` measure **angular coherence weighted by inferential certainty**,
/// not only raw angular coherence. A node that has consolidated knowledge reduces
/// its weight in collective dynamics, analogous to myelination (AXIOMA-015).
///
/// # Preparation for CRATE-004
/// `DiscreteRicciFlow` can read `complex_state(g)` per node to compute
/// cluster-level amplitude before applying Ollivier-Ricci curvature.
/// Clusters with high mean amplitude imply high VFE and are wormhole-collapse candidates.
///
/// # Backward compatibility
/// `phases`, `frequencies`, `node_id`, `state`: unchanged fields, signatures, and semantics.
/// `new()` and `with_phases()`: add `amplitudes = [1.0; 5]` with no behavior change.
///
/// AX-ID: AXIOMA-006, AXIOMA-008, `H_dinámica` (LEY_FUNDACIONAL §3.2)
#[derive(Debug, Clone, Copy)]
pub struct QuantumOscillator {
    /// φ_{i,g} — phases per Clifford grade (`g = 0..=4`), in radians.
    pub phases: [f64; 5],
    /// A_{i,g} — amplitude per Clifford grade (`g = 0..=4`), in `[0.0, 1.0]`.
    ///
    /// Initializes to `1.0` for all grades (maximum-certainty prior /
    /// maximum `r_sync` contribution).
    ///
    /// Decreases when node `FisherInfo::trace` decreases (active learning).
    /// Increases again when the node re-enters a high-VFE regime (new exploration).
    ///
    /// El decaimiento es **responsabilidad del sistema externo** que llama
    /// `update_amplitude_from_fisher()` tras cada `VFEMinimizer::update()`.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008
    pub amplitudes: [f64; 5],
    /// ω_{i,g} — frecuencias naturales (rad/s), una por grado.
    pub frequencies: [f64; 5],
    /// Identificador del nodo en el manifold.
    pub node_id: NodeId,
    /// Estado de vida del oscilador (Active, Saturated, Pruned).
    pub state: OscillatorState,
}

/// 8-lane block-SoA storage for oscillator state.
///
/// The internal memory layout is `[grade][lane]` for each numeric field, enabling
/// grade-wise contiguous loads for SIMD-capable hot paths.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct OscillatorBlock {
    /// Phase values `[grade][lane]`.
    pub phases: [[f64; 8]; 5],
    /// Amplitude values `[grade][lane]`.
    pub amplitudes: [[f64; 8]; 5],
    /// Natural frequencies `[grade][lane]`.
    pub frequencies: [[f64; 8]; 5],
    /// Node identifiers by lane.
    pub node_ids: [NodeId; 8],
    /// Lifecycle state by lane.
    pub states: [OscillatorState; 8],
}

impl Default for OscillatorBlock {
    /// Default block values represent empty/pruned lanes, not active oscillators.
    ///
    /// `amplitudes` are intentionally zeroed so unused lanes do not contribute to
    /// synchrony reductions. Active oscillators are created with
    /// [`QuantumOscillator::new`], which initializes amplitudes to `[1.0; 5]`.
    fn default() -> Self {
        Self {
            phases: [[0.0; 8]; 5],
            amplitudes: [[0.0; 8]; 5],
            frequencies: [[0.0; 8]; 5],
            node_ids: [NodeId::INVALID; 8],
            states: [OscillatorState::Pruned { at_ns: 0 }; 8],
        }
    }
}

/// Block-SoA slab wrapper that preserves the existing oscillator-facing API.
///
/// `lanes` remains as an AoS compatibility projection for public methods and
/// tests, while `blocks` is the hot-path storage used by Kuramoto and synchrony.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
#[derive(Debug)]
pub struct OscillatorSlab {
    lanes: UnsafeCell<Vec<QuantumOscillator>>,
    blocks: UnsafeCell<Vec<OscillatorBlock>>,
    sync_state: Cell<SlabSyncState>,
    #[cfg(test)]
    rebuild_count: Cell<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlabSyncState {
    Clean,
    LanesDirty,
    BlocksDirty,
}

impl Default for OscillatorSlab {
    fn default() -> Self {
        Self::new()
    }
}

impl OscillatorSlab {
    /// Create an empty slab.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub const fn new() -> Self {
        Self {
            lanes: UnsafeCell::new(Vec::new()),
            blocks: UnsafeCell::new(Vec::new()),
            sync_state: Cell::new(SlabSyncState::Clean),
            #[cfg(test)]
            rebuild_count: Cell::new(0),
        }
    }

    /// Number of oscillators stored in the slab.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn len(&self) -> usize {
        self.lanes_ref().len()
    }

    /// Returns true when the slab is empty.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lanes_ref().is_empty()
    }

    /// Returns an immutable AoS compatibility view.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn as_slice(&self) -> &[QuantumOscillator] {
        self.sync_lanes_if_dirty();
        self.lanes_ref()
    }

    /// Returns a mutable AoS compatibility view and marks all blocks stale.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [QuantumOscillator] {
        self.sync_lanes_if_dirty();
        self.sync_state.set(SlabSyncState::LanesDirty);
        self.lanes_mut()
    }

    /// Immutable oscillator lookup by index.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&QuantumOscillator> {
        self.sync_lanes_if_dirty();
        self.lanes_ref().get(idx)
    }

    /// Mutable oscillator lookup by index.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut QuantumOscillator> {
        self.sync_lanes_if_dirty();
        self.sync_state.set(SlabSyncState::LanesDirty);
        self.lanes_mut().get_mut(idx)
    }

    /// Push one oscillator and update its destination block lane.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn push(&mut self, osc: QuantumOscillator) {
        self.sync_lanes_if_dirty();
        self.sync_blocks_if_dirty();
        let idx = self.lanes_mut().len();
        self.lanes_mut().push(osc);
        let b = idx / 8;
        let lane = idx % 8;
        if b == self.blocks_ref().len() {
            self.blocks_mut_ref().push(OscillatorBlock::default());
        }
        self.blocks_mut_ref()[b].node_ids[lane] = osc.node_id;
        self.blocks_mut_ref()[b].states[lane] = osc.state;
        for g in 0..5 {
            self.blocks_mut_ref()[b].phases[g][lane] = osc.phases[g];
            self.blocks_mut_ref()[b].amplitudes[g][lane] = osc.amplitudes[g];
            self.blocks_mut_ref()[b].frequencies[g][lane] = osc.frequencies[g];
        }
        self.sync_state.set(SlabSyncState::Clean);
    }

    /// Mutable iterator over compatibility lanes.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, QuantumOscillator> {
        self.sync_lanes_if_dirty();
        self.sync_state.set(SlabSyncState::LanesDirty);
        self.lanes_mut().iter_mut()
    }

    /// Immutable iterator over compatibility lanes.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, QuantumOscillator> {
        self.sync_lanes_if_dirty();
        self.lanes_ref().iter()
    }

    #[inline]
    fn sync_blocks_if_dirty(&self) {
        if self.sync_state.get() == SlabSyncState::LanesDirty {
            // SAFETY: interior mutability is provided by `UnsafeCell`; we only refresh
            // blocks from lanes and do not hand out aliased mutable references here.
            unsafe {
                self.rebuild_blocks_from_cells();
            }
            self.sync_state.set(SlabSyncState::Clean);
        }
    }

    #[inline]
    fn sync_lanes_if_dirty(&self) {
        if self.sync_state.get() == SlabSyncState::BlocksDirty {
            // SAFETY: interior mutability is provided by `UnsafeCell`; we only refresh
            // lanes from blocks and do not hand out aliased mutable references here.
            unsafe {
                self.sync_lanes_from_blocks_cells();
            }
            self.sync_state.set(SlabSyncState::Clean);
        }
    }

    /// Immutable view over SoA blocks.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn blocks(&self) -> &[OscillatorBlock] {
        self.sync_blocks_if_dirty();
        self.blocks_ref()
    }

    /// Mutable view over SoA blocks.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn blocks_mut(&mut self) -> &mut [OscillatorBlock] {
        self.sync_blocks_if_dirty();
        self.sync_state.set(SlabSyncState::BlocksDirty);
        self.blocks_mut_ref()
    }

    /// Mutable view over SoA blocks when caller keeps AoS lanes mirrored manually.
    ///
    /// Invariant: every write through this view must be mirrored into the AoS lane
    /// storage before leaving the caller, otherwise lanes and blocks diverge.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub(crate) fn blocks_mut_in_sync(&mut self) -> &mut [OscillatorBlock] {
        self.sync_blocks_if_dirty();
        self.blocks_mut_ref()
    }

    /// Mirrors one phase value into AoS lane storage and SoA block storage.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub(crate) fn set_phase_mirrored(&mut self, idx: usize, g: usize, phase: f64) {
        self.sync_lanes_if_dirty();
        self.sync_blocks_if_dirty();
        // SAFETY: `&mut self` guarantees exclusive access; indices are provided by
        // Kuramoto hot paths and validated by loop bounds there.
        unsafe {
            (&mut *self.lanes.get())[idx].phases[g] = phase;
            let b = idx / 8;
            let lane = idx % 8;
            (&mut *self.blocks.get())[b].phases[g][lane] = phase;
        }
        self.sync_state.set(SlabSyncState::Clean);
    }

    /// Rebuild SoA blocks from AoS lanes.
    ///
    /// HOT PATH SUPPORT: called at insertion and after batched phase updates.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn rebuild_blocks(&mut self) {
        self.sync_lanes_if_dirty();
        #[cfg(test)]
        self.rebuild_count.set(self.rebuild_count.get() + 1);
        // SAFETY: `&mut self` guarantees exclusive access; references derived from
        // `UnsafeCell` do not alias mutable/immutable borrows outside this scope.
        unsafe {
            let lanes = &*self.lanes.get();
            let blocks = &mut *self.blocks.get();
            let n = lanes.len();
            let block_count = n.div_ceil(8);
            let old_len = blocks.len();
            if block_count <= old_len {
                for block in &mut blocks[..block_count] {
                    *block = OscillatorBlock::default();
                }
                blocks.truncate(block_count);
            } else {
                for block in &mut blocks[..old_len] {
                    *block = OscillatorBlock::default();
                }
                blocks.resize_with(block_count, OscillatorBlock::default);
            }
            for (idx, osc) in lanes.iter().enumerate() {
                let b = idx / 8;
                let lane = idx % 8;
                blocks[b].node_ids[lane] = osc.node_id;
                blocks[b].states[lane] = osc.state;
                for g in 0..5 {
                    blocks[b].phases[g][lane] = osc.phases[g];
                    blocks[b].amplitudes[g][lane] = osc.amplitudes[g];
                    blocks[b].frequencies[g][lane] = osc.frequencies[g];
                }
            }
        }
        self.sync_state.set(SlabSyncState::Clean);
    }

    /// Write block values back into AoS lane storage.
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn sync_lanes_from_blocks(&mut self) {
        self.sync_blocks_if_dirty();
        // SAFETY: `&mut self` guarantees exclusive access; references derived from
        // `UnsafeCell` do not alias mutable/immutable borrows outside this scope.
        unsafe {
            let lanes = &mut *self.lanes.get();
            let blocks = &*self.blocks.get();
            for (idx, osc) in lanes.iter_mut().enumerate() {
                let b = idx / 8;
                let lane = idx % 8;
                let block = &blocks[b];
                osc.node_id = block.node_ids[lane];
                osc.state = block.states[lane];
                for g in 0..5 {
                    osc.phases[g] = block.phases[g][lane];
                    osc.amplitudes[g] = block.amplitudes[g][lane];
                    osc.frequencies[g] = block.frequencies[g][lane];
                }
            }
        }
        self.sync_state.set(SlabSyncState::Clean);
    }

    #[inline]
    fn lanes_ref(&self) -> &Vec<QuantumOscillator> {
        // SAFETY: shared access through `UnsafeCell` is read-only here.
        unsafe { &*self.lanes.get() }
    }

    #[inline]
    fn lanes_mut(&mut self) -> &mut Vec<QuantumOscillator> {
        // SAFETY: `&mut self` guarantees unique access to the slab instance.
        unsafe { &mut *self.lanes.get() }
    }

    #[inline]
    fn blocks_ref(&self) -> &Vec<OscillatorBlock> {
        // SAFETY: shared access through `UnsafeCell` is read-only here.
        unsafe { &*self.blocks.get() }
    }

    #[inline]
    fn blocks_mut_ref(&mut self) -> &mut Vec<OscillatorBlock> {
        // SAFETY: `&mut self` guarantees unique access to the slab instance.
        unsafe { &mut *self.blocks.get() }
    }

    unsafe fn rebuild_blocks_from_cells(&self) {
        #[cfg(test)]
        self.rebuild_count.set(self.rebuild_count.get() + 1);
        let lanes = &*self.lanes.get();
        let blocks = &mut *self.blocks.get();
        let block_count = lanes.len().div_ceil(8);
        let old_len = blocks.len();
        if block_count <= old_len {
            for block in &mut blocks[..block_count] {
                *block = OscillatorBlock::default();
            }
            blocks.truncate(block_count);
        } else {
            for block in &mut blocks[..old_len] {
                *block = OscillatorBlock::default();
            }
            blocks.resize_with(block_count, OscillatorBlock::default);
        }
        for (idx, osc) in lanes.iter().enumerate() {
            let b = idx / 8;
            let lane = idx % 8;
            blocks[b].node_ids[lane] = osc.node_id;
            blocks[b].states[lane] = osc.state;
            for g in 0..5 {
                blocks[b].phases[g][lane] = osc.phases[g];
                blocks[b].amplitudes[g][lane] = osc.amplitudes[g];
                blocks[b].frequencies[g][lane] = osc.frequencies[g];
            }
        }
    }

    unsafe fn sync_lanes_from_blocks_cells(&self) {
        let lanes = &mut *self.lanes.get();
        let blocks = &*self.blocks.get();
        for (idx, osc) in lanes.iter_mut().enumerate() {
            let b = idx / 8;
            let lane = idx % 8;
            let block = &blocks[b];
            osc.node_id = block.node_ids[lane];
            osc.state = block.states[lane];
            for g in 0..5 {
                osc.phases[g] = block.phases[g][lane];
                osc.amplitudes[g] = block.amplitudes[g][lane];
                osc.frequencies[g] = block.frequencies[g][lane];
            }
        }
    }

    #[cfg(test)]
    fn rebuild_count_for_test(&self) -> usize {
        self.rebuild_count.get()
    }
}

impl core::ops::Index<usize> for OscillatorSlab {
    type Output = QuantumOscillator;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        self.sync_lanes_if_dirty();
        &self.lanes_ref()[index]
    }
}

impl core::ops::IndexMut<usize> for OscillatorSlab {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.sync_lanes_if_dirty();
        self.sync_state.set(SlabSyncState::LanesDirty);
        &mut self.lanes_mut()[index]
    }
}

impl core::ops::Deref for OscillatorSlab {
    type Target = [QuantumOscillator];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.sync_lanes_if_dirty();
        self.lanes_ref()
    }
}

impl core::ops::DerefMut for OscillatorSlab {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.sync_lanes_if_dirty();
        self.sync_state.set(SlabSyncState::LanesDirty);
        self.lanes_mut()
    }
}

impl<'a> IntoIterator for &'a OscillatorSlab {
    type Item = &'a QuantumOscillator;
    type IntoIter = core::slice::Iter<'a, QuantumOscillator>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.sync_lanes_if_dirty();
        self.lanes_ref().iter()
    }
}

impl<'a> IntoIterator for &'a mut OscillatorSlab {
    type Item = &'a mut QuantumOscillator;
    type IntoIter = core::slice::IterMut<'a, QuantumOscillator>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.sync_lanes_if_dirty();
        self.sync_state.set(SlabSyncState::LanesDirty);
        self.lanes_mut().iter_mut()
    }
}

impl QuantumOscillator {
    #[inline]
    const fn default_state() -> OscillatorState {
        OscillatorState::Active
    }

    /// Initial Fisher trace (uninformative prior).
    /// Used to normalize amplitudes: `amplitude = trace / FISHER_TRACE_INITIAL`.
    pub const FISHER_TRACE_INITIAL: f64 = 1.0;

    /// Canonical constructor: zero phases, maximum initial amplitudes (1.0), provided frequencies.
    ///
    /// Amplitudes `[1.0; 5]` encode maximum-certainty prior. Kuramoto behaves
    /// exactly as before this extension.
    #[inline]
    pub const fn new(node_id: NodeId, frequencies: [f64; 5]) -> Self {
        Self {
            phases: [0.0; 5],
            amplitudes: [1.0; 5],
            frequencies,
            node_id,
            state: Self::default_state(),
        }
    }

    /// Constructor con fases iniciales explícitas. Amplitudes inicializan en `1.0`.
    #[inline]
    pub const fn with_phases(node_id: NodeId, phases: [f64; 5], frequencies: [f64; 5]) -> Self {
        Self {
            phases,
            amplitudes: [1.0; 5],
            frequencies,
            node_id,
            state: Self::default_state(),
        }
    }

    /// Complex quantum oscillator state at Clifford grade `g`.
    ///
    /// ```text
    /// ψ_{i,g} = amplitude[g] · (cos(φ_{i,g}), sin(φ_{i,g}))
    /// ```
    ///
    /// Returned as `(re, im)` instead of `Complex64` to avoid exposing
    /// `num-complex` in the public type; CRATE-004 callers can construct
    /// `Complex64::new(re, im)` directly.
    ///
    /// # Parameter
    /// `g` ∈ `[0, 4]`. Panics in debug if `g > 4`; release requires valid caller input.
    ///
    /// # Uso en CRATE-004
    /// ```
    /// use genesis_dynamics::QuantumOscillator;
    /// use genesis_types::NodeId;
    ///
    /// let node = NodeId::try_new(0).unwrap();
    /// let osc = QuantumOscillator::new(node, [0.0; 5]);
    /// let (re, im) = osc.complex_state(0); // grade 0
    /// let amplitude: f64 = re.hypot(im);
    /// assert!(amplitude >= 0.0);
    /// ```
    ///
    /// AX-ID: AXIOMA-006, LEY_FUNDACIONAL §3.2
    #[inline]
    pub fn complex_state(&self, g: usize) -> (f64, f64) {
        debug_assert!(g < 5, "grade {g} out of range [0,4]");
        let a = self.amplitudes[g];
        let (sin, cos) = self.phases[g].sin_cos();
        (a * cos, a * sin)
    }

    /// Total oscillator amplitude: Euclidean norm of the amplitude vector.
    ///
    /// ```text
    /// |A_i| = √(Σ_g amplitudes[g]²) / √5  ∈ [0.0, 1.0]
    /// ```
    ///
    /// Normalized by √5 so the maximum (`all grades = 1.0`) equals `1.0`.
    /// Used by weighted `r_sync` to detect global inferential certainty.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008
    #[inline]
    pub fn amplitude_norm(&self) -> f64 {
        let a = &self.amplitudes;
        let sq = a[0].mul_add(
            a[0],
            a[1].mul_add(a[1], a[2].mul_add(a[2], a[3].mul_add(a[3], a[4] * a[4]))),
        );
        (sq * 0.2_f64).sqrt()
    }

    /// Updates oscillator amplitudes from the current Fisher trace.
    ///
    /// ```text
    /// amplitude[g] = (fisher_trace / FISHER_TRACE_INITIAL).clamp(0.0, 1.0)
    /// ```
    ///
    /// All grades receive the same amplitude because `FisherInfo` currently exposes
    /// an isotropic scalar trace. When CRATE-004 introduces per-grade Fisher metrics,
    /// this method can be extended to grade-specific amplitudes.
    ///
    /// # Correct call site
    /// This method should be called **immediately after** `VFEMinimizer::update()`
    /// for a node, passing `vfe.fisher_info(id).trace` as input.
    ///
    /// # Backward compatibility
    /// With `fisher_trace = FISHER_TRACE_INITIAL = 1.0` (prior), all amplitudes
    /// remain `1.0` and Kuramoto behavior is unchanged.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008, LEY_FUNDACIONAL §3.2
    #[inline]
    pub fn update_amplitude_from_fisher(&mut self, fisher_trace: f64) {
        let a = fisher_trace.clamp(0.0, 1.0);
        self.amplitudes = [a; 5];
    }

    /// Semantic saturation factor ∈ [0.0, 1.0].
    ///
    /// 0.0 = maximum uncertainty (amplitude ≈ 1.0, full coupling drive).
    /// 1.0 = fully saturated (amplitude ≈ 0.0, domain mastered).
    ///
    /// Used in adaptive Kuramoto coupling (semantic habituation):
    /// ```text
    /// adaptive_Γ = base_Γ · (1.0 − sat_i · sat_j)
    /// ```
    /// When both nodes have converged (`sat → 1`), their mutual coupling drops
    /// to zero — compute is redirected to uncertain, high-surprise pairs.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn saturation_factor(&self) -> f64 {
        1.0 - self.amplitude_norm()
    }

    /// Primary oscillator phase (grade 0 scalar).
    /// Used to compute `r_sync` in `synchrony.rs`.
    #[inline]
    pub const fn primary_phase(&self) -> f64 {
        self.phases[0]
    }

    /// Marks the oscillator as saturated. Idempotent if already saturated or pruned.
    pub const fn mark_saturated(&mut self, now_ns: u64) {
        if matches!(self.state, OscillatorState::Active) {
            self.state = OscillatorState::Saturated { since_ns: now_ns };
        }
    }

    /// Marks the oscillator as pruned. Valid from Active or Saturated states.
    pub const fn mark_pruned(&mut self, now_ns: u64) {
        if !matches!(self.state, OscillatorState::Pruned { .. }) {
            self.state = OscillatorState::Pruned { at_ns: now_ns };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_initializes_phases_to_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        let osc = QuantumOscillator::new(id, [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(osc.phases, [0.0; 5]);
        assert_eq!(osc.node_id, id);
    }

    #[test]
    fn new_initializes_amplitudes_to_one() {
        let id = NodeId::try_new(1).expect("NodeId válido por construcción");
        let osc = QuantumOscillator::new(id, [1.0; 5]);
        assert_eq!(
            osc.amplitudes, [1.0; 5],
            "amplitudes deben inicializar en 1.0 (prior de máxima certeza)"
        );
    }

    #[test]
    fn with_phases_preserves_all_fields() {
        let id = NodeId::try_new(7).expect("NodeId válido por construcción");
        let phases = [0.1, 0.2, 0.3, 0.4, 0.5];
        let freqs = [1.0, 1.1, 1.2, 1.3, 1.4];
        let osc = QuantumOscillator::with_phases(id, phases, freqs);
        assert_eq!(osc.phases, phases);
        assert_eq!(osc.frequencies, freqs);
        assert_eq!(osc.node_id.get(), 7);
        assert_eq!(
            osc.amplitudes, [1.0; 5],
            "with_phases también inicializa amplitudes en 1.0"
        );
    }

    #[test]
    fn primary_phase_returns_grade0() {
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        let phases = [2.5, 0.1, 0.2, 0.3, 0.4];
        let osc = QuantumOscillator::with_phases(id, phases, [0.0; 5]);
        assert_eq!(osc.primary_phase(), 2.5);
    }

    // ── complex_state tests ───────────────────────────────────────────────────

    /// Con amplitud 1.0 y fase 0.0, complex_state debe ser (1.0, 0.0).
    #[test]
    fn complex_state_unit_amplitude_zero_phase() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let osc = QuantumOscillator::new(id, [0.0; 5]);
        let (re, im) = osc.complex_state(0);
        assert!(
            (re - 1.0).abs() < 1e-15,
            "re debe ser 1.0 con φ=0, got {re}"
        );
        assert!(im.abs() < 1e-15, "im debe ser 0.0 con φ=0, got {im}");
    }

    /// With amplitude `0.0`, `complex_state` must be `(0.0, 0.0)` for any phase.
    #[test]
    fn complex_state_zero_amplitude_gives_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.amplitudes = [0.0; 5];
        for g in 0..5 {
            let (re, im) = osc.complex_state(g);
            assert_eq!(re, 0.0, "re debe ser 0 con amplitude=0 en grado {g}");
            assert_eq!(im, 0.0, "im debe ser 0 con amplitude=0 en grado {g}");
        }
    }

    /// Verifica que |complex_state(g)| = amplitude[g].
    #[test]
    fn complex_state_norm_equals_amplitude() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        let phases = [1.0, 2.0, 3.0, 4.0, 5.0];
        let amps = [1.0, 0.8, 0.6, 0.4, 0.2];
        osc.amplitudes = amps;
        for g in 0..5 {
            osc.phases[g] = phases[g];
            let (re, im) = osc.complex_state(g);
            let norm = re.hypot(im);
            assert!(
                (norm - amps[g]).abs() < 1e-14,
                "grado {g}: |ψ| = {norm}, amplitude = {}",
                amps[g]
            );
        }
    }

    // ── amplitude_norm tests ──────────────────────────────────────────────────

    /// Con amplitudes = [1.0; 5], amplitude_norm debe ser 1.0.
    #[test]
    fn amplitude_norm_all_ones_is_one() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let osc = QuantumOscillator::new(id, [0.0; 5]);
        assert!(
            (osc.amplitude_norm() - 1.0).abs() < 1e-14,
            "amplitude_norm con [1.0;5] debe ser 1.0"
        );
    }

    /// Con amplitudes = [0.0; 5], amplitude_norm debe ser 0.0.
    #[test]
    fn amplitude_norm_all_zeros_is_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.amplitudes = [0.0; 5];
        assert_eq!(osc.amplitude_norm(), 0.0);
    }

    /// `amplitude_norm` stays in `[0,1]` for amplitudes in `[0,1]`.
    #[test]
    fn amplitude_norm_bounded_in_unit_interval() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        // Validate several intermediate values.
        for v in [0.0, 0.2, 0.5, 0.7, 1.0] {
            osc.amplitudes = [v; 5];
            let n = osc.amplitude_norm();
            assert!(
                (0.0..=1.0 + 1e-14).contains(&n),
                "amplitude_norm = {n} fuera de [0,1] para amplitude = {v}"
            );
        }
    }

    // ── update_amplitude_from_fisher tests ────────────────────────────────────

    /// Con fisher_trace = FISHER_TRACE_INITIAL, amplitudes no cambian de 1.0.
    #[test]
    fn update_amplitude_fisher_prior_stays_one() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.update_amplitude_from_fisher(QuantumOscillator::FISHER_TRACE_INITIAL);
        assert_eq!(
            osc.amplitudes, [1.0; 5],
            "fisher_trace = INITIAL → amplitudes deben permanecer en 1.0"
        );
    }

    /// Con fisher_trace = 0.0 (dominio completamente saturado), amplitudes → 0.0.
    #[test]
    fn update_amplitude_fisher_saturated_gives_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.update_amplitude_from_fisher(0.0);
        assert_eq!(
            osc.amplitudes, [0.0; 5],
            "fisher_trace = 0 → amplitudes deben ser 0.0"
        );
    }

    /// Amplitude is clamped to `[0,1]` even for out-of-range inputs.
    #[test]
    fn update_amplitude_clamps_to_unit_interval() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);

        osc.update_amplitude_from_fisher(2.0); // > INITIAL
        for &a in &osc.amplitudes {
            assert!(a <= 1.0, "amplitude {a} debe ser ≤ 1.0");
        }

        osc.update_amplitude_from_fisher(-1.0); // negativo
        for &a in &osc.amplitudes {
            assert!(a >= 0.0, "amplitude {a} debe ser ≥ 0.0");
        }
    }

    /// Verifies monotonic behavior: more learning implies lower amplitude.
    #[test]
    fn update_amplitude_monotone_with_fisher_trace() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);

        // trace alto (aprendizaje inicial) → amplitud alta
        osc.update_amplitude_from_fisher(1.0);
        let a_high = osc.amplitudes[0];

        // trace medio → amplitud media
        osc.update_amplitude_from_fisher(0.5);
        let a_mid = osc.amplitudes[0];

        // Low trace (saturated domain) -> low amplitude.
        osc.update_amplitude_from_fisher(0.1);
        let a_low = osc.amplitudes[0];

        assert!(
            a_high >= a_mid && a_mid >= a_low,
            "amplitudes deben ser monótonas con fisher_trace: {a_high} ≥ {a_mid} ≥ {a_low}"
        );
    }
}

/// Lifecycle state of a quantum oscillator.
///
/// Transition is one-way: `Active -> Saturated -> Pruned`.
/// A pruned oscillator does not participate in Kuramoto or VFE updates.
///
/// AX-ID: AXIOMA-008 (Saturated), AXIOMA-016 (Pruned)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OscillatorState {
    /// Active oscillator: participates in Kuramoto and VFE.
    #[default]
    Active,
    /// Fisher-saturated: `ΔG < ε` for `FISHER_SATIATION_WINDOW` iterations.
    /// The oscillator stops accepting new learning inputs but still contributes
    /// to the `r_sync` order parameter.
    /// `since_ns`: nanosecond timestamp when saturation occurred.
    Saturated {
        /// Timestamp in nanoseconds when the oscillator entered the Saturated state.
        since_ns: u64,
    },
    /// Pruned by heat-equation dynamics (AXIOMA-016).
    /// The oscillator is inactive: phases are not updated and it contributes no Ω signal.
    /// `at_ns`: timestamp of pruning.
    Pruned {
        /// Timestamp in nanoseconds when the oscillator was pruned.
        at_ns: u64,
    },
}

impl OscillatorState {
    /// Returns true if the oscillator can receive learning inputs.
    #[inline]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Returns true if the oscillator contributes to `r_sync`.
    /// Saturated oscillators do contribute; pruned oscillators do not.
    #[inline]
    pub const fn contributes_to_sync(self) -> bool {
        !matches!(self, Self::Pruned { .. })
    }
}

#[cfg(test)]
mod slab_tests {
    use super::*;

    fn make_osc(id: u64, phase_base: f64) -> QuantumOscillator {
        let node = NodeId::try_new(id).expect("valid NodeId");
        QuantumOscillator::with_phases(
            node,
            [
                phase_base,
                phase_base + 0.1,
                phase_base + 0.2,
                phase_base + 0.3,
                phase_base + 0.4,
            ],
            [1.0, 1.1, 1.2, 1.3, 1.4],
        )
    }

    #[test]
    fn slab_push_updates_block_lane_incrementally() {
        let mut slab = OscillatorSlab::new();
        slab.push(make_osc(1, 0.5));
        assert_eq!(slab.blocks().len(), 1);
        assert!((slab.blocks()[0].phases[0][0] - 0.5).abs() < 1e-12);
        assert_eq!(slab.blocks()[0].node_ids[0].get(), 1);
    }

    #[test]
    fn slab_rebuild_blocks_matches_lane_storage() {
        let mut slab = OscillatorSlab::new();
        for i in 0..10u64 {
            slab.push(make_osc(i, i as f64 * 0.01));
        }
        slab[3].phases[2] = 42.0;
        slab.rebuild_blocks();
        assert!((slab.blocks()[0].phases[2][3] - 42.0).abs() < 1e-12);
    }

    #[test]
    fn slab_sync_lanes_from_blocks_propagates_block_changes() {
        let mut slab = OscillatorSlab::new();
        slab.push(make_osc(7, 0.2));
        slab.blocks_mut()[0].phases[4][0] = 9.0;
        slab.sync_lanes_from_blocks();
        assert!((slab[0].phases[4] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn slab_tail_lane_behavior_partial_block() {
        let mut slab = OscillatorSlab::new();
        for i in 0..9u64 {
            slab.push(make_osc(i, i as f64));
        }
        assert_eq!(slab.blocks().len(), 2);
        assert_eq!(slab.blocks()[1].node_ids[0].get(), 8);
        assert_eq!(slab.blocks()[1].node_ids[1], NodeId::INVALID);
        assert!(matches!(
            slab.blocks()[1].states[1],
            OscillatorState::Pruned { .. }
        ));
    }

    #[test]
    fn slab_push_does_not_trigger_full_rebuild() {
        let mut slab = OscillatorSlab::new();
        assert_eq!(slab.rebuild_count_for_test(), 0);
        slab.push(make_osc(0, 0.0));
        assert_eq!(
            slab.rebuild_count_for_test(),
            0,
            "incremental push must patch target lane without full rebuild"
        );

        slab[0].phases[0] = 3.0;
        let before = slab.rebuild_count_for_test();
        let _ = slab.blocks();
        assert!(
            slab.rebuild_count_for_test() > before,
            "dirty lane sync through blocks() must trigger a rebuild"
        );
    }
}
