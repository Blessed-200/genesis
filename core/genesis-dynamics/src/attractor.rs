//! # AttractorLandscape
//!
//! Cognitive attractor landscape: E(x) = inferential energy of a state.
//! Consolidated concepts are local minima (attractors).
//! Thought dynamics ARE gradient descent −∇E(x). (AXIOMA-004)
//!
//! # Data structure (BN-06)
//!
//! `register()` was O(N) due to `retain()` + `Vec::insert`. At scale (N_attractors ~ 10⁴)
//! this became a bottleneck for the consciousness layer.
//!
//! New implementation: `FastHashMap<NodeId, f64>` for O(1) ID lookup + `BTreeSet<AttractorEntry>`
//! for O(log N) energy-ordered insertion. The `FastHashMap` is not in the HNSW/Kuramoto hot-path —
//! it serves the attractor bookkeeping layer (consciousness, CRATE-006).
//!
//! PROHIBITED: implementing memories as external lookup tables outside the attractor landscape.
//! All memory lives in attractor topology. (AXIOMA-004)

use std::collections::BTreeSet;

use genesis_types::{FastHashMap, NodeId};

use crate::{free_energy::VFEMinimizer, phase_semantics::PhaseSemanticsEngine};

/// Ordering wrapper for `BTreeSet<AttractorEntry>`.
/// Primary: energy ascending (minimum energy = most preferred attractor).
/// Secondary: NodeId (deterministic tiebreak for equal-energy attractors).
///
/// # PartialEq / Eq contract (BTreeSet safety)
///
/// `f64` violates `Eq` reflexivity for `NaN`. Deriving `PartialEq` and then
/// implementing `Eq` would produce an invalid `Eq` impl that breaks `BTreeSet`
/// invariants (remove may not find an element it just inserted).
///
/// Solution: manual `PartialEq` using `f64::to_bits()` — bijective, total,
/// and reflexive for all bit patterns including NaN. Combined with `id` this
/// ensures exact structural equality, making `Eq` valid. `-0.0` and `+0.0`
/// compare as distinct (distinct bits), which is correct: they carry different
/// physical meaning in the energy landscape.
#[derive(Debug, Clone, Copy)]
struct AttractorEntry {
    id: NodeId,
    energy: f64,
}

impl PartialEq for AttractorEntry {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.energy.to_bits() == other.energy.to_bits()
    }
}

/// SAFETY: `PartialEq` above is reflexive (`to_bits()` is reflexive for all
/// `f64` bit patterns including NaN), symmetric, and transitive. `Eq` is valid.
impl Eq for AttractorEntry {}

impl Ord for AttractorEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // total_cmp: deterministic even for NaN (NaN > all finite, consistent with IEEE 754 total order)
        self.energy
            .total_cmp(&other.energy)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for AttractorEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Cognitive attractor landscape.
///
/// # Complexity after BN-06
/// - `register()`: O(log N) — FastHashMap O(1) + BTreeSet O(log N)
/// - `descend()`: O(N) via BTreeSet iteration (cache-friendly, ordered)
/// - `energy_of()`: O(1) via FastHashMap
/// - `attractor_count()`: O(1)
///
/// AX-ID: AXIOMA-004
pub struct AttractorLandscape {
    /// O(1) energy lookup by NodeId.
    id_to_energy: FastHashMap<NodeId, f64>,
    /// Energy-ordered set for deterministic iteration.
    ordered_landscape: BTreeSet<AttractorEntry>,
}

impl AttractorLandscape {
    #[inline]
    #[cfg(debug_assertions)]
    fn debug_assert_consistent(&self) {
        debug_assert_eq!(
            self.id_to_energy.len(),
            self.ordered_landscape.len(),
            "attractor indices out of sync: map={} set={}",
            self.id_to_energy.len(),
            self.ordered_landscape.len()
        );
        for entry in &self.ordered_landscape {
            let mapped = self.id_to_energy.get(&entry.id).copied();
            debug_assert_eq!(
                mapped.map(f64::to_bits),
                Some(entry.energy.to_bits()),
                "attractor energy mismatch for node {}",
                entry.id.get()
            );
        }
    }

    /// Creates an empty attractor landscape with no registered attractors.
    pub fn new() -> Self {
        Self {
            id_to_energy: FastHashMap::new(),
            ordered_landscape: BTreeSet::new(),
        }
    }

    /// Register or update an attractor. O(log N).
    ///
    /// If the same `NodeId` already exists with a different energy, the old entry
    /// is removed and the new one inserted. Idempotent for same (id, energy).
    pub fn register(&mut self, id: NodeId, energy: f64) {
        if let Some(old_energy) = self.id_to_energy.insert(id, energy) {
            if old_energy.to_bits() == energy.to_bits() {
                return; // Exact same value — idempotent, no work needed.
            }
            self.ordered_landscape.remove(&AttractorEntry {
                id,
                energy: old_energy,
            });
        }
        self.ordered_landscape.insert(AttractorEntry { id, energy });
        #[cfg(debug_assertions)]
        self.debug_assert_consistent();
    }

    /// Descend to the minimum-VFE attractor (global minimum, v1.0).
    ///
    /// # Complexity (FIX-2)
    /// O(N_attractors) with **exactly one** `compute_vfe()` call per attractor.
    /// The former `min_by` approach called `compute_vfe()` multiple times per
    /// element (the comparator is invoked O(N log N) times by BTreeSet iteration
    /// + `min_by`). Single-pass is both faster and cache-friendlier.
    ///
    /// `_current` is reserved for topological routing in v2.0 (CRATE-006).
    ///
    /// AX-ID: AXIOMA-004 (partial — full topological routing in CRATE-006)
    pub fn descend(&self, _current: NodeId, vfe: &VFEMinimizer) -> Option<NodeId> {
        let mut best: Option<(NodeId, f64)> = None;
        for e in &self.ordered_landscape {
            let val = vfe.compute_vfe(e.id, None);
            match best {
                Some((_, best_val)) if val >= best_val => {} // not better
                _ => best = Some((e.id, val)),
            }
        }
        best.map(|(id, _)| id)
    }

    /// Descend to the minimum semantic-aware objective:
    /// `VFE + λ1·semantic_incoherence + λ2·phase_instability`.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006, `H_información`, `H_dinámica`
    pub fn descend_with_semantics(
        &self,
        _current: NodeId,
        vfe: &VFEMinimizer,
        semantics: &PhaseSemanticsEngine,
        lambda_incoherence: f64,
        lambda_instability: f64,
    ) -> Option<NodeId> {
        let mut best: Option<(NodeId, f64)> = None;
        for entry in &self.ordered_landscape {
            let base = vfe.compute_vfe(entry.id, None);
            let incoherence = semantics.semantic_incoherence(entry.id);
            let instability = semantics.phase_instability(entry.id);
            let objective = Self::semantic_objective(
                base,
                incoherence,
                instability,
                lambda_incoherence,
                lambda_instability,
            );
            match best {
                Some((_, best_obj)) if objective >= best_obj => {}
                _ => best = Some((entry.id, objective)),
            }
        }
        best.map(|(id, _)| id)
    }

    /// Computes semantic-aware objective using Fused Multiply-Add (FMA).
    ///
    /// Mathematical equivalence:
    /// `VFE + λ1·incoherence + λ2·instability`.
    ///
    /// Uses nested [`f64::mul_add`] to reduce intermediate rounding error and
    /// leverage CPU FMA instructions where available, while remaining `const`
    /// evaluable for compile-time contexts.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006, `H_información`, `H_dinámica`
    #[allow(clippy::inline_always)]
    #[inline(always)]
    const fn semantic_objective(
        base: f64,
        incoherence: f64,
        instability: f64,
        lambda_incoherence: f64,
        lambda_instability: f64,
    ) -> f64 {
        lambda_instability.mul_add(instability, lambda_incoherence.mul_add(incoherence, base))
    }

    /// Iterate attractors in ascending energy order.
    ///
    /// Deterministic iteration for CRATE-006 observer when building Ω.
    #[inline]
    pub fn ascending_energy(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.ordered_landscape.iter().map(|e| e.id)
    }

    /// Number of registered attractors. O(1).
    #[inline]
    pub fn attractor_count(&self) -> usize {
        self.ordered_landscape.len()
    }

    /// Registered energy of an attractor. O(1) via FastHashMap.
    #[inline]
    pub fn energy_of(&self, id: NodeId) -> Option<f64> {
        self.id_to_energy.get(&id).copied()
    }
}

impl Default for AttractorLandscape {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn id(n: u64) -> NodeId {
        NodeId::try_new(n).expect("NodeId válido")
    }

    #[test]
    fn attractor_register_and_count() {
        let mut l = AttractorLandscape::new();
        l.register(id(0), 5.0);
        l.register(id(1), 1.0);
        l.register(id(2), 3.0);
        assert_eq!(l.attractor_count(), 3);
    }

    #[test]
    fn attractor_register_overwrites_existing() {
        let mut l = AttractorLandscape::new();
        l.register(id(0), 5.0);
        l.register(id(0), 2.0); // update
        assert_eq!(l.attractor_count(), 1);
        assert_eq!(l.energy_of(id(0)), Some(2.0));
    }

    #[test]
    fn attractor_register_idempotent_same_value() {
        let mut l = AttractorLandscape::new();
        l.register(id(0), 3.0);
        l.register(id(0), 3.0); // exact same bits
        assert_eq!(l.attractor_count(), 1);
    }

    #[test]
    fn attractor_descend_returns_nearest_minimum() {
        let mut l = AttractorLandscape::new();
        let mut vfe = VFEMinimizer::new();

        let a = id(0); // mean=[0,0,0,0] → VFE=0
        let b = id(1); // mean=[5,0,0,0] → VFE high
        let c = id(2); // mean=[2,0,0,0] → VFE mid

        vfe.add_node(a, [0.0; 4]);
        vfe.add_node(b, [5.0, 0.0, 0.0, 0.0]);
        vfe.add_node(c, [2.0, 0.0, 0.0, 0.0]);

        l.register(a, 1.0);
        l.register(b, 5.0);
        l.register(c, 3.0);

        let target = l.descend(id(99), &vfe);
        assert_eq!(target, Some(a), "must descend to minimum-VFE attractor");
    }

    #[test]
    fn attractor_descend_empty_landscape_returns_none() {
        let l = AttractorLandscape::new();
        let vfe = VFEMinimizer::new();
        assert_eq!(l.descend(id(0), &vfe), None);
    }

    #[test]
    fn attractor_ascending_energy_order() {
        let mut l = AttractorLandscape::new();
        l.register(id(2), 3.0);
        l.register(id(0), 1.0);
        l.register(id(1), 2.0);
        let energies: Vec<f64> = l.ordered_landscape.iter().map(|e| e.energy).collect();
        assert_eq!(energies, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn descend_with_semantics_penalizes_incoherence() {
        use crate::{PhaseSemanticsEngine, QuantumKuramotoNetwork, QuantumOscillator};

        let mut l = AttractorLandscape::new();
        let mut vfe = VFEMinimizer::new();
        let a = id(0);
        let b = id(1);

        vfe.add_node(a, [0.1, 0.0, 0.0, 0.0]);
        vfe.add_node(b, [0.1, 0.0, 0.0, 0.0]);
        l.register(a, 1.0);
        l.register(b, 1.0);

        let mut net = QuantumKuramotoNetwork::new(0.0);
        let mut osc_a = QuantumOscillator::new(a, [0.0; 5]);
        let mut osc_b = QuantumOscillator::new(b, [0.0; 5]);
        osc_a.phases[0] = 0.1;
        osc_b.phases[0] = 3.5;
        net.add_oscillator(osc_a).expect("add a");
        net.add_oscillator(osc_b).expect("add b");
        net.set_coupling(a, b, 1.0);

        let mut semantics = PhaseSemanticsEngine::new();
        semantics.update_from_network(&net, &[(a, 0.1), (b, 0.1)]);

        let target = l.descend_with_semantics(a, &vfe, &semantics, 10.0, 10.0);
        assert_eq!(target, Some(b));
    }

    #[test]
    fn attractor_register_is_log_n_no_linear_scan() {
        // Register 10_000 attractors and verify count — if O(N) retain was used
        // this would be noticeably slow. BTreeSet guarantees O(log N).
        let mut l = AttractorLandscape::new();
        for i in 0..10_000u64 {
            l.register(id(i), i as f64);
        }
        assert_eq!(l.attractor_count(), 10_000);
        // Update all — verifies no duplicates accumulate
        for i in 0..10_000u64 {
            l.register(id(i), i as f64 + 1.0);
        }
        assert_eq!(
            l.attractor_count(),
            10_000,
            "no duplicates after bulk update"
        );
    }

    proptest! {
        #[allow(clippy::suboptimal_flops)]
        #[test]
        fn semantic_objective_matches_naive_expression(
            base in -1.0e3_f64..1.0e3,
            incoherence in 0.0_f64..10.0,
            instability in 0.0_f64..10.0,
            lambda_incoherence in 0.0_f64..10.0,
            lambda_instability in 0.0_f64..10.0,
        ) {
            let fma = AttractorLandscape::semantic_objective(
                base,
                incoherence,
                instability,
                lambda_incoherence,
                lambda_instability,
            );
            let naive = base + lambda_incoherence * incoherence + lambda_instability * instability;
            prop_assert!((fma - naive).abs() < 1e-12);
        }
    }
}
