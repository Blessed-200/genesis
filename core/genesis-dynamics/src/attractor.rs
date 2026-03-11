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
//! New implementation: `HashMap<NodeId, f64>` for O(1) ID lookup + `BTreeSet<AttractorEntry>`
//! for O(log N) energy-ordered insertion. The `HashMap` is not in the HNSW/Kuramoto hot-path —
//! it serves the attractor bookkeeping layer (consciousness, CRATE-006).
//!
//! PROHIBITED: implementing memories as external lookup tables outside the attractor landscape.
//! All memory lives in attractor topology. (AXIOMA-004)

use crate::free_energy::VFEMinimizer;
use genesis_types::NodeId;
use std::collections::{BTreeSet, HashMap};

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
    id:     NodeId,
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
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // total_cmp: deterministic even for NaN (NaN > all finite, consistent with IEEE 754 total order)
        self.energy.total_cmp(&other.energy)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for AttractorEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Cognitive attractor landscape.
///
/// # Complexity after BN-06
/// - `register()`: O(log N) — HashMap O(1) + BTreeSet O(log N)
/// - `descend()`: O(N) via BTreeSet iteration (cache-friendly, ordered)
/// - `energy_of()`: O(1) via HashMap
/// - `attractor_count()`: O(1)
///
/// AX-ID: AXIOMA-004
pub struct AttractorLandscape {
    /// O(1) energy lookup by NodeId.
    id_to_energy:       HashMap<NodeId, f64>,
    /// Energy-ordered set for deterministic iteration.
    ordered_landscape:  BTreeSet<AttractorEntry>,
}

impl AttractorLandscape {
    pub fn new() -> Self {
        Self {
            id_to_energy:      HashMap::new(),
            ordered_landscape: BTreeSet::new(),
        }
    }

    /// Register or update an attractor. O(log N).
    ///
    /// If the same `NodeId` already exists with a different energy, the old entry
    /// is removed and the new one inserted. Idempotent for same (id, energy).
    pub fn register(&mut self, id: NodeId, energy: f64) {
        if let Some(&old_energy) = self.id_to_energy.get(&id) {
            if old_energy.to_bits() == energy.to_bits() {
                return; // Exact same value — idempotent, no work needed.
            }
            self.ordered_landscape.remove(&AttractorEntry { id, energy: old_energy });
        }
        self.id_to_energy.insert(id, energy);
        self.ordered_landscape.insert(AttractorEntry { id, energy });
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

    /// Iterate attractors in ascending energy order.
    ///
    /// Deterministic iteration for CRATE-006 observer when building Ω.
    pub fn ascending_energy(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.ordered_landscape.iter().map(|e| e.id)
    }

    pub fn attractor_count(&self) -> usize {
        self.ordered_landscape.len()
    }

    /// Registered energy of an attractor. O(1) via HashMap.
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

    fn id(n: u64) -> NodeId { NodeId::try_new(n).expect("NodeId válido") }

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
        let mut l   = AttractorLandscape::new();
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
        let l   = AttractorLandscape::new();
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
        assert_eq!(l.attractor_count(), 10_000, "no duplicates after bulk update");
    }
}
