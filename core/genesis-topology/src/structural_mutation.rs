//! Deterministic Structural Mutation Kernel (SMK) for local HNSW rewiring.
//!
//! AX-ID: AXIOMA-013, H_estructura
//!
//! INVARIANTS:
//! 1. Commutative signatures are stable under adjacency iteration reordering.
//! 2. `try_add_edge` is idempotent when the edge already exists.
//! 3. `try_remove_edge` rejects removals that would disconnect local topology.
//! 4. `try_rewire` is transactional: no degraded graph state is left behind.
//! 5. Slot signature `0` means no prior valid data for stability checks.
//! 6. Layer-0 edge counters remain synchronized on every edge mutation.

use genesis_types::proof::{AxiomSet, Proof, Witness};
use genesis_types::{GenesisError, NodeId};

use crate::hnsw::HnswGraph;
use smallvec::SmallVec;

/// Proof-carrying witness for one accepted structural mutation.
///
/// AX-ID: AXIOMA-013, H_estructura
#[derive(Debug, Clone)]
pub struct StructuralMutationWitness {
    /// Mutation source node.
    pub source: NodeId,
    /// Optional old target for remove/rewire.
    pub old_target: Option<NodeId>,
    /// Optional new target for add/rewire.
    pub new_target: Option<NodeId>,
    /// Local structural Hamiltonian delta.
    pub delta_h_structural: f64,
    /// Cryptographic proof over the mutation witness payload.
    pub proof: Proof,
}

/// Local deterministic kernel that applies bounded-energy topology mutations.
///
/// AX-ID: AXIOMA-013, H_estructura
pub struct StructuralMutationKernel {
    current_max_delta: f64,
    initial_max_delta: f64,
    equilibrium_signature_tolerance: f64,
    previous_signatures_by_id: Vec<(u64, u64)>,
    swap_buffer: Vec<u64>,
    pending_signatures: Vec<(u64, u64)>,
    sweep_count: u64,
    consecutive_stable_sweeps: u32,
    failed_memory: SmallVec<[FailedMutationMemory; 64]>,
}

#[derive(Debug, Clone, Copy)]
struct FailedMutationMemory {
    edge_hash: u64,
    rejection_count: u16,
    last_epoch: u32,
}

/// Structural phase metrics for external convergence instrumentation.
///
/// AX-ID: AXIOMA-013, H_estructura
#[derive(Debug, Clone, Copy)]
pub struct StructuralPhaseMetrics {
    /// Global structural energy proxy.
    pub global_energy: f64,
    /// Accepted/intents ratio for last sweep.
    pub accepted_ratio: f64,
    /// Fraction of rejected intents.
    pub rewiring_pressure: f64,
    /// Signature diversity ratio.
    pub topology_entropy: f64,
    /// Fraction of nodes with degree > 1 at layer 0.
    pub shortcut_density: f64,
}

impl StructuralMutationKernel {
    /// Creates a kernel with a strict acceptance threshold for positive ΔH.
    ///
    /// AX-ID: AXIOMA-013, H_estructura
    pub fn new(max_delta_increase: f64) -> Self {
        Self {
            current_max_delta: max_delta_increase,
            initial_max_delta: max_delta_increase,
            equilibrium_signature_tolerance: 1.0e-12,
            previous_signatures_by_id: Vec::new(),
            swap_buffer: Vec::new(),
            pending_signatures: Vec::new(),
            sweep_count: 0,
            consecutive_stable_sweeps: 0,
            failed_memory: SmallVec::new(),
        }
    }

    /// Executes a deterministic single sweep over live nodes.
    ///
    /// AX-ID: AXIOMA-013, H_estructura
    pub fn discover_hypotheses(
        &mut self,
        graph: &HnswGraph,
    ) -> Result<Vec<(NodeId, NodeId, NodeId)>, GenesisError> {
        let total_slots = graph.total_slots();
        self.swap_buffer.clear();
        self.swap_buffer.resize(total_slots, 0);
        self.pending_signatures.clear();
        self.sweep_count = self.sweep_count.saturating_add(1);
        let mut intents = Vec::new();
        for slot in 0..total_slots {
            let Some(id) = graph.node_id_at_slot(slot) else {
                continue;
            };
            let signature =
                graph.smk_local_energy_signature(id, self.equilibrium_signature_tolerance)?;
            let raw = id.get();
            let stable = self
                .previous_signatures_by_id
                .binary_search_by_key(&raw, |(nid, _)| *nid)
                .ok()
                .is_some_and(|pos| self.previous_signatures_by_id[pos].1 == signature);
            if stable {
                self.swap_buffer[slot] = signature;
                continue;
            }
            if let Some((old_b, new_b)) = graph.smk_find_local_rewire(id, self.current_max_delta)? {
                let edge_hash =
                    id.get() ^ old_b.get().rotate_left(21) ^ new_b.get().rotate_left(42);
                if self.should_suppress(edge_hash) {
                    self.swap_buffer[slot] = signature;
                    continue;
                }
                intents.push((id, old_b, new_b));
            }
            self.swap_buffer[slot] = signature;
            self.pending_signatures.push((raw, signature));
        }
        self.decay_failed_memory();
        Ok(intents)
    }

    /// Applies discovered intents on a mutable snapshot in deterministic order.
    ///
    /// AX-ID: AXIOMA-013, H_estructura
    pub fn apply_intents(
        &mut self,
        graph: &mut HnswGraph,
        intents: &[(NodeId, NodeId, NodeId)],
    ) -> Result<usize, GenesisError> {
        let mut accepted = 0usize;
        for &(src, old_b, new_b) in intents {
            let edge_hash = src.get() ^ old_b.get().rotate_left(21) ^ new_b.get().rotate_left(42);
            if graph
                .try_rewire(src, old_b, new_b, self.current_max_delta)?
                .is_some()
            {
                accepted += 1;
                if let Some(slot) = graph.slot_of_id(src) {
                    self.swap_buffer[slot] = graph
                        .smk_local_energy_signature(src, self.equilibrium_signature_tolerance)?;
                    self.pending_signatures
                        .push((src.get(), self.swap_buffer[slot]));
                }
                if let Some(slot) = graph.slot_of_id(old_b) {
                    self.swap_buffer[slot] = 0;
                }
                if let Some(slot) = graph.slot_of_id(new_b) {
                    self.swap_buffer[slot] = 0;
                }
            } else {
                self.record_failed_intent(edge_hash);
            }
        }
        if accepted == 0 && !intents.is_empty() {
            self.consecutive_stable_sweeps = self.consecutive_stable_sweeps.saturating_add(1);
            if self.consecutive_stable_sweeps > 3 {
                let decay = 0.5_f64.powi((self.consecutive_stable_sweeps - 3).cast_signed());
                self.current_max_delta =
                    (self.initial_max_delta * decay).max(self.initial_max_delta * 0.01);
            }
        } else {
            self.consecutive_stable_sweeps = 0;
        }
        Ok(accepted)
    }

    /// Commits signature state only after a successful snapshot publication.
    ///
    /// AX-ID: AXIOMA-013, H_estructura
    pub fn commit_signatures(&mut self) {
        self.previous_signatures_by_id = self.pending_signatures.clone();
        self.previous_signatures_by_id
            .sort_unstable_by_key(|(id, _)| *id);
        self.previous_signatures_by_id.dedup_by_key(|(id, _)| *id);
    }

    /// Returns convergence diagnostics `(sweep_count, current_max_delta)`.
    ///
    /// AX-ID: AXIOMA-013, H_estructura
    pub const fn convergence_stats(&self) -> (u64, f64) {
        (self.sweep_count, self.current_max_delta)
    }

    fn record_failed_intent(&mut self, edge_hash: u64) {
        for item in &mut self.failed_memory {
            if item.edge_hash == edge_hash {
                item.rejection_count = item.rejection_count.saturating_add(1);
                item.last_epoch = self.sweep_count as u32;
                return;
            }
        }
        if self.failed_memory.len() == self.failed_memory.capacity() {
            self.failed_memory.remove(0);
        }
        self.failed_memory.push(FailedMutationMemory {
            edge_hash,
            rejection_count: 1,
            last_epoch: self.sweep_count as u32,
        });
    }

    fn should_suppress(&self, edge_hash: u64) -> bool {
        self.failed_memory.iter().any(|item| {
            item.edge_hash == edge_hash
                && item.rejection_count >= 3
                && (self.sweep_count as u32).saturating_sub(item.last_epoch) < 16
        })
    }

    fn decay_failed_memory(&mut self) {
        let epoch = self.sweep_count as u32;
        for item in &mut self.failed_memory {
            let age = epoch.saturating_sub(item.last_epoch);
            if age > 32 && item.rejection_count > 0 {
                item.rejection_count -= 1;
                item.last_epoch = epoch;
            }
        }
        self.failed_memory.retain(|item| item.rejection_count > 0);
    }

    /// Computes phase metrics for external observability.
    ///
    /// AX-ID: AXIOMA-013, H_estructura
    pub fn phase_metrics(
        &self,
        graph: &HnswGraph,
        intents_count: usize,
        accepted_count: usize,
    ) -> StructuralPhaseMetrics {
        let global_energy = graph.global_structural_energy();
        let accepted_ratio = if intents_count == 0 {
            0.0
        } else {
            accepted_count as f64 / intents_count as f64
        };
        let rewiring_pressure = 1.0 - accepted_ratio;
        let nonzero = self.swap_buffer.iter().filter(|&&s| s != 0).count();
        let topology_entropy = if self.swap_buffer.is_empty() {
            0.0
        } else {
            nonzero as f64 / self.swap_buffer.len() as f64
        };
        let shortcut_density = graph.shortcut_density();
        StructuralPhaseMetrics {
            global_energy,
            accepted_ratio,
            rewiring_pressure,
            topology_entropy,
            shortcut_density,
        }
    }

    fn build_witness(
        source: NodeId,
        old_target: Option<NodeId>,
        new_target: Option<NodeId>,
        delta_h_structural: f64,
    ) -> Proof {
        let mut witness: Witness = Witness::new();
        witness.extend_from_slice(&source.get().to_le_bytes());
        witness.extend_from_slice(&old_target.unwrap_or(NodeId::INVALID).get().to_le_bytes());
        witness.extend_from_slice(&new_target.unwrap_or(NodeId::INVALID).get().to_le_bytes());
        witness.extend_from_slice(&delta_h_structural.to_le_bytes());
        Proof::new(
            AxiomSet::from_slice(genesis_types::proof::AxiomID::STRUCTURAL_REQUIRED),
            witness,
            0,
        )
    }

    pub(crate) fn accepted_witness(
        source: NodeId,
        old_target: Option<NodeId>,
        new_target: Option<NodeId>,
        delta_h_structural: f64,
    ) -> StructuralMutationWitness {
        StructuralMutationWitness {
            source,
            old_target,
            new_target,
            delta_h_structural,
            proof: Self::build_witness(source, old_target, new_target, delta_h_structural),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StructuralMutationKernel;
    use genesis_types::NodeId;

    #[test]
    fn accepted_witness_serializes_targets_and_delta() {
        let source = NodeId::try_new(1).expect("id");
        let old_target = NodeId::try_new(2).expect("id");
        let new_target = NodeId::try_new(3).expect("id");
        let witness = StructuralMutationKernel::accepted_witness(
            source,
            Some(old_target),
            Some(new_target),
            -0.25,
        );

        assert_eq!(witness.source, source);
        assert_eq!(witness.old_target, Some(old_target));
        assert_eq!(witness.new_target, Some(new_target));
        assert!((witness.delta_h_structural + 0.25).abs() < 1e-12);
        assert_ne!(witness.proof.hash, [0_u8; 32]);
    }

    #[test]
    fn failed_intents_are_suppressed_after_repeated_rejection() {
        let mut kernel = StructuralMutationKernel::new(0.1);
        let edge_hash = 0xCAFE_BABE_u64;
        assert!(!kernel.should_suppress(edge_hash));
        kernel.record_failed_intent(edge_hash);
        kernel.record_failed_intent(edge_hash);
        kernel.record_failed_intent(edge_hash);
        assert!(kernel.should_suppress(edge_hash));
        assert_eq!(kernel.convergence_stats(), (0, 0.1));
    }

    #[test]
    fn adaptive_sweep_behavior_decays_max_delta_on_stagnation() {
        let mut kernel = StructuralMutationKernel::new(1.0);
        let mut graph = crate::hnsw::HnswGraph::new(16);
        // Mock intents that will fail (empty graph, so try_rewire might fail or we just mock intents)
        let id = NodeId::try_new(1).expect("id");
        let intents = vec![(id, id, id)];

        // Force consecutive stable sweeps (accepted = 0)
        for _ in 0..10 {
            kernel.apply_intents(&mut graph, &intents).expect("apply");
        }

        let (_, current_delta) = kernel.convergence_stats();
        assert!(current_delta < 1.0, "Delta should have decayed from 1.0, got {current_delta}");
    }
}
