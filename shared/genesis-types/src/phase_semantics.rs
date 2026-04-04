//! Canonical primitive phase-semantics types shared across GÉNESIS crates.
//!
//! This module centralizes semantic markers and records used by runtime
//! interpretation layers while keeping CRATE-000 free of dynamics logic.
//!
//! AX-ID: AXIOMA-004, AXIOMA-006, H_dinámica, H_información

use crate::NodeId;

/// Maximum inline capacity for a semantic cluster node list.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
pub const SEMANTIC_CLUSTER_MAX_NODES: usize = 8;

/// Semantic partition of wrapped phase `[0, 2π)`.
///
/// AX-ID: AXIOMA-006, H_dinámica
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PhaseRegion {
    /// `[0, π/4)`
    Certainty = 0,
    /// `[π/4, π/2)`
    Integration = 1,
    /// `[π/2, π)`
    Exploration = 2,
    /// `[π, 3π/2)`
    Tension = 3,
    /// `[3π/2, 2π)`
    Release = 4,
}

/// Primary semantic interpretation marker per node.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SemanticMarker {
    /// Coherent low-surprise regime.
    Certainty = 0,
    /// Integration regime with convergent coupling.
    Integration = 1,
    /// Broad search / novelty regime.
    Exploration = 2,
    /// Divergent neighboring semantics.
    Conflict = 3,
    /// Low-amplitude semantic collapse.
    Collapse = 4,
}

/// Metastable cognitive regime of the network.
///
/// AX-ID: AXIOMA-005, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MetaState {
    /// High stability and coherent semantics.
    StableMeaning = 0,
    /// Dynamic exploratory regime.
    ExploratoryFlux = 1,
    /// High local semantic tension.
    CognitiveTension = 2,
    /// System-wide semantic degradation.
    SemanticCollapse = 3,
}

/// Local semantic state for one node.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct NodeSemanticState {
    /// Node identifier.
    pub node: NodeId,
    /// Current semantic marker.
    pub marker: SemanticMarker,
    /// Primary phase (grade-0).
    pub phase: f64,
    /// Primary amplitude summary (`amplitude_norm`).
    pub amplitude: f64,
    /// Stability score in `[0,1]` derived from short-term phase variance.
    pub stability: f64,
}

/// Global distributed cognitive field summary.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CognitiveFieldState {
    /// Most frequent marker in network.
    pub dominant_marker: SemanticMarker,
    /// Mean phase alignment (`r_sync`-like).
    pub coherence: f64,
    /// Marker entropy in nats.
    pub semantic_entropy: f64,
    /// Mean neighbor divergence.
    pub tension: f64,
}

/// Edge-level semantic tension descriptor.
///
/// AX-ID: AXIOMA-004, H_dinámica
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct SemanticTensionEdge {
    /// First endpoint.
    pub node_a: NodeId,
    /// Second endpoint.
    pub node_b: NodeId,
    /// Wrapped phase divergence in radians `[0, π]`.
    pub divergence: f64,
}

/// Short persistent semantic trace per node.
///
/// AX-ID: AXIOMA-004
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SemanticTrace {
    /// Previous semantic marker.
    pub previous_marker: SemanticMarker,
    /// Consecutive duration in same marker.
    pub duration: u32,
}

/// Resonant local cluster of semantically aligned nodes.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct SemanticCluster {
    /// Cluster member nodes in canonical sorted order.
    pub nodes: [NodeId; SEMANTIC_CLUSTER_MAX_NODES],
    /// Number of active entries in `nodes`.
    pub node_count: u8,
    /// Cluster representative marker.
    pub marker: SemanticMarker,
    /// Mean local coherence in `[0,1]`.
    pub coherence: f64,
}

/// Network-level semantic state abstraction.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct NetworkSemanticState {
    /// Dominant semantic marker.
    pub dominant_state: SemanticMarker,
    /// Global phase coherence.
    pub coherence: f64,
    /// Marker diversity (entropy).
    pub diversity: f64,
    /// Temporal marker variability proxy.
    pub metastability: f64,
}

impl SemanticCluster {
    /// Canonicalizes a candidate node slice into fixed-capacity storage.
    ///
    /// The input must satisfy all of the following:
    /// - length `<= SEMANTIC_CLUSTER_MAX_NODES`
    /// - strictly increasing `NodeId` order
    /// - no `NodeId::INVALID` sentinel entries
    ///
    /// Returns an error when any invariant is violated.
    ///
    /// # Errors
    /// Returns `GenesisError::InvariantViolation` for capacity/ordering
    /// violations and `GenesisError::NodeIdOutOfRange` when the invalid
    /// sentinel `NodeId::INVALID` appears in the input.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006
    pub fn canonicalize_nodes(
        nodes: &[NodeId],
    ) -> Result<([NodeId; SEMANTIC_CLUSTER_MAX_NODES], u8), crate::GenesisError> {
        if nodes.len() > SEMANTIC_CLUSTER_MAX_NODES {
            return Err(crate::GenesisError::InvariantViolation { axiom_id: 4 });
        }

        let mut previous: Option<NodeId> = None;
        for &node in nodes {
            if node == NodeId::INVALID {
                return Err(crate::GenesisError::NodeIdOutOfRange {
                    raw: NodeId::INVALID.get(),
                });
            }
            if let Some(prev) = previous {
                if node <= prev {
                    return Err(crate::GenesisError::InvariantViolation { axiom_id: 4 });
                }
            }
            previous = Some(node);
        }

        let mut fixed = [NodeId::INVALID; SEMANTIC_CLUSTER_MAX_NODES];
        fixed[..nodes.len()].copy_from_slice(nodes);

        Ok((fixed, nodes.len() as u8))
    }

    /// Builds a semantic cluster from an ordered node slice.
    ///
    /// Returns an error when validation fails.
    ///
    /// # Errors
    /// Propagates canonicalization failures from `canonicalize_nodes`,
    /// including ordering/capacity violations and invalid sentinel entries.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006
    pub fn from_nodes(
        nodes: &[NodeId],
        marker: SemanticMarker,
        coherence: f64,
    ) -> Result<Self, crate::GenesisError> {
        let (fixed, node_count) = Self::canonicalize_nodes(nodes)?;

        Ok(Self {
            nodes: fixed,
            node_count,
            marker,
            coherence,
        })
    }

    /// Returns the canonical fixed-storage key used for deterministic identity
    /// checks and deduplication.
    ///
    /// # Errors
    /// Returns `GenesisError::InvariantViolation` when `node_count` exceeds the
    /// fixed node storage length.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006
    pub fn canonical_key(
        &self,
    ) -> Result<([NodeId; SEMANTIC_CLUSTER_MAX_NODES], u8), crate::GenesisError> {
        let node_count = usize::from(self.node_count);
        if node_count > self.nodes.len() {
            return Err(crate::GenesisError::InvariantViolation { axiom_id: 4 });
        }

        let active = &self.nodes[..node_count];
        let (fixed, canonical_count) = Self::canonicalize_nodes(active)?;

        if fixed[node_count..]
            .iter()
            .any(|node| *node != NodeId::INVALID)
        {
            return Err(crate::GenesisError::InvariantViolation { axiom_id: 4 });
        }

        Ok((fixed, canonical_count))
    }

    /// Returns active cluster members as a slice.
    ///
    /// # Errors
    /// Returns `GenesisError::InvariantViolation` when `node_count` exceeds the
    /// fixed node storage length.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006
    pub fn nodes(&self) -> Result<&[NodeId], crate::GenesisError> {
        let node_count = usize::from(self.node_count);
        if node_count > self.nodes.len() {
            return Err(crate::GenesisError::InvariantViolation { axiom_id: 4 });
        }
        Ok(&self.nodes[..node_count])
    }
}

#[cfg(test)]
mod tests {
    use super::{SemanticCluster, SemanticMarker, SEMANTIC_CLUSTER_MAX_NODES};
    use crate::{GenesisError, NodeId};

    #[test]
    fn semantic_cluster_from_nodes_valid_slice_roundtrips() {
        let input = [
            NodeId::try_new(1).expect("1 is inside the valid NodeId range"),
            NodeId::try_new(4).expect("4 is inside the valid NodeId range"),
            NodeId::try_new(9).expect("9 is inside the valid NodeId range"),
        ];
        let cluster = SemanticCluster::from_nodes(&input, SemanticMarker::Exploration, 0.75)
            .expect("valid sorted unique input must produce a cluster");

        assert_eq!(usize::from(cluster.node_count), input.len());
        assert_eq!(
            cluster.nodes().expect("node_count is valid"),
            input.as_slice()
        );
        assert_eq!(
            cluster.nodes().expect("node_count is valid").len(),
            input.len()
        );
    }

    #[test]
    fn semantic_cluster_from_nodes_rejects_oversized_input() {
        let oversized = [NodeId::INVALID; SEMANTIC_CLUSTER_MAX_NODES + 1];
        assert!(SemanticCluster::from_nodes(&oversized, SemanticMarker::Certainty, 1.0).is_err());
    }

    #[test]
    fn semantic_cluster_from_nodes_rejects_invalid_sentinel_within_capacity() {
        let within_capacity = [
            NodeId::try_new(1).expect("1 is inside the valid NodeId range"),
            NodeId::INVALID,
        ];
        assert!(
            SemanticCluster::from_nodes(&within_capacity, SemanticMarker::Certainty, 1.0).is_err()
        );
    }

    #[test]
    fn semantic_cluster_from_nodes_rejects_non_monotonic_or_duplicate_input() {
        let duplicate = [
            NodeId::try_new(2).expect("2 is inside the valid NodeId range"),
            NodeId::try_new(2).expect("2 is inside the valid NodeId range"),
        ];
        assert!(SemanticCluster::from_nodes(&duplicate, SemanticMarker::Conflict, 0.2).is_err());

        let non_monotonic = [
            NodeId::try_new(5).expect("5 is inside the valid NodeId range"),
            NodeId::try_new(3).expect("3 is inside the valid NodeId range"),
        ];
        assert!(
            SemanticCluster::from_nodes(&non_monotonic, SemanticMarker::Conflict, 0.2).is_err()
        );
    }

    #[test]
    fn semantic_cluster_canonical_key_rejects_out_of_range_node_count() {
        let invalid = SemanticCluster {
            nodes: [NodeId::INVALID; SEMANTIC_CLUSTER_MAX_NODES],
            node_count: (SEMANTIC_CLUSTER_MAX_NODES + 1) as u8,
            marker: SemanticMarker::Certainty,
            coherence: 1.0,
        };

        assert_eq!(
            invalid.canonical_key(),
            Err(GenesisError::InvariantViolation { axiom_id: 4 })
        );
        assert_eq!(
            invalid.nodes(),
            Err(GenesisError::InvariantViolation { axiom_id: 4 })
        );
    }
}
