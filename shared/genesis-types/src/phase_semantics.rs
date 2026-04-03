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
#[repr(C)]
pub enum PhaseRegion {
    /// `[0, π/4)`
    Certainty,
    /// `[π/4, π/2)`
    Integration,
    /// `[π/2, π)`
    Exploration,
    /// `[π, 3π/2)`
    Tension,
    /// `[3π/2, 2π)`
    Release,
}

/// Primary semantic interpretation marker per node.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub enum SemanticMarker {
    /// Coherent low-surprise regime.
    Certainty,
    /// Integration regime with convergent coupling.
    Integration,
    /// Broad search / novelty regime.
    Exploration,
    /// Divergent neighboring semantics.
    Conflict,
    /// Low-amplitude semantic collapse.
    Collapse,
}

/// Metastable cognitive regime of the network.
///
/// AX-ID: AXIOMA-005, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum MetaState {
    /// High stability and coherent semantics.
    StableMeaning,
    /// Dynamic exploratory regime.
    ExploratoryFlux,
    /// High local semantic tension.
    CognitiveTension,
    /// System-wide semantic degradation.
    SemanticCollapse,
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
    /// Builds a semantic cluster from an ordered node slice.
    ///
    /// Returns `None` when the slice exceeds `SEMANTIC_CLUSTER_MAX_NODES`.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006
    #[must_use]
    pub fn from_nodes(nodes: &[NodeId], marker: SemanticMarker, coherence: f64) -> Option<Self> {
        if nodes.len() > SEMANTIC_CLUSTER_MAX_NODES {
            return None;
        }

        let mut fixed = [NodeId::INVALID; SEMANTIC_CLUSTER_MAX_NODES];
        fixed[..nodes.len()].copy_from_slice(nodes);

        Some(Self {
            nodes: fixed,
            node_count: nodes.len() as u8,
            marker,
            coherence,
        })
    }

    /// Returns active cluster members as a slice.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006
    #[must_use]
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes[..usize::from(self.node_count)]
    }
}
