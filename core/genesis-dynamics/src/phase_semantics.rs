//! Phase Semantics Engine — interpretación cognitiva del campo oscilatorio.
//!
//! Traduce fase/amplitud/gradiente VFE en estados semánticos locales y métricas
//! globales de campo, sin modificar directamente la dinámica de Kuramoto.
//!
//! AX-ID: AXIOMA-003, AXIOMA-004, AXIOMA-006, `H_dinámica`, `H_información`

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use genesis_types::NodeId;
use smallvec::SmallVec;

use crate::kuramoto::QuantumKuramotoNetwork;

const CERTAINTY_AMPLITUDE_MIN: f64 = 0.8;
const CERTAINTY_DELTA_G_MAX: f64 = 0.15;
const CERTAINTY_COHERENCE_MIN: f64 = 0.9;
const COLLAPSE_AMPLITUDE_MAX: f64 = 0.2;
const COLLAPSE_DELTA_G_MIN: f64 = 1.0;
const TENSION_DIVERGENCE_MIN: f64 = 1.2;
const STABILITY_VARIANCE_SCALE: f64 = 1.0;
const RESONANCE_DIVERGENCE_MAX: f64 = 0.35;

/// Semantic partition of wrapped phase `[0, 2π)`.
///
/// AX-ID: AXIOMA-006, `H_dinámica`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// AX-ID: AXIOMA-004, `H_dinámica`
#[derive(Debug, Clone, Copy, PartialEq)]
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
pub struct SemanticTrace {
    /// Previous semantic marker.
    pub previous_marker: SemanticMarker,
    /// Consecutive duration in same marker.
    pub duration: u32,
}

/// Resonant local cluster of semantically aligned nodes.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticCluster {
    /// Cluster member nodes.
    pub nodes: SmallVec<[NodeId; 8]>,
    /// Cluster representative marker.
    pub marker: SemanticMarker,
    /// Mean local coherence in `[0,1]`.
    pub coherence: f64,
}

/// Network-level semantic state abstraction.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Copy)]
struct NodeSemanticEntry {
    state: NodeSemanticState,
    trace: SemanticTrace,
    delta_g: f64,
    phase_variance: f64,
    local_coherence: f64,
    local_divergence: f64,
}

#[derive(Debug)]
struct ClusterAssignment {
    nodes: SmallVec<[NodeId; 8]>,
    marker: SemanticMarker,
    coherence: f64,
}

/// Engine that interprets oscillator dynamics as semantic field descriptors.
///
/// This structure is deterministic: node entries are stored sorted by `NodeId`
/// and all floating-point ordering uses `total_cmp` when needed.
///
/// AX-ID: AXIOMA-004, AXIOMA-006, `H_dinámica`, `H_información`
#[derive(Debug)]
pub struct PhaseSemanticsEngine {
    entries: Vec<NodeSemanticEntry>,
    tension_edges: Vec<SemanticTensionEdge>,
    clusters: SmallVec<[SemanticCluster; 8]>,
    field_state: CognitiveFieldState,
    network_state: NetworkSemanticState,
    metastate: MetaState,
}

impl Default for PhaseSemanticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseSemanticsEngine {
    /// Creates an empty phase semantics engine.
    ///
    /// AX-ID: AXIOMA-004
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            tension_edges: Vec::new(),
            clusters: SmallVec::new(),
            field_state: CognitiveFieldState {
                dominant_marker: SemanticMarker::Exploration,
                coherence: 0.0,
                semantic_entropy: 0.0,
                tension: 0.0,
            },
            network_state: NetworkSemanticState {
                dominant_state: SemanticMarker::Exploration,
                coherence: 0.0,
                diversity: 0.0,
                metastability: 0.0,
            },
            metastate: MetaState::ExploratoryFlux,
        }
    }

    /// Updates semantic interpretation from a Kuramoto snapshot and per-node `Δg`.
    ///
    /// `delta_g` must contain `(NodeId, gradient_magnitude)` pairs. Missing nodes
    /// default to `0.0` gradient.
    ///
    /// AX-ID: AXIOMA-003, AXIOMA-006
    pub fn update_from_network(
        &mut self,
        network: &QuantumKuramotoNetwork,
        delta_g: &[(NodeId, f64)],
    ) {
        self.entries.clear();
        self.tension_edges.clear();
        self.clusters.clear();

        self.entries.reserve(network.oscillators.len());
        self.tension_edges.reserve(network.coupling.len());

        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;

        for osc in &network.oscillators {
            let phase = wrap_phase(osc.primary_phase());
            sum_cos += phase.cos();
            sum_sin += phase.sin();

            let amplitude = osc.amplitude_norm();
            let gradient = lookup_delta_g(delta_g, osc.node_id);
            self.entries.push(NodeSemanticEntry {
                state: NodeSemanticState {
                    node: osc.node_id,
                    marker: SemanticMarker::Exploration,
                    phase,
                    amplitude,
                    stability: 0.0,
                },
                trace: SemanticTrace {
                    previous_marker: SemanticMarker::Exploration,
                    duration: 1,
                },
                delta_g: gradient,
                phase_variance: 0.0,
                local_coherence: 1.0,
                local_divergence: 0.0,
            });
        }

        self.entries.sort_by_key(|entry| entry.state.node);
        let neighbor_index = build_neighbor_index(&self.entries, &network.coupling);
        let phases: Vec<f64> = self.entries.iter().map(|entry| entry.state.phase).collect();
        for (idx, entry) in self.entries.iter_mut().enumerate() {
            let phase = entry.state.phase;
            let (coherence, divergence, mean_phase) =
                local_phase_stats_indexed(idx, phase, &phases, &neighbor_index);
            let variance = wrapped_distance_sq(phase, mean_phase);
            let stability = (1.0 - variance / STABILITY_VARIANCE_SCALE).clamp(0.0, 1.0);
            let marker = marker_from_signals(
                phase,
                entry.state.amplitude,
                entry.delta_g,
                coherence,
                divergence,
            );
            entry.state.marker = marker;
            entry.state.stability = stability;
            entry.trace.previous_marker = marker;
            entry.phase_variance = variance;
            entry.local_coherence = coherence;
            entry.local_divergence = divergence;
        }
        self.refresh_traces();
        self.build_tension_edges(network);
        self.build_clusters(network);
        self.compute_field_and_network_state(sum_cos, sum_sin);
        self.metastate = infer_metastate(self.network_state, self.field_state, &self.entries);
    }

    /// Returns semantic state for one node.
    ///
    /// AX-ID: AXIOMA-004
    #[must_use]
    pub fn node_semantic_state(&self, node: NodeId) -> NodeSemanticState {
        self.entries
            .binary_search_by_key(&node, |entry| entry.state.node)
            .ok()
            .map_or(
                NodeSemanticState {
                    node,
                    marker: SemanticMarker::Exploration,
                    phase: 0.0,
                    amplitude: 0.0,
                    stability: 0.0,
                },
                |idx| self.entries[idx].state,
            )
    }

    /// Returns global network semantic summary.
    ///
    /// AX-ID: AXIOMA-004
    #[must_use]
    pub const fn network_semantic_state(&self) -> NetworkSemanticState {
        self.network_state
    }

    /// Returns distributed cognitive field summary.
    ///
    /// AX-ID: AXIOMA-004
    #[must_use]
    pub const fn cognitive_field_state(&self) -> CognitiveFieldState {
        self.field_state
    }

    /// Returns local semantic resonance clusters.
    ///
    /// AX-ID: AXIOMA-004, AXIOMA-006
    #[must_use]
    pub fn semantic_clusters(&self) -> SmallVec<[SemanticCluster; 8]> {
        self.clusters.clone()
    }

    /// Returns current metastability regime.
    ///
    /// AX-ID: AXIOMA-005
    #[must_use]
    pub const fn metastate(&self) -> MetaState {
        self.metastate
    }

    /// Returns semantic tension edges above threshold.
    ///
    /// AX-ID: AXIOMA-004
    #[must_use]
    pub fn semantic_tension_edges(&self) -> &[SemanticTensionEdge] {
        &self.tension_edges
    }

    /// Node-local semantic incoherence proxy for attractor penalization.
    ///
    /// AX-ID: AXIOMA-004, `H_información`
    #[must_use]
    pub fn semantic_incoherence(&self, node: NodeId) -> f64 {
        self.entries
            .binary_search_by_key(&node, |entry| entry.state.node)
            .ok()
            .map_or(0.0, |idx| self.entries[idx].local_divergence)
    }

    /// Node-local phase instability proxy for attractor penalization.
    ///
    /// AX-ID: AXIOMA-006, `H_dinámica`
    #[must_use]
    pub fn phase_instability(&self, node: NodeId) -> f64 {
        self.entries
            .binary_search_by_key(&node, |entry| entry.state.node)
            .ok()
            .map_or(0.0, |idx| self.entries[idx].phase_variance)
    }

    fn refresh_traces(&mut self) {
        for entry in &mut self.entries {
            let same_marker = entry.trace.previous_marker == entry.state.marker;
            let same_mask = (same_marker as u32).wrapping_neg();
            let next_duration = entry.trace.duration.saturating_add(1);
            // CRYSTAL: O88 — inevitable
            entry.trace.duration = (next_duration & same_mask) | (1_u32 & !same_mask);
            entry.trace.previous_marker =
                [entry.state.marker, entry.trace.previous_marker][same_marker as usize];
        }
    }

    fn build_tension_edges(&mut self, network: &QuantumKuramotoNetwork) {
        for &(a, b, _, _) in &network.coupling {
            let state_a = self.node_semantic_state(a);
            let state_b = self.node_semantic_state(b);
            if state_a.marker != state_b.marker {
                let divergence = wrapped_distance(state_a.phase, state_b.phase);
                if divergence >= TENSION_DIVERGENCE_MIN {
                    self.tension_edges.push(SemanticTensionEdge {
                        node_a: a,
                        node_b: b,
                        divergence,
                    });
                }
            }
        }
        self.tension_edges.sort_by(|lhs, rhs| {
            lhs.divergence
                .total_cmp(&rhs.divergence)
                .reverse()
                .then_with(|| lhs.node_a.cmp(&rhs.node_a))
                .then_with(|| lhs.node_b.cmp(&rhs.node_b))
        });
    }

    fn build_clusters(&mut self, network: &QuantumKuramotoNetwork) {
        for entry in &self.entries {
            if let Some(assignment) =
                assign_node_to_cluster(entry, &network.coupling, &self.entries)
            {
                if !self
                    .clusters
                    .iter()
                    .any(|cluster| cluster.nodes == assignment.nodes)
                {
                    self.clusters.push(SemanticCluster {
                        nodes: assignment.nodes,
                        marker: assignment.marker,
                        coherence: assignment.coherence,
                    });
                }
            }
        }

        self.clusters.sort_by(|lhs, rhs| {
            rhs.coherence
                .total_cmp(&lhs.coherence)
                .then_with(|| lhs.marker.cmp(&rhs.marker))
                .then_with(|| lhs.nodes.len().cmp(&rhs.nodes.len()).reverse())
        });
    }

    fn compute_field_and_network_state(&mut self, sum_cos: f64, sum_sin: f64) {
        let n = self.entries.len() as f64;
        if n == 0.0 {
            self.field_state = CognitiveFieldState {
                dominant_marker: SemanticMarker::Exploration,
                coherence: 0.0,
                semantic_entropy: 0.0,
                tension: 0.0,
            };
            self.network_state = NetworkSemanticState {
                dominant_state: SemanticMarker::Exploration,
                coherence: 0.0,
                diversity: 0.0,
                metastability: 0.0,
            };
            return;
        }

        let coherence = (sum_cos.hypot(sum_sin) / n).clamp(0.0, 1.0);
        let (dominant_marker, entropy) = dominant_and_entropy(&self.entries);
        let tension = if self.tension_edges.is_empty() {
            0.0
        } else {
            self.tension_edges.iter().map(|e| e.divergence).sum::<f64>()
                / self.tension_edges.len() as f64
        };
        let metastability = self
            .entries
            .iter()
            .map(|entry| 1.0 - entry.state.stability)
            .sum::<f64>()
            / n;

        self.field_state = CognitiveFieldState {
            dominant_marker,
            coherence,
            semantic_entropy: entropy,
            tension,
        };
        self.network_state = NetworkSemanticState {
            dominant_state: dominant_marker,
            coherence,
            diversity: entropy,
            metastability,
        };
    }
}

#[inline]
fn assign_node_to_cluster(
    entry: &NodeSemanticEntry,
    coupling: &[(NodeId, NodeId, f64, f64)],
    entries: &[NodeSemanticEntry],
) -> Option<ClusterAssignment> {
    let mut nodes: SmallVec<[NodeId; 8]> = SmallVec::new();
    nodes.push(entry.state.node);
    let mut coherence_sum = 1.0;
    let mut count = 1.0;

    for &(src, dst, _, _) in coupling {
        if src != entry.state.node {
            continue;
        }
        let neighbor = node_semantic_state_from_entries(entries, dst);
        if neighbor.marker != entry.state.marker {
            continue;
        }
        let divergence = wrapped_distance(entry.state.phase, neighbor.phase);
        if divergence <= RESONANCE_DIVERGENCE_MAX {
            nodes.push(dst);
            coherence_sum += 1.0 - (divergence / PI).clamp(0.0, 1.0);
            count += 1.0;
        }
    }

    if nodes.len() <= 1 {
        return None;
    }

    nodes.sort_unstable();
    Some(ClusterAssignment {
        nodes,
        marker: entry.state.marker,
        coherence: coherence_sum / count,
    })
}

impl Ord for SemanticMarker {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        marker_rank(*self).cmp(&marker_rank(*other))
    }
}

impl PartialOrd for SemanticMarker {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
const fn marker_rank(marker: SemanticMarker) -> u8 {
    match marker {
        SemanticMarker::Certainty => 0,
        SemanticMarker::Integration => 1,
        SemanticMarker::Exploration => 2,
        SemanticMarker::Conflict => 3,
        SemanticMarker::Collapse => 4,
    }
}

#[inline]
fn wrap_phase(phase: f64) -> f64 {
    let wrapped = phase.rem_euclid(TAU);
    if wrapped == TAU {
        0.0
    } else {
        wrapped
    }
}

#[inline]
fn wrapped_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(TAU - d)
}

#[inline]
fn wrapped_distance_sq(a: f64, b: f64) -> f64 {
    let d = wrapped_distance(a, b);
    d * d
}

#[inline]
fn lookup_delta_g(delta_g: &[(NodeId, f64)], node: NodeId) -> f64 {
    delta_g
        .binary_search_by_key(&node, |&(id, _)| id)
        .ok()
        .map_or(0.0, |idx| delta_g[idx].1)
}

#[inline]
fn node_semantic_state_from_entries(entries: &[NodeSemanticEntry], node: NodeId) -> NodeSemanticState {
    entries
        .binary_search_by_key(&node, |entry| entry.state.node)
        .ok()
        .map_or(
            NodeSemanticState {
                node,
                marker: SemanticMarker::Exploration,
                phase: 0.0,
                amplitude: 0.0,
                stability: 0.0,
            },
            |idx| entries[idx].state,
        )
}

#[inline]
fn phase_region(phase: f64) -> PhaseRegion {
    if phase < FRAC_PI_4 {
        PhaseRegion::Certainty
    } else if phase < FRAC_PI_2 {
        PhaseRegion::Integration
    } else if phase < PI {
        PhaseRegion::Exploration
    } else if phase < 1.5 * PI {
        PhaseRegion::Tension
    } else {
        PhaseRegion::Release
    }
}

#[inline]
fn marker_from_signals(
    phase: f64,
    amplitude: f64,
    delta_g: f64,
    local_coherence: f64,
    local_divergence: f64,
) -> SemanticMarker {
    if amplitude <= COLLAPSE_AMPLITUDE_MAX && delta_g >= COLLAPSE_DELTA_G_MIN {
        return SemanticMarker::Collapse;
    }
    if amplitude >= CERTAINTY_AMPLITUDE_MIN
        && delta_g <= CERTAINTY_DELTA_G_MAX
        && local_coherence >= CERTAINTY_COHERENCE_MIN
    {
        return SemanticMarker::Certainty;
    }
    if delta_g >= 0.8 {
        return SemanticMarker::Exploration;
    }
    if local_divergence >= TENSION_DIVERGENCE_MIN {
        return SemanticMarker::Conflict;
    }

    match phase_region(phase) {
        PhaseRegion::Certainty => SemanticMarker::Certainty,
        PhaseRegion::Integration => SemanticMarker::Integration,
        PhaseRegion::Exploration | PhaseRegion::Release => SemanticMarker::Exploration,
        PhaseRegion::Tension => SemanticMarker::Conflict,
    }
}

fn build_neighbor_index(
    entries: &[NodeSemanticEntry],
    edges: &[(NodeId, NodeId, f64, f64)],
) -> Vec<Vec<usize>> {
    let node_ids: Vec<NodeId> = entries.iter().map(|entry| entry.state.node).collect();
    let mut node_to_neighbors = vec![Vec::new(); entries.len()];
    for &(src, dst, _, _) in edges {
        let Ok(src_idx) = node_ids.binary_search(&src) else {
            continue;
        };
        let Ok(dst_idx) = node_ids.binary_search(&dst) else {
            continue;
        };
        node_to_neighbors[src_idx].push(dst_idx);
    }
    node_to_neighbors
}

fn local_phase_stats_indexed(
    node_idx: usize,
    phase: f64,
    phases: &[f64],
    node_to_neighbors: &[Vec<usize>],
) -> (f64, f64, f64) {
    let mut coherence_acc = 0.0;
    let mut divergence_acc = 0.0;
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    let mut count = 0usize;

    for &neighbor_idx in &node_to_neighbors[node_idx] {
        let neighbor_phase = phases[neighbor_idx];
        let divergence = wrapped_distance(phase, neighbor_phase);
        coherence_acc += 1.0 - (divergence / PI).clamp(0.0, 1.0);
        divergence_acc += divergence;
        sum_cos += neighbor_phase.cos();
        sum_sin += neighbor_phase.sin();
        count += 1;
    }

    if count == 0 {
        (1.0, 0.0, 0.0)
    } else {
        let count_f = count as f64;
        (
            coherence_acc / count_f,
            divergence_acc / count_f,
            wrap_phase(sum_sin.atan2(sum_cos)),
        )
    }
}

fn dominant_and_entropy(entries: &[NodeSemanticEntry]) -> (SemanticMarker, f64) {
    let mut counts = [0usize; 5];
    for entry in entries {
        counts[usize::from(marker_rank(entry.state.marker))] += 1;
    }

    let mut dominant_idx = 0usize;
    for idx in 1..counts.len() {
        if counts[idx] > counts[dominant_idx] {
            dominant_idx = idx;
        }
    }

    let total = entries.len() as f64;
    let inv_total = 1.0 / total;
    // loop-invariant, hoisted
    // CRYSTAL: O57 — inevitable
    let mut entropy = 0.0;
    for count in counts {
        if count == 0 {
            continue;
        }
        let p = count as f64 * inv_total;
        entropy -= p * p.ln();
    }

    let marker = match dominant_idx {
        0 => SemanticMarker::Certainty,
        1 => SemanticMarker::Integration,
        2 => SemanticMarker::Exploration,
        3 => SemanticMarker::Conflict,
        _ => SemanticMarker::Collapse,
    };

    (marker, entropy)
}

fn infer_metastate(
    network: NetworkSemanticState,
    field: CognitiveFieldState,
    entries: &[NodeSemanticEntry],
) -> MetaState {
    let mean_delta = if entries.is_empty() {
        0.0
    } else {
        entries.iter().map(|entry| entry.delta_g).sum::<f64>() / entries.len() as f64
    };

    if matches!(network.dominant_state, SemanticMarker::Collapse) {
        return MetaState::SemanticCollapse;
    }
    if field.tension >= 1.0 {
        return MetaState::CognitiveTension;
    }
    if network.coherence >= 0.85 && network.metastability <= 0.2 && mean_delta <= 0.2 {
        return MetaState::StableMeaning;
    }
    MetaState::ExploratoryFlux
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oscillator::QuantumOscillator;

    fn id(raw: u64) -> NodeId {
        NodeId::try_new(raw).expect("NodeId válido")
    }

    fn network_with_phases(phases: &[f64], amplitudes: &[f64]) -> QuantumKuramotoNetwork {
        let mut network = QuantumKuramotoNetwork::new(0.0);
        for (idx, (&phase, &amplitude)) in phases.iter().zip(amplitudes).enumerate() {
            let mut osc = QuantumOscillator::new(id(idx as u64), [0.0; 5]);
            osc.phases[0] = phase;
            osc.amplitudes = [amplitude; 5];
            network.add_oscillator(osc).expect("insert oscillator");
        }
        for idx in 0..phases.len().saturating_sub(1) {
            let src = id(idx as u64);
            let dst = id((idx + 1) as u64);
            network.set_coupling(src, dst, 1.0);
            network.set_coupling(dst, src, 1.0);
        }
        network
    }

    #[test]
    fn uniform_phase_high_amplitude_low_delta_g_maps_to_certainty() {
        let network = network_with_phases(&[0.1, 0.1, 0.1], &[1.0, 1.0, 1.0]);
        let mut engine = PhaseSemanticsEngine::new();
        let delta = [(id(0), 0.01), (id(1), 0.02), (id(2), 0.03)];
        engine.update_from_network(&network, &delta);

        assert_eq!(
            engine.node_semantic_state(id(0)).marker,
            SemanticMarker::Certainty
        );
        assert_eq!(
            engine.network_semantic_state().dominant_state,
            SemanticMarker::Certainty
        );
    }

    #[test]
    fn phase_dispersion_with_high_delta_g_maps_to_exploration() {
        let network = network_with_phases(&[1.8, 2.4, 5.3], &[0.9, 0.9, 0.9]);
        let mut engine = PhaseSemanticsEngine::new();
        let delta = [(id(0), 1.2), (id(1), 1.1), (id(2), 1.3)];
        engine.update_from_network(&network, &delta);

        assert_eq!(
            engine.network_semantic_state().dominant_state,
            SemanticMarker::Exploration
        );
    }

    #[test]
    fn different_phase_distribution_same_sync_can_change_semantics() {
        let mut engine = PhaseSemanticsEngine::new();
        let network_a = network_with_phases(&[0.1, 0.1, 0.1], &[0.9, 0.9, 0.9]);
        let network_b = network_with_phases(&[3.2, 3.2, 3.2], &[0.9, 0.9, 0.9]);
        let delta = [(id(0), 0.2), (id(1), 0.2), (id(2), 0.2)];

        engine.update_from_network(&network_a, &delta);
        let state_a = engine.network_semantic_state().dominant_state;

        engine.update_from_network(&network_b, &delta);
        let state_b = engine.network_semantic_state().dominant_state;

        assert_ne!(state_a, state_b);
    }

    #[test]
    fn neighbor_semantic_mismatch_creates_high_tension_edges() {
        let network = network_with_phases(&[0.1, 3.5], &[1.0, 0.1]);
        let mut engine = PhaseSemanticsEngine::new();
        let delta = [(id(0), 0.05), (id(1), 1.2)];
        engine.update_from_network(&network, &delta);

        let edges = engine.semantic_tension_edges();
        assert!(!edges.is_empty());
        assert!(edges[0].divergence >= TENSION_DIVERGENCE_MIN);
    }
}
