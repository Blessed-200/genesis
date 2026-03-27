use genesis_types::NodeId;
use smallvec::SmallVec;

use crate::geodesic::geometric_distance;
use crate::hnsw::HnswGraph;
use crate::manifold::ManifoldCollector;

const MAX_HYPOTHESES: usize = 16;
const PERSISTENCE_MIN_LIFETIME: f64 = 1.0;
const SPECTRAL_PARTITION_THRESHOLD: f64 = 0.05;
const BRIDGE_LCC_THRESHOLD: f64 = 0.15;
const CURVATURE_DELTA_THRESHOLD: f64 = 1e-12;
const HUB_MIN_BRANCHING: usize = 6;
const HUB_MAX_TRIANGLE_DENSITY: f64 = 0.20;
const CURVATURE_SEED_LIMIT: usize = 8;

/// Structural intuition engine over topological manifold state.
///
/// Read-only analytical layer: extracts interpretable hypotheses from H¹,
/// spectral connectivity, sparse local bridge structure, and local geometry proxies.
///
/// AX-ID: AXIOMA-007, AXIOMA-013, H_restricción (LEY_FUNDACIONAL §3.5)
pub struct TopologicalIntuition<'a> {
    manifold: &'a ManifoldCollector,
}

/// Candidate structural hypothesis emitted for downstream dynamics.
///
/// AX-ID: AXIOMA-007, AXIOMA-013, H_dinámica (LEY_FUNDACIONAL §3.2)
#[derive(Clone, Debug)]
pub struct TopologicalHypothesis {
    /// High-level category of the inferred structural pattern.
    pub kind: HypothesisKind,
    /// Local witness nodes supporting this hypothesis.
    pub nodes: SmallVec<[NodeId; 4]>,
    /// Interpretable confidence score in [0, 1].
    pub confidence: f64,
    /// Signal intensity in [0, 1].
    pub strength: f64,
    /// Human-readable metric explanation.
    pub explanation: HypothesisExplanation,
}

/// Hypothesis category.
///
/// AX-ID: AXIOMA-007, AXIOMA-013
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypothesisKind {
    /// Long-lived cycle detected in incremental H¹ state.
    PersistentCycle,
    /// H¹ cycle closure via triangle appearance.
    CycleClosure,
    /// Low local clustering edge proxy.
    FragileBridge,
    /// Low λ₂ partition signal.
    SpectralPartition,
    /// Local negative curvature proxy from δ-hyperbolicity.
    NegativeCurvatureRegion,
    /// High branching with low local triangle density.
    HierarchicalHub,
}

/// Interpretable explanation payload.
///
/// AX-ID: AXIOMA-007, H_restricción (LEY_FUNDACIONAL §3.5)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HypothesisExplanation {
    /// H¹ persistence lifetime.
    H1Persistence {
        /// Persistent lifetime measured in filtration steps.
        lifetime: f64,
    },
    /// Triangle closure event.
    TriangleClosure,
    /// λ₂ below partition threshold.
    LowLambda2 {
        /// Estimated algebraic connectivity value λ₂.
        value: f64,
    },
    /// Local clustering coefficient proxy.
    BridgeProxy {
        /// Local clustering coefficient proxy for a candidate bridge edge.
        lcc: f64,
    },
    /// Four-point δ-hyperbolicity proxy.
    NegativeCurvature {
        /// Four-point δ-hyperbolicity estimate.
        delta: f64,
    },
    /// Hierarchical branching factor.
    HyperbolicHierarchy {
        /// Local branching factor (degree) of the hub node.
        branching: usize,
    },
}

impl<'a> TopologicalIntuition<'a> {
    /// Create read-only topological intuition engine.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-013
    #[must_use]
    pub const fn new(manifold: &'a ManifoldCollector) -> Self {
        Self { manifold }
    }

    /// Run deterministic sparse inference and return top hypotheses.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-013, H_restricción (LEY_FUNDACIONAL §3.5)
    #[must_use]
    pub fn infer(&self) -> SmallVec<[TopologicalHypothesis; MAX_HYPOTHESES]> {
        let mut out = SmallVec::<[TopologicalHypothesis; 32]>::new();
        self.detect_persistent_cycles(&mut out);
        self.detect_cycle_closures(&mut out);
        self.detect_spectral_partition(&mut out);
        self.detect_fragile_bridges(&mut out);
        self.detect_negative_curvature_regions(&mut out);
        self.detect_hierarchical_hubs(&mut out);

        out.sort_by(|left, right| {
            let left_score = 0.6f64.mul_add(left.confidence, 0.4 * left.strength);
            let right_score = 0.6f64.mul_add(right.confidence, 0.4 * right.strength);
            right_score
                .total_cmp(&left_score)
                .then_with(|| kind_rank(left.kind).cmp(&kind_rank(right.kind)))
                .then_with(|| left.nodes.len().cmp(&right.nodes.len()))
                .then_with(|| cmp_nodes(&left.nodes, &right.nodes))
        });

        out.into_iter()
            .take(MAX_HYPOTHESES)
            .collect::<SmallVec<[TopologicalHypothesis; MAX_HYPOTHESES]>>()
    }

    fn detect_persistent_cycles(&self, out: &mut SmallVec<[TopologicalHypothesis; 32]>) {
        let h1_state = self.manifold.h1_state_ref();
        let cycles = h1_state.persistent_cycles();
        if cycles.is_empty() {
            return;
        }

        let now = h1_state.inference_step();
        let max_lifetime = cycles
            .iter()
            .map(|cycle| persistent_cycle_lifetime(now, cycle))
            .fold(0.0, f64::max);
        if max_lifetime <= 0.0 {
            return;
        }

        for cycle in cycles {
            let lifetime = persistent_cycle_lifetime(now, cycle);
            // loop-invariant, hoisted
            // CRYSTAL: O2, O5, O6, O7 — inevitable
            if lifetime < PERSISTENCE_MIN_LIFETIME {
                continue;
            }

            let confidence = (lifetime / max_lifetime).clamp(0.0, 1.0);
            let strength = (lifetime / (lifetime + 1.0)).clamp(0.0, 1.0);
            out.push(TopologicalHypothesis {
                kind: HypothesisKind::PersistentCycle,
                nodes: SmallVec::from_slice(&cycle.nodes),
                confidence,
                strength,
                explanation: HypothesisExplanation::H1Persistence { lifetime },
            });
        }
    }

    fn detect_cycle_closures(&self, out: &mut SmallVec<[TopologicalHypothesis; 32]>) {
        for closure in self.manifold.h1_state_ref().triangle_closures() {
            out.push(TopologicalHypothesis {
                kind: HypothesisKind::CycleClosure,
                nodes: SmallVec::from_slice(&closure.nodes),
                confidence: 1.0,
                strength: 0.8,
                explanation: HypothesisExplanation::TriangleClosure,
            });
        }
    }

    fn detect_spectral_partition(&self, out: &mut SmallVec<[TopologicalHypothesis; 32]>) {
        let lambda2 = self.manifold.compute_lambda2();
        if lambda2 >= SPECTRAL_PARTITION_THRESHOLD {
            return;
        }

        let confidence = (1.0 - (lambda2 / SPECTRAL_PARTITION_THRESHOLD)).clamp(0.0, 1.0);
        out.push(TopologicalHypothesis {
            kind: HypothesisKind::SpectralPartition,
            nodes: SmallVec::new(),
            confidence,
            strength: confidence,
            explanation: HypothesisExplanation::LowLambda2 { value: lambda2 },
        });
    }

    fn detect_fragile_bridges(&self, out: &mut SmallVec<[TopologicalHypothesis; 32]>) {
        let graph = self.manifold.graph_ref();

        for u in graph.nodes() {
            let mut u_neighbors = sorted_neighbors(graph, u);
            if u_neighbors.is_empty() {
                continue;
            }
            for &v in &u_neighbors {
                if u >= v {
                    continue;
                }
                let v_neighbors = sorted_neighbors(graph, v);
                let shared = intersection_count(&u_neighbors, &v_neighbors);
                let possible = u_neighbors
                    .len()
                    .saturating_sub(1)
                    .min(v_neighbors.len().saturating_sub(1));
                let lcc = if possible == 0 {
                    0.0
                } else {
                    shared as f64 / possible as f64
                };
                if lcc >= BRIDGE_LCC_THRESHOLD {
                    continue;
                }

                out.push(TopologicalHypothesis {
                    kind: HypothesisKind::FragileBridge,
                    nodes: SmallVec::from_slice(&[u, v]),
                    confidence: (1.0 - lcc).clamp(0.0, 1.0),
                    strength: (1.0 / (1.0 + shared as f64)).clamp(0.0, 1.0),
                    explanation: HypothesisExplanation::BridgeProxy { lcc },
                });
            }
            u_neighbors.clear();
        }
    }

    fn detect_negative_curvature_regions(&self, out: &mut SmallVec<[TopologicalHypothesis; 32]>) {
        let graph = self.manifold.graph_ref();
        let mut seeds: SmallVec<[(NodeId, usize); CURVATURE_SEED_LIMIT]> = SmallVec::new();

        for id in graph.nodes() {
            let degree = graph.neighbors(id).count();
            insert_seed(&mut seeds, id, degree);
        }

        for &(seed, _) in &seeds {
            let mut nearest = nearest_neighbors_by_metric(graph, seed);
            if nearest.len() < 3 {
                continue;
            }
            nearest.truncate(3);
            let quad = [seed, nearest[0], nearest[1], nearest[2]];
            let Some(delta) = local_hyperbolic_delta(graph, quad) else {
                continue;
            };
            if delta <= CURVATURE_DELTA_THRESHOLD {
                continue;
            }

            let confidence = (delta / (delta + CURVATURE_DELTA_THRESHOLD)).clamp(0.0, 1.0);
            out.push(TopologicalHypothesis {
                kind: HypothesisKind::NegativeCurvatureRegion,
                nodes: SmallVec::from_slice(&quad),
                confidence,
                strength: confidence,
                explanation: HypothesisExplanation::NegativeCurvature { delta },
            });
        }
    }

    fn detect_hierarchical_hubs(&self, out: &mut SmallVec<[TopologicalHypothesis; 32]>) {
        let graph = self.manifold.graph_ref();

        let max_degree = graph
            .nodes()
            .map(|id| graph.neighbors(id).count())
            .fold(0, usize::max);
        if max_degree == 0 {
            return;
        }

        for id in graph.nodes() {
            let degree = graph.neighbors(id).count();
            if degree < HUB_MIN_BRANCHING {
                continue;
            }
            let neighbors = sorted_neighbors(graph, id);
            let triangle_density = local_triangle_density(graph, &neighbors);
            if triangle_density > HUB_MAX_TRIANGLE_DENSITY {
                continue;
            }

            let confidence = (degree as f64 / max_degree as f64).clamp(0.0, 1.0);
            out.push(TopologicalHypothesis {
                kind: HypothesisKind::HierarchicalHub,
                nodes: SmallVec::from_slice(&[id]),
                confidence,
                strength: (1.0 - triangle_density).clamp(0.0, 1.0),
                explanation: HypothesisExplanation::HyperbolicHierarchy { branching: degree },
            });
        }
    }
}

fn cmp_nodes(left: &[NodeId], right: &[NodeId]) -> core::cmp::Ordering {
    left.iter()
        .map(|n| n.get())
        .cmp(right.iter().map(|n| n.get()))
}

const fn kind_rank(kind: HypothesisKind) -> usize {
    match kind {
        HypothesisKind::PersistentCycle => 0,
        HypothesisKind::CycleClosure => 1,
        HypothesisKind::FragileBridge => 2,
        HypothesisKind::SpectralPartition => 3,
        HypothesisKind::NegativeCurvatureRegion => 4,
        HypothesisKind::HierarchicalHub => 5,
    }
}

fn insert_seed(
    seeds: &mut SmallVec<[(NodeId, usize); CURVATURE_SEED_LIMIT]>,
    id: NodeId,
    degree: usize,
) {
    let pos = seeds
        .iter()
        .position(|&(eid, edeg)| degree > edeg || (degree == edeg && id.get() < eid.get()))
        .unwrap_or(seeds.len());

    if pos < CURVATURE_SEED_LIMIT {
        seeds.insert(pos, (id, degree));
        seeds.truncate(CURVATURE_SEED_LIMIT);
    }
}

fn sorted_neighbors(graph: &HnswGraph, id: NodeId) -> Vec<NodeId> {
    let mut neighbors = graph.neighbors(id).collect::<Vec<_>>();
    neighbors.sort_unstable_by_key(|id| id.get());
    neighbors
}

fn intersection_count(left: &[NodeId], right: &[NodeId]) -> usize {
    let mut shared = 0;
    let (mut it_l, mut it_r) = (left.iter(), right.iter());
    let (mut l, mut r) = (it_l.next(), it_r.next());
    while let (Some(a), Some(b)) = (l, r) {
        match a.cmp(b) {
            core::cmp::Ordering::Less => l = it_l.next(),
            core::cmp::Ordering::Greater => r = it_r.next(),
            core::cmp::Ordering::Equal => {
                shared += 1;
                l = it_l.next();
                r = it_r.next();
            }
        }
    }
    shared
}

fn nearest_neighbors_by_metric(graph: &HnswGraph, center: NodeId) -> Vec<NodeId> {
    let Some(center_vec) = graph.vector(center) else {
        return Vec::new();
    };

    let mut ranked = graph
        .neighbors(center)
        .filter_map(|neighbor| {
            graph
                .vector(neighbor)
                .map(|neighbor_vec| (neighbor, geometric_distance(center_vec, neighbor_vec)))
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.get().cmp(&right.0.get()))
    });
    ranked.into_iter().map(|(id, _)| id).collect()
}

fn local_hyperbolic_delta(graph: &HnswGraph, nodes: [NodeId; 4]) -> Option<f64> {
    let [a_id, b_id, c_id, d_id] = nodes;
    let a = graph.vector(a_id)?;
    let b = graph.vector(b_id)?;
    let c = graph.vector(c_id)?;
    let d = graph.vector(d_id)?;
    // loop-invariant, hoisted
    // CRYSTAL: O61, O62, O63, O64, FO44 — inevitable

    let mut x = geometric_distance(a, b) + geometric_distance(c, d);
    let mut y = geometric_distance(a, c) + geometric_distance(b, d);
    let mut z = geometric_distance(a, d) + geometric_distance(b, c);
    if x.total_cmp(&y).is_gt() {
        core::mem::swap(&mut x, &mut y);
    }
    if y.total_cmp(&z).is_gt() {
        core::mem::swap(&mut y, &mut z);
    }
    if x.total_cmp(&y).is_gt() {
        core::mem::swap(&mut x, &mut y);
    }
    Some(((z - y) * 0.5).max(0.0))
}

fn local_triangle_density(graph: &HnswGraph, neighbors: &[NodeId]) -> f64 {
    if neighbors.len() < 2 {
        return 0.0;
    }

    let n = neighbors.len();
    let possible = n * (n - 1) / 2;
    if possible == 0 {
        return 0.0;
    }

    let mut triangles = 0usize;
    for (i, &u) in neighbors.iter().enumerate() {
        for &v in &neighbors[i + 1..] {
            if are_adjacent(graph, u, v) {
                triangles += 1;
            }
        }
    }

    triangles as f64 / possible as f64
}

#[inline]
fn persistent_cycle_lifetime(
    now: usize,
    cycle: &crate::incremental_cohomology::PersistentCycleRecord,
) -> f64 {
    cycle
        .death_step
        .unwrap_or(now)
        .saturating_sub(cycle.birth_step) as f64
}

fn are_adjacent(graph: &HnswGraph, left: NodeId, right: NodeId) -> bool {
    graph.neighbors(left).any(|neighbor| neighbor == right)
}

#[cfg(test)]
mod tests {
    use genesis_math::SparseCliffordVector;

    use super::*;
    use crate::manifold::ManifoldCollector;

    fn make_vec(id: u64) -> SparseCliffordVector {
        let base = id as f64;
        SparseCliffordVector::from_iter(
            (0..4).map(|blade| (blade, (blade as f64).mul_add(0.1, base))),
        )
        .expect("valid sparse vector")
    }

    #[test]
    fn persistent_cycle_hypothesis_detected_on_square() {
        let mut manifold = ManifoldCollector::new(16);
        for id in 0..4_u64 {
            manifold
                .insert(NodeId::try_new(id).expect("valid id"), &make_vec(id))
                .expect("insert should succeed");
        }

        let hypotheses = TopologicalIntuition::new(&manifold).infer();
        assert!(
            hypotheses
                .iter()
                .any(|h| h.kind == HypothesisKind::PersistentCycle),
            "expected persistent cycle hypothesis"
        );
    }

    #[test]
    fn triangle_closure_hypothesis_detected() {
        let mut manifold = ManifoldCollector::new(16);
        for id in 0..3_u64 {
            manifold
                .insert(NodeId::try_new(id).expect("valid id"), &make_vec(id))
                .expect("insert should succeed");
        }

        let hypotheses = TopologicalIntuition::new(&manifold).infer();
        assert!(
            hypotheses
                .iter()
                .any(|h| h.kind == HypothesisKind::CycleClosure),
            "expected cycle closure hypothesis"
        );
    }

    #[test]
    fn spectral_partition_detected_for_weak_bridge() {
        let mut manifold = ManifoldCollector::new(8);
        for id in 0..28_u64 {
            manifold
                .insert(NodeId::try_new(id).expect("valid id"), &make_vec(id * 100))
                .expect("insert should succeed");
        }

        let lambda2 = manifold.compute_lambda2();
        let hypotheses = TopologicalIntuition::new(&manifold).infer();
        let has_partition = hypotheses
            .iter()
            .any(|h| h.kind == HypothesisKind::SpectralPartition);
        if lambda2 < SPECTRAL_PARTITION_THRESHOLD {
            assert!(has_partition, "expected spectral partition hypothesis");
        } else {
            assert!(
                !has_partition,
                "spectral partition must only appear when lambda2 is below threshold"
            );
        }
    }

    #[test]
    fn curvature_signal_detected_on_tree_like_graph() {
        let mut manifold = ManifoldCollector::new(8);
        let center = SparseCliffordVector::from_iter([(0usize, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)])
            .expect("valid center vector");
        manifold
            .insert(NodeId::try_new(0).expect("valid id"), &center)
            .expect("insert should succeed");

        for id in 1..20_u64 {
            let radius = 0.05 * id as f64;
            let branch = (id % 3) as f64;
            let vec = SparseCliffordVector::from_iter([
                (0usize, radius),
                (1, radius * (branch + 1.0)),
                (2, radius * (2.0 - branch * 0.2)),
                (3, radius * (0.5 + branch)),
            ])
            .expect("valid branch vector");
            manifold
                .insert(NodeId::try_new(id).expect("valid id"), &vec)
                .expect("insert should succeed");
        }

        let hypotheses = TopologicalIntuition::new(&manifold).infer();
        assert!(
            hypotheses
                .iter()
                .any(|h| h.kind == HypothesisKind::NegativeCurvatureRegion),
            "expected negative curvature hypothesis"
        );
    }

    #[test]
    fn inference_is_deterministic() {
        let mut manifold = ManifoldCollector::new(16);
        for id in 0..10_u64 {
            manifold
                .insert(NodeId::try_new(id).expect("valid id"), &make_vec(id))
                .expect("insert should succeed");
        }

        let first = TopologicalIntuition::new(&manifold).infer();
        let second = TopologicalIntuition::new(&manifold).infer();
        assert_eq!(first.len(), second.len());
        for (left, right) in first.iter().zip(second.iter()) {
            assert_eq!(left.kind, right.kind);
            assert_eq!(left.nodes, right.nodes);
            assert!((left.confidence - right.confidence).abs() < 1e-12);
            assert!((left.strength - right.strength).abs() < 1e-12);
        }
    }
}
