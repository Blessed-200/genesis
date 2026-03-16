/// AX-ID: AXIOMA-007, AXIOMA-009
/// Vietoris-Rips complex up to dimension 2.
/// Used by `CohomologyValidator` to compute H¹.
/// No external libraries. No persistent homology.
use genesis_types::NodeId;

use crate::geodesic::geometric_distance;
use crate::hnsw::HnswGraph;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Edge {
    u: NodeId,
    v: NodeId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Triangle {
    u: NodeId,
    v: NodeId,
    w: NodeId,
}

/// Vietoris-Rips complex built from an `HnswGraph`, capped at dimension 2.
///
/// Simplices:
/// - 0-simplices: all nodes
/// - 1-simplices: edges (u, v) where d(u, v) <= epsilon
/// - 2-simplices: triangles (u, v, w) where all three pairwise distances <= epsilon
///
/// AX-ID: AXIOMA-007, AXIOMA-009
pub struct RipsComplex {
    /// Compact simplex storage by fixed-size arrays instead of heap-allocated Vec per simplex.
    dim0: Vec<[NodeId; 1]>,
    dim1: Vec<[NodeId; 2]>,
    dim2: Vec<[NodeId; 3]>,
}

impl RipsComplex {
    /// Build the Rips complex from the HNSW graph.
    ///
    /// Only edges already present in the graph (at layer 0) are considered,
    /// which gives the graph-induced Rips complex. This is correct for our
    /// topological purposes because the HNSW graph already captures the
    /// neighbourhood structure at the given scale.
    ///
    /// AX-ID: AXIOMA-007
    #[allow(clippy::similar_names)]
    pub fn build(graph: &HnswGraph, epsilon: f64) -> Self {
        let node_ids: Vec<NodeId> = graph.nodes().collect();
        let node_count = node_ids.len();
        let dim0 = node_ids.iter().copied().map(|id| [id]).collect();

        if node_count == 0 {
            return Self {
                dim0,
                dim1: Vec::new(),
                dim2: Vec::new(),
            };
        }

        let max_id = node_ids
            .iter()
            .map(|id| id.get() as usize)
            .max()
            .unwrap_or(0);
        let mut id_to_idx = vec![usize::MAX; max_id + 1];
        for (idx, id) in node_ids.iter().copied().enumerate() {
            id_to_idx[id.get() as usize] = idx;
        }

        // Compact adjacency by internal node index.
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        let mut edges: Vec<Edge> = Vec::new();

        for (u_idx, &u) in node_ids.iter().enumerate() {
            if let Some(u_vec) = graph.get_vector(u) {
                for v in graph.neighbors(u).filter(|&v| u.get() < v.get()) {
                    let v_raw = v.get() as usize;
                    if v_raw >= id_to_idx.len() {
                        continue;
                    }
                    let v_idx = id_to_idx[v_raw];
                    if v_idx == usize::MAX {
                        continue;
                    }
                    if let Some(v_vec) = graph.get_vector(v) {
                        let d = geometric_distance(u_vec, v_vec);
                        if d <= epsilon {
                            edges.push(Edge { u, v });
                            adjacency[u_idx].push(v_idx);
                            adjacency[v_idx].push(u_idx);
                        }
                    }
                }
            }
        }

        for nb in &mut adjacency {
            nb.sort_unstable();
            nb.dedup();
        }

        edges.sort_unstable_by_key(|e| (e.u.get(), e.v.get()));
        edges.dedup();
        let dim1 = edges.iter().map(|e| [e.u, e.v]).collect();

        // Triangle generation using two-pointer intersection over sorted
        // neighbour lists. This avoids per-edge bitset clear/fill churn and
        // keeps memory accesses linear and branch-stable.
        let mut triangles: Vec<Triangle> = Vec::new();

        for (u_idx, u_nb) in adjacency.iter().enumerate() {
            for &v_idx in u_nb.iter().filter(|&&v_idx| v_idx > u_idx) {
                let v_nb = &adjacency[v_idx];
                let mut left_cursor = 0usize;
                let mut right_cursor = 0usize;
                while left_cursor < u_nb.len() && right_cursor < v_nb.len() {
                    let left_neighbor = u_nb[left_cursor];
                    let right_neighbor = v_nb[right_cursor];
                    if left_neighbor <= v_idx {
                        left_cursor += 1;
                        continue;
                    }
                    if right_neighbor <= v_idx {
                        right_cursor += 1;
                        continue;
                    }
                    match left_neighbor.cmp(&right_neighbor) {
                        std::cmp::Ordering::Equal => {
                            triangles.push(Triangle {
                                u: node_ids[u_idx],
                                v: node_ids[v_idx],
                                w: node_ids[left_neighbor],
                            });
                            left_cursor += 1;
                            right_cursor += 1;
                        }
                        std::cmp::Ordering::Less => left_cursor += 1,
                        std::cmp::Ordering::Greater => right_cursor += 1,
                    }
                }
            }
        }

        triangles.sort_unstable_by_key(|t| (t.u.get(), t.v.get(), t.w.get()));
        triangles.dedup();
        let dim2 = triangles.iter().map(|t| [t.u, t.v, t.w]).collect();

        Self { dim0, dim1, dim2 }
    }

    /// Iterate simplices of a given dimension (0, 1, or 2).
    ///
    /// AX-ID: AXIOMA-007
    pub fn simplices_of_dim(&self, d: usize) -> impl Iterator<Item = &[NodeId]> {
        let dim0 = self.dim0.iter().map(<[NodeId; 1]>::as_slice);
        let dim1 = self.dim1.iter().map(<[NodeId; 2]>::as_slice);
        let dim2 = self.dim2.iter().map(<[NodeId; 3]>::as_slice);
        match d {
            0 => EitherIter::Dim0(dim0),
            1 => EitherIter::Dim1(dim1),
            2 => EitherIter::Dim2(dim2),
            _ => EitherIter::Empty(std::iter::empty()),
        }
    }

    /// Number of simplices of each dimension.
    pub const fn counts(&self) -> (usize, usize, usize) {
        (self.dim0.len(), self.dim1.len(), self.dim2.len())
    }
}

enum EitherIter<I0, I1, I2, IE> {
    Dim0(I0),
    Dim1(I1),
    Dim2(I2),
    Empty(IE),
}

impl<'a, I0, I1, I2, IE> Iterator for EitherIter<I0, I1, I2, IE>
where
    I0: Iterator<Item = &'a [NodeId]>,
    I1: Iterator<Item = &'a [NodeId]>,
    I2: Iterator<Item = &'a [NodeId]>,
    IE: Iterator<Item = &'a [NodeId]>,
{
    type Item = &'a [NodeId];

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Dim0(it) => it.next(),
            Self::Dim1(it) => it.next(),
            Self::Dim2(it) => it.next(),
            Self::Empty(it) => it.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use genesis_math::SparseCliffordVector;
    use genesis_types::NodeId;

    use super::*;
    use crate::hnsw::HnswGraph;

    fn make_vec(id: u64) -> SparseCliffordVector {
        let s = (id as f64).mul_add(0.1, 0.05);
        SparseCliffordVector::from_iter((0..4).map(|b| (b, s * (b as f64 + 1.0)))).unwrap()
    }

    #[test]
    fn rips_builds_correctly() {
        let mut g = HnswGraph::new(16);
        for i in 0..5u64 {
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }
        let complex = RipsComplex::build(&g, 1.0);
        let (n0, n1, n2) = complex.counts();
        assert_eq!(n0, 5, "expected 5 vertices");
        // 1 and 2-simplices depend on connectivity — just verify no panic
        println!("RipsComplex: {} verts, {} edges, {} triangles", n0, n1, n2);
    }
}
