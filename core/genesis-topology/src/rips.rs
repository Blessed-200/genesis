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

/// Borrowed CSR adjacency view.
///
/// AX-ID: AXIOMA-007
pub struct CsrAdjacency<'a> {
    /// Flat neighbour index payload in CSR row-major order.
    pub data: &'a [usize],
    /// Row offsets where node `i` spans `offsets[i]..offsets[i+1]` inside `data`.
    pub offsets: &'a [usize],
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
    adjacency_data: Vec<usize>,
    adjacency_offsets: Vec<usize>,
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
                adjacency_data: Vec::new(),
                adjacency_offsets: vec![0],
            };
        }

        let mut id_to_idx: Vec<(u64, usize)> = node_ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.get(), idx))
            .collect();
        id_to_idx.sort_unstable_by_key(|&(id, _)| id);

        let observed_degree_sum: usize =
            node_ids.iter().map(|&id| graph.neighbors(id).count()).sum();
        let observed_avg_degree = observed_degree_sum as f64 / node_count as f64;
        let epsilon_scale = (epsilon / (1.0 + epsilon.abs())).clamp(0.05, 1.0);
        let expected_avg_degree = (observed_avg_degree * epsilon_scale).ceil().max(1.0) as usize;
        let expected_edge_count = (node_count * expected_avg_degree) / 2;

        let mut edges: Vec<Edge> = Vec::with_capacity(expected_edge_count);
        let mut accepted_edges: Vec<(usize, usize)> = Vec::with_capacity(expected_edge_count);

        for (u_idx, &u) in node_ids.iter().enumerate() {
            if let Some(u_vec) = graph.vector(u) {
                for v in graph.neighbors(u).filter(|&v| u.get() < v.get()) {
                    let Ok(pos) = id_to_idx.binary_search_by_key(&v.get(), |&(id, _)| id) else {
                        continue;
                    };
                    let v_idx = id_to_idx[pos].1;
                    if let Some(v_vec) = graph.vector(v) {
                        let d = geometric_distance(u_vec, v_vec);
                        if d <= epsilon {
                            edges.push(Edge { u, v });
                            accepted_edges.push((u_idx, v_idx));
                        }
                    }
                }
            }
        }
        let mut adjacency_offsets = vec![0usize; node_count + 1];
        for &(u_idx, v_idx) in &accepted_edges {
            adjacency_offsets[u_idx + 1] += 1;
            adjacency_offsets[v_idx + 1] += 1;
        }
        for i in 1..=node_count {
            adjacency_offsets[i] += adjacency_offsets[i - 1];
        }
        let mut adjacency_data = vec![0usize; adjacency_offsets[node_count]];
        let mut write_heads = adjacency_offsets[..node_count].to_vec();
        for &(u_idx, v_idx) in &accepted_edges {
            let u_write = write_heads[u_idx];
            adjacency_data[u_write] = v_idx;
            write_heads[u_idx] += 1;
            let v_write = write_heads[v_idx];
            adjacency_data[v_write] = u_idx;
            write_heads[v_idx] += 1;
        }
        for u_idx in 0..node_count {
            let start = adjacency_offsets[u_idx];
            let end = adjacency_offsets[u_idx + 1];
            adjacency_data[start..end].sort_unstable();
        }

        edges.sort_unstable_by_key(|e| (e.u.get(), e.v.get()));
        edges.dedup();
        let dim1 = edges.iter().map(|e| [e.u, e.v]).collect();

        // Triangle generation using two-pointer intersection over sorted
        // neighbour lists. This avoids per-edge bitset clear/fill churn and
        // keeps memory accesses linear and branch-stable.
        let mut triangles: Vec<Triangle> = Vec::new();

        for u_idx in 0..node_count {
            let u_start = adjacency_offsets[u_idx];
            let u_end = adjacency_offsets[u_idx + 1];
            let u_nb = &adjacency_data[u_start..u_end];
            for &v_idx in u_nb.iter().filter(|&&v_idx| v_idx > u_idx) {
                let v_start = adjacency_offsets[v_idx];
                let v_end = adjacency_offsets[v_idx + 1];
                let v_nb = &adjacency_data[v_start..v_end];
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

        Self {
            dim0,
            dim1,
            dim2,
            adjacency_data,
            adjacency_offsets,
        }
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
    ///
    /// ```
    /// use genesis_math::SparseCliffordVector;
    /// use genesis_topology::hnsw::HnswGraph;
    /// use genesis_topology::rips::RipsComplex;
    /// use genesis_types::NodeId;
    ///
    /// let mut graph = HnswGraph::new(16);
    /// for i in 0..3_u64 {
    ///     let v = SparseCliffordVector::from_iter((0..4).map(|b| (b, (i + b as u64) as f64)))
    ///         .expect("finite vector");
    ///     graph.insert(NodeId::try_new(i).expect("valid id"), &v).expect("insert");
    /// }
    /// let rips = RipsComplex::build(&graph, 10.0);
    /// let (_n0, _n1, _n2) = rips.counts();
    /// ```
    ///
    /// AX-ID: AXIOMA-007
    pub const fn counts(&self) -> (usize, usize, usize) {
        (self.dim0.len(), self.dim1.len(), self.dim2.len())
    }

    /// Returns the CSR adjacency backing used for edge/triangle traversal.
    ///
    /// Contract: `data` is the flattened adjacency payload and `offsets` is the
    /// CSR row-offset table where node `i` maps to `data[offsets[i]..offsets[i+1]]`.
    ///
    /// ```
    /// use genesis_math::SparseCliffordVector;
    /// use genesis_topology::hnsw::HnswGraph;
    /// use genesis_topology::rips::RipsComplex;
    /// use genesis_types::NodeId;
    ///
    /// let mut graph = HnswGraph::new(16);
    /// for i in 0..4_u64 {
    ///     let v = SparseCliffordVector::from_iter((0..4).map(|b| (b, (i + b as u64) as f64)))
    ///         .expect("finite vector");
    ///     graph.insert(NodeId::try_new(i).expect("valid id"), &v).expect("insert");
    /// }
    /// let rips = RipsComplex::build(&graph, 10.0);
    /// let csr = rips.adjacency_csr();
    /// assert_eq!(csr.offsets.len(), rips.counts().0 + 1);
    /// assert!(csr.offsets.windows(2).all(|w| w[0] <= w[1]));
    /// assert_eq!(*csr.offsets.last().unwrap(), csr.data.len());
    /// ```
    ///
    /// AX-ID: AXIOMA-007
    pub fn adjacency_csr(&self) -> CsrAdjacency<'_> {
        CsrAdjacency {
            data: &self.adjacency_data,
            offsets: &self.adjacency_offsets,
        }
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

    #[test]
    fn adjacency_csr_returns_data_and_offsets_in_order() {
        let mut g = HnswGraph::new(16);
        for i in 0..4u64 {
            g.insert(
                NodeId::try_new(i).expect("valid NodeId by construction"),
                &make_vec(i),
            )
            .unwrap();
        }
        let rips = RipsComplex::build(&g, 1.0);
        let csr = rips.adjacency_csr();
        assert_eq!(csr.data, rips.adjacency_data.as_slice());
        assert_eq!(csr.offsets, rips.adjacency_offsets.as_slice());
    }
}
