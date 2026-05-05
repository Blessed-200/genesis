#![allow(clippy::cast_precision_loss)]

use genesis_math::SparseCliffordVector;
use genesis_topology::HnswGraph;
use genesis_types::NodeId;

/// Builds a deterministic sparse G(1,3) vector fixture from a scalar seed.
///
/// # Panics
/// Panics if `SparseCliffordVector::from_iter` returns an error. This helper
/// always feeds finite coefficients derived from `seed`, so panic indicates an
/// unexpected constructor contract regression.
#[must_use]
pub fn make_vec(seed: u64) -> SparseCliffordVector {
    let s = (seed as f64).mul_add(0.01, 0.05);
    SparseCliffordVector::from_iter((0..4).map(|blade| (blade, s * (blade as f64 + 1.0))))
        .expect("finite vector")
}

/// Builds an HNSW graph benchmark fixture with deterministic node vectors.
///
/// # Panics
/// Panics if `NodeId::try_new(i as u64)` rejects any generated id or if
/// `graph.insert` returns an error for an inserted fixture vector.
#[allow(dead_code)]
#[must_use]
pub fn build_test_graph(size: usize) -> HnswGraph {
    let mut graph = HnswGraph::new(16);
    for i in 0..size {
        let vector = make_vec(i as u64);
        graph
            .insert(NodeId::try_new(i as u64).expect("valid NodeId"), &vector)
            .expect("insert must succeed");
    }
    graph
}
