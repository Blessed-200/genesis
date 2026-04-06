#![allow(dead_code, clippy::cast_precision_loss)]

use genesis_math::SparseCliffordVector;
use genesis_topology::HnswGraph;
use genesis_types::NodeId;

pub(crate) fn make_vec(seed: u64) -> SparseCliffordVector {
    let s = (seed as f64).mul_add(0.01, 0.05);
    SparseCliffordVector::from_iter((0..4).map(|b| (b, s * (b as f64 + 1.0))))
        .expect("finite vector")
}

pub(crate) fn build_test_graph(size: usize) -> HnswGraph {
    let mut graph = HnswGraph::new(16);
    for i in 0..size {
        let v = make_vec(i as u64);
        graph
            .insert(NodeId::try_new(i as u64).expect("valid NodeId"), &v)
            .expect("insert must succeed");
    }
    graph
}
