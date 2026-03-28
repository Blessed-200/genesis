#![allow(clippy::cast_precision_loss)]

use std::sync::OnceLock;

use genesis_math::{fast_metric_distance_sq, SparseCliffordVector};
use genesis_topology::{HnswGraph, RipsComplex};
use genesis_types::NodeId;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};

fn make_vec(seed: u64) -> SparseCliffordVector {
    let s = seed as f64 * 0.01 + 0.05;
    SparseCliffordVector::from_iter((0..4).map(|b| (b, s * (b as f64 + 1.0))))
        .expect("finite vector")
}

fn graph_fixture(nodes: usize) -> HnswGraph {
    let mut graph = HnswGraph::new(16);
    for i in 0..nodes {
        let v = make_vec(i as u64);
        graph
            .insert(NodeId::try_new(i as u64).expect("valid NodeId"), &v)
            .expect("insert must succeed");
    }
    graph
}

fn hnsw_fixture() -> &'static (HnswGraph, SparseCliffordVector) {
    static FIXTURE: OnceLock<(HnswGraph, SparseCliffordVector)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let graph = graph_fixture(8_192);
        let query = make_vec(4_096);
        (graph, query)
    })
}

fn distance_fixture() -> &'static (SparseCliffordVector, Vec<SparseCliffordVector>) {
    static FIXTURE: OnceLock<(SparseCliffordVector, Vec<SparseCliffordVector>)> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let query = make_vec(777);
        let candidates = (0..4_096).map(make_vec).collect();
        (query, candidates)
    })
}

fn rips_fixture() -> &'static HnswGraph {
    static FIXTURE: OnceLock<HnswGraph> = OnceLock::new();
    FIXTURE.get_or_init(|| graph_fixture(4_096))
}

#[library_benchmark]
fn hnsw_search_hotpath() -> usize {
    let (graph, query) = hnsw_fixture();
    graph.search_nearest(query, 10).len()
}

#[library_benchmark]
fn metric_distance_sq_hotpath() -> f64 {
    let (query, candidates) = distance_fixture();
    let mut total = 0.0;
    for candidate in candidates.iter().take(1024) {
        total += fast_metric_distance_sq(query, candidate);
    }
    total
}

#[library_benchmark]
fn rips_build_hotpath() -> usize {
    let graph = rips_fixture();
    RipsComplex::build(graph, 0.5).counts().0
}

library_benchmark_group!(
    name = hotpaths;
    benchmarks = hnsw_search_hotpath, metric_distance_sq_hotpath, rips_build_hotpath
);
main!(library_benchmark_groups = hotpaths);
