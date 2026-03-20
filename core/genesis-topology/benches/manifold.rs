#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]

use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use genesis_math::SparseCliffordVector;
use genesis_topology::{
    hnsw::{benchmark_batch_distance_4, benchmark_scalar_distance_4x},
    HnswGraph,
};
use genesis_types::NodeId;

const SIGNED_U53_SCALE: f64 = 2.0 / (1_u64 << 53) as f64;

const fn next_u64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    *seed
}

fn make_vec(seed: &mut u64) -> SparseCliffordVector {
    let dense = core::array::from_fn(|_| {
        let bits = next_u64(seed) >> 11;
        (bits as f64).mul_add(SIGNED_U53_SCALE, -1.0)
    });
    SparseCliffordVector::from_dense(&dense).expect("finite deterministic vector")
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    let mut seed = 0xDEAD_BEEF_CAFE_BABE;
    let mut graph = HnswGraph::new(200);
    for i in 0..10_000_u64 {
        let v = make_vec(&mut seed);
        graph
            .insert(NodeId::try_new(i).expect("valid NodeId"), &v)
            .expect("insert must succeed");
    }

    let mut queries = Vec::with_capacity(100);
    for _ in 0..100 {
        queries.push(make_vec(&mut seed));
    }

    for query in queries.iter().take(10) {
        black_box(graph.search_nearest(query, 5));
    }

    group.bench_function(BenchmarkId::new("search_10k_nodes_k5", 100), |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for i in 0..usize::try_from(iters).unwrap_or(usize::MAX) {
                let q = &queries[i % queries.len()];
                black_box(graph.search_nearest(black_box(q), 5));
            }
            start.elapsed()
        });
    });

    let query = make_vec(&mut seed);
    let candidates = [
        make_vec(&mut seed),
        make_vec(&mut seed),
        make_vec(&mut seed),
        make_vec(&mut seed),
    ];

    group.bench_function("batch_distance_4_raw", |b| {
        b.iter(|| {
            for _ in 0..10_000 {
                black_box(benchmark_batch_distance_4(
                    black_box(&query),
                    black_box(&candidates),
                ));
            }
        });
    });

    group.bench_function("scalar_distance_4x_raw", |b| {
        b.iter(|| {
            for _ in 0..10_000 {
                black_box(benchmark_scalar_distance_4x(
                    black_box(&query),
                    black_box(&candidates),
                ));
            }
        });
    });

    let iterations = 50_000_u64;
    let mut sink = 0.0_f64;
    let start_batch = Instant::now();
    for _ in 0..iterations {
        let out = benchmark_batch_distance_4(black_box(&query), black_box(&candidates));
        sink += black_box(out[0]);
    }
    let batch_ns = start_batch.elapsed().as_nanos() as f64 / iterations as f64;

    let start_scalar = Instant::now();
    for _ in 0..iterations {
        let out = benchmark_scalar_distance_4x(black_box(&query), black_box(&candidates));
        sink += black_box(out[0]);
    }
    let scalar_ns = start_scalar.elapsed().as_nanos() as f64 / iterations as f64;
    black_box(sink);

    eprintln!("batch_distance_4_raw: {batch_ns:.2} ns per call");
    eprintln!("scalar_distance_4x_raw: {scalar_ns:.2} ns per call");
    eprintln!(
        "batch_distance_4 / scalar_4x = {:.2}x",
        scalar_ns / batch_ns
    );

    group.finish();
}

criterion_group!(benches, bench_hnsw_search);
criterion_main!(benches);
