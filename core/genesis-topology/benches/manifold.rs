#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]

use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use genesis_math::SparseCliffordVector;
use genesis_topology::{
    hnsw::{benchmark_batch_distance_4, benchmark_scalar_distance_4x},
    HnswGraph, ManifoldCollector,
};
use genesis_types::NodeId;

const SIGNED_U53_SCALE: f64 = 2.0 / (1_u64 << 53) as f64;

const COMPUTE_LAMBDA2_N1000_AVG_GATE_NS: u128 = 14_000_000;
const COMPUTE_LAMBDA2_N1000_P99_GATE_NS: u128 = 16_000_000;

fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

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

fn bench_compute_lambda2_n1000(c: &mut Criterion) {
    let mut manifold = ManifoldCollector::new(4);
    for i in 0..1_000_u64 {
        let mut seed = i ^ 0x9E37_79B9_7F4A_7C15;
        let v = make_vec(&mut seed);
        manifold
            .insert(NodeId::try_new(i).expect("valid NodeId"), &v)
            .expect("insert must succeed");
    }

    c.bench_function("compute_lambda2_n1000", |b| {
        b.iter_custom(|iters| {
            let iterations = usize::try_from(iters).unwrap_or(usize::MAX);
            let mut samples_ns = Vec::with_capacity(iterations);
            let mut total = Duration::ZERO;

            for _ in 0..iterations {
                let start = Instant::now();
                let lambda2 = manifold.compute_lambda2();
                let elapsed = start.elapsed();
                black_box(lambda2);
                samples_ns.push(elapsed.as_nanos());
                total += elapsed;
            }

            samples_ns.sort_unstable();
            let p99_ns = percentile(&samples_ns, 0.99);
            let avg_ns = total.as_nanos() / u128::from(iters.max(1));
            assert!(
                avg_ns <= COMPUTE_LAMBDA2_N1000_AVG_GATE_NS,
                "compute_lambda2_n1000 average latency {avg_ns} ns exceeds 14 ms gate"
            );
            assert!(
                p99_ns <= COMPUTE_LAMBDA2_N1000_P99_GATE_NS,
                "compute_lambda2_n1000 p99 latency {p99_ns} ns exceeds 16 ms gate"
            );
            eprintln!("[compute_lambda2_n1000] average={avg_ns} ns, p99={p99_ns} ns");

            total
        })
    });
}

criterion_group!(benches, bench_hnsw_search, bench_compute_lambda2_n1000);
criterion_main!(benches);
