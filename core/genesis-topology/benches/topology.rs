#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::semicolon_if_nothing_returned,
    clippy::uninlined_format_args
)]

use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use genesis_math::SparseCliffordVector;
use genesis_topology::{
    benchmark_rank_by_gaussian_elimination, benchmark_xor_row_elimination, geometric_distance,
    CliffordHashTable, CohomologyValidator, HnswGraph, ManifoldCollector, RipsComplex,
};
use genesis_types::NodeId;

fn make_vec(id: u64) -> SparseCliffordVector {
    let s = (id as f64).mul_add(0.01, 0.05);
    SparseCliffordVector::from_iter((0..4).map(|b| (b, s * (b as f64 + 1.0)))).unwrap()
}

fn bench_rank_by_gaussian_elimination_throughput(c: &mut Criterion) {
    c.bench_function("rank_by_gaussian_elimination_throughput", |b| {
        b.iter(|| {
            black_box(benchmark_rank_by_gaussian_elimination(
                128,
                256,
                0xA5A5_1234_D3C1_9E37,
            ))
        })
    });
}

fn bench_h1_query_loop_latency(c: &mut Criterion) {
    let mut g = HnswGraph::new(16);
    for i in 0..3_000u64 {
        let v = make_vec(i);
        g.insert(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            &v,
        )
        .unwrap();
    }
    let complex = RipsComplex::build(&g, 0.5);

    c.bench_function("h1_query_loop_latency", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(CohomologyValidator::check_h1(black_box(&complex)));
            }
            start.elapsed()
        })
    });
}

fn bench_hnsw_layer0_neighbor_scan(c: &mut Criterion) {
    let mut g = HnswGraph::new(16);
    for i in 0..10_000u64 {
        let v = make_vec(i);
        g.insert(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            &v,
        )
        .unwrap();
    }
    let soa = g.layer0_soa();

    c.bench_function("hnsw_layer0_neighbor_scan", |b| {
        b.iter(|| {
            let mut sum = 0.0_f64;
            for &(start, end) in &soa.neighbor_offsets {
                for idx in start..end {
                    sum += soa.neighbor_distances[idx];
                }
            }
            black_box(sum)
        })
    });
}

fn bench_xor_row_elimination_throughput(c: &mut Criterion) {
    c.bench_function("xor_row_elimination_throughput", |b| {
        b.iter(|| {
            black_box(benchmark_xor_row_elimination(
                256,
                1024,
                0xDEAD_BEEF_CAFE_BABE,
            ))
        })
    });
}

fn bench_hnsw_insert_1000(c: &mut Criterion) {
    c.bench_function("hnsw_insert_1000", |b| {
        b.iter(|| {
            let mut g = HnswGraph::new(16);
            for i in 0..1000u64 {
                let v = make_vec(i);
                g.insert(
                    NodeId::try_new(i).expect("NodeId válido por construcción"),
                    black_box(&v),
                )
                .unwrap();
            }
            black_box(g.node_count())
        })
    });
}

fn bench_hnsw_search_k10_in_1000(c: &mut Criterion) {
    let mut g = HnswGraph::new(16);
    for i in 0..1000u64 {
        let v = make_vec(i);
        g.insert(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            &v,
        )
        .unwrap();
    }
    let query = make_vec(500);
    c.bench_function("hnsw_search_k10_in_1000", |b| {
        b.iter(|| black_box(g.search_nearest(black_box(&query), 10)))
    });
}

fn bench_geometric_distance_pair(c: &mut Criterion) {
    let a = make_vec(1);
    let b = make_vec(2);
    c.bench_function("geometric_distance_pair", |b_| {
        b_.iter(|| black_box(geometric_distance(black_box(&a), black_box(&b))))
    });
}

fn bench_rips_build_10k(c: &mut Criterion) {
    let mut g = HnswGraph::new(16);
    for i in 0..10_000u64 {
        let v = make_vec(i);
        g.insert(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            &v,
        )
        .unwrap();
    }

    c.bench_function("rips_build_10k", |b| {
        b.iter(|| {
            let complex = RipsComplex::build(black_box(&g), black_box(0.5));
            black_box(complex.counts())
        })
    });
}

fn bench_h1_check_10k(c: &mut Criterion) {
    let mut g = HnswGraph::new(16);
    for i in 0..10_000u64 {
        let v = make_vec(i);
        g.insert(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            &v,
        )
        .unwrap();
    }
    let complex = RipsComplex::build(&g, 0.5);

    let cold_start_begin = Instant::now();
    let cold_start_result = CohomologyValidator::check_h1(&complex);
    let cold_start_elapsed = cold_start_begin.elapsed();
    eprintln!(
        "[cohomology h1 cold-start] result={cold_start_result}, elapsed_ns={}",
        cold_start_elapsed.as_nanos()
    );

    c.bench_function("cohomology_h1_check_10k", |b| {
        b.iter(|| black_box(CohomologyValidator::check_h1(black_box(&complex))))
    });
}
fn percentile(sorted: &[u128], p: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn bench_manifold_lambda2_percentiles(c: &mut Criterion) {
    let mut manifold = ManifoldCollector::new(16);
    for i in 0..5_000u64 {
        let v = make_vec(i);
        manifold
            .insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &v,
            )
            .unwrap();
    }

    c.bench_function("manifold_compute_lambda2_p50_p95_p99", |b| {
        b.iter_custom(|iters| {
            let mut samples_ns = Vec::with_capacity(iters as usize);
            let mut total = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();
                let lambda2 = manifold.compute_lambda2();
                let elapsed = start.elapsed();
                black_box(lambda2);
                samples_ns.push(elapsed.as_nanos());
                total += elapsed;
            }

            samples_ns.sort_unstable();
            let p50 = percentile(&samples_ns, 0.50);
            let p95 = percentile(&samples_ns, 0.95);
            let p99 = percentile(&samples_ns, 0.99);
            eprintln!(
                "[manifold λ2] latency ns => p50={p50}, p95={p95}, p99={p99}, spread={}",
                p99.saturating_sub(p50)
            );

            total
        })
    });
}

fn bench_compute_lambda2_n1000(c: &mut Criterion) {
    let mut manifold = ManifoldCollector::new(16);
    for i in 0..1_000u64 {
        let v = make_vec(i);
        manifold
            .insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &v,
            )
            .unwrap();
    }

    c.bench_function("compute_lambda2_n1000", |b| {
        b.iter(|| black_box(manifold.compute_lambda2()))
    });
}

fn bench_lsh_candidates_adversarial_percentiles(c: &mut Criterion) {
    let mut table = CliffordHashTable::new();
    let adversarial_vec = make_vec(42);

    // Carga adversarial: todos los nodos en exactamente los mismos buckets en todas las tablas.
    for i in 0..20_000u64 {
        table.insert(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            &adversarial_vec,
        );
    }

    c.bench_function("lsh_candidates_adversarial_p50_p95_p99", |b| {
        b.iter_custom(|iters| {
            let mut samples_ns = Vec::with_capacity(iters as usize);
            let mut total = Duration::ZERO;

            for _ in 0..iters {
                let start = Instant::now();
                let count = table.candidates(&adversarial_vec).count();
                let elapsed = start.elapsed();
                black_box(count);
                samples_ns.push(elapsed.as_nanos());
                total += elapsed;
            }

            samples_ns.sort_unstable();
            let p50 = percentile(&samples_ns, 0.50);
            let p95 = percentile(&samples_ns, 0.95);
            let p99 = percentile(&samples_ns, 0.99);
            eprintln!("[lsh adversarial] candidates latency ns => p50={p50}, p95={p95}, p99={p99}");

            total
        })
    });
}

criterion_group!(
    benches,
    bench_rank_by_gaussian_elimination_throughput,
    bench_h1_query_loop_latency,
    bench_hnsw_layer0_neighbor_scan,
    bench_xor_row_elimination_throughput,
    bench_hnsw_insert_1000,
    bench_hnsw_search_k10_in_1000,
    bench_geometric_distance_pair,
    bench_lsh_candidates_adversarial_percentiles,
    bench_compute_lambda2_n1000,
    bench_manifold_lambda2_percentiles,
    bench_rips_build_10k,
    bench_h1_check_10k,
);
criterion_main!(benches);
