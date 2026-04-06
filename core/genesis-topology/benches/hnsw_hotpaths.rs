#![allow(clippy::cast_precision_loss)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
mod bench_utils;

use bench_utils::{build_test_graph, make_vec};
use genesis_topology::benchmark_incremental_d2_xor_columns;
use genesis_types::NodeId;

fn bench_enforce_density_limit(c: &mut Criterion) {
    let mut group = c.benchmark_group("enforce_density_limit");
    for &size in &[1_000usize, 10_000, 100_000] {
        let mut graph = build_test_graph(size);
        let existing_id = NodeId::try_new((size / 2) as u64).expect("valid NodeId");
        let probe = make_vec(size as u64 + 1);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_n| {
            b.iter(|| {
                graph
                    .insert(existing_id, black_box(&probe))
                    .expect("insert must succeed");
                black_box(graph.node_count())
            });
        });
    }
    group.finish();
}

fn bench_incremental_d2_xor_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_d2_xor_columns");
    let num_edges = 65_536usize;
    let iterations = 256usize;

    for &(label, density_permille) in &[
        ("density_1pct", 10u16),
        ("density_10pct", 100u16),
        ("density_50pct", 500u16),
    ] {
        group.bench_with_input(
            BenchmarkId::new(label, num_edges),
            &density_permille,
            |b, &density| {
                b.iter(|| {
                    black_box(benchmark_incremental_d2_xor_columns(
                        black_box(num_edges),
                        black_box(density),
                        black_box(iterations),
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_enforce_density_limit,
    bench_incremental_d2_xor_columns
);
criterion_main!(benches);
