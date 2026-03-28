#![allow(clippy::cast_precision_loss)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use genesis_dynamics::VFEMinimizer;
use genesis_math::SparseCliffordVector;
use genesis_types::NodeId;

fn build_vfe(n: usize) -> (VFEMinimizer, Vec<NodeId>) {
    let mut vfe = VFEMinimizer::new();
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = NodeId::try_new(i as u64).expect("valid NodeId");
        vfe.add_node(id, [i as f64 * 0.0001, 0.0, 0.0, 0.0]);
        ids.push(id);
    }
    (vfe, ids)
}

fn bench_compute_vfe_with_grad(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_vfe_with_grad");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let (vfe, ids) = build_vfe(n);
        let obs = SparseCliffordVector::from_iter([(0, 0.5), (1, -0.2), (2, 0.3), (3, 0.7)])
            .expect("finite observation");

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_size| {
            b.iter(|| {
                let mut total = 0.0f64;
                for &id in ids.iter().take(1024) {
                    let (value, grad) =
                        vfe.compute_vfe_with_grad(black_box(id), Some(black_box(&obs)));
                    total += value + grad[0];
                }
                black_box(total)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compute_vfe_with_grad);
criterion_main!(benches);
