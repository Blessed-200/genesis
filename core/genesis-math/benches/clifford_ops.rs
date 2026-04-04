//! Criterion throughput benchmark for dense G(1,3) geometric products.
//!
//! AX-ID: AXIOMA-001, AXIOMA-011

#![allow(clippy::cast_precision_loss)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use genesis_math::{sparse_geometric_product, SparseCliffordVector};

fn deterministic_dense_inputs() -> Vec<SparseCliffordVector> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut out = Vec::with_capacity(8192);

    for _ in 0..8192 {
        let mut dense = [0.0f64; 16];
        for coeff in &mut dense {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            *coeff = unit.mul_add(4.0, -2.0);
        }
        out.push(SparseCliffordVector::from_dense(&dense).expect("deterministic dense input"));
    }

    out
}

fn bench_geometric_product_throughput(c: &mut Criterion) {
    let inputs = deterministic_dense_inputs();
    let mut group = c.benchmark_group("clifford_ops");
    group.throughput(Throughput::Elements(inputs.len() as u64));

    group.bench_function("geometric_product_throughput_ops", |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f64;
            for idx in 0..inputs.len() {
                let a = &inputs[idx];
                let b = &inputs[(idx + 1) & (inputs.len() - 1)];
                if let Some(product) = sparse_geometric_product(black_box(a), black_box(b)) {
                    acc += product.max_abs_coeff;
                }
            }
            black_box(acc)
        });
    });

    group.finish();
}

criterion_group!(clifford_ops_benches, bench_geometric_product_throughput);
criterion_main!(clifford_ops_benches);
