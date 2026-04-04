//! Criterion throughput benchmarks for core G(1,3) Clifford operations.
//!
//! AX-ID: AXIOMA-001, AXIOMA-011

#![allow(clippy::cast_precision_loss)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use genesis_math::experimental::kernel_dense_g13::dense_geometric_product_g13;
use genesis_math::{
    compute_clifford_norm_sq, fast_metric_distance, sparse_geometric_product, SparseCliffordVector,
};

const DENSE_BATCH: usize = 1024;
const SPARSE_BATCH: usize = 4096;
const SINGLE_BATCH: usize = 8192;
const NORM_BATCH: usize = 4096;
const PROJECTION_BATCH: usize = 2048;
const DISTANCE_BATCH: usize = 2048;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn dense_vectors(count: usize) -> Vec<SparseCliffordVector> {
    let mut state = 0xA2D4_5C61_9312_F005u64;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut dense = [0.0f64; 16];
        for c in &mut dense {
            let bits = lcg_next(&mut state) >> 11;
            let u = (bits as f64) * (1.0 / ((1u64 << 53) as f64));
            *c = u.mul_add(4.0, -2.0);
        }
        out.push(SparseCliffordVector::from_dense(&dense).expect("dense fixture must be finite"));
    }
    out
}

fn sparse_vectors(count: usize) -> Vec<SparseCliffordVector> {
    let mut state = 0x7F12_0DE3_AA19_C0DEu64;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let active = ((lcg_next(&mut state) as usize) % 5) + 4;
        let mut dense = [0.0f64; 16];
        let mut used = 0u16;
        let mut filled = 0usize;
        while filled < active {
            let idx = (lcg_next(&mut state) as usize) & 0xF;
            let bit = 1u16 << idx;
            if (used & bit) == 0 {
                used |= bit;
                let bits = lcg_next(&mut state) >> 11;
                let u = (bits as f64) * (1.0 / ((1u64 << 53) as f64));
                dense[idx] = u.mul_add(2.0, -1.0);
                filled += 1;
            }
        }
        out.push(SparseCliffordVector::from_dense(&dense).expect("sparse fixture must be finite"));
    }
    out
}

fn single_blade_vectors(count: usize) -> Vec<SparseCliffordVector> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let blade = i & 0xF;
        let coeff = ((i as f64) + 1.0) * 0.001;
        out.push(SparseCliffordVector::from_iter([(blade, coeff)]).expect("single blade fixture"));
    }
    out
}

fn throughput_geometric_product(c: &mut Criterion) {
    let dense = dense_vectors(DENSE_BATCH + 1);
    let sparse = sparse_vectors(SPARSE_BATCH + 1);
    let single = single_blade_vectors(SINGLE_BATCH + 1);

    let mut group = c.benchmark_group("throughput_geometric_product_ratio_simd_vs_scalar");
    group.throughput(Throughput::Elements(DENSE_BATCH as u64));
    group.bench_function("geo_product_dense_throughput", |b| {
        b.iter(|| {
            let mut sink = 0.0f64;
            for i in 0..DENSE_BATCH {
                let p = sparse_geometric_product(black_box(&dense[i]), black_box(&dense[i + 1]))
                    .expect("dense product should produce output");
                sink += p.max_abs_coeff;
            }
            black_box(sink)
        });
    });

    group.throughput(Throughput::Elements(SPARSE_BATCH as u64));
    group.bench_function("geo_product_sparse_throughput", |b| {
        b.iter(|| {
            let mut sink = 0.0f64;
            for i in 0..SPARSE_BATCH {
                if let Some(p) =
                    sparse_geometric_product(black_box(&sparse[i]), black_box(&sparse[i + 1]))
                {
                    sink += p.max_abs_coeff;
                }
            }
            black_box(sink)
        });
    });

    group.throughput(Throughput::Elements(SINGLE_BATCH as u64));
    group.bench_function("geo_product_single_blade_throughput", |b| {
        b.iter(|| {
            let mut sink = 0.0f64;
            for i in 0..SINGLE_BATCH {
                let p = sparse_geometric_product(black_box(&single[i]), black_box(&single[i + 1]))
                    .expect("single blade product should produce output");
                sink += p.max_abs_coeff;
            }
            black_box(sink)
        });
    });

    group.bench_function(
        "geo_product_dense_baseline_dense_geometric_product_g13",
        |b| {
            b.iter(|| {
                let mut sink = 0.0f64;
                for i in 0..DENSE_BATCH {
                    let out = dense_geometric_product_g13(
                        black_box(&dense[i].coeffs),
                        black_box(&dense[i + 1].coeffs),
                    );
                    sink += out[0];
                }
                black_box(sink)
            });
        },
    );

    group.bench_function("geo_product_dense_baseline_naive_matmul16x16", |b| {
        b.iter(|| {
            let mut sink = 0.0f64;
            for i in 0..DENSE_BATCH {
                let a = &dense[i].coeffs;
                let m = &dense[i + 1].coeffs;
                let mut out = [0.0f64; 16];
                for r in 0..16 {
                    let ar = a[r];
                    for c in 0..16 {
                        out[c] += ar * m[c];
                    }
                }
                sink += out[0];
            }
            black_box(sink)
        });
    });
    group.finish();
}

fn throughput_norms(c: &mut Criterion) {
    let dense = dense_vectors(NORM_BATCH);
    let mut group = c.benchmark_group("throughput_norms");
    group.throughput(Throughput::Elements(NORM_BATCH as u64));
    group.bench_function("clifford_norm_throughput", |b| {
        b.iter(|| {
            let mut sink = 0.0f64;
            for v in &dense {
                sink += compute_clifford_norm_sq(black_box(&v.coeffs));
            }
            black_box(sink)
        });
    });
    group.finish();
}

fn throughput_projections(c: &mut Criterion) {
    let dense = dense_vectors(PROJECTION_BATCH + 1);
    let mut group = c.benchmark_group("throughput_projections");

    for grade in 0..=4usize {
        group.throughput(Throughput::Elements(PROJECTION_BATCH as u64));
        group.bench_with_input(
            BenchmarkId::new("grade_project_throughput", grade),
            &grade,
            |b, &g| {
                b.iter(|| {
                    let mut sink = 0.0f64;
                    for v in dense.iter().take(PROJECTION_BATCH) {
                        sink += black_box(v).grade_project(black_box(g)).max_abs_coeff;
                    }
                    black_box(sink)
                });
            },
        );
    }

    group.throughput(Throughput::Elements(DISTANCE_BATCH as u64));
    group.bench_function("geometric_distance_throughput", |b| {
        b.iter(|| {
            let mut sink = 0.0f64;
            for i in 0..DISTANCE_BATCH {
                sink += fast_metric_distance(black_box(&dense[i]), black_box(&dense[i + 1]));
            }
            black_box(sink)
        });
    });

    group.finish();
}

criterion_group!(
    clifford_ops,
    throughput_geometric_product,
    throughput_norms,
    throughput_projections
);
criterion_main!(clifford_ops);
