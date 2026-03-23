//! Criterion benchmarks for genesis-math geometry operations.
//!
//! Run with: cargo bench -p genesis-math
//!
//! Targets (Mandato §5.1, hardware: Intel Ice Lake or later with AVX-512):
//! - sparse_geo_product_16x16   < 175 cycles (wall-time p99)
//! - clifford_norm_sq_16        < 45 cycles
//! - grade_project_all_16       < 20 cycles
//! - from_iter_16_pairs         < 60 cycles
//! - metric_scalar_product_16x16  < 25 cycles

#![allow(clippy::cast_precision_loss, clippy::doc_markdown)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use genesis_math::experimental::kernel_dense_g13::dense_geometric_product_g13;
use genesis_math::{compute_clifford_norm_sq, sparse_geometric_product, SparseCliffordVector};
use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;

// ── Benchmark helpers ─────────────────────────────────────────────────────────

const CLIFFORD_NORM_WEIGHTS_F64_BENCH: [f64; 16] = [
    1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
];

fn derive_metadata_two_pass_baseline(mut buf: [f64; 16]) -> (u16, f64, f64) {
    let mut active_mask = 0u16;
    let mut max_abs_coeff = 0.0f64;
    for (k, coeff) in buf.iter_mut().enumerate() {
        let abs = coeff.abs();
        if abs > COGNITIVE_PLANCK_CONSTANT {
            active_mask |= 1u16 << k;
            if abs > max_abs_coeff {
                max_abs_coeff = abs;
            }
        } else {
            *coeff = 0.0;
        }
    }
    let clifford_norm_sq = compute_clifford_norm_sq(&buf);
    (active_mask, max_abs_coeff, clifford_norm_sq)
}

fn derive_metadata_single_pass(mut buf: [f64; 16]) -> (u16, f64, f64) {
    let mut active_mask = 0u16;
    let mut max_abs_coeff = 0.0f64;
    let mut clifford_norm_sq = 0.0f64;
    for k in 0..16 {
        let coeff = buf[k];
        let abs = coeff.abs();
        if abs > COGNITIVE_PLANCK_CONSTANT {
            active_mask |= 1u16 << k;
            if abs > max_abs_coeff {
                max_abs_coeff = abs;
            }
            clifford_norm_sq += coeff * coeff * CLIFFORD_NORM_WEIGHTS_F64_BENCH[k];
        } else {
            buf[k] = 0.0;
        }
    }
    (active_mask, max_abs_coeff, clifford_norm_sq)
}

fn deterministic_dense_inputs() -> Vec<[f64; 16]> {
    let mut state = 0xA5A5_1234_5678_9ABCu64;
    let mut out = Vec::with_capacity(4096);
    for _ in 0..4096 {
        let mut buf = [0.0f64; 16];
        for (blade, coeff) in buf.iter_mut().enumerate() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            if (state & 0b111) == ((blade as u64) & 0b111) {
                *coeff = unit * 4.0 - 2.0;
            } else {
                *coeff = (unit * 2.0 - 1.0) * (COGNITIVE_PLANCK_CONSTANT * 0.25);
            }
        }
        out.push(buf);
    }
    out
}

fn bench_from_dense_single_pass(c: &mut Criterion) {
    let inputs = deterministic_dense_inputs();
    let mut group = c.benchmark_group("from_dense_metadata");

    group.bench_function("from_dense_two_pass_baseline", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0.0f64;
            for buf in &inputs {
                let (mask, max_abs, norm_sq) = derive_metadata_two_pass_baseline(*buf);
                checksum += f64::from(mask) + max_abs + norm_sq;
            }
            black_box(checksum)
        });
    });

    group.bench_function("from_dense_single_pass", |bencher| {
        bencher.iter(|| {
            let mut checksum = 0.0f64;
            for buf in &inputs {
                let (mask, max_abs, norm_sq) = derive_metadata_single_pass(*buf);
                checksum += f64::from(mask) + max_abs + norm_sq;
            }
            black_box(checksum)
        });
    });

    group.finish();
}

/// Builds a multivector with all 16 blades active, coefficient = 1.0 each.
fn full_mv() -> SparseCliffordVector {
    SparseCliffordVector::from_iter((0..16).map(|i| (i, 1.0))).unwrap()
}

/// Builds a multivector with a single blade active.
fn single_blade(blade: usize, coef: f64) -> SparseCliffordVector {
    SparseCliffordVector::from_iter([(blade, coef)]).unwrap()
}

fn sparse_mv_from_mask(mask: u16, seed: f64) -> SparseCliffordVector {
    SparseCliffordVector::from_iter((0..16).filter_map(|i| {
        if (mask & (1 << i)) != 0 {
            Some((i, (i as f64).mul_add(0.25, seed)))
        } else {
            None
        }
    }))
    .unwrap()
}

fn contiguous_sparse_batch() -> Vec<SparseCliffordVector> {
    const BATCH: usize = 4096;
    (0..BATCH)
        .map(|i| {
            SparseCliffordVector::from_iter((0..16).filter_map(|j| {
                let active = ((i + j) % 5) == 0;
                if active {
                    Some((j, ((i * (j + 1)) as f64 * 0.03125).cos()))
                } else {
                    None
                }
            }))
            .unwrap()
        })
        .collect()
}

fn bench_sparse_geo_product_mask_patterns(c: &mut Criterion) {
    let patterns: [(&str, u16, u16); 4] = [
        ("1x1", 0x0001, 0x0002),
        ("1xN", 0x0001, 0x00FF),
        ("NxN", 0x00FF, 0x0F0F),
        ("dense", 0xFFFF, 0xFFFF),
    ];

    let mut group = c.benchmark_group("product_mask_patterns");
    for (name, mask_a, mask_b) in patterns {
        let a = sparse_mv_from_mask(mask_a, 1.0);
        let b = sparse_mv_from_mask(mask_b, 2.0);
        group.bench_function(name, |bencher| {
            bencher.iter(|| black_box(sparse_geometric_product(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

// ── Benchmark: sparse_geometric_product — 16×16 active blades ────────────────

fn bench_sparse_geo_product(c: &mut Criterion) {
    let a = full_mv();
    let b = full_mv();

    c.benchmark_group("product")
        .bench_function("sparse_geo_product_16x16", |bencher| {
            bencher.iter(|| black_box(sparse_geometric_product(black_box(&a), black_box(&b))));
        });
}

// ── Benchmark: compute_clifford_norm_sq — 16 active blades ───────────────────

fn bench_clifford_norm_sq(c: &mut Criterion) {
    let mv = full_mv();
    let buf = mv.coeffs;

    c.benchmark_group("norm")
        .bench_function("clifford_norm_sq_16", |bencher| {
            bencher.iter(|| black_box(compute_clifford_norm_sq(black_box(&buf))));
        });
}

// ── Benchmark: grade_project — all 16 blades, each grade ─────────────────────

fn bench_grade_project(c: &mut Criterion) {
    let mv = full_mv();
    let mut group = c.benchmark_group("grade_project");

    for grade in 0usize..=4 {
        group.bench_with_input(
            BenchmarkId::new("grade_project_16blades", grade),
            &grade,
            |bencher, &g| {
                bencher.iter(|| black_box(black_box(&mv).grade_project(g)));
            },
        );
    }
    group.finish();
}

// ── Benchmark: from_iter — 16 pairs ──────────────────────────────────────────

fn bench_from_iter(c: &mut Criterion) {
    let pairs: Vec<(usize, f64)> = (0..16).map(|i| (i, (i + 1) as f64)).collect();

    c.benchmark_group("constructor")
        .bench_function("from_iter_16_pairs", |bencher| {
            bencher.iter(|| {
                black_box(SparseCliffordVector::from_iter(black_box(
                    pairs.iter().copied(),
                )))
            });
        });
}

// ── Benchmark: metric_scalar_product — 16×16 active blades ───────────────────

fn bench_clifford_inner_product(c: &mut Criterion) {
    let a = full_mv();
    let b = full_mv();

    c.benchmark_group("inner_product")
        .bench_function("metric_scalar_product_16x16", |bencher| {
            bencher.iter(|| black_box(black_box(&a).metric_scalar_product(black_box(&b))));
        });
}

// ── Benchmark: single-blade product (minimal case) ───────────────────────────

fn bench_single_blade_product(c: &mut Criterion) {
    let e0 = single_blade(0b0001, 1.0);
    let e1 = single_blade(0b0010, 1.0);

    c.benchmark_group("product")
        .bench_function("sparse_geo_product_1x1", |bencher| {
            bencher.iter(|| black_box(sparse_geometric_product(black_box(&e0), black_box(&e1))));
        });
}

// ── Benchmark: dense kernel experimental vs sparse reference ───────────────────

fn bench_dense_kernel_compare(c: &mut Criterion) {
    let a = full_mv();
    let b = full_mv();

    let mut group = c.benchmark_group("product_dense_compare");
    group.bench_function("dense_kernel_g13_16x16", |bencher| {
        bencher.iter(|| {
            black_box(dense_geometric_product_g13(
                black_box(&a.coeffs),
                black_box(&b.coeffs),
            ))
        });
    });
    group.bench_function("sparse_kernel_g13_16x16", |bencher| {
        bencher.iter(|| black_box(sparse_geometric_product(black_box(&a), black_box(&b))));
    });
    group.finish();
}

fn bench_naive_matmul_16x16(c: &mut Criterion) {
    let a = full_mv();
    let b = full_mv();
    let mat_a = [[1.0_f64 / 16.0; 16]; 16];
    let bm = [[1.0_f64 / 16.0; 16]; 16];

    let mut group = c.benchmark_group("comparison_baseline");
    group.bench_function("dense_kernel_g13_16x16", |bencher| {
        bencher.iter(|| {
            black_box(dense_geometric_product_g13(
                black_box(&a.coeffs),
                black_box(&b.coeffs),
            ))
        });
    });
    group.bench_function("naive_matmul_16x16_f64", |bencher| {
        bencher.iter(|| {
            let mut out = [[0.0_f64; 16]; 16];
            for i in 0..16 {
                for k in 0..16 {
                    for j in 0..16 {
                        out[i][j] += mat_a[i][k] * bm[k][j];
                    }
                }
            }
            criterion::black_box(out)
        });
    });
    group.finish();
}

fn bench_bivector_norm_sq_of_product(c: &mut Criterion) {
    use genesis_math::bivector_norm_sq_of_product;
    let a = full_mv();
    let b = full_mv();

    c.benchmark_group("product")
        .bench_function("bivector_norm_sq_of_product_16x16", |bencher| {
            bencher.iter(|| black_box(bivector_norm_sq_of_product(black_box(&a), black_box(&b))));
        });
}

#[inline]
fn matmul4x4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0f64;
            for k in 0..4 {
                sum = a[i][k].mul_add(b[k][j], sum);
            }
            out[i][j] = sum;
        }
    }
    out
}

fn matrix4_from_mv(mv: &SparseCliffordVector) -> [[f64; 4]; 4] {
    core::array::from_fn(|i| core::array::from_fn(|j| mv.coeffs[i * 4 + j]))
}

fn bench_elite_algebra_vs_matrix4(c: &mut Criterion) {
    let a_mv = full_mv();
    let b_mv = SparseCliffordVector::from_iter((0..16).map(|i| (i, (i as f64).sin()))).unwrap();
    let a_mat = matrix4_from_mv(&a_mv);
    let b_mat = matrix4_from_mv(&b_mv);

    let mut group = c.benchmark_group("elite_algebra_compare");
    group.bench_function("clifford_g13_geometric_product_sparse", |bencher| {
        bencher.iter(|| black_box(sparse_geometric_product(black_box(&a_mv), black_box(&b_mv))));
    });
    group.bench_function("matrix4x4_multiplication_fma", |bencher| {
        bencher.iter(|| black_box(matmul4x4(black_box(&a_mat), black_box(&b_mat))));
    });
    group.finish();
}

// ── Benchmark: sparse_geometric_product — contiguous sparse batch ───────────

fn bench_sparse_geo_product_contiguous_batch(c: &mut Criterion) {
    let batch = contiguous_sparse_batch();
    let pairs = batch.len() - 1;

    c.benchmark_group("product_batch_contiguous")
        .throughput(criterion::Throughput::Elements(pairs as u64))
        .bench_function("sparse_geo_product_contiguous_batch", |bencher| {
            bencher.iter(|| {
                let mut checksum = 0.0;
                for i in 0..pairs {
                    let out =
                        sparse_geometric_product(black_box(&batch[i]), black_box(&batch[i + 1]))
                            .expect("sparse batch product should be valid");
                    checksum += out.coeffs[0];
                }
                black_box(checksum)
            });
        });
}

// ── Criterion entry points ────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_sparse_geo_product,
    bench_clifford_norm_sq,
    bench_grade_project,
    bench_from_iter,
    bench_clifford_inner_product,
    bench_single_blade_product,
    bench_sparse_geo_product_mask_patterns,
    bench_dense_kernel_compare,
    bench_bivector_norm_sq_of_product,
    bench_sparse_geo_product_contiguous_batch,
    bench_from_dense_single_pass,
    bench_elite_algebra_vs_matrix4,
);
criterion_group!(comparison_baseline, bench_naive_matmul_16x16);
criterion_main!(benches, comparison_baseline);
