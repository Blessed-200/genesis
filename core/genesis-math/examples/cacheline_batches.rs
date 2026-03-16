#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use genesis_math::experimental::kernel_dense_g13::dense_geometric_product_g13;
use genesis_math::{sparse_geometric_product, SparseCliffordVector};

const BATCH: usize = 262_144;
const REPEATS: usize = 16;

fn build_dense_batch() -> Vec<[f64; 16]> {
    (0..BATCH)
        .map(|i| {
            let mut coeffs = [0.0; 16];
            for (j, coeff) in coeffs.iter_mut().enumerate() {
                *coeff = ((i + j) as f64 * 0.125).sin();
            }
            coeffs
        })
        .collect()
}

fn build_sparse_batch() -> Vec<SparseCliffordVector> {
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

fn bench_dense_contiguous(batch: &[[f64; 16]]) -> (f64, f64) {
    let pairs = batch.len() - 1;
    let mut checksum = 0.0;
    let start = Instant::now();
    for _ in 0..REPEATS {
        for i in 0..pairs {
            let out = dense_geometric_product_g13(&batch[i], &batch[i + 1]);
            checksum += out[0];
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput = (pairs * REPEATS) as f64 / elapsed;
    (throughput, checksum)
}

fn bench_sparse_contiguous(batch: &[SparseCliffordVector]) -> (f64, f64) {
    let pairs = batch.len() - 1;
    let mut checksum = 0.0;
    let start = Instant::now();
    for _ in 0..REPEATS {
        for i in 0..pairs {
            let out = sparse_geometric_product(&batch[i], &batch[i + 1]).unwrap();
            checksum += out.coeffs[0];
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput = (pairs * REPEATS) as f64 / elapsed;
    (throughput, checksum)
}

fn main() {
    println!(
        "layout size={} align={}",
        std::mem::size_of::<SparseCliffordVector>(),
        std::mem::align_of::<SparseCliffordVector>()
    );

    let dense = build_dense_batch();
    let sparse = build_sparse_batch();

    let (dense_tput, dense_sum) = bench_dense_contiguous(&dense);
    let (sparse_tput, sparse_sum) = bench_sparse_contiguous(&sparse);

    println!("dense_contiguous throughput_ops_s={dense_tput:.3} checksum={dense_sum:.6}");
    println!("sparse_contiguous throughput_ops_s={sparse_tput:.3} checksum={sparse_sum:.6}");
}
