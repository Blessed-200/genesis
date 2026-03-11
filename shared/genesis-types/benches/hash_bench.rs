use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ring::digest::{digest, SHA256};

fn blake3_hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

fn sha256_baseline(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(digest(&SHA256, data).as_ref());
    out
}

fn hash_bench(c: &mut Criterion) {
    let witness = vec![0xAB; 4096];

    c.bench_function("blake3", |b| {
        b.iter(|| blake3_hash(black_box(&witness)));
    });

    c.bench_function("sha256_baseline", |b| {
        b.iter(|| sha256_baseline(black_box(&witness)));
    });
}

criterion_group!(benches, hash_bench);
criterion_main!(benches);
