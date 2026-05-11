use criterion::{criterion_group, criterion_main, Criterion};
use genesis_core::EngramStore;

fn rand_blades(seed: u64) -> [f64; 16] {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut out = [0.0; 16];
    for v in &mut out {
        x = x.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        *v = ((x >> 11) as f64 / (u64::MAX >> 11) as f64) * 2.0 - 1.0;
    }
    out
}

fn bench_pattern_complete(c: &mut Criterion) {
    let mut store = EngramStore::new(120_000, 0.00001);
    for i in 0..100_000_u64 {
        store
            .encode(rand_blades(i), i + 1, 1.0 + (i % 10) as f64, 0)
            .expect("encode");
    }
    store.dream_cycle(1).expect("dream cycle");
    let query = rand_blades(42);

    c.bench_function("cortical_pattern_complete_100k", |b| {
        b.iter(|| {
            let _ = store
                .pattern_complete(&query, 1.0, 2)
                .expect("pattern_complete");
        });
    });
}

criterion_group!(benches, bench_pattern_complete);
criterion_main!(benches);
