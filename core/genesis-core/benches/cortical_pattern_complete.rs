use criterion::{criterion_group, criterion_main, Criterion};
use genesis_core::EngramStore;

fn rand_blades(seed: u64) -> [f64; 16] {
    let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut out = [0.0; 16];
    #[allow(clippy::cast_precision_loss)]
    for v in &mut out {
        x = x.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(3_037_000_493);
        *v = ((x >> 11) as f64 / (u64::MAX >> 11) as f64).mul_add(2.0, -1.0);
    }
    out
}

fn bench_pattern_complete(c: &mut Criterion) {
    let mut store = EngramStore::new(120_000, 0.000_01);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..100_000_u64 {
        store
            .encode(rand_blades(i), i + 1, ((i % 10) as f64).mul_add(1.0, 1.0), 0)
            .expect("encode");
    }
    store.dream_cycle(1);
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
