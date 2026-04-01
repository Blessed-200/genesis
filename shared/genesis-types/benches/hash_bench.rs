use ahash::AHashMap;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use genesis_types::NodeId;
use std::collections::HashMap;

const SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];

fn make_keys(n: usize) -> Vec<NodeId> {
    (0..n)
        .map(|i| NodeId::try_new(i as u64).expect("benchmark key must be a valid NodeId"))
        .collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_table_insert_nodeid");
    for &size in &SIZES {
        let keys = make_keys(size);

        group.bench_with_input(BenchmarkId::new("siphash_std", size), &keys, |b, keys| {
            b.iter(|| {
                let mut map = HashMap::<NodeId, usize>::with_capacity(keys.len());
                for (idx, &key) in keys.iter().enumerate() {
                    map.insert(key, idx);
                }
                black_box(map);
            });
        });

        group.bench_with_input(BenchmarkId::new("ahash", size), &keys, |b, keys| {
            b.iter(|| {
                let mut map = AHashMap::<NodeId, usize>::with_capacity(keys.len());
                for (idx, &key) in keys.iter().enumerate() {
                    map.insert(key, idx);
                }
                black_box(map);
            });
        });
    }
    group.finish();
}

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_table_lookup_nodeid");
    for &size in &SIZES {
        let keys = make_keys(size);

        let mut sip = HashMap::<NodeId, usize>::with_capacity(keys.len());
        let mut fast = AHashMap::<NodeId, usize>::with_capacity(keys.len());
        for (idx, &key) in keys.iter().enumerate() {
            sip.insert(key, idx);
            fast.insert(key, idx);
        }

        group.bench_with_input(BenchmarkId::new("siphash_std", size), &keys, |b, keys| {
            b.iter(|| {
                let mut acc = 0usize;
                for &key in keys {
                    acc = acc.wrapping_add(*sip.get(&key).expect("key must exist"));
                }
                black_box(acc);
            });
        });

        group.bench_with_input(BenchmarkId::new("ahash", size), &keys, |b, keys| {
            b.iter(|| {
                let mut acc = 0usize;
                for &key in keys {
                    acc = acc.wrapping_add(*fast.get(&key).expect("key must exist"));
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_table_iter_nodeid");
    for &size in &SIZES {
        let keys = make_keys(size);

        let mut sip = HashMap::<NodeId, usize>::with_capacity(keys.len());
        let mut fast = AHashMap::<NodeId, usize>::with_capacity(keys.len());
        for (idx, &key) in keys.iter().enumerate() {
            sip.insert(key, idx);
            fast.insert(key, idx);
        }

        group.bench_with_input(BenchmarkId::new("siphash_std", size), &size, |b, _| {
            b.iter(|| {
                let mut acc = 0u64;
                for (&k, &v) in &sip {
                    acc = acc.wrapping_add(k.get().wrapping_add(v as u64));
                }
                black_box(acc);
            });
        });

        group.bench_with_input(BenchmarkId::new("ahash", size), &size, |b, _| {
            b.iter(|| {
                let mut acc = 0u64;
                for (&k, &v) in &fast {
                    acc = acc.wrapping_add(k.get().wrapping_add(v as u64));
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

fn hash_bench(c: &mut Criterion) {
    bench_insert(c);
    bench_lookup(c);
    bench_iter(c);
}

criterion_group!(benches, hash_bench);
criterion_main!(benches);

// Expected trend: ahash throughput is typically 2x–4x faster than std SipHash
// for non-adversarial integer-like keys such as NodeId in table-heavy workloads.
