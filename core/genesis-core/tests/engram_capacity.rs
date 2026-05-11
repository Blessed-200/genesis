use genesis_core::EngramStore;

fn blades(seed: f64) -> [f64; 16] {
    let mut out = [0.0; 16];
    out[1] = seed;
    out[2] = seed * 0.5;
    out[4] = seed * 0.25;
    out
}

#[test]
fn capacity_stress_keeps_strongest_engrams() {
    let mut store = EngramStore::new(5, 0.0);
    for i in 0..20_u64 {
        store
            .encode(blades(i as f64), i, (i + 1) as f64, 0)
            .expect("encode");
    }
    store.dream_cycle(0);

    let ids = store.causal_ids();
    assert_eq!(ids, vec![15, 16, 17, 18, 19]);
}

#[test]
fn weighted_strengths_are_monotone_in_selected_survivors() {
    let mut store = EngramStore::new(4, 0.0);
    store.encode(blades(0.0), 4, 4.0, 0).expect("encode");
    store.encode(blades(0.0), 1, 1.0, 0).expect("encode");
    store.encode(blades(0.0), 3, 3.0, 0).expect("encode");
    store.encode(blades(0.0), 2, 2.0, 0).expect("encode");
    store.encode(blades(0.0), 8, 8.0, 0).expect("encode");
    store.dream_cycle(0);

    let mut strengths = store.weighted_strengths(0);
    strengths.sort_unstable_by_key(|(id, _)| *id);
    let ids: Vec<u64> = strengths.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, vec![2, 3, 4, 8]);
}
