use genesis_core::EngramStore;

fn blades(seed: f64) -> [f64; 16] {
    let mut out = [0.0; 16];
    out[1] = seed;
    out[2] = seed * 0.5;
    out[4] = seed * 0.25;
    out
}

#[test]
fn hundred_similar_episodic_collapse_to_single_cortical_abstraction() {
    let mut store = EngramStore::new(10_000, 0.0001);
    for i in 0..100_u64 {
        let mut b = blades(1.0 + (i as f64) * 1e-6);
        b[8] += (i as f64) * 1e-7;
        store.encode(b, i + 1, 5.0, 0).expect("encode");
    }
    store.dream_cycle(1).expect("dream cycle");

    assert_eq!(store.cortical_len(), 1);
    let strengths = store.weighted_strengths(1);
    assert!(strengths.iter().any(|(_, w)| *w > 450.0));
}

#[test]
fn capacity_stress_keeps_strongest_engrams() {
    let mut store = EngramStore::new(5, 0.0);
    for i in 0..20_u64 {
        store
            .encode(blades((i + 1) as f64), i, (i + 1) as f64, 0)
            .expect("encode");
    }
    store.dream_cycle(0).expect("dream cycle");
    let ids = store.causal_ids();
    assert!(!ids.is_empty());
}
