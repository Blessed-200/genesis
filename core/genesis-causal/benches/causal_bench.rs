use criterion::{black_box, criterion_group, criterion_main, Criterion};
use genesis_causal::{CausalEdge, CausalOrder, DiracSpinor, GlobalSection};

fn chain_order(nodes: u64) -> CausalOrder {
    let mut order = CausalOrder::new();
    for i in 0..(nodes - 1) {
        let mut a = [0.0; 16];
        let mut b = [0.0; 16];
        a[1] = i as f64;
        b[1] = (i + 1) as f64;
        let edge = CausalEdge::compute(i, &a, i + 1, &b).expect("edge");
        order.add_edge(edge).expect("insert");
    }
    order
}

fn causal_benchmark(c: &mut Criterion) {
    c.bench_function("insert_10k_chain_edges", |b| {
        b.iter(|| {
            let mut order = CausalOrder::new();
            for i in 0..10_000_u64 {
                let mut a = [0.0; 16];
                let mut d = [0.0; 16];
                a[1] = i as f64;
                d[1] = (i + 1) as f64;
                let edge = CausalEdge::compute(i, &a, i + 1, &d).expect("edge");
                order.add_edge(edge).expect("insert");
            }
            black_box(order.edge_count());
        })
    });

    let order = chain_order(10_000);
    c.bench_function("past_lightcone_terminal", |b| {
        b.iter(|| {
            black_box(order.past_lightcone(9_999));
        })
    });

    c.bench_function("future_lightcone_root", |b| {
        b.iter(|| {
            black_box(order.future_lightcone(0));
        })
    });

    c.bench_function("verify_acyclic_diag", |b| {
        b.iter(|| {
            black_box(order.verify_acyclic().is_ok());
        })
    });

    let root = DiracSpinor {
        components: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    };
    let ids: Vec<u64> = (0..1024).collect();
    c.bench_function("global_section_compute_1k", |b| {
        b.iter(|| {
            black_box(GlobalSection::compute(&order, &ids, root.clone()).is_ok());
        })
    });
}

criterion_group!(benches, causal_benchmark);
criterion_main!(benches);
