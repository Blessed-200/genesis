#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::OnceLock;
use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use genesis_dynamics::{
    synchrony_order, synchrony_order_fast, QuantumKuramotoNetwork, QuantumOscillator, VFEMinimizer,
};
use genesis_types::NodeId;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn build_network_sparse(n: usize, k: usize, temperature: f64) -> QuantumKuramotoNetwork {
    let mut net = QuantumKuramotoNetwork::new(temperature);
    for i in 0..n {
        let freq = (i as f64) * 0.01;
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
            [freq; 5],
        ))
        .expect("NodeId válido por construcción");
    }
    // Ring coupling: each node connected to k neighbors
    for i in 0..n {
        for d in 1..=(k / 2) {
            let j = (i + d) % n;
            net.set_coupling(
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                NodeId::try_new(j as u64).expect("NodeId válido por construcción"),
                0.5,
            );
            net.set_coupling(
                NodeId::try_new(j as u64).expect("NodeId válido por construcción"),
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                0.5,
            );
        }
    }
    net
}

fn build_vfe(n: usize) -> VFEMinimizer {
    let mut vfe = VFEMinimizer::new();
    for i in 0..n {
        let mean = [(i as f64) * 0.001, 0.0, 0.0, 0.0];
        vfe.add_node(
            NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
            mean,
        );
    }
    vfe
}

// ── Guardrails of baseline ─────────────────────────────────────────────────────

const BASELINE_KURAMOTO_STEP_1000_NS: f64 = 1_400_000.0;
/// Guardrail for the current scalar implementation (10 000 trigonometric ops).
/// En hardware moderno, ~200 000 ns. A margin of 250 000 ns.
const BASELINE_SYNCHRONY_ORDER_1000_NS: f64 = 250_000.0;
/// Guardrail for `synchrony_order_fast` at N=1000 (serial path, N < `RAYON_THRESHOLD=4096`).
///
/// ## Threshold derivation
///
/// Serial path: 1000 nodes × 5 grades = 5000 scalar `sin_cos` evaluations.
/// Baseline estimate: ~10ns per evaluation ≈ 50µs total.
/// CI variance budget: +60% on shared runners.
/// Final guardrail: `80_000ns` (80µs).
const BASELINE_SYNCHRONY_ORDER_FAST_1000_NS: f64 = 80_000.0;
const BASELINE_VFE_COMPUTE_1000_NS: f64 = 4_200.0;

fn guardrail_baselines() {
    static ONCE: OnceLock<()> = OnceLock::new();
    if ONCE.get().is_some() {
        return;
    }
    let mut net = build_network_sparse(1000, 8, 0.01);
    net.step(0.01);

    let start = Instant::now();
    for _ in 0..50 {
        net.step(0.01);
    }
    let kuramoto_ns = start.elapsed().as_nanos() as f64 / 50.0;

    let net_sync = build_network_sparse(1000, 8, 0.0);
    let start = Instant::now();
    for _ in 0..2000 {
        black_box(synchrony_order(&net_sync));
    }
    let sync_ns = start.elapsed().as_nanos() as f64 / 2000.0;

    let start = Instant::now();
    for _ in 0..2000 {
        black_box(synchrony_order_fast(&net_sync));
    }
    let sync_fast_ns = start.elapsed().as_nanos() as f64 / 2000.0;

    let vfe = build_vfe(1000);
    let ids: Vec<NodeId> = (0..1000u64)
        .map(|id| NodeId::try_new(id).expect("NodeId válido por construcción"))
        .collect();
    let start = Instant::now();
    for _ in 0..500 {
        let mut total = 0.0;
        for &id in &ids {
            total += vfe.compute_vfe(id, None);
        }
        black_box(total);
    }
    let vfe_ns = start.elapsed().as_nanos() as f64 / 500.0;

    assert!(
        kuramoto_ns <= BASELINE_KURAMOTO_STEP_1000_NS,
        "kuramoto_step_1000_nodes guardrail exceeded: {kuramoto_ns:.0}ns > {BASELINE_KURAMOTO_STEP_1000_NS:.0}ns"
    );
    assert!(
        sync_ns <= BASELINE_SYNCHRONY_ORDER_1000_NS,
        "synchrony_order_1000_nodes guardrail exceeded: {sync_ns:.0}ns > {BASELINE_SYNCHRONY_ORDER_1000_NS:.0}ns"
    );
    assert!(
        sync_fast_ns <= BASELINE_SYNCHRONY_ORDER_FAST_1000_NS,
        "synchrony_order_fast_1000_nodes guardrail exceeded: {sync_fast_ns:.0}ns > {BASELINE_SYNCHRONY_ORDER_FAST_1000_NS:.0}ns"
    );
    assert!(
        vfe_ns <= BASELINE_VFE_COMPUTE_1000_NS,
        "vfe_compute_1000_nodes guardrail exceeded: {vfe_ns:.0}ns > {BASELINE_VFE_COMPUTE_1000_NS:.0}ns"
    );

    let _ = ONCE.set(());
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_kuramoto_step_1000_nodes(c: &mut Criterion) {
    guardrail_baselines();
    let mut net = build_network_sparse(1000, 8, 0.01);
    // Pre-warm: one step to force internal index rebuild
    net.step(0.01);
    c.bench_function("kuramoto_step_1000_nodes", |b| {
        b.iter(|| {
            net.step(black_box(0.01));
        });
    });
}

fn bench_synchrony_order_1000_nodes(c: &mut Criterion) {
    guardrail_baselines();
    let net = build_network_sparse(1000, 8, 0.0);
    c.bench_function("synchrony_order_1000_nodes", |b| {
        b.iter(|| {
            black_box(synchrony_order(black_box(&net)));
        });
    });
}

fn bench_synchrony_order_fast_1000_nodes(c: &mut Criterion) {
    guardrail_baselines();
    let net = build_network_sparse(1000, 8, 0.0);
    c.bench_function("synchrony_order_fast_1000_nodes", |b| {
        b.iter(|| {
            black_box(synchrony_order_fast(black_box(&net)));
        });
    });
}

fn bench_vfe_compute_1000_nodes(c: &mut Criterion) {
    guardrail_baselines();
    let vfe = build_vfe(1000);
    let ids: Vec<NodeId> = (0..1000u64)
        .map(|id| NodeId::try_new(id).expect("NodeId válido por construcción"))
        .collect();
    c.bench_function("vfe_compute_1000_nodes", |b| {
        b.iter(|| {
            let mut total = 0.0f64;
            for &id in &ids {
                total += vfe.compute_vfe(black_box(id), None);
            }
            black_box(total)
        });
    });
}

criterion_group!(
    benches,
    bench_kuramoto_step_1000_nodes,
    bench_synchrony_order_1000_nodes,
    bench_synchrony_order_fast_1000_nodes,
    bench_vfe_compute_1000_nodes,
);
criterion_main!(benches);
