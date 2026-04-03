#![allow(clippy::cast_precision_loss)]

use std::cell::RefCell;
use std::hint::black_box;
use std::sync::OnceLock;

use genesis_dynamics::{synchrony_order, QuantumKuramotoNetwork, QuantumOscillator, VFEMinimizer};
use genesis_types::NodeId;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};

fn build_network(n: usize, k: usize, temperature: f64) -> QuantumKuramotoNetwork {
    let mut net = QuantumKuramotoNetwork::new(temperature);
    for i in 0..n {
        let freq = i as f64 * 0.01;
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(i as u64).expect("valid NodeId"),
            [freq; 5],
        ))
        .expect("valid NodeId");
    }
    for i in 0..n {
        for d in 1..=(k / 2) {
            let j = (i + d) % n;
            let ni = NodeId::try_new(i as u64).expect("valid NodeId");
            let nj = NodeId::try_new(j as u64).expect("valid NodeId");
            net.set_coupling(ni, nj, 0.5);
            net.set_coupling(nj, ni, 0.5);
        }
    }
    net
}

fn build_vfe(n: usize) -> (VFEMinimizer, Vec<NodeId>) {
    let mut vfe = VFEMinimizer::new();
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = NodeId::try_new(i as u64).expect("valid NodeId");
        vfe.add_node(id, [i as f64 * 0.001, 0.0, 0.0, 0.0]);
        ids.push(id);
    }
    (vfe, ids)
}

fn kuramoto_fixture() -> QuantumKuramotoNetwork {
    build_network(2_048, 8, 0.01)
}

thread_local! {
    static SYNCHRONY_FIXTURE: RefCell<Option<QuantumKuramotoNetwork>> = const { RefCell::new(None) };
    static KURAMOTO_FIXTURE: RefCell<Option<QuantumKuramotoNetwork>> = const { RefCell::new(None) };
}

fn vfe_fixture() -> &'static (VFEMinimizer, Vec<NodeId>) {
    static FIXTURE: OnceLock<(VFEMinimizer, Vec<NodeId>)> = OnceLock::new();
    FIXTURE.get_or_init(|| build_vfe(4_096))
}

#[library_benchmark]
fn synchrony_order_hotpath() -> f64 {
    SYNCHRONY_FIXTURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let net = slot.get_or_insert_with(|| build_network(2_048, 8, 0.0));
        black_box(synchrony_order(black_box(net)))
    })
}

#[library_benchmark]
fn kuramoto_step_hotpath() {
    KURAMOTO_FIXTURE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let net = slot.get_or_insert_with(kuramoto_fixture);
        net.step(black_box(0.01));
        black_box(net.node_count());
    });
}

#[library_benchmark]
fn vfe_compute_hotpath() -> f64 {
    let (vfe, ids) = vfe_fixture();
    let mut total = 0.0;
    for &id in ids.iter().take(1024) {
        total += black_box(vfe.compute_vfe(black_box(id), None));
    }
    black_box(total)
}

library_benchmark_group!(
    name = hotpaths;
    benchmarks = synchrony_order_hotpath, kuramoto_step_hotpath, vfe_compute_hotpath
);
main!(library_benchmark_groups = hotpaths);
