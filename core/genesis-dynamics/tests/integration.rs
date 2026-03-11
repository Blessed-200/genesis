#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args
)]

use genesis_dynamics::*;
use genesis_types::NodeId;

#[test]
fn stress_kuramoto_10k_steps_no_nan() {
    let n = 50usize;
    let mut net = QuantumKuramotoNetwork::new(0.05);
    for i in 0..n {
        let freq = (i as f64) * 0.1;
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
            [freq; 5],
        ))
        .expect("NodeId válido por construcción");
    }
    for i in 0..n {
        for j in [1, 2, 3] {
            let jj = (i + j) % n;
            net.set_coupling(
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                NodeId::try_new(jj as u64).expect("NodeId válido por construcción"),
                0.3,
            );
        }
    }
    for step in 0..10_000 {
        net.step(0.01);
        if step % 1000 == 0 {
            for osc in net.phases() {
                for &p in &osc.phases {
                    assert!(
                        p.is_finite(),
                        "NaN/Inf en step {} nodo {:?}",
                        step,
                        osc.node_id
                    );
                }
            }
        }
    }
    let r = synchrony_order(&net);
    assert!(r.is_finite(), "r_sync no finito tras 10k steps");
}

#[test]
fn stress_vfe_convergence_100_nodes() {
    let mut minimizer = VFEMinimizer::new();
    for i in 0..100u64 {
        minimizer.add_node(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            [0.0; 4],
        );
    }
    let obs = [1.0f64, 0.5, -0.3, 0.1];
    for _ in 0..500 {
        for i in 0..100u64 {
            minimizer.update(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &obs,
                0.05,
            );
        }
    }
    for i in 0..100u64 {
        let vfe = minimizer.compute_vfe(
            NodeId::try_new(i).expect("NodeId válido por construcción"),
            Some(&obs),
        );
        assert!(vfe < 0.01, "nodo {} no convergió: VFE = {:.6}", i, vfe);
    }
}

#[test]
fn critical_coupling_threshold() {
    use genesis_dynamics::kuramoto_critical_coupling;
    let kc = kuramoto_critical_coupling(1.0);
    assert!(
        (kc - 1.5958).abs() < 0.001,
        "K_c = {:.4}, expected ≈ 1.5958",
        kc
    );

    let n = 30usize;
    let mut net = QuantumKuramotoNetwork::new(1.0);
    for i in 0..n {
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
    }
    let gamma_super = kc * 2.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                net.set_coupling(
                    NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                    NodeId::try_new(j as u64).expect("NodeId válido por construcción"),
                    gamma_super / n as f64,
                );
            }
        }
    }
    for _ in 0..2000 {
        net.step(0.01);
    }
    let r = synchrony_order(&net);
    assert!(r > 0.5, "Con Γ > K_c debe emerger sincronía: r = {:.4}", r);
}
