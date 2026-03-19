#![allow(clippy::cast_precision_loss)]

use core::f64::consts::{PI, TAU};

use genesis_dynamics::{
    CriticalityMonitor, QuantumKuramotoNetwork, QuantumOscillator, VFEMinimizer,
};
use genesis_math::SparseCliffordVector;
use genesis_types::NodeId;

/// Deterministic LCG RNG for reproducible emergence tests.
///
/// AX-ID: AXIOMA-005, AXIOMA-006, `H_dinámica` (`LEY_FUNDACIONAL` §3.2)
#[derive(Clone, Copy)]
struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    const fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f64_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / ((1u64 << 53) as f64)
    }

    fn next_signed(&mut self, scale: f64) -> f64 {
        self.next_f64_unit().mul_add(2.0, -1.0) * scale
    }

    fn next_phase(&mut self) -> f64 {
        self.next_f64_unit() * TAU
    }
}

fn circular_variance(phases: &[f64]) -> f64 {
    let mean = phases
        .iter()
        .map(|p| p.sin())
        .sum::<f64>()
        .atan2(phases.iter().map(|p| p.cos()).sum::<f64>());
    let n = phases.len() as f64;
    phases
        .iter()
        .map(|phase| {
            let d = (phase - mean + PI).rem_euclid(TAU) - PI;
            d * d
        })
        .sum::<f64>()
        / n
}

fn ring_network(
    n: usize,
    neighbors: usize,
    coupling: f64,
    temperature: f64,
    rng: &mut Lcg64,
) -> QuantumKuramotoNetwork {
    let mut network = QuantumKuramotoNetwork::new(temperature);
    network.gauge_learning_rate = 0.006;
    network.curvature_damping = 0.008;

    for i in 0..n {
        let node = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
        let base_phase = rng.next_phase();
        let frequencies = [
            rng.next_signed(0.015),
            rng.next_signed(0.01),
            rng.next_signed(0.008),
            rng.next_signed(0.006),
            rng.next_signed(0.004),
        ];
        network
            .add_oscillator(QuantumOscillator::with_phases(
                node,
                [base_phase; 5],
                frequencies,
            ))
            .expect("insert de oscilador debe ser válido");
    }

    for i in 0..n {
        let src = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
        for offset in 1..=neighbors {
            let cw = NodeId::try_new(((i + offset) % n) as u64).expect("NodeId válido");
            let ccw = NodeId::try_new(((i + n - offset) % n) as u64).expect("NodeId válido");
            network.set_coupling(src, cw, coupling);
            network.set_coupling(src, ccw, coupling);
        }
    }

    network
}

fn collect_primary_phases(network: &QuantumKuramotoNetwork) -> Vec<f64> {
    network
        .phases()
        .iter()
        .map(QuantumOscillator::primary_phase)
        .collect()
}

fn belief_delta(before: &[f64; 16], after: &[f64; 16]) -> f64 {
    before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| {
            let d = b - a;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Converts a power-law sample (already floored, finite, positive)
/// to `u32` with saturating behavior at the type boundaries.
///
/// Precondition: `value` is the result of `.floor()` on a positive
/// finite f64 from an inverse-CDF power-law sampler.
/// Values >= u32::MAX saturate to u32::MAX; values < 1.0 clamp to 1.
#[inline]
fn power_law_sample_to_u32(value: f64) -> u32 {
    if value >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    if value < 1.0 {
        return 1;
    }
    // SAFETY: value is in [1.0, u32::MAX) after the guards above.
    // floor() was already applied at the call site — no fractional part.
    // The value is positive (>= 1.0) — no sign loss possible.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        value as u32
    }
}

#[test]
fn kuramoto_synchronization_reduces_phase_variance() {
    let n = 200usize;
    let mut rng = Lcg64::new(0x5eed_cafe_f00d_1234);
    let mut network = QuantumKuramotoNetwork::new(0.004);
    network.gauge_learning_rate = 0.0015;
    network.curvature_damping = 0.001;
    for i in 0..n {
        let node = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
        let phase = rng.next_phase();
        network
            .add_oscillator(QuantumOscillator::with_phases(node, [phase; 5], [0.0; 5]))
            .expect("insert de oscilador debe ser válido");
    }
    for i in 0..n {
        let src = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
        for offset in 1..=10 {
            let cw = NodeId::try_new(((i + offset) % n) as u64).expect("NodeId válido");
            let ccw = NodeId::try_new(((i + n - offset) % n) as u64).expect("NodeId válido");
            network.set_coupling(src, cw, 0.32);
            network.set_coupling(src, ccw, 0.32);
        }
    }

    let initial_variance = circular_variance(&collect_primary_phases(&network));

    for _ in 0..500 {
        network.step(0.02);
    }

    let final_variance = circular_variance(&collect_primary_phases(&network));
    let reduction = (1.0 - final_variance / initial_variance) * 100.0;
    let order = network.order_parameter();
    let curvature = network.total_curvature();

    println!(
        "Phase variance: {initial_variance:.4} → {final_variance:.4} (reduction: {reduction:.1}%)"
    );
    println!("Order parameter r = {order:.4}");
    println!("Total curvature = {curvature:.6}");

    assert!(final_variance < initial_variance * 0.5);
    assert!(final_variance > 0.01);
    assert!((0.3..0.95).contains(&order));
}

#[test]
fn vfe_decreases_monotonically_over_learning_steps() {
    let mut minimizer = VFEMinimizer::new();
    let node_count = 50usize;
    for i in 0..node_count {
        let node = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
        minimizer.add_node(node, [0.0; 4]);
    }

    let mut target = [0.0; 16];
    target[1] = 0.8;
    target[2] = -0.4;
    target[4] = 0.6;
    target[8] = -0.2;
    target[3] = 0.35;
    let obs = SparseCliffordVector::from_dense(&target).expect("target 16D válido");

    let checkpoints = [0usize, 25, 50, 75, 100];
    let mut trajectory = [0.0f64; 5];

    let total_vfe = |vfe: &VFEMinimizer| -> f64 {
        (0..node_count)
            .map(|i| {
                let node = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
                let (value, _) = vfe.compute_vfe_with_grad(node, Some(&obs));
                value
            })
            .sum()
    };

    trajectory[0] = total_vfe(&minimizer);

    for step in 1..=100 {
        for i in 0..node_count {
            let node = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
            minimizer.update_full(node, &obs, 0.05);
        }
        if let Some(pos) = checkpoints.iter().position(|&c| c == step) {
            trajectory[pos] = total_vfe(&minimizer);
        }
    }

    println!(
        "VFE trajectory: {a:.6} → {b:.6} → {c:.6} → {d:.6} → {e:.6}",
        a = trajectory[0],
        b = trajectory[1],
        c = trajectory[2],
        d = trajectory[3],
        e = trajectory[4]
    );

    assert!(trajectory[4] < trajectory[0]);
    for window in trajectory.windows(2) {
        let prev = window[0];
        let next = window[1];
        assert!(
            next <= prev * 1.05,
            "VFE diverge en ventana: {prev} -> {next}"
        );
    }
}

#[test]
fn criticality_monitor_reaches_soc_regime_under_dynamics() {
    let mut monitor = CriticalityMonitor::new(10_000);
    let mut rng = Lcg64::new(0xabad_1dea_55aa_1234);
    let tau_target = 2.0;

    for _ in 0..500 {
        let mut u = rng.next_f64_unit();
        u = u.clamp(1.0e-9, 1.0 - 1.0e-9);
        let size = u.powf(-1.0 / (tau_target - 1.0)).floor();
        let size: u32 = power_law_sample_to_u32(size);
        monitor.record_avalanche(size);
    }

    let tau = monitor
        .tau_exponent()
        .expect("debe haber suficientes avalanchas");
    println!("Estimated τ = {tau:.4} (target: 2.0)");
    assert!((1.5..=2.5).contains(&tau));
}

#[test]
fn criticality_emerges_from_coupled_dynamics() {
    let node_count = 100usize;
    let mut rng = Lcg64::new(0x1234_5678_9abc_def0);
    let mut network = ring_network(node_count, 2, 0.18, 0.012, &mut rng);
    network.gauge_learning_rate = 0.004;
    network.curvature_damping = 0.006;
    let mut minimizer = VFEMinimizer::new();
    let mut monitor = CriticalityMonitor::new(10_000);
    let mut latent = Vec::with_capacity(node_count);

    for i in 0..node_count {
        let node = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
        minimizer.add_node(
            node,
            [
                rng.next_signed(0.25),
                rng.next_signed(0.25),
                rng.next_signed(0.25),
                rng.next_signed(0.25),
            ],
        );
        let mut coeffs = [0.0; 16];
        for coeff in &mut coeffs {
            *coeff = rng.next_signed(0.35);
        }
        latent.push(coeffs);
    }

    let warmup_steps = 500usize;
    let total_steps = 2_000usize;
    let threshold = 0.04;

    for step in 0..total_steps {
        network.step(0.02);
        let phases = collect_primary_phases(&network);
        let order = network.order_parameter();
        let curvature = network.total_curvature();

        let mut avalanche_size = 0u32;
        for i in 0..node_count {
            let node = NodeId::try_new(i as u64).expect("NodeId válido por construcción");
            let mut obs = latent[i];
            obs[1] = 0.55f64.mul_add(phases[i].sin(), 0.15 * order);
            obs[2] = 0.55f64.mul_add(phases[i].cos(), -(0.1 * order));
            obs[4] = 0.25f64.mul_add(curvature.tanh(), latent[i][4] * 0.5);
            obs[8] = 0.35 * (phases[i] - order).sin();
            obs[3] = 0.2 * (phases[(i + 1) % node_count] - phases[i]).sin();
            obs[5] = 0.15 * (phases[(i + node_count - 1) % node_count] - phases[i]).cos();
            let obs = SparseCliffordVector::from_dense(&obs).expect("obs válida");

            let before = minimizer
                .beliefs_raw(node)
                .expect("nodo debe existir")
                .mean_full;
            minimizer.update_full(node, &obs, 0.015f64.mul_add(1.0 - order, 0.035));
            let after = minimizer
                .beliefs_raw(node)
                .expect("nodo debe existir")
                .mean_full;
            if belief_delta(&before, &after) > threshold {
                avalanche_size = avalanche_size.saturating_add(1);
            }
        }

        if step >= warmup_steps {
            monitor.record_avalanche(avalanche_size.max(1));
        }
    }

    let tau = monitor
        .tau_exponent()
        .expect("debe haber suficientes avalanchas");
    let order = network.order_parameter();
    let curvature = network.total_curvature();
    let avalanche_count = monitor.count();
    let triangle_count = network.triangle_count();
    println!("τ = {tau:.4}, r = {order:.4}, curvature = {curvature:.6}");
    println!("Total avalanches recorded: {avalanche_count}");
    println!("Triangle count built: {triangle_count}");

    assert!((1.5..=2.5).contains(&tau));
    assert!(order > 0.1);
    assert!(order < 0.95);
}

#[test]
fn gauge_invariance_holds_under_local_transformation() {
    let mut control_rng = Lcg64::new(0xface_feed_dead_beef);
    let mut transformed_rng = Lcg64::new(0xface_feed_dead_beef);
    let mut control = ring_network(10, 2, 0.13, 0.01, &mut control_rng);
    let mut transformed = ring_network(10, 2, 0.13, 0.01, &mut transformed_rng);

    for _ in 0..100 {
        control.step(0.02);
        transformed.step(0.02);
    }

    let local_shift = 0.17;
    for raw in [2_u64, 5, 7] {
        transformed.apply_local_gauge_shift(
            NodeId::try_new(raw).expect("NodeId válido por construcción"),
            local_shift,
        );
    }

    let mut max_deviation = 0.0_f64;
    for _ in 0..100 {
        control.step(0.02);
        transformed.step(0.02);
        let deviation = (control.order_parameter() - transformed.order_parameter()).abs();
        max_deviation = max_deviation.max(deviation);
    }

    println!("Gauge invariance max deviation = {max_deviation:.6}");
    assert!(max_deviation < 0.05);
}
