#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args
)]

use genesis_dynamics::*;
use genesis_types::NodeId;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct CountingAllocator;

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn stress_kuramoto_10k_steps_no_nan() {
    let _guard = lock_tests();
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
    let _guard = lock_tests();
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
    let _guard = lock_tests();
    use genesis_dynamics::kuramoto_critical_coupling;
    let kc = kuramoto_critical_coupling(1.0);
    assert!(
        (kc - 1.5958).abs() < 0.001,
        "K_c = {:.4}, expected ≈ 1.5958",
        kc
    );

    let n = 30usize;
    let mut net = QuantumKuramotoNetwork::new(1.0);
    net.gauge_learning_rate = 0.0;
    net.curvature_damping = 0.0;
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

#[test]
fn heap_audit_update_cycle_performs_zero_allocations() {
    let _guard = lock_tests();
    let n = 64usize;
    let mut net = QuantumKuramotoNetwork::new(0.2);
    for i in 0..n {
        net.add_oscillator(QuantumOscillator::new(
            NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
            [0.01 * i as f64; 5],
        ))
        .expect("NodeId válido por construcción");
    }

    for i in 0..n {
        let next_idx = (i + 1) % n;
        net.set_coupling(
            NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
            NodeId::try_new(next_idx as u64).expect("NodeId válido por construcción"),
            0.15,
        );
    }

    // WARMUP: run until all internal scratch buffers reach
    // steady-state capacity. 64 steps is sufficient for
    // gauge_scratch, coupling_idx, and triangle index.
    // align(64) structs may trigger realloc on first resize —
    // this warmup absorbs all one-time setup allocations.
    for _ in 0..64 {
        net.step(0.01);
    }

    // Force gauge field rebuild if dirty flag is pending.
    // Ensures reverse_edges and edge_to_triangles are materialized
    // before the zero-allocation window begins.
    let _ = net.order_parameter();
    let _ = net.total_curvature();

    // Reset counter to zero immediately before the measurement
    // window. This eliminates any allocations from test binary
    // initialization, parallel test threads, or runtime setup
    // that occurred before this point.
    // The _guard mutex ensures no other test runs concurrently
    // (TEST_LOCK), so the window is clean.
    ALLOC_CALLS.store(0, Ordering::SeqCst);

    let before = ALLOC_CALLS.load(Ordering::SeqCst);
    for _ in 0..128 {
        net.step(0.01);
    }
    let after = ALLOC_CALLS.load(Ordering::SeqCst);

    // Steady-state invariant: net.step() performs zero heap
    // allocations in the measured window. Counter reset to 0
    // immediately before the window eliminates init noise.
    // TEST_LOCK ensures single-threaded execution of this test.
    // AX-ID: AXIOMA-013, H_dinámica (LEY_FUNDACIONAL §3.2)
    assert_eq!(
        after - before,
        0,
        "el ciclo principal net.step() debe realizar 0 reservaciones en heap"
    );
}
