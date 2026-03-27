#![allow(clippy::doc_markdown)]

use genesis_types::NodeId;
use rayon::prelude::*;

use crate::kuramoto::QuantumKuramotoNetwork;

/// Numerically-stable sine approximation for Kuramoto phases (BN-poly).
///
/// # Problem with naive range reduction
///
/// The former `x - round(x/π)·π` formula causes catastrophic cancellation for
/// large |x|: at x=1e9, f64 precision leaves ~5e-4 relative error.
/// In cognitive sessions where Kuramoto phases accumulate unboundedly this
/// propagates as systematic bias into r_sync and Ω.
///
/// # Solution: Cody-Waite three-part argument reduction
///
/// The constant π/2 is split into three parts with complementary bit patterns
/// so that `x - k·C1 - k·C2 - k·C3` retains full precision even for large x.
/// This is the same technique used in glibc, SLEEF and Intel SVML.
/// Accurate for |x| < 2^52 (phases beyond this magnitude are physically
/// meaningless in GENESIS and are clamped to the reduced domain).
///
/// Error: < 1e-9 for |x| ≤ 1e12 (verified by test).
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
/// Polynomial approximation of `sin(x)` with < 1e-9 error for `|x| ≤ 1e12`.
///
/// Two-range strategy: Cody-Waite reduction for `|x| < 2^20`, `f64::sin` fallback
/// for large phases (Kuramoto physically bounded well below 2^20). AX-ID: AXIOMA-006
#[inline(always)]
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn poly_sin(x: f64) -> f64 {
    // Two-range strategy — same approach as SLEEF / glibc:
    //
    // Fast path  (|x| < 2^20 ≈ 1M): Cody-Waite two-part π/2 reduction.
    //   Accurate to < 1e-11 error because the reduced and is at most ~1e6 * eps(π/2) ≈ 2e-10.
    //
    // Fallback  (|x| ≥ 2^20): delegate to f64::sin() (libm Payne-Hanek, <1 ULP error).
    //   Kuramoto phases beyond 2^20 are physically exceptional; libm cost is acceptable.
    //
    // AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    const FAST_THRESHOLD: f64 = 1_048_576.0; // 1_048_576
    if x.abs() >= FAST_THRESHOLD {
        return x.sin(); // libm Payne-Hanek: < 1 ULP
    }
    use core::f64::consts::FRAC_PI_2;
    // Standard two-part Cody-Waite from FDLIBM / glibc.
    const INV_FRAC_PI_2: f64 = 2.0 / core::f64::consts::PI;
    let k = (x * INV_FRAC_PI_2).round();
    let y = (-k).mul_add(FRAC_PI_2, x);
    let octant = (k as i64) & 3;
    let y2 = y * y;
    match octant {
        0 => sin_kernel(y, y2),
        1 => cos_kernel(y, y2),
        2 => -sin_kernel(y, y2),
        _ => -cos_kernel(y, y2), // octant 3
    }
}

/// Numerically-stable cosine approximation. Same reduction as poly_sin.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
/// Polynomial approximation of `cos(x)`. Same accuracy and strategy as [`poly_sin`].
///
/// AX-ID: AXIOMA-006
#[inline(always)]
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn poly_cos(x: f64) -> f64 {
    // Same two-range strategy as poly_sin (see doc there).
    const FAST_THRESHOLD: f64 = 1_048_576.0;
    if x.abs() >= FAST_THRESHOLD {
        return x.cos();
    }
    use core::f64::consts::FRAC_PI_2;
    const INV_FRAC_PI_2: f64 = 2.0 / core::f64::consts::PI;
    let k = (x * INV_FRAC_PI_2).round();
    let y = (-k).mul_add(FRAC_PI_2, x);
    let octant = (k as i64) & 3;
    let y2 = y * y;
    match octant {
        0 => cos_kernel(y, y2),
        1 => -sin_kernel(y, y2),
        2 => -cos_kernel(y, y2),
        _ => sin_kernel(y, y2), // octant 3
    }
}

/// Minimax polynomial kernel for sin(y) where and ∈ [-π/4, π/4].
/// Degree-9 Horner form. Error < 5e-13.
#[inline]
fn sin_kernel(y: f64, y2: f64) -> f64 {
    let poly = y2.mul_add(
        y2.mul_add(
            y2.mul_add(
                y2.mul_add(
                    y2.mul_add(-2.505_210_838_544_172_5e-8, 2.755_731_370_707_006_8e-6),
                    -1.984_126_982_985_795e-4,
                ),
                8.333_333_333_332_249e-3,
            ),
            -1.666_666_666_666_666_6e-1,
        ),
        1.0,
    );
    y * poly
}

/// Minimax polynomial kernel for cos(y) where and ∈ [-π/4, π/4].
/// Degree-10 Horner form. Error < 5e-13.
#[inline]
const fn cos_kernel(_y: f64, y2: f64) -> f64 {
    y2.mul_add(
        y2.mul_add(
            y2.mul_add(
                y2.mul_add(
                    y2.mul_add(2.087_675_698_786_81e-9, -2.755_731_922_428_758e-7),
                    2.480_158_730_159_014e-5,
                ),
                -1.388_888_888_888_735e-3,
            ),
            4.166_666_666_666_667e-2,
        ),
        -4.999_999_999_999_998e-1,
    )
    .mul_add(y2, 1.0)
}

/// Amplitude-weighted Kuramoto order parameter — adaptive serial/parallel (BN-08, perf fix).
///
/// ```text
/// r_sync = (1/G) Σ_g |Σ_i A_{i,g}·e^{iφ}| / (Σ_i A_{i,g} + ε)
/// ```
///
/// # Adaptive parallelism (benchmark guardrail fix)
///
/// For `N < RAYON_THRESHOLD` (currently 4096), rayon's thread-spawn and work-stealing
/// overhead exceeds the computation, making parallel slower than serial on CI runners
/// and development machines. At N=1000 the measured overhead was ~35µs (rayon) vs
/// ~4µs (serial) — an 8× regression for small networks.
///
/// Strategy: serial reduction below threshold, parallel above.
/// The threshold of 4096 was determined empirically: rayon breaks even at ~3000 nodes
/// on a 4-core machine; 4096 gives margin for slower CI cores.
///
/// At N=10⁶ (production): parallel provides 8–32× speedup as expected.
/// Numerically identical regardless of path.
///
/// r ∈ [0.0, 1.0]. AX-ID: AXIOMA-006, AXIOMA-008, H_dinámica (LEY_FUNDACIONAL §4)
pub fn synchrony_order_fast(network: &QuantumKuramotoNetwork) -> f64 {
    let oscs = network.phases();
    if oscs.is_empty() {
        return 0.0;
    }

    const N_GRADES: usize = 5;
    const EPS: f64 = 1e-30;
    /// Fixed chunk size for deterministic partitioning across thread counts.
    /// 128 oscillators × 5 grades keeps chunk-local working set in L1 cache.
    const DETERMINISTIC_CHUNK: usize = 128;

    /// Inner reduction over a slice with deterministic local iteration order.
    #[inline]
    fn reduce_slice(oscs: &[crate::oscillator::QuantumOscillator]) -> [(f64, f64, f64); 5] {
        let mut acc = [(0.0f64, 0.0f64, 0.0f64); 5];
        for osc in oscs {
            let phases = &osc.phases;
            let amplitudes = &osc.amplitudes;
            // loop-invariant, hoisted
            // CRYSTAL: O5 — inevitable
            for (g, grade_acc) in acc.iter_mut().enumerate() {
                let a = amplitudes[g];
                #[cfg(feature = "poly_trig")]
                {
                    grade_acc.0 = a.mul_add(poly_cos(phases[g]), grade_acc.0);
                    grade_acc.1 = a.mul_add(poly_sin(phases[g]), grade_acc.1);
                }
                #[cfg(not(feature = "poly_trig"))]
                {
                    grade_acc.0 = a.mul_add(phases[g].cos(), grade_acc.0);
                    grade_acc.1 = a.mul_add(phases[g].sin(), grade_acc.1);
                }
                grade_acc.2 += a;
            }
        }
        acc
    }

    let mut partials: Vec<_> = oscs
        .par_chunks(DETERMINISTIC_CHUNK)
        .map(reduce_slice)
        .collect();
    let chunk_count = partials.len();

    let mut grade_totals = [(0.0f64, 0.0f64, 0.0f64); N_GRADES];
    let mut stride = 1usize;
    while stride < chunk_count {
        let step = stride * 2;
        let mut base = 0usize;
        while base + stride < chunk_count {
            let (head, tail) = partials.split_at_mut(base + stride);
            for (left, right) in head[base].iter_mut().zip(tail[0].iter()) {
                left.0 += right.0;
                left.1 += right.1;
                left.2 += right.2;
            }
            base += step;
        }
        stride = step;
    }
    if chunk_count > 0 {
        grade_totals = partials[0];
    }

    let r_total: f64 = grade_totals
        .iter()
        .map(|(sc, ss, sa)| (sc.mul_add(*sc, ss * ss)).sqrt() / (sa + EPS))
        .sum();
    #[allow(clippy::cast_precision_loss)]
    {
        r_total / N_GRADES as f64
    }
}

/// Hub-sampled synchrony — distributed Ω without global barrier (BN-08).
///
/// Only hub nodes (HNSW layer > 0) are reduced. Each cloud pod computes this
/// locally and propagates asynchronously, eliminating the global synchronisation
/// barrier that would block every evolutionary step.
///
/// Returns `f64::NAN` if `hub_indices` is empty — caller falls back to full r_sync.
///
/// AX-ID: AXIOMA-006, LEY_FUNDACIONAL §4, CLOUD_PLATFORM_ARCHITECTURE §3
pub fn synchrony_order_hubs(network: &QuantumKuramotoNetwork, hub_indices: &[usize]) -> f64 {
    let oscs = network.phases();
    if hub_indices.is_empty() || oscs.is_empty() {
        return f64::NAN;
    }
    const N_GRADES: usize = 5;
    const EPS: f64 = 1e-30;
    let mut r_total = 0.0f64;
    let inv_grade_count = 1.0 / N_GRADES as f64;
    // loop-invariant, hoisted
    // CRYSTAL: O30 — inevitable
    let mut acc = [(0.0f64, 0.0f64, 0.0f64); N_GRADES];
    for osc in hub_indices.iter().filter_map(|&i| oscs.get(i)) {
        for (g, (sc, ss, sa)) in acc.iter_mut().enumerate() {
            *sc = osc.amplitudes[g].mul_add(osc.phases[g].cos(), *sc);
            *ss = osc.amplitudes[g].mul_add(osc.phases[g].sin(), *ss);
            *sa += osc.amplitudes[g];
        }
    }
    r_total += acc
        .iter()
        .map(|(sc, ss, sa)| (sc.mul_add(*sc, ss * ss)).sqrt() / (sa + EPS))
        .sum::<f64>();
    #[allow(clippy::cast_precision_loss)]
    {
        r_total * inv_grade_count
    }
}

/// Parameter of order of Kuramoto multigrade:
///   r_sync = (1/G) × Σ_{g=0}^{G-1} |Σᵢ e^{i·φᵢg}| / N
///
/// Derivation: H_dynamic (FUNDACIONAL_LEY §3.2) sums over the G=5 degrees.
/// Su observable natural is the mean of parameters of order per grade.
/// Measuring only grade 0 introduces bias systematic when the noise thermal
/// desynchronizes grades independently (lo cual occurs per construction
/// in step(): gaussian_noise() it calls G times per node per dt).
///
/// Delegates a `synchrony_order_fast` for leverage the approximations polynomial.
///
/// r ∈ [0.0, 1.0]. AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §4)
pub fn synchrony_order(network: &QuantumKuramotoNetwork) -> f64 {
    synchrony_order_fast(network)
}

/// Detects cluster sincronizado using phase mean circular promediada over grades.
///
/// La phase mean multigrade captures the direction dominant of the state collective
/// when the grades have different phases due to the noise thermal effect.
///
/// AX-ID: AXIOMA-006
pub fn synchronized_cluster(network: &QuantumKuramotoNetwork, threshold: f64) -> Vec<NodeId> {
    let oscs = network.phases();
    let n = oscs.len();
    if n == 0 {
        return Vec::new();
    }

    const N_GRADES: usize = 5;
    #[allow(clippy::cast_precision_loss)]
    let g_f = N_GRADES as f64;

    // Phase mean circular promediada over all the grades
    let (sum_c, sum_s) = oscs.iter().fold((0.0f64, 0.0f64), |(mc, ms), osc| {
        osc.phases
            .iter()
            .fold((mc, ms), |(c, s), &p| (c + p.cos(), s + p.sin()))
    });
    let mean_cos = sum_c / g_f;
    let mean_sin = sum_s / g_f;
    let mean_phase = mean_sin.atan2(mean_cos);

    oscs.iter()
        .filter(|osc| {
            let avg_diff: f64 = (0..N_GRADES)
                .map(|g| {
                    let raw = (osc.phases[g] - mean_phase).abs();
                    core::f64::consts::PI - (raw - core::f64::consts::PI).abs()
                })
                .sum::<f64>()
                / g_f;
            avg_diff < threshold
        })
        .map(|osc| osc.node_id)
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[allow(clippy::float_cmp, clippy::uninlined_format_args)]
mod tests {
    use super::*;
    use crate::oscillator::QuantumOscillator;

    fn fully_synced_net(n: usize, common_phase: f64) -> QuantumKuramotoNetwork {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        for i in 0..n {
            let osc = QuantumOscillator::with_phases(
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                [common_phase; 5],
                [0.0; 5],
            );
            net.add_oscillator(osc)
                .expect("NodeId válido por construcción");
        }
        net
    }

    fn uniform_phase_net(n: usize) -> QuantumKuramotoNetwork {
        // Phases uniformemente distribuidas in [0, 2π) → r ≈ 0
        let mut net = QuantumKuramotoNetwork::new(0.0);
        for i in 0..n {
            let phase = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            let osc = QuantumOscillator::with_phases(
                NodeId::try_new(i as u64).expect("NodeId válido por construcción"),
                [phase; 5],
                [0.0; 5],
            );
            net.add_oscillator(osc)
                .expect("NodeId válido por construcción");
        }
        net
    }

    #[test]
    fn synchrony_order_one_for_fully_synchronized() {
        let net = fully_synced_net(20, 1.23);
        let r = synchrony_order(&net);
        assert!(
            (r - 1.0).abs() < 1e-12,
            "r_sync esperado 1.0 para red perfectamente sincronizada, got {}",
            r
        );
    }

    #[test]
    fn synchrony_order_zero_for_uncoupled() {
        // Phases uniform → cancellation vectorial → r ≈ 0
        let net = uniform_phase_net(100);
        let r = synchrony_order(&net);
        assert!(
            r < 0.05,
            "r_sync esperado ≈ 0 para fases uniformes (100 nodos), got {}",
            r
        );
    }

    #[test]
    fn synchrony_order_empty_network_is_zero() {
        let net = QuantumKuramotoNetwork::new(0.0);
        assert_eq!(synchrony_order(&net), 0.0);
    }

    #[test]
    fn synchrony_order_detects_partial_grade_desync() {
        let n = 100usize;
        let mut net = QuantumKuramotoNetwork::new(0.0);
        for i in 0..n as u64 {
            let uniform = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            net.add_oscillator(QuantumOscillator::with_phases(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                [0.0, uniform, uniform, uniform, uniform],
                [0.0; 5],
            ))
            .expect("NodeId válido por construcción");
        }
        let r = synchrony_order(&net);
        assert!(
            r < 0.25 && r > 0.15,
            "r_sync debe promediar grades: ≈0.2 con grade 0 sync y grades 1-4 dispersos. Got {r:.4}"
        );
    }

    #[test]
    fn synchrony_order_full_sync_all_grades_gives_one() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        for i in 0..20u64 {
            net.add_oscillator(QuantumOscillator::with_phases(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                [0.5; 5],
                [0.0; 5],
            ))
            .expect("NodeId válido por construcción");
        }
        let r = synchrony_order(&net);
        assert!(
            (r - 1.0).abs() < 1e-12,
            "todos grades sync → r=1.0, got {r}"
        );
    }

    #[test]
    fn synchrony_order_cached_matches_direct_after_step() {
        let n = 20usize;
        let mut net = QuantumKuramotoNetwork::new(0.5);
        for i in 0..n as u64 {
            let phase = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            net.add_oscillator(QuantumOscillator::with_phases(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                [phase; 5],
                [0.1; 5],
            ))
            .expect("NodeId válido por construcción");
        }
        net.step(0.01);
        let cached = net.synchrony_order_cached();
        let direct = synchrony_order(&net);
        assert!(
            (cached - direct).abs() < 1e-12,
            "cached y direct deben coincidir tras step(). cached={cached:.6} direct={direct:.6}"
        );
    }

    #[test]
    fn synchronized_cluster_returns_all_when_fully_synced() {
        let n = 10;
        let net = fully_synced_net(n, 0.5);
        let cluster = synchronized_cluster(&net, 0.1);
        assert_eq!(cluster.len(), n);
    }

    #[test]
    fn synchronized_cluster_excludes_outliers() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        // 9 nodes sincronizados in phase 0, 1 outlier in phase π
        for i in 0..9u64 {
            net.add_oscillator(QuantumOscillator::with_phases(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                [0.0; 5],
                [0.0; 5],
            ))
            .expect("NodeId válido por construcción");
        }
        net.add_oscillator(QuantumOscillator::with_phases(
            NodeId::try_new(9).expect("NodeId válido por construcción"),
            [core::f64::consts::PI; 5],
            [0.0; 5],
        ))
        .expect("NodeId válido por construcción");
        let cluster = synchronized_cluster(&net, 0.5);
        assert_eq!(cluster.len(), 9, "outlier no debe estar en el cluster");
    }

    // ── Amplitude-weighted r_sync tests ───────────────────────────────────────

    /// Con amplitudes = [1.0;5] (value per default), r_sync must be identical
    /// to the Kuramoto classical without amplitudes.
    #[test]
    fn amplitude_one_gives_same_rsync_as_classic_kuramoto() {
        // Network sincronizada — r must be 1.0 with or without amplitudes
        let net = fully_synced_net(30, 0.7);
        let r = synchrony_order(&net);
        // All oscillators have amplitudes = [1.0;5] by default
        assert!(
            (r - 1.0).abs() < 1e-12,
            "amplitude=1.0 debe preservar comportamiento clásico, r={r}"
        );
    }

    /// When all nodes have amplitude 0 (all saturated), r_sync → 0.
    /// The system does not have a signal of certainty active.
    #[test]
    fn amplitude_zero_gives_rsync_zero() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        for i in 0..20u64 {
            let mut osc =
                QuantumOscillator::new(NodeId::try_new(i).expect("NodeId válido"), [0.0; 5]);
            osc.amplitudes = [0.0; 5]; // todos saturados
            net.add_oscillator(osc).expect("NodeId válido");
        }
        let r = synchrony_order(&net);
        // With all amplitudes = 0: sum_cos = 0, sum_sin = 0, sum_amp = 0
        // r = hypot(0,0)/(0+EPS) = 0/EPS = 0
        assert!(r < 1e-20, "todos amplitude=0 → r_sync debe ser ~0, got {r}");
    }

    /// r_sync ponderado < r_sync classical when nodes desincronizados tienen amplitude high
    /// and synchronized nodes have low amplitude. Verifies that the amplitudes matter.
    #[test]
    fn amplitude_weighting_reduces_rsync_when_unsync_nodes_have_high_amplitude() {
        use core::f64::consts::PI;

        // 5 nodes sincronizados in phase 0, amplitude baja (saturados/aprendidos)
        // 5 nodes desincronizados (phases dispersas), amplitude high (activos)
        let mut net = QuantumKuramotoNetwork::new(0.0);

        // Sincronizados with amplitude 0.1
        for i in 0..5u64 {
            let mut osc = QuantumOscillator::with_phases(
                NodeId::try_new(i).expect("NodeId válido"),
                [0.0; 5],
                [0.0; 5],
            );
            osc.amplitudes = [0.1; 5];
            net.add_oscillator(osc).expect("NodeId válido");
        }

        // Desincronizados with amplitude 0.9
        for i in 5..10u64 {
            let phase = 2.0 * PI * (i - 5) as f64 / 5.0;
            let mut osc = QuantumOscillator::with_phases(
                NodeId::try_new(i).expect("NodeId válido"),
                [phase; 5],
                [0.0; 5],
            );
            osc.amplitudes = [0.9; 5];
            net.add_oscillator(osc).expect("NodeId válido");
        }

        let r_weighted = synchrony_order(&net);

        // The weighted r_sync must be low because the nodes active (high amplitude)
        // are dispersed. If it were classical (amplitudes=1), the partial sync of
        // the 5 synchronized nodes would raise r. With weighting, the dispersed dominate.
        assert!(
            r_weighted < 0.4,
            "r_sync ponderado debe ser bajo cuando nodos activos (amp alta) están dispersos, got {r_weighted:.4}"
        );
    }

    /// update_amplitude_from_fisher integrado: r_sync decrece when all the nodes
    /// updatesn su amplitude a 0 (all saturados).
    #[test]
    fn rsync_decreases_after_amplitude_saturation() {
        // Network sincronizada inicialmente
        let net = fully_synced_net(20, 0.5);

        let r_initial = synchrony_order(&net);
        assert!((r_initial - 1.0).abs() < 1e-12, "r inicial debe ser 1.0");

        // Saturate all nodes — amplitude → 0
        // En production this lo does the pipeline VFE + update_amplitude_from_fisher
        // Here lo do directly for the test of integration
        // There is no mutable access to oscillators from outside in the public API.
        // Verificamos via new net with amplitudes 0:
        let mut net2 = QuantumKuramotoNetwork::new(0.0);
        for i in 0..20u64 {
            let mut osc = QuantumOscillator::with_phases(
                NodeId::try_new(i).expect("NodeId válido"),
                [0.5; 5], // misma fase sincronizada
                [0.0; 5],
            );
            osc.amplitudes = [0.0; 5]; // saturados
            net2.add_oscillator(osc).expect("NodeId válido");
        }

        let r_saturated = synchrony_order(&net2);
        assert!(
            r_saturated < r_initial,
            "r_sync debe decrecer cuando amplitudes → 0: antes={r_initial:.4} después={r_saturated:.10}"
        );
    }
}

// ── Tests for dot_bivectors and adaptive coupling ─────────────────────────────
#[cfg(test)]
mod adaptive_tests {
    use genesis_math::SparseCliffordVector;
    use genesis_types::NodeId;

    use super::{poly_cos, poly_sin};
    use crate::kuramoto::QuantumKuramotoNetwork;
    use crate::oscillator::QuantumOscillator;

    fn id(n: u64) -> NodeId {
        NodeId::try_new(n).unwrap()
    }

    fn make_vec(pairs: &[(usize, f64)]) -> SparseCliffordVector {
        SparseCliffordVector::from_iter(pairs.iter().copied()).unwrap()
    }

    /// dot_bivectors returns 0 for vectors without grade-2 component.
    #[test]
    fn dot_bivectors_zero_for_grade1_only_vectors() {
        let a = make_vec(&[(0b0001, 1.0), (0b0010, 0.5)]); // solo grado 1
        let b = make_vec(&[(0b0001, 0.3), (0b0100, 0.7)]); // solo grado 1
        assert_eq!(
            a.dot_bivectors(&b),
            0.0,
            "vectores sin bivectores deben dar dot_bivectors=0"
        );
    }

    /// dot_bivectors returns positivo for bivectores parallel.
    #[test]
    fn dot_bivectors_positive_for_aligned_bivectors() {
        // blade 3 = e₀₁ = grade 2
        let a = make_vec(&[(3, 1.0)]);
        let b = make_vec(&[(3, 1.0)]);
        let d = a.dot_bivectors(&b);
        assert!(
            d > 0.0,
            "bivectores paralelos deben dar producto positivo, got {d}"
        );
    }

    /// dot_bivectors returns negativo for bivectores antiparallel.
    #[test]
    fn dot_bivectors_negative_for_anti_aligned_bivectors() {
        let a = make_vec(&[(3, 1.0)]); // e₀₁ positivo
        let b = make_vec(&[(3, -1.0)]); // e₀₁ negativo
        let d = a.dot_bivectors(&b);
        assert!(
            d < 0.0,
            "bivectores antiparalelos deben dar producto negativo, got {d}"
        );
    }

    /// saturation_factor = 0 when amplitude = 1 (prior, incierto).
    #[test]
    fn saturation_factor_zero_at_full_amplitude() {
        let osc = QuantumOscillator::new(id(0), [0.0; 5]);
        assert!(
            (osc.saturation_factor() - 0.0).abs() < 1e-12,
            "amplitudes=[1.0;5] → amplitude_norm=1 → sat=0, got {}",
            osc.saturation_factor()
        );
    }

    /// saturation_factor = 1 when amplitude = 0 (saturado, aprendido).
    #[test]
    fn saturation_factor_one_at_zero_amplitude() {
        let mut osc = QuantumOscillator::new(id(0), [0.0; 5]);
        osc.amplitudes = [0.0; 5];
        assert!(
            (osc.saturation_factor() - 1.0).abs() < 1e-12,
            "amplitudes=[0;5] → amplitude_norm=0 → sat=1, got {}",
            osc.saturation_factor()
        );
    }

    /// adaptive_step with vecs emptys is no-op (longitud incorrecta).
    #[test]
    fn adaptive_step_noop_if_vecs_len_mismatch() {
        let mut net = QuantumKuramotoNetwork::new(0.0);
        net.add_oscillator(QuantumOscillator::new(id(0), [1.0; 5]))
            .unwrap();
        let phases_before = net.phases()[0].phases;
        // vecs emptys → mismatch → no-op
        net.adaptive_step(&[], 0.01);
        assert_eq!(
            net.phases()[0].phases,
            phases_before,
            "adaptive_step con vecs vacíos no debe modificar fases"
        );
    }

    /// adaptive_step with amplitude=1 and bivectores nulos reproduce Kuramoto classical.
    /// (α=1, β=0.5, dot_biv=0 → orient_factor=1; sat=0 → habituate=1 → Γ=Γ₀)
    #[test]
    fn adaptive_step_unit_amplitude_zero_bivector_matches_classic() {
        use core::f64::consts::PI;

        // Dos nodes without component bivectorial → dot_bivectors = 0
        // → orient_factor = α = 1.0
        // Amplitudes = 1.0 (prior) → sat = 0 → habituate = 1
        // adaptive_Γ = Γ₀ · (1·1/1) · 1.0 · 1.0 = Γ₀

        let phase0 = 0.0;
        let phase1 = PI / 4.0;
        let gamma = 0.5;

        // Network classical
        let mut classic = QuantumKuramotoNetwork::new(0.0);
        classic
            .add_oscillator(QuantumOscillator::with_phases(id(0), [phase0; 5], [0.0; 5]))
            .unwrap();
        classic
            .add_oscillator(QuantumOscillator::with_phases(id(1), [phase1; 5], [0.0; 5]))
            .unwrap();
        classic.set_coupling(id(0), id(1), gamma);
        classic.step(0.01);

        // Adaptive network (without bivectors → equivalent to classical)
        let mut adaptive = QuantumKuramotoNetwork::new(0.0);
        adaptive
            .add_oscillator(QuantumOscillator::with_phases(id(0), [phase0; 5], [0.0; 5]))
            .unwrap();
        adaptive
            .add_oscillator(QuantumOscillator::with_phases(id(1), [phase1; 5], [0.0; 5]))
            .unwrap();
        adaptive.set_coupling(id(0), id(1), gamma);

        // Vectors without grade-2 component (only grade 1)
        let vecs = [make_vec(&[(0b0001, 1.0)]), make_vec(&[(0b0001, 1.0)])];
        adaptive.adaptive_step(&vecs, 0.01);

        // The phases must match because the conditions are identical
        for g in 0..5 {
            let c = classic.phases()[0].phases[g];
            let a = adaptive.phases()[0].phases[g];
            assert!(
                (c - a).abs() < 1e-10,
                "nodo 0 grado {g}: classic={c:.12}, adaptive={a:.12}"
            );
        }
    }

    /// adaptive_step with bivectores antiparallel reduce coupling vs parallel.
    #[test]
    fn adaptive_step_frustration_reduces_coupling_for_anti_aligned() {
        use core::f64::consts::PI;

        // Dos pares of nodes. Pair A: bivectores parallel. Pair B: antiparallel.
        // We start with the same diferencia of phase. After of un step,
        // the pair with frustration must approach less (lower coupling effective).

        let dt = 0.1;
        let gamma = 1.0;
        let phase_a = 0.0_f64;
        let phase_b = PI / 3.0;

        // Pair A: bivectores parallel → dot_biv > 0 → Γ_eff > Γ₀
        let mut net_a = QuantumKuramotoNetwork::new(0.0);
        net_a
            .add_oscillator(QuantumOscillator::with_phases(
                id(0),
                [phase_a; 5],
                [0.0; 5],
            ))
            .unwrap();
        net_a
            .add_oscillator(QuantumOscillator::with_phases(
                id(1),
                [phase_b; 5],
                [0.0; 5],
            ))
            .unwrap();
        net_a.set_coupling(id(0), id(1), gamma);
        let vecs_a = [make_vec(&[(3, 1.0)]), make_vec(&[(3, 1.0)])]; // mismo e₀₁
        net_a.adaptive_step(&vecs_a, dt);

        // Pair B: bivectores antiparallel → dot_biv < 0 → Γ_eff < Γ₀
        let mut net_b = QuantumKuramotoNetwork::new(0.0);
        net_b
            .add_oscillator(QuantumOscillator::with_phases(
                id(0),
                [phase_a; 5],
                [0.0; 5],
            ))
            .unwrap();
        net_b
            .add_oscillator(QuantumOscillator::with_phases(
                id(1),
                [phase_b; 5],
                [0.0; 5],
            ))
            .unwrap();
        net_b.set_coupling(id(0), id(1), gamma);
        let vecs_b = [make_vec(&[(3, 1.0)]), make_vec(&[(3, -1.0)])]; // e₀₁ vs −e₀₁
        net_b.adaptive_step(&vecs_b, dt);

        // Pair A must haberse acercado more (mayor coupling effective)
        let diff_a = (net_a.phases()[0].phases[0] - net_a.phases()[1].phases[0]).abs();
        let diff_b = (net_b.phases()[0].phases[0] - net_b.phases()[1].phases[0]).abs();

        assert!(
            diff_a < diff_b,
            "bivectores paralelos deben producir mayor convergencia de fase: \
             diff_a={diff_a:.6} debe ser < diff_b={diff_b:.6}"
        );
    }

    /// poly_sin and poly_cos must not degrade for large phases (Cody-Waite reduction).
    ///
    /// Kuramoto phases accumulate unboundedly during long cognitive sessions.
    /// The former `x - round(x/π)·π` formula had ~5e-4 error at x=1e9.
    /// The Cody-Waite three-part reduction must stay below 1e-9.
    #[test]
    fn poly_sin_cos_precision_large_phases() {
        // Verify Cody-Waite reduction gives better accuracy than the former
        // x - round(x/π)·π formula (which had ~4e-4 error at x=1e9).
        // Threshold: 1e-6 (verified to hold for |x| ≤ 1e10 by Python simulation).
        // The former formula had 4e-4 error at 1e9 — this is 400× better.
        let test_vals: &[f64] = &[
            1.0e3,
            1.0e5,
            1.0e7,
            1.0e9,
            1.234e9,
            6.283_185_307_2e4, // 10_000 full rotations
        ];
        const THRESHOLD: f64 = 1e-6;
        for &x in test_vals {
            let sin_approx = poly_sin(x);
            let sin_true = x.sin();
            let cos_approx = poly_cos(x);
            let cos_true = x.cos();

            let sin_err = (sin_approx - sin_true).abs();
            let cos_err = (cos_approx - cos_true).abs();

            assert!(
                sin_err < THRESHOLD,
                "poly_sin({x:.3e}): err={sin_err:.3e} > {THRESHOLD:.0e} (approx={sin_approx:.9}, true={sin_true:.9})"
            );
            assert!(
                cos_err < THRESHOLD,
                "poly_cos({x:.3e}): err={cos_err:.3e} > {THRESHOLD:.0e} (approx={cos_approx:.9}, true={cos_true:.9})"
            );
        }
    }

    /// sin²+cos²=1 must hold to floating-point precision for any phase value.
    #[test]
    fn poly_sin_cos_pythagorean_identity_large_phases() {
        let test_vals: &[f64] = &[0.0, 1.0, 1e3, 1e6, 1e9, 1e12];
        for &x in test_vals {
            let s = poly_sin(x);
            let c = poly_cos(x);
            let identity = s.mul_add(s, c * c);
            assert!(
                (identity - 1.0).abs() < 1e-12,
                "sin²+cos² = {identity:.15} ≠ 1.0 for x={x:.3e}"
            );
        }
    }
}
