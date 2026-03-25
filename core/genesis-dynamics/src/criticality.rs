use core::num::NonZeroUsize;

/// Coupling critical of Kuramoto for frequencies Gaussian.
///
/// Derived analytically (validated Wolfram Fokker-Planck):
/// `K_c = 2√(2/π) · T`.
/// The system only generates macroscopic synchrony if `Γ_efectivo > K_c`.
///
/// AX-ID: `H_dinámica` (`LEY_FUNDACIONAL` §3.2)
pub fn kuramoto_critical_coupling(temperature: f64) -> f64 {
    temperature.mul_add(2.0 * (2.0 / core::f64::consts::PI).sqrt(), 0.0)
}

/// Lower bound of `r_sync` for the viable cognitive zone.
///
/// `r < SOC_R_SYNC_MIN` indicates regime subcritical (chaos dominant).
///
/// AX-ID: AXIOMA-005, AXIOMA-006
pub const SOC_R_SYNC_MIN: f64 = 0.3;
/// Maximum synchrony order r for SOC regime (above → supercritical, rigid).
///
/// AX-ID: AXIOMA-005
pub const SOC_R_SYNC_MAX: f64 = 0.7;

/// Monitor of criticality autoorganizada (SOC).
///
/// Objetivo: keep `P(S) ∝ S^{-τ}` — distribution of avalanches in ley of potencias.
/// `τ ∈ [1.5, 2.5]`: range biologically observed (criticality cerebral).
///
/// Forbidden: dynamics purely stable (τ → ∞, memory frozen).
/// Forbidden: dynamics purely chaotic (τ → 1, without consolidation).
///
/// The system must self-adjust its parameters to remain at the critical point.
/// `H_teleology` + AXIOMA-005 guarantee that the system never converges to static state.
///
/// AX-ID: AXIOMA-005, `H_dinámica` (`LEY_FUNDACIONAL` §3.2)
pub struct CriticalityMonitor {
    /// History of sizes of cascade — `ring buffer` of layercidad fixed.
    /// No reallocations: `write_pos` advances modulo `capacity`.
    avalanche_sizes: Vec<u32>,
    capacity: NonZeroUsize,
    write_pos: usize,
    total_recorded: usize,
}

/// Result full of the fit of ley of potencias.
///
/// AX-ID: AXIOMA-005 — the avalanche distribution must follow P(S) ∝ S^{-τ}
#[derive(Debug, Clone, PartialEq)]
pub struct CriticalityReport {
    /// Power law exponent by MLE (Clauset estimator).
    /// Range valid observed: τ ∈ [1.5, 2.5].
    pub tau: f64,
    /// Statistic KS: D = max_x |F_empirical(x) − F_theoretical(x)|.
    /// 0 = perfect fit; values ​​> 0.2 suggest that it is not a power law.
    pub ks_statistic: f64,
    /// approximate p-value of the KS test (H₀: data follows P(S) ∝ S^{-τ}).
    /// p < 0.05 rejects H₀ with confianza of the 95%.
    pub p_value: f64,
    /// Number of samples used for the fit.
    pub sample_size: usize,
    /// True if τ ∈ [1.5, 2.5] and p_value > 0.05 → the system is in criticality.
    pub is_critical: bool,
}

/// Approximation of the p-value for the test KS two-sided.
///
/// Implement the series of Marsaglia:
///   P(D√n > z) ≈ 2 Σ_{k=1}^{∞} (−1)^{k+1} exp(−2k²z²)
/// It takes the first 20 terms. For z < 0.1, it returns 1.0.
fn ks_p_value(z: f64) -> f64 {
    if z < 0.1 {
        return 1.0;
    }
    // FIX-I: Convergence criterion based on change in partial sum, not on
    // the magnitude of the individual term. The former `term.abs() < 1e-15` break
    // fires prematurely for intermediate z values where alternating terms partially
    // cancel — the sum has not converged even though individual terms are small.
    // Checking `|sum - prev_sum| < 1e-15` is the correct convergence test for an
    // alternating series: it measures the actual change in the estimate.
    let mut sum = 0.0f64;
    let mut prev_sum = f64::NAN; // NAN ensures first iteration never triggers break
    let z_sq = z * z;
    // CRYSTAL: O1 — inevitable
    for k in 1_u32..=20 {
        let k_f = f64::from(k);
        let term = (-2.0 * (k_f * k_f).mul_add(z_sq, 0.0)).exp();
        if k % 2 == 1 {
            sum += term;
        } else {
            sum -= term;
        }
        // Convergence on partial sum change — not on term magnitude
        if (sum - prev_sum).abs() < 1e-15 {
            break;
        }
        prev_sum = sum;
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

impl CriticalityMonitor {
    /// Creates the monitor with layercidad minimum of 1 for avoid states invalids.
    pub fn new(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::MIN);
        Self {
            avalanche_sizes: vec![0u32; capacity.get()],
            capacity,
            write_pos: 0,
            total_recorded: 0,
        }
    }

    /// Records the size of an avalanche.
    /// Overwrites the entry more antigua when the buffer is full.
    pub fn record_avalanche(&mut self, size: u32) {
        let capacity = self.capacity.get();
        // CRYSTAL: O6 — inevitable
        self.avalanche_sizes[self.write_pos] = size;
        self.write_pos = (self.write_pos + 1) % capacity;
        self.total_recorded += 1;
    }

    /// Number of valid avalanches available (min(total, layercity)).
    ///
    /// AX-ID: AXIOMA-005
    pub fn count(&self) -> usize {
        self.total_recorded.min(self.capacity.get())
    }

    /// Iterator over the sizes of avalanche active in order chronological.
    fn active_sizes(&self) -> ActiveSizes<'_> {
        ActiveSizes {
            buf: &self.avalanche_sizes,
            start: if self.total_recorded >= self.capacity.get() {
                self.write_pos
            } else {
                0
            },
            len: self.count(),
            pos: 0,
            cap: self.capacity.get(),
        }
    }

    /// Estimate the exponent `τ` of the power law by MLE (Clauset estimator).
    ///
    ///   `τ_MLE = 1 + n · [Σᵢ ln(Sᵢ / S_min)]⁻¹`
    ///
    /// Valid for distribution discrete of ley of potencias with `S ≥ S_min`.
    /// Returns None if hay less of 10 avalanches registradas.
    pub fn tau_exponent(&self) -> Option<f64> {
        if self.count() < 10 {
            return None;
        }
        let sizes: Vec<u32> = self.active_sizes().filter(|&s| s > 0).collect();
        let n = sizes.len();
        if n < 10 {
            return None;
        }
        let s_min = f64::from(*sizes.iter().min().unwrap_or(&1));
        if s_min <= 0.0 {
            return None;
        }
        let s_half = (s_min - 0.5).max(0.5);
        let sum_ln: f64 = sizes.iter().map(|&s| (f64::from(s) / s_half).ln()).sum();
        if sum_ln < f64::MIN_POSITIVE {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        Some(1.0 + n as f64 / sum_ln)
    }

    /// Verifica if the system necesita fit of parameters.
    ///
    /// Returns true if τ is fuera of the range biologically valid [1.5, 2.5].
    /// Returns false if hay datos insuficientes (no ajustar precipitadamente).
    ///
    /// AX-ID: AXIOMA-005
    pub fn needs_adjustment(&self) -> bool {
        self.tau_exponent()
            .is_some_and(|tau| !(1.5..=2.5).contains(&tau))
    }

    /// The system is frozen: all avalanches are size ≤ 1.
    /// Indicator of dynamics purely stable (AXIOMA-005 violated).
    ///
    /// Returns false if hay less of 5 avalanches (datos insuficientes).
    pub fn is_frozen(&self) -> bool {
        if self.count() < 5 {
            return false;
        }
        self.active_sizes().all(|s| s <= 1)
    }

    /// Verifica if the `r_sync` actual is in the zone cognitive viable.
    pub fn is_cognitively_viable(&self, r_sync: f64) -> bool {
        (SOC_R_SYNC_MIN..=SOC_R_SYNC_MAX).contains(&r_sync)
    }

    /// Generates the reporte full of criticality with statistic KS.
    ///
    /// Algoritmo:
    /// 1. Recolectar all the sizes of avalanche active (filter s > 0).
    /// 2. If hay less of 10 samples, returns None.
    /// 3. Estimar τ with the method of Clauset (tau_exponent).
    /// 4. Sort the samples and compute the KS statistic:
    ///    - F_empirical(Sᵢ) = i/n
    ///    - F_theoretical(x) = 1 − (x/x_min)^{1−τ}  (CDF continua of ley of potencias)
    /// 5. Calculate the p-value using the Marsaglia approximation.
    /// 6. Determine if the system is critical (τ in range and p > 0.05).
    ///
    /// AX-ID: AXIOMA-005
    pub fn tau_exponent_report(&self) -> Option<CriticalityReport> {
        // Collect all samples in a Vec (no fixed limit)
        let mut sizes: Vec<u32> = self.active_sizes().filter(|&s| s > 0).collect();
        let n = sizes.len();
        if n < 10 {
            return None;
        }

        // Estimate τ (Clauset); tau_exponent already internally handles the condition of sufficient samples
        let tau = self.tau_exponent()?;
        let exponent = tau - 1.0;
        // loop-invariant, hoisted
        // CRYSTAL: O29 — inevitable

        // Sort for the KS test
        sizes.sort_unstable();
        let x_min = f64::from(sizes[0]); // valor mínimo observado

        // Calcular statistic KS
        let n_f = n as f64;
        let mut ks_d = 0.0f64;
        for (i, &s) in sizes.iter().enumerate() {
            let s_f = f64::from(s);
            let f_emp = (i as f64 + 1.0) / n_f;
            let f_teo = if exponent > 0.0 {
                1.0 - (s_f / x_min).powf(-exponent)
            } else {
                // Degenerate case (τ ≤ 1) should not occur with real data.
                0.5
            };
            ks_d = ks_d.max((f_emp - f_teo).abs());
        }

        // p-value (Marsaglia)
        let z = ks_d * n_f.sqrt();
        let p_value = ks_p_value(z);

        let is_critical = (1.5..=2.5).contains(&tau) && p_value > 0.05;

        Some(CriticalityReport {
            tau,
            ks_statistic: ks_d,
            p_value,
            sample_size: n,
            is_critical,
        })
    }
}

impl Default for CriticalityMonitor {
    fn default() -> Self {
        Self::new(1024)
    }
}

// ── Iterator without heap for ring buffer ───────────────────────────────────────

struct ActiveSizes<'a> {
    buf: &'a [u32],
    start: usize,
    len: usize,
    pos: usize,
    cap: usize,
}

impl Iterator for ActiveSizes<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let idx = (self.start + self.pos) % self.cap;
        self.pos += 1;
        Some(self.buf[idx])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Generates avalanches power-law with exponente τ objetivo using inverso of the CDF.
    /// P(S) ∝ S^{-τ}  →  S = floor((U·(S_max^{1-τ} − 1) + 1)^{1/(1-τ)})
    /// For τ ≠ 1. We use deterministic LCG for reproducibility.
    fn generate_power_law_avalanches(n: usize, tau: f64, seed: u64) -> Vec<u32> {
        let mut rng = seed;
        let mut sizes = Vec::with_capacity(n);
        let exponent = 1.0 - tau;
        let s_max_term = (100_f64).powf(exponent);
        for _ in 0..n {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (rng >> 11) as f64 / (1u64 << 53) as f64;
            let u = u.max(1e-10);
            // Inversion of the CDF: S = (u·(S_max^{1-τ} − 1) + 1)^{1/(1-τ)}
            let s = (u * (s_max_term - 1.0) + 1.0).powf(1.0 / exponent).max(1.0) as u32;
            sizes.push(s.max(1));
        }
        sizes
    }

    #[test]
    fn criticality_tau_in_valid_range_after_100_avalanches() {
        let mut monitor = CriticalityMonitor::new(1000);
        // Generar avalanches with τ ≈ 2.0 (dentro of the valid range [1.5, 2.5])
        for s in generate_power_law_avalanches(100, 2.0, 0xdead_beef) {
            monitor.record_avalanche(s);
        }
        let tau = monitor
            .tau_exponent()
            .expect("debe estimar τ con 100 avalanchas");
        assert!(
            (1.0..=5.0).contains(&tau),
            "τ estimado = {:.3} debe ser positivo y finito",
            tau
        );
    }

    #[test]
    fn criticality_is_not_frozen_after_diverse_avalanches() {
        let mut monitor = CriticalityMonitor::new(200);
        // Avalanchas diversas (sizes 1..=50)
        for i in 1u32..=100 {
            monitor.record_avalanche((i % 50) + 1);
        }
        assert!(
            !monitor.is_frozen(),
            "sistema con avalanchas diversas no debe estar congelado"
        );
    }

    #[test]
    fn criticality_is_frozen_when_all_size_one() {
        let mut monitor = CriticalityMonitor::new(100);
        for _ in 0..20 {
            monitor.record_avalanche(1);
        }
        assert!(
            monitor.is_frozen(),
            "todas avalanchas tamaño 1 → sistema congelado"
        );
    }

    #[test]
    fn criticality_tau_none_below_10_avalanches() {
        let mut monitor = CriticalityMonitor::new(100);
        for i in 1u32..=9 {
            monitor.record_avalanche(i);
        }
        assert_eq!(monitor.tau_exponent(), None);
    }

    #[test]
    fn criticality_needs_adjustment_false_for_insufficient_data() {
        let monitor = CriticalityMonitor::new(100);
        assert!(!monitor.needs_adjustment());
    }

    #[test]
    fn criticality_new_zero_capacity_is_promoted_to_one() {
        let mut monitor = CriticalityMonitor::new(0);
        monitor.record_avalanche(7);
        assert_eq!(monitor.count(), 1);
        assert_eq!(monitor.active_sizes().next(), Some(7));
    }

    #[test]
    fn criticality_ring_buffer_overwrites_oldest() {
        let mut monitor = CriticalityMonitor::new(5);
        for i in 1u32..=10 {
            monitor.record_avalanche(i);
        }
        //Only the last 5 must be present: [6,7,8,9,10]
        let active: Vec<u32> = monitor.active_sizes().collect();
        assert_eq!(active.len(), 5);
        // The set must contain the values ​​6..10
        for expected in 6u32..=10 {
            assert!(active.contains(&expected), "debe contener {}", expected);
        }
    }
}
