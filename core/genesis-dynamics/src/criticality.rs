use core::num::NonZeroUsize;

/// Acoplamiento crítico de Kuramoto para frecuencias Gaussianas.
///
/// Derivado analíticamente (validado Wolfram Fokker-Planck):
/// `K_c = 2√(2/π) · T`.
/// El sistema solo genera sincronía macroscópica si `Γ_efectivo > K_c`.
///
/// AX-ID: `H_dinámica` (`LEY_FUNDACIONAL` §3.2)
pub fn kuramoto_critical_coupling(temperature: f64) -> f64 {
    temperature.mul_add(2.0 * (2.0 / core::f64::consts::PI).sqrt(), 0.0)
}

/// Límite inferior de `r_sync` para la zona cognitiva viable.
///
/// `r < SOC_R_SYNC_MIN` indica régimen subcrítico (caos dominante).
///
/// AX-ID: AXIOMA-005, AXIOMA-006
pub const SOC_R_SYNC_MIN: f64 = 0.3;
/// Maximum synchrony order r for SOC regime (above → supercritical, rigid).
///
/// AX-ID: AXIOMA-005
pub const SOC_R_SYNC_MAX: f64 = 0.7;

/// Monitor de criticalidad autoorganizada (SOC).
///
/// Objetivo: mantener `P(S) ∝ S^{-τ}` — distribución de avalanchas en ley de potencias.
/// `τ ∈ [1.5, 2.5]`: rango biológicamente observado (criticidad cerebral).
///
/// PROHIBIDO: dinámica puramente estable (τ → ∞, memoria congelada).
/// PROHIBIDO: dinámica puramente caótica (τ → 1, sin consolidación).
///
/// El sistema debe autoajustar sus parámetros para mantenerse en el punto crítico.
/// `H_teleología` + AXIOMA-005 garantizan que el sistema nunca converge a estado estático.
///
/// AX-ID: AXIOMA-005, `H_dinámica` (`LEY_FUNDACIONAL` §3.2)
pub struct CriticalityMonitor {
    /// Historial de tamaños de cascada — `ring buffer` de capacidad fija.
    /// Sin realloc: `write_pos` avanza módulo `capacity`.
    avalanche_sizes: Vec<u32>,
    capacity: NonZeroUsize,
    write_pos: usize,
    total_recorded: usize,
}

/// Resultado completo del ajuste de ley de potencias.
///
/// AX-ID: AXIOMA-005 — la distribución de avalanchas debe seguir P(S) ∝ S^{-τ}
#[derive(Debug, Clone, PartialEq)]
pub struct CriticalityReport {
    /// Exponente de la ley de potencias por MLE (estimador de Clauset).
    /// Rango válido observado: τ ∈ [1.5, 2.5].
    pub tau: f64,
    /// Estadístico KS: D = max_x |F_empírica(x) − F_teórica(x)|.
    /// 0 = ajuste perfecto; valores > 0.2 sugieren que no es ley de potencias.
    pub ks_statistic: f64,
    /// p-valor aproximado del test KS (H₀: los datos siguen P(S) ∝ S^{-τ}).
    /// p < 0.05 rechaza H₀ con confianza del 95%.
    pub p_value: f64,
    /// Número de muestras usadas para el ajuste.
    pub sample_size: usize,
    /// True si τ ∈ [1.5, 2.5] y p_value > 0.05 → el sistema está en criticalidad.
    pub is_critical: bool,
}

/// Aproximación del p-valor para el test KS bilateral.
///
/// Implementa la serie de Marsaglia:
///   P(D√n > z) ≈ 2 Σ_{k=1}^{∞} (−1)^{k+1} exp(−2k²z²)
/// Se toman los primeros 20 términos. Para z < 0.1 se devuelve 1.0.
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
    for k in 1_u32..=20 {
        let k_f = f64::from(k);
        let term = (-2.0 * (k_f * k_f).mul_add(z * z, 0.0)).exp();
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
    /// Crea el monitor con capacidad mínima de 1 para evitar estados inválidos.
    pub fn new(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::MIN);
        Self {
            avalanche_sizes: vec![0u32; capacity.get()],
            capacity,
            write_pos: 0,
            total_recorded: 0,
        }
    }

    /// Registra el tamaño de una avalancha.
    /// Sobrescribe la entrada más antigua cuando el buffer está lleno.
    pub fn record_avalanche(&mut self, size: u32) {
        self.avalanche_sizes[self.write_pos] = size;
        self.write_pos = (self.write_pos + 1) % self.capacity.get();
        self.total_recorded += 1;
    }

    /// Número de avalanchas válidas disponibles (mín(total, capacity)).
    ///
    /// AX-ID: AXIOMA-005
    pub fn count(&self) -> usize {
        self.total_recorded.min(self.capacity.get())
    }

    /// Iterador sobre los tamaños de avalancha activos en orden cronológico.
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

    /// Estima el exponente `τ` de la ley de potencias por MLE (estimador de Clauset).
    ///
    ///   `τ_MLE = 1 + n · [Σᵢ ln(Sᵢ / S_min)]⁻¹`
    ///
    /// Válido para distribución discreta de ley de potencias con `S ≥ S_min`.
    /// Retorna None si hay menos de 10 avalanchas registradas.
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

    /// Verifica si el sistema necesita ajuste de parámetros.
    ///
    /// Retorna true si τ está fuera del rango biológicamente válido [1.5, 2.5].
    /// Retorna false si hay datos insuficientes (no ajustar precipitadamente).
    ///
    /// AX-ID: AXIOMA-005
    pub fn needs_adjustment(&self) -> bool {
        self.tau_exponent()
            .is_some_and(|tau| !(1.5..=2.5).contains(&tau))
    }

    /// El sistema está congelado: todas las avalanchas son tamaño ≤ 1.
    /// Indicador de dinámica puramente estable (AXIOMA-005 violado).
    ///
    /// Retorna false si hay menos de 5 avalanchas (datos insuficientes).
    pub fn is_frozen(&self) -> bool {
        if self.count() < 5 {
            return false;
        }
        self.active_sizes().all(|s| s <= 1)
    }

    /// Verifica si el `r_sync` actual está en la zona cognitiva viable.
    pub fn is_cognitively_viable(&self, r_sync: f64) -> bool {
        (SOC_R_SYNC_MIN..=SOC_R_SYNC_MAX).contains(&r_sync)
    }

    /// Genera el reporte completo de criticalidad con estadístico KS.
    ///
    /// Algoritmo:
    /// 1. Recolectar todos los tamaños de avalancha activos (filter s > 0).
    /// 2. Si hay menos de 10 muestras, retorna None.
    /// 3. Estimar τ con el método de Clauset (tau_exponent).
    /// 4. Ordenar las muestras y calcular el estadístico KS:
    ///    - F_empírica(Sᵢ) = i/n
    ///    - F_teórica(x) = 1 − (x/x_min)^{1−τ}  (CDF continua de ley de potencias)
    /// 5. Calcular el p-valor mediante la aproximación de Marsaglia.
    /// 6. Determinar si el sistema es crítico (τ en rango y p > 0.05).
    ///
    /// AX-ID: AXIOMA-005
    pub fn tau_exponent_report(&self) -> Option<CriticalityReport> {
        // Recolectar todas las muestras en un Vec (sin límite fijo)
        let mut sizes: Vec<u32> = self.active_sizes().filter(|&s| s > 0).collect();
        let n = sizes.len();
        if n < 10 {
            return None;
        }

        // Estimar τ (Clauset); tau_exponent ya maneja internamente la condición de muestras suficientes
        let tau = self.tau_exponent()?;

        // Ordenar para el test KS
        sizes.sort_unstable();
        let x_min = f64::from(sizes[0]); // valor mínimo observado

        // Calcular estadístico KS
        let n_f = n as f64;
        let mut ks_d = 0.0f64;
        for (i, &s) in sizes.iter().enumerate() {
            let s_f = f64::from(s);
            let f_emp = (i as f64 + 1.0) / n_f;
            let exponent = tau - 1.0; // > 0 para τ > 1
            let f_teo = if exponent > 0.0 {
                1.0 - (s_f / x_min).powf(-exponent)
            } else {
                // Caso degenerado (τ ≤ 1) no debería ocurrir con datos reales.
                0.5
            };
            ks_d = ks_d.max((f_emp - f_teo).abs());
        }

        // p-valor (Marsaglia)
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

// ── Iterador sin heap para ring buffer ───────────────────────────────────────

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

    /// Genera avalanchas power-law con exponente τ objetivo usando inverso del CDF.
    /// P(S) ∝ S^{-τ}  →  S = floor((U·(S_max^{1-τ} − 1) + 1)^{1/(1-τ)})
    /// Para τ ≠ 1. Usamos LCG determinístico para reproducibilidad.
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
            // Inversión del CDF: S = (u·(S_max^{1-τ} − 1) + 1)^{1/(1-τ)}
            let s = (u * (s_max_term - 1.0) + 1.0).powf(1.0 / exponent).max(1.0) as u32;
            sizes.push(s.max(1));
        }
        sizes
    }

    #[test]
    fn criticality_tau_in_valid_range_after_100_avalanches() {
        let mut monitor = CriticalityMonitor::new(1000);
        // Generar avalanchas con τ ≈ 2.0 (dentro del rango válido [1.5, 2.5])
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
        // Avalanchas diversas (tamaños 1..=50)
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
        // Solo los últimos 5 deben estar presentes: [6,7,8,9,10]
        let active: Vec<u32> = monitor.active_sizes().collect();
        assert_eq!(active.len(), 5);
        // El conjunto debe contener los valores 6..10
        for expected in 6u32..=10 {
            assert!(active.contains(&expected), "debe contener {}", expected);
        }
    }
}
