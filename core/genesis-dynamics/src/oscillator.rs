#![allow(clippy::float_cmp, clippy::must_use_candidate)]

use genesis_types::NodeId;

/// Oscilador cuántico por nodo con amplitudes complejas por grado de Clifford.
///
/// # Estructura de estado
///
/// Cada nodo `i` tiene 5 grados de Clifford (0=escalar…4=pseudoescalar).
/// Para cada grado `g`, el estado cuántico complejo es:
///
/// ```text
/// ψ_{i,g} = amplitude[g] · e^{iφ_{i,g}}
/// ```
///
/// donde:
/// - `phases[g]` — ángulo en radianes, acumulado libremente (sin wrapping)
/// - `amplitudes[g]` — módulo ∈ [0.0, 1.0], certeza inferencial del oscilador
///
/// # Amplitudes: semántica y acoplamiento Fisher
///
/// `amplitudes[g]` inicializa en `1.0` para todos los grados.
/// Con esta inicialización, el comportamiento de Kuramoto es **idéntico al anterior**:
/// el parámetro de orden `r_sync` reproduce exactamente los valores previos.
///
/// Cuando `VFEMinimizer` actualiza un nodo, el sistema externo (CRATE-003 pipeline)
/// acopla la amplitud con `FisherInfo::trace` normalizado:
///
/// ```text
/// amplitude[g] = (FisherInfo::trace / TRACE_INITIAL).clamp(0.0, 1.0)
/// ```
///
/// - Nodo con **alta VFE** (alta sorpresa, dominio no aprendido):
///   `FisherInfo::trace ≈ 1.0` → `amplitude ≈ 1.0` → contribuye plenamente a `r_sync`
/// - Nodo con **baja VFE** (dominio saturado, AXIOMA-008):
///   `FisherInfo::trace → 0` → `amplitude → 0` → contribución a `r_sync` suprimida
///
/// Esto hace que `r_sync` mida **coherencia angular ponderada por certeza inferencial**,
/// no solo coherencia angular bruta. Un nodo que ha consolidado conocimiento reduce
/// su peso en la dinámica colectiva, análogo a la mielinización (AXIOMA-015).
///
/// # Preparación para CRATE-004
/// `DiscreteRicciFlow` puede leer `complex_state(g)` por nodo para calcular
/// la amplitud colectiva del cluster antes de aplicar curvatura Ollivier-Ricci.
/// Clusters con amplitud media alta → alta VFE → candidatos para colapso de wormhole.
///
/// # Retrocompatibilidad
/// `phases`, `frequencies`, `node_id`, `state`: campo, firma y semántica idénticos.
/// `new()` y `with_phases()`: añaden campo `amplitudes = [1.0; 5]` (sin cambio de comportamiento).
///
/// AX-ID: AXIOMA-006, AXIOMA-008, `H_dinámica` (LEY_FUNDACIONAL §3.2)
#[derive(Debug, Clone, Copy)]
pub struct QuantumOscillator {
    /// φ_{i,g} — fases por grado Clifford (g = 0..=4), en radianes.
    pub phases: [f64; 5],
    /// ω_{i,g} — frecuencias naturales (rad/s), una por grado.
    pub frequencies: [f64; 5],
    /// A_{i,g} — amplitud por grado Clifford (g = 0..=4), ∈ [0.0, 1.0].
    ///
    /// Inicializa en `1.0` para todos los grados (prior de máxima certeza /
    /// máxima contribución a `r_sync`).
    ///
    /// Decrece cuando `FisherInfo::trace` del nodo decrece (aprendizaje activo).
    /// Vuelve a subir si el nodo entra en zona de alta VFE (exploración nueva).
    ///
    /// El decaimiento es **responsabilidad del sistema externo** que llama
    /// `update_amplitude_from_fisher()` tras cada `VFEMinimizer::update()`.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008
    pub amplitudes: [f64; 5],
    /// Identificador del nodo en el manifold.
    pub node_id: NodeId,
    /// Estado de vida del oscilador (Active, Saturated, Pruned).
    pub state: OscillatorState,
}

impl QuantumOscillator {
    #[inline]
    const fn default_state() -> OscillatorState {
        OscillatorState::Active
    }

    /// Traza inicial de Fisher (prior no informativo).
    /// Usada para normalizar amplitudes: `amplitude = trace / FISHER_TRACE_INITIAL`.
    pub const FISHER_TRACE_INITIAL: f64 = 1.0;

    /// Constructor canónico: fases y amplitudes iniciales máximas (1.0), frecuencias dadas.
    ///
    /// Amplitudes `[1.0; 5]` ← prior de máxima certeza. Kuramoto se comporta
    /// exactamente igual que antes de esta extensión.
    #[inline]
    pub const fn new(node_id: NodeId, frequencies: [f64; 5]) -> Self {
        Self {
            phases: [0.0; 5],
            amplitudes: [1.0; 5],
            frequencies,
            node_id,
            state: Self::default_state(),
        }
    }

    /// Constructor con fases iniciales explícitas. Amplitudes inicializan en `1.0`.
    #[inline]
    pub const fn with_phases(node_id: NodeId, phases: [f64; 5], frequencies: [f64; 5]) -> Self {
        Self {
            phases,
            amplitudes: [1.0; 5],
            frequencies,
            node_id,
            state: Self::default_state(),
        }
    }

    /// Estado cuántico complejo del oscilador en el grado Clifford `g`.
    ///
    /// ```text
    /// ψ_{i,g} = amplitude[g] · (cos(φ_{i,g}), sin(φ_{i,g}))
    /// ```
    ///
    /// Retornado como `(re, im)` en lugar de `Complex64` para evitar
    /// dependencia de `num-complex` en el tipo público — los callers en
    /// CRATE-004 pueden construir `Complex64::new(re, im)` directamente.
    ///
    /// # Parámetro
    /// `g` ∈ [0, 4]. Panics en debug si `g > 4`; comportamiento indefinido en release.
    ///
    /// # Uso en CRATE-004
    /// ```
    /// use genesis_dynamics::QuantumOscillator;
    /// use genesis_types::NodeId;
    ///
    /// let node = NodeId::try_new(0).expect("NodeId válido");
    /// let osc = QuantumOscillator::new(node, [0.0; 5]);
    /// let (re, im) = osc.complex_state(0); // grado 0
    /// let cluster_amplitude = re.hypot(im);
    /// assert!(cluster_amplitude >= 0.0);
    /// ```
    ///
    /// AX-ID: AXIOMA-006, LEY_FUNDACIONAL §3.2
    #[inline]
    pub fn complex_state(&self, g: usize) -> (f64, f64) {
        debug_assert!(g < 5, "grado {g} fuera de rango [0,4]");
        let a = self.amplitudes[g];
        let phi = self.phases[g];
        (a * phi.cos(), a * phi.sin())
    }

    /// Amplitud total del oscilador: norma euclidiana del vector de amplitudes.
    ///
    /// ```text
    /// |A_i| = √(Σ_g amplitudes[g]²) / √5  ∈ [0.0, 1.0]
    /// ```
    ///
    /// Normalizado por √5 para que el máximo (todos grades = 1.0) sea 1.0.
    /// Usado en `r_sync` ponderado para detectar certeza inferencial global del nodo.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008
    #[inline]
    pub fn amplitude_norm(&self) -> f64 {
        let sq: f64 = self
            .amplitudes
            .iter()
            .fold(0.0, |acc, &a| a.mul_add(a, acc));
        (sq / 5.0_f64).sqrt()
    }

    /// Actualiza las amplitudes del oscilador desde la traza de Fisher actual.
    ///
    /// ```text
    /// amplitude[g] = (fisher_trace / FISHER_TRACE_INITIAL).clamp(0.0, 1.0)
    /// ```
    ///
    /// Todos los grados reciben la misma amplitud porque la traza de Fisher
    /// en `FisherInfo` es escalar (isótropa). Cuando CRATE-004 tenga métricas
    /// de Fisher por grado, este método puede extenderse para amplitudes por grado.
    ///
    /// # Llamada correcta
    /// Este método debe llamarse **inmediatamente después** de `VFEMinimizer::update()`
    /// para un nodo, pasando `vfe.fisher_info(id).trace` como argumento.
    ///
    /// # Retrocompatibilidad
    /// Con `fisher_trace = FISHER_TRACE_INITIAL = 1.0` (prior), todas las amplitudes
    /// permanecen en 1.0 y el comportamiento de Kuramoto es idéntico al anterior.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008, LEY_FUNDACIONAL §3.2
    #[inline]
    pub fn update_amplitude_from_fisher(&mut self, fisher_trace: f64) {
        let a = (fisher_trace / Self::FISHER_TRACE_INITIAL).clamp(0.0, 1.0);
        self.amplitudes = [a; 5];
    }

    /// Semantic saturation factor ∈ [0.0, 1.0].
    ///
    /// 0.0 = maximum uncertainty (amplitude ≈ 1.0, full coupling drive).
    /// 1.0 = fully saturated (amplitude ≈ 0.0, domain mastered).
    ///
    /// Used in adaptive Kuramoto coupling (semantic habituation):
    /// ```text
    /// adaptive_Γ = base_Γ · (1.0 − sat_i · sat_j)
    /// ```
    /// When both nodes have converged (`sat → 1`), their mutual coupling drops
    /// to zero — compute is redirected to uncertain, high-surprise pairs.
    ///
    /// AX-ID: AXIOMA-006, AXIOMA-008, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub fn saturation_factor(&self) -> f64 {
        1.0 - self.amplitude_norm()
    }

    /// Fase primaria del oscilador (grado 0, escalar).
    /// Usada para calcular `r_sync` en `synchrony.rs`.
    #[inline]
    pub const fn primary_phase(&self) -> f64 {
        self.phases[0]
    }

    /// Marca el oscilador como saturado. Idempotente si ya está saturado o podado.
    pub const fn mark_saturated(&mut self, now_ns: u64) {
        if matches!(self.state, OscillatorState::Active) {
            self.state = OscillatorState::Saturated { since_ns: now_ns };
        }
    }

    /// Marca el oscilador como podado. Solo desde Active o Saturated.
    pub const fn mark_pruned(&mut self, now_ns: u64) {
        if !matches!(self.state, OscillatorState::Pruned { .. }) {
            self.state = OscillatorState::Pruned { at_ns: now_ns };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_initializes_phases_to_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        let osc = QuantumOscillator::new(id, [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(osc.phases, [0.0; 5]);
        assert_eq!(osc.node_id, id);
    }

    #[test]
    fn new_initializes_amplitudes_to_one() {
        let id = NodeId::try_new(1).expect("NodeId válido por construcción");
        let osc = QuantumOscillator::new(id, [1.0; 5]);
        assert_eq!(
            osc.amplitudes, [1.0; 5],
            "amplitudes deben inicializar en 1.0 (prior de máxima certeza)"
        );
    }

    #[test]
    fn with_phases_preserves_all_fields() {
        let id = NodeId::try_new(7).expect("NodeId válido por construcción");
        let phases = [0.1, 0.2, 0.3, 0.4, 0.5];
        let freqs = [1.0, 1.1, 1.2, 1.3, 1.4];
        let osc = QuantumOscillator::with_phases(id, phases, freqs);
        assert_eq!(osc.phases, phases);
        assert_eq!(osc.frequencies, freqs);
        assert_eq!(osc.node_id.get(), 7);
        assert_eq!(
            osc.amplitudes, [1.0; 5],
            "with_phases también inicializa amplitudes en 1.0"
        );
    }

    #[test]
    fn primary_phase_returns_grade0() {
        let id = NodeId::try_new(0).expect("NodeId válido por construcción");
        let phases = [2.5, 0.1, 0.2, 0.3, 0.4];
        let osc = QuantumOscillator::with_phases(id, phases, [0.0; 5]);
        assert_eq!(osc.primary_phase(), 2.5);
    }

    // ── complex_state tests ───────────────────────────────────────────────────

    /// Con amplitud 1.0 y fase 0.0, complex_state debe ser (1.0, 0.0).
    #[test]
    fn complex_state_unit_amplitude_zero_phase() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let osc = QuantumOscillator::new(id, [0.0; 5]);
        let (re, im) = osc.complex_state(0);
        assert!(
            (re - 1.0).abs() < 1e-15,
            "re debe ser 1.0 con φ=0, got {re}"
        );
        assert!(im.abs() < 1e-15, "im debe ser 0.0 con φ=0, got {im}");
    }

    /// Con amplitud 0.0, complex_state debe ser (0.0, 0.0) para cualquier fase.
    #[test]
    fn complex_state_zero_amplitude_gives_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.amplitudes = [0.0; 5];
        for g in 0..5 {
            let (re, im) = osc.complex_state(g);
            assert_eq!(re, 0.0, "re debe ser 0 con amplitude=0 en grado {g}");
            assert_eq!(im, 0.0, "im debe ser 0 con amplitude=0 en grado {g}");
        }
    }

    /// Verifica que |complex_state(g)| = amplitude[g].
    #[test]
    fn complex_state_norm_equals_amplitude() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        let phases = [1.0, 2.0, 3.0, 4.0, 5.0];
        let amps = [1.0, 0.8, 0.6, 0.4, 0.2];
        osc.amplitudes = amps;
        for g in 0..5 {
            osc.phases[g] = phases[g];
            let (re, im) = osc.complex_state(g);
            let norm = re.hypot(im);
            assert!(
                (norm - amps[g]).abs() < 1e-14,
                "grado {g}: |ψ| = {norm}, amplitude = {}",
                amps[g]
            );
        }
    }

    // ── amplitude_norm tests ──────────────────────────────────────────────────

    /// Con amplitudes = [1.0; 5], amplitude_norm debe ser 1.0.
    #[test]
    fn amplitude_norm_all_ones_is_one() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let osc = QuantumOscillator::new(id, [0.0; 5]);
        assert!(
            (osc.amplitude_norm() - 1.0).abs() < 1e-14,
            "amplitude_norm con [1.0;5] debe ser 1.0"
        );
    }

    /// Con amplitudes = [0.0; 5], amplitude_norm debe ser 0.0.
    #[test]
    fn amplitude_norm_all_zeros_is_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.amplitudes = [0.0; 5];
        assert_eq!(osc.amplitude_norm(), 0.0);
    }

    /// amplitude_norm está en [0,1] para amplitudes en [0,1].
    #[test]
    fn amplitude_norm_bounded_in_unit_interval() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        // Probar varios valores intermedios
        for v in [0.0, 0.2, 0.5, 0.7, 1.0] {
            osc.amplitudes = [v; 5];
            let n = osc.amplitude_norm();
            assert!(
                (0.0..=1.0 + 1e-14).contains(&n),
                "amplitude_norm = {n} fuera de [0,1] para amplitude = {v}"
            );
        }
    }

    // ── update_amplitude_from_fisher tests ────────────────────────────────────

    /// Con fisher_trace = FISHER_TRACE_INITIAL, amplitudes no cambian de 1.0.
    #[test]
    fn update_amplitude_fisher_prior_stays_one() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.update_amplitude_from_fisher(QuantumOscillator::FISHER_TRACE_INITIAL);
        assert_eq!(
            osc.amplitudes, [1.0; 5],
            "fisher_trace = INITIAL → amplitudes deben permanecer en 1.0"
        );
    }

    /// Con fisher_trace = 0.0 (dominio completamente saturado), amplitudes → 0.0.
    #[test]
    fn update_amplitude_fisher_saturated_gives_zero() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);
        osc.update_amplitude_from_fisher(0.0);
        assert_eq!(
            osc.amplitudes, [0.0; 5],
            "fisher_trace = 0 → amplitudes deben ser 0.0"
        );
    }

    /// La amplitud está clampeada en [0,1] incluso con valores fuera de rango.
    #[test]
    fn update_amplitude_clamps_to_unit_interval() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);

        osc.update_amplitude_from_fisher(2.0); // > INITIAL
        for &a in &osc.amplitudes {
            assert!(a <= 1.0, "amplitude {a} debe ser ≤ 1.0");
        }

        osc.update_amplitude_from_fisher(-1.0); // negativo
        for &a in &osc.amplitudes {
            assert!(a >= 0.0, "amplitude {a} debe ser ≥ 0.0");
        }
    }

    /// Verifica comportamiento monotónico: más aprendizaje → menor amplitud.
    #[test]
    fn update_amplitude_monotone_with_fisher_trace() {
        let id = NodeId::try_new(0).expect("NodeId válido");
        let mut osc = QuantumOscillator::new(id, [0.0; 5]);

        // trace alto (aprendizaje inicial) → amplitud alta
        osc.update_amplitude_from_fisher(1.0);
        let a_high = osc.amplitudes[0];

        // trace medio → amplitud media
        osc.update_amplitude_from_fisher(0.5);
        let a_mid = osc.amplitudes[0];

        // trace bajo (dominio saturado) → amplitud baja
        osc.update_amplitude_from_fisher(0.1);
        let a_low = osc.amplitudes[0];

        assert!(
            a_high >= a_mid && a_mid >= a_low,
            "amplitudes deben ser monótonas con fisher_trace: {a_high} ≥ {a_mid} ≥ {a_low}"
        );
    }
}

/// Estado de vida de un oscilador cuántico.
///
/// La transición es unidireccional: Active → Saturated → Pruned.
/// Un oscilador Pruned no participa en el cálculo de Kuramoto ni en VFE.
///
/// AX-ID: AXIOMA-008 (Saturated), AXIOMA-016 (Pruned)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OscillatorState {
    /// Oscilador activo: participa en Kuramoto y VFE.
    #[default]
    Active,
    /// Fisher saturado: ΔG < ε durante FISHER_SATIATION_WINDOW iteraciones.
    /// El oscilador no acepta nuevos inputs de aprendizaje pero sigue
    /// contribuyendo al parámetro de orden r_sync.
    /// `since_ns`: timestamp en nanosegundos del momento de saturación.
    Saturated {
        /// Timestamp in nanoseconds when the oscillator entered the Saturated state.
        since_ns: u64,
    },
    /// Podado por la ecuación de calor (AXIOMA-016).
    /// El oscilador es inactivo: sus fases no se actualizan y no contribuye a Ω.
    /// `at_ns`: timestamp del momento de poda.
    Pruned {
        /// Timestamp in nanoseconds when the oscillator was pruned.
        at_ns: u64,
    },
}

impl OscillatorState {
    /// Retorna true si el oscilador puede recibir inputs de aprendizaje.
    #[inline]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Retorna true si el oscilador contribuye al parámetro de orden r_sync.
    /// Los saturados SÍ contribuyen; los podados NO.
    #[inline]
    pub const fn contributes_to_sync(self) -> bool {
        !matches!(self, Self::Pruned { .. })
    }
}
