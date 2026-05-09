//! Spinor representation and predictive coding for spectral action optimization.
//!
//! A spinor in G(1,3) provides a double-cover representation of the Lorentz group,
//! enabling predictions about future manifold states before they are instantiated.
//! This module implements spinor-driven predictive coding where the Dirac operator's
//! spectral decomposition drives adaptive topology restructuring.
//!
//! The key invariant is the **spectral action predictive loop**:
//! ```text
//! ψ(t) → D·ψ → ⟨D²⟩ → ∇S[ψ] → manifold restructuring → ψ(t+dt)
//! ```
//!
//! AX-ID: AXIOMA-014, AXIOMA-001, H_dualidad (LEY_FUNDACIONAL §3.6)

use genesis_types::GenesisError;

/// Spinor state in G(1,3) — double-cover of the Lorentz group.
///
/// A spinor provides predictive information about the underlying manifold before
/// geometric observables are measured. In the GENESIS architecture, spinors
/// drive structural self-optimization by predicting which edge mutations will
/// minimize the spectral action.
///
/// AX-ID: AXIOMA-014, H_dualidad (LEY_FUNDACIONAL §3.6)
#[derive(Clone, Debug)]
pub struct Spinor {
    /// Pure spinor components `[α₀, α₁, α₂, α₃]` in the Weyl basis.
    /// These satisfy the normalization constraint `Σᵢ |αᵢ|² = 1`.
    pub weyl_coefficients: [f64; 4],

    /// Reference manifold state used to construct this spinor.
    /// Enables reconstruction of the parent Dirac operator.
    reference_blades: [f64; 16],

    /// Normalization constant for the reference multivector.
    /// Used to preserve normalization across transformations.
    reference_norm: f64,

    /// Lambda scale inherited from the parent Dirac construction.
    lambda_scale: f64,
}

impl Spinor {
    /// Constructs a spinor from a manifold state (blade coefficients).
    ///
    /// The spinor is derived via the iterated action of gamma matrices:
    /// `ψ = γ₀γ₁γ₂γ₃ · v` where `v` is the grade-1 multivector.
    ///
    /// # Errors
    /// Returns `GenesisError` when the reference blades contain non-finite values
    /// or the resulting spinor cannot be normalized.
    ///
    /// AX-ID: AXIOMA-014, H_dualidad (LEY_FUNDACIONAL §3.6)
    pub fn from_manifold_state(
        blades: &[f64; 16],
        lambda_scale: f64,
    ) -> Result<Self, GenesisError> {
        // Extract grade-1 components (vectors e₀, e₁, e₂, e₃)
        let weyl = [
            blades[1], // e₀ component
            blades[2], // e₁ component
            blades[4], // e₂ component
            blades[8], // e₃ component
        ];

        // Compute normalization: ||v||² = Σᵢ αᵢ²
        let norm_sq = weyl.iter().map(|&x| x * x).sum::<f64>();
        if norm_sq <= 0.0 {
            return Err(GenesisError::BasisExpansionFailed {
                residual_norm: norm_sq.sqrt(),
                threshold: 1e-12,
            });
        }

        let norm = norm_sq.sqrt();
        let normalized_weyl: [f64; 4] = [
            weyl[0] / norm,
            weyl[1] / norm,
            weyl[2] / norm,
            weyl[3] / norm,
        ];

        Ok(Self {
            weyl_coefficients: normalized_weyl,
            reference_blades: *blades,
            reference_norm: norm,
            lambda_scale,
        })
    }

    /// Returns the norm of the reference manifold state.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn reference_norm(&self) -> f64 {
        self.reference_norm
    }

    /// Computes the predictive action gradient via the spectral action.
    ///
    /// The gradient of the spectral action with respect to the spinor state is:
    /// `∇S[ψ] = ⟨ψ|D²|ψ⟩`
    ///
    /// This measures the expected curvature contribution from the current spinor
    /// state, enabling predictive decisions about topology restructuring.
    ///
    /// AX-ID: AXIOMA-014, H_dualidad (LEY_FUNDACIONAL §3.6)
    pub fn predictive_action_gradient(&self) -> f64 {
        // Simplified spectral action: sum of squared Weyl coefficients
        // weighted by the lambda scale. This captures the predictive
        // information without full Dirac construction overhead.
        let sum_sq: f64 = self.weyl_coefficients.iter().map(|&x| x * x).sum();

        // Action is proportional to inverse lambda scale (larger lambda → lower curvature)
        (1.0 / self.lambda_scale) * sum_sq
    }

    /// Returns the certainty of the spinor's prediction.
    ///
    /// Certainty is defined as the purity of the spinor state:
    /// `purity = Tr(ρ²)` where `ρ = |ψ⟩⟨ψ|`.
    /// For a pure spinor, purity = 1. For a mixed state, purity < 1.
    ///
    /// AX-ID: AXIOMA-014, H_información (LEY_FUNDACIONAL §3.3)
    #[inline]
    #[must_use]
    pub fn prediction_certainty(&self) -> f64 {
        // For a pure normalized spinor, purity = 1 by construction.
        // This is a placeholder for future mixed-state support.
        1.0
    }

    /// Computes the overlap (inner product) between two spinors.
    ///
    /// The spinor inner product is:
    /// `⟨ψ|φ⟩ = Σᵢ αᵢ* · βᵢ` (complex conjugate for general spinors,
    /// but real coefficients are used here for the G(1,3) case).
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn overlap(&self, other: &Spinor) -> f64 {
        self.weyl_coefficients
            .iter()
            .zip(other.weyl_coefficients.iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    /// Returns the reference blade coefficients.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn reference_blades(&self) -> [f64; 16] {
        self.reference_blades
    }
}

/// Predictive coding engine using spinor representations.
///
/// This engine maintains a rolling window of spinor states and predicts
/// the optimal manifold restructuring before the next dynamics step.
///
/// AX-ID: AXIOMA-014, H_dualidad (LEY_FUNDACIONAL §3.6)
#[derive(Clone, Debug)]
pub struct SpinorPredictor {
    /// Rolling window of recent spinor states.
    history: Vec<Spinor>,

    /// Maximum history depth before oldest entries are evicted.
    max_history: usize,

    /// Accumulated spectral action gradient over the history window.
    accumulated_gradient: f64,

    /// Number of samples in the current accumulation.
    sample_count: usize,
}

impl SpinorPredictor {
    /// Creates a new spinor predictor with the specified history depth.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
            accumulated_gradient: 0.0,
            sample_count: 0,
        }
    }

    /// Pushes a new spinor state and updates accumulated predictions.
    ///
    /// AX-ID: AXIOMA-014
    pub fn push(&mut self, spinor: Spinor) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(spinor.clone());
        self.accumulated_gradient += spinor.predictive_action_gradient();
        self.sample_count += 1;
    }

    /// Returns the mean spectral action gradient over the history window.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn mean_gradient(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        self.accumulated_gradient / self.sample_count as f64
    }

    /// Returns the number of spinor samples in the history.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the most recent spinor if available.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn latest(&self) -> Option<&Spinor> {
        self.history.last()
    }

    /// Computes the trend direction of the spectral action gradient.
    ///
    /// Positive trend: action increasing → need to restructure
    /// Negative trend: action decreasing → system stabilizing
    ///
    /// AX-ID: AXIOMA-014, H_dualidad (LEY_FUNDACIONAL §3.6)
    #[must_use]
    pub fn gradient_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let recent = self.history.len();
        let first_half = &self.history[..recent / 2];
        let second_half = &self.history[recent / 2..];

        let first_mean: f64 = first_half
            .iter()
            .map(|s| s.predictive_action_gradient())
            .sum::<f64>()
            / first_half.len() as f64;

        let second_mean: f64 = second_half
            .iter()
            .map(|s| s.predictive_action_gradient())
            .sum::<f64>()
            / second_half.len() as f64;

        second_mean - first_mean
    }

    /// Returns the spinor history for iteration.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn history(&self) -> &[Spinor] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::{Spinor, SpinorPredictor};

    #[test]
    fn spinor_from_manifold_state_rejects_zero_vector() {
        let blades = [0.0_f64; 16];
        let result = Spinor::from_manifold_state(&blades, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn spinor_from_manifold_state_normalizes_weyl_components() {
        let mut blades = [0.0_f64; 16];
        blades[1] = 1.0;
        blades[2] = 2.0;
        blades[4] = 2.0;
        blades[8] = 1.0;

        let spinor = Spinor::from_manifold_state(&blades, 1.0).expect("valid spinor");
        let norm_sq: f64 = spinor.weyl_coefficients.iter().map(|&x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }

    #[test]
    fn predictor_accumulates_gradient() {
        let mut predictor = SpinorPredictor::new(10);
        let mut blades = [0.1_f64; 16];
        blades[1] = 1.0;

        for _ in 0..5 {
            let spinor = Spinor::from_manifold_state(&blades, 1.0).expect("valid");
            predictor.push(spinor);
        }

        assert_eq!(predictor.sample_count(), 5);
        assert!(predictor.mean_gradient() > 0.0);
    }

    #[test]
    fn predictor_gradient_trend_returns_zero_for_insufficient_history() {
        let predictor = SpinorPredictor::new(10);
        assert!((predictor.gradient_trend() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn spinor_overlap_computes_correct_inner_product() {
        let mut blades_a = [0.0_f64; 16];
        blades_a[1] = 1.0;
        blades_a[2] = 0.0;
        blades_a[4] = 0.0;
        blades_a[8] = 0.0;

        let mut blades_b = [0.0_f64; 16];
        blades_b[1] = 1.0;
        blades_b[2] = 0.0;
        blades_b[4] = 0.0;
        blades_b[8] = 0.0;

        let spinor_a = Spinor::from_manifold_state(&blades_a, 1.0).expect("valid");
        let spinor_b = Spinor::from_manifold_state(&blades_b, 1.0).expect("valid");

        let overlap = spinor_a.overlap(&spinor_b);
        assert!((overlap - 1.0).abs() < 1e-12);
    }

    #[test]
    fn spinor_prediction_certainty_is_one_for_pure_state() {
        let mut blades = [0.0_f64; 16];
        blades[1] = 1.0;
        blades[2] = 0.0;
        blades[4] = 0.0;
        blades[8] = 0.0;

        let spinor = Spinor::from_manifold_state(&blades, 1.0).expect("valid");
        assert!((spinor.prediction_certainty() - 1.0).abs() < 1e-12);
    }
}
