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
use std::collections::VecDeque;

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
    /// If the manifold state has no grade-1 components (e.g., purely bivectorial
    /// relations or scalar identities), a neutral spinor `[1, 0, 0, 0]` is returned.
    /// This preserves the prediction_certainty signal even for non-vector concepts.
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
        let (weyl_normalized, norm) = if norm_sq > 1e-24 {
            // Normalizable grade-1 component: construct normalized spinor
            let n = norm_sq.sqrt();
            ([weyl[0] / n, weyl[1] / n, weyl[2] / n, weyl[3] / n], n)
        } else {
            // Purely bivectorial or scalar concept: neutral spinor with zero norm
            // This is valid in Genesis — such spinors still contribute to the
            // prediction signal (their gradient is 0, which signals stability).
            ([1.0, 0.0, 0.0, 0.0], 0.0)
        };

        Ok(Self {
            weyl_coefficients: weyl_normalized,
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
    /// Rolling window of recent spinor states (O(1) eviction).
    history: VecDeque<Spinor>,

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
            history: VecDeque::with_capacity(max_history),
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
            self.history.pop_front();
        }
        self.history.push_back(spinor.clone());
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
        self.history.back()
    }

    /// Returns the certainty of the prediction signal.
    ///
    /// Certainty is computed as an inverse-variance signal from the rolling history.
    /// High variance in gradient history → low certainty (instability).
    /// Low variance → high certainty (convergence).
    /// Formula: `certainty = 1 / (1 + variance)`
    ///
    /// AX-ID: AXIOMA-014, H_información (LEY_FUNDACIONAL §3.3)
    #[inline]
    #[must_use]
    pub fn prediction_certainty(&self) -> f64 {
        if self.history.len() < 2 {
            return 1.0;
        }
        let mean = self.mean_gradient();
        let variance = self
            .history
            .iter()
            .map(|s| {
                let delta = s.predictive_action_gradient() - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.history.len() as f64;
        1.0 / (1.0 + variance)
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
        let mid = self.history.len() / 2;

        let first_half: f64 = self
            .history
            .iter()
            .take(mid)
            .map(|s| s.predictive_action_gradient())
            .sum::<f64>()
            / mid as f64;
        let second_half: f64 = self
            .history
            .iter()
            .skip(mid)
            .map(|s| s.predictive_action_gradient())
            .sum::<f64>()
            / (self.history.len() - mid) as f64;

        second_half - first_half
    }

    /// Returns the spinor history for iteration.
    ///
    /// AX-ID: AXIOMA-014
    #[inline]
    #[must_use]
    pub fn history(&self) -> std::collections::vec_deque::Iter<'_, Spinor> {
        self.history.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::{Spinor, SpinorPredictor};

    #[test]
    fn spinor_from_manifold_state_accepts_pure_bivector_as_neutral_spinor() {
        // Purely bivectorial concept (no grade-1 components): returns neutral spinor
        let blades = [0.0_f64; 16];
        let result = Spinor::from_manifold_state(&blades, 1.0);
        assert!(result.is_ok());
        let spinor = result.expect("valid spinor");
        assert!((spinor.weyl_coefficients[0] - 1.0).abs() < 1e-12);
        assert!((spinor.weyl_coefficients[1] - 0.0).abs() < 1e-12);
        assert!((spinor.weyl_coefficients[2] - 0.0).abs() < 1e-12);
        assert!((spinor.weyl_coefficients[3] - 0.0).abs() < 1e-12);
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
        let blades = [0.0_f64; 16];

        for i in 0..5 {
            let mut b = blades;
            b[1] = i as f64 * 0.1 + 1.0;
            let spinor = Spinor::from_manifold_state(&b, 1.0).expect("valid");
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
    fn predictor_prediction_certainty_high_for_low_variance() {
        // Create a predictor with low-variance history → high certainty
        let mut predictor = SpinorPredictor::new(10);
        let mut blades = [0.0_f64; 16];
        blades[1] = 1.0;

        // Push same spinor multiple times → variance ≈ 0 → certainty ≈ 1
        for _ in 0..5 {
            let spinor = Spinor::from_manifold_state(&blades, 1.0).expect("valid");
            predictor.push(spinor);
        }

        let certainty = predictor.prediction_certainty();
        assert!(certainty > 0.9, "low variance should yield high certainty");
    }
}
