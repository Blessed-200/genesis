use genesis_types::{GenesisError, METRIC_WEIGHTS};

/// Probability distribution over the 16 blades of G(1,3).
///
/// Invariants:
/// - `weights.iter().sum() == 1.0`
/// - `weights[i] >= 0.0` for all i.
///
/// AX-ID: AXIOMA-001, AXIOMA-003, H_información
#[derive(Clone, Debug)]
pub struct BeliefDistribution {
    /// Normalized blade probabilities.
    pub weights: [f64; 16],
    entropy: f64,
    weighted_mean: f64,
}

impl BeliefDistribution {
    /// Builds a normalized belief from raw non-negative weights.
    ///
    /// AX-ID: AXIOMA-003
    pub fn from_weights(weights: [f64; 16]) -> Result<Self, GenesisError> {
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(GenesisError::InvalidInput("non-finite belief weight"));
        }
        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(GenesisError::InvalidInput("all weights are non-positive"));
        }
        let mut normalized = weights;
        for w in &mut normalized {
            if *w < 0.0 {
                return Err(GenesisError::InvalidInput("negative belief weight"));
            }
            *w /= sum;
        }
        let entropy = normalized
            .iter()
            .zip(METRIC_WEIGHTS.iter())
            .filter(|(w, _)| **w > 0.0)
            .map(|(w, mw)| -w * (w / mw).ln())
            .sum();
        let weighted_mean = normalized.iter().zip(METRIC_WEIGHTS.iter()).map(|(w, mw)| w * mw).sum();
        Ok(Self { weights: normalized, entropy, weighted_mean })
    }

    /// Returns a uniform distribution over all blades.
    ///
    /// AX-ID: AXIOMA-003
    #[must_use]
    pub fn uniform() -> Self { Self::from_weights([1.0 / 16.0; 16]).expect("uniform distribution is valid") }

    /// Returns a point mass at blade `k`.
    ///
    /// AX-ID: AXIOMA-001
    pub fn point_mass(k: usize) -> Result<Self, GenesisError> {
        if k >= 16 { return Err(GenesisError::BladeIndexOutOfRange { index: k }); }
        let mut w = [0.0; 16];
        w[k] = 1.0;
        Self::from_weights(w)
    }

    /// Computes `D_KL(self || other)` with metric-aware weighting.
    ///
    /// AX-ID: AXIOMA-003, H_información
    #[must_use]
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        self.weights.iter().zip(other.weights.iter()).zip(METRIC_WEIGHTS.iter()).filter(|((p, _), _)| **p > 1e-30).map(|((p, q), mw)| {
            let q_safe = (*q).max(1e-30);
            p * (p / (q_safe * mw)).ln() * mw
        }).sum()
    }

    /// Returns cached entropy.
    #[inline]
    #[must_use]
    pub const fn entropy(&self) -> f64 { self.entropy }

    /// Returns cached metric-weighted mean.
    #[inline]
    #[must_use]
    pub const fn weighted_mean(&self) -> f64 { self.weighted_mean }
}

#[cfg(test)]
mod tests {
    use super::BeliefDistribution;

    #[test]
    fn belief_normalization_invariant() {
        let b = BeliefDistribution::from_weights([1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0]).expect("valid");
        let s: f64 = b.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }
}
