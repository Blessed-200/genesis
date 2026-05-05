use genesis_types::constants::SPECTRAL_LAMBDA_MIN;

pub struct ZetaRegularizer {
    pub s: f64,
    pub lambda_min: f64,
}

impl ZetaRegularizer {
    #[must_use]
    pub const fn new(s: f64) -> Self {
        Self {
            s,
            lambda_min: SPECTRAL_LAMBDA_MIN,
        }
    }

    #[must_use]
    pub fn regularize(lambda: f64) -> f64 {
        lambda.max(SPECTRAL_LAMBDA_MIN)
    }

    #[must_use]
    pub fn evaluate_from_trace(&self, d_squared_trace: f64) -> f64 {
        let lambda_eff = d_squared_trace.abs().max(self.lambda_min) / 16.0;
        lambda_eff.powf(-self.s)
    }

    #[must_use]
    pub fn spectral_dimension(&self, d_squared_trace: f64) -> f64 {
        let lambda_eff = d_squared_trace.abs().max(self.lambda_min) / 16.0;
        2.0 * lambda_eff.ln().abs()
    }
}

#[cfg(test)]
mod tests {
    use super::ZetaRegularizer;
    use genesis_types::constants::SPECTRAL_LAMBDA_MIN;

    #[test]
    fn zeta_regularizer_clamps_and_estimates_dimension() {
        let zeta = ZetaRegularizer::new(2.0);
        assert!((ZetaRegularizer::regularize(0.0) - SPECTRAL_LAMBDA_MIN).abs() < 1e-20);
        let value = zeta.evaluate_from_trace(16.0);
        assert!((value - 1.0).abs() < 1e-12);
        assert!(zeta.spectral_dimension(16.0).abs() < 1e-12);
        assert!(zeta.evaluate_from_trace(0.0).is_finite());
    }
}
