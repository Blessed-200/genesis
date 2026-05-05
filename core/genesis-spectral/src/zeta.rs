use genesis_types::constants::SPECTRAL_LAMBDA_MIN;

pub struct ZetaRegularizer {
    pub s: f64,
    pub lambda_min: f64,
}

impl ZetaRegularizer {
    pub fn new(s: f64) -> Self {
        Self {
            s,
            lambda_min: SPECTRAL_LAMBDA_MIN,
        }
    }

    pub fn regularize(lambda: f64) -> f64 {
        lambda.max(SPECTRAL_LAMBDA_MIN)
    }

    pub fn evaluate_from_trace(&self, d_squared_trace: f64) -> f64 {
        let lambda_eff = d_squared_trace.abs().max(self.lambda_min) / 16.0;
        lambda_eff.powf(-self.s)
    }

    pub fn spectral_dimension(&self, d_squared_trace: f64) -> f64 {
        let lambda_eff = d_squared_trace.abs().max(self.lambda_min) / 16.0;
        2.0 * lambda_eff.ln().abs()
    }
}
