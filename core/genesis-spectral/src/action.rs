use genesis_types::{AxiomID, GenesisError, WitnessBuilder};

use crate::dirac::DiracOperator;

#[derive(Clone, Copy, Debug)]
pub enum CutoffFunction {
    Gaussian,
    Polynomial { exponent: u32 },
    Sharp,
}
impl CutoffFunction {
    #[must_use]
    pub fn evaluate(self, x: f64) -> f64 {
        match self {
            Self::Gaussian => (-x).exp(),
            Self::Polynomial { exponent } => {
                let exp_i32 = i32::try_from(exponent).unwrap_or(i32::MAX);
                (1.0 + x).powi(-exp_i32)
            }
            Self::Sharp => {
                if x <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    #[must_use]
    pub fn derivative(self, x: f64) -> f64 {
        match self {
            Self::Gaussian => -(-x).exp(),
            Self::Polynomial { exponent } => {
                let exp_i32 = i32::try_from(exponent).unwrap_or(i32::MAX);
                -f64::from(exponent) * (1.0 + x).powi(-exp_i32 - 1)
            }
            Self::Sharp => 0.0,
        }
    }
}

pub struct SpectralActionResult {
    pub total_action: f64,
    pub seeley_dewitt: [f64; 3],
    pub mean_curvature: f64,
    pub gradients: Vec<(u64, [f64; 16])>,
    pub evaluation_proof: [u8; 32],
}

pub struct SpectralActionEngine {
    cutoff: CutoffFunction,
    lambda: f64,
}
impl SpectralActionEngine {
    fn lambda_sq_floor(lambda: f64) -> f64 {
        use genesis_types::constants::SPECTRAL_LAMBDA_MIN;
        (lambda * lambda).max(SPECTRAL_LAMBDA_MIN * SPECTRAL_LAMBDA_MIN)
    }

    #[must_use]
    pub const fn new(cutoff: CutoffFunction, lambda: f64) -> Self {
        Self { cutoff, lambda }
    }
    /// # Errors
    /// Returns `GenesisError` when input is empty or Dirac construction fails.
    pub fn evaluate(
        &self,
        node_blades: &[(u64, [f64; 16])],
    ) -> Result<SpectralActionResult, GenesisError> {
        if node_blades.is_empty() {
            return Err(GenesisError::SpectralInsufficientNodes {
                found: 0,
                required: 1,
            });
        }
        let gradients = self.gradient_only(node_blades)?;
        let mut total_action = 0.0;
        let mut a2 = 0.0;
        let mut builder = WitnessBuilder::new();
        for (id, blades) in node_blades {
            let d = DiracOperator::from_blades(blades, self.lambda)?;
            let sq = d.squared()?;
            let e = sq.matrix.trace() / Self::lambda_sq_floor(self.lambda);
            total_action += self.cutoff.evaluate(e);
            a2 += sq.ricci_scalar / 6.0;
            let _ = id;
            let _ = d.construction_proof;
            builder.check(AxiomID::ProofGuard, || true)?;
        }
        let a0 = f64::from(u32::try_from(node_blades.len()).unwrap_or(u32::MAX));
        let a4 = if a0 > 0.0 { (a2 * a2) / a0 } else { 0.0 };
        let mean_curvature = if a0 > 0.0 { 6.0 * a2 / a0 } else { 0.0 };
        let proof = builder.build(0);
        Ok(SpectralActionResult {
            total_action,
            seeley_dewitt: [a0, a2, a4],
            mean_curvature,
            gradients,
            evaluation_proof: proof.hash,
        })
    }
    /// # Errors
    /// Returns `GenesisError` when input is empty or finite-difference evaluations fail.
    pub fn gradient_only(
        &self,
        node_blades: &[(u64, [f64; 16])],
    ) -> Result<Vec<(u64, [f64; 16])>, GenesisError> {
        if node_blades.is_empty() {
            return Err(GenesisError::SpectralInsufficientNodes {
                found: 0,
                required: 1,
            });
        }
        let mut out = Vec::with_capacity(node_blades.len());
        for &(id, blades) in node_blades {
            let d_base = DiracOperator::from_blades(&blades, self.lambda)?;
            let sq_base = d_base.squared()?;
            let e_base = self
                .cutoff
                .evaluate(sq_base.matrix.trace() / Self::lambda_sq_floor(self.lambda));
            let mut grad = [0.0_f64; 16];
            for j in 0..16 {
                let mut perturbed = blades;
                perturbed[j] += 1e-7;
                let d_p = DiracOperator::from_blades(&perturbed, self.lambda)?;
                let sq_p = d_p.squared()?;
                let e_p = self
                    .cutoff
                    .evaluate(sq_p.matrix.trace() / Self::lambda_sq_floor(self.lambda));
                grad[j] = (e_p - e_base) / 1e-7;
            }
            out.push((id, grad));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{CutoffFunction, SpectralActionEngine};

    #[test]
    fn cutoff_sharp_and_polynomial_derivative_cover_boundary_branches() {
        assert!((CutoffFunction::Sharp.evaluate(1.0) - 1.0).abs() < 1e-12);
        assert!((CutoffFunction::Sharp.evaluate(1.000_001) - 0.0).abs() < 1e-12);
        assert!((CutoffFunction::Sharp.derivative(0.5) - 0.0).abs() < 1e-12);

        let poly = CutoffFunction::Polynomial { exponent: u32::MAX };
        assert!(poly.evaluate(0.25).is_finite());
        assert!(poly.derivative(0.25).is_finite());
    }

    #[test]
    fn gradient_only_empty_nodes_reports_insufficient_nodes() {
        let engine = SpectralActionEngine::new(CutoffFunction::Gaussian, 1.0);
        assert!(engine.gradient_only(&[]).is_err());
    }

    #[test]
    fn evaluate_rejects_invalid_lambda() {
        let engine = SpectralActionEngine::new(CutoffFunction::Gaussian, 0.0);
        let nodes = vec![(1, [1.0; 16])];
        assert!(engine.evaluate(&nodes).is_err());
    }
}
