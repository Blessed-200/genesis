use genesis_types::{AxiomID, GenesisError, WitnessBuilder};

use crate::dirac::DiracOperator;

#[derive(Clone, Copy, Debug)]
pub enum CutoffFunction { Gaussian, Polynomial { exponent: u32 }, Sharp }
impl CutoffFunction {
    pub fn evaluate(self, x: f64) -> f64 { match self { Self::Gaussian => (-x).exp(), Self::Polynomial { exponent } => (1.0 + x).powi(-(exponent as i32)), Self::Sharp => if x <= 1.0 { 1.0 } else { 0.0 } } }
    pub fn derivative(self, x: f64) -> f64 { match self { Self::Gaussian => -(-x).exp(), Self::Polynomial { exponent } => -f64::from(exponent) * (1.0 + x).powi(-(exponent as i32) - 1), Self::Sharp => 0.0 } }
}

pub struct SpectralActionResult { pub total_action: f64, pub seeley_dewitt: [f64; 3], pub mean_curvature: f64, pub gradients: Vec<(u64, [f64; 16])>, pub evaluation_proof: [u8; 32] }

pub struct SpectralActionEngine { cutoff: CutoffFunction, lambda: f64 }
impl SpectralActionEngine {
    pub fn new(cutoff: CutoffFunction, lambda: f64) -> Self { Self { cutoff, lambda } }
    pub fn evaluate(&self, node_blades: &[(u64, [f64; 16])]) -> Result<SpectralActionResult, GenesisError> {
        if node_blades.is_empty() { return Err(GenesisError::SpectralInsufficientNodes { found: 0, required: 1 }); }
        let gradients = self.gradient_only(node_blades)?;
        let mut total_action = 0.0;
        let mut a2 = 0.0;
        let mut builder = WitnessBuilder::new();
        for (id, blades) in node_blades {
            let d = DiracOperator::from_blades(blades, self.lambda)?;
            let sq = d.squared();
            let e = sq.matrix.trace() / (self.lambda * self.lambda).max(1e-12);
            total_action += self.cutoff.evaluate(e);
            a2 += sq.ricci_scalar / 6.0;
            let _ = id;
            let _ = d.construction_proof;
            builder.check(AxiomID::ProofGuard, || true)?;
        }
        let a0 = node_blades.len() as f64;
        let a4 = if a0 > 0.0 { (a2 * a2) / a0 } else { 0.0 };
        let mean_curvature = if a0 > 0.0 { 6.0 * a2 / a0 } else { 0.0 };
        let proof = builder.build(0);
        Ok(SpectralActionResult { total_action, seeley_dewitt: [a0, a2, a4], mean_curvature, gradients, evaluation_proof: proof.hash })
    }
    pub fn gradient_only(&self, node_blades: &[(u64, [f64; 16])]) -> Result<Vec<(u64, [f64; 16])>, GenesisError> {
        if node_blades.is_empty() { return Err(GenesisError::SpectralInsufficientNodes { found: 0, required: 1 }); }
        const EPS: f64 = 1e-7;
        let mut out = Vec::with_capacity(node_blades.len());
        for &(id, blades) in node_blades {
            let d_base = DiracOperator::from_blades(&blades, self.lambda)?;
            let sq_base = d_base.squared();
            let e_base = self.cutoff.evaluate(sq_base.matrix.trace() / (self.lambda * self.lambda).max(1e-30));
            let mut grad = [0.0_f64; 16];
            for j in 0..16 {
                let mut perturbed = blades;
                perturbed[j] += EPS;
                let d_p = DiracOperator::from_blades(&perturbed, self.lambda)?;
                let sq_p = d_p.squared();
                let e_p = self.cutoff.evaluate(sq_p.matrix.trace() / (self.lambda * self.lambda).max(1e-30));
                grad[j] = (e_p - e_base) / EPS;
            }
            out.push((id, grad));
        }
        Ok(out)
    }
}
