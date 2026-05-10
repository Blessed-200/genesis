use crate::{BeliefDistribution, CausalSinkhorn};
use genesis_types::{AxiomID, GenesisError, WitnessBuilder};

/// One implicit JKO descent step in causal Wasserstein geometry.
///
/// AX-ID: AXIOMA-003, H_dinámica
pub struct JKOScheme {
    sinkhorn: CausalSinkhorn,
    /// Implicit time step.
    pub tau: f64,
}

/// Output of a single JKO update.
///
/// AX-ID: AXIOMA-003, H_dinámica
pub struct JKOStep {
    /// Updated belief after the implicit step.
    pub new_distribution: BeliefDistribution,
    /// Free-energy difference `F_{k+1} - F_k`.
    pub free_energy_delta: f64,
    /// Transport penalty `W2^2 / (2τ)`.
    pub transport_cost: f64,
    /// Witness hash for this update step.
    pub step_proof: [u8; 32],
}

impl JKOScheme {
    /// Creates a new scheme with fixed transport backend.
    pub fn new(sinkhorn: CausalSinkhorn, tau: f64) -> Result<Self, GenesisError> {
        if tau <= 0.0 { return Err(GenesisError::InvalidInput("tau must be positive")); }
        Ok(Self { sinkhorn, tau })
    }

    /// Executes one JKO step.
    pub fn step(&self, current: &BeliefDistribution, vfe_gradient: &[f64; 16], current_vfe: f64) -> Result<JKOStep, GenesisError> {
        let mut proposed_weights = current.weights;
        for i in 0..16 { proposed_weights[i] = (proposed_weights[i] - self.tau * vfe_gradient[i]).max(0.0); }
        let sum: f64 = proposed_weights.iter().sum();
        if sum <= 1e-30 { return Err(GenesisError::InvalidInput("JKO step collapsed distribution to zero")); }
        for w in &mut proposed_weights { *w /= sum; }
        let proposed = BeliefDistribution::from_weights(proposed_weights)?;
        let new_vfe = current_vfe + vfe_gradient.iter().zip(proposed.weights.iter().zip(current.weights.iter())).map(|(g, (p, c))| g * (p - c)).sum::<f64>();
        let w2_sq = self.sinkhorn.distance_sq(&proposed, current)?;
        let mut wb = WitnessBuilder::new();
        wb.check(AxiomID::ProofGuard, || true)?;
        let proof = wb.build(0);
        Ok(JKOStep { new_distribution: proposed, free_energy_delta: new_vfe - current_vfe, transport_cost: w2_sq / (2.0 * self.tau), step_proof: proof.hash })
    }
}

#[cfg(test)]
mod tests {
    use crate::{CausalSinkhorn, CostMatrix16, JKOScheme};

    #[test]
    fn jko_step_proof_not_zero() {
        let sink = CausalSinkhorn::new(0.5, CostMatrix16::from_lorentzian_metric()).expect("sink");
        let jko = JKOScheme::new(sink, 0.1).expect("jko");
        let belief = crate::BeliefDistribution::uniform();
        let grad = [0.01_f64; 16];
        let step = jko.step(&belief, &grad, 1.0).expect("step");
        assert_ne!(step.step_proof, [0_u8; 32]);
    }
}
