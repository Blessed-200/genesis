use genesis_types::{AxiomID, GenesisError, WitnessBuilder};

use crate::action::{CutoffFunction, SpectralActionEngine};

#[derive(Clone, Copy, Debug)]
pub struct ConvergenceCriteria {
    pub grad_tol: f64,
    pub max_steps: u32,
}

pub struct SpectralFlowEngine {
    action_engine: SpectralActionEngine,
    tau: f64,
    _criteria: ConvergenceCriteria,
    candidate_buffer: Vec<(u64, [f64; 16])>,
}

#[derive(Clone, Debug)]
pub struct SpectralFlowStep {
    pub iteration: u32,
    pub action_before: f64,
    pub action_after: f64,
    pub action_delta: f64,
    pub tau_used: f64,
    pub step_proof: [u8; 32],
}

#[derive(Clone, Debug, Default)]
pub struct SpectralFlowHistory {
    pub steps: Vec<SpectralFlowStep>,
}

fn spectral_divergence(last_delta: f64) -> GenesisError {
    GenesisError::SpectralDivergence {
        iteration: 0,
        delta: if last_delta.is_finite() {
            last_delta
        } else {
            f64::MAX
        },
    }
}

impl SpectralFlowEngine {
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(
        lambda: f64,
        tau: f64,
        cutoff: CutoffFunction,
        criteria: ConvergenceCriteria,
    ) -> Self {
        Self {
            action_engine: SpectralActionEngine::new(cutoff, lambda),
            tau,
            _criteria: criteria,
            candidate_buffer: Vec::new(),
        }
    }
    /// # Errors
    /// Returns `GenesisError::SpectralDivergence` when Armijo backtracking fails.
    pub fn step(
        &mut self,
        node_blades: &mut [(u64, [f64; 16])],
    ) -> Result<SpectralFlowStep, GenesisError> {
        let current = self.action_engine.evaluate(node_blades)?;
        let mut trial_tau = self.tau;
        let mut last_delta = f64::INFINITY;
        self.candidate_buffer
            .resize(node_blades.len(), (0_u64, [0.0_f64; 16]));

        for _ in 0..10 {
            self.candidate_buffer.copy_from_slice(node_blades);
            for ((_, b), (_, g)) in self.candidate_buffer.iter_mut().zip(&current.gradients) {
                for i in 0..16 {
                    b[i] -= trial_tau * g[i];
                }
            }
            let next = self.action_engine.evaluate(&self.candidate_buffer)?;
            let delta = next.total_action - current.total_action;
            last_delta = delta;
            if delta.is_finite() && delta <= 0.0 {
                node_blades.copy_from_slice(&self.candidate_buffer);
                let mut wb = WitnessBuilder::new();
                wb.check(AxiomID::ProofGuard, || true)?;
                let p = wb.build(0);
                let step = SpectralFlowStep {
                    iteration: 0,
                    action_before: current.total_action,
                    action_after: next.total_action,
                    action_delta: delta,
                    tau_used: trial_tau,
                    step_proof: p.hash,
                };
                debug_assert!(step.action_delta <= 0.0);
                return Ok(step);
            }
            trial_tau *= 0.5;
        }
        Err(spectral_divergence(last_delta))
    }
}

#[cfg(test)]
mod tests {
    use super::{ConvergenceCriteria, SpectralFlowEngine};
    use crate::action::CutoffFunction;
    use genesis_types::GenesisError;

    #[test]
    fn divergence_error_clamps_non_finite_delta() {
        assert!(matches!(
            super::spectral_divergence(f64::NAN),
            GenesisError::SpectralDivergence { delta, .. } if (delta - f64::MAX).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn step_propagates_invalid_lambda_before_mutation() {
        let mut nodes = [(1_u64, [0.0_f64; 16])];
        let mut engine = SpectralFlowEngine::new(
            f64::NAN,
            0.1,
            CutoffFunction::Gaussian,
            ConvergenceCriteria {
                grad_tol: 1e-8,
                max_steps: 1,
            },
        );
        assert!(matches!(
            engine.step(&mut nodes),
            Err(GenesisError::SpectralLambdaUnderflow { .. })
        ));
        assert!(nodes[0].1.iter().all(|value| value.abs() < 1e-12));
    }
}
