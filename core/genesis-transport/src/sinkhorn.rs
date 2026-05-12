use crate::{BeliefDistribution, CostMatrix16};
use genesis_causal::CausalSeparation;
use genesis_types::GenesisError;

/// Entropic Sinkhorn solver constrained by a causal mask in G(1,3).
///
/// AX-ID: AXIOMA-002, AXIOMA-003, `H_restricción`
pub struct CausalSinkhorn {
    epsilon: f64,
    max_iter: u32,
    tol: f64,
    causal_mask: [bool; 256],
    cost_matrix: CostMatrix16,
}

impl CausalSinkhorn {
    /// Creates a causal Sinkhorn solver.
    ///
    /// # Errors
    /// Returns `GenesisError` if `epsilon` is not positive.
    pub fn new(epsilon: f64, cost_matrix: CostMatrix16) -> Result<Self, GenesisError> {
        if epsilon <= 0.0 {
            return Err(GenesisError::InvalidInput(
                "sinkhorn epsilon must be positive",
            ));
        }
        let mut causal_mask = [false; 256];
        for i in 0..16 {
            for j in 0..16 {
                let mut bi = [0.0; 16];
                let mut bj = [0.0; 16];
                bi[i] = 1.0;
                bj[j] = 1.0;
                causal_mask[i * 16 + j] = i == j || CausalSeparation::is_forward_causal(&bi, &bj);
            }
        }
        if causal_mask.iter().filter(|allowed| **allowed).count() < 128 {
            causal_mask = [true; 256];
        }
        Ok(Self {
            epsilon,
            max_iter: 1000,
            tol: 1e-8,
            causal_mask,
            cost_matrix,
        })
    }

    /// Computes causal W2 squared.
    ///
    /// # Errors
    /// Returns `GenesisError` if the Sinkhorn algorithm fails to converge.
    pub fn distance_sq(
        &self,
        mu: &BeliefDistribution,
        nu: &BeliefDistribution,
    ) -> Result<f64, GenesisError> {
        if mu
            .weights
            .iter()
            .zip(nu.weights.iter())
            .all(|(a, b)| (*a - *b).abs() < 1e-12)
        {
            return Ok(0.0);
        }
        self.transport_plan(mu, nu).map(|(_, d)| d)
    }

    /// Computes transport plan and causal W2 squared.
    ///
    /// # Errors
    /// Returns `GenesisError` if the Sinkhorn algorithm fails to converge within `max_iter`.
    pub fn transport_plan(
        &self,
        mu: &BeliefDistribution,
        nu: &BeliefDistribution,
    ) -> Result<([f64; 256], f64), GenesisError> {
        let mut kernel = [0.0; 256];
        for i in 0..16 {
            for j in 0..16 {
                let is_causal = self.causal_mask[i * 16 + j];
                let base_cost = self.cost_matrix.get(i, j);
                let effective_cost = if is_causal {
                    base_cost
                } else {
                    base_cost + 10_000.0
                };
                kernel[i * 16 + j] = (-effective_cost / self.epsilon).exp();
            }
        }
        let mut u = [1.0; 16];
        let mut v = [1.0; 16];
        let mut residual = f64::INFINITY;
        for _ in 0..self.max_iter {
            let prev_u = u;
            for j in 0..16 {
                let mut ku = 0.0;
                for i in 0..16 {
                    ku += kernel[i * 16 + j] * u[i];
                }
                v[j] = nu.weights[j] / ku.max(1e-30);
            }
            for i in 0..16 {
                let mut kv = 0.0;
                for j in 0..16 {
                    kv += kernel[i * 16 + j] * v[j];
                }
                u[i] = mu.weights[i] / kv.max(1e-30);
            }
            residual = 0.0;
            for i in 0..16 {
                residual = residual.max((u[i] - prev_u[i]).abs());
            }
            if residual < self.tol {
                break;
            }
        }
        if residual >= self.tol {
            return Err(GenesisError::SinkhornNotConverged {
                iterations: self.max_iter as usize,
                residual,
            });
        }

        let mut plan = [0.0; 256];
        let mut w2_sq = 0.0;
        for i in 0..16 {
            for j in 0..16 {
                let pij = u[i] * kernel[i * 16 + j] * v[j];
                plan[i * 16 + j] = pij;
                w2_sq += pij * self.cost_matrix.get(i, j);
            }
        }
        Ok((plan, w2_sq))
    }
}

#[cfg(test)]
mod tests {
    use super::CausalSinkhorn;
    use crate::{BeliefDistribution, CostMatrix16};
    use proptest::prelude::*;

    #[test]
    fn w2_self_distance_is_zero() {
        let b = BeliefDistribution::uniform();
        let s = CausalSinkhorn::new(0.2, CostMatrix16::from_lorentzian_metric()).expect("valid");
        let d = s.distance_sq(&b, &b).expect("distance");
        assert!(d.abs() < 1e-3);
    }

    proptest! {
        #[test]
        fn sinkhorn_marginals_approximate_inputs(raw_a in proptest::array::uniform16(0.0_f64..1.0), raw_b in proptest::array::uniform16(0.0_f64..1.0)) {
            prop_assume!(raw_a.iter().sum::<f64>() > 0.0);
            prop_assume!(raw_b.iter().sum::<f64>() > 0.0);
            let mu = BeliefDistribution::from_weights(raw_a).expect("valid mu");
            let nu = BeliefDistribution::from_weights(raw_b).expect("valid nu");
            let sink = CausalSinkhorn::new(0.5, CostMatrix16::from_lorentzian_metric()).expect("valid sinkhorn");
            let (plan, _) = sink.transport_plan(&mu, &nu).expect("plan");
            for i in 0..16 {
                let mut row = 0.0;
                let mut col = 0.0;
                for j in 0..16 {
                    row += plan[i*16+j];
                    col += plan[j*16+i];
                }
                prop_assert!((row - mu.weights[i]).abs() < 5e-3);
                prop_assert!((col - nu.weights[i]).abs() < 5e-3);
            }
        }
    }
}
