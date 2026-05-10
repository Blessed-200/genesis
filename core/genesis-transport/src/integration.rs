use genesis_dynamics::VFEMinimizer;
use genesis_types::NodeId;

/// Input terms that define the integrated objective `J_t`.
///
/// `VFE_t` is produced by `VFEMinimizer`, `CausalPenalty_t` by
/// `genesis-causal` validations, `TransportCost_t` by JKO/Sinkhorn, and
/// `SpectralInstability_t` by `genesis-spectral` metrics.
///
/// AX-ID: AXIOMA-002, AXIOMA-003, AXIOMA-006, H_dinámica, H_estructura, H_compresión
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JtTerms {
    /// Variational free-energy contribution at time t.
    pub vfe_t: f64,
    /// Causal consistency penalty at time t.
    pub causal_penalty_t: f64,
    /// Optimal transport cost contribution at time t.
    pub transport_cost_t: f64,
    /// Spectral instability contribution at time t.
    pub spectral_instability_t: f64,
}

/// Full trace of `J_t` computation including per-term contributions.
///
/// AX-ID: AXIOMA-003, H_dinámica, H_estructura, H_compresión
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JtBreakdown {
    /// Original terms used in the computation.
    pub terms: JtTerms,
    /// Aggregated objective value.
    pub j_t: f64,
}

impl JtBreakdown {
    /// Computes the unweighted integrated objective.
    #[must_use]
    pub fn from_terms(terms: JtTerms) -> Self {
        let j_t = terms.vfe_t
            + terms.causal_penalty_t
            + terms.transport_cost_t
            + terms.spectral_instability_t;
        Self { terms, j_t }
    }

    /// Reconstructs `J_t` from trace fields for regression validation.
    #[must_use]
    pub fn reconstruct(&self) -> f64 {
        self.terms.vfe_t
            + self.terms.causal_penalty_t
            + self.terms.transport_cost_t
            + self.terms.spectral_instability_t
    }

    /// Returns the isolated contribution of each term.
    #[must_use]
    pub fn contributions(&self) -> [f64; 4] {
        [
            self.terms.vfe_t,
            self.terms.causal_penalty_t,
            self.terms.transport_cost_t,
            self.terms.spectral_instability_t,
        ]
    }
}

/// Computes `VFE_t` from a configured minimizer and current belief state.
///
/// AX-ID: AXIOMA-003, H_información
#[must_use]
pub fn vfe_t(minimizer: &VFEMinimizer, node_id: NodeId, obs: Option<&[f64; 4]>) -> f64 {
    minimizer.compute_vfe(node_id, obs)
}

#[cfg(test)]
mod tests {
    use super::{JtBreakdown, JtTerms};

    #[test]
    fn breakdown_reconstructs_exact_sum() {
        let breakdown = JtBreakdown::from_terms(JtTerms {
            vfe_t: 0.2,
            causal_penalty_t: 0.4,
            transport_cost_t: 0.6,
            spectral_instability_t: 0.8,
        });
        assert!((breakdown.j_t - 2.0).abs() < 1e-12);
        assert!((breakdown.reconstruct() - breakdown.j_t).abs() < 1e-12);
    }
}
