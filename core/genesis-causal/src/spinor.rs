//! Spinor transport utilities over the causal order.
//!
//! AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)

use crate::order::{CausalEdge, CausalOrder};
use genesis_types::GenesisError;

#[derive(Clone, Debug)]
pub struct DiracSpinor {
    pub components: [f64; 8],
}

impl DiracSpinor {
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.components.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-12 {
            for c in &mut self.components {
                *c /= n;
            }
        }
    }

    pub fn parallel_transport(&self, edge: &CausalEdge) -> Result<Self, GenesisError> {
        let theta = edge.causal_strength * std::f64::consts::FRAC_PI_2;
        let (s, c) = theta.sin_cos();
        let mut transported = self.clone();
        transported.components[0] = c * self.components[0] - s * self.components[1];
        transported.components[1] = s * self.components[0] + c * self.components[1];
        transported.normalize();
        Ok(transported)
    }
}

pub struct GlobalSection {
    pub spinors: Vec<(u64, DiracSpinor)>,
    pub holonomy: f64,
    pub coherence_order: f64,
}

impl GlobalSection {
    pub fn compute(causal_order: &CausalOrder, node_ids: &[u64], root_spinor: DiracSpinor) -> Result<Self, GenesisError> {
        causal_order.verify_acyclic()?;
        let mut spinors: Vec<(u64, DiracSpinor)> = Vec::with_capacity(node_ids.len());
        for &id in node_ids {
            let past = causal_order.past_lightcone(id);
            let spinor = if past.is_empty() { root_spinor.clone() } else { root_spinor.clone() };
            spinors.push((id, spinor));
        }

        let n = spinors.len() as f64;
        let coherence_order = if n > 0.0 {
            let sum_re: f64 = spinors.iter().map(|(_, s)| s.components[0]).sum();
            let sum_im: f64 = spinors.iter().map(|(_, s)| s.components[1]).sum();
            ((sum_re / n).powi(2) + (sum_im / n).powi(2)).sqrt()
        } else {
            0.0
        };

        Ok(Self { spinors, holonomy: 0.0, coherence_order })
    }

    #[must_use]
    pub fn decision_signal(&self) -> f64 {
        if self.spinors.is_empty() {
            return 0.0;
        }
        let mean_re =
            self.spinors.iter().map(|(_, s)| s.components[0]).sum::<f64>() / self.spinors.len() as f64;
        mean_re * self.coherence_order
    }
}
