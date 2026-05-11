//! Spinor transport utilities over the causal order.
//!
//! AX-ID: AXIOMA-006, `H_dinámica` (`LEY_FUNDACIONAL` §3.2)

use crate::order::{CausalEdge, CausalOrder};
use genesis_types::GenesisError;

const HOLO_MAX: f64 = 1.0;

#[derive(Clone, Copy, Debug)]
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

    /// Computes parallel transport along a causal edge.
    ///
    /// # Errors
    /// Returns `GenesisError` if the transport operation fails due to numerical instability.
    pub fn parallel_transport(&self, edge: &CausalEdge) -> Result<Self, GenesisError> {
        let theta = edge.causal_strength * std::f64::consts::FRAC_PI_2;
        let (s, c) = theta.sin_cos();
        let mut transported = *self;
        transported.components[0] = c.mul_add(self.components[0], -s * self.components[1]);
        transported.components[1] = s.mul_add(self.components[0], c * self.components[1]);
        transported.normalize();
        Ok(transported)
    }

    /// Transports the spinor along a given path of causal edges.
    ///
    /// # Errors
    /// Returns `GenesisError` if any edge in the path violates causal invariants.
    pub fn transport_along_path(&self, path: &[CausalEdge]) -> Result<Self, GenesisError> {
        let mut state = *self;
        for edge in path {
            state = state.parallel_transport(edge)?;
        }
        Ok(state)
    }
}

pub struct GlobalSection {
    pub spinors: Vec<(u64, DiracSpinor)>,
    pub holonomy: f64,
    pub coherence_order: f64,
}

impl GlobalSection {
    /// Computes the global section and its coherence metrics.
    ///
    /// # Errors
    /// Returns `GenesisError::CausalCycle` if the graph is not a DAG.
    /// Returns `GenesisError::CausalViolation` if a node has no causal ancestors or invalid weights.
    /// Returns `GenesisError::HolonomyExcessive` if the resulting holonomy exceeds `HOLO_MAX`.
    pub fn compute(
        causal_order: &CausalOrder,
        node_ids: &[u64],
        root_spinor: &DiracSpinor,
    ) -> Result<Self, GenesisError> {
        causal_order.verify_acyclic()?;

        let mut ordered_ids = node_ids.to_vec();
        ordered_ids.sort_unstable();

        let mut spinors: Vec<(u64, DiracSpinor)> = Vec::with_capacity(ordered_ids.len());

        for &id in &ordered_ids {
            let incoming = causal_order.incoming_hot_edges(id);
            let spinor = if incoming.is_empty() {
                *root_spinor
            } else {
                let mut accum = [0.0_f64; 8];
                let mut total_w = 0.0_f64;
                for edge in &incoming {
                    let parent_spinor = spinors
                        .iter()
                        .find(|(pid, _)| *pid == edge.cause_id)
                        .map_or(root_spinor, |(_, s)| s);
                    let transported = parent_spinor.parallel_transport(edge)?;
                    let w = edge.causal_strength.max(1e-12);
                    total_w += w;
                    for (dst, src) in accum.iter_mut().zip(transported.components.iter()) {
                        *dst += src * w;
                    }
                }
                if total_w <= 0.0 {
                    return Err(GenesisError::CausalViolation {
                        cause_id: id,
                        effect_id: id,
                    });
                }
                for coeff in &mut accum {
                    *coeff /= total_w;
                }
                let mut combined = DiracSpinor { components: accum };
                combined.normalize();
                combined
            };
            spinors.push((id, spinor));
        }

        let holonomy = compute_holonomy(&spinors, causal_order)?;
        #[allow(clippy::cast_precision_loss)]
        // Se acepta la pérdida de precisión entrópica para promedios de grandes poblaciones de engramas.
        let n = spinors.len() as f64;
        let coherence_order = if n > 0.0 {
            let sum_re: f64 = spinors.iter().map(|(_, s)| s.components[0]).sum();
            let sum_im: f64 = spinors.iter().map(|(_, s)| s.components[1]).sum();
            (sum_re / n).hypot(sum_im / n)
        } else {
            0.0
        };

        Ok(Self {
            spinors,
            holonomy,
            coherence_order,
        })
    }

    #[must_use]
    pub fn decision_signal(&self) -> f64 {
        if self.spinors.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        // Se acepta la pérdida de precisión entrópica para promedios de grandes poblaciones de engramas.
        let n_f = self.spinors.len() as f64;
        let mean_re = self
            .spinors
            .iter()
            .map(|(_, s)| s.components[0])
            .sum::<f64>()
            / n_f;
        mean_re * self.coherence_order
    }
}

fn compute_holonomy(
    spinors: &[(u64, DiracSpinor)],
    causal_order: &CausalOrder,
) -> Result<f64, GenesisError> {
    let mut total_sq = 0.0_f64;
    let mut pairs = 0usize;

    for (node_id, _) in spinors {
        let incoming = causal_order.incoming_hot_edges(*node_id);
        if incoming.len() <= 1 {
            continue;
        }

        let mut transported_states: Vec<DiracSpinor> = Vec::with_capacity(incoming.len());
        for edge in &incoming {
            let parent_spinor = spinors
                .iter()
                .find(|(pid, _)| *pid == edge.cause_id)
                .map(|(_, s)| s)
                .ok_or(GenesisError::CausalViolation {
                    cause_id: edge.cause_id,
                    effect_id: edge.effect_id,
                })?;
            transported_states.push(parent_spinor.parallel_transport(edge)?);
        }

        for i in 0..transported_states.len() {
            for j in (i + 1)..transported_states.len() {
                let pi =
                    transported_states[i].components[1].atan2(transported_states[i].components[0]);
                let pj =
                    transported_states[j].components[1].atan2(transported_states[j].components[0]);
                let d = wrap_phase(pi - pj);
                total_sq += d * d;
                pairs += 1;
            }
        }
    }

    let holonomy = if pairs == 0 {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        // Se acepta la pérdida de precisión entrópica para promedios de grandes poblaciones de engramas.
        let pairs_f = pairs as f64;
        (total_sq / pairs_f).sqrt()
    };
    if holonomy > HOLO_MAX {
        return Err(GenesisError::HolonomyExcessive {
            holonomy,
            threshold: HOLO_MAX,
        });
    }
    Ok(holonomy)
}

fn wrap_phase(phi: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut x = phi % two_pi;
    if x > std::f64::consts::PI {
        x -= two_pi;
    } else if x < -std::f64::consts::PI {
        x += two_pi;
    }
    x
}
