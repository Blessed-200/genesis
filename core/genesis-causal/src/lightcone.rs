//! Past-lightcone filtering utilities for causal inference.
//!
//! AX-ID: AXIOMA-002, H_información (LEY_FUNDACIONAL §3.3)

use crate::order::CausalOrder;
use smallvec::SmallVec;

pub struct LightconeFilter<'a> {
    causal_order: &'a CausalOrder,
}

impl<'a> LightconeFilter<'a> {
    #[must_use]
    pub fn new(causal_order: &'a CausalOrder) -> Self {
        Self { causal_order }
    }

    pub fn causal_inputs(&self, target: u64, candidates: &[u64]) -> SmallVec<[u64; 16]> {
        let past = self.causal_order.past_lightcone(target);
        candidates.iter().filter(|&&id| past.contains(&id)).copied().collect()
    }

    #[must_use]
    pub fn validate_inference(&self, inference: &CausalInference) -> bool {
        let past = self.causal_order.past_lightcone(inference.conclusion_id);
        inference.premise_ids.iter().all(|id| past.contains(id))
    }
}

#[derive(Debug)]
pub struct CausalInference {
    pub premise_ids: Vec<u64>,
    pub conclusion_id: u64,
    pub inferential_strength: f64,
}
