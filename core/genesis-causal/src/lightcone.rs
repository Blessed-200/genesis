//! Past-lightcone filtering utilities for causal inference.
//!
//! AX-ID: AXIOMA-002, H_información (LEY_FUNDACIONAL §3.3)

use crate::order::CausalOrder;
use genesis_types::GenesisError;
use smallvec::SmallVec;

pub struct LightconeFilter<'a> {
    causal_order: &'a CausalOrder,
}

impl<'a> LightconeFilter<'a> {
    #[must_use]
    pub fn new(causal_order: &'a CausalOrder) -> Self {
        Self { causal_order }
    }

    #[must_use]
    pub fn causal_inputs(&self, target: u64, candidates: &[u64]) -> SmallVec<[u64; 16]> {
        let past = self.causal_order.past_lightcone(target);
        candidates
            .iter()
            .filter(|&&id| past.contains(&id))
            .copied()
            .collect()
    }

    pub fn validate_inference(&self, inference: &CausalInference) -> Result<(), GenesisError> {
        let past = self.causal_order.past_lightcone(inference.conclusion_id);
        for &premise in &inference.premise_ids {
            if !past.contains(&premise) {
                return Err(GenesisError::AcausalInference {
                    premise_id: premise,
                    conclusion_id: inference.conclusion_id,
                });
            }
        }
        Ok(())
    }

    pub fn explain_inference_failure(
        &self,
        inference: &CausalInference,
    ) -> Result<(), GenesisError> {
        self.validate_inference(inference)
    }

    #[must_use]
    pub fn is_before(&self, a: u64, b: u64) -> bool {
        self.causal_order.is_before(a, b)
    }

    #[must_use]
    pub fn ancestors_of(&self, node: u64) -> SmallVec<[u64; 16]> {
        self.causal_order.ancestors_of(node)
    }

    #[must_use]
    pub fn descendants_of(&self, node: u64) -> SmallVec<[u64; 16]> {
        self.causal_order.descendants_of(node)
    }

    #[must_use]
    pub fn cone_overlap(&self, a: u64, b: u64) -> SmallVec<[u64; 16]> {
        self.causal_order.cone_overlap(a, b)
    }
}

#[derive(Debug)]
pub struct CausalInference {
    pub premise_ids: Vec<u64>,
    pub conclusion_id: u64,
    pub inferential_strength: f64,
}
