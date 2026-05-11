use genesis_spectral::DiracOperator;
use genesis_types::GenesisError;

/// A Clifford Engram: a causal memory unit.
/// Not a database row. A geometric imprint in G(1,3).
///
/// Emotional weight equals VFE magnitude at encoding time.
/// High surprisal events create deeper topological imprints.
/// Low surprisal events decay over time.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, AXIOMA-003
#[derive(Clone, Debug)]
pub struct CliffordEngram {
    /// The full 16-blade multivector representation of the experience.
    pub blades: [f64; 16],
    /// Causal timestamp: node id in the `CausalOrder` DAG.
    pub causal_id: u64,
    /// Emotional weight represented as VFE at encoding time.
    pub vfe_weight: f64,
    /// Number of times this engram has been retrieved.
    pub retrieval_count: u32,
    /// Cycle at which this engram was encoded.
    pub encoded_at: u64,
    /// Precomputed spectral projection used for fast resonance retrieval.
    pub spectral_state: [f64; 16],
}

impl CliffordEngram {
    #[must_use]
    pub fn new(
        blades: [f64; 16],
        causal_id: u64,
        vfe_weight: f64,
        cycle: u64,
        spectral_state: [f64; 16],
    ) -> Self {
        Self {
            blades,
            causal_id,
            vfe_weight,
            retrieval_count: 0,
            encoded_at: cycle,
            spectral_state,
        }
    }

    #[must_use]
    pub fn strength(&self) -> f64 {
        self.vfe_weight * (1.0 + 0.1 * f64::from(self.retrieval_count))
    }

    #[must_use]
    pub fn decay(&self, current_cycle: u64, lambda: f64) -> f64 {
        let delta = current_cycle.saturating_sub(self.encoded_at) as f64;
        let effective_weight = self.vfe_weight.max(1e-6);
        (-lambda * delta / effective_weight).exp()
    }
}

/// Causal memory collection with Dirac-based pattern completion.
///
/// AX-ID: AXIOMA-002, AXIOMA-003, H_información
pub struct EngramStore {
    /// Sorted by causal_id for O(log N) lookup.
    engrams: Vec<CliffordEngram>,
    /// Forgetting rate lambda.
    decay_lambda: f64,
    /// Maximum engrams before pruning.
    capacity: usize,
    /// Fixed lambda used for spectral precomputation.
    spectral_lambda: f64,
}

impl EngramStore {
    #[must_use]
    pub fn new(capacity: usize, decay_lambda: f64) -> Self {
        Self {
            engrams: Vec::with_capacity(capacity),
            decay_lambda,
            capacity,
            spectral_lambda: 1.0,
        }
    }

    pub fn encode(
        &mut self,
        blades: [f64; 16],
        causal_id: u64,
        vfe_weight: f64,
        current_cycle: u64,
    ) -> Result<(), GenesisError> {
        let spectral_state =
            DiracOperator::from_blades(&blades, self.spectral_lambda)?.apply(&blades)?;
        let engram =
            CliffordEngram::new(blades, causal_id, vfe_weight, current_cycle, spectral_state);
        let pos = self.engrams.partition_point(|e| e.causal_id < causal_id);
        self.engrams.insert(pos, engram);
        if self.engrams.len() > self.capacity {
            self.prune(current_cycle);
        }
        Ok(())
    }

    pub fn pattern_complete(
        &mut self,
        partial_query: &[f64; 16],
        _lambda: f64,
        current_cycle: u64,
    ) -> Result<Option<CliffordEngram>, GenesisError> {
        if self.engrams.is_empty() {
            return Ok(None);
        }

        let query_dirac = DiracOperator::from_blades(partial_query, self.spectral_lambda)?;
        let query_applied = query_dirac.apply(partial_query)?;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = None;

        for (idx, engram) in self.engrams.iter().enumerate() {
            let decay = engram.decay(current_cycle, self.decay_lambda);
            if decay < 1e-6 {
                continue;
            }

            let mut dot = 0.0;
            let mut query_norm_sq = 0.0;
            let mut engram_norm_sq = 0.0;
            for i in 0..16 {
                let q = query_applied[i];
                let e = engram.spectral_state[i];
                dot += q * e;
                query_norm_sq += q * q;
                engram_norm_sq += e * e;
            }
            let denom = (query_norm_sq.sqrt() * engram_norm_sq.sqrt()).max(1e-12);
            let constructive_interference = dot / denom;
            let score = constructive_interference * (engram.strength() * decay);

            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            self.engrams[idx].retrieval_count += 1;
            return Ok(Some(self.engrams[idx].clone()));
        }
        Ok(None)
    }

    pub fn dream_cycle(&mut self, current_cycle: u64) {
        self.prune(current_cycle);
    }

    #[must_use]
    pub fn causal_ids(&self) -> Vec<u64> {
        self.engrams.iter().map(|engram| engram.causal_id).collect()
    }

    #[must_use]
    pub fn weighted_strengths(&self, current_cycle: u64) -> Vec<(u64, f64)> {
        self.engrams
            .iter()
            .map(|engram| {
                (
                    engram.causal_id,
                    engram.strength() * engram.decay(current_cycle, self.decay_lambda),
                )
            })
            .collect()
    }

    fn prune(&mut self, current_cycle: u64) {
        self.engrams
            .retain(|e| e.decay(current_cycle, self.decay_lambda) * e.strength() > 1e-6);
        if self.engrams.len() > self.capacity {
            if self.capacity == 0 {
                self.engrams.clear();
                return;
            }
            self.engrams
                .select_nth_unstable_by(self.capacity - 1, |a, b| {
                    let sa = a.strength() * a.decay(current_cycle, self.decay_lambda);
                    let sb = b.strength() * b.decay(current_cycle, self.decay_lambda);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
            self.engrams.truncate(self.capacity);
            self.engrams.sort_unstable_by_key(|e| e.causal_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EngramStore;

    fn blades(seed: f64) -> [f64; 16] {
        let mut out = [0.0; 16];
        out[1] = seed;
        out[2] = seed * 0.5;
        out[4] = seed * 0.25;
        out
    }

    #[test]
    fn pattern_complete_finds_most_resonant_engram() {
        let mut store = EngramStore::new(16, 0.01);
        store.encode(blades(0.2), 1, 1.0, 0).expect("encode");
        store.encode(blades(1.0), 2, 2.0, 0).expect("encode");
        store.encode(blades(2.0), 3, 1.0, 0).expect("encode");
        let q = blades(1.02);
        let recovered = store
            .pattern_complete(&q, 1.0, 5)
            .expect("completion")
            .expect("engram");
        assert_eq!(recovered.causal_id, 2);
    }

    #[test]
    fn high_vfe_engrams_survive_dream_cycle() {
        let mut store = EngramStore::new(8, 0.1);
        store.encode(blades(0.5), 1, 10.0, 0).expect("encode");
        store.encode(blades(0.6), 2, 0.01, 0).expect("encode");
        store.dream_cycle(1000);
        assert!(store.engrams.iter().any(|e| e.causal_id == 1));
        assert!(!store.engrams.iter().any(|e| e.causal_id == 2));
    }

    #[test]
    fn retrieval_reinforces_engram() {
        let mut store = EngramStore::new(8, 0.001);
        store.encode(blades(1.0), 7, 5.0, 0).expect("encode");
        let q = blades(1.0);
        let before = store.engrams[0].strength();
        let _ = store.pattern_complete(&q, 1.0, 1).expect("completion");
        let _ = store.pattern_complete(&q, 1.0, 2).expect("completion");
        let after = store.engrams[0].strength();
        assert!(after > before);
        assert_eq!(store.engrams[0].retrieval_count, 2);
    }

    #[test]
    fn decay_lambda_controls_forgetting_rate() {
        let mut slow = EngramStore::new(4, 0.01);
        let mut fast = EngramStore::new(4, 1.0);
        slow.encode(blades(0.9), 10, 1.0, 0).expect("encode");
        fast.encode(blades(0.9), 10, 1.0, 0).expect("encode");

        let slow_decay = slow.engrams[0].decay(10, slow.decay_lambda);
        let fast_decay = fast.engrams[0].decay(10, fast.decay_lambda);
        assert!(fast_decay < slow_decay);
    }

    #[test]
    fn prune_over_capacity_keeps_highest_weighted_engrams() {
        let mut store = EngramStore::new(3, 0.0);
        store.encode(blades(0.0), 1, 1.0, 0).expect("encode");
        store.encode(blades(0.0), 2, 5.0, 0).expect("encode");
        store.encode(blades(0.0), 3, 3.0, 0).expect("encode");
        store.encode(blades(0.0), 4, 7.0, 0).expect("encode");
        store.dream_cycle(0);

        let survivors = store.causal_ids();
        assert_eq!(survivors, vec![2, 3, 4]);
    }

    #[test]
    fn prune_preserves_causal_order_after_partition() {
        let mut store = EngramStore::new(2, 0.0);
        store.encode(blades(0.0), 50, 2.0, 0).expect("encode");
        store.encode(blades(0.0), 10, 9.0, 0).expect("encode");
        store.encode(blades(0.0), 30, 8.0, 0).expect("encode");
        store.dream_cycle(0);
        assert_eq!(store.causal_ids(), vec![10, 30]);
    }

    #[test]
    fn prune_removes_sub_threshold_even_under_capacity() {
        let mut store = EngramStore::new(8, 1.0);
        store.encode(blades(0.0), 1, 1e-6, 0).expect("encode");
        store.encode(blades(0.0), 2, 5.0, 0).expect("encode");
        store.dream_cycle(20);
        assert_eq!(store.causal_ids(), vec![2]);
    }

    #[test]
    fn zero_capacity_store_never_panics_and_keeps_no_engrams() {
        let mut store = EngramStore::new(0, 0.1);
        store.encode(blades(1.0), 1, 1.0, 0).expect("encode");
        assert!(store.causal_ids().is_empty());
    }
}
