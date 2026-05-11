use genesis_spectral::DiracOperator;
use genesis_types::GenesisError;

const RESONANCE_COLLAPSE_THRESHOLD: f64 = 0.95;
const MAX_CAUSAL_GAP_FOR_FUSION: u64 = 1000;
const TRAUMA_VFE_THRESHOLD: f64 = 5.0;
const DEGENERATE_NORM_EPSILON_SQ: f64 = 1e-24;

#[derive(Clone, Debug)]
pub struct CliffordEngram {
    pub blades: [f64; 16],
    pub causal_id: u64,
    pub vfe_weight: f64,
    pub retrieval_count: u32,
    pub encoded_at: u64,
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
        (-lambda * delta / self.vfe_weight.max(1e-6)).exp()
    }
}

/// SoA cortical long-term memory optimized for contiguous SIMD-friendly scans.
pub struct CorticalStore {
    pub spectral_states_flat: Vec<f64>,
    pub decays: Vec<f64>,
    pub strengths: Vec<f64>,
    pub causal_ids: Vec<u64>,
}

impl CorticalStore {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            spectral_states_flat: Vec::with_capacity(capacity * 16),
            decays: Vec::with_capacity(capacity),
            strengths: Vec::with_capacity(capacity),
            causal_ids: Vec::with_capacity(capacity),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.causal_ids.len()
    }

    pub fn push(&mut self, causal_id: u64, spectral: [f64; 16], strength: f64, decay: f64) {
        self.causal_ids.push(causal_id);
        self.strengths.push(strength);
        self.decays.push(decay);
        self.spectral_states_flat.extend_from_slice(&spectral);
    }
}

pub struct EngramStore {
    episodic_buffer: Vec<CliffordEngram>,
    cortical_store: CorticalStore,
    decay_lambda: f64,
    capacity: usize,
    spectral_lambda: f64,
    // Last query blades and last normalized spectral query state.
    // Invalidated by replacing the tuple whenever the query changes.
    cached_query_dirac: Option<([f64; 16], [f64; 16])>,
}

impl EngramStore {
    #[must_use]
    pub fn new(capacity: usize, decay_lambda: f64) -> Self {
        Self {
            episodic_buffer: Vec::with_capacity(capacity.min(1024)),
            cortical_store: CorticalStore::new(capacity),
            decay_lambda,
            capacity,
            spectral_lambda: 1.0,
            cached_query_dirac: None,
        }
    }

    pub fn encode(
        &mut self,
        blades: [f64; 16],
        causal_id: u64,
        vfe_weight: f64,
        current_cycle: u64,
    ) -> Result<(), GenesisError> {
        let mut spectral_state =
            DiracOperator::from_blades(&blades, self.spectral_lambda)?.apply(&blades)?;
        normalize_unit(&mut spectral_state)?;
        self.episodic_buffer.push(CliffordEngram::new(
            blades,
            causal_id,
            vfe_weight,
            current_cycle,
            spectral_state,
        ));
        Ok(())
    }

    pub fn pattern_complete(
        &mut self,
        partial_query: &[f64; 16],
        _lambda: f64,
        current_cycle: u64,
    ) -> Result<Option<CliffordEngram>, GenesisError> {
        if self.episodic_buffer.is_empty() && self.cortical_store.len() == 0 {
            return Ok(None);
        }
        let query = if let Some((cached_blades, cached_state)) = self.cached_query_dirac {
            if cached_blades == *partial_query {
                cached_state
            } else {
                self.compute_and_cache_query(partial_query)?
            }
        } else {
            self.compute_and_cache_query(partial_query)?
        };

        let mut best_score = f64::NEG_INFINITY;
        let mut best_epi = None;

        for (idx, e) in self.episodic_buffer.iter().enumerate() {
            let decay = e.decay(current_cycle, self.decay_lambda);
            if decay < 1e-6 {
                continue;
            }
            let mut dot = 0.0;
            for i in 0..16 {
                dot += query[i] * e.spectral_state[i];
            }
            let score = dot * e.strength() * decay;
            if score > best_score {
                best_score = score;
                best_epi = Some(idx);
            }
        }

        let mut best_cortical = None;
        for idx in 0..self.cortical_store.len() {
            let base = idx * 16;
            let mut dot = 0.0;
            for i in 0..16 {
                dot += query[i] * self.cortical_store.spectral_states_flat[base + i];
            }
            let score = dot * self.cortical_store.strengths[idx] * self.cortical_store.decays[idx];
            if score > best_score {
                best_score = score;
                best_cortical = Some(idx);
                best_epi = None;
            }
        }

        if let Some(idx) = best_epi {
            self.episodic_buffer[idx].retrieval_count += 1;
            return Ok(Some(self.episodic_buffer[idx].clone()));
        }
        if let Some(idx) = best_cortical {
            let base = idx * 16;
            let mut spectral = [0.0; 16];
            spectral.copy_from_slice(&self.cortical_store.spectral_states_flat[base..base + 16]);
            return Ok(Some(CliffordEngram::new(
                [0.0; 16],
                self.cortical_store.causal_ids[idx],
                self.cortical_store.strengths[idx],
                current_cycle,
                spectral,
            )));
        }
        Ok(None)
    }

    pub fn dream_cycle(&mut self, current_cycle: u64) -> Result<(), GenesisError> {
        self.prune(current_cycle);
        self.consolidate_to_cortex(current_cycle)
    }

    fn prune(&mut self, current_cycle: u64) {
        self.episodic_buffer
            .retain(|e| e.decay(current_cycle, self.decay_lambda) * e.strength() > 1e-6);
        if self.episodic_buffer.len() > self.capacity {
            self.episodic_buffer
                .select_nth_unstable_by(self.capacity.saturating_sub(1), |a, b| {
                    let sa = a.strength() * a.decay(current_cycle, self.decay_lambda);
                    let sb = b.strength() * b.decay(current_cycle, self.decay_lambda);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
            self.episodic_buffer.truncate(self.capacity);
        }
        self.episodic_buffer.sort_unstable_by_key(|e| e.causal_id);
    }

    fn consolidate_to_cortex(&mut self, _current_cycle: u64) -> Result<(), GenesisError> {
        if self.episodic_buffer.len() < 2 {
            return Ok(());
        }
        let mut consumed = vec![false; self.episodic_buffer.len()];
        for i in 0..self.episodic_buffer.len() {
            if consumed[i] {
                continue;
            }
            let mut cluster = vec![i];
            consumed[i] = true;
            for (j, flag) in consumed.iter_mut().enumerate().skip(i + 1) {
                if *flag {
                    continue;
                }
                let reson = cosine_dot(
                    &self.episodic_buffer[i].spectral_state,
                    &self.episodic_buffer[j].spectral_state,
                );
                let causal_gap = self.episodic_buffer[j]
                    .causal_id
                    .abs_diff(self.episodic_buffer[i].causal_id);
                if reson >= RESONANCE_COLLAPSE_THRESHOLD && causal_gap <= MAX_CAUSAL_GAP_FOR_FUSION
                {
                    cluster.push(j);
                    *flag = true;
                }
            }
            if cluster.len() <= 1 {
                continue;
            }
            let mut spectral = [0.0; 16];
            let mut total_strength = 0.0;
            let mut causal_id = u64::MAX;
            for &idx in &cluster {
                let e = &self.episodic_buffer[idx];
                let s = e.strength();
                total_strength += s;
                causal_id = causal_id.min(e.causal_id);
                for (k, val) in spectral.iter_mut().enumerate() {
                    *val += e.spectral_state[k] * s;
                }
            }
            normalize_unit(&mut spectral)?;
            self.cortical_store
                .push(causal_id, spectral, total_strength, 1.0);
        }
        self.episodic_buffer = self
            .episodic_buffer
            .iter()
            .enumerate()
            .filter(|(i, _)| !consumed[*i])
            .map(|(_, e)| e.clone())
            .collect();
        Ok(())
    }

    /// Predicts the next cognitive state by applying the most resonant memory abstraction.
    ///
    /// The selected engram acts as a spectral transition operator over the current
    /// multivector state, yielding the geometrically expected successor state.
    ///
    /// AX-ID: AXIOMA-003, `H_información` (`LEY_FUNDACIONAL` §3.3)
    ///
    /// # Errors
    /// Returns `GenesisError` when spectral operator construction or application fails.
    pub fn predict_next_state(
        &mut self,
        current_state: &[f64; 16],
        current_cycle: u64,
    ) -> Result<Option<[f64; 16]>, GenesisError> {
        let best = self.pattern_complete(current_state, self.spectral_lambda, current_cycle)?;
        let Some(engram) = best else {
            return Ok(None);
        };

        let transition = DiracOperator::from_blades(&engram.blades, self.spectral_lambda)?;
        let predicted = transition.apply(current_state)?;
        Ok(Some(predicted))
    }

    /// Measures intrinsic surprise for an incoming state relative to memory prediction.
    ///
    /// A value near `1.0` marks a highly unexpected state with high information gain,
    /// while a value near `0.0` marks a state already explained by memory dynamics.
    ///
    /// AX-ID: AXIOMA-003, `H_información` (`LEY_FUNDACIONAL` §3.3)
    ///
    /// # Errors
    /// Returns `GenesisError` when predictive spectral processing fails.
    pub fn calculate_intrinsic_surprise(
        &mut self,
        incoming_state: &[f64; 16],
        current_cycle: u64,
    ) -> Result<f64, GenesisError> {
        let best = self.pattern_complete(incoming_state, self.spectral_lambda, current_cycle)?;
        let Some(engram) = best else {
            return Ok(1.0);
        };

        let transition = DiracOperator::from_blades(&engram.blades, self.spectral_lambda)?;
        let predicted = transition.apply(incoming_state)?;
        let prediction_surprise = cosine_surprise(&predicted, incoming_state);
        let state_surprise = cosine_surprise(&engram.blades, incoming_state);
        Ok(prediction_surprise.max(state_surprise))
    }

    /// Returns the repulsion penalty induced by high-free-energy traumatic memories.
    ///
    /// High-VFE episodic engrams create geometric exclusion pressure against similar
    /// futures, allowing action selection to reject states near costly memory regions.
    ///
    /// AX-ID: AXIOMA-003, `H_información` (`LEY_FUNDACIONAL` §3.3)
    #[must_use]
    pub fn trauma_repulsion(&self, proposed_future: &[f64; 16], current_cycle: u64) -> f64 {
        let mut total_repulsion = 0.0_f64;
        for engram in &self.episodic_buffer {
            if engram.vfe_weight < TRAUMA_VFE_THRESHOLD {
                continue;
            }
            let decay = engram.decay(current_cycle, self.decay_lambda);
            if decay < 1e-6 {
                continue;
            }
            let mut similarity = 0.0_f64;
            for i in 0..16 {
                similarity += engram.spectral_state[i] * proposed_future[i];
            }
            if similarity > 0.0 {
                total_repulsion += similarity * engram.vfe_weight * decay;
            }
        }
        total_repulsion
    }

    fn compute_and_cache_query(
        &mut self,
        partial_query: &[f64; 16],
    ) -> Result<[f64; 16], GenesisError> {
        // HOT PATH: O(1) per pattern-completion call. The query Dirac operator is
        // constructed once outside all memory scans and cached for repeated queries;
        // each stored engram keeps its own spectral representation from `encode`.
        let mut query = DiracOperator::from_blades(partial_query, self.spectral_lambda)?
            .apply(partial_query)?;
        normalize_unit(&mut query)?;
        self.cached_query_dirac = Some((*partial_query, query));
        Ok(query)
    }

    #[must_use]
    pub fn causal_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.episodic_buffer.iter().map(|e| e.causal_id).collect();
        ids.extend(self.cortical_store.causal_ids.iter().copied());
        ids.sort_unstable();
        ids
    }

    #[must_use]
    pub fn weighted_strengths(&self, current_cycle: u64) -> Vec<(u64, f64)> {
        let mut out: Vec<(u64, f64)> = self
            .episodic_buffer
            .iter()
            .map(|e| {
                (
                    e.causal_id,
                    e.strength() * e.decay(current_cycle, self.decay_lambda),
                )
            })
            .collect();
        out.extend(
            self.cortical_store
                .causal_ids
                .iter()
                .zip(
                    self.cortical_store
                        .strengths
                        .iter()
                        .zip(self.cortical_store.decays.iter()),
                )
                .map(|(id, (s, d))| (*id, s * d)),
        );
        out
    }

    #[must_use]
    pub fn cortical_len(&self) -> usize {
        self.cortical_store.len()
    }
}

fn cosine_dot(a: &[f64; 16], b: &[f64; 16]) -> f64 {
    let mut dot = 0.0;
    for i in 0..16 {
        dot += a[i] * b[i];
    }
    dot
}

fn cosine_surprise(a: &[f64; 16], b: &[f64; 16]) -> f64 {
    let mut dot = 0.0_f64;
    let mut a_norm = 0.0_f64;
    let mut b_norm = 0.0_f64;
    for i in 0..16 {
        dot += a[i] * b[i];
        a_norm += a[i] * a[i];
        b_norm += b[i] * b[i];
    }
    let denom = a_norm.sqrt() * b_norm.sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    let cosine = (dot / denom).clamp(-1.0, 1.0);
    (1.0 - cosine) * 0.5
}

fn normalize_unit(values: &mut [f64; 16]) -> Result<(), GenesisError> {
    let mut norm_sq = 0.0_f64;
    for value in values.iter() {
        norm_sq += value * value;
    }
    if norm_sq < DEGENERATE_NORM_EPSILON_SQ {
        return Err(GenesisError::InvalidInput(
            "degenerate engram: all blades are zero",
        ));
    }
    let norm = norm_sq.sqrt();
    for value in values.iter_mut() {
        *value /= norm;
    }
    Ok(())
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
    fn cluster_consolidation_creates_cortical_abstraction() {
        let mut store = EngramStore::new(256, 0.001);
        for i in 0..100_u64 {
            let mut b = blades(1.0 + (i as f64) * 1e-5);
            b[8] += (i as f64) * 1e-6;
            store.encode(b, i + 1, 10.0, 0).expect("encode");
        }
        store.dream_cycle(1).expect("dream cycle");
        assert!(store.cortical_len() >= 1);
        assert!(store.weighted_strengths(1).iter().any(|(_, w)| *w > 500.0));
    }

    #[test]
    fn predict_next_state_returns_nonzero_for_known_pattern() {
        let mut store = EngramStore::new(16, 0.01);
        let pattern = blades(1.0);
        store.encode(pattern, 7, 3.0, 0).expect("encode");

        let predicted = store
            .predict_next_state(&blades(1.01), 1)
            .expect("prediction")
            .expect("predicted state");

        let norm_sq: f64 = predicted.iter().map(|value| value * value).sum();
        assert!(norm_sq > 1e-12);
    }

    #[test]
    fn intrinsic_surprise_is_high_for_novel_state() {
        let mut empty = EngramStore::new(16, 0.01);
        let surprise = empty
            .calculate_intrinsic_surprise(&blades(1.0), 0)
            .expect("empty surprise");
        assert!((surprise - 1.0).abs() < 1e-12);

        let mut store = EngramStore::new(16, 0.01);
        store.encode(blades(1.0), 1, 1.0, 0).expect("encode");
        let opposite = blades(-1.0);
        let surprise = store
            .calculate_intrinsic_surprise(&opposite, 1)
            .expect("opposite surprise");
        assert!(surprise > 0.8);
    }

    #[test]
    fn trauma_repulsion_penalizes_similar_high_vfe_states() {
        let mut store = EngramStore::new(16, 0.01);
        store.encode(blades(1.0), 1, 10.0, 0).expect("encode");
        let similar = store.episodic_buffer[0].spectral_state;

        let repulsion = store.trauma_repulsion(&similar, 1);
        assert!(repulsion > 0.0);

        let mut axis = [0.0_f64; 16];
        let mut min_idx = 0;
        let mut min_abs = f64::INFINITY;
        for i in 0..16 {
            let abs = similar[i].abs();
            if abs < min_abs {
                min_abs = abs;
                min_idx = i;
            }
        }
        axis[min_idx] = 1.0;
        let projection = similar[min_idx];
        let mut orthogonal = axis;
        for i in 0..16 {
            orthogonal[i] -= projection * similar[i];
        }

        let orthogonal_repulsion = store.trauma_repulsion(&orthogonal, 1);
        assert!(orthogonal_repulsion.abs() < 1e-12);
    }

    #[test]
    fn causal_gap_prevents_acausal_fusion() {
        let mut store = EngramStore::new(16, 0.001);
        let pattern = blades(1.0);
        store.encode(pattern, 0, 10.0, 0).expect("encode first");
        store
            .encode(pattern, 10_000, 10.0, 0)
            .expect("encode second");

        store.dream_cycle(1).expect("dream cycle");

        assert_eq!(store.cortical_len(), 0);
    }

    #[test]
    fn zero_capacity_store_never_panics_and_keeps_no_episodic() {
        let mut store = EngramStore::new(0, 0.1);
        store.encode(blades(1.0), 1, 1.0, 0).expect("encode");
        store.dream_cycle(0).expect("dream cycle");
        assert!(store.causal_ids().is_empty() || store.cortical_len() > 0);
    }
}
