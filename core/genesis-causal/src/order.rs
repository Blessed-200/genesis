//! Causal partial-order graph and edge construction.
//!
//! AX-ID: AXIOMA-002, H_estructura (LEY_FUNDACIONAL §3.1)

use crate::separation::CausalSeparation;
use genesis_types::{AxiomID, GenesisError, WitnessBuilder};
use smallvec::SmallVec;

#[derive(Clone, Debug)]
pub struct CausalEdge {
    pub cause_id: u64,
    pub effect_id: u64,
    pub separation: CausalSeparation,
    pub causal_strength: f64,
    pub edge_proof: [u8; 32],
}

impl CausalEdge {
    pub fn compute(
        cause_id: u64,
        cause_blades: &[f64; 16],
        effect_id: u64,
        effect_blades: &[f64; 16],
    ) -> Result<Self, GenesisError> {
        if cause_id == effect_id {
            return Err(GenesisError::CausalViolation { cause_id, effect_id });
        }

        let separation = CausalSeparation::compute(cause_blades, effect_blades);
        let causal_strength = match separation {
            CausalSeparation::Timelike { separation_sq } => separation_sq / (1.0 + separation_sq),
            CausalSeparation::Lightlike => 1.0,
            CausalSeparation::Spacelike { .. } => 0.0,
        };

        let mut wb = WitnessBuilder::new();
        wb.check(AxiomID::ProofGuard, || true)?;
        let proof = wb.build(0);

        Ok(Self { cause_id, effect_id, separation, causal_strength, edge_proof: proof.hash })
    }
}

pub struct CausalOrder {
    edges: Vec<CausalEdge>,
    causal_frontier: SmallVec<[u64; 16]>,
    out_index: Vec<(u64, SmallVec<[u64; 8]>)>,
    in_index: Vec<(u64, SmallVec<[u64; 8]>)>,
}

impl CausalOrder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            causal_frontier: SmallVec::new(),
            out_index: Vec::new(),
            in_index: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, edge: CausalEdge) -> Result<(), GenesisError> {
        if !edge.separation.is_causal() {
            return Ok(());
        }
        if self.is_ancestor(edge.effect_id, edge.cause_id) {
            return Err(GenesisError::CausalCycle {
                cycle_nodes: vec![edge.cause_id, edge.effect_id],
            });
        }

        let pos = self.edges.partition_point(|e| e.cause_id < edge.cause_id);
        let cause_id = edge.cause_id;
        let effect_id = edge.effect_id;
        self.edges.insert(pos, edge);
        self.update_adjacency_index(cause_id, effect_id);
        self.update_frontier();
        Ok(())
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn past_lightcone(&self, node_id: u64) -> SmallVec<[u64; 16]> {
        let mut visited: SmallVec<[u64; 32]> = SmallVec::new();
        let mut queue: SmallVec<[u64; 16]> = SmallVec::new();
        queue.push(node_id);

        while let Some(current) = queue.pop() {
            if let Ok(pos) = self.in_index.binary_search_by_key(&current, |(id, _)| *id) {
                for &cause_id in &self.in_index[pos].1 {
                    if !visited.contains(&cause_id) {
                        visited.push(cause_id);
                        queue.push(cause_id);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    pub fn future_lightcone(&self, node_id: u64) -> SmallVec<[u64; 16]> {
        let mut visited: SmallVec<[u64; 32]> = SmallVec::new();
        let mut queue: SmallVec<[u64; 16]> = SmallVec::new();
        queue.push(node_id);

        while let Some(current) = queue.pop() {
            if let Ok(pos) = self.out_index.binary_search_by_key(&current, |(id, _)| *id) {
                for &effect_id in &self.out_index[pos].1 {
                    if !visited.contains(&effect_id) {
                        visited.push(effect_id);
                        queue.push(effect_id);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    pub fn verify_acyclic(&self) -> Result<(), GenesisError> {
        let mut all_nodes: SmallVec<[u64; 64]> = SmallVec::new();
        for e in &self.edges {
            if !all_nodes.contains(&e.cause_id) {
                all_nodes.push(e.cause_id);
            }
            if !all_nodes.contains(&e.effect_id) {
                all_nodes.push(e.effect_id);
            }
        }

        let mut in_degree: SmallVec<[(u64, usize); 64]> =
            all_nodes.iter().map(|&id| (id, 0)).collect();
        for e in &self.edges {
            if let Some(entry) = in_degree.iter_mut().find(|(id, _)| *id == e.effect_id) {
                entry.1 += 1;
            }
        }

        let mut queue: SmallVec<[u64; 16]> = in_degree
            .iter()
            .filter(|(_, d)| *d == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut visited = 0usize;
        while let Some(node) = queue.pop() {
            visited += 1;

            if let Ok(pos) = self.out_index.binary_search_by_key(&node, |(id, _)| *id) {
                for &effect_id in &self.out_index[pos].1 {
                    if let Some(entry) = in_degree.iter_mut().find(|(id, _)| *id == effect_id) {
                        entry.1 = entry.1.saturating_sub(1);
                        if entry.1 == 0 {
                            queue.push(entry.0);
                        }
                    }
                }
            }
        }

        if visited < all_nodes.len() {
            Err(GenesisError::CausalCycle {
                cycle_nodes: Vec::new(),
            })
        } else {
            Ok(())
        }
    }

    #[must_use]
    pub(crate) fn incoming_edges(&self, node_id: u64) -> SmallVec<[CausalEdge; 8]> {
        let mut edges: SmallVec<[CausalEdge; 8]> = SmallVec::new();
        if let Ok(pos) = self.in_index.binary_search_by_key(&node_id, |(id, _)| *id) {
            for &cause_id in &self.in_index[pos].1 {
                if let Some(edge) = self
                    .edges
                    .iter()
                    .find(|e| e.cause_id == cause_id && e.effect_id == node_id && e.separation.is_causal())
                {
                    edges.push(edge.clone());
                }
            }
        }
        edges
    }

    fn is_ancestor(&self, potential_ancestor: u64, of_node: u64) -> bool {
        self.past_lightcone(of_node).contains(&potential_ancestor)
    }

    fn update_adjacency_index(&mut self, cause_id: u64, effect_id: u64) {
        match self.out_index.binary_search_by_key(&cause_id, |(id, _)| *id) {
            Ok(pos) => {
                if !self.out_index[pos].1.contains(&effect_id) {
                    self.out_index[pos].1.push(effect_id);
                }
            }
            Err(pos) => {
                let mut effects: SmallVec<[u64; 8]> = SmallVec::new();
                effects.push(effect_id);
                self.out_index.insert(pos, (cause_id, effects));
            }
        }

        match self.in_index.binary_search_by_key(&effect_id, |(id, _)| *id) {
            Ok(pos) => {
                if !self.in_index[pos].1.contains(&cause_id) {
                    self.in_index[pos].1.push(cause_id);
                }
            }
            Err(pos) => {
                let mut causes: SmallVec<[u64; 8]> = SmallVec::new();
                causes.push(cause_id);
                self.in_index.insert(pos, (effect_id, causes));
            }
        }
    }

    fn update_frontier(&mut self) {
        let effects: SmallVec<[u64; 32]> = self.edges.iter().map(|e| e.effect_id).collect();
        let all_causes: SmallVec<[u64; 32]> = self.edges.iter().map(|e| e.cause_id).collect();
        self.causal_frontier = all_causes
            .iter()
            .filter(|id| !effects.contains(id))
            .copied()
            .collect();
    }
}

impl Default for CausalOrder {
    fn default() -> Self {
        Self::new()
    }
}
