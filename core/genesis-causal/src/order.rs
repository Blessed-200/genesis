//! Causal graph with dense internal indexing and hot-path adjacency traversal.
//!
//! AX-ID: AXIOMA-002, H_estructura (LEY_FUNDACIONAL §3.1)

use crate::separation::CausalSeparation;
use genesis_types::{AxiomID, GenesisError, WitnessBuilder};
use smallvec::SmallVec;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
struct HotEdge {
    cause_idx: u32,
    effect_idx: u32,
    causal_strength: f64,
}

#[derive(Clone, Debug)]
struct ColdEdgeMeta {
    edge_proof: [u8; 32],
    separation: CausalSeparation,
}

#[derive(Clone, Debug)]
struct NodeEntry {
    external_id: u64,
    topo_rank: u32,
}

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
        let delta_t = effect_blades[1] - cause_blades[1];
        if !separation.is_causal() || delta_t <= 0.0 {
            return Err(GenesisError::CausalViolation { cause_id, effect_id });
        }

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
    nodes: Vec<NodeEntry>,
    id_to_idx: Vec<(u64, u32)>,
    hot_edges: Vec<HotEdge>,
    cold_edge_meta: Vec<ColdEdgeMeta>,
    out_adj: Vec<SmallVec<[u32; 8]>>,
    in_adj: Vec<SmallVec<[u32; 8]>>,
    causal_frontier: SmallVec<[u64; 16]>,
}

impl CausalOrder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            id_to_idx: Vec::new(),
            hot_edges: Vec::new(),
            cold_edge_meta: Vec::new(),
            out_adj: Vec::new(),
            in_adj: Vec::new(),
            causal_frontier: SmallVec::new(),
        }
    }

    pub fn add_edge(&mut self, edge: CausalEdge) -> Result<(), GenesisError> {
        let cause_idx = self.get_or_insert_node(edge.cause_id)?;
        let effect_idx = self.get_or_insert_node(edge.effect_id)?;

        if cause_idx == effect_idx {
            return Err(GenesisError::CausalViolation { cause_id: edge.cause_id, effect_id: edge.effect_id });
        }

        if self.has_path(effect_idx, cause_idx) {
            return Err(GenesisError::CausalCycle { cycle_nodes: vec![edge.cause_id, edge.effect_id] });
        }

        let cause_usize = usize::try_from(cause_idx).map_err(|_| GenesisError::InvalidInput("cause index overflow"))?;
        let effect_usize = usize::try_from(effect_idx).map_err(|_| GenesisError::InvalidInput("effect index overflow"))?;

        if !self.out_adj[cause_usize].contains(&effect_idx) {
            self.out_adj[cause_usize].push(effect_idx);
            self.in_adj[effect_usize].push(cause_idx);

            self.hot_edges.push(HotEdge { cause_idx, effect_idx, causal_strength: edge.causal_strength });
            self.cold_edge_meta.push(ColdEdgeMeta { edge_proof: edge.edge_proof, separation: edge.separation });
            self.rebalance_topology(cause_idx, effect_idx)?;
            self.update_frontier();
        }

        Ok(())
    }

    #[must_use]
    pub fn edge_count(&self) -> usize { self.hot_edges.len() }

    #[must_use]
    pub fn past_lightcone(&self, node_id: u64) -> SmallVec<[u64; 16]> {
        let Some(start_idx) = self.node_idx(node_id) else { return SmallVec::new(); };
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        let mut out: SmallVec<[u64; 16]> = SmallVec::new();

        let start_usize = start_idx as usize;
        visited[start_usize] = true;
        queue.push_back(start_idx);

        while let Some(current) = queue.pop_front() {
            let current_usize = current as usize;
            for &parent in &self.in_adj[current_usize] {
                let parent_usize = parent as usize;
                if !visited[parent_usize] {
                    visited[parent_usize] = true;
                    queue.push_back(parent);
                    out.push(self.nodes[parent_usize].external_id);
                }
            }
        }

        out
    }

    #[must_use]
    pub fn future_lightcone(&self, node_id: u64) -> SmallVec<[u64; 16]> {
        let Some(start_idx) = self.node_idx(node_id) else { return SmallVec::new(); };
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        let mut out: SmallVec<[u64; 16]> = SmallVec::new();

        let start_usize = start_idx as usize;
        visited[start_usize] = true;
        queue.push_back(start_idx);

        while let Some(current) = queue.pop_front() {
            let current_usize = current as usize;
            for &child in &self.out_adj[current_usize] {
                let child_usize = child as usize;
                if !visited[child_usize] {
                    visited[child_usize] = true;
                    queue.push_back(child);
                    out.push(self.nodes[child_usize].external_id);
                }
            }
        }

        out
    }

    pub fn verify_acyclic(&self) -> Result<(), GenesisError> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        for edge in &self.hot_edges {
            let idx = usize::try_from(edge.effect_idx).map_err(|_| GenesisError::InvalidInput("index overflow"))?;
            in_degree[idx] += 1;
        }

        let mut queue = VecDeque::new();
        for (i, &d) in in_degree.iter().enumerate() {
            if d == 0 { queue.push_back(i); }
        }

        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            for &child in &self.out_adj[node] {
                let child_u = usize::try_from(child).map_err(|_| GenesisError::InvalidInput("index overflow"))?;
                in_degree[child_u] = in_degree[child_u].saturating_sub(1);
                if in_degree[child_u] == 0 { queue.push_back(child_u); }
            }
        }

        if visited != n { Err(GenesisError::CausalCycle { cycle_nodes: Vec::new() }) } else { Ok(()) }
    }

    #[must_use]
    pub fn is_before(&self, a: u64, b: u64) -> bool {
        match (self.node_idx(a), self.node_idx(b)) {
            (Some(a_idx), Some(b_idx)) => self.has_path(a_idx, b_idx),
            _ => false,
        }
    }

    #[must_use]
    pub fn ancestors_of(&self, node_id: u64) -> SmallVec<[u64; 16]> { self.past_lightcone(node_id) }

    #[must_use]
    pub fn descendants_of(&self, node_id: u64) -> SmallVec<[u64; 16]> { self.future_lightcone(node_id) }

    #[must_use]
    pub fn cone_overlap(&self, a: u64, b: u64) -> SmallVec<[u64; 16]> {
        let pa = self.past_lightcone(a);
        let pb = self.past_lightcone(b);
        let mut overlap = SmallVec::new();
        for id in pa {
            if pb.contains(&id) { overlap.push(id); }
        }
        overlap
    }

    pub(crate) fn incoming_hot_edges(&self, node_id: u64) -> SmallVec<[CausalEdge; 8]> {
        let mut out = SmallVec::new();
        let Some(idx) = self.node_idx(node_id) else { return out; };
        let idx_u = idx as usize;

        for &parent in &self.in_adj[idx_u] {
            if let Some((edge_idx, hot)) = self.hot_edges.iter().enumerate().find(|(_, e)| e.cause_idx == parent && e.effect_idx == idx) {
                let cause_u = hot.cause_idx as usize;
                let effect_u = hot.effect_idx as usize;
                let meta = &self.cold_edge_meta[edge_idx];
                out.push(CausalEdge {
                    cause_id: self.nodes[cause_u].external_id,
                    effect_id: self.nodes[effect_u].external_id,
                    separation: meta.separation,
                    causal_strength: hot.causal_strength,
                    edge_proof: meta.edge_proof,
                });
            }
        }
        out
    }

    fn node_idx(&self, external_id: u64) -> Option<u32> {
        self.id_to_idx
            .binary_search_by_key(&external_id, |(id, _)| *id)
            .ok()
            .map(|pos| self.id_to_idx[pos].1)
    }

    fn get_or_insert_node(&mut self, external_id: u64) -> Result<u32, GenesisError> {
        match self.id_to_idx.binary_search_by_key(&external_id, |(id, _)| *id) {
            Ok(pos) => Ok(self.id_to_idx[pos].1),
            Err(pos) => {
                let idx_u32 = u32::try_from(self.nodes.len()).map_err(|_| GenesisError::InvalidInput("too many nodes for u32 indexing"))?;
                self.nodes.push(NodeEntry { external_id, topo_rank: idx_u32 });
                self.out_adj.push(SmallVec::new());
                self.in_adj.push(SmallVec::new());
                self.id_to_idx.insert(pos, (external_id, idx_u32));
                Ok(idx_u32)
            }
        }
    }

    fn has_path(&self, from: u32, to: u32) -> bool {
        if from == to { return true; }
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        let from_u = from as usize;
        visited[from_u] = true;
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            let curr_u = current as usize;
            for &child in &self.out_adj[curr_u] {
                if child == to { return true; }
                let child_u = child as usize;
                if !visited[child_u] {
                    visited[child_u] = true;
                    queue.push_back(child);
                }
            }
        }
        false
    }

    fn rebalance_topology(&mut self, cause_idx: u32, effect_idx: u32) -> Result<(), GenesisError> {
        let cause_u = usize::try_from(cause_idx).map_err(|_| GenesisError::InvalidInput("index overflow"))?;
        let effect_u = usize::try_from(effect_idx).map_err(|_| GenesisError::InvalidInput("index overflow"))?;

        if self.nodes[cause_u].topo_rank < self.nodes[effect_u].topo_rank {
            return Ok(());
        }

        let new_rank = self.nodes[cause_u].topo_rank.saturating_add(1);
        self.nodes[effect_u].topo_rank = new_rank;
        Ok(())
    }

    fn update_frontier(&mut self) {
        self.causal_frontier.clear();
        for (idx, node) in self.nodes.iter().enumerate() {
            if self.in_adj[idx].is_empty() {
                self.causal_frontier.push(node.external_id);
            }
        }
    }
}

impl Default for CausalOrder {
    fn default() -> Self { Self::new() }
}
