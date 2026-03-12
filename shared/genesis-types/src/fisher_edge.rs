//! Métrica de Fisher entre pares de nodos: G^Fisher_{ij}.
//!
//! Expuesta en genesis-types para que genesis-evolution (CRATE-004) pueda
//! verificar DualityConsistency sin importar genesis-dynamics (CRATE-003).
//!
//! AX-ID: LEY_FUNDACIONAL §3.6, §5.6 (DualityConsistency)
//
// AX-ID: LEY_FUNDACIONAL §3.6, §5.6 (DualityConsistency)

use crate::NodeId;

/// Métrica de Fisher escalarizada por arista (i,j): G^Fisher_{ij}.
///
/// Almacena pares canónicos (i ≤ j) ordenados por (i,j) para búsqueda O(log E).
/// `is_current(n)` responde si el nodo n tiene su Fisher actualizado tras la
/// última expansión dimensional — requerido por AxiomID::DualityConsistency.
///
/// AX-ID: AXIOMA-008, LEY_FUNDACIONAL §3.6
#[derive(Clone, Debug, Default)]
pub struct FisherEdgeMetric {
    edges: Vec<((NodeId, NodeId), f64)>,
    node_degrees: Vec<(NodeId, u32)>,
}

impl FisherEdgeMetric {
    /// Construye una nueva `FisherEdgeMetric` desde un vector de aristas.
    ///
    /// Los pares se normalizan a forma canónica (i ≤ j) y se ordenan.
    pub fn new(edges: Vec<((NodeId, NodeId), f64)>) -> Self {
        let mut metric = Self {
            edges,
            node_degrees: Vec::new(),
        };
        metric.normalize();
        metric
    }

    /// Obtiene el valor de Fisher para la arista (i, j). O(log E).
    /// Retorna 0.0 si la arista no existe.
    pub fn get(&self, i: NodeId, j: NodeId) -> f64 {
        let key = canonical_edge(i, j);
        self.edges
            .binary_search_by_key(&key, |(pair, _)| *pair)
            .map_or(0.0, |idx| self.edges[idx].1)
    }

    /// Update the Fisher value for an existing edge. O(log E).
    ///
    /// If the edge exists, replaces its value (no re-sort needed — edges maintain
    /// order by (i,j) pair, not by value). If the edge does not exist, inserts it.
    ///
    /// This is the primary mutation path for `DiscreteRicciFlow` (CRATE-004):
    /// each Ricci step updates O(E) edge values without rebuilding the sorted Vec.
    /// For existing edges: O(log E). For new edges: O(E) shift (rare during Ricci).
    ///
    /// AX-ID: LEY_FUNDACIONAL §3.6, BN-10
    pub fn set(&mut self, i: NodeId, j: NodeId, value: f64) {
        let key = canonical_edge(i, j);
        match self.edges.binary_search_by_key(&key, |(pair, _)| *pair) {
            Ok(pos) => {
                // Edge exists — update value in-place, no re-sort needed.
                self.edges[pos].1 = value;
            }
            Err(pos) => {
                // New edge — insert at sorted position.
                self.edges.insert(pos, (key, value));
                self.increment_degree(key.0);
                self.increment_degree(key.1);
            }
        }
    }

    /// Remove an edge from the metric. O(E) shift. O(log E) search.
    ///
    /// Removes both the edge value and updates `current_nodes` if either node
    /// no longer appears in any edge.
    ///
    /// AX-ID: LEY_FUNDACIONAL §3.6, BN-10
    pub fn remove(&mut self, i: NodeId, j: NodeId) {
        let key = canonical_edge(i, j);
        if let Ok(pos) = self.edges.binary_search_by_key(&key, |(pair, _)| *pair) {
            self.edges.remove(pos);
            self.decrement_degree(key.0);
            self.decrement_degree(key.1);
        }
    }

    /// Itera sobre todas las aristas como (i, j, valor).
    pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, f64)> + '_ {
        self.edges.iter().map(|((i, j), w)| (*i, *j, *w))
    }

    /// Retorna true si el nodo `n` tiene su Fisher actualizado.
    /// Requerido por AxiomID::DualityConsistency.
    pub fn is_current(&self, n: NodeId) -> bool {
        self.node_degrees
            .binary_search_by_key(&n, |(id, _)| *id)
            .is_ok()
    }

    fn increment_degree(&mut self, n: NodeId) {
        match self.node_degrees.binary_search_by_key(&n, |(id, _)| *id) {
            Ok(pos) => {
                self.node_degrees[pos].1 = self.node_degrees[pos].1.saturating_add(1);
            }
            Err(pos) => {
                self.node_degrees.insert(pos, (n, 1));
            }
        }
    }

    fn decrement_degree(&mut self, n: NodeId) {
        if let Ok(pos) = self.node_degrees.binary_search_by_key(&n, |(id, _)| *id) {
            let degree = self.node_degrees[pos].1;
            if degree <= 1 {
                self.node_degrees.remove(pos);
            } else {
                self.node_degrees[pos].1 = degree - 1;
            }
        }
    }

    fn normalize(&mut self) {
        for ((i, j), _) in &mut self.edges {
            if j < i {
                core::mem::swap(i, j);
            }
        }
        self.edges.sort_unstable_by_key(|(key, _)| *key);
        self.edges.dedup_by(|lhs, rhs| lhs.0 == rhs.0);

        self.node_degrees.clear();
        for i in 0..self.edges.len() {
            let (left, right) = self.edges[i].0;
            self.increment_degree(left);
            self.increment_degree(right);
        }
    }
}

fn canonical_edge(i: NodeId, j: NodeId) -> (NodeId, NodeId) {
    if i <= j {
        (i, j)
    } else {
        (j, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(n: u64) -> NodeId {
        NodeId::try_new(n).expect("NodeId válido")
    }

    #[test]
    fn get_existing_edge_returns_value() {
        let m = FisherEdgeMetric::new(vec![((node(0), node(1)), 0.5)]);
        assert!((m.get(node(0), node(1)) - 0.5).abs() < 1e-15);
        assert!((m.get(node(1), node(0)) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn get_missing_edge_returns_zero() {
        let m = FisherEdgeMetric::new(vec![]);
        assert_eq!(m.get(node(0), node(1)), 0.0);
    }

    #[test]
    fn is_current_reflects_nodes_in_edges() {
        let m = FisherEdgeMetric::new(vec![((node(2), node(5)), 1.0)]);
        assert!(m.is_current(node(2)));
        assert!(m.is_current(node(5)));
        assert!(!m.is_current(node(0)));
    }

    #[test]
    fn edges_iterator_yields_all_entries() {
        let m = FisherEdgeMetric::new(vec![((node(0), node(1)), 0.3), ((node(1), node(2)), 0.7)]);
        let v: Vec<_> = m.edges().collect();
        assert_eq!(v.len(), 2);
    }
    #[test]
    fn set_updates_existing_edge_in_place() {
        let mut m = FisherEdgeMetric::new(vec![((node(0), node(1)), 0.5)]);
        m.set(node(0), node(1), 0.9);
        assert!(
            (m.get(node(0), node(1)) - 0.9).abs() < 1e-15,
            "set must update existing edge"
        );
        assert_eq!(m.edges.len(), 1, "no duplicate created");
    }

    #[test]
    fn set_inserts_new_edge() {
        let mut m = FisherEdgeMetric::new(vec![]);
        m.set(node(3), node(7), 0.4);
        assert!((m.get(node(3), node(7)) - 0.4).abs() < 1e-15);
        assert_eq!(m.edges.len(), 1);
        assert!(m.is_current(node(3)));
        assert!(m.is_current(node(7)));
    }

    #[test]
    fn remove_deletes_edge_and_updates_current_nodes() {
        let mut m =
            FisherEdgeMetric::new(vec![((node(0), node(1)), 0.5), ((node(1), node(2)), 0.7)]);
        m.remove(node(0), node(1));
        assert_eq!(m.get(node(0), node(1)), 0.0, "removed edge returns 0");
        assert!(!m.is_current(node(0)), "node 0 no longer has edges");
        assert!(m.is_current(node(1)), "node 1 still has edge to node 2");
    }

    #[test]
    fn set_maintains_sorted_order_for_binary_search() {
        let mut m = FisherEdgeMetric::new(vec![]);
        // Insert in non-sequential order
        m.set(node(5), node(9), 1.0);
        m.set(node(1), node(3), 2.0);
        m.set(node(2), node(7), 3.0);
        // All lookups must work (relies on sorted order)
        assert!((m.get(node(5), node(9)) - 1.0).abs() < 1e-15);
        assert!((m.get(node(1), node(3)) - 2.0).abs() < 1e-15);
        assert!((m.get(node(2), node(7)) - 3.0).abs() < 1e-15);
    }
}
