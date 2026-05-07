//! genesis-causal crate.
//!
//! Causal structure, lightcone filtering, and spinor coherence over G(1,3).
//!
//! AX-ID: AXIOMA-002, H_estructura (LEY_FUNDACIONAL §3.1)

pub mod error;
pub mod lightcone;
pub mod order;
pub mod separation;
pub mod spinor;

pub use lightcone::{CausalInference, LightconeFilter};
pub use order::{CausalEdge, CausalOrder};
pub use separation::{CausalSeparation, LIGHTLIKE_TOL};
pub use spinor::{DiracSpinor, GlobalSection};

#[cfg(test)]
mod tests {
    use super::*;

    fn blades_txyz(t: f64, x: f64, y: f64, z: f64) -> [f64; 16] {
        let mut b = [0.0; 16];
        b[1] = t;
        b[2] = x;
        b[4] = y;
        b[8] = z;
        b
    }

    #[test]
    fn causal_order_rejects_cycles() {
        let a = blades_txyz(0.0, 0.0, 0.0, 0.0);
        let b = blades_txyz(1.0, 0.0, 0.0, 0.0);
        let mut order = CausalOrder::new();
        let ab = CausalEdge::compute(1, &a, 2, &b).expect("A->B edge should build");
        order.add_edge(ab).expect("A->B should be inserted");
        let ba = CausalEdge::compute(2, &b, 1, &a).expect("B->A edge should build");
        let err = order.add_edge(ba).expect_err("B->A should create a cycle");
        assert!(matches!(err, genesis_types::GenesisError::CausalCycle { .. }));
    }

    #[test]
    fn lightcone_filter_rejects_acausal_inputs() {
        let a = blades_txyz(0.0, 0.0, 0.0, 0.0);
        let b = blades_txyz(1.0, 0.0, 0.0, 0.0);
        let c = blades_txyz(2.0, 0.0, 0.0, 0.0);
        let mut order = CausalOrder::new();
        order.add_edge(CausalEdge::compute(1, &a, 2, &b).expect("A->B edge should build")).expect("A->B should insert");
        order.add_edge(CausalEdge::compute(2, &b, 3, &c).expect("B->C edge should build")).expect("B->C should insert");

        let filter = LightconeFilter::new(&order);
        let inputs = filter.causal_inputs(1, &[2, 3]);
        assert!(inputs.is_empty());
    }

    #[test]
    fn global_section_coherence_order_is_normalized() {
        let order = CausalOrder::new();
        let root = DiracSpinor { components: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
        let section = GlobalSection::compute(&order, &[10, 20, 30], root).expect("section should compute");
        assert!((0.0..=1.0).contains(&section.coherence_order));
    }

    #[test]
    fn spacelike_separation_not_added_to_dag() {
        let a = blades_txyz(0.0, 0.0, 0.0, 0.0);
        let b = blades_txyz(0.1, 1.0, 0.0, 0.0);
        let edge = CausalEdge::compute(1, &a, 2, &b).expect("edge should compute");
        assert!(matches!(edge.separation, CausalSeparation::Spacelike { .. }));

        let mut order = CausalOrder::new();
        order.add_edge(edge).expect("spacelike edge should be ignored without error");
        assert_eq!(order.edge_count(), 0);
    }

    #[test]
    fn verify_acyclic_passes_on_dag() {
        let a = blades_txyz(0.0, 0.0, 0.0, 0.0);
        let b = blades_txyz(1.0, 0.0, 0.0, 0.0);
        let c = blades_txyz(2.0, 0.0, 0.0, 0.0);
        let mut order = CausalOrder::new();
        order.add_edge(CausalEdge::compute(1, &a, 2, &b).expect("A->B edge should build")).expect("A->B should insert");
        order.add_edge(CausalEdge::compute(2, &b, 3, &c).expect("B->C edge should build")).expect("B->C should insert");
        order.verify_acyclic().expect("valid DAG should remain acyclic");
    }

    #[test]
    fn causal_strength_lightlike_is_maximum() {
        let a = blades_txyz(0.0, 0.0, 0.0, 0.0);
        let b = blades_txyz(1.0, 1.0, 0.0, 0.0);
        let edge = CausalEdge::compute(1, &a, 2, &b).expect("lightlike edge");
        assert!(matches!(edge.separation, CausalSeparation::Lightlike));
        assert!((edge.causal_strength - 1.0).abs() < 1e-10);
    }

    #[test]
    fn past_lightcone_is_transitive() {
        let a = blades_txyz(0.0, 0.0, 0.0, 0.0);
        let b = blades_txyz(1.0, 0.0, 0.0, 0.0);
        let c = blades_txyz(2.0, 0.0, 0.0, 0.0);
        let mut order = CausalOrder::new();
        order.add_edge(CausalEdge::compute(1, &a, 2, &b).expect("A->B edge")).expect("insert A->B");
        order.add_edge(CausalEdge::compute(2, &b, 3, &c).expect("B->C edge")).expect("insert B->C");
        let past_c = order.past_lightcone(3);
        assert!(past_c.contains(&1), "A should be in past of C");
        assert!(past_c.contains(&2), "B should be in past of C");
    }

    #[test]
    fn long_cycle_detected() {
        let nodes: Vec<_> = (0..4)
            .map(|i| {
                let mut b = [0.0_f64; 16];
                b[1] = i as f64;
                b
            })
            .collect();
        let mut order = CausalOrder::new();
        order.add_edge(CausalEdge::compute(0, &nodes[0], 1, &nodes[1]).expect("0->1")).expect("insert 0->1");
        order.add_edge(CausalEdge::compute(1, &nodes[1], 2, &nodes[2]).expect("1->2")).expect("insert 1->2");
        order.add_edge(CausalEdge::compute(2, &nodes[2], 3, &nodes[3]).expect("2->3")).expect("insert 2->3");

        if let Ok(edge) = CausalEdge::compute(3, &nodes[3], 0, &nodes[0]) {
            assert!(order.add_edge(edge).is_err(), "long cycle must be rejected");
        }
    }

    #[test]
    fn global_section_holonomy_zero_on_chain() {
        let a = blades_txyz(0.0, 0.0, 0.0, 0.0);
        let b = blades_txyz(1.0, 0.0, 0.0, 0.0);
        let c = blades_txyz(2.0, 0.0, 0.0, 0.0);
        let mut order = CausalOrder::new();
        order.add_edge(CausalEdge::compute(1, &a, 2, &b).expect("A->B edge")).expect("insert A->B");
        order.add_edge(CausalEdge::compute(2, &b, 3, &c).expect("B->C edge")).expect("insert B->C");

        let root = DiracSpinor { components: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
        let section = GlobalSection::compute(&order, &[1, 2, 3], root).expect("compute section");
        assert!(section.holonomy < 1e-10, "holonomy must be zero on linear chain");
    }

}
