use genesis_types::NodeId;
use genesis_dynamics::VFEMinimizer;
use genesis_transport::{vfe_t, JtBreakdown, JtTerms};

fn sample_terms() -> JtTerms {
    JtTerms {
        vfe_t: 0.7,
        causal_penalty_t: 0.2,
        transport_cost_t: 1.1,
        spectral_instability_t: 0.4,
    }
}

#[test]
fn jt_traceability_is_deterministic_and_reconstructible() {
    let breakdown = JtBreakdown::from_terms(sample_terms());
    assert!((breakdown.j_t - 2.4).abs() < 1e-12);
    assert!((breakdown.reconstruct() - breakdown.j_t).abs() < 1e-12);

    let contributions = breakdown.contributions();
    assert!((contributions[0] - breakdown.terms.vfe_t).abs() < 1e-12);
    assert!((contributions[1] - breakdown.terms.causal_penalty_t).abs() < 1e-12);
    assert!((contributions[2] - breakdown.terms.transport_cost_t).abs() < 1e-12);
    assert!((contributions[3] - breakdown.terms.spectral_instability_t).abs() < 1e-12);
}

#[test]
fn jt_term_sensitivity_is_linear_and_isolated() {
    let baseline = JtBreakdown::from_terms(sample_terms());
    let mut modified = sample_terms();
    modified.transport_cost_t += 0.35;

    let shifted = JtBreakdown::from_terms(modified);
    let delta = shifted.j_t - baseline.j_t;
    assert!((delta - 0.35).abs() < 1e-12);

    let mut modified2 = sample_terms();
    modified2.spectral_instability_t -= 0.1;
    let shifted2 = JtBreakdown::from_terms(modified2);
    let delta2 = shifted2.j_t - baseline.j_t;
    assert!((delta2 + 0.1).abs() < 1e-12);
}

#[test]
fn compatibility_with_base_crates_is_preserved() {
    let mut minimizer = VFEMinimizer::new();
    let node_id = NodeId::try_new(7).expect("valid node id");
    minimizer.add_node(node_id, [0.0; 4]);
    let obs = [0.0_f64; 4];
    let vfe = vfe_t(&minimizer, node_id, Some(&obs));

    let breakdown = JtBreakdown::from_terms(JtTerms {
        vfe_t: vfe,
        causal_penalty_t: 0.0,
        transport_cost_t: 0.0,
        spectral_instability_t: 0.0,
    });

    assert!(breakdown.j_t.is_finite());
    assert!((breakdown.j_t - vfe).abs() < 1e-12);
}
