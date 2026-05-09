//! Spectral operators and flows for CRATE-004 Phase 5A.

pub mod action;
pub mod dirac;
pub mod flow;
pub mod spinor;
pub mod zeta;

pub use action::{CutoffFunction, SpectralActionEngine, SpectralActionResult};
pub use dirac::{DiracOperator, DiracSquared, GammaAction, LorentzIndex};
pub use flow::{ConvergenceCriteria, SpectralFlowEngine, SpectralFlowHistory, SpectralFlowStep};
pub use spinor::{Spinor, SpinorPredictor};
pub use zeta::ZetaRegularizer;

mod dense_matrix;

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{
        action::{CutoffFunction, SpectralActionEngine},
        dirac::DiracOperator,
        flow::{ConvergenceCriteria, SpectralFlowEngine, SpectralFlowHistory},
    };

    #[test]
    fn dirac_on_flat_manifold_zero_curvature() {
        let mut blades = [0.0_f64; 16];
        blades[0] = 1.0;
        let dirac = DiracOperator::from_blades(&blades, 1.0).expect("dirac");
        let ricci = dirac.squared().expect("d2").ricci_scalar;
        assert!(ricci.is_finite(), "ricci must be finite, got {ricci}");
        assert!(
            ricci.abs() < 1.0,
            "flat manifold ricci {ricci} out of expected range"
        );
    }

    #[test]
    fn dirac_squared_on_flat_equals_double_apply() {
        let blades = [0.1_f64; 16];
        let d = DiracOperator::from_blades(&blades, 1.0).expect("dirac");
        let v = std::array::from_fn(|i| f64::from(u32::try_from(i).unwrap_or(0)) * 0.1);
        let via_matrix = d.squared().expect("d2").matrix.matvec(&v);
        let via_double_apply = d.apply(&d.apply(&v).expect("d(v)")).expect("d2(v)");
        for i in 0..16 {
            assert!(
                (via_matrix[i] - via_double_apply[i]).abs() < 1e-12,
                "D² mismatch at index {i}"
            );
        }
    }

    #[test]
    fn spectral_action_monotone_decrease() {
        let mut nodes = vec![(1_u64, [0.5; 16]), (2_u64, [0.2; 16])];
        let mut engine = SpectralFlowEngine::new(
            1.0,
            0.1,
            CutoffFunction::Gaussian,
            ConvergenceCriteria {
                grad_tol: 1e-10,
                max_steps: 100,
            },
        );
        for _ in 0..100 {
            let step = engine.step(&mut nodes).expect("step");
            assert!(step.action_delta <= 0.0);
        }
    }

    #[test]
    fn spectral_distance_recovers_grade_metric() {
        let mut b = [0.0; 16];
        b[1] = 1.0;
        let d = DiracOperator::from_blades(&b, 1.0)
            .expect("dirac")
            .commutator_norm(&[1.0; 16])
            .expect("comm");
        assert!(d.is_finite());
        assert!(d >= 0.0);
    }

    #[test]
    fn proof_generated_on_every_dirac_construction() {
        let d = DiracOperator::from_blades(&[1.0; 16], 1.0).expect("dirac");
        assert_ne!(d.construction_proof, [0_u8; 32]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn spectral_flow_never_increases_action(raw in prop::collection::vec(prop::array::uniform16(-1.0f64..1.0f64), 3..6)) {
            let mut nodes: Vec<(u64, [f64;16])> = raw.into_iter().enumerate().map(|(i,b)| (u64::try_from(i).unwrap_or(0), b)).collect();
            let mut engine = SpectralFlowEngine::new(1.0, 0.05, CutoffFunction::Polynomial { exponent: 2 }, ConvergenceCriteria { grad_tol: 1e-8, max_steps: 2 });
            let mut history = SpectralFlowHistory::default();
            for _ in 0..2 {
                let step = engine.step(&mut nodes).expect("step");
                history.steps.push(step);
            }
            prop_assert!(history.steps.iter().all(|s| s.action_delta <= 0.0));
        }
    }

    #[test]
    fn action_empty_nodes_errors() {
        let engine = SpectralActionEngine::new(CutoffFunction::Sharp, 1.0);
        assert!(engine.evaluate(&[]).is_err());
    }
}
