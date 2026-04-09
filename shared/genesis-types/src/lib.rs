//! # genesis-types — GÉNESIS Primitive Layer (CRATE-000)
//!
//! The **foundation** of the GÉNESIS cognitive core. This crate has:
//!
//! - **Zero** internal genesis dependencies (leaf node in the dependency DAG).
//! - **No** cognitive logic — only types, constants, errors, and signals.
//! - **No** hot-path code — all items here are O(1) or trivially small.
//!
//! Every other genesis crate depends on this one directly.
//! Nothing in this crate may import from `genesis-math` or above.
//!
//! ## API Changes from Audit Rev 1
//!
//! - `SpikeComponents::from_pairs` now takes `(u16, f64)` pairs.
//! - `SpikeComponents` exposes `indices`, `values`, `count` (`SoA`); `presence_mask`
//!   is removed.
//! - `SpikeEvent::timestamp_ns` is `Timestamp`, `origin_node_id` is `NodeId`.
//! - `SpikeEvent::collapse_grade` is `Option<u16>`.
//! - `DomainConsolidationSignal<State>` requires a state type parameter
//!   (`Saturated` or `Certified`).
//! - `ConsolidationKind` enum is removed; `is_irrevocable()` is removed.
//! - `DomainResetSignal::domain` and `justification` are `&'static str`.
//! - `DomainResetSignal::timestamp_ns` is `Timestamp`.
//! - `FISHER_SATIATION_WINDOW` and `SINKHORN_MAX_ITER` are `NonZeroUsize`.
//!
//! AX-ID: Cross-cutting (`MACRO_ARCHITECTURE` §CRATE-000)

#![deny(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::no_effect_underscore_binding,
    clippy::semicolon_if_nothing_returned,
    clippy::uninlined_format_args,
    clippy::unreadable_literal
)]

pub mod constants;
pub mod error;
pub mod fisher_edge;
mod multivector_types;
pub mod phase_semantics;
pub mod proof;
pub mod quantities;
pub mod signal;

// ============================================================================
// FLAT RE-EXPORTS
// ============================================================================

// — Constants —
pub use constants::{
    validate_constant_ordering,
    CLIFFORD_BASIS_SIZE,
    COGNITIVE_PLANCK_CONSTANT,
    DELTA_DUALITY,
    // Variational extensions v1.1.0
    DENSITY_PENALTY,
    FISHER_SATIATION_EPSILON,
    FISHER_SATIATION_WINDOW,
    HEAT_DIFFUSION_CONVERGENCE_EPSILON,
    JL_RESIDUAL_EXPANSION_DELTA,
    KAPPA_REDUNDANCY,
    // Proof system
    LAMBDA2_MIN,
    LAMBDA_DIM_FIXED,
    LAMBDA_DIM_LOG,
    MAX_CLIFFORD_GRADE,
    METRIC_WEIGHTS,
    MINKOWSKI_SIGNATURE,
    PHASE_DISTINCTION_THRESHOLD,
    PROOF_MAX_AGE_NS,
    PROOF_PENALTY,
    REDUNDANCY_RADIUS,
    SINKHORN_CONVERGENCE_EPSILON,
    SINKHORN_MAX_ITER,
    SINKHORN_REGULARISATION,
    SOC_TAU_MAX,
    SOC_TAU_MIN,
    SPATIAL_BASIS_MASK,
    SYNCHRONY_COLLAPSE_THRESHOLD,
    WORMHOLE_CURVATURE_THRESHOLD,
};
// — Errors —
pub use error::GenesisError;
// — Fast non-cryptographic hash tables —
pub use ahash::{AHashMap, AHashSet};

/// Fast hash map alias for non-cryptographic table operations.
///
/// AX-ID: AXIOMA-017, H_estructura (LEY_FUNDACIONAL §3.1)
pub type FastHashMap<K, V> = AHashMap<K, V>;

/// Fast hash set alias for non-cryptographic membership operations.
///
/// AX-ID: AXIOMA-017, H_estructura (LEY_FUNDACIONAL §3.1)
pub type FastHashSet<T> = AHashSet<T>;
// — Fisher Edge Metric —
pub use fisher_edge::FisherEdgeMetric;
// — Multivector Types —
pub use multivector_types::DerivedMetadata;
// — Proof System —
pub use proof::{AxiomGuard, AxiomID, Mutation, Proof, WitnessBuilder};
pub use quantities::{
    Amplitude, ComplexPhasor, Frequency, LearningRate, Phase, SyncOrder, Temperature, TimeStep,
};
// — Phase semantics primitives —
pub use phase_semantics::{
    CognitiveFieldState, MetaState, NetworkSemanticState, NodeSemanticState, PhaseRegion,
    SemanticCluster, SemanticMarker, SemanticTensionEdge, SemanticTrace,
    SEMANTIC_CLUSTER_MAX_NODES,
};
// — Signals —
pub use signal::{
    BladeIndex, Certified, CognitiveDomain, ConsciousnessDomain, DomainConsolidationSignal,
    CouplingEdge, DomainResetSignal, DomainSignal, DynamicsDomain, GaussianPair, NodeId,
    PhysicsDomain, Saturated, SpikeComponents, SpikeComponentsError, SpikeEvent, Timestamp,
    TopologyDomain, SPIKE_MAX_COMPONENTS,
};

/// Curated constants grouped by semantic domain.
pub mod domains {
    /// Core Clifford substrate constants.
    pub mod substrate {
        pub use crate::constants::{
            CLIFFORD_BASIS_SIZE, COGNITIVE_PLANCK_CONSTANT, MAX_CLIFFORD_GRADE,
            MINKOWSKI_SIGNATURE, SPATIAL_BASIS_MASK,
        };
    }

    /// Runtime control and convergence constants.
    pub mod control {
        pub use crate::constants::{
            FISHER_SATIATION_EPSILON, FISHER_SATIATION_WINDOW, HEAT_DIFFUSION_CONVERGENCE_EPSILON,
            JL_RESIDUAL_EXPANSION_DELTA, SINKHORN_CONVERGENCE_EPSILON, SINKHORN_MAX_ITER,
            SINKHORN_REGULARISATION, SOC_TAU_MAX, SOC_TAU_MIN, SYNCHRONY_COLLAPSE_THRESHOLD,
            WORMHOLE_CURVATURE_THRESHOLD,
        };
    }
}

/// Minimal import surface for downstream crates.
pub mod prelude {
    pub use crate::constants::{
        COGNITIVE_PLANCK_CONSTANT, MINKOWSKI_SIGNATURE, SPATIAL_BASIS_MASK,
    };
    pub use crate::error::GenesisError;
    pub use crate::proof::{AxiomGuard, AxiomID, Mutation, Proof};
    pub use crate::signal::{NodeId, SpikeEvent, Timestamp};
}

#[cfg(test)]
#[test]
fn derived_metadata_layout_is_stable() {
    assert_eq!(core::mem::size_of::<DerivedMetadata>(), 64);
    assert_eq!(core::mem::align_of::<DerivedMetadata>(), 64);
}

// ============================================================================
// CRATE-LEVEL INTEGRATION TESTS
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Verify that the full public surface compiles and is accessible
    /// from the crate root — simulates how downstream crates import.
    ///
    /// AX-ID: Cross-cutting (MACRO_ARCHITECTURE §CRATE-000)
    #[test]
    fn all_public_exports_accessible() {
        // Constants
        let _ = COGNITIVE_PLANCK_CONSTANT;
        let _ = MINKOWSKI_SIGNATURE;
        let _ = SPATIAL_BASIS_MASK;
        let _ = FISHER_SATIATION_EPSILON;
        let _ = FISHER_SATIATION_WINDOW.get(); // NonZeroUsize
        let _ = SOC_TAU_MIN;
        let _ = SOC_TAU_MAX;
        let _ = SYNCHRONY_COLLAPSE_THRESHOLD;
        let _ = WORMHOLE_CURVATURE_THRESHOLD;
        let _ = MAX_CLIFFORD_GRADE;
        let _ = CLIFFORD_BASIS_SIZE;
        let _ = SINKHORN_MAX_ITER.get(); // NonZeroUsize
        let _ = SINKHORN_CONVERGENCE_EPSILON;
        let _ = SINKHORN_REGULARISATION;
        let _ = JL_RESIDUAL_EXPANSION_DELTA;
        let _ = HEAT_DIFFUSION_CONVERGENCE_EPSILON;
        let _ = SPIKE_MAX_COMPONENTS;
        let _ = METRIC_WEIGHTS;

        // Phase-semantics primitives
        let _phase_region = PhaseRegion::Certainty;
        let _marker = SemanticMarker::Exploration;
        let _metastate = MetaState::ExploratoryFlux;
        let _node_state = NodeSemanticState {
            node: NodeId::try_new(1).expect("1 is inside the valid NodeId range"),
            marker: SemanticMarker::Integration,
            phase: 0.0,
            amplitude: 1.0,
            stability: 1.0,
        };
        let _field_state = CognitiveFieldState {
            dominant_marker: SemanticMarker::Certainty,
            coherence: 1.0,
            semantic_entropy: 0.0,
            tension: 0.0,
        };
        let _network_state = NetworkSemanticState {
            dominant_state: SemanticMarker::Certainty,
            coherence: 1.0,
            diversity: 0.0,
            metastability: 0.0,
        };
        let _tension_edge = SemanticTensionEdge {
            node_a: NodeId::try_new(2).expect("2 is inside the valid NodeId range"),
            node_b: NodeId::try_new(3).expect("3 is inside the valid NodeId range"),
            divergence: 0.5,
        };
        let _trace = SemanticTrace {
            previous_marker: SemanticMarker::Conflict,
            duration: 4,
        };
        let _cluster = SemanticCluster::from_nodes(
            &[
                NodeId::try_new(5).expect("5 is inside the valid NodeId range"),
                NodeId::try_new(8).expect("8 is inside the valid NodeId range"),
            ],
            SemanticMarker::Exploration,
            0.75,
        )
        .expect("cluster fits fixed-capacity representation");

        // Proof system constants
        let _ = LAMBDA2_MIN;
        let _ = PROOF_PENALTY;
        let _ = PROOF_MAX_AGE_NS;
        let _ = DENSITY_PENALTY;
        let _ = DELTA_DUALITY;
        let _ = KAPPA_REDUNDANCY;
        let _ = REDUNDANCY_RADIUS;
        let _ = PHASE_DISTINCTION_THRESHOLD;
        let _ = LAMBDA_DIM_FIXED;
        let _ = LAMBDA_DIM_LOG;

        // Newtypes
        let _node = NodeId::try_new(0).expect("0 is inside the valid NodeId range");
        let _ts = Timestamp::new(0);

        // Errors
        let _ = GenesisError::GlobalClockForbidden;
        let _ = GenesisError::CohomologyNonTrivial;
        let _ = GenesisError::FirewallBlocked;

        // SpikeComponents — from_pairs now takes (u16, f64).
        let components = SpikeComponents::from_pairs([] as [(u16, f64); 0]);
        let spike = SpikeEvent::new(
            components,
            Timestamp::new(0),
            NodeId::try_new(0).expect("0 is inside the valid NodeId range"),
            16u64,
            None,
        );
        assert_eq!(spike.cardinality(), 0);

        // DomainConsolidationSignal — requires explicit state type.
        let sat = DomainConsolidationSignal::<Saturated>::saturated("test", Timestamp::new(0), 0.0);
        assert_eq!(sat.domain, "test");

        let cert = DomainConsolidationSignal::<Certified>::certified("test2", Timestamp::new(1));
        assert_eq!(cert.domain, "test2");

        // DomainResetSignal — &'static str fields, Timestamp for timestamp_ns.
        let reset = DomainResetSignal {
            domain: "test",
            timestamp_ns: Timestamp::new(0),
            justification: "ok",
        };
        assert_eq!(reset.domain, "test");

        // Proof system types
        let _ = AxiomID::MinkowskiSignature;
        let _ = AxiomID::DimensionalAdmission;
        let _ = AxiomID::EXPANSION_REQUIRED;
        assert_eq!(AxiomID::EXPANSION_REQUIRED.len(), 7);
        let mut builder = WitnessBuilder::new();
        builder.check(AxiomID::PlanckConstant, || true).unwrap();
        let proof = builder.build(0);
        assert!(AxiomGuard::verify(&proof, &[AxiomID::PlanckConstant]));
    }

    /// Verify the Minkowski signature invariant is consistent with
    /// `SPATIAL_BASIS_MASK`.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-002
    #[test]
    fn spatial_mask_consistent_with_minkowski_signature() {
        for (i, &sig) in MINKOWSKI_SIGNATURE.iter().enumerate() {
            let bit_set = (SPATIAL_BASIS_MASK & (1 << i)) != 0;
            let is_spatial = sig < 0.0;
            assert_eq!(
                bit_set, is_spatial,
                "Bit {} of SPATIAL_BASIS_MASK inconsistent with MINKOWSKI_SIGNATURE[{}]={}",
                i, i, sig
            );
        }
    }

    /// Verify the error classification system does not misclassify canonical
    /// representatives.
    ///
    /// AX-ID: Cross-cutting
    #[test]
    fn error_classification_is_mutually_exclusive_for_key_errors() {
        let inv = GenesisError::GlobalClockForbidden;
        let topo = GenesisError::CohomologyNonTrivial;

        assert!(inv.is_invariant_violation());
        assert!(!inv.is_topological_rejection());

        assert!(topo.is_topological_rejection());
        assert!(!topo.is_invariant_violation());
    }

    /// Verify `SpikeEvent` is `Copy` — compile-time proof of zero per-spike
    /// heap allocation.
    ///
    /// AX-ID: AXIOMA-018
    #[test]
    fn spike_event_copy_bound_compile_time_no_heap() {
        fn require_copy<T: Copy>(_: T) {}
        let sc = SpikeComponents::from_pairs([(0u16, 1.0)]);
        let e = SpikeEvent::new(
            sc,
            Timestamp::new(1),
            NodeId::try_new(1).expect("1 is inside the valid NodeId range"),
            16u64,
            Some(0u16),
        );
        require_copy(e); // Compile error if SpikeEvent is not Copy
        let _copy = e; // Second use — valid only if e is Copy
        let _ = e.cardinality(); // Original still usable
    }

    /// `DomainResetSignal::validate_against` accepts `Saturated` and
    /// returns `Ok(())` when domains match.
    ///
    /// AX-ID: AXIOMA-008, AXIOMA-009
    #[test]
    fn domain_reset_validate_against_saturated_ok_from_crate_root() {
        let certified = DomainConsolidationSignal::<Saturated>::saturated(
            "math::topology",
            Timestamp::new(1),
            1e-8,
        );
        let reset = DomainResetSignal {
            domain: "math::topology",
            timestamp_ns: Timestamp::new(2),
            justification: "test",
        };
        assert!(reset.validate_against(&certified).is_ok());
    }

    /// Passing `DomainConsolidationSignal<Certified>` to `validate_against`
    /// is a **compile error**. This test documents the compile-time guarantee.
    ///
    /// If the body were uncommented, rustc would emit:
    ///   error[E0308]: mismatched types
    ///     expected `&DomainConsolidationSignal<Saturated>`
    ///     found    `&DomainConsolidationSignal<Certified>`
    ///
    /// AX-ID: AXIOMA-009
    #[test]
    fn domain_reset_validate_certified_is_compile_error() {
        // DO NOT UNCOMMENT:
        //
        // let certified = DomainConsolidationSignal::<Certified>::certified("math::topology", Timestamp::new(1));
        // let reset = DomainResetSignal {
        //     domain:        "math::topology",
        //     timestamp_ns:  Timestamp::new(2),
        //     justification: "test",
        // };
        // let _ = reset.validate_against(&certified); // error[E0308]
    }

    /// Verify that blade indices > 255 (u16 range) are accepted.
    /// This was the primary blocker (CONVERGENTE-1) with the old u8 design.
    ///
    /// AX-ID: AXIOMA-014 (GramSchmidtExpander expansion to D > 10,000)
    #[test]
    fn spike_components_accepts_blade_index_above_255() {
        let sc = SpikeComponents::from_pairs([(256u16, 1.0), (1000u16, -0.5)]);
        assert_eq!(sc.cardinality(), 2);
        assert!(sc.is_active(256));
        assert_eq!(sc.get(1000), -0.5);
    }

    /// Verify `FISHER_SATIATION_WINDOW` and `SINKHORN_MAX_ITER` are
    /// `NonZeroUsize` and accessible via `.get()`.
    ///
    /// AX-ID: AXIOMA-008, AXIOMA-015
    #[test]
    fn nonzero_usize_constants_accessible_via_get() {
        assert_eq!(FISHER_SATIATION_WINDOW.get(), 50);
        assert_eq!(SINKHORN_MAX_ITER.get(), 1_000);
    }
}
