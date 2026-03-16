//! # GÉNESIS Error Taxonomy
//!
//! Centralised, typed error hierarchy for all genesis crates.
//! Every variant maps to a specific layer of the cognitive architecture.
//!
//! **Design contract:** No genesis crate may define its own top-level error
//! enum. All errors must be variants of `GenesisError` or wrap it.
//! This ensures that error propagation never breaks the dependency graph
//! defined in `MACRO_ARCHITECTURE` §3.
//!
//! **`IrrevocableDomainReset` removed (CORRECCIÓN ESTRUCTURAL-2):** The previous
//! `IrrevocableDomainReset` variant was the runtime enforcement of AXIOMA-009.
//! It has been replaced by the compile-time type-state in
//! `DomainConsolidationSignal<State>`: `DomainResetSignal::validate_against`
//! now only accepts `&DomainConsolidationSignal<Saturated>`, making a reset
//! against a `Certified` domain a compile error rather than a runtime error.
//! The `DomainMismatch` variant replaces it for the remaining domain-name
//! mismatch check.
//!
//! AX-ID: Cross-cutting (`MACRO_ARCHITECTURE` §CRATE-000)

#![allow(clippy::must_use_candidate)]

use thiserror::Error;

/// Payload policy tier for [`GenesisError`] variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorPayloadTier {
    /// Tier 1: invariant/fatal path. Payload must be fixed-size and non-alloc.
    Tier1Invariant,
    /// Tier 2: operational path. Contextual allocation is permitted.
    Tier2Operational,
}

/// Compact source tag for algebra signature violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureViolationCode {
    /// Violation detected while constructing from sparse `(index, coeff)` input.
    FromSparseInput,
    /// Violation detected while constructing from a dense `[f64; 16]` buffer.
    FromDenseInput,
    /// Hot-path deterministic strict validation: non-finite `max_abs_coeff` metadata.
    HotPathNonFiniteMetadata,
    /// Hot-path deterministic strict validation: non-finite coefficient in input operands.
    HotPathNonFiniteInputCoeff,
    /// Cold-path deterministic strict finalization: non-finite coefficient in result buffer.
    ColdPathNonFiniteResultCoeff,
}

/// Compact source tag for VFE non-finite terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfeTermCode {
    /// `D_KL(Q||P)` term became NaN/Inf.
    KullbackLeibler,
    /// `-ln P(o)` term became NaN/Inf.
    ObservationLogProb,
    /// Final aggregated free-energy scalar became NaN/Inf.
    AggregateFreeEnergy,
    /// Fallback when a specific term cannot be identified.
    Unknown,
}

/// Non-alloc domain fingerprint used in invariant/signal mismatch paths.
pub type DomainCode = u32;

/// Computes a stable compact domain identifier (FNV-1a 32-bit).
pub const fn domain_code(input: &str) -> DomainCode {
    let bytes = input.as_bytes();
    let mut hash = 0x811c9dc5u32;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(0x01000193);
        i += 1;
    }
    hash
}

// ============================================================================
// TOP-LEVEL ERROR ENUM
// ============================================================================

/// Unified error type for the entire GÉNESIS cognitive core.
///
/// Variants are organised by the crate layer that produces them,
/// matching the dependency hierarchy in `MACRO_ARCHITECTURE` §3.
///
/// ## Silent thermal gate-outs
///
/// The Cauchy-Schwarz Planck gate in `sparse_geometric_product` is **not**
/// represented here. When a product falls below `COGNITIVE_PLANCK_CONSTANT`,
/// the function returns `Option::None`. That signal path must never allocate
/// a `GenesisError` — doing so would violate AXIOMA-001 by adding heap cost
/// to the hot-path. Callers treat `None` as thermal silence, not as failure.
///
/// ## Exhaustiveness
///
/// `#[non_exhaustive]` is applied because new variants will be added as
/// CRATE-001 through CRATE-005 are implemented. All `match` expressions in
/// downstream code must include a wildcard arm (`_ => …`).
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum GenesisError {
    // ------------------------------------------------------------------
    // CRATE-001: genesis-math — Clifford Algebra errors
    // ------------------------------------------------------------------
    /// A geometric product or transformation would violate the Minkowski
    /// signature (1,3). The operation is aborted before any mutation occurs.
    ///
    /// AX-ID: AXIOMA-001
    #[error(
        "Algebra error: Minkowski signature (1,3) violation — code={code:?}, blade_index={blade_index}, normalized_value={normalized_value:?}"
    )]
    SignatureViolation {
        /// Compact violation classification.
        code: SignatureViolationCode,
        /// Blade index associated with the violation when available.
        /// `u16` to support future dimensional expansion beyond 255 blades.
        /// `u16::MAX` = sentinel "unknown".
        blade_index: u16,
        /// Optional normalized value tag for the offending coefficient.
        /// `Some(0)=NaN`, `Some(1)=+Inf`, `Some(2)=-Inf`, `None=unknown/other`.
        normalized_value: Option<u8>,
    },

    /// A basis index exceeds the maximum allowed for G(1,3) (valid: 0..=15).
    ///
    /// AX-ID: AXIOMA-001
    #[error("Algebra error: blade index {index} out of range [0, 15] for G(1,3)")]
    BladeIndexOutOfRange {
        /// The out-of-range blade index. Valid indices for G(1,3) are 0..=15.
        index: usize,
    },

    /// Attempted an operation requiring equal `total_dim` on two vectors
    /// with mismatched dimensionality.
    ///
    /// AX-ID: AXIOMA-001
    #[error("Algebra error: dimension mismatch — lhs.total_dim={lhs}, rhs.total_dim={rhs}")]
    DimensionMismatch {
        /// `total_dim` of the left-hand operand.
        lhs: usize,
        /// `total_dim` of the right-hand operand.
        rhs: usize,
    },

    // ------------------------------------------------------------------
    // CRATE-002: genesis-topology — Manifold / Cohomology errors
    // ------------------------------------------------------------------
    /// A candidate hypothesis has non-trivial first cohomology group H¹ ≠ 0.
    /// The hypothesis is rejected as internally inconsistent (hallucination).
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-009
    #[error("Topology error: H¹ cohomology non-trivial — hypothesis rejected as inconsistent")]
    CohomologyNonTrivial,

    /// HNSW graph navigation found no neighbour within the bivector metric
    /// bound around the query vector.
    ///
    /// AX-ID: AXIOMA-013
    #[error(
        "Topology error: HNSW search returned no neighbours within \
         bivector metric bound {metric_bound:.6}"
    )]
    HnswNoNeighbours {
        /// Upper bound of the bivector metric d(A,B) = ln(|⟨AB̃⟩₂| + ε).
        metric_bound: f64,
    },

    /// An insertion would require global Delaunay re-triangulation, which is
    /// architecturally prohibited (`ENGINEERING_BLUEPRINT` §4.1).
    ///
    /// AX-ID: AXIOMA-013, `ENGINEERING_BLUEPRINT` §4.1
    #[error("Topology error: global Delaunay re-triangulation required — operation forbidden")]
    DelaunayForbidden,

    /// Gram-Schmidt manifold expansion failed: the residual vector is not
    /// sufficiently orthogonal to the current basis to warrant a new dimension.
    ///
    /// AX-ID: AXIOMA-014
    #[error(
        "Topology error: basis expansion failed — residual norm {residual_norm:.6} \
         below orthogonality threshold {threshold:.6}"
    )]
    BasisExpansionFailed {
        /// L2 norm of the residual vector after projection onto the current basis.
        residual_norm: f64,
        /// Minimum residual norm required to justify adding a new basis dimension.
        threshold: f64,
    },

    // ------------------------------------------------------------------
    // CRATE-003: genesis-dynamics — Oscillator / VFE errors
    // ------------------------------------------------------------------
    /// The Kuramoto network attempted to step with zero oscillators.
    ///
    /// AX-ID: AXIOMA-006
    #[error("Dynamics error: Kuramoto network is empty — cannot execute phase step")]
    KuramotoNetworkEmpty,

    /// Duplicate oscillator insertion was attempted for an existing `NodeId`.
    #[error("Dynamics error: duplicate NodeId {raw} in Kuramoto network")]
    KuramotoDuplicateNodeId {
        /// Raw node identifier that already exists in the network.
        raw: u64,
    },

    /// A Lindblad phase collapse was attempted before the synchrony order
    /// parameter reached `SYNCHRONY_COLLAPSE_THRESHOLD`.
    ///
    /// AX-ID: AXIOMA-006
    #[error(
        "Dynamics error: premature Lindblad collapse attempted — \
         order parameter r={order:.4} below threshold {threshold:.4}"
    )]
    PrematureCollapse {
        /// Measured Kuramoto order parameter at the time of the premature attempt.
        order: f64,
        /// Minimum order parameter required to trigger a valid Lindblad collapse.
        threshold: f64,
    },

    /// Free energy computation produced a non-finite value (NaN or ±∞).
    ///
    /// AX-ID: AXIOMA-003
    #[error("Dynamics error: VFE computation produced non-finite value — term={term:?}")]
    VfeNonFinite {
        /// Identifies which term of F = `D_KL(Q||P)` − ln P(o) produced the
        /// non-finite value.
        term: VfeTermCode,
    },

    // ------------------------------------------------------------------
    // CRATE-004: genesis-evolution — Ricci / Fisher / Heat errors
    // ------------------------------------------------------------------
    /// The Sinkhorn transport algorithm failed to converge within
    /// `SINKHORN_MAX_ITER` iterations (Ollivier-Ricci curvature aborted).
    ///
    /// AX-ID: AXIOMA-015, `ENGINEERING_BLUEPRINT` §4.1
    #[error(
        "Evolution error: Sinkhorn-Knopp transport did not converge after {iterations} \
         iterations (residual={residual:.2e})"
    )]
    SinkhornNotConverged {
        /// Number of Sinkhorn-Knopp iterations executed before aborting.
        iterations: usize,
        /// L∞ residual of the transport plan marginals at the time of abort.
        residual: f64,
    },

    /// The Fisher gate has declared domain satiation; ingestion into this
    /// branch is blocked until an explicit `DomainResetSignal` is issued.
    ///
    /// AX-ID: AXIOMA-008
    #[error("Evolution error: domain satiation reached — ingestion blocked for branch '{domain}'")]
    DomainSatiated {
        /// Hierarchical domain identifier whose ingestion channel is now locked.
        domain: &'static str,
    },

    /// A Ricci flow step would produce a degenerate metric (det(g) ≤ 0),
    /// collapsing the manifold geometry. The step is aborted.
    ///
    /// AX-ID: AXIOMA-015
    #[error(
        "Evolution error: Ricci flow step would produce degenerate metric \
         (det={determinant:.2e}) — step aborted"
    )]
    RicciMetricDegenerate {
        /// Computed determinant of the metric tensor after the proposed step.
        determinant: f64,
    },

    // ------------------------------------------------------------------
    // CRATE-005: genesis-io — Projection / Pipeline errors
    // ------------------------------------------------------------------
    /// The sensory projector Π produced a vector whose topological distance
    /// from the original stimulus exceeds the TDA persistence threshold.
    ///
    /// AX-ID: AXIOMA-019
    #[error(
        "IO error: homeomorphism violated — TDA Wasserstein distance {distance:.4} \
         exceeds threshold {threshold:.4}"
    )]
    ProjectionHomeomorphismViolated {
        /// Wasserstein-2 distance between the persistence diagrams of the
        /// original stimulus and its Clifford projection.
        distance: f64,
        /// Maximum acceptable Wasserstein distance for the homeomorphism
        /// invariant to be considered preserved.
        threshold: f64,
    },

    /// The cognitive pipeline firewall blocked manifestation because the
    /// output state's cohomology was non-trivial (H¹ ≠ 0).
    ///
    /// AX-ID: AXIOMA-007, `CLOUD_PLATFORM_ARCHITECTURE` §5.1
    #[error(
        "IO error: cohomological firewall — output state has H¹ ≠ 0; \
         manifestation aborted"
    )]
    FirewallBlocked,

    /// A `ToolFunctor` failed to embed a multivector into a valid external action.
    ///
    /// AX-ID: AXIOMA-017
    #[error("IO error: ToolFunctor embedding failed for action '{action}' — {reason}")]
    FunctorEmbeddingFailed {
        /// Identifier of the external action the functor attempted to embed.
        action: String,
        /// Diagnostic reason for the embedding failure.
        reason: String,
    },

    // ------------------------------------------------------------------
    // CROSS-CUTTING — Architectural Invariant Violations
    // ------------------------------------------------------------------
    /// A forbidden external clock / global time-step was detected in the
    /// execution path.
    ///
    /// AX-ID: AXIOMA-002
    #[error("Invariant violation: global simulation clock detected — forbidden by AXIOMA-002")]
    GlobalClockForbidden,

    /// An external loss function was injected into the learning path.
    /// Only VFE minimisation is permitted as a learning signal.
    ///
    /// `loss_fn` is `&'static str` because this represents a programming error
    /// identified at compile time. Heap allocation for a fatal invariant
    /// violation is architecturally indefensible (`ENGINEERING_BLUEPRINT` §6).
    ///
    /// AX-ID: AXIOMA-003
    #[error(
        "Invariant violation: external loss function '{loss_fn}' injected — \
         forbidden by AXIOMA-003; only VFE minimisation is permitted"
    )]
    ExternalLossForbidden {
        /// Static name of the forbidden loss function (e.g., `"cross_entropy"`).
        loss_fn: &'static str,
    },

    /// A `HashMap` or `BTreeMap` was used in a hot-path computation.
    ///
    /// AX-ID: `MACRO_ARCHITECTURE` §5.2, `ENGINEERING_BLUEPRINT` §6
    #[error(
        "Invariant violation: associative map (HashMap/BTreeMap) used in hot-path \
         '{location}' — forbidden; use flat ordered Vec"
    )]
    HashMapInHotPath {
        /// Static source location where the forbidden map usage was detected.
        location: &'static str,
    },

    /// A `DomainResetSignal` targets a different domain than the
    /// `DomainConsolidationSignal<Saturated>` it was applied against.
    ///
    /// This replaces the old runtime `IrrevocableDomainReset` check, which
    /// is now enforced at compile time via the type-state in
    /// `DomainConsolidationSignal<State>` (CORRECCIÓN ESTRUCTURAL-2).
    ///
    /// AX-ID: AXIOMA-008, AXIOMA-009
    #[error(
        "Signal error: domain mismatch — reset_code=0x{reset:08x}, signal_code=0x{signal:08x}"
    )]
    DomainMismatch {
        /// Compact domain identifier named in the `DomainResetSignal`.
        reset: DomainCode,
        /// Compact domain identifier named in the `DomainConsolidationSignal<Saturated>`.
        signal: DomainCode,
    },

    /// A raw `NodeId` exceeded the supported runtime range.
    ///
    /// Updated by BN-05: the former hard cap of 999_999 is replaced by
    /// `MAX_ALLOWED_NODE_ID = 100_000_000` in `VFEMinimizer`. This variant
    /// is now emitted as an explicit error instead of a silent `return`.
    #[error("Signal error: node id {raw} is out of the supported range")]
    NodeIdOutOfRange {
        /// Raw node identifier supplied by the caller.
        raw: u64,
    },

    /// A caller supplied semantically invalid input for a fallible API.
    ///
    /// AX-ID: Cross-cutting input contract enforcement.
    #[error("Invalid input: {0}")]
    InvalidInput(&'static str),

    /// A `NodeId` exceeded the defensive maximum for `VFEMinimizer`'s direct index.
    ///
    /// The direct-index Vec is bounded at `MAX_ALLOWED_NODE_ID = 100_000_000`
    /// entries to prevent runaway memory allocation from adversarial or buggy IDs.
    /// This is 100× the production target of N~10⁶ nodes.
    ///
    /// AX-ID: BN-05
    #[error("VFE error: NodeId {raw} exceeds maximum allowed index {max_allowed}")]
    NodeIdTooLarge {
        /// Raw NodeId value that triggered the limit.
        raw: usize,
        /// The defensive maximum (`MAX_ALLOWED_NODE_ID = 100_000_000`).
        max_allowed: usize,
    },

    /// A node was not found in the specified structure.
    ///
    /// Returned by `beliefs_raw`, `remove_node`, `remove_oscillator`, and
    /// `ManifoldCollector::remove_node` when the requested node is not registered.
    ///
    /// AX-ID: CRATE-004 prerequisite (FIX-H)
    #[error("Node not found: NodeId {id:?}")]
    NodeNotFound {
        /// The NodeId that was not found.
        id: crate::NodeId,
    },

    /// The internal `Vec` index overflowed `u32::MAX` during `add_node`.
    ///
    /// `VFEMinimizer` uses `Vec<u32>` as its direct index, supporting up to
    /// ~4 billion internal node slots. This error fires only if more than
    /// `u32::MAX` nodes are registered — effectively unreachable in practice.
    ///
    /// AX-ID: BN-05
    #[error("VFE error: internal node index {index} overflows u32::MAX")]
    InternalIndexOverflow {
        /// The internal belief Vec index that overflowed.
        index: usize,
    },

    // ------------------------------------------------------------------
    // PROOF SYSTEM — Mutation certification errors
    // ------------------------------------------------------------------
    /// Una mutación estructural intentó ejecutarse sin generar un Proof válido.
    ///
    /// AX-ID: `LEY_FUNDACIONAL` §5.5 (`ProofGuard`)
    #[error("Proof violation: '{mutation_name}' sin Proof válido")]
    ProofMissing {
        /// Nombre de la mutación que violó el protocolo.
        mutation_name: &'static str,
    },

    /// Un Proof falló verificación (hash corrupto o axioma faltante).
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.4
    #[error("Proof inválido: hash o axioma {axiom_id} fallido")]
    ProofInvalid {
        /// ID del axioma que falló (0-6 según `AxiomID`).
        axiom_id: u8,
    },

    /// Un axioma fue verificado antes de la mutación y resultó falso.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §3
    #[error("Invariante violado: axioma {axiom_id} falló pre-mutación")]
    InvariantViolation {
        /// ID del axioma violado (0-6 según `AxiomID`).
        axiom_id: u8,
    },
}

// ============================================================================
// CLASSIFICATION HELPERS
// ============================================================================

impl GenesisError {
    /// Logging/IO adapter for compact algebra signature violations.
    ///
    /// Keeps hot-path payloads allocation-free while reconstructing a readable
    /// explanation at boundary layers.
    pub fn signature_violation_diagnostic(
        code: SignatureViolationCode,
        blade_index: u16,
        normalized_value: Option<u8>,
    ) -> String {
        let value_hint = match normalized_value {
            Some(0) => "value=nan",
            Some(1) => "value=+inf",
            Some(2) => "value=-inf",
            _ => "value=unknown",
        };

        let where_hint = if blade_index == u16::MAX {
            "index=unknown"
        } else {
            "index=known"
        };

        let reason = match code {
            SignatureViolationCode::FromSparseInput => {
                "non-finite coefficient while constructing from sparse input"
            }
            SignatureViolationCode::FromDenseInput => {
                "non-finite coefficient while constructing from dense input"
            }
            SignatureViolationCode::HotPathNonFiniteMetadata => {
                "hot-path strict validation failed: non-finite max_abs_coeff metadata"
            }
            SignatureViolationCode::HotPathNonFiniteInputCoeff => {
                "hot-path strict validation failed: non-finite input coefficient"
            }
            SignatureViolationCode::ColdPathNonFiniteResultCoeff => {
                "cold-path strict finalization failed: non-finite result coefficient"
            }
        };

        if blade_index == u16::MAX {
            format!("Signature violation detail: reason={reason}; {where_hint}; {value_hint}")
        } else {
            format!(
                "Signature violation detail: reason={reason}; blade_index={blade_index}; {value_hint}"
            )
        }
    }

    /// Classifies this error according to payload policy tiers.
    pub const fn payload_tier(&self) -> ErrorPayloadTier {
        match self {
            Self::SignatureViolation { .. }
            | Self::BladeIndexOutOfRange { .. }
            | Self::DimensionMismatch { .. }
            | Self::CohomologyNonTrivial
            | Self::HnswNoNeighbours { .. }
            | Self::DelaunayForbidden
            | Self::BasisExpansionFailed { .. }
            | Self::KuramotoNetworkEmpty
            | Self::KuramotoDuplicateNodeId { .. }
            | Self::PrematureCollapse { .. }
            | Self::VfeNonFinite { .. }
            | Self::SinkhornNotConverged { .. }
            | Self::RicciMetricDegenerate { .. }
            | Self::ProjectionHomeomorphismViolated { .. }
            | Self::FirewallBlocked
            | Self::GlobalClockForbidden
            | Self::ExternalLossForbidden { .. }
            | Self::HashMapInHotPath { .. }
            | Self::DomainMismatch { .. }
            | Self::NodeIdOutOfRange { .. }
            | Self::NodeIdTooLarge { .. }
            | Self::InternalIndexOverflow { .. }
            | Self::NodeNotFound { .. }
            | Self::ProofMissing { .. }
            | Self::ProofInvalid { .. }
            | Self::InvariantViolation { .. }
            | Self::DomainSatiated { .. } => ErrorPayloadTier::Tier1Invariant,
            Self::FunctorEmbeddingFailed { .. } | Self::InvalidInput(_) => {
                ErrorPayloadTier::Tier2Operational
            }
        }
    }

    /// Expands compact Tier-1 payloads into richer strings at IO/logging boundaries.
    pub fn to_boundary_message(&self) -> String {
        match self {
            Self::SignatureViolation {
                code,
                blade_index,
                normalized_value,
            } => Self::signature_violation_diagnostic(*code, *blade_index, *normalized_value),
            Self::VfeNonFinite { term } => {
                format!("VFE non-finite detail: term={term:?}")
            }
            Self::DomainMismatch { reset, signal } => format!(
                "Domain mismatch detail: reset_code=0x{reset:08x}, signal_code=0x{signal:08x}"
            ),
            _ => self.to_string(),
        }
    }

    /// Returns `true` if this error represents an **architectural invariant
    /// violation** — a programming error that must never occur at runtime.
    ///
    /// These errors should cause a hard panic in debug builds and be treated
    /// as fatal (process abort) in release builds.
    pub const fn is_invariant_violation(&self) -> bool {
        matches!(
            self,
            Self::GlobalClockForbidden
                | Self::ExternalLossForbidden { .. }
                | Self::HashMapInHotPath { .. }
                | Self::DelaunayForbidden
                | Self::ProofMissing { .. }
                | Self::ProofInvalid { .. }
                | Self::InvariantViolation { .. }
        )
    }

    /// Returns `true` if this error is a **topological rejection** that the
    /// pipeline must handle gracefully by discarding the offending input.
    ///
    /// Topological rejections do not indicate bugs — they indicate that an
    /// input or generated state failed the H¹ consistency gate (AXIOMA-007).
    pub const fn is_topological_rejection(&self) -> bool {
        matches!(self, Self::CohomologyNonTrivial | Self::FirewallBlocked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_violation_is_compact_and_boundary_message_expands_it() {
        let err = GenesisError::SignatureViolation {
            code: SignatureViolationCode::FromSparseInput,
            blade_index: 3,
            normalized_value: Some(0),
        };
        assert_eq!(
            err,
            GenesisError::SignatureViolation {
                code: SignatureViolationCode::FromSparseInput,
                blade_index: 3,
                normalized_value: Some(0),
            }
        );

        let msg = err.to_boundary_message();
        assert!(
            msg.contains("non-finite coefficient while constructing from sparse input"),
            "Message: {msg}"
        );
        assert!(msg.contains("blade_index=3"), "Message: {msg}");
    }

    #[test]
    fn signature_violation_diagnostic_supports_hot_and_cold_codes() {
        let hot = GenesisError::signature_violation_diagnostic(
            SignatureViolationCode::HotPathNonFiniteMetadata,
            u16::MAX,
            None,
        );
        assert!(
            hot.contains("hot-path strict validation failed"),
            "Message: {hot}"
        );
        assert!(hot.contains("index=unknown"), "Message: {hot}");

        let cold = GenesisError::signature_violation_diagnostic(
            SignatureViolationCode::ColdPathNonFiniteResultCoeff,
            7,
            Some(1),
        );
        assert!(
            cold.contains("cold-path strict finalization failed"),
            "Message: {cold}"
        );
        assert!(cold.contains("blade_index=7"), "Message: {cold}");
        assert!(cold.contains("value=+inf"), "Message: {cold}");
    }

    #[test]
    fn blade_index_out_of_range_shows_index() {
        let err = GenesisError::BladeIndexOutOfRange { index: 99 };
        let msg = err.to_string();
        assert!(msg.contains("99"), "Message: {msg}");
        assert!(
            msg.contains("15"),
            "Must reference max blade index. Message: {msg}"
        );
    }

    #[test]
    fn dimension_mismatch_shows_both_dims() {
        let err = GenesisError::DimensionMismatch { lhs: 256, rhs: 512 };
        let msg = err.to_string();
        assert!(msg.contains("256"), "Message: {msg}");
        assert!(msg.contains("512"), "Message: {msg}");
    }

    #[test]
    fn hnsw_no_neighbours_uses_metric_bound_not_radius() {
        let err = GenesisError::HnswNoNeighbours { metric_bound: 0.42 };
        let msg = err.to_string();
        assert!(msg.contains("0.420000"), "Message: {msg}");
    }

    #[test]
    fn sinkhorn_not_converged_shows_iterations() {
        let err = GenesisError::SinkhornNotConverged {
            iterations: 1000,
            residual: 3.14e-4,
        };
        let msg = err.to_string();
        assert!(msg.contains("1000"), "Message: {msg}");
    }

    #[test]
    fn global_clock_is_invariant_violation() {
        assert!(GenesisError::GlobalClockForbidden.is_invariant_violation());
    }

    /// ExternalLossForbidden uses &'static str — verify it compiles
    /// without any .to_string() or String allocation.
    ///
    /// AX-ID: ENGINEERING_BLUEPRINT §6
    #[test]
    fn external_loss_forbidden_accepts_static_str_without_allocation() {
        let err = GenesisError::ExternalLossForbidden {
            loss_fn: "cross_entropy",
        };
        assert!(err.is_invariant_violation());
        let msg = err.to_string();
        assert!(msg.contains("cross_entropy"), "Message: {msg}");
    }

    /// HashMapInHotPath uses &'static str — same contract.
    ///
    /// AX-ID: MACRO_ARCHITECTURE §5.2
    #[test]
    fn hashmap_in_hot_path_accepts_static_str_without_allocation() {
        let err = GenesisError::HashMapInHotPath {
            location: "genesis_math::product::sparse_geometric_product",
        };
        assert!(err.is_invariant_violation());
        let msg = err.to_string();
        assert!(msg.contains("genesis_math"), "Message: {msg}");
    }

    #[test]
    fn delaunay_is_invariant_violation() {
        assert!(GenesisError::DelaunayForbidden.is_invariant_violation());
    }

    #[test]
    fn cohomology_non_trivial_is_topological_rejection() {
        assert!(GenesisError::CohomologyNonTrivial.is_topological_rejection());
    }

    #[test]
    fn firewall_blocked_is_topological_rejection() {
        assert!(GenesisError::FirewallBlocked.is_topological_rejection());
    }

    #[test]
    fn signature_violation_is_not_topological_rejection() {
        let err = GenesisError::SignatureViolation {
            code: SignatureViolationCode::FromDenseInput,
            blade_index: 0,
            normalized_value: None,
        };
        assert!(!err.is_topological_rejection());
    }

    #[test]
    fn error_clone_equals_original() {
        let err = GenesisError::BladeIndexOutOfRange { index: 7 };
        assert_eq!(err, err.clone());
    }

    #[test]
    fn distinct_errors_not_equal() {
        let a = GenesisError::GlobalClockForbidden;
        let b = GenesisError::FirewallBlocked;
        assert_ne!(a, b);
    }

    #[test]
    fn domain_satiated_message_contains_domain() {
        let err = GenesisError::DomainSatiated {
            domain: "physics::electromagnetism",
        };
        let msg = err.to_string();
        assert!(msg.contains("physics::electromagnetism"), "Message: {msg}");
    }

    /// DomainMismatch replaces the runtime IrrevocableDomainReset check.
    ///
    /// AX-ID: AXIOMA-008, AXIOMA-009
    #[test]
    fn domain_mismatch_message_contains_both_domains() {
        let err = GenesisError::DomainMismatch {
            reset: domain_code("math::clifford"),
            signal: domain_code("physics::em"),
        };
        let msg = err.to_string();
        assert!(msg.contains("reset_code"), "Message: {msg}");
        assert!(msg.contains("signal_code"), "Message: {msg}");
        // DomainMismatch is NOT an invariant violation (it is a signal error).
        assert!(!err.is_invariant_violation());
        assert!(!err.is_topological_rejection());
    }

    /// Verify the three classification buckets are mutually exclusive for
    /// their canonical representatives.
    ///
    /// AX-ID: Cross-cutting
    #[test]
    fn classification_is_mutually_exclusive() {
        let inv = GenesisError::GlobalClockForbidden;
        let topo = GenesisError::CohomologyNonTrivial;
        let sig = GenesisError::DomainMismatch {
            reset: domain_code("a"),
            signal: domain_code("b"),
        };

        assert!(inv.is_invariant_violation());
        assert!(!inv.is_topological_rejection());

        assert!(topo.is_topological_rejection());
        assert!(!topo.is_invariant_violation());

        assert!(!sig.is_invariant_violation());
        assert!(!sig.is_topological_rejection());
    }

    #[test]
    fn payload_tiers_split_invariant_and_operational_errors() {
        assert_eq!(
            GenesisError::GlobalClockForbidden.payload_tier(),
            ErrorPayloadTier::Tier1Invariant
        );
        assert_eq!(
            GenesisError::DomainSatiated {
                domain: "physics::electromagnetism",
            }
            .payload_tier(),
            ErrorPayloadTier::Tier1Invariant
        );
    }
}
