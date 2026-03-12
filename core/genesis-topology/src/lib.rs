//! # genesis-topology — Cognitive Manifold Substrate (CRATE-002)
//!
//! Provides the topological infrastructure: HNSW graph for O(log N) semantic
//! search, incremental H¹ cohomology for consistency validation, spectral
//! connectivity (λ₂), and edge density monitoring for `H_restricción`.
//!
//! AX-ID: AXIOMA-007, AXIOMA-009, AXIOMA-013, AXIOMA-014
//!
//! # Compiler directives — lint policy
//!
//! These crate-level lints enforce production-grade engineering standards.
//! All public API must be documented. All unsafe must be justified.
//! All clippy::pedantic issues not explicitly allowed must be zero.
#![deny(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::missing_errors_doc,    // added in favour of explicit fallibility docs
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::semicolon_if_nothing_returned,
    clippy::uninlined_format_args,
)]

/// Cohomology validation — H¹ = 0 filter for hypothesis consistency.
///
/// AX-ID: AXIOMA-007, AXIOMA-009
pub mod cohomology;
/// genesis-topology — The Cognitive Manifold
///
/// CRATE-002 — The Cognitive Manifold
///
/// CRATE-002: provides the topological substrate where GÉNESIS concepts exist.
/// All distance operations use the exclusive bivector metric from genesis-math.
///
/// AX-ID: AXIOMA-007, AXIOMA-009, AXIOMA-013, AXIOMA-014
/// H implemented: `H_restricción` (H¹, λ₂, edge density)
/// Ω contribution: λ₂, H¹, `edge_density_ratio`
/// Bivector distance metric for G(1,3) — the HNSW traversal metric.
pub mod geodesic;
/// Hierarchical Navigable Small World graph for O(log N) ANN search.
///
/// AX-ID: AXIOMA-013
pub mod hnsw;
/// Incremental H¹ computation via union-find + incremental boundary matrix.
///
/// Near-O(α(N)) amortised for streaming edge insertions. AX-ID: AXIOMA-007
pub mod incremental_cohomology;
/// Locality-Sensitive Hashing over Clifford vectors for O(log N) lookup.
///
/// AX-ID: AXIOMA-013
pub mod lsh;
/// ManifoldCollector — orchestrates HNSW + H¹ + λ₂ with Hamiltonian hooks.
///
/// AX-ID: AXIOMA-007, AXIOMA-013
pub mod manifold;
/// Vietoris-Rips complex construction up to dimension 2.
///
/// AX-ID: AXIOMA-007
pub mod rips;

pub use cohomology::CohomologyValidator;
#[allow(deprecated)]
pub use geodesic::fast_bivector_distance;
pub use geodesic::{bivector_interaction, geometric_distance};
pub use hnsw::HnswGraph;
pub use incremental_cohomology::IncrementalH1State;
pub use lsh::CliffordHashTable;
pub use manifold::{HyperbolicCoord, ManifoldCollector};
pub use rips::RipsComplex;
