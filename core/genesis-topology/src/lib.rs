#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::semicolon_if_nothing_returned,
    clippy::uninlined_format_args,
    clippy::unreadable_literal
)]

pub mod cohomology;
pub mod incremental_cohomology;
/// genesis-topology — The Cognitive Manifold
///
/// CRATE-002: provides the topological substrate where GÉNESIS concepts exist.
/// All distance operations use the exclusive bivector metric from genesis-math.
///
/// AX-ID: AXIOMA-007, AXIOMA-009, AXIOMA-013, AXIOMA-014
/// H implemented: `H_restricción` (H¹, λ₂, edge density)
/// Ω contribution: λ₂, H¹, `edge_density_ratio`
pub mod geodesic;
pub mod hnsw;
pub mod lsh;
pub mod manifold;
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
