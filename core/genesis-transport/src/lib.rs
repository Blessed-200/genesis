//! `genesis-transport` — Causally consistent optimal transport on G(1,3).
//!
//! CRATE-006 in the GÉNESIS workspace.
//!
//! AX-ID: AXIOMA-001, AXIOMA-002, AXIOMA-003, `H_compresión`

#![deny(missing_docs)]

/// Belief distributions over 16 blades.
pub mod belief;
/// Boundary projections Π and Π*.
pub mod boundary;
/// Lorentzian cost matrix primitives.
pub mod cost;
/// Implicit JKO descent scheme.
pub mod jko;
/// Causally constrained Sinkhorn solver.
pub mod sinkhorn;

pub use belief::BeliefDistribution;
pub use boundary::{SensoryEvent, SensoryProjector};
pub use cost::CostMatrix16;
pub use jko::{JKOScheme, JKOStep};
pub use sinkhorn::CausalSinkhorn;
