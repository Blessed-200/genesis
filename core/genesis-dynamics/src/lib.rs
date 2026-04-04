//! # genesis-dynamics — Cognitive Resonance Engine
//!
//! This crate implements `H_dinámica` and `H_información` from the global
//! Hamiltonian and provides the runtime dynamics layer for the GÉNESIS system.
//! The implemented equations follow the formal derivation in
//! `LEY_FUNDACIONAL.md`.
//!
//! ## Modules
//! - `oscillator`: `QuantumOscillator` state evolution over five Clifford grades
//!   (0..=4) with amplitude and saturation controls.
//! - `kuramoto`: `QuantumKuramotoNetwork` integration via Euler–Maruyama for
//!   coupled oscillator updates.
//! - `synchrony`: order parameter `r_sync` computation and synchronized-cluster
//!   extraction.
//! - `free_energy`: `VFEMinimizer` as the sole learning engine
//!   (AXIOMA-003).
//! - `attractor`: `AttractorLandscape` gradient-based descent over attractor
//!   energy `E(x)`.
//! - `criticality`: `CriticalityMonitor` for SOC regime tracking
//!   `P(S) ∝ S^{-τ}` (AXIOMA-005).
//!
//! ## SIMD compilation options
//! Some Kuramoto hot-path primitives use compile-time SIMD feature gating:
//! AVX2 on x86_64 and NEON on aarch64. Without explicit target features,
//! the crate uses the portable scalar fallback.
//!
//! - Portable default: `cargo build -p genesis-dynamics --release`
//! - AVX2/FMA (x86_64): `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build -p genesis-dynamics --release`
//! - NEON (aarch64): `RUSTFLAGS="-C target-feature=+neon" cargo build -p genesis-dynamics --release`
//!
//! AX-ID: AXIOMA-003, AXIOMA-004, AXIOMA-005, AXIOMA-006,
//!        `H_dinámica`, `H_información` (`LEY_FUNDACIONAL` §3.2, §3.3)
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
#![allow(dead_code)] // internal utility fns used by tests

/// Kuramoto attractor energy landscape — concept recognition via gradient descent.
///
/// AX-ID: AXIOMA-004, `H_información`
pub mod attractor;
/// Self-Organised Criticality monitor — ensures system operates near the phase transition.
///
/// AX-ID: AXIOMA-005
pub mod criticality;
/// Variational Free Energy minimiser — the sole learning mechanism.
///
/// AX-ID: AXIOMA-003, AXIOMA-008, `H_información`
pub mod free_energy;
/// Compensated summation primitives for numerically stable reductions.
///
/// AX-ID: AXIOMA-006, `H_dinámica`
pub mod kahan;
/// Quantum Kuramoto network — synchronisation dynamics over G(1,3) phases.
///
/// AX-ID: AXIOMA-006, `H_dinámica`
pub mod kuramoto;
/// Quantum oscillator state: amplitude, phase, and saturation per Clifford grade.
///
/// AX-ID: AXIOMA-006
pub mod oscillator;
/// Phase semantics engine — interpretable cognitive field over oscillator dynamics.
///
/// AX-ID: AXIOMA-004, AXIOMA-006
pub mod phase_semantics;
/// Synchrony order parameters and cluster extraction from Kuramoto network.
///
/// AX-ID: AXIOMA-006
pub mod synchrony;

pub use attractor::AttractorLandscape;
pub use criticality::{
    kuramoto_critical_coupling, CriticalityMonitor, CriticalityReport, SOC_R_SYNC_MAX,
    SOC_R_SYNC_MIN,
};
pub use free_energy::{Belief, FisherEdgeMetric, FisherInfo, VFEMinimizer};
pub use genesis_types::{
    CognitiveFieldState, MetaState, NetworkSemanticState, NodeSemanticState, PhaseRegion,
    SemanticCluster, SemanticMarker, SemanticTensionEdge, SemanticTrace,
};
pub use kuramoto::QuantumKuramotoNetwork;
pub use oscillator::QuantumOscillator;
pub use phase_semantics::{ClusterRejectionReason, PhaseSemanticsEngine};
pub use synchrony::{synchronized_cluster, synchrony_order, synchrony_order_fast};
