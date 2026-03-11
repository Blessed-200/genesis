//! # genesis-dynamics — Motor de Resonancia Cognitiva
//!
//! Implementa `H_dinámica` y `H_información` del Hamiltoniano total.
//! Cada módulo deriva sus ecuaciones de `LEY_FUNDACIONAL.md`.
//!
//! ## Módulos
//! - `oscillator`: `QuantumOscillator` — 5 fases por nodo (grados Clifford 0..=4)
//! - `kuramoto`: `QuantumKuramotoNetwork` — integración Euler-Maruyama
//! - `synchrony`: parámetro de orden `r_sync`, cluster sincronizado
//! - `free_energy`: `VFEMinimizer` — único motor de aprendizaje (AXIOMA-003)
//! - `attractor`: `AttractorLandscape` — descenso por gradiente en `E(x)`
//! - `criticality`: `CriticalityMonitor` — SOC `P(S) ∝ S^{-τ}` (AXIOMA-005)
//!
//! AX-ID: AXIOMA-003, AXIOMA-004, AXIOMA-005, AXIOMA-006,
//!        `H_dinámica`, `H_información` (`LEY_FUNDACIONAL` §3.2, §3.3)

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::manual_range_contains,
    clippy::must_use_candidate,
    clippy::unreadable_literal
)]

pub mod attractor;
pub mod criticality;
pub mod free_energy;
pub mod kuramoto;
pub mod oscillator;
pub mod synchrony;

pub use attractor::AttractorLandscape;
pub use criticality::{
    kuramoto_critical_coupling, CriticalityMonitor, CriticalityReport, SOC_R_SYNC_MAX, SOC_R_SYNC_MIN,
};
pub use free_energy::{Belief, FisherEdgeMetric, FisherInfo, VFEMinimizer};
pub use kuramoto::QuantumKuramotoNetwork;
pub use oscillator::QuantumOscillator;
pub use synchrony::{synchronized_cluster, synchrony_order, synchrony_order_fast};
