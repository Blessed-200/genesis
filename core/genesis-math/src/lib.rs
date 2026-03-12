//! `genesis-math` — G(1,3) Clifford algebra substrate.
//!
//! CRATE-001 in the GÉNESIS workspace.
//!
//! AX-ID: AXIOMA-001 (G(1,3) substrate), AXIOMA-002 (time as geometric dimension)
//!
//! # Primary type
//! [`SparseCliffordVector`] — dense 160-byte multivector. See [`multivector`].
//!
//! # Module map
//! - [`sign`]       — `BladeIndex`, `Sign`, `CAYLEY_SIGN`, `compute_clifford_sign`
//! - [`basis`]      — `CliffordBasis`, `CANONICAL_G13`, `GRADE_TABLE`
//! - [`grade`]      — `CLIFFORD_NORM_WEIGHTS`, `compute_clifford_norm_sq`, projections
//! - [`multivector`]— `SparseCliffordVector` (primary type)
//! - [`product`]    — `sparse_geometric_product`
//!
//! # Design constraints (ENGINEERING_BLUEPRINT §1, §6)
//! - `HashMap` / `BTreeMap` PROHIBITED everywhere.
//! - Signature (1,3) preserved in all transformations (AXIOMA-001).
//! - No global simulation clock (AXIOMA-002).
//! - No external basis parameter in geometric product (Mandato §4.2).
//! - CS gate uses `max_abs_coeff`, NOT the Clifford/L2 norm (Mandato §2.3).
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
    clippy::unreadable_literal,
    clippy::uninlined_format_args,
)]

pub mod basis;
pub mod dual;
pub mod experimental;
pub mod grade;
pub mod multivector;
pub mod product;
pub mod sign;

pub use basis::{CliffordBasis, CANONICAL_G13};
pub use dual::{geometric_product_dual, Dual, SparseDualVector};
pub use grade::{
    compute_clifford_norm, compute_clifford_norm_sq, grade_project_ct, grades_present,
    is_homogeneous, max_grade, min_grade, reverse, CLIFFORD_NORM_WEIGHTS, REVERSE_SIGN,
};
pub use multivector::SparseCliffordVector;
pub use multivector::{fast_metric_distance, fast_metric_distance_from_dense};
pub use product::{
    bivector_norm_sq_of_product, bivector_norm_sq_of_product_lhs_dense, sparse_geometric_product,
    sparse_geometric_product_with_mode, BivectorProduct, GeometricProductMode,
};
pub use sign::{compute_clifford_sign, fast_cayley_product, BladeIndex, Sign, CAYLEY_SIGN};

// ── GeometricProduct trait ────────────────────────────────────────────────────

/// Core algebra trait for G(1,3) multivectors.
///
/// Implemented by [`SparseCliffordVector`]. The `basis` parameter has been
/// removed from `geo_product` (Mandato §4.2): `CAYLEY_SIGN` is compile-time
/// and self-contained.
///
/// AX-ID: AXIOMA-001
pub trait GeometricProduct: Sized {
    /// Full geometric product A * B. Returns `None` when the result is
    /// below the cognitive noise floor (Cauchy-Schwarz gate).
    fn geo_product(&self, rhs: &Self) -> Option<Self>;

    /// Producto escalar métrico Σᵢ aᵢbᵢηᵢᵢ.
    ///
    /// NO es la norma de Lorentz ⟨A·Ã⟩₀. Para grado k ≥ 2, el valor
    /// difiere de `clifford_norm_sq` por el signo del reverso.
    /// Use `clifford_norm_sq` para la norma Lorentz-invariante.
    fn metric_scalar_product(&self, rhs: &Self) -> f64;

    /// Grade-k projection of the multivector.
    #[must_use]
    fn grade_project(&self, grade: usize) -> Self;
}
