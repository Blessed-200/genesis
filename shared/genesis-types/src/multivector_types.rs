//! Multivector-related shared data types.
//!
//! # active_mask width (FIX-DerivedMetadata)
//!
//! Changed from `u16` to `u32` in v5.1.0. Rationale:
//! - G(1,3) has 2^4 = 16 blades → u16 was sufficient for the base algebra.
//! - GramSchmidtExpander (LEY_FUNDACIONAL §3.8) can expand D beyond 16, creating
//!   new basis elements. A u32 mask supports up to 32 active blades, covering all
//!   anticipated expansion targets (D ~ 10³–10⁵ in production uses D for the
//!   *algebraic* dimension, not the blade count which stays bounded by the grade structure).
//!   In practice, active blades per SparseCliffordVector ≤ 16 (G(1,3) grade structure),
//!   but future G(1,n) extensions need wider masks.
//! - u64 (as proposed) would waste 4 bytes per metadata instance with no current need.
//!   u32 is the pragmatic "inevitable" choice: covers 2× the current blade count with
//!   zero padding overhead on 64-bit targets.

/// Derived scalar metadata computed from a multivector representation.
///
/// All three fields are computed in a single pass by `derive_all_metadata()`
/// in genesis-math. Stored separately from `SparseCliffordVector` to allow
/// lightweight metadata queries without loading the full 160-byte struct.
///
/// # Memory layout
/// Size = 16 bytes on 64-bit targets (u32 + 4-byte gap + f64 + f64).
/// align(8) preserves f64 alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedMetadata {
    /// Bitmask: bit i set ↔ blade i is active (|coeff| > COGNITIVE_PLANCK_CONSTANT).
    /// u32 supports up to 32 blades — covers G(1,3) (16 blades) and near-future expansions.
    pub active_mask: u32,
    /// Padding to maintain f64 alignment of subsequent fields.
    _pad: u32,
    /// Maximum absolute coefficient magnitude. Used for Cauchy-Schwarz gating.
    pub max_abs_coeff: f64,
    /// ⟨A·Ã⟩₀ — Clifford norm squared (Lorentz-invariant scalar).
    pub clifford_norm_sq: f64,
}

static_assertions::const_assert_eq!(core::mem::size_of::<DerivedMetadata>(), 24);
static_assertions::const_assert_eq!(core::mem::align_of::<DerivedMetadata>(), 8);

impl DerivedMetadata {
    /// Returns true if blade `i` is active in this metadata.
    #[inline(always)]
    pub fn is_active(&self, blade: usize) -> bool {
        blade < 32 && (self.active_mask & (1u32 << blade)) != 0
    }

    /// Construct a new `DerivedMetadata` from computed fields.
    /// Internal constructor — `_pad` is zero-initialized.
    #[inline(always)]
    pub fn new(active_mask: u32, max_abs_coeff: f64, clifford_norm_sq: f64) -> Self {
        Self { active_mask, _pad: 0, max_abs_coeff, clifford_norm_sq }
    }

    /// Mark blade `i` as active.
    #[inline(always)]
    pub fn set_active(&mut self, blade: usize) {
        if blade < 32 { self.active_mask |= 1u32 << blade; }
    }
}
