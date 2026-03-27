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
/// Size = 64 bytes on 64-bit targets after cache-line alignment.
/// `align(64)` preserves a single alignment contract for SIMD and prefetch paths.
// CRYSTAL: FO1 — inevitable
// CRYSTAL: FO2 — inevitable
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivedMetadata {
    /// Maximum absolute coefficient magnitude. Used for Cauchy-Schwarz gating.
    pub max_abs_coeff: f64,
    /// ⟨A·Ã⟩₀ — Clifford norm squared (Lorentz-invariant scalar).
    pub clifford_norm_sq: f64,
    /// Bitmask: bit i set ↔ blade i is active (|coeff| > COGNITIVE_PLANCK_CONSTANT).
    /// u32 supports up to 32 blades — covers G(1,3) (16 blades) and near-future expansions.
    pub active_mask: u32,
    /// Explicit trailing metadata slot to keep constructor output deterministic.
    _pad: u32,
}

static_assertions::const_assert_eq!(core::mem::size_of::<DerivedMetadata>(), 64);
static_assertions::const_assert_eq!(core::mem::align_of::<DerivedMetadata>(), 64);

impl DerivedMetadata {
    /// Returns true if blade `i` is active in this metadata.
    #[inline] // bit-mask check: always inlined by optimizer anyway
    pub const fn is_active(&self, blade: usize) -> bool {
        blade < 32 && ((self.active_mask >> blade) & 1) == 1
    }

    /// Construct a new `DerivedMetadata` from computed fields.
    /// Internal constructor — `_pad` is zero-initialized.
    #[inline]
    pub const fn new(active_mask: u32, max_abs_coeff: f64, clifford_norm_sq: f64) -> Self {
        Self {
            max_abs_coeff,
            clifford_norm_sq,
            active_mask,
            _pad: 0,
        }
    }

    /// Mark blade `i` as active.
    #[inline]
    #[allow(clippy::missing_const_for_fn)] // &mut self not const-stable on MSRV 1.75
    pub fn set_active(&mut self, blade: usize) {
        if blade < 32 {
            self.active_mask |= 1u32 << blade;
        }
    }

    /// Mark blade `i` as active without bounds checks.
    ///
    /// # Safety
    /// Caller must guarantee `blade < 32`.
    ///
    /// AX-ID: AXIOMA-001
    #[inline]
    pub unsafe fn set_active_unchecked(&mut self, blade: usize) {
        debug_assert!(blade < 32);
        self.active_mask |= 1u32 << blade;
    }
}

#[cfg(test)]
mod tests {
    use super::DerivedMetadata;

    #[test]
    fn set_active_unchecked_sets_bit_when_precondition_holds() {
        let mut md = DerivedMetadata::new(0, 0.0, 0.0);
        // SAFETY: blade 7 is within [0, 31].
        unsafe { md.set_active_unchecked(7) };
        assert!(md.is_active(7));
    }

    #[test]
    fn set_active_ignores_out_of_range_blade() {
        let mut md = DerivedMetadata::new(0, 0.0, 0.0);
        md.set_active(40);
        assert_eq!(md.active_mask, 0);
    }

    #[test]
    fn is_active_returns_false_for_out_of_range_blade() {
        let md = DerivedMetadata::new(1, 0.0, 0.0);
        assert!(!md.is_active(40));
    }
}
