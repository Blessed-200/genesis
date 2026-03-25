//! Clifford product sign kernel for G(1,3): compile-time Cayley table,
//! `BladeIndex` newtype, and `Sign` enum.
//!
//! AX-ID: AXIOMA-001
//!
//! # Compile-time Cayley table
//! G(1,3) has exactly 16 basis blades → 16×16 = 256 blade pairs.
//! `CAYLEY_SIGN: [[i8; 16]; 16]` is computed at compile time and baked
//! into `.rodata`. At runtime, sign lookup is two array indices + one XOR.
//!
//! # Integer domain
//! All sign values are `i8` (+1 or −1). Float conversion occurs only at
//! the accumulation site in `product.rs`.
//!
//! # Blade bitmask encoding
//! ```text
//!   bit 0 → e₀  temporal  η₀₀ = +1
//!   bit 1 → e₁  spatial   η₁₁ = −1
//!   bit 2 → e₂  spatial   η₂₂ = −1
//!   bit 3 → e₃  spatial   η₃₃ = −1
//! ```

use genesis_types::error::GenesisError;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum valid blade bitmask for G(1,3): 0b1111 = 15.
pub const MAX_BLADE_MASK: usize = 0b1111;

/// Total number of basis blades: 2^4 = 16.
pub const BLADE_COUNT: usize = MAX_BLADE_MASK + 1;

// ── BladeIndex newtype ────────────────────────────────────────────────────────

/// Typed blade index for G(1,3): guaranteed 0..=15 by construction.
///
/// Eliminates out-of-range indexing without `debug_assert` in hot paths.
/// Stored as `u8` to minimise register pressure.
///
/// AX-ID: AXIOMA-001
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BladeIndex(u8);

impl BladeIndex {
    /// Minimum valid blade index (scalar blade).
    pub const MIN: Self = Self(0);
    /// Maximum valid blade index (pseudoscalar blade e₀₁₂₃).
    pub const MAX: Self = Self(15);

    /// Validated constructor. Returns `Err` for `idx > 15`.
    ///
    /// Validates unconditionally in both debug and release modes.
    ///
    /// # Errors
    /// Returns `GenesisError::BladeIndexOutOfRange` if `idx >= 16`.
    #[inline]
    pub const fn new(idx: u8) -> Result<Self, GenesisError> {
        if idx > 15 {
            Err(GenesisError::BladeIndexOutOfRange {
                index: idx as usize,
            })
        } else {
            Ok(Self(idx))
        }
    }

    /// Unchecked constructor for sites where the bound is guaranteed by
    /// construction (e.g., iterating `0u8..16`).
    ///
    /// # Safety
    /// Caller must ensure `idx ≤ 15`. Violating this causes UB only if
    /// the result is used to index `CAYLEY_SIGN` — which is `[_; 16]`,
    /// so the actual UB is an out-of-bounds slice read.
    #[allow(dead_code)]
    #[inline]
    // SAFETY: caller guarantees `idx < 16` (blade count in G(1,3) = 2^4 = 16).
    pub(crate) const unsafe fn new_unchecked(idx: u8) -> Self {
        debug_assert!(idx <= 15, "BladeIndex::new_unchecked requires idx <= 15");
        Self(idx)
    }

    /// Returns the inner `u8` value.
    #[inline]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Returns the index as `usize` for array indexing.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

// ── Sign enum ─────────────────────────────────────────────────────────────────

/// Algebraic sign ∈ {+1, −1}.
///
/// Eliminates the invalid value space of `i8` and prevents ±0.0 contamination
/// that would occur with `f64` sign values.
///
/// AX-ID: AXIOMA-001
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Represents the algebraic sign of a Cayley product entry.
///
/// Stored as `i8` for SIMD-friendly packing. Derived from parity
/// of blade permutations in G(1,3) via `CAYLEY_SIGN` table.
#[repr(i8)]
pub enum Sign {
    /// Positive sign: even number of basis vector permutations.
    Pos = 1,
    /// Negative sign: odd number of basis vector permutations.
    Neg = -1,
}

impl Sign {
    /// Convert from `i8`. Panics in debug if `v ∉ {1, −1}`.
    #[inline]
    pub fn from_i8(v: i8) -> Self {
        debug_assert!(v == 1 || v == -1, "Sign only admits ±1, got {v}");
        if v > 0 {
            Self::Pos
        } else {
            Self::Neg
        }
    }

    /// Returns the sign as `i8` ∈ {1, −1}.
    #[inline]
    pub const fn as_i8(self) -> i8 {
        self as i8
    }

    /// Returns the sign as `f64` ∈ {1.0, −1.0}. Cast is lossless and exact.
    #[inline]
    pub fn as_f64(self) -> f64 {
        f64::from(self as i8)
    }
}

impl std::ops::Mul for Sign {
    type Output = Self;
    /// Product of two signs: Pos×Pos = Pos, Pos×Neg = Neg×Pos = Neg, Neg×Neg = Pos.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if self == rhs {
            Self::Pos
        } else {
            Self::Neg
        }
    }
}

// ── Compile-time sign computation ─────────────────────────────────────────────

/// Computes the Clifford product sign for blade pair `(ia, ib)` in G(1,3).
///
/// This is the canonical `const fn` implementation used to generate the
/// compile-time `CAYLEY_SIGN` table. Also serves as the runtime oracle in tests.
///
/// # Algorithm: S = (−1)^{P + Nₛ}
///
/// **P** — transposition parity: anticommutations needed to bring the merged
/// blade sequence into canonical ascending order. Computed via POPCNT on
/// shifted bitmasks.
///
/// **Nₛ** — spatial contraction count: bits 1,2,3 present in both `ia` and `ib`.
/// Each spatial self-contraction eₖ² = ηₖₖ = −1 flips the sign.
///
/// AX-ID: AXIOMA-001
#[allow(clippy::inline_always)]
// Inlining forzado: función en hot-path del producto geométrico.
// Benchmark kuramoto_step_1000_nodes = 1.11 ms para N=1000.
// Sin inline(always) el compilador puede crear frame overhead en
// el inner loop de sparse_geometric_product (≥ 10⁸ llamadas/step).
#[inline(always)]
pub const fn compute_clifford_sign(indices_a: usize, indices_b: usize) -> i8 {
    // ── Step 1: Transposition parity P ───────────────────────────────────────
    let mut transpositions: u32 = 0;
    let mut b = indices_b;
    while b != 0 {
        let lowest = b & b.wrapping_neg();
        let pos = lowest.trailing_zeros();
        let shift = pos + 1;
        if shift < usize::BITS {
            transpositions += (indices_a >> shift).count_ones();
        }
        b ^= lowest;
    }
    let parity_sign: i8 = 1 - 2 * ((transpositions & 1) as i8);

    // ── Step 2: Minkowski metric correction Nₛ ───────────────────────────────
    // Temporal bit (0): η₀₀ = +1 → excluded (no sign change).
    // Spatial bits (1,2,3): ηₖₖ = −1 → each flips sign.
    let contractions = (indices_a & indices_b) & 0b1110_usize;
    let spatial_count = contractions.count_ones();
    let metric_sign: i8 = 1 - 2 * ((spatial_count & 1) as i8);

    parity_sign * metric_sign
}

// ── Compile-time Cayley table ─────────────────────────────────────────────────

/// Precomputed sign table for all 16×16 basis blade pairs in G(1,3).
///
/// `CAYLEY_SIGN[i][j]` = sign of (blade_i · blade_j) ∈ {+1i8, −1i8}.
///
/// **Memory:** 256 bytes in `.rodata`. Guaranteed L1-cache resident.
/// **Access cost:** one 2D array index + XOR for the result blade.
///
/// AX-ID: AXIOMA-001
pub const CAYLEY_SIGN: [[i8; BLADE_COUNT]; BLADE_COUNT] = {
    let mut table = [[0i8; BLADE_COUNT]; BLADE_COUNT];
    let mut i = 0usize;
    while i < BLADE_COUNT {
        let mut j = 0usize;
        while j < BLADE_COUNT {
            table[i][j] = compute_clifford_sign(i, j);
            j += 1;
        }
        i += 1;
    }
    table
};

// ── Hot-path entry point ──────────────────────────────────────────────────────

/// Returns the basis-blade geometric product kernel `(I ⊕ J, σ(I,J))` in O(1).
///
/// Mathematical definition:
/// `$ e_I e_J = \sigma(I,J)\,e_{I \oplus J},\quad \sigma(I,J)\in\{-1,+1\} $`
/// with `\(\sigma(I,J)=\texttt{CAYLEY\_SIGN}[I][J]\)`.
///
/// AX-ID: AXIOMA-001
/// See also: [`CAYLEY_SIGN`]
#[allow(clippy::inline_always)]
// Inlining forzado: función en hot-path del producto geométrico.
// Benchmark kuramoto_step_1000_nodes = 1.11 ms para N=1000.
// Sin inline(always) el compilador puede crear frame overhead en
// el inner loop de sparse_geometric_product (≥ 10⁸ llamadas/step).
#[inline(always)]
pub fn fast_cayley_product(indices_a: usize, indices_b: usize) -> (usize, i8) {
    debug_assert!(
        indices_a <= MAX_BLADE_MASK,
        "blade {indices_a:#06b} > G(1,3) max {MAX_BLADE_MASK:#06b}"
    );
    debug_assert!(
        indices_b <= MAX_BLADE_MASK,
        "blade {indices_b:#06b} > G(1,3) max {MAX_BLADE_MASK:#06b}"
    );
    (indices_a ^ indices_b, CAYLEY_SIGN[indices_a][indices_b])
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[allow(clippy::float_cmp, clippy::needless_range_loop)]
mod tests {
    use super::*;

    // ── BladeIndex ────────────────────────────────────────────────────────────

    #[test]
    fn blade_index_accepts_zero_to_fifteen() {
        assert!(BladeIndex::new(0).is_ok());
        assert!(BladeIndex::new(15).is_ok());
    }

    #[test]
    fn blade_index_rejects_index_16_and_above() {
        assert!(BladeIndex::new(16).is_err(), "16 must be rejected");
        assert!(BladeIndex::new(255).is_err(), "255 must be rejected");
    }

    #[test]
    fn blade_index_get_round_trip() {
        for v in 0u8..=15 {
            let b = BladeIndex::new(v).unwrap();
            assert_eq!(b.get(), v);
            assert_eq!(b.as_usize(), v as usize);
        }
    }

    // ── Sign ─────────────────────────────────────────────────────────────────

    #[test]
    fn sign_from_i8_pos() {
        let s = Sign::from_i8(1);
        assert_eq!(s, Sign::Pos);
        assert_eq!(s.as_i8(), 1i8);
        assert_eq!(s.as_f64(), 1.0f64);
    }

    #[test]
    fn sign_from_i8_neg() {
        let s = Sign::from_i8(-1);
        assert_eq!(s, Sign::Neg);
        assert_eq!(s.as_i8(), -1i8);
        assert_eq!(s.as_f64(), -1.0f64);
    }

    #[test]
    fn sign_multiplication_table() {
        assert_eq!(Sign::Pos * Sign::Pos, Sign::Pos);
        assert_eq!(Sign::Pos * Sign::Neg, Sign::Neg);
        assert_eq!(Sign::Neg * Sign::Pos, Sign::Neg);
        assert_eq!(Sign::Neg * Sign::Neg, Sign::Pos);
    }

    #[test]
    fn sign_as_f64_is_bit_exact_no_rounding() {
        // i8 → f64 for ±1 is lossless.
        assert_eq!(Sign::Pos.as_f64().to_bits(), 1.0f64.to_bits());
        assert_eq!(Sign::Neg.as_f64().to_bits(), (-1.0f64).to_bits());
    }

    // ── Verification Gate 1: Minkowski signature ──────────────────────────────

    #[test]
    fn vg1_basis_vector_squares_exact() {
        let cases: [(usize, i8); 4] = [
            (0b0001, 1),  // e₀² = +1
            (0b0010, -1), // e₁² = −1
            (0b0100, -1), // e₂² = −1
            (0b1000, -1), // e₃² = −1
        ];
        for (blade, expected) in cases {
            let (idx, sign) = fast_cayley_product(blade, blade);
            assert_eq!(idx, 0, "eₖ² must be scalar (blade 0)");
            assert_eq!(sign, expected, "sign for blade {blade:#06b}");
        }
    }

    #[test]
    fn vg1_anticommutativity_all_six_pairs() {
        let bases = [0b0001usize, 0b0010, 0b0100, 0b1000];
        for i in 0..4 {
            for j in (i + 1)..4 {
                let (k_ij, s_ij) = fast_cayley_product(bases[i], bases[j]);
                let (k_ji, s_ji) = fast_cayley_product(bases[j], bases[i]);
                assert_eq!(k_ij, k_ji, "blade mismatch pair ({i},{j})");
                assert_eq!(s_ij + s_ji, 0i8, "signs must cancel for pair ({i},{j})");
            }
        }
    }

    #[test]
    fn sign_is_always_plus_or_minus_one_exhaustive() {
        for a in 0..BLADE_COUNT {
            for b in 0..BLADE_COUNT {
                let s = CAYLEY_SIGN[a][b];
                assert!(
                    s == 1i8 || s == -1i8,
                    "CAYLEY_SIGN[{a}][{b}] = {s}, must be ±1"
                );
            }
        }
    }

    #[test]
    fn cayley_table_matches_runtime_sign_exhaustive() {
        for a in 0..BLADE_COUNT {
            for b in 0..BLADE_COUNT {
                assert_eq!(
                    CAYLEY_SIGN[a][b],
                    compute_clifford_sign(a, b),
                    "table vs runtime mismatch at [{a}][{b}]"
                );
            }
        }
    }

    #[test]
    fn pseudoscalar_squared_is_minus_one() {
        let (idx, sign) = fast_cayley_product(0b1111, 0b1111);
        assert_eq!(idx, 0, "I² must be scalar");
        assert_eq!(sign, -1i8, "I² = −1 in G(1,3)");
    }

    #[test]
    fn bivector_e01_squared_is_plus_one() {
        let (idx, sign) = fast_cayley_product(0b0011, 0b0011);
        assert_eq!(idx, 0);
        assert_eq!(sign, 1i8);
    }

    #[test]
    fn bivector_e12_squared_is_minus_one() {
        let (idx, sign) = fast_cayley_product(0b0110, 0b0110);
        assert_eq!(idx, 0);
        assert_eq!(sign, -1i8);
    }

    #[test]
    fn result_blade_is_always_xor_exhaustive() {
        for a in 0..BLADE_COUNT {
            for b in 0..BLADE_COUNT {
                let (k, _) = fast_cayley_product(a, b);
                assert_eq!(k, a ^ b);
            }
        }
    }

    #[test]
    fn scalar_blade_is_left_identity() {
        for b in 0..BLADE_COUNT {
            let (k, sign) = fast_cayley_product(0, b);
            assert_eq!(k, b, "1 * blade {b:#06b} result blade");
            assert_eq!(sign, 1i8, "1 * blade {b:#06b} sign");
        }
    }
}
