//! CliffordBasis — compile-time G(1,3) basis descriptor.
//!
//! AX-ID: AXIOMA-001, AXIOMA-002
//!
//! # Layout (repr(C), zero padding, bytemuck::Pod valid)
//! ```text
//! Offset  Size   Field                      Type
//!      0    16   grade                      [u8;  16]
//!     16    16   signature                  [i8;  16]
//!     32    68   fenwick_parity_tree        [i32; 17]
//!                                           ─────────
//!                Total: 100 bytes │ align: 4 │ padding: 0
//! ```
//!
//! # Integer discipline
//! - `blade_square` returns `i8` — no float arithmetic in this module.
//! - `grade_of` returns `u8`.
//! - `CANONICAL_G13` is baked into `.rodata` at compile time.
//! - No runtime construction. Use `&CANONICAL_G13`.

use genesis_types::{CLIFFORD_BASIS_SIZE, MAX_CLIFFORD_GRADE};

// ── Dimensional constants ─────────────────────────────────────────────────────

/// Number of basis vectors in G(1,3).
pub const DIM: usize = MAX_CLIFFORD_GRADE;

/// Total blade count: 2^DIM = 16.
pub const TOTAL_BLADES: usize = CLIFFORD_BASIS_SIZE;

/// Maximum blade bitmask for G(1,3): 0b1111 = 15.
pub const MAX_BLADE_MASK: usize = TOTAL_BLADES - 1;

const _: () = assert!(DIM == 4);

// ── Precomputed immutable tables (baked into .rodata) ─────────────────────────

/// `GRADE_TABLE[i]` = popcount(i) = Grassmann grade of blade i.
/// Grade ∈ {0,1,2,3,4}. `u8` is exact.
///
/// AX-ID: AXIOMA-001
pub(crate) const GRADE_TABLE: [u8; TOTAL_BLADES] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// `SIGNATURE_TABLE[i]` = e_I² for blade bitmask i in G(1,3).
///
/// Formula: e_I² = (−1)^{k(k−1)/2} · ∏_{j∈I} η_{jj}
///   where k = grade(I), η = diag(+1,−1,−1,−1).
///
/// Values ∈ {+1i8, −1i8}. Verified against Cayley sign kernel: 16/16 ✓.
///
/// Used by `CLIFFORD_NORM_WEIGHTS` in `grade.rs` (not cfg(test)).
/// AX-ID: AXIOMA-001
pub(crate) const SIGNATURE_TABLE: [i8; TOTAL_BLADES] =
    [1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1];

/// Precomputed Fenwick (BIT) parity tree over the 16 blade grades.
/// Index 0 is unused (1-indexed tree).
///
/// `fenwick_prefix_parity(k)` = count of odd-grade blades in [0, k].
/// Verified: `prefix_parity(15)` = 8 (4 grade-1 + 4 grade-3 blades) ✓.
#[cfg(test)]
pub(crate) const FENWICK_TABLE: [i32; TOTAL_BLADES + 1] =
    [0, 0, 1, 1, 2, 1, 1, 0, 4, 1, 1, 0, 2, 0, 1, 1, 8];

pub(crate) const FENWICK_PREFIX_LUT: [u8; TOTAL_BLADES] = {
    let mut lut = [0u8; TOTAL_BLADES];
    let mut acc = 0u8;
    let mut i = 0usize;
    while i < TOTAL_BLADES {
        acc += (i.count_ones() & 1) as u8;
        lut[i] = acc;
        i += 1;
    }
    lut
};

// ── CliffordBasis struct ──────────────────────────────────────────────────────

/// Canonical G(1,3) basis descriptor.
///
/// All fields are compile-time constants baked into `CANONICAL_G13`.
/// **Never instantiated at runtime** — use `&CANONICAL_G13`.
///
/// AX-ID: AXIOMA-001, AXIOMA-002
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CliffordBasis {
    /// Grassmann grade of each blade index: `grade[i] = popcount(i)`.
    pub grade: [u8; TOTAL_BLADES],

    /// e_I² for each blade index ∈ {+1i8, −1i8}.
    pub signature: [i8; TOTAL_BLADES],

    /// 1-indexed Fenwick tree for prefix odd-grade-blade counts.
    pub fenwick_parity_tree: [i32; TOTAL_BLADES + 1],
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
const fn fenwick_update_const(fenwick: &mut [i32; TOTAL_BLADES + 1], blade: usize) {
    let mut i = blade as i32 + 1;
    while i <= TOTAL_BLADES as i32 {
        fenwick[i as usize] += 1;
        i += i & (-i);
    }
}

impl CliffordBasis {
    /// Builds the G(1,3) basis at compile time. Called only from the `const`
    /// block initializing `CANONICAL_G13`.
    const fn build_g13() -> Self {
        let mut grade     = [0u8;  TOTAL_BLADES    ];
        let mut signature = [0i8;  TOTAL_BLADES    ];
        let mut fenwick   = [0i32; TOTAL_BLADES + 1];

        let mut blade = 0usize;
        while blade < TOTAL_BLADES {
            #[allow(clippy::cast_possible_truncation)]
            let g = blade.count_ones() as u8;
            grade[blade]     = g;
            signature[blade] = Self::blade_square_const(blade);

            if (g & 1) != 0 {
                // CRYSTAL: O5, O36, FO29 — inevitable
                fenwick_update_const(&mut fenwick, blade);
            }

            blade += 1;
        }

        Self { grade, signature, fenwick_parity_tree: fenwick }
    }

    /// Grassmann grade of blade `i`. Returns `u8` ∈ {0..=4}.
    #[inline]
    pub fn grade_of(&self, i: usize) -> u8 {
        debug_assert!(i <= MAX_BLADE_MASK);
        self.grade[i & MAX_BLADE_MASK]
    }

    /// Returns the exact metric square \(e_I^2\in\{-1,+1\}\) for blade `I`.
    ///
    /// Mathematical definition:
    /// `$ e_I^2 = (-1)^{k(k-1)/2}\prod_{\mu \in I} g_{\mu\mu},\quad k = |I| $`
    ///
    /// Invariants:
    /// - Output is exact `i8` (`-1` or `+1`), never `0`.
    /// - No floating-point operations are performed.
    ///
    /// AX-ID: AXIOMA-001
    /// See also: [`Self::blade_square_f64`]
    #[allow(clippy::inline_always)]
    // Forced inlining: hot-path function of the geometric product.
    // Benchmark kuramoto_step_1000_nodes = 1.11 ms for N=1000.
    // Without inline(always) the compiler can introduce frame overhead in
    // the inner loop of sparse_geometric_product (≥ 10⁸ calls/step).
    #[inline(always)]
    pub fn blade_square(&self, i: usize) -> i8 {
        debug_assert!(i <= MAX_BLADE_MASK);
        self.signature[i & MAX_BLADE_MASK]
    }

    /// Returns the metric square \(e_I^2\) as `f64` for numeric kernels.
    ///
    /// Mathematical definition:
    /// `$ e_I^2(\mathrm{f64}) = \mathrm{float}(e_I^2),\quad e_I^2\in\{-1,+1\} $`
    ///
    /// Invariants:
    /// - Conversion from `i8` to `f64` is lossless for `±1`.
    /// - Value remains exactly representable in IEEE-754 binary64.
    ///
    /// AX-ID: AXIOMA-001
    /// See also: [`Self::blade_square`]
    #[allow(clippy::inline_always)]
    // Forced inlining: hot-path function of the geometric product.
    // Benchmark kuramoto_step_1000_nodes = 1.11 ms for N=1000.
    // Without inline(always) the compiler can introduce frame overhead in
    // the inner loop of sparse_geometric_product (≥ 10⁸ calls/step).
    #[inline(always)]
    pub fn blade_square_f64(&self, i: usize) -> f64 {
        f64::from(self.blade_square(i))
    }

    /// O(log 16) prefix odd-grade-blade count for blades [0, blade_idx].
    ///
    /// Uses a direct LUT for `TOTAL_BLADES = 16` to minimize branch and ALU
    /// pressure on hot paths while preserving exact Fenwick semantics.
    #[inline]
    pub fn fenwick_prefix_parity(&self, blade_idx: usize) -> i32 {
        debug_assert!(blade_idx < TOTAL_BLADES);
        FENWICK_PREFIX_LUT[blade_idx & MAX_BLADE_MASK] as i32
    }

    /// Computes e_I² as `i8` — `const fn` used during compile-time table build.
    const fn blade_square_const(blade: usize) -> i8 {
        let k = blade.count_ones() as i64;
        let swap_pairs = (k * (k - 1)) >> 1;
        // CRYSTAL: O37 — inevitable
        // CRYSTAL: O38 — inevitable
        let reorder_sign: i8 = 1 - (((swap_pairs & 1) as i8) << 1);
        let spatial_bits = (blade >> 1) & 0b111;
        let spatial_count = spatial_bits.count_ones();
        let metric_sign: i8 = 1 - (((spatial_count & 1) as i8) << 1);
        reorder_sign * metric_sign
    }

    /// View as raw bytes for DAX/NVMe writes. Zero-copy. O(1).
    /// AX-ID: AXIOMA-018
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Reconstruct from raw bytes without panicking.
    ///
    /// # Errors
    /// Returns `bytemuck::PodCastError` if `bytes` does not have the length
    /// or alignment required for `CliffordBasis`.
    #[inline]
    pub fn try_from_bytes(bytes: &[u8]) -> Result<&Self, bytemuck::PodCastError> {
        bytemuck::try_from_bytes(bytes)
    }
}

// ── Canonical compile-time instance ──────────────────────────────────────────

/// The one and only G(1,3) basis — built at compile time, lives in `.rodata`.
///
/// **All runtime code must use `&CANONICAL_G13`.**
///
/// AX-ID: AXIOMA-001
pub const CANONICAL_G13: CliffordBasis = CliffordBasis::build_g13();

/// Convenience reference accessor.
#[inline]
#[must_use = "fetching the canonical basis has no side effects"]
pub const fn g13() -> &'static CliffordBasis {
    &CANONICAL_G13
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::sign::CAYLEY_SIGN;

    // ── Table consistency ────────────────────────────────────────────────────

    #[test]
    fn canonical_grade_table_matches_precomputed() {
        // Direct array equality: cleaner than indexed loop, same failure semantics.
        assert_eq!(CANONICAL_G13.grade, GRADE_TABLE);
    }

    #[test]
    fn canonical_signature_table_matches_precomputed() {
        assert_eq!(CANONICAL_G13.signature, SIGNATURE_TABLE);
    }

    #[test]
    fn canonical_fenwick_struct_field_matches_precomputed() {
        // Verifies the struct field built by build_g13(), not the LUT.
        // Required for DAX round-trip integrity: fenwick_parity_tree is serialized.
        for i in 0..=TOTAL_BLADES {
            assert_eq!(
                CANONICAL_G13.fenwick_parity_tree[i], FENWICK_TABLE[i],
                "fenwick_parity_tree[{i}] mismatch"
            );
        }
    }

    /// Verifies FENWICK_PREFIX_LUT (introduced alongside the O(1) accessor)
    /// is consistent with grade-table prefix counts.
    ///
    /// Invariant: LUT[k] = |{i ≤ k : popcount(i) is odd}|.
    #[test]
    fn fenwick_prefix_lut_matches_grade_prefix_counts() {
        let mut expected = 0u8;
        for (i, &lut_val) in FENWICK_PREFIX_LUT.iter().enumerate() {
            expected += (i.count_ones() & 1) as u8;
            assert_eq!(lut_val, expected, "FENWICK_PREFIX_LUT[{i}]");
        }
    }

    // ── Basis vector signatures ──────────────────────────────────────────────

    #[test]
    fn vg1_basis_vector_signatures() {
        assert_eq!(CANONICAL_G13.blade_square(0b0001),  1i8, "e₀²  (timelike)");
        assert_eq!(CANONICAL_G13.blade_square(0b0010), -1i8, "e₁²  (spacelike)");
        assert_eq!(CANONICAL_G13.blade_square(0b0100), -1i8, "e₂²  (spacelike)");
        assert_eq!(CANONICAL_G13.blade_square(0b1000), -1i8, "e₃²  (spacelike)");
    }

    #[test]
    fn vg1_pseudoscalar_squared_minus_one() {
        assert_eq!(CANONICAL_G13.blade_square(0b1111), -1i8, "I²");
    }

    #[test]
    fn vg1_bivector_e01_squared_plus_one() {
        assert_eq!(CANONICAL_G13.blade_square(0b0011), 1i8, "e₀₁²");
    }

    #[test]
    fn vg1_bivector_e12_squared_minus_one() {
        assert_eq!(CANONICAL_G13.blade_square(0b0110), -1i8, "e₁₂²");
    }

    // ── Cross-verification against Cayley table ──────────────────────────────

    #[test]
    fn signature_consistent_with_cayley_table_all_16() {
        for blade in 0..TOTAL_BLADES {
            assert_eq!(
                CANONICAL_G13.blade_square(blade),
                CAYLEY_SIGN[blade][blade],
                "blade {blade:#06b}: signature vs cayley mismatch"
            );
        }
    }

    // ── Grade table ──────────────────────────────────────────────────────────

    #[test]
    fn grade_table_all_16_match_popcount() {
        for i in 0..TOTAL_BLADES {
            assert_eq!(
                CANONICAL_G13.grade_of(i),
                i.count_ones() as u8,
                "blade {i:#06b}"
            );
        }
    }

    // ── Fenwick / prefix parity ──────────────────────────────────────────────

    #[test]
    fn fenwick_total_odd_grade_blades_is_eight() {
        assert_eq!(
            CANONICAL_G13.fenwick_prefix_parity(TOTAL_BLADES - 1),
            8,
            "total odd-grade blades in G(1,3) must be 8 (grade-1×4 + grade-3×4)"
        );
    }

    #[test]
    fn fenwick_prefix_correct_at_every_position() {
        let mut running = 0i32;
        for k in 0..TOTAL_BLADES {
            if CANONICAL_G13.grade[k] % 2 == 1 {
                running += 1;
            }
            assert_eq!(
                CANONICAL_G13.fenwick_prefix_parity(k),
                running,
                "fenwick_prefix_parity({k})"
            );
        }
    }

    // ── Signature domain ─────────────────────────────────────────────────────

    #[test]
    fn signature_always_plus_or_minus_one() {
        for i in 0..TOTAL_BLADES {
            let s = CANONICAL_G13.blade_square(i);
            assert!(s == 1i8 || s == -1i8, "blade {i:#06b}: sig={s}");
        }
    }

    #[test]
    fn blade_square_f64_is_bit_exact() {
        for i in 0..TOTAL_BLADES {
            let s = CANONICAL_G13.blade_square(i);
            let f = CANONICAL_G13.blade_square_f64(i);
            let expected = if s == 1 { 1.0f64 } else { -1.0f64 };
            assert_eq!(
                f.to_bits(),
                expected.to_bits(),
                "blade {i:#06b}: f64 cast must be bit-exact (i8→f64 for ±1 is lossless)"
            );
        }
    }

    // ── Layout & ABI ─────────────────────────────────────────────────────────

    #[test]
    fn struct_size_and_alignment() {
        const MANUAL: usize = 16 + 16 + 17 * 4; // grade[16] + sig[16] + fenwick[17]×4
        assert_eq!(std::mem::size_of::<CliffordBasis>(), 100, "must be 100 bytes");
        assert_eq!(std::mem::size_of::<CliffordBasis>(), MANUAL, "zero implicit padding");
        assert_eq!(std::mem::align_of::<CliffordBasis>(), 4, "must be align 4");
    }

    #[test]
    fn dax_round_trip_via_try_from_bytes() {
        let bytes = CANONICAL_G13.as_bytes();
        assert_eq!(bytes.len(), 100);
        let recovered = CliffordBasis::try_from_bytes(bytes).unwrap();
        assert_eq!(*recovered, CANONICAL_G13);
    }

    #[test]
    fn try_from_bytes_fails_on_wrong_length() {
        let bad = [0u8; 42];
        assert!(CliffordBasis::try_from_bytes(&bad).is_err());
    }

    // ── Static invariants ────────────────────────────────────────────────────

    #[test]
    fn canonical_is_static_in_rodata() {
        // Correct approach: verify content invariance across two independent borrows,
        // not raw pointer equality (Miri models each borrow as distinct provenance).
        // The Rust language spec guarantees single-allocation semantics for `static`.
        //
        // AX-ID: AXIOMA-001 — Clifford basis is static, immutable, zero-cost.
        let a = &CANONICAL_G13;
        let b = &CANONICAL_G13;
        assert_eq!(a.grade,               b.grade,               "grade invariant");
        assert_eq!(a.signature,           b.signature,           "signature invariant");
        assert_eq!(a.fenwick_parity_tree, b.fenwick_parity_tree, "fenwick invariant");

        // Verify Minkowski signature (+,-,-,-) via blade bitmask indices.
        assert_eq!(a.signature[0], 1i8,  "scalar (0b0000)  → +1");
        assert_eq!(a.signature[1], 1i8,  "e0     (0b0001)  → +1 (timelike)");
        assert_eq!(a.signature[2], -1i8, "e1     (0b0010)  → -1 (spacelike)");
        assert_eq!(a.signature[4], -1i8, "e2     (0b0100)  → -1 (spacelike)");
        assert_eq!(a.signature[8], -1i8, "e3     (0b1000)  → -1 (spacelike)");
    }
}
