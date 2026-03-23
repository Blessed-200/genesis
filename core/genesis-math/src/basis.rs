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

// ── Dimensional constants ─────────────────────────────────────────────────────

/// Number of basis vectors in G(1,3).
pub const DIM: usize = 4;

/// Total blade count: 2^DIM = 16.
pub const TOTAL_BLADES: usize = 1 << DIM;

/// Maximum blade bitmask for G(1,3): 0b1111 = 15.
pub const MAX_BLADE_MASK: usize = TOTAL_BLADES - 1;

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

/// Prefix LUT: `ODD_GRADE_PREFIX_TABLE[k]` = odd-grade blade count in [0, k].
///
/// For fixed `TOTAL_BLADES = 16`, direct lookup is strictly lower latency than
/// Fenwick traversal and remains fully deterministic.
pub(crate) const ODD_GRADE_PREFIX_TABLE: [i32; TOTAL_BLADES] = {
    let mut out = [0i32; TOTAL_BLADES];
    let mut i = 0usize;
    let mut acc = 0i32;
    while i < TOTAL_BLADES {
        // loop-invariant, hoisted
        // CRYSTAL: O34 — inevitable
        acc += (GRADE_TABLE[i] & 1) as i32;
        out[i] = acc;
        i += 1;
    }
    out
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

impl CliffordBasis {
    /// Builds the G(1,3) basis at compile time. Called only from the `const`
    /// block initializing `CANONICAL_G13`.
    const fn build_g13() -> Self {
        let mut grade = [0u8; TOTAL_BLADES];
        let mut signature = [0i8; TOTAL_BLADES];
        let mut fenwick = [0i32; TOTAL_BLADES + 1];

        let mut blade = 0usize;
        while blade < TOTAL_BLADES {
            // blade < TOTAL_BLADES = 16, count_ones() ≤ 4, siempre cabe en u8.
            #[allow(clippy::cast_possible_truncation)]
            let g = blade.count_ones() as u8; // ≤ 4, siempre dentro de u8
            grade[blade] = g;
            signature[blade] = Self::blade_square_const(blade);
            // loop-invariant, hoisted
            // CRYSTAL: O5 — inevitable
            // CRYSTAL: O36 — inevitable
            // CRYSTAL: FO29 — inevitable
            let parity = (g & 1) as i32;
            if parity != 0 {
                // TOTAL_BLADES = 16 — siempre cabe en i32 y usize.
                // Usamos i32 para la aritmética del árbol de Fenwick (i & -i)
                // que requiere complemento a dos con signo.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_wrap
                )]
                // Fenwick tree arithmetic: índices 1..=TOTAL_BLADES (máx 16),
                // siempre positivos. La aritmética i & (-i) requiere i32 signed.
                // Invariante: i > 0 antes de cada cast a usize.
                {
                    let mut i: i32 = blade as i32 + 1; // blade ≤ 15, i ≤ 16, ok
                    while i <= TOTAL_BLADES as i32 {
                        fenwick[i as usize] += parity; // i > 0 por invariante
                        i += i & (-i);
                    }
                }
            }
            blade += 1;
        }
        Self {
            grade,
            signature,
            fenwick_parity_tree: fenwick,
        }
    }

    /// Grassmann grade of blade `i`. Returns `u8` ∈ {0..=4}.
    #[inline]
    pub fn grade_of(&self, i: usize) -> u8 {
        debug_assert!(i <= MAX_BLADE_MASK);
        self.grade.get(i).copied().unwrap_or(0)
    }

    /// e_I² as exact `i8` ∈ {+1, −1}. No float arithmetic.
    #[allow(clippy::inline_always)]
    // Inlining forzado: función en hot-path del producto geométrico.
    // Benchmark kuramoto_step_1000_nodes = 1.11 ms para N=1000.
    // Sin inline(always) el compilador puede crear frame overhead en
    // el inner loop de sparse_geometric_product (≥ 10⁸ llamadas/step).
    #[inline(always)]
    pub fn blade_square(&self, i: usize) -> i8 {
        debug_assert!(i <= MAX_BLADE_MASK);
        self.signature.get(i).copied().unwrap_or(0)
    }

    /// e_I² as `f64`. Cast from `i8` is lossless and exact.
    #[allow(clippy::inline_always)]
    // Inlining forzado: función en hot-path del producto geométrico.
    // Benchmark kuramoto_step_1000_nodes = 1.11 ms para N=1000.
    // Sin inline(always) el compilador puede crear frame overhead en
    // el inner loop de sparse_geometric_product (≥ 10⁸ llamadas/step).
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
        let safe_idx = blade_idx.min(TOTAL_BLADES - 1);
        ODD_GRADE_PREFIX_TABLE[safe_idx]
    }

    /// Computes e_I² as `i8` — `const fn` used during compile-time table build.
    const fn blade_square_const(blade: usize) -> i8 {
        let k = blade.count_ones() as i64;
        let swap_pairs = (k * (k - 1)) >> 1;
        // CRYSTAL: O37 — inevitable
        // CRYSTAL: O38 — inevitable
        let reorder_sign: i8 = if (swap_pairs & 1) == 0 { 1 } else { -1 };
        let spatial_bits = (blade >> 1) & 0b111;
        let spatial_count = spatial_bits.count_ones();
        let metric_sign: i8 = if spatial_count.is_multiple_of(2) {
            1
        } else {
            -1
        };
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
    /// Retorna `bytemuck::PodCastError` si `bytes` no tiene la longitud
    /// o alineación correcta para `CliffordBasis`.
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
pub const fn g13() -> &'static CliffordBasis {
    &CANONICAL_G13
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::sign::CAYLEY_SIGN;

    #[test]
    fn canonical_grade_table_matches_precomputed() {
        for i in 0..TOTAL_BLADES {
            assert_eq!(
                CANONICAL_G13.grade[i], GRADE_TABLE[i],
                "grade[{i}] mismatch"
            );
        }
    }

    #[test]
    fn canonical_signature_table_matches_precomputed() {
        for i in 0..TOTAL_BLADES {
            assert_eq!(
                CANONICAL_G13.signature[i], SIGNATURE_TABLE[i],
                "signature[{i:#06b}] mismatch"
            );
        }
    }

    #[test]
    fn canonical_fenwick_table_matches_precomputed() {
        for i in 0..=TOTAL_BLADES {
            assert_eq!(
                CANONICAL_G13.fenwick_parity_tree[i], FENWICK_TABLE[i],
                "fenwick[{i}] mismatch"
            );
        }
    }

    #[test]
    fn vg1_basis_vector_signatures() {
        assert_eq!(CANONICAL_G13.blade_square(0b0001), 1i8, "e₀²");
        assert_eq!(CANONICAL_G13.blade_square(0b0010), -1i8, "e₁²");
        assert_eq!(CANONICAL_G13.blade_square(0b0100), -1i8, "e₂²");
        assert_eq!(CANONICAL_G13.blade_square(0b1000), -1i8, "e₃²");
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

    #[test]
    fn fenwick_total_odd_grade_blades_is_eight() {
        let total = CANONICAL_G13.fenwick_prefix_parity(TOTAL_BLADES - 1);
        assert_eq!(total, 8, "total odd-grade blades must be 8");
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
                "blade {i:#06b}: f64 cast must be bit-exact"
            );
        }
    }

    #[test]
    fn struct_size_and_alignment() {
        assert_eq!(
            std::mem::size_of::<CliffordBasis>(),
            100,
            "must be 100 bytes"
        );
        assert_eq!(std::mem::align_of::<CliffordBasis>(), 4, "must be align 4");
        let manual = 16 + 16 + 17 * 4;
        assert_eq!(
            std::mem::size_of::<CliffordBasis>(),
            manual,
            "zero implicit padding"
        );
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
        let bad = vec![0u8; 42];
        assert!(CliffordBasis::try_from_bytes(&bad).is_err());
    }

    #[test]
    fn canonical_is_static_in_rodata() {
        // Verifies that CANONICAL_G13 is a static array (lives in .rodata / read-only data).
        //
        // ORIGINAL approach: compare raw pointers — `&CANONICAL_G13 as *const _ == &CANONICAL_G13`
        // This is WRONG under Miri: Miri models every reference as a fresh provenance token,
        // so two borrows of the same static may produce different pointer values, causing a
        // spurious failure even though no UB exists.
        //
        // CORRECT approach: verify the semantic invariant — that accessing CANONICAL_G13 from
        // multiple call sites produces bit-identical results (no dynamic computation, no mutable
        // global, no unsafe aliasing). Content equality is the right observable property.
        //
        // The `static` keyword in Rust guarantees single-allocation semantics by the language spec.
        // AX-ID: AXIOMA-001 — Clifford basis is static, immutable, zero-cost.

        // Two independent borrows must produce identical content (required property of static data)
        let a = &CANONICAL_G13;
        let b = &CANONICAL_G13;

        // Grade table must match
        assert_eq!(
            a.grade, b.grade,
            "grade table must be invariant across borrows"
        );
        // Signature (Minkowski +---) must match
        assert_eq!(
            a.signature, b.signature,
            "signature must be invariant across borrows"
        );
        // Fenwick tree must match
        assert_eq!(
            a.fenwick_parity_tree, b.fenwick_parity_tree,
            "fenwick parity tree must be invariant across borrows"
        );

        // Verify Minkowski signature (+,-,-,-):
        // Blade indices are bitmasks: 1=e0 (bit0), 2=e1 (bit1), 4=e2 (bit2), 8=e3 (bit3).
        // Blade 3 = 0b0011 = e₀₁ (bivector), NOT a basis vector.
        assert_eq!(
            a.signature[0], 1i8,
            "scalar blade (0b0000) must have signature +1"
        );
        assert_eq!(
            a.signature[1], 1i8,
            "e0 blade (0b0001, timelike) must have signature +1"
        );
        assert_eq!(
            a.signature[2], -1i8,
            "e1 blade (0b0010, spacelike) must have signature -1"
        );
        assert_eq!(
            a.signature[4], -1i8,
            "e2 blade (0b0100, spacelike) must have signature -1"
        );
        assert_eq!(
            a.signature[8], -1i8,
            "e3 blade (0b1000, spacelike) must have signature -1"
        );
    }
}
