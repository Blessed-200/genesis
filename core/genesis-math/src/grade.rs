//! Grade-projection utilities, Clifford reverse, and norm computation for G(1,3).
//!
//! AX-ID: AXIOMA-001
//!
//! # CLIFFORD_NORM_WEIGHTS
//! Compile-time `[i8; 16]` table for the Lorentz-invariant norm:
//!   ⟨A·Ã⟩₀ = Σᵢ `coeffs[i]²` × `CLIFFORD_NORM_WEIGHTS[i]`
//!
//! Values: `REVERSE_SIGN[grade(i)] × SIGNATURE_TABLE[i]`
//! Verified: [+1,+1,−1,−1,−1,−1,+1,+1,−1,−1,+1,+1,+1,+1,−1,−1]
//!
//! # All projections
//! Operate over `active_mask` with `trailing_zeros()` — only active blades
//! are touched. Output is assembled via `from_dense_buf` in `multivector.rs`.

use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;

use crate::basis::{GRADE_TABLE, SIGNATURE_TABLE, TOTAL_BLADES};
use crate::multivector::SparseCliffordVector;

/// Reverse sign for Clifford blades, indexed by Grassmann grade.
///
/// Formula: (−1)^{k(k−1)/2} for grade k.
/// ```text
///   grade 0: +1    grade 1: +1    grade 2: −1    grade 3: −1    grade 4: +1
/// ```
/// AX-ID: AXIOMA-001
pub const REVERSE_SIGN: [i8; 5] = [1, 1, -1, -1, 1];

/// Weights for the Lorentz-invariant Clifford norm ⟨A·Ã⟩₀.
///
/// ``CLIFFORD_NORM_WEIGHTS[i]`` = REVERSE_SIGN[grade(i)] × `SIGNATURE_TABLE[i]`
///
/// Verified values:
/// `[+1,+1,−1,−1,−1,−1,+1,+1,−1,−1,+1,+1,+1,+1,−1,−1]`
///
/// Stored as `i8` for compatibility with SIMD gather instructions.
/// Runtime conversion to `f64` occurs only at the `compute_clifford_norm_sq`
/// accumulation site.
///
/// AX-ID: AXIOMA-001
pub const CLIFFORD_NORM_WEIGHTS: [i8; TOTAL_BLADES] = {
    let mut w = [0i8; TOTAL_BLADES];
    let mut i = 0usize;
    while i < TOTAL_BLADES {
        let grade = GRADE_TABLE[i] as usize;
        let rev = REVERSE_SIGN[grade];
        let sig = SIGNATURE_TABLE[i];
        w[i] = rev * sig;
        i += 1;
    }
    w
};

/// Compile-time f64 version of `CLIFFORD_NORM_WEIGHTS` for SIMD accumulation.
///
/// The compiler vectorizes `compute_clifford_norm_sq` over this table into
/// VFMADD231PD instructions (AVX-512) when `opt-level=3` and target includes
/// AVX-512F.
///
/// AX-ID: AXIOMA-001
pub(crate) const CLIFFORD_NORM_WEIGHTS_F64: [f64; TOTAL_BLADES] = {
    let mut w = [0.0f64; TOTAL_BLADES];
    let mut i = 0usize;
    while i < TOTAL_BLADES {
        w[i] = CLIFFORD_NORM_WEIGHTS[i] as f64;
        i += 1;
    }
    w
};

/// ⟨A·Ã⟩₀ with Kahan compensated summation.
///
/// Kahan summation reduces floating-point error from O(n·ε) to O(ε)
/// for the signed Lorentz metric accumulation. Critical for correct
/// null-vector detection in G(1,3) where timelike and spacelike
/// contributions partially cancel.
///
/// INVARIANT: result matches `derive_all_metadata` `clifford_norm_sq`.
/// AX-ID: AXIOMA-001
#[inline]
pub fn compute_clifford_norm_sq(coeffs: &[f64; 16]) -> f64 {
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for (coeff, w) in coeffs.iter().zip(CLIFFORD_NORM_WEIGHTS_F64.iter()) {
        let y = (coeff * coeff).mul_add(*w, -comp);
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Clifford norm scalar: sqrt(|⟨A·Ã⟩₀|).
///
/// **PROHIBITED** for use in the CS gate. Only for physics invariants.
///
/// AX-ID: AXIOMA-001
#[inline]
pub fn compute_clifford_norm(coeffs: &[f64; 16]) -> f64 {
    compute_clifford_norm_sq(coeffs).abs().sqrt()
}

// ── Grade-projection utilities ────────────────────────────────────────────────

/// Extracts blades of exactly the specified Grassmann grade.
///
/// Iterates over `active_mask` with `trailing_zeros()` — only touches
/// active blades. Output assembled via `SparseCliffordVector::from_dense_buf`.
///
/// AX-ID: AXIOMA-001
#[inline]
pub fn grade_project(v: &SparseCliffordVector, grade: usize) -> SparseCliffordVector {
    debug_assert!(grade <= 4, "grade > 4 violates G(1,3) invariant");
    #[allow(clippy::cast_possible_truncation)]
    let g = grade as u8; // grade ≤ 4 (G(1,3) tiene grados 0..4)
    let mut buf = [0.0f64; 16];
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        if GRADE_TABLE[i] == g {
            buf[i] = v.coeffs[i];
        }
        mask &= mask - 1;
    }
    SparseCliffordVector::from_dense_buf(&buf)
}

/// Marker trait for compile-time valid grades in G(1,3).
pub trait ValidGrade {}

/// Type-level grade wrapper used to constrain const generic `G`.
pub struct Grade<const G: usize>;

impl ValidGrade for Grade<0> {}
impl ValidGrade for Grade<1> {}
impl ValidGrade for Grade<2> {}
impl ValidGrade for Grade<3> {}
impl ValidGrade for Grade<4> {}

/// Compile-time mask containing all blade indices of grade `G`.
pub const fn grade_mask<const G: usize>() -> u16
where
    Grade<G>: ValidGrade,
{
    let mut mask = 0u16;
    let mut i = 0usize;
    while i < TOTAL_BLADES {
        if GRADE_TABLE[i] as usize == G {
            mask |= 1u16 << i;
        }
        i += 1;
    }
    mask
}

/// Compile-time grade projection constrained to valid grades (`0..=4`).
///
/// ```compile_fail
/// use genesis_math::{grade_project_ct, SparseCliffordVector};
///
/// let v = SparseCliffordVector::zero();
/// let _ = grade_project_ct::<5>(&v);
/// ```
#[inline]
pub fn grade_project_ct<const G: usize>(v: &SparseCliffordVector) -> SparseCliffordVector
where
    Grade<G>: ValidGrade,
{
    let mut buf = [0.0f64; 16];
    let mut mask = v.active_mask & grade_mask::<G>();
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        buf[i] = v.coeffs[i];
        mask &= mask - 1;
    }
    SparseCliffordVector::from_dense_buf(&buf)
}

/// Even-grade sub-multivector: scalar (k=0) + bivectors (k=2) + 4-vector (k=4).
///
/// Single pass over `active_mask`.
#[inline]
pub fn even_grade(v: &SparseCliffordVector) -> SparseCliffordVector {
    let mut buf = [0.0f64; 16];
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        if (GRADE_TABLE[i] & 1) == 0 {
            buf[i] = v.coeffs[i];
        }
        mask &= mask - 1;
    }
    SparseCliffordVector::from_dense_buf(&buf)
}

/// Odd-grade sub-multivector: vectors (k=1) + trivectors (k=3).
#[inline]
pub fn odd_grade(v: &SparseCliffordVector) -> SparseCliffordVector {
    let mut buf = [0.0f64; 16];
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        // CRYSTAL: O62 — inevitable
        if (GRADE_TABLE[i] & 1) == 1 {
            buf[i] = v.coeffs[i];
        }
        mask &= mask - 1;
    }
    SparseCliffordVector::from_dense_buf(&buf)
}

/// Clifford reverse: Ã of multivector A.
///
/// Each blade of grade k is scaled by `REVERSE_SIGN[k]` ∈ {+1i8, −1i8}.
///
/// The only float operation is `coef * (sign as f64)`. No float branching,
/// no ±0.0 risk.
///
/// For the Clifford norm: ‖A‖² = ⟨A · Ã⟩₀.
///
/// AX-ID: AXIOMA-001
pub fn reverse(v: &SparseCliffordVector) -> SparseCliffordVector {
    let mut buf = [0.0f64; 16];
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        let grade = GRADE_TABLE[i] as usize;
        let val = if REVERSE_SIGN[grade] == 1 {
            v.coeffs[i]
        } else {
            -v.coeffs[i]
        };
        buf[i] = if val.abs() > COGNITIVE_PLANCK_CONSTANT {
            val
        } else {
            0.0
        };
        mask &= mask - 1;
    }
    SparseCliffordVector::from_dense_buf(&buf)
}

/// Returns a `u8` bitmask of all grades present in the multivector.
///
/// Bit k set ↔ at least one blade of grade k is active.
/// G(1,3) has grades 0..=4 → fits in `u8`. Zero allocation.
pub const fn grades_present(v: &SparseCliffordVector) -> u8 {
    // active_mask already encodes which blades are present.
    let mut result = 0u8;
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        result |= 1u8 << GRADE_TABLE[i];
        mask &= mask - 1;
    }
    result
}

/// Returns the maximum grade present, or 0 for the zero multivector.
pub const fn max_grade(v: &SparseCliffordVector) -> u8 {
    let mut best = 0u8;
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        let g = GRADE_TABLE[i];
        // CRYSTAL: O66 — inevitable
        // CRYSTAL: FO57 — inevitable
        if g > best {
            best = g;
        }
        mask &= mask - 1;
    }
    best
}

/// Returns the minimum grade present, or 0 for the zero multivector.
pub const fn min_grade(v: &SparseCliffordVector) -> u8 {
    if v.active_mask == 0 {
        return 0;
    }
    let mut best = 4u8;
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        let g = GRADE_TABLE[i];
        // CRYSTAL: O67 — inevitable
        // CRYSTAL: FO58 — inevitable
        if g < best {
            best = g;
        }
        mask &= mask - 1;
    }
    best
}

/// True if all active blades have the same grade.
pub const fn is_homogeneous(v: &SparseCliffordVector) -> bool {
    if v.active_mask == 0 {
        return true;
    }
    let first_idx = v.active_mask.trailing_zeros() as usize;
    let first_grade = GRADE_TABLE[first_idx];
    let mut mask = v.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        if GRADE_TABLE[i] != first_grade {
            return false;
        }
        mask &= mask - 1;
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[allow(
    clippy::needless_range_loop,
    clippy::approx_constant,
    clippy::bool_assert_comparison
)]
mod tests {
    use super::*;

    fn e(bit: usize) -> SparseCliffordVector {
        SparseCliffordVector::from_iter([(bit, 1.0)]).unwrap()
    }

    fn mv(pairs: &[(usize, f64)]) -> SparseCliffordVector {
        SparseCliffordVector::from_iter(pairs.iter().copied()).unwrap()
    }

    // ── CLIFFORD_NORM_WEIGHTS verification ───────────────────────────────────

    #[test]
    fn clifford_norm_weights_verified_exhaustive() {
        let expected: [i8; 16] = [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1];
        assert_eq!(
            CLIFFORD_NORM_WEIGHTS, expected,
            "CLIFFORD_NORM_WEIGHTS must match mathematically verified values"
        );
    }

    #[test]
    fn reverse_sign_table_all_five_grades_correct() {
        let expected: [i8; 5] = [1, 1, -1, -1, 1];
        assert_eq!(REVERSE_SIGN, expected);
    }

    // ── Lorentz invariance of Clifford norm ───────────────────────────────────

    #[test]
    fn clifford_norm_sq_e0_is_plus_one() {
        // e₀: coeffs[1] = 1.0, CLIFFORD_NORM_WEIGHTS[1] = +1 → norm_sq = +1.
        let mut c = [0.0f64; 16];
        c[1] = 1.0;
        let n = compute_clifford_norm_sq(&c);
        assert!((n - 1.0).abs() < 1e-15, "e₀ norm_sq = {n}, expected +1");
    }

    #[test]
    fn clifford_norm_sq_e1_is_minus_one() {
        let mut c = [0.0f64; 16];
        c[2] = 1.0; // blade index 0b0010 = e₁
        let n = compute_clifford_norm_sq(&c);
        assert!((n - (-1.0)).abs() < 1e-15, "e₁ norm_sq = {n}, expected −1");
    }

    #[test]
    fn null_vector_has_zero_clifford_norm_sq() {
        // e₀ + e₁: norm_sq = 1²×(+1) + 1²×(−1) = 0 (lightlike).
        let v = mv(&[(1, 1.0), (2, 1.0)]);
        assert!(
            v.clifford_norm_sq.abs() < 1e-12,
            "e₀+e₁ norm_sq = {}, expected 0 (null/lightlike)",
            v.clifford_norm_sq
        );
    }

    #[test]
    fn clifford_norm_is_lorentz_invariant_under_boost() {
        // Boost of rapidity η: e₀ → cosh(η)e₀ + sinh(η)e₁, e₁ → sinh(η)e₀ + cosh(η)e₁
        let eta = 1.5f64;
        let (ch, sh) = (eta.cosh(), eta.sinh());

        // 4-vector A = 3e₀ + e₁ (blade indices 1 and 2)
        let a = mv(&[(1, 3.0), (2, 1.0)]);
        let norm_sq_a = a.clifford_norm_sq;

        // Boosted: A' = (3ch+sh)e₀ + (3sh+ch)e₁
        let a_boosted = mv(&[(1, 3.0f64.mul_add(ch, sh)), (2, 3.0f64.mul_add(sh, ch))]);
        let norm_sq_boosted = a_boosted.clifford_norm_sq;

        assert!(
            (norm_sq_a - norm_sq_boosted).abs() < 1e-10,
            "Clifford norm_sq must be Lorentz-invariant: {norm_sq_a} vs {norm_sq_boosted}"
        );
    }

    // ── grade_project ─────────────────────────────────────────────────────────

    #[test]
    fn grade_project_extracts_grade_zero() {
        let v = mv(&[(0, 1.0), (1, 2.0), (3, 3.0)]);
        let g0 = grade_project(&v, 0);
        assert_eq!(g0.coeffs[0], 1.0);
        assert_eq!(g0.active_mask, 0b0000_0000_0000_0001u16);
    }

    #[test]
    fn grade_project_extracts_bivectors() {
        let v = mv(&[(0b0000, 1.0), (0b0001, 2.0), (0b0011, 3.0), (0b1111, 4.0)]);
        let g2 = grade_project(&v, 2);
        assert_eq!(g2.coeffs[0b0011], 3.0);
        assert_eq!(g2.active_mask.count_ones(), 1);
    }

    #[test]
    fn grade_project_empty_when_absent() {
        let v = e(0b0001); // only grade 1
        let g2 = grade_project(&v, 2);
        assert_eq!(g2.active_mask, 0);
    }

    #[test]
    fn even_grade_contains_grades_0_2_4() {
        let v = mv(&[
            (0b0000, 1.0), // grade 0 ✓
            (0b0001, 2.0), // grade 1 ✗
            (0b0011, 3.0), // grade 2 ✓
            (0b0111, 4.0), // grade 3 ✗
            (0b1111, 5.0), // grade 4 ✓
        ]);
        let even = even_grade(&v);
        let cnt = even.active_mask.count_ones();
        assert_eq!(cnt, 3, "grades 0, 2, 4 → 3 active blades");
        // Verify all active blades have even grade.
        let mut mask = even.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            assert_eq!(GRADE_TABLE[i] % 2, 0, "blade {i:#06b} must be even grade");
            mask &= mask - 1;
        }
    }

    #[test]
    fn odd_grade_contains_grades_1_3() {
        let v = mv(&[
            (0b0000, 1.0), // grade 0 ✗
            (0b0001, 2.0), // grade 1 ✓
            (0b0011, 3.0), // grade 2 ✗
            (0b0111, 4.0), // grade 3 ✓
        ]);
        let odd = odd_grade(&v);
        let cnt = odd.active_mask.count_ones();
        assert_eq!(cnt, 2);
        let mut mask = odd.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            assert_eq!(GRADE_TABLE[i] % 2, 1, "blade {i:#06b} must be odd grade");
            mask &= mask - 1;
        }
    }

    // ── reverse operator ──────────────────────────────────────────────────────

    #[test]
    fn reverse_scalar_unchanged() {
        let s = mv(&[(0, 3.0)]);
        let rs = reverse(&s);
        assert_eq!(rs.coeffs[0].to_bits(), s.coeffs[0].to_bits());
    }

    #[test]
    fn reverse_vector_unchanged() {
        let v = mv(&[(0b0001, 1.0), (0b0010, -2.0)]);
        let rv = reverse(&v);
        assert_eq!(rv.coeffs[0b0001].to_bits(), v.coeffs[0b0001].to_bits());
        assert_eq!(rv.coeffs[0b0010].to_bits(), v.coeffs[0b0010].to_bits());
    }

    #[test]
    fn reverse_bivector_negated_bit_exact() {
        let v = mv(&[(0b0011, 5.0)]); // e₀₁, grade 2
        let rv = reverse(&v);
        assert_eq!(
            rv.coeffs[0b0011].to_bits(),
            (-5.0f64).to_bits(),
            "bivector reverse must be −5.0"
        );
    }

    #[test]
    fn reverse_trivector_negated_bit_exact() {
        let v = mv(&[(0b0111, 2.0)]); // grade 3
        let rv = reverse(&v);
        assert_eq!(rv.coeffs[0b0111].to_bits(), (-2.0f64).to_bits());
    }

    #[test]
    fn reverse_pseudoscalar_unchanged_bit_exact() {
        let v = mv(&[(0b1111, 7.0)]); // grade 4
        let rv = reverse(&v);
        assert_eq!(rv.coeffs[0b1111].to_bits(), 7.0f64.to_bits());
    }

    #[test]
    fn reverse_applied_twice_is_identity() {
        let v = mv(&[
            (0b0000, 1.0),
            (0b0001, 2.0),
            (0b0011, 3.0),
            (0b0111, 4.0),
            (0b1111, 5.0),
        ]);
        let doubled = reverse(&reverse(&v));
        for i in 0..16 {
            assert!(
                (v.coeffs[i] - doubled.coeffs[i]).abs() < 1e-12,
                "blade {i}: reverse² ≠ identity"
            );
        }
    }

    // ── grades_present ────────────────────────────────────────────────────────

    #[test]
    fn grades_present_bitmask() {
        let v = mv(&[(0, 1.0), (0b0011, 2.0)]); // grades 0 and 2
        let mask = grades_present(&v);
        assert_ne!(mask & (1 << 0), 0, "grade 0 must be set");
        assert_ne!(mask & (1 << 2), 0, "grade 2 must be set");
        assert_eq!(mask & (1 << 1), 0, "grade 1 must not be set");
    }

    #[test]
    fn grades_present_all_five() {
        let v = mv(&[
            (0b0000, 1.0),
            (0b0001, 2.0),
            (0b0011, 3.0),
            (0b0111, 4.0),
            (0b1111, 5.0),
        ]);
        assert_eq!(grades_present(&v), 0b00011111u8);
    }

    #[test]
    fn grades_present_empty_is_zero() {
        assert_eq!(grades_present(&SparseCliffordVector::zero()), 0u8);
    }

    // ── max_grade / min_grade / is_homogeneous ────────────────────────────────

    #[test]
    fn max_grade_correct() {
        let v = mv(&[(0, 1.0), (0b0001, 2.0), (0b0011, 3.0)]);
        assert_eq!(max_grade(&v), 2u8);
    }

    #[test]
    fn min_grade_correct() {
        let v = mv(&[(0b0001, 1.0), (0b0011, 2.0), (0b1111, 3.0)]);
        assert_eq!(min_grade(&v), 1u8);
    }

    #[test]
    fn is_homogeneous_pure_vector() {
        let v = mv(&[(0b0001, 1.0), (0b0010, 2.0)]); // both grade 1
        assert!(is_homogeneous(&v));
    }

    #[test]
    fn is_homogeneous_mixed_false() {
        let v = mv(&[(0b0001, 1.0), (0b0011, 2.0)]); // grades 1 and 2
        assert!(!is_homogeneous(&v));
    }

    #[test]
    fn is_homogeneous_empty_is_true() {
        assert!(is_homogeneous(&SparseCliffordVector::zero()));
    }

    // ── grade_project partition of unity ─────────────────────────────────────

    #[test]
    fn grade_projection_partition_of_unity() {
        let v = mv(&[
            (0b0000, 1.0),
            (0b0001, 2.0),
            (0b0011, 3.0),
            (0b0111, 4.0),
            (0b1111, 5.0),
        ]);
        let mut buf = [0.0f64; 16];
        for g in 0..=4 {
            let proj = grade_project(&v, g);
            for i in 0..16 {
                buf[i] += proj.coeffs[i];
            }
        }
        for i in 0..16 {
            assert!(
                (buf[i] - v.coeffs[i]).abs() < 1e-12,
                "blade {i:#06b}: partition-of-unity error"
            );
        }
    }

    #[test]
    fn grade_project_ct_matches_runtime_for_1000_vectors() {
        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        for _ in 0..1000 {
            let mut coeffs = [0.0f64; 16];
            for coeff in &mut coeffs {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let sample = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                if state.trailing_zeros() >= 2 {
                    *coeff = 0.0;
                } else {
                    *coeff = (sample * 2.0) - 1.0;
                }
            }
            let v = SparseCliffordVector::from_dense_buf(&coeffs);
            let ct = grade_project_ct::<2>(&v);
            let rt = grade_project(&v, 2);
            for i in 0..16 {
                assert_eq!(
                    ct.coeffs[i].to_bits(),
                    rt.coeffs[i].to_bits(),
                    "blade {i}: compile-time and runtime projection diverged"
                );
            }
            assert_eq!(ct.active_mask, rt.active_mask);
        }
    }
}
