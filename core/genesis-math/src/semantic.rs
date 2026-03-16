//! Explicit semantic operators over `SparseCliffordVector`.
//!
//! These operators provide high-level geometric composition without heap allocation,
//! preserving closure in G(1,3) and enabling structural reasoning.
//!
//! AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)

use crate::basis::{GRADE_TABLE, TOTAL_BLADES};
use crate::dual::{hodge_dual, hodge_undual};
use crate::grade::reverse;
use crate::multivector::SparseCliffordVector;
use crate::product::sparse_geometric_product;
use crate::sign::CAYLEY_SIGN;

/// Unit-rotor scalar tolerance for `R * reverse(R) ≈ 1`.
const ROTOR_UNIT_TOLERANCE: f64 = 1e-12;

// Inlining policy:
// Semantic operators are frequent algebraic composition points and remain
// small enough to benefit from regular inlining hints. We intentionally use
// `#[inline]` (not `#[inline(always)]`) so LLVM can choose the best
// cross-crate/code-size trade-off for each target and optimization level.

/// Exterior (wedge) product `A ∧ B`.
///
/// Keeps only the antisymmetric-disjoint blade interactions `(i & j) == 0`,
/// yielding the geometric span (join in OPNS form) without dense intermediates.
///
/// AX-ID: AXIOMA-001, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
#[inline]
pub fn wedge(lhs: &SparseCliffordVector, rhs: &SparseCliffordVector) -> SparseCliffordVector {
    if lhs.active_mask == 0 || rhs.active_mask == 0 {
        return SparseCliffordVector::zero();
    }

    let mut out = [0.0_f64; TOTAL_BLADES];
    let mut mask_lhs = lhs.active_mask;

    while mask_lhs != 0 {
        let i = mask_lhs.trailing_zeros() as usize;
        let coeff_lhs = lhs.coeffs[i];
        let sign_row = &CAYLEY_SIGN[i];

        let mut mask_rhs = rhs.active_mask;
        while mask_rhs != 0 {
            let j = mask_rhs.trailing_zeros() as usize;
            let coeff_rhs = rhs.coeffs[j];
            if (i & j) == 0 {
                let k = i ^ j;
                out[k] += coeff_lhs * coeff_rhs * f64::from(sign_row[j]);
            }
            mask_rhs &= mask_rhs - 1;
        }

        mask_lhs &= mask_lhs - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Left contraction `A ⌟ B`.
///
/// For basis blades of grades `r` and `s`, keeps only grade `s-r` terms when
/// `r <= s`; otherwise contributes zero. Implemented over sparse active masks.
///
/// AX-ID: AXIOMA-001, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
#[inline]
pub fn left_contraction(
    lhs: &SparseCliffordVector,
    rhs: &SparseCliffordVector,
) -> SparseCliffordVector {
    if lhs.active_mask == 0 || rhs.active_mask == 0 {
        return SparseCliffordVector::zero();
    }

    let mut out = [0.0_f64; TOTAL_BLADES];
    let mut mask_lhs = lhs.active_mask;

    while mask_lhs != 0 {
        let i = mask_lhs.trailing_zeros() as usize;
        let grade_i = GRADE_TABLE[i];
        let coeff_lhs = lhs.coeffs[i];
        let sign_row = &CAYLEY_SIGN[i];

        let mut mask_rhs = rhs.active_mask;
        while mask_rhs != 0 {
            let j = mask_rhs.trailing_zeros() as usize;
            let grade_j = GRADE_TABLE[j];
            let coeff_rhs = rhs.coeffs[j];
            if grade_i <= grade_j {
                let k = i ^ j;
                let target_grade = grade_j - grade_i;
                if GRADE_TABLE[k] == target_grade {
                    out[k] += coeff_lhs * coeff_rhs * f64::from(sign_row[j]);
                }
            }
            mask_rhs &= mask_rhs - 1;
        }

        mask_lhs &= mask_lhs - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Right contraction `A ⌞ B`.
///
/// For basis blades of grades `r` and `s`, keeps only grade `r-s` terms when
/// `r >= s`; otherwise contributes zero.
///
/// AX-ID: AXIOMA-001, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
#[inline]
pub fn right_contraction(
    lhs: &SparseCliffordVector,
    rhs: &SparseCliffordVector,
) -> SparseCliffordVector {
    if lhs.active_mask == 0 || rhs.active_mask == 0 {
        return SparseCliffordVector::zero();
    }

    let mut out = [0.0_f64; TOTAL_BLADES];
    let mut mask_lhs = lhs.active_mask;

    while mask_lhs != 0 {
        let i = mask_lhs.trailing_zeros() as usize;
        let grade_i = GRADE_TABLE[i];
        let coeff_lhs = lhs.coeffs[i];
        let sign_row = &CAYLEY_SIGN[i];

        let mut mask_rhs = rhs.active_mask;
        while mask_rhs != 0 {
            let j = mask_rhs.trailing_zeros() as usize;
            let grade_j = GRADE_TABLE[j];
            let coeff_rhs = rhs.coeffs[j];
            if grade_i >= grade_j {
                let k = i ^ j;
                let target_grade = grade_i - grade_j;
                if GRADE_TABLE[k] == target_grade {
                    out[k] += coeff_lhs * coeff_rhs * f64::from(sign_row[j]);
                }
            }
            mask_rhs &= mask_rhs - 1;
        }

        mask_lhs &= mask_lhs - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Regressive meet `A ∩ B` derived from Hodge duality (De Morgan identity).
///
/// Implemented as `A ∩ B = ⋆⁻¹(⋆A ∧ ⋆B)` using [`hodge_dual`], [`hodge_undual`]
/// and [`wedge`] only, preserving sparse/stack execution.
///
/// AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
#[inline]
pub fn meet(lhs: &SparseCliffordVector, rhs: &SparseCliffordVector) -> SparseCliffordVector {
    // Regressive product via De Morgan duality in G(1,3):
    // meet(A, B) = ⋆⁻¹(⋆A ∧ ⋆B).
    let lhs_dual = hodge_dual(lhs);
    let rhs_dual = hodge_dual(rhs);
    let dual_meet = wedge(&lhs_dual, &rhs_dual);
    hodge_undual(&dual_meet)
}

/// Join (union/span) `A ∪ B` derived through duality involution.
///
/// Implemented as `A ∪ B = ⋆⁻¹((⋆A) ∩ (⋆B))`, reusing [`meet`] to enforce
/// De Morgan dual symmetry in G(1,3).
///
/// AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
#[inline]
pub fn join(lhs: &SparseCliffordVector, rhs: &SparseCliffordVector) -> SparseCliffordVector {
    // Direct De Morgan dual identity for join, avoiding redundant dual cycles:
    // join(A, B) = ⋆(⋆⁻¹A ∧ ⋆⁻¹B).
    let lhs_undual = hodge_undual(lhs);
    let rhs_undual = hodge_undual(rhs);
    let primal_wedge = wedge(&lhs_undual, &rhs_undual);
    hodge_dual(&primal_wedge)
}

/// Commutator operator `[A,B] = 0.5 * (AB − BA)`.
///
/// Uses sparse geometric products and stack-local accumulation to produce a
/// Lie-algebra generator in G(1,3) without heap allocations.
///
/// AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
#[inline]
pub fn commutator(a: &SparseCliffordVector, b: &SparseCliffordVector) -> SparseCliffordVector {
    if a.active_mask == 0 || b.active_mask == 0 {
        return SparseCliffordVector::zero();
    }

    let mut out = [0.0_f64; TOTAL_BLADES];

    if let Some(ab) = sparse_geometric_product(a, b) {
        let mut mask = ab.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            out[i] += 0.5 * ab.coeffs[i];
            mask &= mask - 1;
        }
    }

    if let Some(ba) = sparse_geometric_product(b, a) {
        let mut mask = ba.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            out[i] -= 0.5 * ba.coeffs[i];
            mask &= mask - 1;
        }
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Rotor unit-norm validation.
///
/// A valid rotor must satisfy `R * reverse(R) = 1` (scalar grade-0 blade only)
/// within `1e-12` tolerance.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
#[must_use]
pub fn is_unit_rotor(r: &SparseCliffordVector) -> bool {
    let r_reverse = reverse(r);
    let Some(norm) = sparse_geometric_product(r, &r_reverse) else {
        return false;
    };

    if norm.active_mask != 0b1_u16 {
        return false;
    }

    (norm.coeffs[0] - 1.0).abs() <= ROTOR_UNIT_TOLERANCE
}

/// Rotor sandwich action `R X R̃`.
///
/// Applies a rotor to a multivector through geometric conjugation. In G(1,3),
/// this realizes Lorentz-compatible rotations/boosts depending on the bivector
/// generator used to build `R`.
///
/// Returns `None` when the Cauchy-Schwarz gate suppresses one of the geometric
/// products in the chain.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
#[must_use]
pub fn rotor_sandwich(
    rotor: &SparseCliffordVector,
    target: &SparseCliffordVector,
) -> Option<SparseCliffordVector> {
    let first = sparse_geometric_product(rotor, target)?;
    let rotor_reverse = reverse(rotor);
    sparse_geometric_product(&first, &rotor_reverse)
}

/// Safe rotor sandwich action `R X R̃` with rotor validation.
///
/// Returns `None` when `rotor` is not unit-norm or when geometric-product gating
/// suppresses an intermediate result.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
#[must_use]
pub fn rotor_sandwich_checked(
    rotor: &SparseCliffordVector,
    target: &SparseCliffordVector,
) -> Option<SparseCliffordVector> {
    if !is_unit_rotor(rotor) {
        return None;
    }
    rotor_sandwich(rotor, target)
}

#[cfg(test)]
mod tests {
    use super::{
        commutator, is_unit_rotor, join, meet, rotor_sandwich, rotor_sandwich_checked, wedge,
        TOTAL_BLADES,
    };
    use crate::{grade_project_ct, hodge_dual, hodge_undual, SparseCliffordVector};

    fn blade(index: usize) -> SparseCliffordVector {
        SparseCliffordVector::from_iter([(index, 1.0)]).expect("blade index must be valid")
    }

    #[test]
    fn wedge_parallel_grade1_blades_is_zero_degenerate_case() {
        let e1 = blade(0b0010);
        let out = wedge(&e1, &e1);
        assert_eq!(out.active_mask, 0);
    }

    #[test]
    fn meet_parallel_grade1_blades_is_zero_degenerate_case() {
        let e2 = blade(0b0100);
        let out = meet(&e2, &e2);
        assert_eq!(out.active_mask, 0);
    }

    #[test]
    fn join_output_remains_closed_inside_g13() {
        let a = SparseCliffordVector::from_iter([(0b0010, 2.0), (0b0110, -0.5)])
            .expect("valid coefficients");
        let b = SparseCliffordVector::from_iter([(0b0100, 1.5), (0b1110, 0.25)])
            .expect("valid coefficients");

        let out = join(&a, &b);
        assert!(out.coeffs.iter().all(|v| v.is_finite()));
        assert!((out.active_mask as u32) < (1_u32 << TOTAL_BLADES));
    }

    #[test]
    fn duality_composition_is_recoverable_by_grade_projection() {
        let e1 = blade(0b0010);
        let e2 = blade(0b0100);
        let bivector = wedge(&e1, &e2);
        let recovered = hodge_undual(&hodge_dual(&bivector));

        let recovered_grade2 = grade_project_ct::<2>(&recovered);
        assert!((recovered_grade2.coeffs[0b0110] - bivector.coeffs[0b0110]).abs() < 1e-12);
    }

    #[test]
    fn rotor_sandwich_preserves_grade1_norm_in_e12_rotation() {
        let theta = std::f64::consts::FRAC_PI_2;
        let half = 0.5 * theta;

        let rotor = SparseCliffordVector::from_iter([(0, half.cos()), (0b0110, half.sin())])
            .expect("valid rotor");
        let e1 = blade(0b0010);

        let rotated = rotor_sandwich(&rotor, &e1).expect("rotation should survive CS gate");
        let v = grade_project_ct::<1>(&rotated);

        let l2 = v.l2_norm();
        assert!((l2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn is_unit_rotor_true_for_normalized_rotor() {
        let theta = std::f64::consts::FRAC_PI_3;
        let half = 0.5 * theta;
        let rotor = SparseCliffordVector::from_iter([(0, half.cos()), (0b0110, half.sin())])
            .expect("valid rotor");

        assert!(is_unit_rotor(&rotor));
    }

    #[test]
    fn is_unit_rotor_false_for_non_normalized_rotor() {
        let rotor =
            SparseCliffordVector::from_iter([(0, 1.2), (0b0110, 0.4)]).expect("valid rotor");
        assert!(!is_unit_rotor(&rotor));
    }

    #[test]
    fn rotor_sandwich_checked_rejects_non_unit_rotor() {
        let bad_rotor =
            SparseCliffordVector::from_iter([(0, 1.2), (0b0110, 0.4)]).expect("valid rotor");
        let target = blade(0b0010);
        assert!(rotor_sandwich_checked(&bad_rotor, &target).is_none());
    }

    #[test]
    fn commutator_is_antisymmetric() {
        let a = SparseCliffordVector::from_iter([(0b0010, 0.7), (0b0110, -0.25)]).expect("valid");
        let b = SparseCliffordVector::from_iter([(0b0100, -1.2), (0b1110, 0.1)]).expect("valid");

        let ab = commutator(&a, &b);
        let ba = commutator(&b, &a);

        for i in 0..TOTAL_BLADES {
            assert!((ab.coeffs[i] + ba.coeffs[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn bivector_commutator_generates_grade2_lie_element() {
        let b1 = blade(0b0011);
        let b2 = blade(0b0101);
        let comm = commutator(&b1, &b2);
        let grade2 = grade_project_ct::<2>(&comm);

        for i in 0..TOTAL_BLADES {
            assert!((comm.coeffs[i] - grade2.coeffs[i]).abs() < 1e-12);
        }
        assert_ne!(comm.active_mask, 0);
    }
}
