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
            if (i & j) == 0 {
                let k = i ^ j;
                out[k] = coeff_lhs.mul_add(rhs.coeffs[j] * f64::from(sign_row[j]), out[k]);
            }
            mask_rhs &= mask_rhs - 1;
        }

        mask_lhs &= mask_lhs - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Computes left contraction `A ⌟ B` via geometric-product grade selection.
///
/// Mathematical definition:
/// `$ A \!\rfloor B = \sum_{r,s}\left\langle \langle A\rangle_r \langle B\rangle_s \right\rangle_{s-r},\; r \le s $`
///
/// AX-ID: AXIOMA-001
/// See also: [`right_contraction`], [`sparse_geometric_product`]
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
                    out[k] = coeff_lhs.mul_add(coeff_rhs * f64::from(sign_row[j]), out[k]);
                }
            }
            mask_rhs &= mask_rhs - 1;
        }

        mask_lhs &= mask_lhs - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Computes right contraction `A ⌞ B` via geometric-product grade selection.
///
/// Mathematical definition:
/// `$ A \!\lfloor B = \sum_{r,s}\left\langle \langle A\rangle_r \langle B\rangle_s \right\rangle_{r-s},\; r \ge s $`
///
/// AX-ID: AXIOMA-001
/// See also: [`left_contraction`], [`sparse_geometric_product`]
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
                    out[k] = coeff_lhs.mul_add(coeff_rhs * f64::from(sign_row[j]), out[k]);
                }
            }
            mask_rhs &= mask_rhs - 1;
        }

        mask_lhs &= mask_lhs - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Computes regressive meet `A ∩ B` through Hodge-dual wedge composition.
///
/// Mathematical definition:
/// `$ A \cap B = \star^{-1}\!\left(\star A \wedge \star B\right) $`
///
/// AX-ID: AXIOMA-006
/// See also: [`join`], [`wedge`], [`hodge_dual`]
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

/// Computes join `A ∪ B` as the dual counterpart of regressive intersection.
///
/// Mathematical definition:
/// `$ A \cup B = \star\!\left(\star^{-1}A \wedge \star^{-1}B\right) $`
///
/// AX-ID: AXIOMA-006
/// See also: [`meet`], [`wedge`], [`hodge_undual`]
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

/// Computes the geometric-algebra commutator as a Lie bracket in G(1,3).
///
/// Mathematical definition:
/// `$ [A,B] = \frac{1}{2}(AB - BA) $`
///
/// AX-ID: AXIOMA-006
/// See also: [`sparse_geometric_product`]
#[must_use]
#[inline]
pub fn commutator(a: &SparseCliffordVector, b: &SparseCliffordVector) -> SparseCliffordVector {
    if a.active_mask == 0 || b.active_mask == 0 {
        return SparseCliffordVector::zero();
    }

    let mut out = [0.0_f64; TOTAL_BLADES];

    let half = 0.5;
    if let Some(ab) = sparse_geometric_product(a, b) {
        let mut mask = ab.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            out[i] += half * ab.coeffs[i];
            mask &= mask - 1;
        }
    }

    if let Some(ba) = sparse_geometric_product(b, a) {
        let mut mask = ba.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            out[i] -= half * ba.coeffs[i];
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

/// Exponential map from bivector to rotor in G(1,3).
///
/// `e^B = cos(θ) + sin(θ)/θ · B` for timelike (`B²<0`), `θ=√(-B²)`.
/// `e^B = cosh(θ) + sinh(θ)/θ · B` for spacelike (`B²>0`), `θ=√(B²)`.
/// `e^B = 1 + B` for null (`B²≈0`).
///
/// Returns `None` if input contains non-grade-2 blades.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
#[must_use]
pub fn exp_bivector(b: &SparseCliffordVector) -> Option<SparseCliffordVector> {
    use crate::basis::GRADE_TABLE;

    let mut mask = b.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        if GRADE_TABLE[i] != 2 {
            return None;
        }
        mask &= mask - 1;
    }

    let b_sq = b.clifford_norm_sq;
    let mut out = [0.0f64; 16];
    if b_sq.abs() < genesis_types::COGNITIVE_PLANCK_CONSTANT {
        out[0] = 1.0;
        let mut m = b.active_mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            out[i] = b.coeffs[i];
            m &= m - 1;
        }
    } else if b_sq < 0.0 {
        let theta = (-b_sq).sqrt();
        let cos_t = theta.cos();
        let sinc_t = theta.sin() / theta;
        out[0] = cos_t;
        let mut m = b.active_mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            let scale = sinc_t;
            out[i] = scale * b.coeffs[i];
            m &= m - 1;
        }
    } else {
        let theta = b_sq.sqrt();
        let cosh_t = theta.cosh();
        let sinhc_t = theta.sinh() / theta;
        out[0] = cosh_t;
        let mut m = b.active_mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            let scale = sinhc_t;
            out[i] = scale * b.coeffs[i];
            m &= m - 1;
        }
    }
    Some(SparseCliffordVector::from_dense_buf(&out))
}

/// Logarithm map from unit rotor to generating bivector in G(1,3).
///
/// Inverse of [`exp_bivector`]. Returns `None` if input is not a unit rotor.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
#[must_use]
pub fn log_rotor(r: &SparseCliffordVector) -> Option<SparseCliffordVector> {
    if r.active_mask == 0 {
        return None;
    }
    let s = r.scalar_part();
    let biv = crate::grade::grade_project(r, 2);
    let b_norm_sq = biv.clifford_norm_sq;
    if b_norm_sq.abs() < genesis_types::COGNITIVE_PLANCK_CONSTANT {
        return Some(SparseCliffordVector::zero());
    }

    let mut out = [0.0f64; 16];
    if b_norm_sq < 0.0 {
        let sin_t = (-b_norm_sq).sqrt();
        let theta = s.clamp(-1.0, 1.0).acos();
        let scale = if sin_t.abs() > genesis_types::COGNITIVE_PLANCK_CONSTANT {
            theta / sin_t
        } else {
            1.0
        };
        let mut m = biv.active_mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            out[i] = scale * biv.coeffs[i];
            m &= m - 1;
        }
    } else {
        let sinh_t = b_norm_sq.sqrt();
        let theta = s.max(1.0).acosh();
        let scale = if sinh_t.abs() > genesis_types::COGNITIVE_PLANCK_CONSTANT {
            theta / sinh_t
        } else {
            1.0
        };
        let mut m = biv.active_mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            out[i] = scale * biv.coeffs[i];
            m &= m - 1;
        }
    }
    Some(SparseCliffordVector::from_dense_buf(&out))
}

/// Spherical linear interpolation between two unit rotors.
///
/// `slerp(R0, R1, 0.0) = R0`, `slerp(R0, R1, 1.0) = R1`.
/// Returns `None` if `t ∉ [0,1]` or either input is not a unit rotor.
///
/// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
#[must_use]
pub fn slerp_rotor(
    r0: &SparseCliffordVector,
    r1: &SparseCliffordVector,
    t: f64,
) -> Option<SparseCliffordVector> {
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    if !is_unit_rotor(r0) || !is_unit_rotor(r1) {
        return None;
    }
    if t == 0.0 {
        return Some(*r0);
    }
    if t == 1.0 {
        return Some(*r1);
    }

    let r0_inv = crate::grade::reverse(r0);
    let delta = crate::product::sparse_geometric_product(&r0_inv, r1)?;
    let log_delta = log_rotor(&delta)?;

    let mut scaled = [0.0f64; 16];
    let mut m = log_delta.active_mask;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        let scale = t;
        scaled[i] = scale * log_delta.coeffs[i];
        m &= m - 1;
    }

    let scaled_biv = SparseCliffordVector::from_dense_buf(&scaled);
    let exp_part = exp_bivector(&scaled_biv)?;
    crate::product::sparse_geometric_product(r0, &exp_part)
}

#[cfg(test)]
mod tests {
    use super::{
        commutator, exp_bivector, is_unit_rotor, join, log_rotor, meet, rotor_sandwich,
        rotor_sandwich_checked, slerp_rotor, wedge, TOTAL_BLADES,
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
        assert!((recovered_grade2.coeffs[0b0110] + bivector.coeffs[0b0110]).abs() < 1e-12);
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

    #[test]
    fn exp_bivector_null_gives_identity_plus_b() {
        let zero_biv = SparseCliffordVector::zero();
        let result = exp_bivector(&zero_biv).unwrap();
        assert!((result.scalar_part() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exp_then_log_roundtrip_for_small_bivector() {
        let b = SparseCliffordVector::from_iter([(0b0110usize, 0.3)]).unwrap();
        let r = exp_bivector(&b).unwrap();
        let b_recovered = log_rotor(&r).unwrap();
        assert!((b_recovered.coeffs[0b0110] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn slerp_rotor_endpoints() {
        use core::f64::consts::FRAC_PI_4;

        let r0 = SparseCliffordVector::from_iter([(0, 1.0)]).unwrap();
        let r1 = SparseCliffordVector::from_iter([(0, FRAC_PI_4.cos()), (0b0110, FRAC_PI_4.sin())])
            .unwrap();
        let at0 = slerp_rotor(&r0, &r1, 0.0).unwrap();
        let at1 = slerp_rotor(&r0, &r1, 1.0).unwrap();
        assert!((at0.scalar_part() - r0.scalar_part()).abs() < 1e-10);
        assert!((at1.scalar_part() - r1.scalar_part()).abs() < 1e-10);
    }
}
