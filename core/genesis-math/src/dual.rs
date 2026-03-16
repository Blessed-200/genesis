//! Forward-mode dual numbers for sparse G(1,3) multivectors.

use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;

use crate::basis::TOTAL_BLADES;
use crate::multivector::SparseCliffordVector;
use crate::sign::CAYLEY_SIGN;

/// Pseudoscalar blade index `e₀₁₂₃` in G(1,3).
const PSEUDOSCALAR_INDEX: usize = 0b1111;

/// Right Hodge dual `⋆A = A I⁻¹`, with `I = e₀₁₂₃`.
///
/// In G(1,3), `I² = -1`, therefore `I⁻¹ = -I`. This implementation performs
/// the multiplication directly on sparse active blades (stack-only, zero alloc).
///
/// AX-ID: AXIOMA-001, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
pub fn hodge_dual(value: &SparseCliffordVector) -> SparseCliffordVector {
    let mut out = [0.0_f64; TOTAL_BLADES];
    let mut mask = value.active_mask;

    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        let k = i ^ PSEUDOSCALAR_INDEX;
        out[k] += -value.coeffs[i] * f64::from(CAYLEY_SIGN[i][PSEUDOSCALAR_INDEX]);
        mask &= mask - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Inverse right Hodge dual `⋆⁻¹A = A I` with `I = e₀₁₂₃`.
///
/// Used to map dual-space constructions back into primal-space blades.
///
/// AX-ID: AXIOMA-001, H_estructura (LEY_FUNDACIONAL §3.1)
#[must_use]
pub fn hodge_undual(value: &SparseCliffordVector) -> SparseCliffordVector {
    let mut out = [0.0_f64; TOTAL_BLADES];
    let mut mask = value.active_mask;

    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        let k = i ^ PSEUDOSCALAR_INDEX;
        out[k] += -value.coeffs[i] * f64::from(CAYLEY_SIGN[i][PSEUDOSCALAR_INDEX]);
        mask &= mask - 1;
    }

    SparseCliffordVector::from_dense_buf(&out)
}

/// Scalar dual number `value + grad ε` for forward AD.
/// Scalar dual number `f + ε·f'` for forward-mode automatic differentiation.
///
/// Used to propagate gradients through sparse G(1,3) geometric products.
/// Arithmetic is standard dual-number algebra: `(a+bε)(c+dε) = ac + (ad+bc)ε`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    /// Primal value (the regular coefficient).
    pub value: f64,
    /// Dual component (the derivative / sensitivity).
    pub grad: f64,
}

/// Dual-valued dense multivector with fixed metadata and stack layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseDualVector {
    /// Dual-valued coefficient array for all 16 blades of G(1,3).
    pub coeffs: [Dual; TOTAL_BLADES],
    /// Dual-valued Lorentz-invariant norm squared `⟨A·Ã⟩₀`.
    pub clifford_norm_sq: Dual,
    /// Dual-valued maximum absolute coefficient (used for CS gate).
    pub max_abs_coeff: Dual,
    /// Active blade bitmask (bit i = 1 iff `|coeffs[i].value| > PLANCK`).
    pub active_mask: u16,
    _pad: [u8; 14],
}

impl SparseDualVector {
    /// Returns the zero dual multivector (all coefficients zero, inactive mask).
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            coeffs: [Dual {
                value: 0.0,
                grad: 0.0,
            }; TOTAL_BLADES],
            clifford_norm_sq: Dual {
                value: 0.0,
                grad: 0.0,
            },
            max_abs_coeff: Dual {
                value: 0.0,
                grad: 0.0,
            },
            active_mask: 0,
            _pad: [0u8; 14],
        }
    }

    /// Constructs a `SparseDualVector` from a dense 16-element dual buffer.
    ///
    /// Computes norm, active mask, and max coefficient in a single pass.
    #[must_use]
    pub fn from_dense_buf(buf: &[Dual; TOTAL_BLADES]) -> Self {
        let mut active_mask = 0u16;
        let mut max_abs_coeff_value = 0.0f64;
        let mut clifford_norm_sq_value = 0.0f64;
        let mut clifford_norm_sq_grad = 0.0f64;

        for (k, dual) in buf.iter().copied().enumerate() {
            let abs = dual.value.abs();
            if abs > COGNITIVE_PLANCK_CONSTANT {
                active_mask |= 1u16 << k;
                if abs > max_abs_coeff_value {
                    max_abs_coeff_value = abs;
                }
            }
            let weight = crate::grade::CLIFFORD_NORM_WEIGHTS_F64[k];
            clifford_norm_sq_value += dual.value * dual.value * weight;
            clifford_norm_sq_grad += 2.0 * dual.value * dual.grad * weight;
        }

        Self {
            coeffs: *buf,
            clifford_norm_sq: Dual {
                value: clifford_norm_sq_value,
                grad: clifford_norm_sq_grad,
            },
            max_abs_coeff: Dual {
                value: max_abs_coeff_value,
                grad: 0.0,
            },
            active_mask,
            _pad: [0u8; 14],
        }
    }
}

impl From<SparseCliffordVector> for SparseDualVector {
    fn from(value: SparseCliffordVector) -> Self {
        let mut coeffs = [Dual {
            value: 0.0,
            grad: 0.0,
        }; TOTAL_BLADES];
        for (idx, coeff) in value.coeffs.into_iter().enumerate() {
            coeffs[idx] = Dual {
                value: coeff,
                grad: 0.0,
            };
        }

        Self {
            coeffs,
            clifford_norm_sq: Dual {
                value: value.clifford_norm_sq,
                grad: 0.0,
            },
            max_abs_coeff: Dual {
                value: value.max_abs_coeff,
                grad: 0.0,
            },
            active_mask: value.active_mask,
            _pad: [0u8; 14],
        }
    }
}

/// Geometric product of two dual sparse multivectors in G(1,3).
///
/// Returns `None` if the Cauchy-Schwarz energy gate fires (product vanishes
/// below `COGNITIVE_PLANCK_CONSTANT`). Propagates dual (gradient) components
/// via the Leibniz rule: `d(AB) = dA·B + A·dB`.
#[must_use]
pub fn geometric_product_dual(
    lhs: &SparseDualVector,
    rhs: &SparseDualVector,
) -> Option<SparseDualVector> {
    if lhs.active_mask == 0 || rhs.active_mask == 0 {
        return None;
    }
    if !(lhs.max_abs_coeff.value.is_finite() && rhs.max_abs_coeff.value.is_finite()) {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    if lhs.max_abs_coeff.value * rhs.max_abs_coeff.value * (TOTAL_BLADES as f64)
        < COGNITIVE_PLANCK_CONSTANT
    {
        return None;
    }

    let mut result = [Dual {
        value: 0.0,
        grad: 0.0,
    }; TOTAL_BLADES];

    let mut lhs_mask = lhs.active_mask;
    while lhs_mask != 0 {
        let lhs_index = lhs_mask.trailing_zeros() as usize;
        let lhs_coeff = lhs.coeffs[lhs_index];
        let sign_row = &CAYLEY_SIGN[lhs_index];

        let mut rhs_mask = rhs.active_mask;
        while rhs_mask != 0 {
            let rhs_index = rhs_mask.trailing_zeros() as usize;
            let result_index = lhs_index ^ rhs_index;
            let rhs_coeff = rhs.coeffs[rhs_index];
            let sign = f64::from(sign_row[rhs_index]);

            result[result_index].value += sign * (lhs_coeff.value * rhs_coeff.value);
            result[result_index].grad += sign
                * lhs_coeff
                    .grad
                    .mul_add(rhs_coeff.value, lhs_coeff.value * rhs_coeff.grad);

            rhs_mask &= rhs_mask - 1;
        }

        lhs_mask &= lhs_mask - 1;
    }

    let out = SparseDualVector::from_dense_buf(&result);
    if out.active_mask == 0 {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{geometric_product_dual, hodge_dual, hodge_undual, Dual, SparseDualVector};
    use crate::SparseCliffordVector;

    #[test]
    fn dual_layout_is_stable() {
        assert_eq!(std::mem::size_of::<Dual>(), 16);
        assert_eq!(std::mem::align_of::<Dual>(), 8);
    }

    #[test]
    fn dual_ad_x_squared_via_geometric_product() {
        let x0 = 3.5;
        let mut coeffs = [Dual {
            value: 0.0,
            grad: 0.0,
        }; 16];
        coeffs[0] = Dual {
            value: x0,
            grad: 1.0,
        };

        let x = SparseDualVector::from_dense_buf(&coeffs);
        let product = geometric_product_dual(&x, &x).expect("x*x should stay above Planck gate");

        assert_eq!(product.coeffs[0].value, x0 * x0);
        assert_eq!(product.coeffs[0].grad, 2.0 * x0);
    }

    #[test]
    fn from_sparse_clifford_vector_sets_zero_gradients() {
        let mv = SparseCliffordVector::from_iter([(0, 1.0), (3, -2.0), (7, 0.5)])
            .expect("finite sparse input should build");
        let dual: SparseDualVector = mv.into();

        for coeff in dual.coeffs {
            assert_eq!(coeff.grad, 0.0);
        }
    }

    #[test]
    fn hodge_double_dual_is_negation_in_g13() {
        let scalar = SparseCliffordVector::from_iter([(0, 1.0)]).unwrap();
        let double = hodge_undual(&hodge_dual(&scalar));
        assert!(
            (double.coeffs[0] + 1.0).abs() < 1e-12,
            "hodge_undual(hodge_dual(scalar)) should be -scalar, got {}",
            double.coeffs[0]
        );

        let e1 = SparseCliffordVector::from_iter([(0b0001, 1.0)]).unwrap();
        let double_e1 = hodge_undual(&hodge_dual(&e1));
        let orig = e1.coeffs[0b0001];
        let got = double_e1.coeffs[0b0001];
        assert!(
            (got + orig).abs() < 1e-12,
            "hodge_undual(hodge_dual(e1)) should be -e1, got {got}"
        );

        let biv = SparseCliffordVector::from_iter([(0b0011, 1.0)]).unwrap();
        let double_biv = hodge_undual(&hodge_dual(&biv));
        let orig_biv = biv.coeffs[0b0011];
        let got_biv = double_biv.coeffs[0b0011];
        assert!(
            (got_biv + orig_biv).abs() < 1e-12,
            "hodge_undual(hodge_dual(biv)) should be -biv, got {got_biv}"
        );
    }
}
