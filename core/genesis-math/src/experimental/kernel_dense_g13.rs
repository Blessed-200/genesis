//! Experimental dense geometric product kernel for fixed G(1,3).
//!
//! Goal: establish a lower-bound reference path for dense 16x16 inputs.
//! Arithmetic floor for dense case:
//! - 256 multiplications (all bilinear monomials a_i * b_j)
//! - 240 additions (16 outputs × (16 terms - 1))
//!
//! Decisión (no promover a producción):
//! - El truco branchless de aplicar signo por XOR sobre bit 63 del escalar es
//!   útil como referencia microarquitectónica y baseline escalar.
//! - En throughput sostenido, kernels AVX2/AVX-512 con permute/gather y FMA/ADD
//!   vectorial superan consistentemente este camino.
//! - Este archivo se conserva como referencia arquitectónica y oráculo de
//!   equivalencia, sin impacto en el dispatch productivo.

use crate::basis::TOTAL_BLADES;
use crate::product::CAYLEY_SIGN_F64;

/// Dense product result buffer for G(1,3).
pub type DenseBuf = [f64; TOTAL_BLADES];

/// Experimental dense kernel over full 16-blade vectors.
///
/// This function assumes all blades are active from a semantic perspective
/// (coefficients may still be zero-valued). It performs exactly 256 scalar
/// products and 240 accumulation additions.
#[inline]
#[allow(clippy::many_single_char_names)]
// Notación canónica GA: i = blade_a, j = blade_b, k = blade_resultado.
// Renombrar diverge de la literatura estándar (Hestenes 2003, §2.1).
pub fn dense_geometric_product_g13(a: &DenseBuf, b: &DenseBuf) -> DenseBuf {
    let mut out = [0.0; TOTAL_BLADES];

    // HOT PATH: O(16²), dense baseline kernel for G(1,3).
    // Use `i`-outer / `j`-inner traversal so both `CAYLEY_SIGN[i][j]` and `b[j]`
    // are consumed sequentially; only the `out[i ^ j]` scatter remains.
    for i in 0..TOTAL_BLADES {
        let a_i = a[i];
        let sign_row = &CAYLEY_SIGN_F64[i];
        for j in 0..TOTAL_BLADES {
            out[i ^ j] += a_i * b[j] * sign_row[j];
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sparse_geometric_product, SparseCliffordVector};

    fn dense_from_fn(mut f: impl FnMut(usize) -> f64) -> SparseCliffordVector {
        SparseCliffordVector::from_iter((0..16).map(|i| (i, f(i)))).unwrap()
    }

    #[test]
    fn dense_kernel_matches_sparse_reference_for_multiple_deterministic_inputs() {
        let cases = [
            (
                dense_from_fn(|i| (i as f64) - 2.0),
                dense_from_fn(|i| 0.5f64.mul_add(i as f64, 1.0)),
            ),
            (
                dense_from_fn(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) }),
                dense_from_fn(|i| ((i * i) as f64).mul_add(0.25, -3.0)),
            ),
            (
                dense_from_fn(|i| ((i as f64) - 7.5) / 3.0),
                dense_from_fn(|i| if i % 3 == 0 { 1.0 } else { -0.75 }),
            ),
            (
                dense_from_fn(|i| ((i as f64) + 1.0).recip()),
                dense_from_fn(|i| (i as f64 - 8.0) * 0.125),
            ),
        ];

        for (a, b) in cases {
            let exp = dense_geometric_product_g13(&a.coeffs, &b.coeffs);
            let reference = sparse_geometric_product(&a, &b).unwrap();

            for (lhs, rhs) in exp.iter().zip(reference.coeffs.iter()) {
                assert!((lhs - rhs).abs() < 1e-12);
            }
        }
    }
}
