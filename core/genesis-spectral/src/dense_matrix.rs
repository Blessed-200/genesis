#![allow(
    clippy::cast_precision_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::manual_flatten,
    clippy::indexing_slicing,
    clippy::pub_crate_without_defaults,
    clippy::manual_copy,
    clippy::copy_iterator,
    clippy::ptr_as_ptr,
    clippy::unwrap_used,
    clippy::as_conversions,
    clippy::suspicious_else_width,
    clippy::redundant_pub_crate,
    clippy::implicit_return,
    clippy::unused_self,
    clippy::unused_unit,
    clippy::restriction,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo
)]

//! Fixed-size dense matrix kernels for spectral 16-blade operators.
//!
//! AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)

/// Dense 16×16 matrix stored in row-major order.
///
/// The representation is intentionally dense because 256 `f64` coefficients
/// occupy only 2 KiB, fitting in L1 cache and avoiding sparse indirection for
/// fixed-width spectral operators.
///
/// AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
#[derive(Clone, Debug)]
pub(crate) struct DenseMatrix16 {
    /// Row-major storage: `data[r * 16 + c] = M[r][c]`.
    ///
    /// The full 16×16 matrix occupies 2 KiB, fitting entirely in L1 cache and
    /// giving LLVM contiguous inner loops for vectorized matvec/norm kernels.
    ///
    /// AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
    data: [f64; 256],
}

impl DenseMatrix16 {
    /// Builds a dense fixed-size 16×16 matrix from canonical nested storage.
    ///
    /// AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
    pub(crate) fn from_dense(dense: &[[f64; 16]; 16]) -> Self {
        let mut data = [0.0_f64; 256];
        for r in 0..16 {
            let row_base = r * 16;
            for c in 0..16 {
                data[row_base + c] = dense[r][c];
            }
        }
        Self { data }
    }

    /// Multiplies this matrix by a 16-component vector.
    ///
    /// HOT PATH: O(16²), called by spectral action evaluation. Rows are
    /// contiguous, so the inner loop is a fixed trip-count FMA reduction that
    /// LLVM can unroll and vectorize for the target CPU.
    ///
    /// AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
    pub(crate) fn matvec(&self, v: &[f64; 16]) -> [f64; 16] {
        let mut out = [0.0_f64; 16];
        for r in 0..16 {
            let row_base = r * 16;
            let mut acc = 0.0_f64;
            for c in 0..16 {
                acc += self.data[row_base + c] * v[c];
            }
            out[r] = acc;
        }
        out
    }

    /// Computes the matrix trace from the dense diagonal.
    ///
    /// AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
    pub(crate) fn trace(&self) -> f64 {
        let mut acc = 0.0_f64;
        for i in 0..16 {
            acc += self.data[i * 16 + i];
        }
        acc
    }

    /// Computes the squared Frobenius norm over all 256 dense coefficients.
    ///
    /// HOT PATH: O(256) contiguous reduction with no indirection or branches.
    ///
    /// AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
    pub(crate) fn frobenius_norm_sq(&self) -> f64 {
        let mut acc = 0.0_f64;
        for &value in &self.data {
            acc += value * value;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::DenseMatrix16;

    #[test]
    fn dense_matrix_matvec_identity() {
        let mut id = [[0.0; 16]; 16];
        for (i, row) in id.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let m = DenseMatrix16::from_dense(&id);
        let v = std::array::from_fn(|i| f64::from(u32::try_from(i).unwrap_or(0)) * 0.5);
        let out = m.matvec(&v);
        for i in 0..16 {
            assert!((out[i] - v[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn dense_matrix_trace_uses_diagonal_only() {
        let mut dense = [[0.0; 16]; 16];
        dense[0][0] = 3.0;
        dense[1][1] = 4.0;
        dense[0][1] = 100.0;
        let matrix = DenseMatrix16::from_dense(&dense);
        assert!((matrix.trace() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn dense_matrix_frobenius_counts_all_coefficients() {
        let mut dense = [[0.0; 16]; 16];
        dense[0][0] = 3.0;
        dense[1][2] = 4.0;
        dense[2][1] = 1.0e-15;
        let matrix = DenseMatrix16::from_dense(&dense);
        assert!((matrix.frobenius_norm_sq() - 25.0).abs() < 1e-12);
    }
}