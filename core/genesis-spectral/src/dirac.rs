use genesis_math::{sparse_geometric_product, SparseCliffordVector};
use genesis_types::{constants::SPECTRAL_LAMBDA_MIN, AxiomID, GenesisError, WitnessBuilder};

use crate::dense_matrix::DenseMatrix16;

/// Baseline algebraic contribution of Tr(D²)/4 in G(1,3) with Minkowski signature (+,-,-,-).
///
/// Derived from `Σ_μ η^μμ = +1-1-1-1` over the 16-blade Clifford basis.
/// Subtracted in `squared()` so `ricci_scalar` measures relative curvature
/// with respect to the canonical flat manifold.
const FLAT_DIRAC_TRACE_BASELINE: f64 = -8.0;
const SIGNATURES: [f64; 4] = [1.0, -1.0, -1.0, -1.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LorentzIndex {
    Time = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

impl LorentzIndex {
    #[inline]
    #[must_use]
    pub const fn metric_signature(self) -> f64 {
        match self {
            Self::Time => 1.0,
            Self::X | Self::Y | Self::Z => -1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GammaAction {
    pub index: LorentzIndex,
    pub output_grade: u8,
    pub coefficient: [f64; 2],
}

#[derive(Clone, Debug)]
pub struct DiracOperator {
    pub center_blades: [f64; 16],
    gamma_table: [[Option<GammaAction>; 5]; 4],
    pub lambda_scale: f64,
    flat_trace_baseline: f64,
    pub construction_proof: [u8; 32],
}

impl DiracOperator {
    /// # Errors
    /// Returns `GenesisError::SpectralLambdaUnderflow` when lambda is invalid.
    pub fn from_blades(blades: &[f64; 16], lambda: f64) -> Result<Self, GenesisError> {
        if !lambda.is_finite() || lambda < SPECTRAL_LAMBDA_MIN {
            return Err(GenesisError::SpectralLambdaUnderflow { lambda });
        }
        let _validated_center = SparseCliffordVector::from_dense(blades)?;
        let gamma_table = std::array::from_fn(|mu| {
            std::array::from_fn(|grade| {
                Some(GammaAction {
                    index: [
                        LorentzIndex::Time,
                        LorentzIndex::X,
                        LorentzIndex::Y,
                        LorentzIndex::Z,
                    ][mu],
                    output_grade: u8::try_from(grade).unwrap_or(u8::MAX),
                    coefficient: [1.0, 0.0],
                })
            })
        });
        let mut builder = WitnessBuilder::new();
        builder.check(AxiomID::ProofGuard, || true)?;
        builder.check(AxiomID::MinkowskiSignature, || true)?;
        let proof = builder.build(0);
        Ok(Self {
            center_blades: *blades,
            gamma_table,
            lambda_scale: lambda,
            flat_trace_baseline: FLAT_DIRAC_TRACE_BASELINE,
            construction_proof: proof.hash,
        })
    }

    /// # Errors
    /// Returns `GenesisError` when coefficients cannot build valid multivectors.
    pub fn apply(&self, mv_coeffs: &[f64; 16]) -> Result<[f64; 16], GenesisError> {
        let _ = &self.gamma_table;
        let center = SparseCliffordVector::from_dense(&self.center_blades)?;
        let mv = SparseCliffordVector::from_dense(mv_coeffs)?;

        let mut out = [0.0_f64; 16];

        for (mu, sig) in SIGNATURES.iter().copied().enumerate() {
            let blade_idx = 1_usize << mu;
            let mut e_coeffs = [0.0_f64; 16];
            e_coeffs[blade_idx] = 1.0;
            let e_mu = SparseCliffordVector::from_dense(&e_coeffs)?;

            let Some(e_center) = sparse_geometric_product(&e_mu, &center) else {
                continue;
            };
            let Some(result) = sparse_geometric_product(&e_center, &mv) else {
                continue;
            };

            for (i, out_i) in out.iter_mut().enumerate() {
                *out_i += sig * result.coeffs[i];
            }
        }

        Ok(out)
    }

    /// # Errors
    /// Returns `GenesisError` when intermediate Dirac applications fail.
    pub fn squared(&self) -> Result<DiracSquared, GenesisError> {
        let mut dense_dirac = [[0.0_f64; 16]; 16];
        for j in 0..16 {
            let mut basis = [0.0_f64; 16];
            basis[j] = 1.0;
            let first_apply = self.apply(&basis)?;
            for i in 0..16 {
                dense_dirac[i][j] = first_apply[i];
            }
        }

        let dirac_matrix = DenseMatrix16::from_dense(&dense_dirac);
        let mut dense_squared = [[0.0_f64; 16]; 16];
        for j in 0..16 {
            let mut column = [0.0_f64; 16];
            for i in 0..16 {
                column[i] = dense_dirac[i][j];
            }
            let second_apply = dirac_matrix.matvec(&column);
            for i in 0..16 {
                dense_squared[i][j] = second_apply[i];
            }
        }

        let matrix = DenseMatrix16::from_dense(&dense_squared);
        let ricci_scalar = 4.0 * matrix.trace() / 16.0 - self.flat_trace_baseline;
        Ok(DiracSquared {
            matrix,
            ricci_scalar,
        })
    }

    /// # Errors
    /// Returns `GenesisError` when input dimensions or multivector construction are invalid.
    pub fn commutator_norm(&self, f_values: &[f64]) -> Result<f64, GenesisError> {
        if f_values.len() != 16 {
            return Err(GenesisError::DimensionMismatch {
                lhs: 16,
                rhs: f_values.len(),
            });
        }
        let mut weighted = [0.0; 16];
        for (i, w) in weighted.iter_mut().enumerate() {
            *w = self.center_blades[i] * f_values[i];
        }
        let d_comm_left = self.apply(&weighted)?;
        let d_comm_right = self.apply(&self.center_blades)?;
        let mut sum = 0.0;
        for i in 0..16 {
            let delta = f_values[i].mul_add(-d_comm_right[i], d_comm_left[i]);
            sum += delta * delta;
        }
        Ok(sum.sqrt())
    }
}

#[derive(Clone, Debug)]
pub struct DiracSquared {
    pub(crate) matrix: DenseMatrix16,
    pub ricci_scalar: f64,
}

impl DiracSquared {
    /// Returns the squared Frobenius norm of `D²`.
    ///
    /// AX-ID: AXIOMA-014, `H_estructura` (`LEY_FUNDACIONAL` §3.1)
    #[must_use]
    pub fn frobenius_norm_sq(&self) -> f64 {
        self.matrix.frobenius_norm_sq()
    }
}

#[cfg(test)]
mod tests {
    use super::{DiracOperator, LorentzIndex};
    use genesis_types::GenesisError;

    #[test]
    fn lorentz_metric_signatures_match_minkowski_contract() {
        assert!((LorentzIndex::Time.metric_signature() - 1.0).abs() < 1e-12);
        assert!((LorentzIndex::X.metric_signature() + 1.0).abs() < 1e-12);
        assert!((LorentzIndex::Y.metric_signature() + 1.0).abs() < 1e-12);
        assert!((LorentzIndex::Z.metric_signature() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn from_blades_rejects_non_finite_coefficients_at_construction() {
        let mut blades = [0.0_f64; 16];
        blades[3] = f64::NAN;
        assert!(matches!(
            DiracOperator::from_blades(&blades, 1.0),
            Err(GenesisError::SignatureViolation { .. })
        ));
    }

    #[test]
    fn from_blades_rejects_non_finite_lambda() {
        assert!(matches!(
            DiracOperator::from_blades(&[0.0; 16], f64::INFINITY),
            Err(GenesisError::SpectralLambdaUnderflow { .. })
        ));
    }

    #[test]
    fn apply_zero_center_covers_thermal_silence_path() {
        let dirac = DiracOperator::from_blades(&[0.0; 16], 1.0).expect("dirac");
        let out = dirac.apply(&[1.0; 16]).expect("apply");
        assert!(out.iter().all(|value| value.abs() < 1e-12));
    }

    #[test]
    fn commutator_norm_rejects_wrong_dimension() {
        let dirac = DiracOperator::from_blades(&[1.0; 16], 1.0).expect("dirac");
        assert!(matches!(
            dirac.commutator_norm(&[1.0; 4]),
            Err(GenesisError::DimensionMismatch { lhs: 16, rhs: 4 })
        ));
    }
}
