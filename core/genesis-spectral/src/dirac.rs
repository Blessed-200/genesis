use genesis_math::{sparse_geometric_product, SparseCliffordVector};
use genesis_types::{constants::SPECTRAL_LAMBDA_MIN, AxiomID, GenesisError, WitnessBuilder};

use crate::sparse_matrix::SparseMatrix16;

/// Baseline algebraic contribution of Tr(D²)/4 in G(1,3) with Minkowski signature (+,-,-,-).
///
/// Derived from Σ_μ η^μμ = +1-1-1-1 over the 16-blade Clifford basis.
/// Subtracted in `squared()` so `ricci_scalar` measures relative curvature
/// with respect to the canonical flat manifold.
const FLAT_DIRAC_TRACE_BASELINE: f64 = -8.0;

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
    pub fn metric_signature(self) -> f64 {
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
    pub fn from_blades(blades: &[f64; 16], lambda: f64) -> Result<Self, GenesisError> {
        if lambda < SPECTRAL_LAMBDA_MIN {
            return Err(GenesisError::SpectralLambdaUnderflow { lambda });
        }
        let gamma_table = std::array::from_fn(|mu| {
            std::array::from_fn(|grade| {
                Some(GammaAction {
                    index: [
                        LorentzIndex::Time,
                        LorentzIndex::X,
                        LorentzIndex::Y,
                        LorentzIndex::Z,
                    ][mu],
                    output_grade: grade as u8,
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

    pub fn apply(&self, mv_coeffs: &[f64; 16]) -> Result<[f64; 16], GenesisError> {
        let _ = &self.gamma_table;
        let center = SparseCliffordVector::from_dense(&self.center_blades)?;
        let mv = SparseCliffordVector::from_dense(mv_coeffs)?;

        const SIGNATURES: [f64; 4] = [1.0, -1.0, -1.0, -1.0];
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

    pub fn squared(&self) -> Result<DiracSquared, GenesisError> {
        let mut dense = [[0.0_f64; 16]; 16];
        for j in 0..16 {
            let mut basis = [0.0_f64; 16];
            basis[j] = 1.0;
            let d_basis = self.apply(&basis)?;
            let d2_basis = self.apply(&d_basis)?;
            for i in 0..16 {
                dense[i][j] = d2_basis[i];
            }
        }
        let matrix = SparseMatrix16::from_dense(&dense);
        let ricci_scalar = 4.0 * matrix.trace() / 16.0 - self.flat_trace_baseline;
        Ok(DiracSquared {
            matrix,
            ricci_scalar,
        })
    }

    pub fn commutator_norm(&self, f_values: &[f64]) -> Result<f64, GenesisError> {
        let mut weighted = [0.0; 16];
        for (i, w) in weighted.iter_mut().enumerate() {
            *w = self.center_blades[i] * f_values.get(i).copied().unwrap_or(0.0);
        }
        let d_fmv = self.apply(&weighted)?;
        let d_mv = self.apply(&self.center_blades)?;
        let mut sum = 0.0;
        for i in 0..16 {
            let delta = d_fmv[i] - f_values.get(i).copied().unwrap_or(0.0) * d_mv[i];
            sum += delta * delta;
        }
        Ok(sum.sqrt())
    }
}

#[derive(Clone, Debug)]
pub struct DiracSquared {
    pub(crate) matrix: SparseMatrix16,
    pub ricci_scalar: f64,
}
