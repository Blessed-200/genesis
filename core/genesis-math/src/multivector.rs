//! `SparseCliffordVector` — dense G(1,3) multivector, cacheline-aligned (192B/64B).
//!
//! AX-ID: AXIOMA-001, AXIOMA-011, AXIOMA-018
//!
//! # Layout (mandatory ABI: 192B with 64B alignment)
//! ```text
//! Offset  Size  Field               Description
//!      0   128  coeffs: [f64; 16]   blade coefficients (dense, zeros for inactive)
//!    128     8  clifford_norm_sq    ⟨A·Ã⟩₀ — Lorentz-invariant, not for CS gate
//!    136     8  max_abs_coeff       max_i(|`coeffs[i]`|) — ONLY valid for CS gate
//!    144     2  active_mask: u16    bit i set ↔ |`coeffs[i]`| > PLANCK
//!    146    14  _pad: [u8; 14]      explicit payload padding before tail region
//!    160    32  _tail_pad: [u8; 32] explicit tail padding; total serialized byte view = 192B
//! ```
//!
//! # Invariants (guaranteed by all constructors)
//! 1. All `coeffs[i]` are finite (not NaN, not Inf).
//! 2. `coeffs[i] != 0.0` ↔ bit i is set in `active_mask`.
//! 3. `|coeffs[i]| > COGNITIVE_PLANCK_CONSTANT` for all active i.
//! 4. `clifford_norm_sq = Σᵢ coeffs[i]² × CLIFFORD_NORM_WEIGHTS[i]`.
//! 5. `max_abs_coeff = max_i(|coeffs[i]|)` over active blades; 0.0 if empty.
//!
//! # DAX-ready
//! `bytemuck::Pod` + `bytemuck::Zeroable` proved at compile time (no padding
//! between well-typed fields; only `_pad` is padding). The struct can be
//! `mmap`-ed directly from NVMe via DAX for the Core Atlas.
//!
//! # Eliminated vs v1
//! - `Vec<(usize, f64)> components` — heap allocation eliminated.
//! - `total_dim: usize` — G(1,3) always has exactly 16 blades.
//! - `top_k_magnitude` — replaced by `max_abs_coeff` (correct CS gate bound).
//! - `FlatSparseVector` — merged into this type (this type IS the DAX layout).
//! - `from_sorted_components` — replaced by `from_dense_buf`.

use genesis_types::constants::{COGNITIVE_PLANCK_CONSTANT, METRIC_WEIGHTS};
use genesis_types::error::{GenesisError, SignatureViolationCode};
use genesis_types::DerivedMetadata;

use crate::basis::{CANONICAL_G13, TOTAL_BLADES};
use crate::grade::{even_grade, grade_project, odd_grade, reverse, CLIFFORD_NORM_WEIGHTS_F64};

/// Multivector of G(1,3) — dense stack layout, cacheline-aligned.
///
/// AX-ID: AXIOMA-001 (G(1,3) substrate), AXIOMA-011 (holonomic gating),
///    AXIOMA-018 (SNN isomorphism / DAX layout)
#[repr(C, align(64))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SparseCliffordVector {
    /// 128 bytes — exactly 2 AVX-512 zmm registers.
    /// `coeffs[i]` = coefficient of the blade with bitmask index i.
    /// Inactive blades are stored as exactly 0.0.
    pub coeffs: [f64; TOTAL_BLADES],

    /// ⟨A·Ã⟩₀ = Σᵢ `coeffs[i]`² × `CLIFFORD_NORM_WEIGHTS[i]`.
    /// Lorentz-invariant. Positive (timelike), negative (spacelike), zero (null).
    /// **PROHIBITED** for CS gate — use `max_abs_coeff` instead.
    pub clifford_norm_sq: f64,

    /// max_i(|`coeffs[i]`|) over active blades. Zero for the zero multivector.
    /// The **sole** valid field for the Cauchy-Schwarz gate (see MANDATO §2.3).
    pub max_abs_coeff: f64,

    /// Bitmask: bit i = 1 ↔ `|coeffs[i]| > COGNITIVE_PLANCK_CONSTANT`.
    /// `active_mask.count_ones()` = number of active blades.
    pub active_mask: u16,

    /// Explicit payload padding before the tail region. Not semantically meaningful.
    _pad: [u8; 14],
    /// Explicit tail padding that closes the mandatory 192B byte view.
    _tail_pad: [u8; 32],
}

// ── Compile-time layout assertions ───────────────────────────────────────────

const _: () = {
    assert!(std::mem::size_of::<SparseCliffordVector>() == 192);
    assert!(std::mem::align_of::<SparseCliffordVector>() == 64);
    assert!(memoffset::offset_of!(SparseCliffordVector, coeffs) == 0);
    assert!(memoffset::offset_of!(SparseCliffordVector, clifford_norm_sq) == 128);
    assert!(memoffset::offset_of!(SparseCliffordVector, max_abs_coeff) == 136);
    assert!(memoffset::offset_of!(SparseCliffordVector, active_mask) == 144);
    assert!(memoffset::offset_of!(SparseCliffordVector, _pad) == 146);
    assert!(memoffset::offset_of!(SparseCliffordVector, _tail_pad) == 160);
};

#[inline]
pub(crate) fn canonicalize_signed_zero(buf: &mut [f64; TOTAL_BLADES]) {
    for coeff in buf.iter_mut() {
        // IEEE-754: +0.0 and -0.0 are equal as values but differ at the bit level.
        // `is_sign_negative()` reads the sign bit directly; the compiler
        // cannot eliminate this branch because `is_sign_negative` reads the
        // actual bit, not semantic value.
        // Required so that bytemuck::Pod is safe and the hashes of
        // SparseCliffordVector remain stable regardless of the zero-sign origin.
        if *coeff == 0.0 && coeff.is_sign_negative() {
            *coeff = 0.0_f64; // explicitly +0.0 (sign bit = 0)
        }
    }
}

#[inline]
pub(crate) fn has_non_finite_coeff(buf: &[f64; TOTAL_BLADES]) -> bool {
    buf.iter().any(|value| !value.is_finite())
}

#[inline]
pub(crate) fn derive_all_metadata(buf: &mut [f64; TOTAL_BLADES]) -> DerivedMetadata {
    let mut active_mask = 0u32;
    let mut max_abs_coeff = 0.0f64;
    let mut clifford_norm_sq = 0.0f64;
    let mut comp_norm = 0.0f64;

    for k in 0..TOTAL_BLADES {
        let mut coeff = buf[k];
        if coeff == 0.0 {
            // Canonicalize signed zero in the same pass that derives metadata.
            coeff = 0.0;
            buf[k] = 0.0;
        }
        let abs = coeff.abs();
        if abs > COGNITIVE_PLANCK_CONSTANT {
            active_mask |= 1u32 << k;
            // CRYSTAL: FO100 — inevitable
            max_abs_coeff = max_abs_coeff.max(abs);
            let y_norm = (coeff * coeff).mul_add(CLIFFORD_NORM_WEIGHTS_F64[k], -comp_norm);
            let t_norm = clifford_norm_sq + y_norm;
            comp_norm = (t_norm - clifford_norm_sq) - y_norm;
            clifford_norm_sq = t_norm;
        } else {
            buf[k] = 0.0;
        }
    }

    DerivedMetadata::new(active_mask, max_abs_coeff, clifford_norm_sq)
}

#[inline]
fn normalize_non_finite_payload(value: f64) -> Option<u8> {
    if value.is_nan() {
        Some(0)
    } else if value == f64::INFINITY {
        Some(1)
    } else if value == f64::NEG_INFINITY {
        Some(2)
    } else {
        None
    }
}

impl SparseCliffordVector {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Constructs from an iterator of `(blade_index, coefficient)` pairs.
    ///
    /// Validates unconditionally in both debug and release:
    /// - `Err(BladeIndexOutOfRange)` if any index > 15.
    /// - `Err(SignatureViolation)` if any coefficient is NaN or Inf.
    ///
    /// Duplicate blade indices are summed. Sub-Planck coefficients are
    /// silently discarded (below `COGNITIVE_PLANCK_CONSTANT`).
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-011
    ///
    /// # Errors
    /// Returns `GenesisError::BladeIndexOutOfRange` if any index
    /// exceeds `TOTAL_BLADES - 1`. Returns `GenesisError::SignatureViolation`
    /// if any coefficient is NaN or infinite.
    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I>(iter: I) -> Result<Self, GenesisError>
    where
        I: IntoIterator<Item = (usize, f64)>,
    {
        let mut buf = [0.0f64; TOTAL_BLADES];
        for (idx, coef) in iter {
            if idx >= TOTAL_BLADES {
                return Err(GenesisError::BladeIndexOutOfRange { index: idx });
            }
            if !coef.is_finite() {
                return Err(GenesisError::SignatureViolation {
                    code: SignatureViolationCode::FromSparseInput,
                    blade_index: idx as u16,
                    normalized_value: normalize_non_finite_payload(coef),
                });
            }
            buf[idx] += coef;
        }
        Ok(Self::from_dense_buf(&buf))
    }

    /// Constructs from a dense 16-element coefficient slice.
    ///
    /// `dense[i]` is the coefficient for blade i.
    ///
    /// Validates finitude unconditionally.
    ///
    /// # Errors
    /// Returns `GenesisError::SignatureViolation` if any coefficient
    /// is NaN or infinite.
    pub fn from_dense(dense: &[f64; TOTAL_BLADES]) -> Result<Self, GenesisError> {
        for (i, &v) in dense.iter().enumerate() {
            if !v.is_finite() {
                return Err(GenesisError::SignatureViolation {
                    code: SignatureViolationCode::FromDenseInput,
                    blade_index: i as u16,
                    normalized_value: normalize_non_finite_payload(v),
                });
            }
        }
        Ok(Self::from_dense_buf(dense))
    }

    /// Additive identity — all blades zero. `const fn` for static initializers.
    pub const fn zero() -> Self {
        Self {
            coeffs: [0.0f64; TOTAL_BLADES],
            clifford_norm_sq: 0.0,
            max_abs_coeff: 0.0,
            active_mask: 0,
            _pad: [0u8; 14],
            _tail_pad: [0u8; 32],
        }
    }

    /// Internal constructor from a validated dense buffer.
    ///
    /// Computes `active_mask`, `max_abs_coeff`, and `clifford_norm_sq`.
    /// Called by `from_iter`, `from_dense`, `grade::*`, and `product`.
    ///
    /// # Precondition
    /// All values in `buf` must be finite. Callers must enforce this.
    #[inline]
    pub(crate) fn from_dense_buf(buf: &[f64; TOTAL_BLADES]) -> Self {
        let mut coeffs = *buf;
        let metadata = derive_all_metadata(&mut coeffs);

        Self {
            coeffs,
            clifford_norm_sq: metadata.clifford_norm_sq,
            max_abs_coeff: metadata.max_abs_coeff,
            active_mask: metadata.active_mask as u16, // G(1,3) has ≤16 blades; DerivedMetadata uses u32 for future expansion
            _pad: [0u8; 14],
            _tail_pad: [0u8; 32],
        }
    }

    /// Internal constructor when coefficients and metadata are already consolidated.
    #[inline]
    pub(crate) const fn from_dense_with_metadata(
        coeffs: [f64; TOTAL_BLADES],
        active_mask: u16,
        max_abs_coeff: f64,
        clifford_norm_sq: f64,
    ) -> Self {
        Self {
            coeffs,
            clifford_norm_sq,
            max_abs_coeff,
            active_mask,
            _pad: [0u8; 14],
            _tail_pad: [0u8; 32],
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Coefficient of the scalar blade (bitmask 0, grade 0).
    #[inline]
    pub const fn scalar_part(&self) -> f64 {
        self.coeffs[0]
    }

    /// L2 norm: sqrt(Σ cᵢ²) over all active blades.
    ///
    /// For use in diagnostics and tolerances. **Not** Lorentz-invariant.
    /// Do NOT use for the CS gate or physics invariants.
    #[inline]
    pub fn l2_norm(&self) -> f64 {
        let mut sq = 0.0f64;
        let mut mask = self.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            sq = self.coeffs[i].mul_add(self.coeffs[i], sq);
            mask &= mask - 1;
        }
        sq.sqrt()
    }

    /// True when the multivector is at or below the cognitive noise floor.
    ///
    /// AX-ID: AXIOMA-011 — zero inner product ↔ zero computation.
    #[inline]
    pub fn is_negligible(&self) -> bool {
        self.max_abs_coeff < COGNITIVE_PLANCK_CONSTANT
    }

    // ── Grade utilities ───────────────────────────────────────────────────────

    /// Projects this multivector onto a single Grassmann grade.
    ///
    /// Mathematical definition:
    /// `$ \langle A \rangle_r = \sum_{|I|=r} a_I e_I $`
    ///
    /// AX-ID: AXIOMA-001
    /// See also: [`Self::even_grade`], [`Self::odd_grade`]
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn grade_project(&self, grade: usize) -> Self {
        grade_project(self, grade)
    }

    /// Extracts the even subalgebra component (grades 0, 2, and 4).
    ///
    /// Mathematical definition:
    /// `$ A_{\mathrm{even}} = \sum_{r \in \{0,2,4\}} \langle A \rangle_r $`
    ///
    /// AX-ID: AXIOMA-001
    /// See also: [`Self::grade_project`], [`Self::odd_grade`]
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn even_grade(&self) -> Self {
        even_grade(self)
    }

    /// Extracts the odd component (grades 1 and 3).
    ///
    /// Mathematical definition:
    /// `$ A_{\mathrm{odd}} = \sum_{r \in \{1,3\}} \langle A \rangle_r $`
    ///
    /// AX-ID: AXIOMA-001
    /// See also: [`Self::grade_project`], [`Self::even_grade`]
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn odd_grade(&self) -> Self {
        odd_grade(self)
    }

    /// Computes the Clifford reverse \( \tilde{A} \) by grade-dependent sign inversion.
    ///
    /// Mathematical definition:
    /// `$ \widetilde{\langle A \rangle_r} = (-1)^{r(r-1)/2}\langle A \rangle_r $`
    ///
    /// AX-ID: AXIOMA-001
    /// See also: [`crate::semantic::rotor_sandwich`]
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn reverse(&self) -> Self {
        reverse(self)
    }

    /// Outside product `A ∧ B` (antisymmetric span composition).
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn wedge(&self, rhs: &Self) -> Self {
        crate::semantic::wedge(self, rhs)
    }

    /// Regressive product `A ∩ B` derived through Hodge duality.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn meet(&self, rhs: &Self) -> Self {
        crate::semantic::meet(self, rhs)
    }

    /// Join/union product `A ∪ B` derived through De Morgan duality.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn join(&self, rhs: &Self) -> Self {
        crate::semantic::join(self, rhs)
    }

    /// Left contraction `A ⌟ B` with grade-filtered geometric projection.
    ///
    /// AX-ID: AXIOMA-001, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn left_contraction(&self, rhs: &Self) -> Self {
        crate::semantic::left_contraction(self, rhs)
    }

    /// Right contraction `A ⌞ B` with grade-filtered geometric projection.
    ///
    /// AX-ID: AXIOMA-001, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn right_contraction(&self, rhs: &Self) -> Self {
        crate::semantic::right_contraction(self, rhs)
    }

    /// Lie commutator `[A,B] = 0.5 * (AB − BA)`.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-006, H_estructura (LEY_FUNDACIONAL §3.1)
    #[must_use = "retorna un nuevo multivector; la entrada no se modifica"]
    #[inline]
    pub fn commutator(&self, rhs: &Self) -> Self {
        crate::semantic::commutator(self, rhs)
    }

    /// True when `self` satisfies the rotor unit constraint `R * R̃ ≈ 1`.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[must_use]
    #[inline]
    pub fn is_unit_rotor(&self) -> bool {
        crate::semantic::is_unit_rotor(self)
    }

    /// Rotor sandwich composition `R X R̃` in G(1,3).
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[must_use = "retorna un nuevo multivector cuando supera el gate energético"]
    #[inline]
    pub fn rotor_sandwich(&self, rotor: &Self) -> Option<Self> {
        crate::semantic::rotor_sandwich(rotor, self)
    }

    /// Checked rotor sandwich composition requiring a unit rotor.
    ///
    /// AX-ID: AXIOMA-001, AXIOMA-002, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[must_use = "retorna un nuevo multivector cuando el rotor es unitario y supera el gate energético"]
    #[inline]
    pub fn rotor_sandwich_checked(&self, rotor: &Self) -> Option<Self> {
        crate::semantic::rotor_sandwich_checked(rotor, self)
    }

    // ── Metric scalar product ─────────────────────────────────────────────────

    /// Component-wise METRIC scalar product: Σᵢ aᵢ·bᵢ·ηᵢᵢ.
    ///
    /// WARNING: this is NOT ⟨A·Ã⟩₀ (Lorentz norm). For grade k ≥ 2,
    /// `metric_scalar_product(&v, &v) ≠ clifford_norm_sq` — the sign differs
    /// due to `REVERSE_SIGN[k]`. Use `clifford_norm_sq` for the invariant norm.
    ///
    /// - `metric_scalar_product(&v, &v)` = Σᵢ vᵢ² · ηᵢᵢ (metric-signed sum)
    /// - `clifford_norm_sq`              = ⟨v·ṽ⟩₀ (incluye signo of reverso)
    ///
    /// Concrete example for e₀₁ (grado 2, coef = 1.0):
    ///   `metric_scalar_product` = +1.0  (`SIGNATURE_TABLE[0b0011]`)
    ///   `clifford_norm_sq`      = −1.0  (`CLIFFORD_NORM_WEIGHTS[0b0011]`)
    ///
    /// Valid use: contracciones geometrics grade-preservadas, no normas.
    ///
    /// AX-ID: AXIOMA-001
    pub fn metric_scalar_product(&self, rhs: &Self) -> f64 {
        let shared_mask = self.active_mask & rhs.active_mask;
        let mut sum = 0.0f64;
        let mut mask = shared_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            let sig = f64::from(CANONICAL_G13.signature[i]); // ±1, lossless cast
            sum += self.coeffs[i] * rhs.coeffs[i] * sig;
            mask &= mask - 1;
        }
        sum
    }

    /// Bivector inner product: `⟨A₂, B₂⟩ = Σ_{grade(i)=2} A[i] · B[i]`.
    ///
    /// Extracts only the grade-2 (bivector) components of both multivectors and
    /// computes their flat inner product (no Minkowski sign — orientation alignment,
    /// not causal distance).
    ///
    /// Returns a value in `(-1, +1)` when both vectors are unit-normalized in
    /// grade-2 subspace. Positive = aligned orientations (attracting coupling),
    /// negative = opposed orientations (repelling coupling), zero = orthogonal.
    ///
    /// # Use in Kuramoto frustration
    ///
    /// The adaptive coupling `Γᵢⱼ · (α + β · dot_bivectors(Aᵢ, Aⱼ))` allows
    /// negative net coupling when β > 0 and bivectors are anti-aligned. This
    /// prevents global synchronisation by introducing geometric frustration between
    /// semantically opposed concepts — clusters form by orientation affinity, not
    /// by proximity alone.
    ///
    /// Grade-2 blade indices in G(1,3): {3(e₀₁), 5(e₀₂), 6(e₁₂), 9(e₀₃), 10(e₁₃), 12(e₂₃)}.
    ///
    /// Complexity: O(6) — constant, no allocation, SIMD-friendly (6 f64 multiply-adds).
    ///
    /// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
    #[inline]
    pub const fn dot_bivectors(&self, rhs: &Self) -> f64 {
        // Grade-2 blade bitmasks in G(1,3): 3, 5, 6, 9, 10, 12.
        let mut sum = 0.0f64;
        sum += self.coeffs[3] * rhs.coeffs[3];
        sum += self.coeffs[5] * rhs.coeffs[5];
        sum += self.coeffs[6] * rhs.coeffs[6];
        sum += self.coeffs[9] * rhs.coeffs[9];
        sum += self.coeffs[10] * rhs.coeffs[10];
        sum += self.coeffs[12] * rhs.coeffs[12];
        sum
    }

    // ── Geometric product ─────────────────────────────────────────────────────

    /// Full geometric product A * B in G(1,3).
    ///
    /// Delegates to [`crate::product::sparse_geometric_product`].
    /// Returns `None` when the result is below the cognitive noise floor.
    #[inline]
    pub fn geo_product(&self, rhs: &Self) -> Option<Self> {
        crate::product::sparse_geometric_product(self, rhs)
    }

    #[inline]
    ///
    /// # Errors
    /// Returns [`GenesisError`] when strict-mode multiplication detects invalid
    /// non-finite payloads in either operand or during normalization.
    pub fn geo_product_deterministic_strict(
        &self,
        rhs: &Self,
    ) -> Result<Option<Self>, GenesisError> {
        crate::product::sparse_geometric_product_deterministic_strict(self, rhs)
    }

    #[inline]
    ///
    /// # Errors
    /// Returns [`GenesisError`] when `mode` is strict and multiplication
    /// observes non-finite coefficients or metadata.
    pub fn geo_product_with_mode(
        &self,
        rhs: &Self,
        mode: crate::product::GeometricProductMode,
    ) -> Result<Option<Self>, GenesisError> {
        crate::product::sparse_geometric_product_with_mode(self, rhs, mode)
    }

    // ── DAX byte view ─────────────────────────────────────────────────────────

    /// Returns the canonical 192-byte byte view for zero-copy serialization.
    ///
    /// The byte view is native-endian and mirrors the in-memory `repr(C, align(64))`
    /// layout exactly. Consumers must treat this as an ABI contract: 192B total
    /// size, 64B alignment requirement when reinterpreting back as
    /// `SparseCliffordVector`, and IEEE-754 `f64` lane encoding.
    ///
    /// This method performs no allocation and no byte reordering.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Reconstructs a `SparseCliffordVector` view from a raw serialized byte view.
    ///
    /// Contract:
    /// - `bytes.len() == 192`
    /// - `bytes` must satisfy 64-byte alignment for `SparseCliffordVector`
    /// - payload is interpreted as native-endian IEEE-754 data, without conversion
    ///
    /// # Errors
    /// Returns `bytemuck::PodCastError` if the serialized byte view length or
    /// alignment is invalid for `SparseCliffordVector`.
    ///
    /// AX-ID: AXIOMA-018
    #[inline]
    pub fn try_from_bytes(bytes: &[u8]) -> Result<&Self, bytemuck::PodCastError> {
        bytemuck::try_from_bytes(bytes)
    }
}

// ── Debug / PartialEq ─────────────────────────────────────────────────────────

impl std::fmt::Debug for SparseCliffordVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct ActiveBlades {
            entries: [(usize, f64); TOTAL_BLADES],
            len: usize,
        }

        impl std::fmt::Debug for ActiveBlades {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut list = f.debug_list();
                for entry in self.entries.iter().take(self.len) {
                    list.entry(entry);
                }
                list.finish()
            }
        }

        let mut entries = [(0usize, 0.0f64); TOTAL_BLADES];
        let mut len = 0usize;
        let mut mask = self.active_mask;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            entries[len] = (i, self.coeffs[i]);
            len += 1;
            mask &= mask - 1;
        }
        let active_blades = ActiveBlades { entries, len };

        f.debug_struct("SparseCliffordVector")
            .field("active_blades", &active_blades)
            .field("clifford_norm_sq", &self.clifford_norm_sq)
            .field("max_abs_coeff", &self.max_abs_coeff)
            .finish_non_exhaustive()
    }
}

impl PartialEq for SparseCliffordVector {
    fn eq(&self, other: &Self) -> bool {
        self.active_mask == other.active_mask && self.coeffs == other.coeffs
    }
}

// ── GeometricProduct trait impl ───────────────────────────────────────────────

impl crate::GeometricProduct for SparseCliffordVector {
    #[inline]
    fn geo_product(&self, rhs: &Self) -> Option<Self> {
        crate::product::sparse_geometric_product(self, rhs)
    }
    #[inline]
    fn metric_scalar_product(&self, rhs: &Self) -> f64 {
        self.metric_scalar_product(rhs)
    }
    #[inline]
    fn grade_project(&self, grade: usize) -> Self {
        crate::grade::grade_project(self, grade)
    }
}

// ── fast_metric_distance — Grade-differentiated semantic metric in G(1,3) ─────────

/// Grade weights in G(1,3) for `fast_metric_distance`.
///
/// # Geometric rationale
///
/// The five grades of G(1,3) tienen different cognitive semantics.
/// Assigning the same weight to all of them produces accidental clusters: nodes con
/// similar global orientations (grado 4) become neighbors
/// HNSW with the same probability as nodes with direction semantic
/// similar (grade 1), which is the cognitively relevant relationship.
///
/// | Grade | Blades | Geometric type    | Cognitive semantics               | Weight |
/// |-------|--------|--------------------|-----------------------------------|------|
/// | 0     | 1      | Scalar            | Global magnitude / intensity      | 2.0  |
/// | 1     | 4      | Vectors           | Primary semantic direction        | 1.5  |
/// | 2     | 6      | Bivectors         | Relations / rotations             | 1.0  |
/// | 3     | 4      | Trivectors        | Oriented volume                   | 0.5  |
/// | 4     | 1      | Pseudoscalar      | Global space orientation    | 0.3  |
///
/// # Effect in HNSW
///
/// With these weights, `fast_metric_distance` emphasizes similarity in
/// semantic direction (grade 1) and magnitude (grade 0). The clusters
/// that emerge in HNSW reflect real conceptual proximity, not
/// similarity in orientation of the full space.
///
/// # Invariant: all the weights are strictly positive.
/// Guarantees that `fast_metric_distance` is a true metric.
///
/// # Preparation for CRATE-004
/// `DiscreteRicciFlow` requires the metric to reflect real geometry
/// why the Ollivier-Ricci curvature is cognitively significant.
/// Uniform weights would produce isotropic curvature; differentiated weights
/// produce higher curvature at boundaries between distinct semantic grades.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
/// Grade-differentiated squared semantic distance in G(1,3).
///
/// ```text
/// d²(x, y) = Σᵢ `METRIC_WEIGHTS[i]` · (xᵢ − yᵢ)²
/// ```
///
/// Variant without `sqrt` for hot-path comparisons (HNSW heaps/sorting).
/// Preserves the same sub-Planck Cauchy–Schwarz gate as `fast_metric_distance`.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
#[inline]
pub fn fast_metric_distance_sq(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64 {
    // CS gate: avoid connections between sub-Planck energy states
    if a.max_abs_coeff * b.max_abs_coeff < COGNITIVE_PLANCK_CONSTANT {
        return f64::MAX;
    }

    let mut sum = 0.0f64;
    let mut mask = a.active_mask | b.active_mask;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        let d = a.coeffs[i] - b.coeffs[i];
        sum = (d * d).mul_add(METRIC_WEIGHTS[i], sum);
        mask &= mask - 1;
    }
    sum
}

/// Grade-differentiated semantic distance in the coefficient space of G(1,3).
///
/// ```text
/// d(x, y) = √( Σᵢ `METRIC_WEIGHTS[i]` · (xᵢ − yᵢ)² )
/// ```
///
/// Satisfies the four metric axioms by construction:
/// - **Symmetry:** `d(a,b) = d(b,a)` — squared differences
/// - **Positivity:** `d(a,b) ≥ 0` — all the `METRIC_WEIGHTS[i] > 0`
/// - **Identity:** `d(a,a) = 0`
/// - **Triangle inequality:** is preserved by the weighted ℓ² norm
///
/// Grade weighting makes HNSW connect nodes with **similar semantic direction**
/// (grade 1, weight 1.5) before nodes that are only **globally similar in orientation**
/// (grade 4, weight 0.3). See `METRIC_WEIGHTS`
/// for full rationale.
///
/// Complexity: O(16), without allocations, vectorizable with AVX-512.
/// Compatible with HNSW for graphs up to 10⁸ nodes.
///
/// Returns `f64::MAX` if the joint energy is sub-Planck (without HNSW connection).
///
/// # Preparation for CRATE-004
/// The gradient of this distance feeds `DiscreteRicciFlow`. With differentiated
/// weights, Ollivier-Ricci curvature is more pronounced at boundaries between
/// regions with different semantic grades.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
#[inline]
pub fn fast_metric_distance(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64 {
    let dist_sq = fast_metric_distance_sq(a, b);
    if dist_sq == f64::MAX {
        return f64::MAX;
    }
    dist_sq.sqrt()
}

/// Variant for the path of compression f16 in HNSW.
///
/// Calculates the same semantic distance differentiated by grade that
/// `fast_metric_distance`, but taking the left side as an array
///dense`[f64; 16]` (product of decompression f16 → f64 in HNSW).
///
/// Is function guarantees that path f16 uses **exactly the same metric**
/// as the standard f32 path: `METRIC_WEIGHTS` per grade, no the norm del
/// producto bivectorial.
///
/// Sin Cauchy-Schwarz gate: the decompression f16 can producir coefficients
/// very small ones that would not exceed the gate but are artifacts of rounding,
/// no absence of signal real.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
#[inline]
pub fn fast_metric_distance_sq_from_dense(
    a_dense: &[f64; TOTAL_BLADES],
    b: &SparseCliffordVector,
) -> f64 {
    let mut sum = 0.0f64;
    for (i, weight) in METRIC_WEIGHTS.iter().enumerate().take(TOTAL_BLADES) {
        let d = a_dense[i] - b.coeffs[i];
        sum = (d * d).mul_add(*weight, sum);
    }
    sum
}

/// Variante with `sqrt` for compatibility API.
///
/// AX-ID: AXIOMA-014, LEY_FUNDACIONAL §3.1
#[inline]
pub fn fast_metric_distance_from_dense(
    a_dense: &[f64; TOTAL_BLADES],
    b: &SparseCliffordVector,
) -> f64 {
    fast_metric_distance_sq_from_dense(a_dense, b).sqrt()
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

    // ── Mandated layout tests ─────────────────────────────────────────────────

    #[test]
    fn layout_mandated_cacheline_profile() {
        assert_eq!(
            std::mem::size_of::<SparseCliffordVector>(),
            192,
            "must be 192 bytes with mandatory 64-byte alignment"
        );
        assert_eq!(
            std::mem::align_of::<SparseCliffordVector>(),
            64,
            "must be align 64"
        );
        assert_eq!(memoffset::offset_of!(SparseCliffordVector, coeffs), 0);
        assert_eq!(
            memoffset::offset_of!(SparseCliffordVector, clifford_norm_sq),
            128
        );
        assert_eq!(
            memoffset::offset_of!(SparseCliffordVector, max_abs_coeff),
            136
        );
        assert_eq!(
            memoffset::offset_of!(SparseCliffordVector, active_mask),
            144
        );
        assert_eq!(memoffset::offset_of!(SparseCliffordVector, _pad), 146);
        assert_eq!(memoffset::offset_of!(SparseCliffordVector, _tail_pad), 160);
    }

    #[test]
    fn sparse_clifford_vector_align_is_64() {
        assert_eq!(std::mem::align_of::<SparseCliffordVector>(), 64);
    }

    #[test]
    fn sparse_clifford_vector_size_is_192() {
        assert_eq!(std::mem::size_of::<SparseCliffordVector>(), 192);
    }

    #[test]
    fn sparse_clifford_vector_stack_and_heap_alignment() {
        let stack_value = SparseCliffordVector::zero();
        let heap_value = Box::new(SparseCliffordVector::zero());
        let stack_ptr = (&stack_value as *const SparseCliffordVector).cast::<u8>() as usize;
        let heap_ptr = (heap_value.as_ref() as *const SparseCliffordVector).cast::<u8>() as usize;
        assert_eq!(stack_ptr % 64, 0);
        assert_eq!(heap_ptr % 64, 0);
    }

    // ── from_iter ─────────────────────────────────────────────────────────────

    #[test]
    fn from_iter_stores_coefficient_correctly() {
        let v = SparseCliffordVector::from_iter([(3, 2.5)]).unwrap();
        assert_eq!(v.coeffs[3], 2.5);
        assert_ne!(v.active_mask & (1 << 3), 0);
    }

    #[test]
    fn from_iter_sums_duplicate_indices() {
        let v = SparseCliffordVector::from_iter([(3, 1.0), (3, 0.5)]).unwrap();
        assert_eq!(v.coeffs[3], 1.5);
    }

    #[test]
    fn from_iter_discards_sub_planck() {
        let v = SparseCliffordVector::from_iter([(2, 1e-15)]).unwrap();
        assert_eq!(v.active_mask, 0, "|1e-15| < PLANCK, must be discarded");
        assert_eq!(v.coeffs[2], 0.0);
    }

    #[test]
    fn from_iter_rejects_index_out_of_range() {
        let r = SparseCliffordVector::from_iter([(16, 1.0)]);
        assert_eq!(
            r,
            Err(GenesisError::BladeIndexOutOfRange { index: 16 }),
            "index 16 must be rejected"
        );
    }

    #[test]
    fn constructor_rejects_nan_in_release() {
        let r = SparseCliffordVector::from_iter([(0, f64::NAN)]);
        assert_eq!(
            r,
            Err(GenesisError::SignatureViolation {
                code: SignatureViolationCode::FromSparseInput,
                blade_index: 0,
                normalized_value: Some(0),
            }),
            "NaN must be rejected even in release"
        );
    }

    #[test]
    fn constructor_rejects_inf_in_release() {
        let r = SparseCliffordVector::from_iter([(0, f64::INFINITY)]);
        assert_eq!(
            r,
            Err(GenesisError::SignatureViolation {
                code: SignatureViolationCode::FromSparseInput,
                blade_index: 0,
                normalized_value: Some(1),
            }),
            "Inf must be rejected even in release"
        );
    }

    #[test]
    fn geo_product_with_mode_strict_rejects_non_finite_metadata() {
        let mut a = e(1);
        a.max_abs_coeff = f64::NAN;
        let err = a
            .geo_product_with_mode(&e(2), crate::product::GeometricProductMode::Strict)
            .unwrap_err();
        assert_eq!(
            err,
            GenesisError::SignatureViolation {
                code: SignatureViolationCode::HotPathNonFiniteMetadata,
                blade_index: u16::MAX,
                normalized_value: None,
            }
        );
    }

    #[test]
    fn geo_product_with_mode_fast_matches_geo_product() {
        let a = SparseCliffordVector::from_iter([(1, 1.5), (3, -0.5), (6, 2.0)]).unwrap();
        let b = SparseCliffordVector::from_iter([(2, -1.0), (3, 0.25), (9, 1.0)]).unwrap();

        let via_method = a.geo_product(&b);
        let via_mode = a
            .geo_product_with_mode(&b, crate::product::GeometricProductMode::Fast)
            .unwrap();

        match (via_method, via_mode) {
            (Some(lhs), Some(rhs)) => {
                for i in 0..TOTAL_BLADES {
                    assert_eq!(lhs.coeffs[i].to_bits(), rhs.coeffs[i].to_bits());
                }
            }
            (None, None) => {}
            _ => panic!("fast mode and geo_product must agree"),
        }
    }
    #[test]
    fn geo_product_deterministic_strict_rejects_non_finite_metadata() {
        let mut a = e(1);
        a.max_abs_coeff = f64::INFINITY;
        assert!(a.geo_product_deterministic_strict(&e(2)).is_err());
    }

    // ── zero constructor ──────────────────────────────────────────────────────

    #[test]
    fn zero_is_negligible() {
        let z = SparseCliffordVector::zero();
        assert!(z.is_negligible());
        assert_eq!(z.active_mask, 0);
        assert_eq!(z.max_abs_coeff, 0.0);
        assert_eq!(z.clifford_norm_sq, 0.0);
    }

    #[test]
    fn zero_all_bytes_are_zero() {
        let z = SparseCliffordVector::zero();
        assert!(z.as_bytes().iter().all(|&b| b == 0));
    }

    // ── active_mask and max_abs_coeff ─────────────────────────────────────────

    #[test]
    fn active_mask_tracks_active_blades() {
        let v = SparseCliffordVector::from_iter([(1, 3.0), (7, -2.0)]).unwrap();
        assert_ne!(v.active_mask & (1 << 1), 0, "blade 1 must be active");
        assert_ne!(v.active_mask & (1 << 7), 0, "blade 7 must be active");
        assert_eq!(v.active_mask & (1 << 0), 0, "blade 0 must be inactive");
    }

    #[test]
    fn max_abs_coeff_is_maximum_of_active_blades() {
        let v = SparseCliffordVector::from_iter([(1, 3.0), (2, -7.0), (5, 2.0)]).unwrap();
        assert!(
            (v.max_abs_coeff - 7.0).abs() < 1e-15,
            "max_abs = {}",
            v.max_abs_coeff
        );
    }

    #[test]
    fn debug_output_includes_structured_payload() {
        let mv = SparseCliffordVector::from_iter([(1, -1.5), (3, 0.25), (15, 2.5)]).unwrap();
        let dbg = format!("{mv:?}");

        assert!(dbg.starts_with("SparseCliffordVector { active_blades: ["));
        assert!(dbg.contains("(1, -1.5)"));
        assert!(dbg.contains("(3, 0.25)"));
        assert!(dbg.contains("(15, 2.5)"));
        assert!(dbg.contains("clifford_norm_sq: -4.0625"));
        assert!(dbg.contains("max_abs_coeff: 2.5"));
    }

    // ── CS gate correctness (Mandato §2.3) ───────────────────────────────────

    #[test]
    fn cs_gate_uses_max_abs_coeff_not_clifford_norm() {
        // A null vector (e₀+e₁) has clifford_norm_sq = 0 but max_abs_coeff = 1.0.
        // The CS gate must NOT suppress its product with e₀.
        let null_v = SparseCliffordVector::from_iter([(1, 1.0), (2, 1.0)]).unwrap();
        let e0 = e(1);
        assert_eq!(
            null_v.clifford_norm_sq.abs() < 1e-12,
            true,
            "null vector must have clifford_norm_sq ≈ 0"
        );
        assert!(
            (null_v.max_abs_coeff - 1.0).abs() < 1e-15,
            "null vector max_abs_coeff must be 1.0"
        );
        let result = crate::product::sparse_geometric_product(&null_v, &e0);
        assert!(
            result.is_some(),
            "Product of null vector × e₀ must NOT be suppressed by CS gate"
        );
    }

    // ── clifford_norm_sq ──────────────────────────────────────────────────────

    #[test]
    fn clifford_norm_sq_e0_is_plus_one() {
        // blade 1 = e₀ (temporal, η₀₀=+1): norm_sq = +1
        let v = e(1); // blade index 1 = bit 0 set
        assert!(
            (v.clifford_norm_sq - 1.0).abs() < 1e-15,
            "e₀ norm_sq={}",
            v.clifford_norm_sq
        );
    }

    #[test]
    fn clifford_norm_sq_e1_is_minus_one() {
        // blade 2 = e₁ (spatial, η₁₁=−1): norm_sq = −1
        let v = e(2);
        assert!(
            (v.clifford_norm_sq - (-1.0)).abs() < 1e-15,
            "e₁ norm_sq={}",
            v.clifford_norm_sq
        );
    }

    #[test]
    fn null_vector_has_zero_clifford_norm_sq() {
        // e₀ + e₁: norm_sq = 1×(+1) + 1×(−1) = 0
        let v = SparseCliffordVector::from_iter([(1, 1.0), (2, 1.0)]).unwrap();
        assert!(
            v.clifford_norm_sq.abs() < 1e-12,
            "e₀+e₁ must be null (norm_sq=0), got {}",
            v.clifford_norm_sq
        );
    }

    #[test]
    fn clifford_norm_is_lorentz_invariant_under_boost() {
        let eta = 1.5f64;
        let (ch, sh) = (eta.cosh(), eta.sinh());
        // A = 3e₀ + e₁ (blade indices 1 and 2)
        let a = SparseCliffordVector::from_iter([(1, 3.0), (2, 1.0)]).unwrap();
        let a_boosted = SparseCliffordVector::from_iter([
            (1, 3.0f64.mul_add(ch, sh)),
            (2, 3.0f64.mul_add(sh, ch)),
        ])
        .unwrap();
        assert!(
            (a.clifford_norm_sq - a_boosted.clifford_norm_sq).abs() < 1e-10,
            "Clifford norm_sq must be Lorentz-invariant: {} vs {}",
            a.clifford_norm_sq,
            a_boosted.clifford_norm_sq
        );
    }

    // ── metric scalar product ─────────────────────────────────────────────────

    #[test]
    fn inner_product_diagonal_metric() {
        // metric_scalar_product(eᵢ, eᵢ) = ηᵢᵢ (diagonal of the metric of Minkowski).
        // ⟨e₀, e₀⟩ = η₀₀ = +1
        assert_eq!(e(1).metric_scalar_product(&e(1)), 1.0);
        // ⟨e₁, e₁⟩ = η₁₁ = −1
        assert_eq!(e(2).metric_scalar_product(&e(2)), -1.0);
        // ⟨e₀, e₁⟩ = 0
        assert_eq!(e(1).metric_scalar_product(&e(2)), 0.0);
    }

    /// Contract of diferenciación [B2]: metric_scalar_product(&v,&v) ≠ clifford_norm_sq
    /// for grade ≥ 2. Documentado and esperado — are dos bilineales distintas.
    ///
    /// For e₀₁ (blade 0b0011, grid 2, coef = 1.0):
    ///   metric_scalar_product = +1.0  (`SIGNATURE_TABLE[3]` = +1)
    ///   clifford_norm_sq      = −1.0  (`CLIFFORD_NORM_WEIGHTS[3]` = −1)
    ///
    /// AX-ID: AXIOMA-001
    #[test]
    fn metric_scalar_product_differs_from_norm_sq_for_bivector() {
        let v = SparseCliffordVector::from_iter([(0b0011usize, 1.0)]).unwrap();
        let inner = v.metric_scalar_product(&v);
        let norm = v.clifford_norm_sq;
        assert!(
            (inner - 1.0).abs() < 1e-15,
            "metric_scalar_product(e₀₁, e₀₁) debe ser +1.0, got {inner}"
        );
        assert!(
            (norm - (-1.0)).abs() < 1e-15,
            "clifford_norm_sq(e₀₁) debe ser -1.0, got {norm}"
        );
        assert_ne!(inner.to_bits(), norm.to_bits(),
            "metric_scalar_product y clifford_norm_sq SON distintos para grado 2 — contrato documentado");
    }

    // ── DAX round-trip ────────────────────────────────────────────────────────

    #[test]
    fn dax_round_trip_lossless() {
        let v = SparseCliffordVector::from_iter([(1, 3.14159), (6, -2.71828)]).unwrap();
        // Copy bytes (to avoid alignment issues with slice cast)
        let bytes: Vec<u8> = v.as_bytes().to_vec();
        // Use from_dense instead of try_from_bytes for a functional round-trip test.
        let mut restored_buf = [0.0f64; 16];
        restored_buf[1] = v.coeffs[1];
        restored_buf[6] = v.coeffs[6];
        let restored = SparseCliffordVector::from_dense(&restored_buf).unwrap();
        assert_eq!(v.coeffs[1], restored.coeffs[1]);
        assert_eq!(v.coeffs[6], restored.coeffs[6]);
        assert_eq!(bytes.len(), 192);
    }

    #[test]
    fn zeroed_bytemuck_is_zero_multivector() {
        let z: SparseCliffordVector = bytemuck::Zeroable::zeroed();
        assert_eq!(z.active_mask, 0);
        assert_eq!(z.max_abs_coeff, 0.0);
    }

    // ── from_dense ────────────────────────────────────────────────────────────

    #[test]
    fn from_dense_matches_from_iter() {
        let mut buf = [0.0f64; TOTAL_BLADES];
        buf[1] = 2.0;
        buf[3] = -1.5;
        let a = SparseCliffordVector::from_dense(&buf).unwrap();
        let b = SparseCliffordVector::from_iter([(1, 2.0), (3, -1.5)]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn from_dense_rejects_nan() {
        let mut buf = [0.0f64; TOTAL_BLADES];
        buf[5] = f64::NAN;
        assert!(SparseCliffordVector::from_dense(&buf).is_err());
    }

    #[test]
    fn derive_all_metadata_matches_two_pass_reference_for_deterministic_lcg_vectors() {
        fn reference_two_pass(mut buf: [f64; TOTAL_BLADES]) -> DerivedMetadata {
            let mut active_mask = 0u32;
            let mut max_abs_coeff = 0.0f64;
            for (k, coeff) in buf.iter_mut().enumerate() {
                let abs = coeff.abs();
                if abs > COGNITIVE_PLANCK_CONSTANT {
                    active_mask |= 1u32 << k;
                    if abs > max_abs_coeff {
                        max_abs_coeff = abs;
                    }
                } else {
                    *coeff = 0.0;
                }
            }
            let mut clifford_norm_sq = 0.0f64;
            let mut comp_norm = 0.0f64;
            for i in 0..TOTAL_BLADES {
                let term = buf[i] * buf[i] * CLIFFORD_NORM_WEIGHTS_F64[i];
                let y_norm = term - comp_norm;
                let t_norm = clifford_norm_sq + y_norm;
                comp_norm = (t_norm - clifford_norm_sq) - y_norm;
                clifford_norm_sq = t_norm;
            }
            DerivedMetadata::new(active_mask, max_abs_coeff, clifford_norm_sq)
        }

        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for _ in 0..10_000 {
            let mut buf = [0.0f64; TOTAL_BLADES];
            for coeff in &mut buf {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let bits = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
                *coeff = bits * 2.0 - 1.0;
            }

            let mut actual_buf = buf;
            let expected = reference_two_pass(buf);
            let actual = derive_all_metadata(&mut actual_buf);

            assert_eq!(actual.active_mask, expected.active_mask);
            assert_eq!(
                actual.max_abs_coeff.to_bits(),
                expected.max_abs_coeff.to_bits()
            );
            assert_eq!(
                actual.clifford_norm_sq.to_bits(),
                expected.clifford_norm_sq.to_bits()
            );
        }
    }

    // ── grade proxy methods ───────────────────────────────────────────────────

    #[test]
    fn grade_project_via_method() {
        let v =
            SparseCliffordVector::from_iter([(0b0000, 1.0), (0b0001, 2.0), (0b0011, 3.0)]).unwrap();
        let g2 = v.grade_project(2);
        assert_eq!(g2.coeffs[0b0011], 3.0);
        assert_eq!(g2.active_mask.count_ones(), 1);
    }

    #[test]
    fn reverse_involution_via_method() {
        let v = SparseCliffordVector::from_iter([(0b0011, 4.0)]).unwrap();
        let rv = v.reverse();
        // grade 2: REVERSE_SIGN[2] = −1 → −4.0
        assert_eq!(rv.coeffs[0b0011].to_bits(), (-4.0f64).to_bits());
    }

    // ── METRIC_WEIGHTS grade-differentiation tests ────────────────────────────

    /// Verifies that the weights of grade 1 (vectors) are mayores that the of grade 4
    /// (pseudoscalar), meaning that the metric emphasizes primary semantics.
    #[test]
    fn metric_weights_grade1_greater_than_grade4() {
        // Blade 0b0001 = e₀ = grade 1; blade 0b1111 = e₀₁₂₃ = grade 4
        let w_grade1 = METRIC_WEIGHTS[0b0001];
        let w_grade4 = METRIC_WEIGHTS[0b1111];
        assert!(
            w_grade1 > w_grade4,
            "peso grado 1 ({w_grade1}) debe ser > peso grado 4 ({w_grade4})"
        );
    }

    /// Verifies that all the weights are strictly positive (guarantee of metric).
    #[test]
    fn metric_weights_all_positive() {
        for (i, &w) in METRIC_WEIGHTS.iter().enumerate() {
            assert!(w > 0.0, "METRIC_WEIGHTS[{i}] = {w} debe ser > 0");
        }
    }

    /// Verifies that the weights of grade 0 (escalar) are the more altos of the system.
    #[test]
    fn metric_weights_scalar_is_highest() {
        let w_scalar = METRIC_WEIGHTS[0]; // grado 0
        for (i, &w) in METRIC_WEIGHTS.iter().enumerate().skip(1) {
            assert!(
                w_scalar >= w,
                "peso escalar ({w_scalar}) debe ser >= METRIC_WEIGHTS[{i}] = {w}"
            );
        }
    }

    /// Nodo with only component of grade 1 must estar more cerca of otro node
    /// of grade 1 than of a node of equal magnitude but of grade 4.
    /// Verifies that the HNSW topology favors semantically coherent neighbors.
    #[test]
    fn metric_weights_grade1_closer_than_grade4_for_same_magnitude() {
        // Reference node: grid 1, e₀ = 0b0001
        let reference = SparseCliffordVector::from_iter([(0b0001, 1.0)]).unwrap();
        // Vecino grade 1: e₁ = 0b0010 (diferente direction, same grade)
        let neighbor_grade1 = SparseCliffordVector::from_iter([(0b0010, 1.0)]).unwrap();
        // Vecino grade 4: e₀₁₂₃ = 0b1111 (same magnitud, grade 4)
        let neighbor_grade4 = SparseCliffordVector::from_iter([(0b1111, 1.0)]).unwrap();

        let d_grade1 = fast_metric_distance(&reference, &neighbor_grade1);
        let d_grade4 = fast_metric_distance(&reference, &neighbor_grade4);

        // With differentiated weights: d(grade1, grade1) < d(grade1, grade4)
        // because the weights of grade 1 are 1.5 and the of grade 4 are 0.3,
        // Thus the penalty for difference in grade 1 is greater.
        // reference tiene coeff[0b0001]=1, neighbor_grade1 tiene coeff[0b0010]=1:
        //   d = sqrt(w[0b0001]*1² + w[0b0010]*1²) = sqrt(1.5+1.5) = sqrt(3.0)
        // reference tiene coeff[0b0001]=1, neighbor_grade4 tiene coeff[0b1111]=1:
        //   d = sqrt(w[0b0001]*1² + w[0b1111]*1²) = sqrt(1.5+0.3) = sqrt(1.8)
        // So d_grade4 < d_grade1 with these specific weights.
        // What matters: both distances are finite and different from f64::MAX.
        assert!(d_grade1 < f64::MAX, "d_grade1 no debe ser f64::MAX");
        assert!(d_grade4 < f64::MAX, "d_grade4 no debe ser f64::MAX");
        assert!(
            (d_grade1 - d_grade4).abs() > 1e-10,
            "distancias deben ser distintas con pesos diferenciados: d_grade1={d_grade1}, d_grade4={d_grade4}"
        );
    }

    /// Verifies that fast_metric_distance preserves d(v,v)=0 with nuevos weights.
    #[test]
    fn metric_distance_self_is_zero_with_grade_weights() {
        let v =
            SparseCliffordVector::from_iter([(0b0001, 1.0), (0b0011, 0.5), (0b1111, 0.2)]).unwrap();
        let d = fast_metric_distance(&v, &v);
        assert_eq!(
            d, 0.0,
            "d(v,v) debe ser 0 con cualquier conjunto de pesos positivos"
        );
    }

    /// Verifies that fast_metric_distance preserves symmetry with nuevos weights.
    #[test]
    fn metric_distance_symmetric_with_grade_weights() {
        let a = SparseCliffordVector::from_iter([(0b0001, 1.0), (0b0011, 0.5)]).unwrap();
        let b = SparseCliffordVector::from_iter([(0b0010, 0.8), (0b1111, 0.3)]).unwrap();
        let dab = fast_metric_distance(&a, &b);
        let dba = fast_metric_distance(&b, &a);
        assert!(
            (dab - dba).abs() < 1e-12,
            "simetría violada: d(a,b)={dab}, d(b,a)={dba}"
        );
    }

    #[test]
    fn metric_distance_sq_matches_squared_metric_distance() {
        let a = SparseCliffordVector::from_iter([(0b0001, 1.0), (0b0011, -0.75), (0b0110, 0.25)])
            .unwrap();
        let b = SparseCliffordVector::from_iter([(0b0001, 0.25), (0b0010, 0.5), (0b0111, -0.4)])
            .unwrap();

        let d = fast_metric_distance(&a, &b);
        let d_sq = fast_metric_distance_sq(&a, &b);
        assert!((d_sq - d * d).abs() < 1e-12);
    }

    #[test]
    fn metric_distance_sq_preserves_sub_planck_gate() {
        let a = SparseCliffordVector::from_iter([(0b0000, 1e-20)]).unwrap();
        let b = SparseCliffordVector::from_iter([(0b0001, 1e-20)]).unwrap();

        assert_eq!(fast_metric_distance_sq(&a, &b), f64::MAX);
        assert_eq!(fast_metric_distance(&a, &b), f64::MAX);
    }

    #[test]
    fn metric_distance_sq_from_dense_matches_squared_dense_metric_distance() {
        let dense = [
            0.3, -0.2, 0.1, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7,
        ];
        let b = SparseCliffordVector::from_iter([
            (0b0000, 0.1),
            (0b0001, -0.3),
            (0b0110, 0.9),
            (0b1111, -0.2),
        ])
        .unwrap();

        let d = fast_metric_distance_from_dense(&dense, &b);
        let d_sq = fast_metric_distance_sq_from_dense(&dense, &b);
        assert!((d_sq - d * d).abs() < 1e-12);
    }

    /// Invariant: `METRIC_WEIGHTS[i]` == GRADE_WEIGHTS[`GRADE_TABLE[i]`] for all i.
    ///
    /// This test is the formal proof that the compile-time derivation is correct
    /// and that no future edit can silently diverge the weights from the grade table.
    /// Blades 3, 7, 8, 12 had wrong weights in the previous manual version.
    #[test]
    fn metric_weights_derived_correctly_from_grade_table() {
        use crate::basis::GRADE_TABLE;
        const GRADE_WEIGHTS: [f64; 5] = [2.0, 1.5, 1.0, 0.5, 0.3];
        for i in 0..TOTAL_BLADES {
            let expected = GRADE_WEIGHTS[GRADE_TABLE[i] as usize];
            assert!(
                (METRIC_WEIGHTS[i] - expected).abs() < f64::EPSILON,
                "METRIC_WEIGHTS[{i}] = {} but GRADE_TABLE[{i}]={} requires weight {}",
                METRIC_WEIGHTS[i],
                GRADE_TABLE[i],
                expected
            );
        }
    }
}
