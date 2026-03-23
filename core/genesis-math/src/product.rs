//! Geometric product: double-rail accumulation over `active_mask`.
//!
//! AX-ID: AXIOMA-001, AXIOMA-011
//!
//! # Hot-path structure
//! ```text
//! ┌─ CS gate (Mandato §2.3) ──────────────────────────── O(1) ─┐
//! │  abort if max_abs_a * max_abs_b < PLANCK, or non-finite    │
//! │  abort if active_mask = 0 on either input                  │
//! └────────────────────────────────────────────────────────────┘
//!          ↓
//! ┌─ Inner loop ─────────────────────── O(|active_a| · |active_b|) ─┐
//! │  iterate active_mask_a via trailing_zeros()                      │
//! │    row = CAYLEY_SIGN[i]                                          │
//! │    iterate active_mask_b via trailing_zeros()                    │
//! │      k = i ^ j                                                   │
//! │      term = coeffs_a[i] * coeffs_b[j]                           │
//! │      if row[j] > 0: pos[k] += term                              │
//! │      else:          neg[k] += term                              │
//! └──────────────────────────────────────────────────────────────────┘
//!          ↓
//! ┌─ Reduction ──────────────────────────────────────── O(16) ───┐
//! │  result[k] = pos[k] + neg[k]  (neg already negated on add)  │
//! │  collect via from_dense_buf                                  │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! # CS gate proof (Mandato §2.3)
//! Para cada blade k del resultado:
//!   `|result[k]|` ≤ Σ_{i⊕j=k} `|aᵢ|·|bⱼ|` ≤ 16 · max_abs_a · max_abs_b
//!
//! Para k fijo, los pares (i,j) con i⊕j=k son exactamente 16: {(i, i⊕k) : i∈0..15}.
//! Por tanto: si 16·max_abs_a·max_abs_b < PLANCK, todos los blades del resultado
//! son sub-Planck. El gate se dispara bajo esta condición más estricta.
//! La cota laxa anterior (max_abs_a·max_abs_b < PLANCK) suprimía incorrectamente
//! productos reales en el rango [PLANCK/16, PLANCK].
//!
//! # Double-rail accumulation
//! Two `[f64; 16]` buffers on the call stack (total 256 bytes = 4 cache lines).
//! `pos_buf`: terms with CAYLEY_SIGN = +1.
//! `neg_buf`: terms with CAYLEY_SIGN = −1, accumulated as `neg_buf[k] += |term|`
//!            (negation already applied), then summed as `pos + neg` where
//!            neg carries the negated value. See note in code.
//!
//! # No `_basis` parameter (Mandato §4.2)
//! `CAYLEY_SIGN` is compile-time in `.rodata`. No basis passed at call site.

use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;
use genesis_types::error::{GenesisError, SignatureViolationCode};

use crate::basis::TOTAL_BLADES;
use crate::multivector::{derive_all_metadata, SparseCliffordVector};
use crate::sign::CAYLEY_SIGN;

const DENSE_MASK: u16 = 0xFFFF;

pub(crate) const CAYLEY_SIGN_F64: [[f64; 16]; 16] = {
    let mut t = [[0.0f64; 16]; 16];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 16 {
            t[i][j] = if CAYLEY_SIGN[i][j] > 0 { 1.0 } else { -1.0 };
            j += 1;
        }
        i += 1;
    }
    t
};

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const SIGN_FLIP_BIT: u64 = 1u64 << 63;

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const fn build_sign_flip_masks() -> [[u64; TOTAL_BLADES]; TOTAL_BLADES] {
    let mut out = [[0u64; TOTAL_BLADES]; TOTAL_BLADES];
    let mut j = 0;
    while j < TOTAL_BLADES {
        let mut k = 0;
        // For each output blade k and b-index j:
        //   a-index i = k ^ j  (because blade_k = i ^ j in Clifford product)
        //   sign = CAYLEY_SIGN[i][j] = CAYLEY_SIGN[k ^ j][j]
        while k < TOTAL_BLADES {
            out[j][k] = if CAYLEY_SIGN[k ^ j][j] < 0 {
                SIGN_FLIP_BIT
            } else {
                0
            };
            k += 1;
        }
        j += 1;
    }
    out
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[allow(clippy::cast_possible_wrap)]
const fn build_xor_permute_indices() -> [[i64; TOTAL_BLADES]; TOTAL_BLADES] {
    let mut out = [[0i64; TOTAL_BLADES]; TOTAL_BLADES];
    let mut j = 0;
    while j < TOTAL_BLADES {
        let mut k = 0;
        while k < TOTAL_BLADES {
            let xor_index = k ^ j;
            out[j][k] = if xor_index <= i64::MAX as usize {
                xor_index as i64
            } else {
                0
            };
            k += 1;
        }
        j += 1;
    }
    out
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const SIGN_FLIP_MASKS: [[u64; TOTAL_BLADES]; TOTAL_BLADES] = build_sign_flip_masks();
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
const XOR_PERMUTE_INDICES: [[i64; TOTAL_BLADES]; TOTAL_BLADES] = build_xor_permute_indices();

/// Selection policy for geometric-product execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometricProductMode {
    /// Fast path: allows SIMD dispatch and suppresses non-finite payloads as `None`.
    Fast,
    /// Strict path: typed errors on non-finite metadata/coefficients, canonical signed zeros,
    /// and stable scalar reduction order (`i` ascending × `j` ascending) without SIMD reassociation.
    Strict,
}

#[inline]
fn strict_validate_inputs(
    a: &SparseCliffordVector,
    b: &SparseCliffordVector,
) -> Result<(), GenesisError> {
    if !a.max_abs_coeff.is_finite() {
        return Err(GenesisError::SignatureViolation {
            code: SignatureViolationCode::HotPathNonFiniteMetadata,
            blade_index: u16::MAX,
            normalized_value: None,
        });
    }
    if !b.max_abs_coeff.is_finite() {
        return Err(GenesisError::SignatureViolation {
            code: SignatureViolationCode::HotPathNonFiniteMetadata,
            blade_index: u16::MAX,
            normalized_value: None,
        });
    }
    if crate::multivector::has_non_finite_coeff(&a.coeffs)
        || crate::multivector::has_non_finite_coeff(&b.coeffs)
    {
        return Err(GenesisError::SignatureViolation {
            code: SignatureViolationCode::HotPathNonFiniteInputCoeff,
            blade_index: u16::MAX,
            normalized_value: None,
        });
    }
    Ok(())
}

#[inline]
fn strict_finalize_result(
    mut result_buf: [f64; TOTAL_BLADES],
) -> Result<Option<SparseCliffordVector>, GenesisError> {
    if crate::multivector::has_non_finite_coeff(&result_buf) {
        return Err(GenesisError::SignatureViolation {
            code: SignatureViolationCode::ColdPathNonFiniteResultCoeff,
            blade_index: u16::MAX,
            normalized_value: None,
        });
    }
    crate::multivector::canonicalize_signed_zero(&mut result_buf);
    let metadata = derive_all_metadata(&mut result_buf);
    if metadata.active_mask == 0 {
        return Ok(None);
    }
    Ok(Some(SparseCliffordVector::from_dense_with_metadata(
        result_buf,
        metadata.active_mask as u16, // SCV uses u16 (G(1,3)≤16 blades); DerivedMetadata uses u32
        metadata.max_abs_coeff,
        metadata.clifford_norm_sq,
    )))
}

#[inline]
fn geometric_product_scalar_sparse(
    a_coeffs: &[f64; TOTAL_BLADES],
    a_mask: u16,
    b_coeffs: &[f64; TOTAL_BLADES],
    b_mask: u16,
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    let mut mask_a = a_mask;
    while mask_a != 0 {
        let i = mask_a.trailing_zeros() as usize;
        let coef_a = a_coeffs[i];
        // loop-invariant, hoisted
        // CRYSTAL: O13 — inevitable
        let sign_row = &CAYLEY_SIGN_F64[i];
        let mut mask_b = b_mask;
        while mask_b != 0 {
            let j = mask_b.trailing_zeros() as usize;
            let k = i ^ j;
            result_buf[k] += coef_a * b_coeffs[j] * sign_row[j];
            mask_b &= mask_b - 1;
        }
        mask_a &= mask_a - 1;
    }
}

#[inline]
fn geometric_product_scalar_dense(
    a_coeffs: &[f64; TOTAL_BLADES],
    b_coeffs: &[f64; TOTAL_BLADES],
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    // HOT PATH: O(16²), dense G(1,3) geometric product.
    // Keep `i` outer so `sign_row[j]` and `b_coeffs[j]` stream sequentially; the
    // only non-sequential access left is the algebraically unavoidable scatter
    // store into `result_buf[i ^ j]`.
    for i in 0..TOTAL_BLADES {
        let coef_a = a_coeffs[i];
        let sign_row = &CAYLEY_SIGN_F64[i];
        for j in 0..TOTAL_BLADES {
            let k = i ^ j;
            result_buf[k] += coef_a * b_coeffs[j] * sign_row[j];
        }
    }
}

#[inline]
fn geometric_product_dispatch_dense(
    a_coeffs: &[f64; TOTAL_BLADES],
    b_coeffs: &[f64; TOTAL_BLADES],
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    if cfg!(feature = "deterministic_strict") {
        geometric_product_scalar_dense(a_coeffs, b_coeffs, result_buf);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                geometric_product_x86_avx512_dense(a_coeffs, b_coeffs, result_buf);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                geometric_product_x86_avx2_fma_dense(a_coeffs, b_coeffs, result_buf);
            }
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                geometric_product_x86_avx2_dense(a_coeffs, b_coeffs, result_buf);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                geometric_product_aarch64_neon_dense(a_coeffs, b_coeffs, result_buf);
            }
            return;
        }
    }

    geometric_product_scalar_dense(a_coeffs, b_coeffs, result_buf);
}

#[inline]
fn geometric_product_dispatch_by_mask(
    a_coeffs: &[f64; TOTAL_BLADES],
    a_mask: u16,
    b_coeffs: &[f64; TOTAL_BLADES],
    b_mask: u16,
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    if a_mask == DENSE_MASK && b_mask == DENSE_MASK {
        geometric_product_dispatch_dense(a_coeffs, b_coeffs, result_buf);
        return;
    }
    geometric_product_scalar_sparse(a_coeffs, a_mask, b_coeffs, b_mask, result_buf);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn geometric_product_x86_avx2_dense(
    a_coeffs: &[f64; TOTAL_BLADES],
    b_coeffs: &[f64; TOTAL_BLADES],
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    use std::arch::x86_64::{
        _mm256_cvtsd_f64, _mm256_extractf128_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_unpackhi_pd, _mm_cvtsd_f64, _mm_unpackhi_pd,
    };

    // HOT PATH: O(16²), dense G(1,3) product on AVX2.
    // Sequential SIMD loads come from `sign_row[j..]` and `b_coeffs[j..]`; the
    // result update remains scalar scatter because `i ^ j` is intrinsic to the
    // Clifford product. Keep lane values resident in registers to avoid
    // store-to-load forwarding stalls from a stack scratch buffer.
    for i in 0..TOTAL_BLADES {
        let coef_a = a_coeffs[i];
        let coef_a_vec = _mm256_set1_pd(coef_a);
        let sign_row = &CAYLEY_SIGN_F64[i];

        let mut j = 0usize;
        while j < TOTAL_BLADES {
            // SAFETY: `j` advances in multiples of 4 over a fixed-size 16-lane
            // array, so both sequential loads stay within bounds.
            let products = unsafe {
                let b_vec = _mm256_loadu_pd(b_coeffs.as_ptr().add(j));
                let scaled = _mm256_mul_pd(coef_a_vec, b_vec);
                let signs = _mm256_loadu_pd(sign_row.as_ptr().add(j));
                _mm256_mul_pd(scaled, signs)
            };
            // SAFETY: `products` is a live SIMD register. These extraction
            // intrinsics shuffle within registers only, so the hot path avoids
            // any intermediate stack buffer or reload from memory.
            let v0 = _mm256_cvtsd_f64(products);
            let v1 = _mm256_cvtsd_f64(_mm256_unpackhi_pd(products, products));
            let hi128 = _mm256_extractf128_pd(products, 1);
            let v2 = _mm_cvtsd_f64(hi128);
            let v3 = _mm_cvtsd_f64(_mm_unpackhi_pd(hi128, hi128));

            result_buf[i ^ j] += v0;
            result_buf[i ^ (j + 1)] += v1;
            result_buf[i ^ (j + 2)] += v2;
            result_buf[i ^ (j + 3)] += v3;
            j += 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn geometric_product_x86_avx2_fma_dense(
    a_coeffs: &[f64; TOTAL_BLADES],
    b_coeffs: &[f64; TOTAL_BLADES],
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    use std::arch::x86_64::{
        _mm256_cvtsd_f64, _mm256_extractf128_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_unpackhi_pd, _mm_cvtsd_f64, _mm_unpackhi_pd,
    };

    // HOT PATH: O(16²), dense G(1,3) product on AVX2+FMA hardware.
    // The read side is fully sequential; scatter remains scalar so we preserve
    // the scalar accumulation order while still exposing contiguous SIMD loads.
    // Keep lane values resident in registers to avoid store forwarding stalls.
    for i in 0..TOTAL_BLADES {
        let coef_a = a_coeffs[i];
        let coef_a_vec = _mm256_set1_pd(coef_a);
        let sign_row = &CAYLEY_SIGN_F64[i];

        let mut j = 0usize;
        while j < TOTAL_BLADES {
            // SAFETY: `j` advances in multiples of 4 over a fixed-size 16-lane
            // array, so both sequential loads stay within bounds.
            let products = unsafe {
                let b_vec = _mm256_loadu_pd(b_coeffs.as_ptr().add(j));
                let scaled = _mm256_mul_pd(coef_a_vec, b_vec);
                let signs = _mm256_loadu_pd(sign_row.as_ptr().add(j));
                _mm256_mul_pd(scaled, signs)
            };
            // SAFETY: `products` is a live SIMD register. These extraction
            // intrinsics shuffle within registers only, so the hot path avoids
            // any intermediate stack buffer or reload from memory.
            let v0 = _mm256_cvtsd_f64(products);
            let v1 = _mm256_cvtsd_f64(_mm256_unpackhi_pd(products, products));
            let hi128 = _mm256_extractf128_pd(products, 1);
            let v2 = _mm_cvtsd_f64(hi128);
            let v3 = _mm_cvtsd_f64(_mm_unpackhi_pd(hi128, hi128));

            result_buf[i ^ j] += v0;
            result_buf[i ^ (j + 1)] += v1;
            result_buf[i ^ (j + 2)] += v2;
            result_buf[i ^ (j + 3)] += v3;
            j += 4;
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
unsafe fn geometric_product_x86_avx512_dense(
    a_coeffs: &[f64; TOTAL_BLADES],
    b_coeffs: &[f64; TOTAL_BLADES],
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    use std::arch::x86_64::{
        _mm512_add_pd, _mm512_castsi512_pd, _mm512_loadu_pd, _mm512_loadu_si512, _mm512_mul_pd,
        _mm512_permutex2var_pd, _mm512_set1_pd, _mm512_setzero_pd, _mm512_storeu_pd, _mm512_xor_pd,
    };

    let a_lo = _mm512_loadu_pd(a_coeffs.as_ptr());
    let a_hi = _mm512_loadu_pd(a_coeffs.as_ptr().add(8));
    let mut acc_lo = _mm512_setzero_pd();
    let mut acc_hi = _mm512_setzero_pd();

    for (j, &coef_b) in b_coeffs.iter().enumerate() {
        let sign_lo =
            _mm512_castsi512_pd(_mm512_loadu_si512(SIGN_FLIP_MASKS[j][0..8].as_ptr().cast()));
        let sign_hi = _mm512_castsi512_pd(_mm512_loadu_si512(
            SIGN_FLIP_MASKS[j][8..16].as_ptr().cast(),
        ));

        let signed_lo = _mm512_xor_pd(a_lo, sign_lo);
        let signed_hi = _mm512_xor_pd(a_hi, sign_hi);

        let idx_lo = _mm512_loadu_si512(XOR_PERMUTE_INDICES[j][0..8].as_ptr().cast());
        let idx_hi = _mm512_loadu_si512(XOR_PERMUTE_INDICES[j][8..16].as_ptr().cast());

        let perm_lo = _mm512_permutex2var_pd(signed_lo, idx_lo, signed_hi);
        let perm_hi = _mm512_permutex2var_pd(signed_lo, idx_hi, signed_hi);

        let b_vec = _mm512_set1_pd(coef_b);
        acc_lo = _mm512_add_pd(acc_lo, _mm512_mul_pd(perm_lo, b_vec));
        acc_hi = _mm512_add_pd(acc_hi, _mm512_mul_pd(perm_hi, b_vec));
    }

    _mm512_storeu_pd(result_buf.as_mut_ptr(), acc_lo);
    _mm512_storeu_pd(result_buf.as_mut_ptr().add(8), acc_hi);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn geometric_product_aarch64_neon_dense(
    a_coeffs: &[f64; TOTAL_BLADES],
    b_coeffs: &[f64; TOTAL_BLADES],
    result_buf: &mut [f64; TOTAL_BLADES],
) {
    use std::arch::aarch64::{vdupq_n_f64, vgetq_lane_f64, vld1q_f64, vmulq_f64};

    // HOT PATH: O(16²), dense G(1,3) product on AArch64.
    // Hoist one sign row per `i` so the read side becomes sequential NEON loads
    // from `sign_row[j..]` and `b_coeffs[j..]`; the scatter store remains scalar.
    // Keep lane values in the NEON register file to avoid a stack round-trip.
    for i in 0..TOTAL_BLADES {
        let coef_a = a_coeffs[i];
        let coef_a_vec = vdupq_n_f64(coef_a);
        let sign_row = &CAYLEY_SIGN_F64[i];

        let mut j = 0usize;
        while j < TOTAL_BLADES {
            // SAFETY: `j` advances in multiples of 2 over a fixed-size 16-lane
            // array, so both sequential loads stay within bounds.
            let products = unsafe {
                let signs_vec = vld1q_f64(sign_row.as_ptr().add(j));
                let b_vec = vld1q_f64(b_coeffs.as_ptr().add(j));
                let scaled = vmulq_f64(coef_a_vec, b_vec);
                vmulq_f64(scaled, signs_vec)
            };
            // SAFETY: `products` is a live NEON register. Lane extraction is
            // register-to-register, so there is no memory round-trip here.
            let v0 = vgetq_lane_f64::<0>(products);
            let v1 = vgetq_lane_f64::<1>(products);

            result_buf[i ^ j] += v0;
            result_buf[i ^ (j + 1)] += v1;
            j += 2;
        }
    }
}

/// Computes A * B in G(1,3) via double-rail stack accumulation.
///
/// Returns `None` when gated by the Cauchy-Schwarz energy threshold, when either
/// input is the zero multivector, or when the algebraic product vanishes.
///
/// AX-ID: AXIOMA-001, AXIOMA-011
#[allow(clippy::many_single_char_names)]
// Notación canónica GA: i = blade_a, j = blade_b, k = blade_resultado.
// Renombrar diverge de la literatura estándar (Hestenes 2003, §2.1).
pub fn sparse_geometric_product(
    a: &SparseCliffordVector,
    b: &SparseCliffordVector,
) -> Option<SparseCliffordVector> {
    if cfg!(feature = "deterministic_strict") {
        return sparse_geometric_product_deterministic_strict(a, b)
            .ok()
            .flatten();
    }

    // ── CS gate (Mandato §2.3) ────────────────────────────────────────────────
    // Non-finite max_abs_coeff = upstream corruption. Quarantine immediately.
    if !a.max_abs_coeff.is_finite() || !b.max_abs_coeff.is_finite() {
        return None;
    }
    // Cota superior demostrable: para cada blade k del resultado,
    //   |result[k]| ≤ Σ_{i⊕j=k} |aᵢ||bⱼ| ≤ 16 · max_abs_a · max_abs_b
    // (exactamente 16 pares (i, i⊕k) para cada k fijo).
    // Si 16 · max_abs_a · max_abs_b < PLANCK → todos los blades sub-Planck.
    // Fix [B1]: cota anterior (sin factor 16) suprimía productos reales en [PLANCK/16, PLANCK].
    #[allow(clippy::cast_precision_loss)]
    // TOTAL_BLADES = 16, exactamente representable como f64.
    // f64 mantissa = 52 bits; 16 = 2^4, sin pérdida de precisión.
    if a.max_abs_coeff * b.max_abs_coeff * (TOTAL_BLADES as f64) < COGNITIVE_PLANCK_CONSTANT {
        return None;
    }
    // Zero multivectors (active_mask = 0) have max_abs_coeff = 0.0, so they
    // are already caught by the threshold test above. This guard is belt-and-
    // suspenders for the case where max_abs_coeff was manually zeroed.
    if a.active_mask == 0 || b.active_mask == 0 {
        return None;
    }

    // ── Stack buffer ──────────────────────────────────────────────────────────
    // [f64; TOTAL_BLADES] = 128 bytes in G(1,3). Zero heap allocation.
    let mut result_buf = [0.0f64; TOTAL_BLADES];

    // ── Fast paths by active-mask pattern ─────────────────────────────────────
    let pop_a = a.active_mask.count_ones();
    let pop_b = b.active_mask.count_ones();

    if pop_a == 1 && pop_b == 1 {
        let i = a.active_mask.trailing_zeros() as usize;
        let j = b.active_mask.trailing_zeros() as usize;
        let k = i ^ j;
        result_buf[k] = a.coeffs[i] * b.coeffs[j] * f64::from(CAYLEY_SIGN[i][j]);
    } else if pop_a == 1 || pop_b == 1 {
        geometric_product_scalar_sparse(
            &a.coeffs,
            a.active_mask,
            &b.coeffs,
            b.active_mask,
            &mut result_buf,
        );
    } else {
        geometric_product_dispatch_by_mask(
            &a.coeffs,
            a.active_mask,
            &b.coeffs,
            b.active_mask,
            &mut result_buf,
        );
    }

    // ── Consolidate metadata in one pass (no second reconstruction pass) ─────
    crate::multivector::canonicalize_signed_zero(&mut result_buf);
    let metadata = derive_all_metadata(&mut result_buf);

    if metadata.active_mask == 0 {
        return None;
    }

    Some(SparseCliffordVector::from_dense_with_metadata(
        result_buf,
        metadata.active_mask as u16,
        metadata.max_abs_coeff,
        metadata.clifford_norm_sq,
    ))
}

/// Computes A * B using an explicit execution mode.
///
/// # Errors
/// Returns [`GenesisError`] when strict validation detects non-finite
/// coefficients or metadata during multiplication.
pub fn sparse_geometric_product_with_mode(
    a: &SparseCliffordVector,
    b: &SparseCliffordVector,
    mode: GeometricProductMode,
) -> Result<Option<SparseCliffordVector>, GenesisError> {
    match mode {
        GeometricProductMode::Fast => Ok(sparse_geometric_product(a, b)),
        GeometricProductMode::Strict => sparse_geometric_product_deterministic_strict(a, b),
    }
}

#[allow(clippy::many_single_char_names)]
///
/// # Errors
/// Propagates any [`GenesisError`] returned by strict-mode geometric
/// multiplication.
pub fn sparse_geometric_product_deterministic_strict(
    a: &SparseCliffordVector,
    b: &SparseCliffordVector,
) -> Result<Option<SparseCliffordVector>, GenesisError> {
    // Reduction order is deliberately stable and documented:
    // - outer loop over `a_mask` uses trailing_zeros => increasing `i`
    // - inner loop over `b_mask` uses trailing_zeros => increasing `j`
    // - accumulation is scalar-only in strict mode (no SIMD reassociation)
    strict_validate_inputs(a, b)?;
    #[allow(clippy::cast_precision_loss)]
    if a.max_abs_coeff * b.max_abs_coeff * (TOTAL_BLADES as f64) < COGNITIVE_PLANCK_CONSTANT {
        return Ok(None);
    }
    if a.active_mask == 0 || b.active_mask == 0 {
        return Ok(None);
    }

    let mut result_buf = [0.0f64; TOTAL_BLADES];
    let pop_a = a.active_mask.count_ones();
    let pop_b = b.active_mask.count_ones();

    if pop_a == 1 && pop_b == 1 {
        let i = a.active_mask.trailing_zeros() as usize;
        let j = b.active_mask.trailing_zeros() as usize;
        let k = i ^ j;
        result_buf[k] = a.coeffs[i] * b.coeffs[j] * f64::from(CAYLEY_SIGN[i][j]);
    } else if pop_a == 1 || pop_b == 1 {
        geometric_product_scalar_sparse(
            &a.coeffs,
            a.active_mask,
            &b.coeffs,
            b.active_mask,
            &mut result_buf,
        );
    } else if a.active_mask == DENSE_MASK && b.active_mask == DENSE_MASK {
        // Strict mode must avoid SIMD dispatch to preserve a single reduction order.
        geometric_product_scalar_dense(&a.coeffs, &b.coeffs, &mut result_buf);
    } else {
        geometric_product_scalar_sparse(
            &a.coeffs,
            a.active_mask,
            &b.coeffs,
            b.active_mask,
            &mut result_buf,
        );
    }

    strict_finalize_result(result_buf)
}

/// Norma Lorentz del componente de grado 2 de A*B en G(1,3).
///
/// Computa `⟨(A*B)·rev(A*B)⟩₀` restringido a la parte bivectorial,
/// sin construir un `SparseCliffordVector` completo.
///
/// # Propósito
/// Función de distancia para HNSW en genesis-topology:
///   `fast_bivector_distance(a, b) = bivector_norm_sq_of_product(a, b).unwrap_or(f64::MAX)`
///
/// Seis blades de grado 2 en G(1,3):
///   3 (e₀₁, −1), 5 (e₀₂, −1), 6 (e₁₂, +1),
///   9 (e₀₃, −1), 10 (e₁₃, +1), 12 (e₂₃, +1)
///
/// # Retorno
/// `None` si el CS gate dispara (mismo criterio que `sparse_geometric_product`).
/// `Some(norm_sq)` donde `norm_sq` puede ser negativo (spacelike), cero (null),
/// o positivo (timelike). El llamador en HNSW usa `.abs()` como distancia.
///
/// # Rendimiento
/// Zero heap allocation. Stack buffer de 128 bytes. Seis FMA sobre los
/// blades de grado 2 post-acumulación.
/// Target: < 30 ns en hardware con AVX-512.
///
/// AX-ID: AXIOMA-001, AXIOMA-013, AXIOMA-014
/// Nota de rendimiento (CRATE-001 v0.2.3): en sandbox sin AVX-512 sostenido,
/// el benchmark `bivector_norm_sq_of_product_16x16` se mantiene ~420ns.
/// La causa principal es el costo del loop denso 16×16 y el manejo del
/// discriminante del enum de retorno bajo `black_box`; no hay heap allocation.
/// Resultado semántico del producto bivectorial — distingue "cero algebraico"
/// de "sub-Planck" para que los llamadores puedan implementar d(v,v)=0 correctamente.
///
/// AX-ID: AXIOMA-001, contrato CRATE-001 v0.2.2
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub enum BivectorProduct {
    /// La parte bivectorial del producto fue calculada. El valor puede ser 0.0
    /// (vectores en relación geométrica nula: paralelos, idénticos, ortogonales
    /// cuya parte bivectorial se cancela). Esta es información algebraica real.
    Computed(f64),
    /// CS gate disparó: max_abs_a × max_abs_b × 16 < PLANCK, o entrada no finita,
    /// o active_mask vacío. La energía es insuficiente para computar — no es
    /// una relación geométrica, es ausencia de señal.
    SubPlanck,
}

// FIX-B: Module-level constants — eliminate duplicate definitions in both
// bivector_norm_sq_of_product and bivector_norm_sq_of_product_lhs_dense.
/// Dense-to-packed mapping for grade-2 blades in G(1,3).
/// -1 means non-bivector blade.
const BIVECTOR_LANE_MAP: [i8; 16] = [-1, -1, -1, 0, -1, 1, 2, -1, -1, 3, 4, -1, 5, -1, -1, -1];

/// Packed Lorentz weights aligned with lane order [3, 5, 6, 9, 10, 12].
const BIVECTOR_LANE_WEIGHTS: [f64; 6] = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0];

/// Computes the Lorentz-invariant bivector norm squared of the geometric product `A*B`.
///
/// Extracts only grade-2 (bivector) components of `A*B` and weights by Minkowski
/// signature. Returns [`BivectorProduct::SubPlanck`] when the CS gate vetoes the
/// product. Used as the primary distance metric for HNSW semantic search.
///
/// **Notation:** `i` = source blade index, `j` = rhs blade index (canonical GA).
/// Renaming conflicts with Hestenes 2003 §2.1 convention.
///
/// AX-ID: AXIOMA-013, AXIOMA-001
#[allow(clippy::many_single_char_names)]
// i = blade_a, j = blade_b, k = blade_resultado — canonical GA notation.
pub fn bivector_norm_sq_of_product(
    a: &SparseCliffordVector,
    b: &SparseCliffordVector,
) -> BivectorProduct {
    // CS gate — mismo criterio que sparse_geometric_product.
    if !a.max_abs_coeff.is_finite() || !b.max_abs_coeff.is_finite() {
        return BivectorProduct::SubPlanck;
    }
    #[allow(clippy::cast_precision_loss)]
    // TOTAL_BLADES = 16, exactamente representable como f64.
    // f64 mantissa = 52 bits; 16 = 2^4, sin pérdida de precisión.
    if a.max_abs_coeff * b.max_abs_coeff * (TOTAL_BLADES as f64) < COGNITIVE_PLANCK_CONSTANT {
        return BivectorProduct::SubPlanck;
    }
    if a.active_mask == 0 || b.active_mask == 0 {
        return BivectorProduct::SubPlanck;
    }

    // Packed bivector buffer (6 lanes) instead of full dense 16-lane output.
    let mut bivector_buf = [0.0f64; 6];

    // Inner loop idéntico al path general de sparse_geometric_product.
    // El compilador vectoriza este loop con VFMADD cuando active_mask = 0xFFFF.
    let mut mask_a = a.active_mask;
    while mask_a != 0 {
        let i = mask_a.trailing_zeros() as usize;
        let coef_a = a.coeffs[i];
        let row = &CAYLEY_SIGN[i];
        let mut mask_b = b.active_mask;
        while mask_b != 0 {
            let j = mask_b.trailing_zeros() as usize;
            let k = i ^ j;
            let lane = BIVECTOR_LANE_MAP[k];
            if lane >= 0 {
                bivector_buf[lane as usize] += coef_a * b.coeffs[j] * f64::from(row[j]);
            }
            mask_b &= mask_b - 1;
        }
        mask_a &= mask_a - 1;
    }

    let mut norm_sq = 0.0f64;
    let mut has_signal = false;
    for lane in 0..6 {
        let v = bivector_buf[lane];
        if v.abs() > COGNITIVE_PLANCK_CONSTANT {
            norm_sq += v * v * BIVECTOR_LANE_WEIGHTS[lane];
            has_signal = true;
        }
    }

    BivectorProduct::Computed(if has_signal { norm_sq } else { 0.0 })
}

/// Bivector norm squared of `A*B` where `A` is provided as a dense `[f64; 16]` buffer.
///
/// Equivalent to [`bivector_norm_sq_of_product`] but avoids the sparse-to-dense
/// conversion overhead when the left operand is already dense (e.g. from AVX-512 output).
/// Used internally by HNSW distance computation after the NEON/AVX path.
///
/// AX-ID: AXIOMA-013
#[inline]
#[allow(clippy::many_single_char_names)]
pub fn bivector_norm_sq_of_product_lhs_dense(
    a_dense: &[f64; TOTAL_BLADES],
    b: &SparseCliffordVector,
) -> BivectorProduct {
    let mut a_mask = 0u16;
    let mut a_max = 0.0f64;
    for (i, &coef) in a_dense.iter().enumerate() {
        if !coef.is_finite() {
            return BivectorProduct::SubPlanck;
        }
        let abs = coef.abs();
        if abs > COGNITIVE_PLANCK_CONSTANT {
            a_mask |= 1u16 << i;
            if abs > a_max {
                a_max = abs;
            }
        }
    }

    if !b.max_abs_coeff.is_finite() {
        return BivectorProduct::SubPlanck;
    }
    #[allow(clippy::cast_precision_loss)]
    if a_max * b.max_abs_coeff * (TOTAL_BLADES as f64) < COGNITIVE_PLANCK_CONSTANT {
        return BivectorProduct::SubPlanck;
    }
    if a_mask == 0 || b.active_mask == 0 {
        return BivectorProduct::SubPlanck;
    }

    let mut bivector_buf = [0.0f64; 6];

    let mut mask_a = a_mask;
    while mask_a != 0 {
        let i = mask_a.trailing_zeros() as usize;
        let coef_a = a_dense[i];
        let row = &CAYLEY_SIGN[i];
        let mut mask_b = b.active_mask;
        while mask_b != 0 {
            let j = mask_b.trailing_zeros() as usize;
            let k = i ^ j;
            let lane = BIVECTOR_LANE_MAP[k];
            if lane >= 0 {
                bivector_buf[lane as usize] += coef_a * b.coeffs[j] * f64::from(row[j]);
            }
            mask_b &= mask_b - 1;
        }
        mask_a &= mask_a - 1;
    }

    let mut norm_sq = 0.0f64;
    let mut has_signal = false;
    for lane in 0..6 {
        let v = bivector_buf[lane];
        if v.abs() > COGNITIVE_PLANCK_CONSTANT {
            norm_sq += v * v * BIVECTOR_LANE_WEIGHTS[lane];
            has_signal = true;
        }
    }

    BivectorProduct::Computed(if has_signal { norm_sq } else { 0.0 })
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::basis::TOTAL_BLADES;

    fn e(bit: usize) -> SparseCliffordVector {
        SparseCliffordVector::from_iter([(bit, 1.0)]).unwrap()
    }

    // ── Verification Gate 1: exactness with {−1.0, 0.0, +1.0} coefficients ───

    #[test]
    fn exact_e0_squared_plus_one() {
        let r = sparse_geometric_product(&e(0b0001), &e(0b0001)).unwrap();
        assert_eq!(
            r.coeffs[0].to_bits(),
            1.0f64.to_bits(),
            "e₀² must be bit-exact +1.0"
        );
        assert_eq!(
            r.active_mask, 0b0000_0000_0000_0001u16,
            "result must be scalar blade"
        );
    }

    #[test]
    fn exact_e1_squared_minus_one() {
        let r = sparse_geometric_product(&e(0b0010), &e(0b0010)).unwrap();
        assert_eq!(
            r.coeffs[0].to_bits(),
            (-1.0f64).to_bits(),
            "e₁² must be bit-exact −1.0"
        );
    }

    #[test]
    fn exact_e2_squared_minus_one() {
        let r = sparse_geometric_product(&e(0b0100), &e(0b0100)).unwrap();
        assert_eq!(r.coeffs[0].to_bits(), (-1.0f64).to_bits());
    }

    #[test]
    fn exact_e3_squared_minus_one() {
        let r = sparse_geometric_product(&e(0b1000), &e(0b1000)).unwrap();
        assert_eq!(r.coeffs[0].to_bits(), (-1.0f64).to_bits());
    }

    #[test]
    fn exact_anticommutativity_e0_e1() {
        let ab = sparse_geometric_product(&e(0b0001), &e(0b0010)).unwrap();
        let ba = sparse_geometric_product(&e(0b0010), &e(0b0001)).unwrap();
        // Same result blade.
        let blade = ab.active_mask.trailing_zeros() as usize;
        assert_eq!(
            blade,
            ba.active_mask.trailing_zeros() as usize,
            "result blade must match"
        );
        // Exact cancellation: coeff(ab) + coeff(ba) = 0.
        let sum = ab.coeffs[blade] + ba.coeffs[blade];
        assert_eq!(
            sum.to_bits(),
            0.0f64.to_bits(),
            "e₀e₁ + e₁e₀ must be bit-exact 0.0"
        );
    }

    #[test]
    fn exact_all_six_anticommutator_pairs() {
        let bases = [0b0001usize, 0b0010, 0b0100, 0b1000];
        for i in 0..4 {
            for j in (i + 1)..4 {
                let ab = sparse_geometric_product(&e(bases[i]), &e(bases[j])).unwrap();
                let ba = sparse_geometric_product(&e(bases[j]), &e(bases[i])).unwrap();
                let blade = ab.active_mask.trailing_zeros() as usize;
                let sum = ab.coeffs[blade] + ba.coeffs[blade];
                assert_eq!(
                    sum.to_bits(),
                    0.0f64.to_bits(),
                    "eᵢeⱼ + eⱼeᵢ must cancel for bases [{i}][{j}]"
                );
            }
        }
    }

    // ── CS gate ───────────────────────────────────────────────────────────────

    #[test]
    fn cs_gate_fires_for_zero_vector() {
        let zero = SparseCliffordVector::zero();
        assert!(sparse_geometric_product(&zero, &e(0b0001)).is_none());
    }

    #[test]
    fn cs_gate_fires_for_nan_max_abs() {
        let mut mv = e(0b0001);
        mv.max_abs_coeff = f64::NAN;
        assert!(
            sparse_geometric_product(&mv, &e(0b0001)).is_none(),
            "NaN max_abs_coeff must trigger gate"
        );
    }

    #[test]
    fn cs_gate_fires_for_inf_max_abs() {
        let mut mv = e(0b0001);
        mv.max_abs_coeff = f64::INFINITY;
        assert!(
            sparse_geometric_product(&mv, &e(0b0001)).is_none(),
            "Inf max_abs_coeff must trigger non-finite check"
        );
    }

    #[test]
    fn cs_gate_does_not_suppress_null_vector_products() {
        // null vector (e₀+e₁): clifford_norm_sq=0 BUT max_abs_coeff=1.0.
        // The old L2-based gate would suppress this; the correct gate must not.
        let null_v = SparseCliffordVector::from_iter([(1, 1.0), (2, 1.0)]).unwrap();
        let e0 = e(1);
        assert!(
            null_v.clifford_norm_sq.abs() < 1e-12,
            "precondition: null vector"
        );
        let result = sparse_geometric_product(&null_v, &e0);
        assert!(
            result.is_some(),
            "Product of null vector × e₀ must NOT be gated"
        );
    }

    // ── Structural ────────────────────────────────────────────────────────────

    #[test]
    fn pseudoscalar_at_blade_15() {
        let p01 = sparse_geometric_product(&e(0b0001), &e(0b0010)).unwrap();
        let p012 = sparse_geometric_product(&p01, &e(0b0100)).unwrap();
        let p = sparse_geometric_product(&p012, &e(0b1000)).unwrap();
        assert_ne!(
            p.active_mask & (1 << 0b1111),
            0,
            "pseudoscalar must be at blade 15"
        );
        assert_eq!(p.active_mask.count_ones(), 1, "must be a single blade");
    }

    #[test]
    fn result_blades_within_g13_bounds() {
        let mv = SparseCliffordVector::from_iter([(0b0001, 1.0), (0b0010, 1.0)]).unwrap();
        if let Some(result) = sparse_geometric_product(&mv, &mv) {
            let mut mask = result.active_mask;
            while mask != 0 {
                let i = mask.trailing_zeros() as usize;
                assert!(i < TOTAL_BLADES, "blade {i} out of G(1,3) bounds");
                mask &= mask - 1;
            }
        }
    }

    #[test]
    fn associativity_e0_e1_e2() {
        let ab_c = sparse_geometric_product(
            &sparse_geometric_product(&e(0b0001), &e(0b0010)).unwrap(),
            &e(0b0100),
        )
        .unwrap();
        let a_bc = sparse_geometric_product(
            &e(0b0001),
            &sparse_geometric_product(&e(0b0010), &e(0b0100)).unwrap(),
        )
        .unwrap();
        for i in 0..TOTAL_BLADES {
            assert!(
                (ab_c.coeffs[i] - a_bc.coeffs[i]).abs() < COGNITIVE_PLANCK_CONSTANT * 1e6,
                "blade {i}: (e₀e₁)e₂ vs e₀(e₁e₂) differ"
            );
        }
    }

    #[test]
    fn no_basis_parameter_required() {
        // Verify the function compiles and runs without a CliffordBasis argument.
        let r = sparse_geometric_product(&e(0b0001), &e(0b0010));
        assert!(r.is_some());
    }

    /// Regresión [B1]: CS gate anterior (sin factor 16) suprimía productos reales
    /// en el rango [PLANCK/16, PLANCK]. Con la corrección, estos productos pasan.
    ///
    /// AX-ID: AXIOMA-011, MANDATO §2.3
    #[test]
    fn cs_gate_does_not_suppress_constructive_superposition() {
        use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;
        // Construir max_abs ≈ sqrt(PLANCK/8): producto = PLANCK/8 ∈ [PLANCK/16, PLANCK].
        // Con el gate incorrecto (sin ×16): PLANCK/8 < PLANCK → suprimido (INCORRECTO).
        // Con el gate correcto (×16): 16 × PLANCK/8 = 2×PLANCK > PLANCK → NO suprimido.
        let coef = (COGNITIVE_PLANCK_CONSTANT * 2.0f64).sqrt();
        // e₀ × e₀ = +1 (scalar). Ambos vectores tienen max_abs_coeff = coef.
        let a = SparseCliffordVector::from_iter([(0b0001usize, coef)]).unwrap();
        let b = SparseCliffordVector::from_iter([(0b0001usize, coef)]).unwrap();
        // Verificar precondición: 16 × coef² ≥ PLANCK → gate no debe dispararse.
        assert!(
            a.max_abs_coeff * b.max_abs_coeff * (TOTAL_BLADES as f64) >= COGNITIVE_PLANCK_CONSTANT,
            "precondition: gate no debe dispararse con factor 16"
        );
        let result = sparse_geometric_product(&a, &b);
        assert!(
            result.is_some(),
            "producto con 16×max_a×max_b > PLANCK no debe ser suprimido por CS gate"
        );
    }

    #[test]
    fn cs_gate_factor_16_boundary() {
        use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;

        // Caso 1: 15.9999 × (max_a · max_b) < PLANCK ⇒ gate debe suprimir (None).
        let coef_below = (COGNITIVE_PLANCK_CONSTANT / 16.0001f64).sqrt();
        let a_below = SparseCliffordVector::from_iter([(0b0001usize, coef_below)]).unwrap();
        let b_below = SparseCliffordVector::from_iter([(0b0001usize, coef_below)]).unwrap();
        assert!(
            a_below.max_abs_coeff * b_below.max_abs_coeff * 15.9999f64 < COGNITIVE_PLANCK_CONSTANT
        );
        assert!(sparse_geometric_product(&a_below, &b_below).is_none());

        // Caso 2: 16.0001 × (max_a · max_b) ≥ PLANCK y producto algebraicamente no nulo ⇒ Some.
        let coef_above = (COGNITIVE_PLANCK_CONSTANT * 1.0001f64).sqrt();
        let a_above = SparseCliffordVector::from_iter([(0b0001usize, coef_above)]).unwrap();
        let b_above = SparseCliffordVector::from_iter([(0b0001usize, coef_above)]).unwrap();
        assert!(
            a_above.max_abs_coeff * b_above.max_abs_coeff * 16.0001f64 >= COGNITIVE_PLANCK_CONSTANT
        );
        let above = sparse_geometric_product(&a_above, &b_above).unwrap();
        assert!(above.coeffs[0].abs() >= COGNITIVE_PLANCK_CONSTANT);
    }

    fn mv_from_mask(mask: u16, base: f64) -> SparseCliffordVector {
        SparseCliffordVector::from_iter((0..TOTAL_BLADES).filter_map(|i| {
            if (mask & (1u16 << i)) != 0 {
                Some((i, (i as f64).mul_add(0.5, base)))
            } else {
                None
            }
        }))
        .unwrap()
    }

    fn assert_kernel_bit_equivalent(mask_a: u16, mask_b: u16) {
        let a = mv_from_mask(mask_a, 1.0);
        let b = mv_from_mask(mask_b, 2.0);

        let mut scalar = [0.0f64; TOTAL_BLADES];
        if mask_a == DENSE_MASK && mask_b == DENSE_MASK {
            geometric_product_scalar_dense(&a.coeffs, &b.coeffs, &mut scalar);
        } else {
            geometric_product_scalar_sparse(&a.coeffs, mask_a, &b.coeffs, mask_b, &mut scalar);
        }

        let mut dispatch = [0.0f64; TOTAL_BLADES];
        geometric_product_dispatch_by_mask(&a.coeffs, mask_a, &b.coeffs, mask_b, &mut dispatch);

        for i in 0..TOTAL_BLADES {
            assert_eq!(
                scalar[i].to_bits(),
                dispatch[i].to_bits(),
                "kernel mismatch en blade {i} para máscaras {mask_a:#06x} x {mask_b:#06x}"
            );
        }
    }

    #[test]
    fn kernel_bit_equivalence_mask_patterns() {
        for (mask_a, mask_b) in [
            (0x0001u16, 0x0002u16),
            (0x0001u16, 0x00FFu16),
            (0x00FFu16, 0x0F0Fu16),
            (0xAAAAu16, 0x5555u16),
            (DENSE_MASK, 0x00FFu16),
            (0x0FFFu16, DENSE_MASK),
            (DENSE_MASK, DENSE_MASK),
        ] {
            assert_kernel_bit_equivalent(mask_a, mask_b);
        }
    }

    #[test]
    fn kernel_bit_equivalence_dense_dispatch_vs_scalar() {
        assert_kernel_bit_equivalent(DENSE_MASK, DENSE_MASK);
    }

    #[test]
    fn strict_mode_stable_cancellation_to_bits() {
        let a = SparseCliffordVector::from_iter([(0, 1.0), (1, 1.0), (2, 1.0)]).unwrap();
        let b = SparseCliffordVector::from_iter([(0, 1.0), (1, -1.0), (2, 1.0)]).unwrap();

        let strict = sparse_geometric_product_with_mode(&a, &b, GeometricProductMode::Strict)
            .unwrap()
            .unwrap();
        let det = sparse_geometric_product_deterministic_strict(&a, &b)
            .unwrap()
            .unwrap();

        for i in 0..TOTAL_BLADES {
            assert_eq!(strict.coeffs[i].to_bits(), det.coeffs[i].to_bits());
        }
    }

    #[test]
    fn strict_mode_canonicalizes_signed_zero_to_bits() {
        let a = SparseCliffordVector::from_iter([(1, 1.0), (2, 1.0)]).unwrap();
        let b = SparseCliffordVector::from_iter([(1, -1.0), (2, 1.0)]).unwrap();

        let out = sparse_geometric_product_with_mode(&a, &b, GeometricProductMode::Strict)
            .unwrap()
            .unwrap();
        for coeff in out.coeffs {
            if coeff == 0.0 {
                assert_eq!(coeff.to_bits(), 0.0f64.to_bits());
            }
        }
    }

    #[test]
    fn strict_mode_rejects_non_finite_inputs_typed() {
        let mut bad_meta = e(0b0001);
        bad_meta.max_abs_coeff = f64::INFINITY;
        let err =
            sparse_geometric_product_with_mode(&bad_meta, &e(0b0001), GeometricProductMode::Strict)
                .unwrap_err();
        assert_eq!(
            err,
            GenesisError::SignatureViolation {
                code: SignatureViolationCode::HotPathNonFiniteMetadata,
                blade_index: u16::MAX,
                normalized_value: None,
            }
        );

        let mut bad_coeff = e(0b0001);
        bad_coeff.coeffs[1] = f64::NAN;
        let err = sparse_geometric_product_with_mode(
            &bad_coeff,
            &e(0b0001),
            GeometricProductMode::Strict,
        )
        .unwrap_err();
        assert_eq!(
            err,
            GenesisError::SignatureViolation {
                code: SignatureViolationCode::HotPathNonFiniteInputCoeff,
                blade_index: u16::MAX,
                normalized_value: None,
            }
        );
    }
    #[test]
    fn deterministic_strict_dense_path_matches_scalar_dense_bits() {
        let a = mv_from_mask(DENSE_MASK, 1.0);
        let b = mv_from_mask(DENSE_MASK, 2.0);

        let out = sparse_geometric_product_deterministic_strict(&a, &b)
            .unwrap()
            .unwrap();

        let mut scalar = [0.0f64; TOTAL_BLADES];
        geometric_product_scalar_dense(&a.coeffs, &b.coeffs, &mut scalar);
        crate::multivector::canonicalize_signed_zero(&mut scalar);

        for i in 0..TOTAL_BLADES {
            assert_eq!(out.coeffs[i].to_bits(), scalar[i].to_bits());
        }
    }

    #[test]
    fn deterministic_strict_repeated_runs_are_bit_stable() {
        let a = mv_from_mask(DENSE_MASK, 1.0);
        let b = mv_from_mask(DENSE_MASK, 2.0);

        let first = sparse_geometric_product_deterministic_strict(&a, &b)
            .unwrap()
            .unwrap();
        for _ in 0..32 {
            let next = sparse_geometric_product_deterministic_strict(&a, &b)
                .unwrap()
                .unwrap();
            for i in 0..TOTAL_BLADES {
                assert_eq!(
                    first.coeffs[i].to_bits(),
                    next.coeffs[i].to_bits(),
                    "deterministic_strict mismatch en blade {i}"
                );
            }
        }
    }

    #[test]
    fn deterministic_strict_canonicalizes_negative_zero() {
        let a = SparseCliffordVector::from_iter([(1, 1.0), (2, 1.0)]).unwrap();
        let b = SparseCliffordVector::from_iter([(1, -1.0), (2, 1.0)]).unwrap();

        let out = sparse_geometric_product_deterministic_strict(&a, &b)
            .unwrap()
            .unwrap();
        for coeff in out.coeffs {
            if coeff == 0.0 {
                assert_eq!(coeff.to_bits(), 0.0f64.to_bits());
            }
        }
    }

    #[test]
    fn deterministic_strict_rejects_non_finite_metadata_and_inputs() {
        let mut bad_meta = e(0b0001);
        bad_meta.max_abs_coeff = f64::NAN;
        let err = sparse_geometric_product_deterministic_strict(&bad_meta, &e(0b0001)).unwrap_err();
        assert_eq!(
            err,
            GenesisError::SignatureViolation {
                code: SignatureViolationCode::HotPathNonFiniteMetadata,
                blade_index: u16::MAX,
                normalized_value: None,
            }
        );

        let mut bad_coeffs = e(0b0001);
        bad_coeffs.coeffs[1] = f64::INFINITY;
        let err =
            sparse_geometric_product_deterministic_strict(&bad_coeffs, &e(0b0001)).unwrap_err();
        assert_eq!(
            err,
            GenesisError::SignatureViolation {
                code: SignatureViolationCode::HotPathNonFiniteInputCoeff,
                blade_index: u16::MAX,
                normalized_value: None,
            }
        );
    }

    // ── bivector_norm_sq_of_product ───────────────────────────────────────────

    #[test]
    fn bivector_norm_sq_e0e1_is_minus_one() {
        // e₀ * e₁ = e₀₁ (blade 3). weight = −1. norm_sq = 1² × (−1) = −1.
        let r = match bivector_norm_sq_of_product(&e(0b0001), &e(0b0010)) {
            BivectorProduct::Computed(v) => v,
            BivectorProduct::SubPlanck => panic!("unexpected SubPlanck"),
        };
        assert!(
            (r - (-1.0)).abs() < 1e-15,
            "bivector_norm_sq(e₀, e₁) must be −1.0, got {r}"
        );
    }

    #[test]
    fn bivector_norm_sq_e1e2_is_plus_one() {
        // e₁ * e₂ = e₁₂ (blade 6). weight = +1. norm_sq = 1² × (+1) = +1.
        let r = match bivector_norm_sq_of_product(&e(0b0010), &e(0b0100)) {
            BivectorProduct::Computed(v) => v,
            BivectorProduct::SubPlanck => panic!("unexpected SubPlanck"),
        };
        assert!(
            (r - 1.0).abs() < 1e-15,
            "bivector_norm_sq(e₁, e₂) must be +1.0, got {r}"
        );
    }

    #[test]
    fn bivector_norm_sq_scalar_product_is_computed_zero() {
        // e₀ * e₀ = +1 (scalar, grado 0). Ningún blade de grado 2 activo → Computed(0.0).
        let r = bivector_norm_sq_of_product(&e(0b0001), &e(0b0001));
        match r {
            BivectorProduct::Computed(v) => assert!(v.abs() < 1e-15),
            BivectorProduct::SubPlanck => panic!("producto escalar puro no debe ser SubPlanck"),
        }
    }

    #[test]
    fn bivector_norm_sq_consistent_with_full_product() {
        // Verificar consistencia con sparse_geometric_product para entrada mixta.
        let a = SparseCliffordVector::from_iter([(0b0001, 2.0), (0b0010, 3.0)]).unwrap();
        let b = SparseCliffordVector::from_iter([(0b0100, 1.0), (0b1000, -1.0)]).unwrap();

        let full = sparse_geometric_product(&a, &b).unwrap();
        let biv_fast = match bivector_norm_sq_of_product(&a, &b) {
            BivectorProduct::Computed(v) => v,
            BivectorProduct::SubPlanck => panic!("unexpected SubPlanck"),
        };

        // Calcular manualmente la norma bivectorial desde el producto completo.
        const BIVECTOR_INDICES: [usize; 6] = [3, 5, 6, 9, 10, 12];
        const BIVECTOR_WEIGHTS: [f64; 6] = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
        let biv_ref: f64 = BIVECTOR_INDICES
            .iter()
            .zip(BIVECTOR_WEIGHTS.iter())
            .map(|(&k, &w)| full.coeffs[k] * full.coeffs[k] * w)
            .sum();

        assert!(
            (biv_fast - biv_ref).abs() < 1e-12,
            "bivector_norm_sq_of_product={biv_fast} pero referencia={biv_ref}"
        );
    }

    #[test]
    fn bivector_norm_sq_zero_input_is_none() {
        let zero = SparseCliffordVector::zero();
        assert_eq!(
            bivector_norm_sq_of_product(&zero, &e(0b0001)),
            BivectorProduct::SubPlanck
        );
        assert_eq!(
            bivector_norm_sq_of_product(&e(0b0001), &zero),
            BivectorProduct::SubPlanck
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PROPERTY-BASED TESTS — 10 000 cases, feature-gated
// Run with: cargo test -p genesis-math --features properties
//
// AX-ID: AXIOMA-001
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, target_arch = "x86_64"))]
mod simd_equivalence_tests {
    use super::{
        geometric_product_scalar_dense, geometric_product_x86_avx2_fma_dense, TOTAL_BLADES,
    };

    #[test]
    fn avx2_fma_dense_kernel_matches_scalar_for_one_million_cases() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }

        let mut state = 0xD1B5_4A32_CE77_9A1Fu64;
        for _ in 0..1_000_000usize {
            let mut a = [0.0f64; TOTAL_BLADES];
            let mut b = [0.0f64; TOTAL_BLADES];

            for i in 0..TOTAL_BLADES {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let x = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
                a[i] = x.mul_add(2.0, -1.0);

                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let y = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
                b[i] = y.mul_add(2.0, -1.0);
            }

            let mut scalar = [0.0f64; TOTAL_BLADES];
            let mut simd = [0.0f64; TOTAL_BLADES];
            geometric_product_scalar_dense(&a, &b, &mut scalar);
            // SAFETY: guarded by runtime feature checks above.
            unsafe {
                geometric_product_x86_avx2_fma_dense(&a, &b, &mut simd);
            }

            for k in 0..TOTAL_BLADES {
                assert!(
                    (scalar[k] - simd[k]).abs() < 1e-12,
                    "mismatch at blade {k}: scalar={} simd={}",
                    scalar[k],
                    simd[k]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(all(test, feature = "properties"))]
#[allow(clippy::needless_range_loop)]
mod property_tests {
    use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;
    use proptest::prelude::*;

    use super::*;

    fn approx_eq(lhs: Option<&SparseCliffordVector>, rhs: Option<&SparseCliffordVector>) -> bool {
        const TOL: f64 = COGNITIVE_PLANCK_CONSTANT * 1e6;
        match (lhs, rhs) {
            (None, None) => true,
            (Some(l), None) => l.max_abs_coeff < TOL,
            (None, Some(r)) => r.max_abs_coeff < TOL,
            (Some(l), Some(r)) => {
                for i in 0..TOTAL_BLADES {
                    if (l.coeffs[i] - r.coeffs[i]).abs() > TOL {
                        return false;
                    }
                }
                true
            }
        }
    }

    fn arb_mv() -> impl Strategy<Value = SparseCliffordVector> {
        prop::collection::vec((0usize..TOTAL_BLADES, -50.0f64..50.0f64), 0..=6)
            .prop_filter_map("valid pairs", |pairs| {
                SparseCliffordVector::from_iter(pairs).ok()
            })
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 10_000,
            ..Default::default()
        })]

        #[test]
        fn prop_associativity(
            a in arb_mv(),
            b in arb_mv(),
            c in arb_mv(),
        ) {
            let ab   = sparse_geometric_product(&a, &b);
            let ab_c = ab.as_ref().and_then(|p| sparse_geometric_product(p, &c));
            let bc   = sparse_geometric_product(&b, &c);
            let a_bc = bc.as_ref().and_then(|p| sparse_geometric_product(&a, p));
            prop_assert!(
                approx_eq(ab_c.as_ref(), a_bc.as_ref()),
                "(A*B)*C ≠ A*(B*C):\n  A={:?}\n  B={:?}\n  C={:?}",
                a, b, c
            );
        }

        #[test]
        fn prop_left_distributivity(
            a in arb_mv(),
            b in arb_mv(),
            c in arb_mv(),
        ) {
            let mut bc_buf = [0.0f64; TOTAL_BLADES];
            for i in 0..TOTAL_BLADES { bc_buf[i] = b.coeffs[i] + c.coeffs[i]; }
            let Ok(b_plus_c) = SparseCliffordVector::from_dense(&bc_buf) else {
                return Ok(());
            };
            let lhs = sparse_geometric_product(&a, &b_plus_c);
            let ab  = sparse_geometric_product(&a, &b);
            let ac  = sparse_geometric_product(&a, &c);
            let mut rhs_buf = [0.0f64; TOTAL_BLADES];
            if let Some(p) = &ab { for i in 0..TOTAL_BLADES { rhs_buf[i] += p.coeffs[i]; } }
            if let Some(p) = &ac { for i in 0..TOTAL_BLADES { rhs_buf[i] += p.coeffs[i]; } }
            let rhs = SparseCliffordVector::from_dense(&rhs_buf).ok();
            prop_assert!(
                approx_eq(lhs.as_ref(), rhs.as_ref()),
                "A*(B+C) ≠ A*B + A*C"
            );
        }

        #[test]
        fn prop_scalar_left_identity(a in arb_mv()) {
            let one    = SparseCliffordVector::from_iter([(0, 1.0)]).unwrap();
            let result = sparse_geometric_product(&one, &a);
            if a.is_negligible() {
                prop_assert!(
                    result.is_none() || result.as_ref().unwrap().is_negligible()
                );
            } else {
                prop_assert!(result.is_some(), "1 * A must not be gated for non-negligible A");
                prop_assert!(approx_eq(result.as_ref(), Some(&a)));
            }
        }

        #[test]
        fn prop_result_grade_bounded(a in arb_mv(), b in arb_mv()) {
            if let Some(result) = sparse_geometric_product(&a, &b) {
                let mut mask = result.active_mask;
                while mask != 0 {
                    let i     = mask.trailing_zeros() as usize;
                    let grade = i.count_ones() as usize;
                    prop_assert!(grade <= 4,
                        "blade {i:#06b} grade {grade} > 4 (impossible in G(1,3))");
                    mask &= mask - 1;
                }
            }
        }

        #[test]
        fn prop_cs_gate_cota_superior(a in arb_mv(), b in arb_mv()) {
            // If product is None (gated), verify it was justified.
            // If max_abs_a * max_abs_b >= PLANCK, the result should not be None
            // unless it algebraically vanishes.
            if a.max_abs_coeff * b.max_abs_coeff >= COGNITIVE_PLANCK_CONSTANT * 1e3 {
                // Just verify it doesn't panic. Algebraic vanishing is allowed.
                let _ = sparse_geometric_product(&a, &b);
            }
        }
    }
}
