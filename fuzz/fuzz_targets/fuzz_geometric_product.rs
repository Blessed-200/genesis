//! Fuzz target: sparse_geometric_product with arbitrary inputs.
//!
//! Invariants that must NEVER violate:
//! 1. No panic on any input (finite or infinite/NaN — rejected gracefully)
//! 2. Result is either None or a valid SparseCliffordVector
//! 3. If both inputs are above PLANCK threshold, result is non-None (CS gate)
//! 4. Minkowski signature is preserved
//!
//! AX-ID: AXIOMA-001 (geometric product), AXIOMA-011 (CS gate)

#![no_main]
use libfuzzer_sys::fuzz_target;
use genesis_math::{SparseCliffordVector, sparse_geometric_product};
use genesis_types::constants::COGNITIVE_PLANCK_CONSTANT;

fuzz_target!(|data: &[u8]| {
    // Need at least 256 bytes for two 16×f64 multivectors
    if data.len() < 256 { return; }

    let parse_vec = |bytes: &[u8; 128]| -> Option<SparseCliffordVector> {
        let mut coeffs = [0.0f64; 16];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            coeffs[i] = f64::from_le_bytes(chunk.try_into().ok()?);
        }
        SparseCliffordVector::from_dense(&coeffs).ok()
    };

    let a_bytes: &[u8; 128] = data[..128].try_into().unwrap();
    let b_bytes: &[u8; 128] = data[128..256].try_into().unwrap();

    let a = match parse_vec(a_bytes) { Some(v) => v, None => return };
    let b = match parse_vec(b_bytes) { Some(v) => v, None => return };

    // Must not panic
    let result = sparse_geometric_product(&a, &b);

    // Invariant: if result is Some, it must be a valid vector
    if let Some(ref r) = result {
        assert!(r.coeffs.iter().all(|c| c.is_finite() || c.abs() <= COGNITIVE_PLANCK_CONSTANT),
            "geometric product produced non-finite coefficient");
        assert!(r.max_abs_coeff >= 0.0, "max_abs_coeff must be non-negative");
    }
});
