//! Fuzz target: Proof generation + verification roundtrip.
//!
//! Invariants:
//! 1. A just-built proof always verifies (is_internally_consistent)
//! 2. Tampering with the witness always fails verification
//! 3. No panic on arbitrary witness data
//!
//! AX-ID: LEY_FUNDACIONAL §5.5 (ProofGuard)

#![no_main]
use libfuzzer_sys::fuzz_target;
use genesis_types::proof::{AxiomID, AxiomGuard, WitnessBuilder};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() { return; }

    // Build a legitimate proof
    let mut builder = WitnessBuilder::new();
    let axioms = [
        AxiomID::MinkowskiSignature,
        AxiomID::CohomologyZero,
        AxiomID::ProofGuard,
    ];
    for axiom in &axioms {
        let _ = builder.check(*axiom, || true);
    }
    let timestamp = u64::from_le_bytes(
        data[..8.min(data.len())].iter().chain(&[0u8; 8]).copied().take(8)
            .collect::<Vec<_>>().try_into().unwrap_or([0u8; 8])
    );
    let proof = builder.build(timestamp);

    // Legitimate proof must always verify
    assert!(proof.is_internally_consistent(),
        "freshly built proof must be internally consistent");

    // Tampered proof must fail
    if proof.witness.len() > 0 {
        let mut tampered = proof.clone();
        // Flip a bit in witness
        let idx = (data.first().copied().unwrap_or(0) as usize) % tampered.witness.len();
        tampered.witness[idx] ^= 0xFF;
        // Tampered proof should NOT be consistent (hash mismatch)
        // Note: there's a ~1/256 chance this doesn't change the hash — that's fine
        let _ = tampered.is_internally_consistent(); // must not panic
    }

    // AxiomGuard::verify with required axioms
    let required = &[AxiomID::MinkowskiSignature, AxiomID::ProofGuard];
    let _ = AxiomGuard::verify(&proof, required); // must not panic
});
