# GENESIS_PROOF_SPEC.md — Proof System Specification

**Version:** 2.0.0  
**Status:** IMPLEMENTED — `shared/genesis-types/src/proof.rs`  
**Implements:** `H_restricción` guard. LEY_FUNDACIONAL §5.5–5.7.

> **Implementation note.** The proof system is part of `genesis-types` (CRATE-000),
> not a separate crate. All types described here are exported from
> `genesis_types::{AxiomGuard, AxiomID, AxiomSet, Mutation, Proof, WitnessBuilder}`.
> Hash algorithm: **BLAKE3** (`blake3 =1.5.4`). Not SHA-256.

---

## 1. Location

```
shared/genesis-types/src/proof.rs      ← implementation
shared/genesis-types/src/constants.rs  ← LAMBDA2_MIN, PROOF_PENALTY, PROOF_MAX_AGE_NS, ...
shared/genesis-types/src/error.rs      ← ProofMissing, ProofInvalid, InvariantViolation
shared/genesis-types/src/lib.rs        ← re-exports
```

---

## 2. Types

### 2.1 AxiomID

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AxiomID {
    MinkowskiSignature    = 0,   // LEY_FUNDACIONAL §5.1
    CohomologyZero        = 1,   // LEY_FUNDACIONAL §5.2
    AlgebraicConnectivity = 2,   // LEY_FUNDACIONAL §5.3
    PlanckConstant        = 3,   // LEY_FUNDACIONAL §5.4
    ProofGuard            = 4,   // LEY_FUNDACIONAL §5.5
    DualityConsistency    = 5,   // LEY_FUNDACIONAL §5.6
    DimensionalAdmission  = 6,   // LEY_FUNDACIONAL §5.7
}

impl AxiomID {
    pub const STRUCTURAL_REQUIRED: &'static [AxiomID] = &[
        AxiomID::MinkowskiSignature,
        AxiomID::CohomologyZero,
        AxiomID::AlgebraicConnectivity,
        AxiomID::PlanckConstant,
        AxiomID::ProofGuard,
    ];

    pub const EXPANSION_REQUIRED: &'static [AxiomID] = &[
        AxiomID::MinkowskiSignature,
        AxiomID::CohomologyZero,
        AxiomID::AlgebraicConnectivity,
        AxiomID::PlanckConstant,
        AxiomID::ProofGuard,
        AxiomID::DualityConsistency,
        AxiomID::DimensionalAdmission,
    ];
}
```

Discriminants are stable. Do not reorder.

---

### 2.2 AxiomSet

```rust
pub struct AxiomSet(pub u8);   // bitmask; bit i = AxiomID with discriminant i
```

Const-constructible. Key methods:

```rust
pub const fn empty() -> Self
pub const fn from_axiom(axiom: AxiomID) -> Self
pub fn from_slice(axioms: &[AxiomID]) -> Self
pub const fn contains(self, axiom: AxiomID) -> bool
pub const fn union(self, other: Self) -> Self
pub const fn intersects(self, other: Self) -> bool
pub const fn difference(self, other: Self) -> Self
pub const fn is_subset_of(self, other: Self) -> bool
pub const fn is_superset_of(self, other: Self) -> bool
pub const fn is_valid(self) -> bool    // no bits outside 0..=6
```

---

### 2.3 Proof

```rust
pub struct Proof {
    pub axioms_checked: AxiomSet,
    pub witness:        Vec<u8>,   // serialised verification trace
    pub timestamp:      u64,       // nanoseconds since Unix epoch
    pub hash:           [u8; 32],  // BLAKE3(witness)
}

impl Proof {
    pub fn new(axioms: AxiomSet, witness: Vec<u8>, timestamp: u64) -> Self {
        // hash = BLAKE3(witness) computed at construction
    }

    pub fn is_internally_consistent(&self) -> bool {
        // blake3(witness) == self.hash
    }

    pub fn is_fresh(&self, current_ns: u64) -> bool {
        // current_ns.saturating_sub(self.timestamp) < PROOF_MAX_AGE_NS
    }
}
```

---

### 2.4 Mutation trait

```rust
pub trait Mutation: Send + Sync {
    fn propose(&self) -> Result<Proof, GenesisError>;
    fn apply(&self)   -> Result<(), GenesisError>;
    fn name(&self)    -> &'static str;
}
```

`propose()` must generate the Proof before `apply()` executes. Never generate proofs inside hot paths.

---

### 2.5 AxiomGuard

```rust
pub struct AxiomGuard;

impl AxiomGuard {
    pub fn verify(proof: &Proof, required: &[AxiomID]) -> bool;

    pub fn verify_for<'p, M: Mutation>(
        proof: &'p Proof,
        mutation: &M,
    ) -> Option<VerifiedProof<'p, M>>;
}
```

`verify` performs:
1. BLAKE3 integrity: `blake3(proof.witness) == proof.hash`
2. Symbolic witness replay: deserialises each frame and checks `result == 1`
3. Axiom coverage: every axiom in `required` has a verified frame

Returns `false` if any check fails.

---

## 3. WitnessBuilder

```rust
pub struct WitnessBuilder {
    frames: Vec<u8>,
    axioms: AxiomSet,
}

impl WitnessBuilder {
    pub fn new() -> Self

    pub fn check<F: FnOnce() -> bool>(
        &mut self,
        axiom: AxiomID,
        f: F,
    ) -> Result<(), GenesisError>
    // Returns Err(InvariantViolation { axiom_id }) if f() == false

    pub fn build(self, timestamp: u64) -> Proof
    // Calls BLAKE3 once on the complete witness
}
```

---

## 4. Usage — dimensional expansion

```rust
// In GramSchmidtExpander::try_expand (genesis-evolution, Phase 5):

fn propose_expansion(&self, candidate: &SparseCliffordVector) -> Result<Proof, GenesisError> {
    let mut builder = WitnessBuilder::new();

    builder.check(AxiomID::MinkowskiSignature, || {
        self.state.basis.signature == [1.0, -1.0, -1.0, -1.0]
    })?;

    builder.check(AxiomID::CohomologyZero, || {
        self.state.manifold.h1_is_zero_fast()
    })?;

    builder.check(AxiomID::AlgebraicConnectivity, || {
        self.state.manifold.compute_lambda2() >= LAMBDA2_MIN
    })?;

    builder.check(AxiomID::DualityConsistency, || {
        // Fisher metric is current for all nodes affected by this expansion
        let affected = self.state.manifold.find_affected_nodes(candidate);
        affected.iter().all(|&n| self.state.fisher.is_current(n))
    })?;

    builder.check(AxiomID::DimensionalAdmission, || {
        // ΔH_total < C_dim(N) — admission cost gate
        let n = self.state.manifold.node_count();
        let c_dim = LAMBDA_DIM_FIXED + LAMBDA_DIM_LOG * (n as f64).ln();
        let delta_h = self.state.estimate_delta_h_after_expansion(candidate);
        delta_h < c_dim
    })?;

    builder.check(AxiomID::ProofGuard, || true)?;

    Ok(builder.build(current_timestamp_ns()))
}
```

---

## 5. Wire format

Each `WitnessBuilder::check` call appends one frame to `witness`:

```
[axiom_id: u8] [result: u8] [ctx_len: u16 LE] [ctx_bytes: ctx_len × u8]
```

`result = 1` for pass, `0` for fail (which also returns `Err` immediately).
`ctx_len = 0` in the current implementation (context bytes reserved for future use).

`AxiomGuard::verify` deserialises and replays all frames, confirming each `result == 1`.

---

## 6. Constants

```rust
// shared/genesis-types/src/constants.rs

pub const LAMBDA2_MIN:      f64 = 0.1;
pub const PROOF_PENALTY:    f64 = 1000.0;
pub const PROOF_MAX_AGE_NS: u64 = 5_000_000_000;  // 5 seconds

// Variational parameters (LEY_FUNDACIONAL v1.1.0)
pub const DENSITY_PENALTY:              f64 = 50.0;
pub const DELTA_DUALITY:                f64 = 0.01;
pub const KAPPA_REDUNDANCY:             f64 = 0.05;
pub const REDUNDANCY_RADIUS:            f64 = 0.1;
pub const PHASE_DISTINCTION_THRESHOLD:  f64 = 0.3;

// Dimensional admission (LEY_FUNDACIONAL v1.2.0)
pub const LAMBDA_DIM_FIXED: f64 = 0.05;
pub const LAMBDA_DIM_LOG:   f64 = 0.01;
```

---

## 7. Error variants (error.rs)

```rust
#[error("Proof missing: mutation '{mutation_name}' executed without a valid Proof")]
ProofMissing { mutation_name: &'static str },

#[error("Proof invalid: hash or axiom {axiom_id} failed verification")]
ProofInvalid { axiom_id: u8 },

#[error("Invariant violated: axiom {axiom_id} failed pre-mutation check")]
InvariantViolation { axiom_id: u8 },
```

---

## 8. Tests (implemented)

```rust
#[test] fn valid_proof_passes()
#[test] fn tampered_witness_fails()
#[test] fn missing_axiom_fails()
#[test] fn failed_check_returns_error()
#[test] fn proof_is_fresh_within_window()
#[test] fn proof_expired_fails()
#[test] fn duality_consistency_fails_if_fisher_not_current()
#[test] fn expansion_required_includes_duality_and_admission()
#[test] fn axiom_set_bitmask_operations_correct()
#[test] fn structural_required_excludes_duality_and_admission()
```

---

## 9. Cargo.toml (genesis-types)

```toml
[dependencies]
thiserror         = "1.0"
static_assertions = "1.1"
blake3            = "=1.5.4"
arrayvec          = "0.7"
```

**`sha2` is not a dependency.** The proof system uses BLAKE3 exclusively.

---

*Last updated: 2026-03-05 | Implemented: `shared/genesis-types/src/proof.rs`*
