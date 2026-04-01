//! # Proof-Carrying Mutations
//!
//! Every structural self-modification must generate a valid [`Proof`] before
//! execution. This module implements the certification system that guarantees
//! no mutation violates axiomatic invariants.
//!
//! ## Usage flow
//!
//! ```rust
//! use genesis_types::proof::{WitnessBuilder, AxiomID, AxiomGuard};
//!
//! fn propose_structural_mutation() -> Result<genesis_types::proof::Proof, genesis_types::GenesisError> {
//!     let mut builder = WitnessBuilder::new();
//!     builder.check(AxiomID::MinkowskiSignature, || true)?;
//!     builder.check(AxiomID::CohomologyZero,    || true)?;
//!     builder.check(AxiomID::ProofGuard,        || true)?;
//!     Ok(builder.build(0))
//! }
//! ```
//!
//! AX-ID: `LEY_FUNDACIONAL` §5.5 (`ProofGuard`), `GENESIS_PROOF_SPEC` v1.1.0

#![allow(clippy::must_use_candidate)]

use arrayvec::ArrayVec;
use blake3;
use smallvec::SmallVec;

use crate::{error::GenesisError, signal::NodeId};

/// SSO witness buffer — stack-inline for witnesses ≤ 512 bytes (matches
/// `WitnessBuffer::Small` inline capacity, guaranteeing zero-alloc end-to-end),
/// heap spill only for larger witnesses.
///
/// # Why 512, not 64 (BN-03 revision, FIX-5)
///
/// `WitnessBuffer::Small` stores up to 512 bytes inline in `ArrayVec<u8, 512>`.
/// A `Proof::witness` of `SmallVec<[u8; 64]>` would force a heap allocation for
/// any witness in the range 65–512 bytes, silently breaking the zero-alloc contract
/// promised by `WitnessBuilder`. Unifying both inline capacities at 512 bytes eliminates
/// the allocation for all current (20–28 byte) and anticipated proof types.
///
/// Stack cost: 512 bytes per `Proof` struct (acceptable — Proofs are not persisted
/// in hot loops; they are produced, verified, and dropped per mutation).
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.2, BN-03
pub type Witness = SmallVec<[u8; 512]>;

/// Inline capacity of the `Witness` SSO buffer in bytes.
///
/// Used in `debug_assert!` in `WitnessBuilder::build()` to verify that the
/// assembled witness fits inline. If this fires, increase both this constant
/// and the SmallVec inline capacity in the `Witness` type alias above.
///
/// AX-ID: GENESIS_PROOF_SPEC §2.2, FIX-C
pub const WITNESS_INLINE_CAPACITY: usize = 512;

/// Canonical hash of a [`Proof`] (BLAKE3, 32 bytes).
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.2
pub type ProofHash = [u8; 32];

/// Inline capacity for causal premises before spilling to heap.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.5
pub const PREMISES_INLINE_CAPACITY: usize = 4;

/// Premise buffer with heap-free hot path for ≤4 entries.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.5
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Premises {
    /// Stack-only path for short causal histories.
    Small(ArrayVec<ProofHash, PREMISES_INLINE_CAPACITY>),
    /// Spills to heap when the graph requires more premises.
    Large(Vec<ProofHash>),
}

impl Default for Premises {
    fn default() -> Self {
        Self::new()
    }
}

impl Premises {
    /// Creates an empty container of causal premises.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.5
    pub fn new() -> Self {
        Self::Small(ArrayVec::new())
    }

    /// Returns the number of premises.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.5
    pub const fn len(&self) -> usize {
        match self {
            Self::Small(items) => items.len(),
            Self::Large(items) => items.len(),
        }
    }

    /// Returns `true` if there are no premises.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.5
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a causal premise to the end.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.5
    pub fn push(&mut self, premise: ProofHash) {
        match self {
            Self::Small(items) => {
                if items.is_full() {
                    let mut large = Vec::with_capacity(PREMISES_INLINE_CAPACITY + 1);
                    large.extend_from_slice(items.as_slice());
                    large.push(premise);
                    *self = Self::Large(large);
                } else {
                    items.push(premise);
                }
            }
            Self::Large(items) => items.push(premise),
        }
    }

    /// Iterates over the stored premises.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.5
    pub fn iter(&self) -> impl Iterator<Item = &ProofHash> {
        match self {
            Self::Small(items) => items.iter(),
            Self::Large(items) => items.iter(),
        }
    }
}

impl From<Vec<ProofHash>> for Premises {
    fn from(value: Vec<ProofHash>) -> Self {
        if value.len() <= PREMISES_INLINE_CAPACITY {
            let mut small = ArrayVec::<ProofHash, PREMISES_INLINE_CAPACITY>::new();
            small.extend(value);
            Self::Small(small)
        } else {
            Self::Large(value)
        }
    }
}

/// Optional historical-consistency metadata for a [`Proof`].
///
/// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofMeta {
    /// Node that originated the certified mutation or assertion.
    pub origin_node: NodeId,
    /// Logical target domain (mutation or assertion), stable hash.
    pub target_domain_hash: ProofHash,
    /// Hash of the resulting state/assertion after applying the proof.
    pub resulting_state_hash: ProofHash,
}

// ============================================================================
// AxiomID
// ============================================================================

/// Identifiers of axioms verifiable in structural mutations.
///
/// The value `repr(u8)` is stable — changing the discriminants breaks the
/// format of the serialized witness and all existing [`Proof`]s.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AxiomID {
    /// Preserved Minkowski signature (+,-,-,-) — `LEY_FUNDACIONAL` §5.1
    MinkowskiSignature = 0,
    /// H¹(M,F) = 0 — `LEY_FUNDACIONAL` §5.2
    CohomologyZero = 1,
    /// λ₂ ≥ 0.1 — `LEY_FUNDACIONAL` §5.3
    AlgebraicConnectivity = 2,
    /// Immutable `COGNITIVE_PLANCK_CONSTANT` — `LEY_FUNDACIONAL` §5.4
    PlanckConstant = 3,
    /// Every mutation has a Proof — `LEY_FUNDACIONAL` §5.5
    ProofGuard = 4,
    /// Dimensional expansion updates Fisher metric — `LEY_FUNDACIONAL` §5.6
    DualityConsistency = 5,
    /// `ΔH_total` < `C_dim(N)` before creating a new dimension — `LEY_FUNDACIONAL` §5.7
    DimensionalAdmission = 6,
}

/// Compact mask of verified/required axioms.
///
/// Bit `i` represents the presence of `AxiomID = i` for `0..=6`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AxiomSet(pub u8);

impl AxiomSet {
    const VALID_BITS: u8 = 0b0111_1111;

    /// Returns the empty set (`0b0000_0000`).
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Builds a singleton set with one active axiom.
    pub const fn from_axiom(axiom: AxiomID) -> Self {
        Self(1u8 << (axiom as u8))
    }

    /// Constructs an array from a list of [`AxiomID`].
    pub fn from_slice(axioms: &[AxiomID]) -> Self {
        axioms.iter().fold(Self::empty(), |mut set, &axiom| {
            set.insert(axiom);
            set
        })
    }

    /// Returns the internal bit mask.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Inserts an axiom into the set.
    #[allow(clippy::missing_const_for_fn)] // &mut self not const-stable on MSRV 1.75
    pub fn insert(&mut self, axiom: AxiomID) {
        self.0 |= Self::from_axiom(axiom).0;
    }

    /// Returns `true` if the axiom is present in the set.
    pub const fn contains(self, axiom: AxiomID) -> bool {
        (self.0 & Self::from_axiom(axiom).0) != 0
    }

    /// Returns `true` if `self` is subset of `other`.
    pub const fn is_subset_of(self, other: Self) -> bool {
        (self.0 & other.0) == self.0
    }

    /// Returns `true` if `self` is superset of `other`.
    pub const fn is_superset_of(self, other: Self) -> bool {
        other.is_subset_of(self)
    }

    /// Returns the union between `self` and `other`.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self((self.0 | other.0) & Self::VALID_BITS)
    }

    /// Returns `true` if `self` and `other` share at least one axiom.
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Returns the set difference `self - other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self((self.0 & !other.0) & Self::VALID_BITS)
    }

    /// Returns `true` if all the bits belong to the valid range `0..=6`.
    pub const fn is_valid(self) -> bool {
        (self.0 & !Self::VALID_BITS) == 0
    }
}

impl AxiomID {
    /// Invariants required for standard structural mutations.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.1
    pub const STRUCTURAL_REQUIRED: &'static [Self] = &[
        Self::MinkowskiSignature,
        Self::CohomologyZero,
        Self::AlgebraicConnectivity,
        Self::PlanckConstant,
        Self::ProofGuard,
    ];

    /// Invariants required for dimensional expansion.
    /// Includes Fisher duality verification and energy admission.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.1
    pub const EXPANSION_REQUIRED: &'static [Self] = &[
        Self::MinkowskiSignature,
        Self::CohomologyZero,
        Self::AlgebraicConnectivity,
        Self::PlanckConstant,
        Self::ProofGuard,
        Self::DualityConsistency,
        Self::DimensionalAdmission,
    ];
}

// ============================================================================
// Proof
// ============================================================================

/// Cryptographic certificate proving that a mutation preserves axiomatic invariants.
/// The `hash` field is BLAKE3 over `witness`.
///
/// A [`Proof`] is valid if and only if:
/// 1. `blake3(witness) == hash` (integrity)
/// 2. The serialized witness contains a valid frame for each axiom in
///    `axioms_checked` with `result == 1` (positive verification)
///
/// # Memory model (BN-03)
/// `witness` uses SSO via [`Witness`] = `SmallVec<[u8; 512]>`, so the inline
/// threshold is [`WITNESS_INLINE_CAPACITY`] bytes (currently 512).
/// This keeps witnesses of length `<= WITNESS_INLINE_CAPACITY` on stack and
/// spills to heap only beyond that bound, eliminating allocator cost on the
/// common proof path.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.2
#[derive(Debug, Clone)]
pub struct Proof {
    /// List of axioms positively verified in this proof.
    pub axioms_checked: AxiomSet,
    /// Serialized binary verification trace.
    /// Stored as [`Witness`] (`SmallVec<[u8; 512]>`) with inline capacity
    /// [`WITNESS_INLINE_CAPACITY`] (512 bytes).
    /// Frame format: [`axiom_id`: u8, `result`: u8, `ctx_len`: u16le, `ctx`: [u8; `ctx_len`]].
    pub witness: Witness,
    /// Generation timestamp in nanoseconds.
    pub timestamp: u64,
    /// BLAKE3 of the field `witness`. Detects any post-generation modification.
    pub hash: ProofHash,
    /// Optional reference to the parent proof in the causal chain.
    pub parent_proof_hash: Option<ProofHash>,
    /// Explicit causal premises (DAG) of this proof.
    pub premises: Premises,
    /// Optional metaconsistency layer (origin, domain, resulting state).
    pub meta: Option<ProofMeta>,
}

impl Proof {
    /// Builds a [`Proof`] computing the BLAKE3 hash of the witness (SSO, zero malloc).
    pub fn new(axioms: AxiomSet, witness: Witness, timestamp: u64) -> Self {
        let hash = blake3_hash(&witness);
        Self {
            axioms_checked: axioms,
            witness,
            timestamp,
            hash,
            parent_proof_hash: None,
            premises: Premises::new(),
            meta: None,
        }
    }

    /// Verifies that the stored hash matches BLAKE3(witness).
    /// Detects modification of the witness post-generation.
    pub fn is_internally_consistent(&self) -> bool {
        blake3_hash(&self.witness) == self.hash
    }

    /// Verifies that the Proof has not expired (anti-replay).
    ///
    /// A proof is fresh if `current_ns >= timestamp` (not from the future)
    /// and `current_ns - timestamp < PROOF_MAX_AGE_NS` (within the window).
    ///
    /// Correction [A1-1]: `saturating_sub` accepted future timestamps
    /// (`current_ns` < timestamp → 0 < `PROOF_MAX_AGE_NS` → always fresh).
    /// The new implementation explicitly rejects proofs from the future.
    pub const fn is_fresh(&self, current_ns: u64) -> bool {
        use crate::constants::PROOF_MAX_AGE_NS;
        current_ns >= self.timestamp && current_ns - self.timestamp < PROOF_MAX_AGE_NS
    }

    /// Attaches optional causal relations without altering the base hash.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.5
    #[must_use]
    pub fn with_causal_links(
        mut self,
        parent_proof_hash: Option<ProofHash>,
        premises: Premises,
    ) -> Self {
        self.parent_proof_hash = parent_proof_hash;
        self.premises = premises;
        self
    }

    /// Attaches domain metadata for historical validation.
    ///
    /// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
    #[must_use]
    pub const fn with_meta(mut self, meta: ProofMeta) -> Self {
        self.meta = Some(meta);
        self
    }
}

fn blake3_hash(data: &[u8]) -> ProofHash {
    *blake3::hash(data).as_bytes()
}

/// Result of a metaconsistency validation.
///
/// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
#[derive(Debug, Clone, PartialEq)]
pub enum MetaConsistencyError {
    /// Base validation failed (hash+witness+axioms) for the existing layer.
    Base(GenesisError),
    /// The causal graph contains a loop, violating the DAG structure.
    CausalCycle,
    /// Historical conflict exists with another valid proof.
    Conflict {
        /// Hash of the candidate proof attempting to enter history.
        candidate: ProofHash,
        /// Hash of the historical proof already accepted and colliding.
        existing: ProofHash,
    },
}

/// Pluggable rule for detecting historical conflicts.
///
/// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
pub trait ConflictRule {
    /// Returns `true` if `candidate` and `existing` are mutually incompatible.
    fn conflicts(&self, candidate: &Proof, existing: &Proof) -> bool;
}

/// Base conflict heuristic: same node + same domain + different state.
///
/// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeDomainStateConflictRule;

impl ConflictRule for NodeDomainStateConflictRule {
    fn conflicts(&self, candidate: &Proof, existing: &Proof) -> bool {
        match (&candidate.meta, &existing.meta) {
            (Some(c), Some(e)) => {
                c.origin_node == e.origin_node
                    && c.target_domain_hash == e.target_domain_hash
                    && c.resulting_state_hash != e.resulting_state_hash
            }
            _ => false,
        }
    }
}

/// Separate historical-validation layer over internally valid proofs.
///
/// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
#[derive(Debug, Clone)]
pub struct MetaConsistencyValidator<R: ConflictRule = NodeDomainStateConflictRule> {
    rule: R,
}

impl<R: ConflictRule> MetaConsistencyValidator<R> {
    /// Builds a validator with a pluggable conflict rule.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.5
    pub const fn new(rule: R) -> Self {
        Self { rule }
    }

    /// Executes historical validation: DAG integrity + conflict checks against history.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `MetaConsistencyError::CausalCycle` if the causal proof graph contains a cycle.
    /// - `MetaConsistencyError::Conflict` if the candidate proof conflicts with an existing proof.
    ///
    /// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
    pub fn validate_against_history(
        &self,
        candidate: &Proof,
        history: &[&Proof],
    ) -> Result<(), MetaConsistencyError> {
        Self::ensure_dag(candidate, history)?;

        for existing in history {
            if self.rule.conflicts(candidate, existing) {
                return Err(MetaConsistencyError::Conflict {
                    candidate: candidate.hash,
                    existing: existing.hash,
                });
            }
        }

        Ok(())
    }

    fn ensure_dag(candidate: &Proof, history: &[&Proof]) -> Result<(), MetaConsistencyError> {
        let mut nodes: Vec<(ProofHash, &Proof)> = Vec::with_capacity(history.len() + 1);
        for proof in history {
            nodes.push((proof.hash, *proof));
        }
        nodes.push((candidate.hash, candidate));
        nodes.sort_by_key(|(hash, _)| *hash);

        let mut colors = vec![0u8; nodes.len()];
        for idx in 0..nodes.len() {
            if colors[idx] == 0 && Self::dfs_cycle(idx, &nodes, &mut colors) {
                return Err(MetaConsistencyError::CausalCycle);
            }
        }

        Ok(())
    }

    fn dfs_cycle(idx: usize, nodes: &[(ProofHash, &Proof)], colors: &mut [u8]) -> bool {
        colors[idx] = 1;
        let (_, proof) = nodes[idx];

        if let Some(parent) = proof.parent_proof_hash {
            if let Ok(next_idx) = nodes.binary_search_by_key(&parent, |(hash, _)| *hash) {
                if colors[next_idx] == 1
                    || (colors[next_idx] == 0 && Self::dfs_cycle(next_idx, nodes, colors))
                {
                    return true;
                }
            }
        }

        for premise in proof.premises.iter() {
            if let Ok(next_idx) = nodes.binary_search_by_key(premise, |(hash, _)| *hash) {
                if colors[next_idx] == 1
                    || (colors[next_idx] == 0 && Self::dfs_cycle(next_idx, nodes, colors))
                {
                    return true;
                }
            }
        }

        colors[idx] = 2;
        false
    }
}

impl Default for MetaConsistencyValidator<NodeDomainStateConflictRule> {
    fn default() -> Self {
        Self {
            rule: NodeDomainStateConflictRule,
        }
    }
}

// ============================================================================
// VerifiedProof token — compile-time guarantee
// ============================================================================

/// Verified-proof token.
/// **Not constructible outside [`AxiomGuard::verify_for`] or [`AxiomGuard::verify_for_result`].**
///
/// Compile-time guarantee: [`Mutation::apply`] can be called only if
/// `AxiomGuard::verify_for` returns `Some(token)` or
/// `AxiomGuard::verify_for_result` returns `Ok(token)` — that is, if the [`Proof`]
/// was successfully verified. There is no way to construct this token without
/// passing through verification.
///
/// The `PhantomData<&'mutation M>` links the token to the type of mutation and
/// a specific lifetime, preventing reuse across mutations.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.3
pub struct VerifiedProof<'mutation, M: Mutation> {
    _mutation: core::marker::PhantomData<&'mutation M>,
}

// ============================================================================
// Mutation trait
// ============================================================================

/// Trait that every structural mutation must implement.
///
/// Contract: `propose()` verifies invariants and generates a [`Proof`];
/// `apply()` executes the mutation only with a [`VerifiedProof`] issued by
/// [`AxiomGuard::verify_for`] or [`AxiomGuard::verify_for_result`] —
/// compile-time verification guarantee.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.3
pub trait Mutation: Send + Sync {
    /// Verifies axiomatic invariants and generates a certified [`Proof`].
    /// Returns error if some invariant fails.
    /// # Errors
    /// Returns an error if the implementation cannot build a valid proof token.
    fn propose(&self) -> Result<Proof, crate::error::GenesisError>;

    /// Applies the mutation. REQUIRES token of verification issued by
    /// [`AxiomGuard::verify_for`] or [`AxiomGuard::verify_for_result`] —
    /// compile-time guarantee of proof verification via either entry point.
    ///
    /// `where Self: Sized` is required because `VerifiedProof<'_, Self>`
    /// is parameterized over `Self`; trait objects (`dyn Mutation`) do not
    /// require `apply` because all concrete mutations are `Sized` types.
    /// # Errors
    /// Returns an error if applying the verified proof fails for the target state.
    fn apply(&self, _token: VerifiedProof<'_, Self>) -> Result<(), crate::error::GenesisError>
    where
        Self: Sized;

    /// Static mutation name used in error messages.
    fn name(&self) -> &'static str;
}

// ============================================================================
// AxiomGuard
// ============================================================================

/// Verifier for [`Proof`] certificates.
///
/// Performs two independent checks:
/// 1. Cryptographic integrity: `blake3(witness) == proof.hash`
/// 2. Witness replay: each witness frame is deserialized and verified, ensuring
///    all claimed axioms appear with result `1`.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §2.4
pub struct AxiomGuard;

impl AxiomGuard {
    fn verify_with_error(
        proof: &Proof,
        required: &[AxiomID],
    ) -> Result<(), crate::error::GenesisError> {
        if !proof.is_internally_consistent() {
            return Err(crate::error::GenesisError::ProofInvalid { axiom_id: u8::MAX });
        }

        if !proof.axioms_checked.is_valid() {
            return Err(crate::error::GenesisError::ProofInvalid { axiom_id: u8::MAX });
        }

        let required_mask = AxiomSet::from_slice(required);
        if !proof.axioms_checked.is_superset_of(required_mask) {
            let missing = required_mask.difference(proof.axioms_checked).bits();
            let axiom_id = missing.trailing_zeros() as u8;
            return Err(crate::error::GenesisError::ProofInvalid { axiom_id });
        }

        if !Self::replay_witness(&proof.witness, proof.axioms_checked) {
            return Err(crate::error::GenesisError::ProofInvalid { axiom_id: u8::MAX });
        }

        Ok(())
    }

    /// Verifies that `proof` covers all the axioms in `required`.
    ///
    /// Returns `true` only if:
    /// - The witness hash is correct.
    /// - Each axiom in `required` appears in the witness with positive result.
    pub fn verify(proof: &Proof, required: &[AxiomID]) -> bool {
        Self::verify_with_error(proof, required).is_ok()
    }

    /// Verifies a proof and emits a typed token.
    /// **This is the only legitimate path to `apply()`.**
    ///
    /// Returns `Some(token)` if the proof is valid for `required`.
    /// The token binds the type of mutation `M` via `PhantomData`.
    ///
    /// Typical usage:
    /// ```rust
    /// use genesis_types::proof::{AxiomGuard, AxiomID, Mutation, Proof, VerifiedProof, WitnessBuilder};
    ///
    /// # struct DummyMutation;
    /// # impl Mutation for DummyMutation {
    /// #     fn propose(&self) -> Result<Proof, genesis_types::GenesisError> {
    /// #         let mut builder = WitnessBuilder::new();
    /// #         builder.check(AxiomID::MinkowskiSignature, || true)?;
    /// #         Ok(builder.build(0))
    /// #     }
    /// #
    /// #     fn apply(&self, _token: VerifiedProof<'_, Self>) -> Result<(), genesis_types::GenesisError> {
    /// #         Ok(())
    /// #     }
    /// #
    /// #     fn name(&self) -> &'static str {
    /// #         "dummy"
    /// #     }
    /// # }
    /// let mutation = DummyMutation;
    /// let proof = mutation.propose().expect("proof must be buildable");
    ///
    /// // Contract: verify_for emits a typed token when the proof satisfies the required axiom.
    /// let token = AxiomGuard::verify_for::<DummyMutation>(&proof, &[AxiomID::MinkowskiSignature]);
    /// assert!(token.is_some());
    /// ```
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.4
    pub fn verify_for<'p, M: Mutation>(
        proof: &'p Proof,
        required: &[AxiomID],
    ) -> Option<VerifiedProof<'p, M>> {
        Self::verify_for_result::<M>(proof, required).ok()
    }

    /// Verifies a proof and emits a typed token with structured diagnostics.
    ///
    /// Returns [`crate::error::GenesisError::ProofInvalid`] when the proof fails
    /// integrity, axiom coverage, or witness replay.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::GenesisError::ProofInvalid`] in any of the
    /// following contract violations:
    ///
    /// - **Witness integrity violation**: the BLAKE3 hash of the witness
    ///   does not match `proof.hash`, indicating post-issuance alteration.
    /// - **Declared-mutation mismatch**: at least one required axiom in
    ///   `required` is missing from `proof.axioms_checked`, so the mutation
    ///   does not satisfy the minimum application contract.
    /// - **Axiomatic replay failure**: the witness cannot be deserialized
    ///   consistently (truncated frames, invalid IDs, or negative results),
    ///   breaking traceability of the axioms claimed by the proof.
    ///
    /// AX-ID: `GENESIS_PROOF_SPEC` §2.4
    pub fn verify_for_result<'p, M: Mutation>(
        proof: &'p Proof,
        required: &[AxiomID],
    ) -> Result<VerifiedProof<'p, M>, crate::error::GenesisError> {
        Self::verify_with_error(proof, required)?;
        Ok(VerifiedProof {
            _mutation: core::marker::PhantomData,
        })
    }

    /// Verifies the base layer and then applies historical metaconsistency.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - `MetaConsistencyError::Base` if the base proof validation fails.
    /// - Any error produced by the meta-consistency validator
    ///   (causal cycle or historical conflict).
    ///
    /// AX-ID: AXIOMA-009, `GENESIS_PROOF_SPEC` §2.5
    pub fn verify_with_meta<R: ConflictRule>(
        proof: &Proof,
        required: &[AxiomID],
        history: &[&Proof],
        validator: &MetaConsistencyValidator<R>,
    ) -> Result<(), MetaConsistencyError> {
        Self::verify_with_error(proof, required).map_err(MetaConsistencyError::Base)?;
        validator.validate_against_history(proof, history)
    }

    /// Deserializes witness frames and verifies that all claimed axioms are
    /// present with `result == 1`.
    ///
    /// Frame format: [`axiom_id`: u8][result: u8][`ctx_len`: u16le][ctx: bytes]
    fn replay_witness(witness: &[u8], claimed: AxiomSet) -> bool {
        let mut cursor = 0usize;
        let mut verified = AxiomSet::empty();

        while cursor < witness.len() {
            let len = witness.len();
            if cursor + 4 > len {
                return false;
            }

            let axiom_raw = witness[cursor];
            let result = witness[cursor + 1];
            let ctx_len = u16::from_le_bytes([witness[cursor + 2], witness[cursor + 3]]) as usize;
            cursor += 4 + ctx_len;

            if cursor > len || result != 1 {
                return false;
            }

            let axiom = match axiom_raw {
                0..=6 => unsafe { core::mem::transmute::<u8, AxiomID>(axiom_raw) },
                _ => return false,
            };

            verified.insert(axiom);
        }

        // All the axioms claimed must be in the witnessed witness
        (claimed.bits() & verified.bits()) == claimed.bits()
    }
}

// ============================================================================
// WitnessBuilder
// ============================================================================

/// Builds a binary witness by executing axiomatic checks in sequence.
///
/// Each call to [`check`][WitnessBuilder::check] serializes one frame
/// `[axiom_id, result, 0, 0]` into the internal witness buffer. If a check fails,
/// it returns an error immediately.
///
/// AX-ID: `GENESIS_PROOF_SPEC` §3
pub struct WitnessBuilder {
    frames: WitnessBuffer,
    axioms: AxiomSet,
}

#[allow(clippy::large_enum_variant)]
enum WitnessBuffer {
    Small(ArrayVec<u8, 512>),
    Large(Vec<u8>),
}

impl WitnessBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            frames: WitnessBuffer::Small(ArrayVec::new()),
            axioms: AxiomSet::empty(),
        }
    }

    fn push_byte(&mut self, byte: u8) {
        self.ensure_capacity(1);
        match &mut self.frames {
            WitnessBuffer::Small(frames) => frames.push(byte),
            WitnessBuffer::Large(frames) => frames.push(byte),
        }
    }

    fn extend_bytes(&mut self, bytes: &[u8]) {
        self.ensure_capacity(bytes.len());
        match &mut self.frames {
            WitnessBuffer::Small(frames) => {
                if frames.try_extend_from_slice(bytes).is_err() {
                    unreachable!("capacity pre-validated by ensure_capacity")
                }
            }
            WitnessBuffer::Large(frames) => frames.extend_from_slice(bytes),
        }
    }

    fn ensure_capacity(&mut self, additional: usize) {
        let needs_upgrade =
            matches!(&self.frames, WitnessBuffer::Small(frames) if frames.len() + additional > 512);
        if needs_upgrade {
            let old = core::mem::replace(&mut self.frames, WitnessBuffer::Large(Vec::new()));
            if let WitnessBuffer::Small(frames) = old {
                let mut large = Vec::with_capacity(frames.len() + additional);
                large.extend_from_slice(&frames);
                self.frames = WitnessBuffer::Large(large);
            }
        }
    }

    /// Executes check `f` for `axiom` and serializes the frame into the witness.
    ///
    /// If `f()` returns `false`, serializes `result=0` and returns
    /// [`GenesisError::InvariantViolation`][crate::GenesisError::InvariantViolation].
    /// # Errors
    /// Returns `GenesisError::InvariantViolation` when the evaluated axiom check fails.
    pub fn check<F: FnOnce() -> bool>(
        &mut self,
        axiom: AxiomID,
        f: F,
    ) -> Result<(), crate::error::GenesisError> {
        let ok = f();
        // Serialize frame: [axiom_id: u8, result: u8, ctx_len: u16le = 0]
        self.push_byte(axiom as u8);
        self.push_byte(u8::from(ok));
        self.extend_bytes(&0u16.to_le_bytes());

        if !ok {
            return Err(crate::error::GenesisError::InvariantViolation {
                axiom_id: axiom as u8,
            });
        }

        self.axioms.insert(axiom);
        Ok(())
    }

    /// Finalizes the builder and returns a [`Proof`] with the given timestamp.
    ///
    /// # Allocation contract (BN-03 + FIX-5)
    /// For witnesses ≤ 512 bytes: zero heap allocation (inline SmallVec matches
    /// WitnessBuffer::Small inline capacity — no copy crosses stack/heap boundary).
    /// For witnesses > 512 bytes: single heap allocation (SmallVec spill, same
    /// as the builder's WitnessBuffer::Large path).
    pub fn build(self, timestamp: u64) -> Proof {
        let witness: Witness = match self.frames {
            WitnessBuffer::Small(frames) => {
                // FIX-C: Use WITNESS_INLINE_CAPACITY constant — was hardcoded 64 but
                // Witness is now SmallVec<[u8; 512]> to match WitnessBuffer::Small inline capacity.
                debug_assert!(
                    frames.len() <= WITNESS_INLINE_CAPACITY,
                    "Witness exceeds SSO inline capacity ({} > {} bytes): \
                     will heap-allocate in Proof. Increase WITNESS_INLINE_CAPACITY.",
                    frames.len(),
                    WITNESS_INLINE_CAPACITY
                );
                SmallVec::from_slice(&frames)
            }
            WitnessBuffer::Large(vec) => {
                // Take ownership of the existing heap Vec — zero-copy spill.
                SmallVec::from_vec(vec)
            }
        };
        Proof::new(self.axioms, witness, timestamp)
    }
}

impl Default for WitnessBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use core::alloc::{GlobalAlloc, Layout};
    use core::cell::Cell;

    use super::*;

    struct CountingAllocator;

    thread_local! {
        static ALLOC_RECORDING: Cell<bool> = const { Cell::new(false) };
        static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
        static ALLOC_GUARD_DEPTH: Cell<usize> = const { Cell::new(0) };
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

    // SAFETY: Delegates to the system allocator while counting allocations for the current test thread.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_RECORDING.with(|recording| {
                if recording.get() {
                    ALLOC_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            std::alloc::System.alloc(layout)
        }

        // SAFETY: delegates directly to GlobalAlloc::dealloc with matching ptr and layout.
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            std::alloc::System.dealloc(ptr, layout)
        }

        // SAFETY: delegates directly to GlobalAlloc::realloc; ptr was allocated by this allocator.
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOC_RECORDING.with(|recording| {
                if recording.get() {
                    ALLOC_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            std::alloc::System.realloc(ptr, layout, new_size)
        }
    }

    struct AllocationScope {
        start: usize,
    }

    impl AllocationScope {
        fn begin() -> Self {
            ALLOC_GUARD_DEPTH.with(|depth| {
                assert_eq!(
                    depth.get(),
                    0,
                    "nested allocation scopes in the same thread are unsupported"
                );
                depth.set(1);
            });

            ALLOC_COUNT.with(|count| count.set(0));
            ALLOC_RECORDING.with(|recording| recording.set(true));
            let start = ALLOC_COUNT.with(Cell::get);

            Self { start }
        }

        fn delta(&self) -> usize {
            ALLOC_COUNT.with(Cell::get).saturating_sub(self.start)
        }
    }

    impl Drop for AllocationScope {
        fn drop(&mut self) {
            ALLOC_RECORDING.with(|recording| recording.set(false));
            ALLOC_GUARD_DEPTH.with(|depth| depth.set(0));
        }
    }

    // ── Helper ─────────────────────────────────────────────────────────────

    fn build_proof(checks: &[(AxiomID, bool)]) -> Result<Proof, crate::GenesisError> {
        let mut builder = WitnessBuilder::new();
        for &(axiom, result) in checks {
            builder.check(axiom, move || result)?;
        }
        Ok(builder.build(0))
    }

    // ── Proof generation ───────────────────────────────────────────────────

    #[test]
    fn valid_proof_passes() {
        let proof = build_proof(&[(AxiomID::MinkowskiSignature, true)]).unwrap();
        assert!(AxiomGuard::verify(&proof, &[AxiomID::MinkowskiSignature]));
    }

    #[test]
    fn verify_for_result_returns_token_for_valid_proof() {
        struct DummyMutation;

        impl Mutation for DummyMutation {
            fn propose(&self) -> Result<Proof, crate::error::GenesisError> {
                build_proof(&[(AxiomID::MinkowskiSignature, true)])
            }

            fn apply(
                &self,
                _token: VerifiedProof<'_, Self>,
            ) -> Result<(), crate::error::GenesisError>
            where
                Self: Sized,
            {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "dummy"
            }
        }

        let mutation = DummyMutation;
        let proof = mutation.propose().unwrap();
        let token =
            AxiomGuard::verify_for_result::<DummyMutation>(&proof, &[AxiomID::MinkowskiSignature])
                .expect("proof should verify");
        assert!(mutation.apply(token).is_ok());
    }

    #[test]
    fn verify_for_result_returns_proof_invalid_for_missing_axiom() {
        struct MissingMutation;

        impl Mutation for MissingMutation {
            fn propose(&self) -> Result<Proof, crate::error::GenesisError> {
                build_proof(&[(AxiomID::MinkowskiSignature, true)])
            }

            fn apply(
                &self,
                _token: VerifiedProof<'_, Self>,
            ) -> Result<(), crate::error::GenesisError>
            where
                Self: Sized,
            {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "missing"
            }
        }

        let proof = build_proof(&[(AxiomID::MinkowskiSignature, true)]).unwrap();
        let Err(err) =
            AxiomGuard::verify_for_result::<MissingMutation>(&proof, &[AxiomID::CohomologyZero])
        else {
            panic!("missing required axiom must fail")
        };
        assert_eq!(
            err,
            crate::error::GenesisError::ProofInvalid { axiom_id: 1 }
        );
    }

    #[test]
    fn tampered_witness_fails() {
        let mut proof = build_proof(&[(AxiomID::MinkowskiSignature, true)]).unwrap();
        // Flip one bit in the witness → hash mismatch
        proof.witness[0] ^= 0xFF;
        assert!(!AxiomGuard::verify(&proof, &[AxiomID::MinkowskiSignature]));
    }

    #[test]
    fn tampered_hash_fails() {
        let mut proof = build_proof(&[(AxiomID::MinkowskiSignature, true)]).unwrap();
        proof.hash[0] ^= 0xFF;
        assert!(!AxiomGuard::verify(&proof, &[AxiomID::MinkowskiSignature]));
    }

    #[test]
    fn missing_axiom_in_required_fails() {
        // Proof only checks MinkowskiSignature, but CohomologyZero is required
        let proof = build_proof(&[(AxiomID::MinkowskiSignature, true)]).unwrap();
        assert!(!AxiomGuard::verify(&proof, &[AxiomID::CohomologyZero]));
    }

    #[test]
    fn failed_check_returns_invariant_violation_error() {
        let result = build_proof(&[(AxiomID::MinkowskiSignature, false)]);
        match result {
            Err(crate::GenesisError::InvariantViolation { axiom_id }) => {
                assert_eq!(axiom_id, AxiomID::MinkowskiSignature as u8);
            }
            other => panic!("Expected InvariantViolation, got {:?}", other),
        }
    }

    #[test]
    fn proof_is_fresh_within_window() {
        use crate::constants::PROOF_MAX_AGE_NS;
        let proof = build_proof(&[(AxiomID::PlanckConstant, true)]).unwrap();
        // timestamp=0, current=PROOF_MAX_AGE_NS - 1 → within the freshness window
        assert!(proof.is_fresh(PROOF_MAX_AGE_NS - 1));
    }

    #[test]
    fn proof_from_future_is_not_fresh() {
        // current_ns < timestamp must return false.
        // Simulate a future timestamp: create proof with timestamp=100, verify at t=50.
        let mut future_proof = build_proof(&[(AxiomID::PlanckConstant, true)]).unwrap();
        future_proof.timestamp = 100;
        assert!(
            !future_proof.is_fresh(50),
            "proof from the future must be non-fresh"
        );
        // current_ns=0 < timestamp=100 → also non-fresh.
        assert!(
            !future_proof.is_fresh(0),
            "proof from the future must be non-fresh at t=0"
        );
    }

    #[test]
    fn proof_expired_fails_freshness() {
        use crate::constants::PROOF_MAX_AGE_NS;
        let proof = build_proof(&[(AxiomID::PlanckConstant, true)]).unwrap();
        // current = PROOF_MAX_AGE_NS → age == PROOF_MAX_AGE_NS → not fresh
        assert!(!proof.is_fresh(PROOF_MAX_AGE_NS));
    }

    // ── AxiomID sets ────────────────────────────────────────────────────────

    #[test]
    fn structural_required_does_not_contain_duality_or_admission() {
        let structural = AxiomSet::from_slice(AxiomID::STRUCTURAL_REQUIRED);
        assert!(!structural.contains(AxiomID::DualityConsistency));
        assert!(!structural.contains(AxiomID::DimensionalAdmission));
    }

    #[test]
    fn expansion_required_contains_duality_and_admission() {
        let expansion = AxiomSet::from_slice(AxiomID::EXPANSION_REQUIRED);
        assert!(expansion.contains(AxiomID::DualityConsistency));
        assert!(expansion.contains(AxiomID::DimensionalAdmission));
    }

    #[test]
    fn expansion_required_is_superset_of_structural_required() {
        let structural = AxiomSet::from_slice(AxiomID::STRUCTURAL_REQUIRED);
        let expansion = AxiomSet::from_slice(AxiomID::EXPANSION_REQUIRED);
        assert!(expansion.is_superset_of(structural));
    }

    #[test]
    fn axiom_set_algebra_subset_superset_equality() {
        let structural = AxiomSet::from_slice(AxiomID::STRUCTURAL_REQUIRED);
        let expansion = AxiomSet::from_slice(AxiomID::EXPANSION_REQUIRED);
        let structural_clone = AxiomSet::from_slice(AxiomID::STRUCTURAL_REQUIRED);

        assert!(structural.is_subset_of(expansion));
        assert!(expansion.is_superset_of(structural));
        assert_eq!(structural, structural_clone);
    }

    #[test]
    fn axiom_set_const_ops_compile_time() {
        const STRUCTURAL: AxiomSet = AxiomSet::from_axiom(AxiomID::MinkowskiSignature)
            .union(AxiomSet::from_axiom(AxiomID::CohomologyZero))
            .union(AxiomSet::from_axiom(AxiomID::AlgebraicConnectivity))
            .union(AxiomSet::from_axiom(AxiomID::PlanckConstant))
            .union(AxiomSet::from_axiom(AxiomID::ProofGuard));
        const EXPANSION: AxiomSet = STRUCTURAL
            .union(AxiomSet::from_axiom(AxiomID::DualityConsistency))
            .union(AxiomSet::from_axiom(AxiomID::DimensionalAdmission));
        const ONLY_EXPANSION: AxiomSet = EXPANSION.difference(STRUCTURAL);
        const BOTH: AxiomSet = STRUCTURAL.union(EXPANSION);
        assert_eq!(BOTH.bits(), EXPANSION.bits());
        assert_eq!(ONLY_EXPANSION.bits(), 0b0110_0000);
        assert!(STRUCTURAL.intersects(EXPANSION));
    }

    #[test]
    fn axiom_set_const_difference_masks_invalid_bits() {
        const RAW: AxiomSet = AxiomSet(0b1111_1111);
        const REMOVE_HIGH_BIT: AxiomSet = AxiomSet(0b1000_0000);
        const DIFF: AxiomSet = RAW.difference(REMOVE_HIGH_BIT);

        assert_eq!(DIFF.bits(), 0b0111_1111);
        assert!(DIFF.is_valid());
    }

    // ── DualityConsistency specific (GENESIS_PROOF_SPEC §9 new tests) ───────

    #[test]
    fn duality_consistency_fails_if_check_returns_false() {
        let mut builder = WitnessBuilder::new();
        let result = builder.check(AxiomID::DualityConsistency, || false);
        assert!(
            matches!(
                result,
                Err(crate::GenesisError::InvariantViolation { axiom_id: 5 })
            ),
            "Expected InvariantViolation {{ axiom_id: 5 }}, got {:?}",
            result
        );
    }

    #[test]
    fn multi_axiom_proof_all_required_present() {
        let proof = build_proof(&[
            (AxiomID::MinkowskiSignature, true),
            (AxiomID::CohomologyZero, true),
            (AxiomID::AlgebraicConnectivity, true),
            (AxiomID::PlanckConstant, true),
            (AxiomID::ProofGuard, true),
        ])
        .unwrap();
        assert!(AxiomGuard::verify(&proof, AxiomID::STRUCTURAL_REQUIRED));
    }

    #[test]
    fn expansion_proof_all_required_present() {
        let proof = build_proof(&[
            (AxiomID::MinkowskiSignature, true),
            (AxiomID::CohomologyZero, true),
            (AxiomID::AlgebraicConnectivity, true),
            (AxiomID::PlanckConstant, true),
            (AxiomID::ProofGuard, true),
            (AxiomID::DualityConsistency, true),
            (AxiomID::DimensionalAdmission, true),
        ])
        .unwrap();
        assert!(AxiomGuard::verify(&proof, AxiomID::EXPANSION_REQUIRED));
    }

    // ── Witness format ──────────────────────────────────────────────────────

    #[test]
    fn witness_frame_layout_is_4_bytes_per_check() {
        let proof = build_proof(&[
            (AxiomID::MinkowskiSignature, true),
            (AxiomID::CohomologyZero, true),
        ])
        .unwrap();
        // 2 checks × 4 bytes each (no ctx)
        assert_eq!(proof.witness.len(), 8);
    }

    #[test]
    fn witness_builder_small_witness_zero_alloc_256_bytes() {
        let mut builder = WitnessBuilder::new();

        let scope = AllocationScope::begin();
        for _ in 0..64 {
            builder.check(AxiomID::MinkowskiSignature, || true).unwrap();
        }

        assert_eq!(
            scope.delta(),
            0,
            "WitnessBuilder hot path must not allocate while writing <=256 bytes"
        );
    }

    #[test]
    fn witness_builder_upgrades_to_large_after_512() {
        let mut builder = WitnessBuilder::new();
        for _ in 0..128 {
            builder.check(AxiomID::MinkowskiSignature, || true).unwrap();
        }

        assert!(matches!(builder.frames, WitnessBuffer::Small(_)));

        builder.check(AxiomID::MinkowskiSignature, || true).unwrap();

        assert!(matches!(builder.frames, WitnessBuffer::Large(_)));
    }

    #[test]
    fn witness_encodes_axiom_id_correctly() {
        let proof = build_proof(&[(AxiomID::AlgebraicConnectivity, true)]).unwrap();
        // Frame[0] = axiom_id = 2
        assert_eq!(proof.witness[0], 2u8);
        // Frame[1] = result = 1
        assert_eq!(proof.witness[1], 1u8);
    }

    // ── Default impl ────────────────────────────────────────────────────────

    #[test]
    fn witness_builder_default_is_empty() {
        let builder = WitnessBuilder::default();
        let proof = builder.build(0);
        assert_eq!(proof.axioms_checked, AxiomSet::empty());
        assert!(proof.witness.is_empty());
    }

    #[test]
    fn malformed_axiom_id_in_witness_is_rejected() {
        let mut proof = build_proof(&[(AxiomID::MinkowskiSignature, true)]).unwrap();
        proof.witness[0] = 9;
        proof.hash = blake3_hash(&proof.witness);
        assert!(!AxiomGuard::verify(&proof, &[AxiomID::MinkowskiSignature]));
    }

    #[test]
    fn malformed_claimed_axiom_mask_is_rejected() {
        let mut proof = build_proof(&[(AxiomID::MinkowskiSignature, true)]).unwrap();
        proof.axioms_checked = AxiomSet(0b1000_0000);
        assert!(!AxiomGuard::verify(&proof, &[AxiomID::MinkowskiSignature]));
    }

    #[test]
    fn blake3_hash_differs_from_reference_digest_for_test_vector() {
        fn reference_sha_256_baseline(data: &[u8]) -> [u8; 32] {
            assert_eq!(data, b"test");
            [
                0x9f, 0x86, 0xd0, 0x81, 0x88, 0x4c, 0x7d, 0x65, 0x9a, 0x2f, 0xea, 0xa0, 0xc5, 0x5a,
                0xd0, 0x15, 0xa3, 0xbf, 0x4f, 0x1b, 0x2b, 0x0b, 0x82, 0x2c, 0xd1, 0x5d, 0x6c, 0x15,
                0xb0, 0xf0, 0x0a, 0x08,
            ]
        }

        let blake3_digest = blake3_hash(b"test");
        assert_eq!(blake3_digest.len(), 32);
        assert_ne!(blake3_digest, reference_sha_256_baseline(b"test"));
    }

    #[test]
    fn meta_consistency_detects_conflict_between_two_valid_proofs() {
        let mut a = WitnessBuilder::new();
        a.check(AxiomID::MinkowskiSignature, || true)
            .expect("axiom check should pass");
        let proof_a = a
            .build(100)
            .with_meta(ProofMeta {
                origin_node: NodeId::try_new(7).expect("valid node"),
                target_domain_hash: [1u8; 32],
                resulting_state_hash: [9u8; 32],
            })
            .with_causal_links(None, Premises::new());

        let mut premises_b = Premises::new();
        premises_b.push(proof_a.hash);

        let mut b = WitnessBuilder::new();
        b.check(AxiomID::MinkowskiSignature, || true)
            .expect("axiom check should pass");
        b.check(AxiomID::CohomologyZero, || true)
            .expect("axiom check should pass");
        let proof_b = b
            .build(101)
            .with_meta(ProofMeta {
                origin_node: NodeId::try_new(7).expect("valid node"),
                target_domain_hash: [1u8; 32],
                resulting_state_hash: [10u8; 32],
            })
            .with_causal_links(Some(proof_a.hash), premises_b);

        assert!(AxiomGuard::verify(&proof_a, &[AxiomID::MinkowskiSignature]));
        assert!(AxiomGuard::verify(&proof_b, &[AxiomID::MinkowskiSignature]));

        let validator = MetaConsistencyValidator::default();
        let err = AxiomGuard::verify_with_meta(
            &proof_b,
            &[AxiomID::MinkowskiSignature],
            &[&proof_a],
            &validator,
        )
        .expect_err("proofs should conflict at meta layer");

        assert_eq!(
            err,
            MetaConsistencyError::Conflict {
                candidate: proof_b.hash,
                existing: proof_a.hash,
            }
        );
    }
}
