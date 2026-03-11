# GENESIS — Interface Contracts

**Revision:** 4.2.0
**Status:** CANONICAL — source of truth for cross-crate integration
**Scope:** Crates 000–003 (implemented). Crates 004–006 reserved; see `docs/04_IMPLEMENTATION_ROADMAP.md`.

> **Policy.** Every guarantee listed here is backed by a `cargo test` assertion.
> Benchmark figures are median wall-clock from Criterion on a sandbox VM (AMD EPYC,
> no AVX-512, no NEON); production hardware will be faster.
>
> **Last verified:** 385 tests, 0 failures, 0 compiler warnings (v5.3.0-ci-clean).
>
> **Changelog v4.2.0 (2026-03-09):**
> - CRATE-000: `Proof.witness` type corrected to `Witness = SmallVec<[u8;512]>` (was `Vec<u8>`)
> - CRATE-003: `QuantumOscillator` now documents `amplitudes`, `complex_state`, `amplitude_norm`,
>   `update_amplitude_from_fisher`, `FISHER_TRACE_INITIAL` — required interfaces for CRATE-004
> - CRATE-003: `synchrony_order_cached` now documents amplitude-weighted formula
> - CRATE-003: `AttractorLandscape` internal structure corrected (`HashMap+BTreeSet`, was `Vec`)
> - CRATE-002/003: documented 6 prerequisite APIs that must be added as Phase 5 prep
>   before `inelastic_concept_fusion` in CRATE-004 can compile

---

## Dependency Graph

```
genesis-types    (000)
      │
genesis-math     (001)
      │
genesis-topology (002)
      │
genesis-dynamics (003)
```

Each crate imports only from crates numerically below it. No exceptions.

---

## CRATE-000 — `genesis-types`

**Path:** `shared/genesis-types` | **Version:** 0.1.0
**Dependencies:** `thiserror 1`, `static_assertions 1.1`, `blake3 =1.5.4`, `arrayvec 0.7`, `smallvec 1`

---

### `NodeId`

```rust
pub struct NodeId(u64);  // Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord
```

| Member | Type | Contract |
|--------|------|----------|
| `MAX_VALID` | `const u64 = u64::MAX - 1` | All raw values ≤ MAX_VALID are accepted by `try_new`. |
| `INVALID` | `const NodeId` | Sentinel value. Raw = `u64::MAX`. Never returned by constructors. |
| `try_new(raw)` | `(u64) → Result<Self, GenesisError>` | `Err(NodeIdOutOfRange)` iff `raw > MAX_VALID`. |
| `from_raw_unchecked(raw)` | `unsafe (u64) → Self` | Caller guarantees `raw ≤ MAX_VALID` or `raw = u64::MAX` (INVALID). |
| `get()` | `() → u64` | Returns the inner raw value. |

---

### `SpikeComponents`

```rust
pub struct SpikeComponents { ... }  // Copy, 168 bytes, no heap allocation
```

| Method | Contract |
|--------|----------|
| `from_pairs<I: IntoIterator<Item=(u16,f64)>>` | Sub-Planck coefficients silently discarded. |
| `from_typed_pairs<const D: usize>` | Compile-time blade bound check. |
| `try_from_pairs<const D: usize>` | `Err` for NaN/Inf or index ≥ D. |
| `cardinality() → usize` | Number of active blades. |
| `is_active(blade: u16) → bool` | True iff `|coeff[blade]| > PLANCK`. |
| `get(blade: u16) → f64` | Returns 0.0 for inactive blades. |
| `active_pairs() → impl Iterator<Item=(u16, f64)>` | Active blades only. |

`size_of::<SpikeComponents>() == 168` — verified at compile time.

---

### `SpikeEvent`

```rust
pub struct SpikeEvent { ... }  // Copy, ≤208 bytes, no heap
```

Fields: `origin_node_id: NodeId`, `timestamp: Timestamp`, `components: SpikeComponents`,
`is_internal_drive: bool`, `is_phase_collapse: bool`.

`is_internal_drive()` — True iff spike originated from VFE internal drive (no external observation).
`is_phase_collapse()` — True iff spike resulted from Kuramoto phase collapse.

---

### `DomainSignal<D: CognitiveDomain>`

Typed signal carrier parameterised over domain marker:
`PhysicsDomain`, `TopologyDomain`, `DynamicsDomain`, `ConsciousnessDomain`.
Prevents cross-domain routing at compile time.

---

### `DomainConsolidationSignal<State: ConsolidationState>`

Two-state machine: `Saturated → Certified`.
- `DomainConsolidationSignal::<Saturated>::saturated(domain, ts, delta_g)`
- `.certify() → DomainConsolidationSignal<Certified>`

`DomainResetSignal` triggers consolidation rollback for a domain.

---

### Proof System

#### `Witness` type

```rust
/// SSO buffer — stack-inline for witnesses ≤ 512 bytes, heap-spill above.
/// Zero heap allocation for all standard proof sizes (≤7 axiom frames × ~73 bytes/frame).
pub type Witness = SmallVec<[u8; 512]>;

/// Capacidad inline del buffer Witness SSO.
pub const WITNESS_INLINE_CAPACITY: usize = 512;
```

**Critical:** `Witness` is `SmallVec<[u8; 512]>`, NOT `Vec<u8>`. Upstream code must use
the `Witness` type alias. Constructors that accept `Vec<u8>` do not exist.

#### `AxiomID`

```rust
#[repr(u8)]
pub enum AxiomID {
    MinkowskiSignature    = 0,
    CohomologyZero        = 1,
    AlgebraicConnectivity = 2,
    PlanckConstant        = 3,
    ProofGuard            = 4,
    DualityConsistency    = 5,
    DimensionalAdmission  = 6,
}
```

Discriminants are stable across versions. Do not reorder.

```
STRUCTURAL_REQUIRED = [0, 1, 2, 3, 4]
EXPANSION_REQUIRED  = [0, 1, 2, 3, 4, 5, 6]
```

#### `AxiomSet`

```rust
pub struct AxiomSet(pub u8);  // bitmask over AxiomID discriminants
```

Const-constructible. Key methods: `empty()`, `from_axiom()`, `from_slice()`,
`contains()`, `union()`, `intersects()`, `difference()`, `is_subset_of()`,
`is_superset_of()`, `is_valid()`.

#### `Proof`

```rust
pub struct Proof {
    pub axioms_checked: AxiomSet,
    pub witness:        Witness,   // SmallVec<[u8; 512]> — NOT Vec<u8>
    pub timestamp:      u64,       // nanoseconds since Unix epoch
    pub hash:           [u8; 32],  // BLAKE3(witness)
}
```

| Method | Contract |
|--------|----------|
| `new(axioms, witness: Witness, timestamp)` | Computes `hash = BLAKE3(witness)` at construction. Zero heap if witness ≤ 512 bytes. |
| `is_internally_consistent() → bool` | `BLAKE3(witness) == self.hash`. |
| `is_fresh(current_ns) → bool` | `current_ns − timestamp < PROOF_MAX_AGE_NS` (5 seconds). |

#### `AxiomGuard`

| Method | Contract |
|--------|----------|
| `verify(proof, required: &[AxiomID]) → bool` | BLAKE3 integrity + symbolic witness replay. Returns false on hash mismatch, missing axiom, or any frame with result ≠ 1. |
| `verify_for<'p, M: Mutation>(proof, mutation) → Option<VerifiedProof<'p, M>>` | Typed proof binding. `None` on failure. |

#### `WitnessBuilder`

| Method | Contract |
|--------|----------|
| `new()` | Empty builder, zero allocation. |
| `check(axiom, f: FnOnce()→bool) → Result<(), GenesisError>` | Appends a wire frame. `Err(InvariantViolation)` if closure returns false. |
| `build(timestamp: u64) → Proof` | Finalises and computes BLAKE3. Returns `Proof` with `Witness` type. |

**Do not call `verify()` inside hot paths.** Proofs are generated once in `propose()`.

---

### `FisherEdgeMetric`

Stores Fisher information values keyed by ordered `(NodeId, NodeId)` pairs.

| Method | Contract |
|--------|----------|
| `get(i, j) → f64` | Canonicalises key order. Returns 0.0 for unknown pairs. |
| `set(i, j, value)` | Canonicalised storage. |
| `remove(i, j)` | No-op if absent. |
| `edges() → impl Iterator<Item=(NodeId, NodeId, f64)>` | All stored pairs. |
| `is_current(n: NodeId) → bool` | True iff node `n` appears in at least one stored edge. Used by `DualityConsistency` check in proof system. |

---

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `COGNITIVE_PLANCK_CONSTANT` | `1e-12` | CS gate threshold. Immutable at runtime. AX-ID: AXIOMA-011. |
| `MINKOWSKI_SIGNATURE` | `[1.0, -1.0, -1.0, -1.0]` | G(1,3) spacetime signature. AX-ID: AXIOMA-001. |
| `SPATIAL_BASIS_MASK` | `0b1110` | Blade bits for e₁, e₂, e₃. |
| `CLIFFORD_BASIS_SIZE` | `16` | Blade count in G(1,3). |
| `FISHER_SATIATION_EPSILON` | `1e-6` | ΔG below which a domain is saturated. AX-ID: AXIOMA-008. |
| `FISHER_SATIATION_WINDOW` | `50` | Consecutive below-ε iterations required for satiation. |
| `SOC_TAU_MIN / MAX` | `1.5 / 2.5` | Valid range for power-law exponent τ. AX-ID: AXIOMA-005. |
| `SYNCHRONY_COLLAPSE_THRESHOLD` | `0.85` | Kuramoto r above which phase collapse fires. AX-ID: AXIOMA-006. |
| `WORMHOLE_CURVATURE_THRESHOLD` | `-0.5` | Legacy: Ollivier-Ricci K below which WormholeCollapse activates. Superseded by `RICCI_PREEMPTIVE_THRESHOLD` for Ricci-path fusions. |
| `RICCI_PREEMPTIVE_THRESHOLD` | `1e-4` | Edge weight below which preemptive inelastic fusion fires. AX-ID: AXIOMA-015. |
| `SINKHORN_MAX_ITER` | `1000` | Iterations for global Wasserstein-1. |
| `SINKHORN_MAX_LOCAL_DEGREE` | `32` | Max support size for micro-Sinkhorn L1. Cost matrix = 32×32×4 = 4096 bytes (L1 resident). |
| `SINKHORN_LOG_EPSILON` | `5e-2_f32` | Log-domain regularisation for micro-Sinkhorn. |
| `SINKHORN_L1_MAX_ITER` | `15` | Micro-Sinkhorn iterations. Converges in 12–15 for 32×32. |
| `JL_RESIDUAL_EXPANSION_DELTA` | `1e-3` | Residual norm threshold for GramSchmidt expansion. |
| `HEAT_DIFFUSION_CONVERGENCE_EPSILON` | `1e-8` | Heat equation convergence criterion. |
| `LAMBDA2_MIN` | `0.1` | Minimum Fiedler value (algebraic connectivity). LEY §5.3. |
| `PROOF_PENALTY` | `1000.0` | H_restricción weight for unproven mutations. |
| `PROOF_MAX_AGE_NS` | `5_000_000_000` | Proof expiry: 5 seconds. |
| `DENSITY_PENALTY` | `50.0` | κ_s for H_restricción edge density sub-term. LEY §3.5. |
| `DELTA_DUALITY` | `0.01` | δ coupling for H_dualidad. LEY §3.6. |
| `KAPPA_REDUNDANCY` | `0.05` | κ_r for H_compresión. LEY §3.7. |
| `REDUNDANCY_RADIUS` | `0.1` | ε_r search radius for redundancy detection. |
| `PHASE_DISTINCTION_THRESHOLD` | `0.3` | θ_r: phase diff below which two nodes are redundant. |
| `LAMBDA_DIM_FIXED` | `0.05` | Fixed cost of dimensional admission. LEY §3.8. |
| `LAMBDA_DIM_LOG` | `0.01` | Log-scale cost of dimensional admission. |
| `WITNESS_INLINE_CAPACITY` | `512` | Inline byte capacity of `Witness` SSO buffer. |

---

### `GenesisError`

`thiserror`-derived. Key variants: `BladeIndexOutOfRange`, `SignatureViolation`,
`CohomologyNonTrivial`, `InvariantViolation { axiom_id: u8 }`,
`ProofMissing { mutation_name: &'static str }`, `ProofInvalid { axiom_id: u8 }`,
`NodeIdOutOfRange`, `NodeNotFound`.

---

### Prohibited imports

`genesis-math`, `genesis-topology`, `genesis-dynamics`, `genesis-evolution`,
`genesis-consciousness`, `genesis-io`.

---

## CRATE-001 — `genesis-math`

**Path:** `core/genesis-math` | **Version:** 0.2.0
**Dependencies:** `genesis-types`, `bytemuck 1.14`, `memoffset 0.9`
**Features:** `deterministic_strict`, `avx512`, `cacheline64`, `properties`

---

### `SparseCliffordVector`

```rust
pub struct SparseCliffordVector { ... }
// Copy, Clone, Pod, Zeroable (bytemuck), Debug, PartialEq
```

**Memory layout** (compile-time verified via `memoffset::offset_of!`):

| Offset | Field | Bytes | Description |
|--------|-------|-------|-------------|
| 0 | `coeffs: [f64; 16]` | 128 | Blade coefficients. Inactive = exactly `+0.0`. |
| 128 | `clifford_norm_sq: f64` | 8 | `Σᵢ coeffsᵢ² × CLIFFORD_NORM_WEIGHTS[i]`. Lorentz-invariant. May be negative. **Prohibited for CS gate.** |
| 136 | `max_abs_coeff: f64` | 8 | `maxᵢ |coeffsᵢ|`. **Sole valid field for the CS gate.** |
| 144 | `active_mask: u16` | 2 | Bit i = 1 iff `|coeffs[i]| > COGNITIVE_PLANCK_CONSTANT`. |
| 146 | `_pad: [u8; 14]` | 14 | Padding. Not semantically meaningful. |

Total: **160 bytes**, alignment **32** (default).
With feature `cacheline64`: **192 bytes**, alignment **64**.

**Invariants enforced by all constructors:**
- All `coeffs[i]` are finite.
- `active_mask` is consistent with `coeffs`.
- `max_abs_coeff = maxᵢ(|coeffs[i]|)` over active blades.
- `clifford_norm_sq = Σᵢ coeffs[i]² × CLIFFORD_NORM_WEIGHTS[i]`.
- All inactive `coeffs[i]` store exactly `+0.0` (no `-0.0`).

| Method | Contract |
|--------|----------|
| `from_iter<I: IntoIterator<Item=(usize,f64)>>` | `Err(BladeIndexOutOfRange)` if index > 15. `Err(SignatureViolation)` if NaN/Inf. Duplicate indices summed. Sub-Planck discarded. |
| `from_dense(&[f64; 16])` | Same validation. All 16 coefficients provided. |
| `zero() → Self` | Additive identity. `active_mask = 0`. |
| `is_negligible() → bool` | `max_abs_coeff ≤ COGNITIVE_PLANCK_CONSTANT`. |
| `grade_project(grade) → Self` | Projects onto the given Clifford grade subspace. |
| `even_grade() → Self` | Grades 0, 2, 4. |
| `odd_grade() → Self` | Grades 1, 3. |
| `reverse() → Self` | Clifford reverse Ã. |
| `scalar_part() → f64` | `coeffs[0]`. |
| `l2_norm() → f64` | Euclidean L2 norm over all 16 coefficients. |
| `metric_scalar_product(&Self) → f64` | `⟨A·B̃⟩₀` under Minkowski signature. |
| `geo_product(&Self) → Option<Self>` | Delegates to `sparse_geometric_product`. |
| `as_bytes() → &[u8]` | Zero-copy. Valid because `bytemuck::Pod`. |
| `try_from_bytes(&[u8]) → Result<&Self, PodCastError>` | Zero-copy. |

---

### `sparse_geometric_product`

```rust
pub fn sparse_geometric_product(
    a: &SparseCliffordVector,
    b: &SparseCliffordVector,
) -> Option<SparseCliffordVector>
```

Returns `None` when:
- CS gate fires: `a.max_abs_coeff × b.max_abs_coeff × 16 < COGNITIVE_PLANCK_CONSTANT`
- Either `active_mask == 0`
- Algebraic result is zero (all outputs sub-Planck after reduction)

**Dispatch paths (automatic by compile target):**
- Scalar: double-rail flat buffer, O(256) FMAs
- AVX2 (x86-64 `+avx2`): 8-wide f64 vectorisation
- AVX-512 (feature `avx512`, `+avx512f`): 16-wide zmm registers
- NEON (aarch64): 8 × `float64x2_t` accumulators via `vdupq_n_f64` + `vsetq_lane_f64` — no temporary stack arrays

**Feature `deterministic_strict`:** Kahan-compensated accumulation + `-0.0` canonicalisation.
Bit-exact across runs and platforms. Required for Proof generation.

**`CAYLEY_SIGN: [[i8; 16]; 16]`** — 256 bytes, compile-time `.rodata`. L1-resident at steady state.
Index convention: `CAYLEY_SIGN[i][j]` = sign for input blade pair `(i, j)`.

---

### `bivector_norm_sq_of_product`

```rust
pub fn bivector_norm_sq_of_product(a: &SparseCliffordVector, b: &SparseCliffordVector)
    -> BivectorProduct

pub enum BivectorProduct {
    Computed(f64),  // algebraic result; negative = spacelike, zero = null
    SubPlanck,      // CS gate fired or active_mask = 0
}
```

Lorentz-weighted norm-squared of the grade-2 part of `a·b`.
`BIVECTOR_MASK = 0x1668` (blades 3, 5, 6, 9, 10, 12) — module-level constant.
Blade weights: 3→−1, 5→−1, 6→+1, 9→−1, 10→+1, 12→+1 — module-level constant.
Zero heap. 128-byte stack buffer.

**This is not a metric.** The triangle inequality does not hold.
Callers use `.abs().sqrt()` as a proximity measure in HNSW.

Variant `bivector_norm_sq_of_product_lhs_dense(&[f64;16], &SparseCliffordVector)` for f16-decompressed HNSW paths.

---

### `fast_metric_distance`

```rust
pub fn fast_metric_distance(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64
```

`d(a, b) = √(Σᵢ METRIC_WEIGHTS[i] · (aᵢ − bᵢ)²)` — grade-differentiated semantic metric.

Weights by G(1,3) grade: grade-0 (scalar) = 2.0, grade-1 (vectors, semantic primary) = 1.5,
grade-2 (bivectors) = 1.0, grade-3 (trivectors) = 0.5, grade-4 (pseudoscalar) = 0.3.
All weights strictly positive → all four metric axioms satisfied.
Returns `f64::MAX` when CS gate fires. O(16), no allocation.
This is the **primary metric** of the system. `geometric_distance` in `genesis-topology` delegates to this.

Also exported: `fast_metric_distance_from_dense(a_dense: &[f64;16], b: &SparseCliffordVector) → f64`.

---

### Supporting exports

| Symbol | Description |
|--------|-------------|
| `CliffordBasis`, `CANONICAL_G13` | Static G(1,3) basis. Signature `[1,-1,-1,-1]`. |
| `CLIFFORD_NORM_WEIGHTS: [i8; 16]` | Per-blade Lorentz weights for `clifford_norm_sq`. |
| `CAYLEY_SIGN: [[i8; 16]; 16]` | Compile-time Cayley table. 256 bytes. |
| `compute_clifford_sign(i, j) → i8` | POPCNT-based sign. O(1). |
| `fast_cayley_product(i, j) → (usize, i8)` | Blade index + sign. O(1). |
| `compute_clifford_norm_sq(&[f64;16]) → f64` | Weighted norm-squared. |
| `grade_project(v, grade)`, `grade_project_ct<const G>` | Grade projection, compile-time variant for `G ∈ {0,1,2,3,4}`. |
| `even_grade / odd_grade / reverse / grades_present / max_grade / min_grade / is_homogeneous` | Grade analysis. |
| `SparseDualVector`, `geometric_product_dual` | Dual space operations. |
| `GeometricProduct` trait, `GeometricProductMode` enum | Operator abstraction. |
| `experimental::kernel_dense_g13` | **PROHIBITED in production.** Benchmarks only. |

---

### Restrictions for upstream crates

- `clifford_norm_sq` may be negative. Do not use as a distance.
- `max_abs_coeff` is the exclusive field for the CS gate. Do not substitute `l2_norm`.
- `BivectorProduct::Computed(v)`: callers apply `.abs().sqrt()` before use as scalar distance.
- `experimental::kernel_dense_g13` must not appear in production code.
- `active_mask` is `u16`. G(1,3) is fixed at 16 blades.

---

### Prohibited imports

`genesis-topology`, `genesis-dynamics`, `genesis-evolution`, `genesis-consciousness`, `genesis-io`.

---

## CRATE-002 — `genesis-topology`

**Path:** `core/genesis-topology` | **Version:** 0.1.0
**Dependencies:** `genesis-types`, `genesis-math`, `fixedbitset 0.5`, `smallvec 1.13`
**Features:** `hnsw-f16`

---

### Distance functions (`geodesic` module)

#### `geometric_distance` — primary metric

```rust
pub fn geometric_distance(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64
```

Delegates to `fast_metric_distance`. Satisfies all four metric axioms.
Used by `ManifoldCollector`, lambda₂ computation, and all topological operations.

#### `bivector_interaction`

```rust
pub fn bivector_interaction(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64
```

`|BivectorProduct::Computed(v)|` or `f64::MAX`. Not a metric.

#### `fast_bivector_distance`

```rust
#[deprecated(note = "use geometric_distance()")]
pub fn fast_bivector_distance(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64
```

Legacy alias. Delegates to `geometric_distance`. Will be removed in a future version.

---

### `HnswGraph`

```rust
pub struct HnswGraph { ... }  // not Clone
```

| Method | Signature | Contract |
|--------|-----------|----------|
| `new(ef_construction)` | `(usize) → Self` | |
| `insert(id, vec)` | `(NodeId, &SparseCliffordVector) → Result<(), GenesisError>` | Idempotent if `id` already present. `Err` if `|E| > N × log₂(N) × 2`. |
| `search_nearest(query, k)` | `(&mut self, &SparseCliffordVector, usize) → Vec<NodeId>` | `&mut self` for generation-counter visited tracking. Returns ≤ k nearest by `geometric_distance`. |
| `neighbors(id)` | `(NodeId) → impl Iterator<Item=NodeId>` | Layer-0 neighbours. |
| `neighbors_within(id, radius)` | `(NodeId, f64) → impl Iterator<Item=NodeId>` | Layer-0 neighbours with `geometric_distance ≤ radius`. Single `get_idx` call. |
| `get_vector(id)` | `(NodeId) → Option<&SparseCliffordVector>` | `None` if absent. |
| `nodes()` | `() → impl Iterator<Item=NodeId>` | |
| `node_count() → usize` | | |
| `edge_count() → usize` | | Layer-0 only. |
| `compact_index()` | `(&mut self)` | Rebuilds `direct_index` for O(1) lookup. Call after bulk inserts. |

**Feature `hnsw-f16`:** Layer-0 stores vectors as 16×f16.

**Benchmarks (sandbox VM):** insert_1000: ~91 µs/insert. search_k10_in_1000: ~14 µs/query.

---

### `IncrementalH1State`

Maintains exact `dim(H¹) = dim(ker ∂₁) − dim(im ∂₂)` under online insertions.
Amortised complexity: O(K·α(N) + K²) per insertion.

| Method | Contract |
|--------|----------|
| `new()` | Zero allocation. |
| `add_node()` | O(1). |
| `add_edge(u, v) → u32` | Returns edge index. O(α(N)). |
| `add_triangle(a, b, c)` | Gaussian elimination step on D2. |
| `h1_is_zero() → bool` | O(1). |
| `h1_dim() → usize` | O(1). |

---

### `ManifoldCollector`

```rust
pub struct ManifoldCollector { ... }
```

| Method | Signature | Contract |
|--------|-----------|----------|
| `new(ef_construction)` | `(usize) → Self` | |
| `insert(id, vec)` | `(NodeId, &SparseCliffordVector) → Result<(), GenesisError>` | Inserts into HNSW; updates H¹ state; detects triangles. |
| `node_count() → usize` | | |
| `edge_count() → usize` | | |
| `compute_edge_density() → f64` | | `|E| / (N × log₂(N))`. |
| `compute_lambda2() → f64` | | Lanczos + shifted power iteration. Not hot-path. |
| `compute_h1() → usize` | | Full rebuild. Batch validation only. |
| `h1_is_zero_fast() → bool` | | O(1). |
| `h1_dim_fast() → usize` | | O(1). |
| `find_affected_nodes(candidate)` | `(&mut self, &SparseCliffordVector) → Vec<NodeId>` | K nearest neighbours. |
| `hyperbolic_coord(id) → Option<HyperbolicCoord>` | | O(log N). Returns `None` until CRATE-004 calls `set_hyperbolic_coord`. |
| `set_hyperbolic_coord(id, coord)` | `(NodeId, HyperbolicCoord)` | O(log N) sorted-vec insert. |
| `hyperbolic_coord_count() → usize` | | |

**`LambdaWorkspace`:** thread-local, geometric growth. No fixed node limit.

#### CRATE-004 prerequisite API (to be added in Phase 5 prep)

```rust
/// Removes node `id` from HNSW graph and H¹ state.
/// Complexity: O(K) for edge cleanup.
/// Required by `inelastic_concept_fusion` in genesis-evolution.
pub fn remove_node(&mut self, id: NodeId) -> Result<(), GenesisError>
```

This method does not exist in v0.1.0. Must be added to `ManifoldCollector` (and underlying
`HnswGraph`) before CRATE-004's `wormhole.rs` can compile.

---

### `HyperbolicCoord`

```rust
pub struct HyperbolicCoord { pub x: f64, pub y: f64 }
// Invariant: x² + y² < 1 (Poincaré disk)
```

| Method | Contract |
|--------|----------|
| `new(x, y) → Option<Self>` | `None` if `x²+y² ≥ 1` or non-finite. |
| `norm_sq() → f64` | Always < 1.0. |
| `hyperbolic_distance_to_origin() → f64` | `2·arctanh(|coord|)`. |

Populated exclusively by `DiscreteRicciFlow` (CRATE-004).
Before CRATE-004: all `hyperbolic_coord()` calls return `None`.

---

### `CohomologyValidator`

```rust
pub fn check_h1(complex: &RipsComplex) -> bool
```

Full Z₂ Gaussian elimination. Use for batch validation only.

---

### `RipsComplex`

```rust
pub fn build(graph: &HnswGraph, epsilon: f64) -> Self
pub fn simplices_of_dim(&self, d: usize) -> impl Iterator<Item = &[NodeId]>
pub fn counts(&self) -> (usize, usize, usize)
```

---

### `CliffordHashTable`

```rust
pub fn insert(&mut self, id: NodeId, vec: &SparseCliffordVector)
pub fn candidates<'a>(&'a self, query: &SparseCliffordVector) -> impl Iterator<Item = NodeId> + 'a
```

---

### Restrictions for upstream crates

- `search_nearest` is `&mut self`. Do not call from immutable contexts.
- `compute_lambda2()` and `compute_h1()` are not hot-path.
- `RipsComplex::build` is O(N·K²). Call only for full revalidation.
- `HnswGraph` does not implement `Clone`.
- `fast_bivector_distance` is deprecated. Use `geometric_distance`.

---

### Prohibited imports

`genesis-dynamics`, `genesis-evolution`, `genesis-consciousness`, `genesis-io`.

---

## CRATE-003 — `genesis-dynamics`

**Path:** `core/genesis-dynamics` | **Version:** 0.1.0
**Dependencies:** `genesis-types`, `genesis-math`, `genesis-topology`

---

### `QuantumOscillator`

```rust
pub struct QuantumOscillator {
    pub node_id:     NodeId,
    pub phases:      [f64; 5],      // one phase per Clifford grade 0..=4
    pub amplitudes:  [f64; 5],      // inferential certainty per grade ∈ [0.0, 1.0], init 1.0
    pub frequencies: [f64; 5],
    pub state:       OscillatorState,
}

pub enum OscillatorState {
    Active,
    Saturated { since_ns: u64 },
    Pruned    { at_ns: u64 },
}
```

Lifecycle is **unidirectional**: `Active → Saturated → Pruned`. Irreversible.

| Method | Contract |
|--------|----------|
| `new(node_id, frequencies)` | Phases = `[0.0; 5]`. Amplitudes = `[1.0; 5]`. State = `Active`. |
| `with_phases(node_id, phases, frequencies)` | Explicit phase initialisation. Amplitudes = `[1.0; 5]`. |
| `primary_phase() → f64` | `phases[0]` (grade-0). |
| `mark_saturated(now_ns)` | `Active → Saturated`. No-op if already saturated or pruned. |
| `mark_pruned(now_ns)` | Transitions to `Pruned` from any non-pruned state. |
| `is_active() → bool` | True iff `Active`. |
| `contributes_to_sync() → bool` | True iff `Active` or `Saturated`. Pruned excluded from all Kuramoto sums. |

#### Amplitude methods (required by CRATE-004 `DiscreteRicciFlow`)

```rust
/// ψ_{i,g} = A_{i,g} · (cos φ_{i,g}, sin φ_{i,g}) — complex state per grade.
/// CRATE-004 reads re.hypot(im) as per-node cluster amplitude for Ricci weighting.
pub fn complex_state(&self, g: usize) -> (f64, f64)

/// Normalised total amplitude: √(Σ_g amplitudes[g]²/5) ∈ [0.0, 1.0].
pub fn amplitude_norm(&self) -> f64

/// Sets all grade amplitudes from scalar Fisher trace.
///   amplitudes[g] = (fisher_trace / FISHER_TRACE_INITIAL).clamp(0.0, 1.0)  ∀g
/// With fisher_trace = FISHER_TRACE_INITIAL (prior): no change to amplitudes.
/// Call after every VFEMinimizer::update().
pub fn update_amplitude_from_fisher(&mut self, fisher_trace: f64)

/// Prior Fisher trace (1.0). With trace = FISHER_TRACE_INITIAL → amplitudes unchanged.
pub const FISHER_TRACE_INITIAL: f64 = 1.0;
```

**Amplitude semantics:**
- `amplitudes[g] = 1.0`: uninformative prior, maximum contribution to r_sync
- `amplitudes[g] → 0`: saturated domain, suppressed contribution (AXIOMA-008)
- With `amplitudes = [1.0;5]` (default): Kuramoto behaviour identical to classical formulation

---

### `QuantumKuramotoNetwork`

Multi-grade Kuramoto network. Equation of motion (derived from `H_dinámica`, not postulated):

```
dφᵢₘ/dt = ωᵢₘ + Σⱼ Γᵢⱼ sin(φⱼₘ − φᵢₘ) + ηᵢₘ √(2kT dt)
```

Integration: Euler-Maruyama. Noise: Box-Muller with internal LCG. No `rand` dependency.
Coupling: sorted `Vec<(NodeId, NodeId, f64)>` with CSR offset index. No `HashMap` in hot path.

| Method | Signature | Contract |
|--------|-----------|----------|
| `new(temperature)` | `(f64) → Self` | `temperature = 0.0` → deterministic. |
| `add_oscillator(osc)` | `(QuantumOscillator) → Result<NodeId, GenesisError>` | `Err` if count exceeds density limit. |
| `set_coupling(i, j, γ)` | `(NodeId, NodeId, f64)` | Directed. Call twice for symmetric. |
| `set_coupling_batch(pairs)` | `(&[(NodeId, NodeId, f64)])` | Single index rebuild. |
| `node_count() → usize` | | |
| `phases() → &[QuantumOscillator]` | | Read-only view. |
| `phase_diff_norm(i, j) → f64` | `(NodeId, NodeId) → f64` | L2 norm of `‖φᵢ − φⱼ‖` over all 5 grades. Used by `compute_h_compression` in CRATE-004. |
| `step(dt)` | `(f64)` | Euler-Maruyama step. Snapshot before update. Skips Pruned. |
| `synchrony_order_cached() → f64` | `(&mut self)` | Amplitude-weighted r_sync (see formula below). Recomputes only if `sync_dirty`. |

**Amplitude-weighted synchrony order parameter:**
```
r_sync = (1/G) Σ_{g=0}^{G-1} |Σ_i A_{i,g}·e^{iφ_{i,g}}| / (Σ_i A_{i,g} + ε)
```
With `A_{i,g} = 1.0` (default prior): reduces exactly to classical Kuramoto. Backward compatible.
With learned nodes (`A → 0`): saturated nodes fade from r_sync; high-VFE nodes dominate.

#### CRATE-004 prerequisite APIs (to be added in Phase 5 prep)

```rust
/// Per-node amplitude_norm. O(1) via direct_index.
/// Required by `inelastic_concept_fusion` for post-merge amplitude weighting.
pub fn amplitude_norm(&self, id: NodeId) -> f64

/// Sets the same amplitude for all grades of oscillator `id`.
/// Required by `inelastic_concept_fusion` to propagate merged amplitude.
pub fn set_amplitude_all_grades(&mut self, id: NodeId, amplitude: f64)

/// Removes oscillator `id` from all internal structures.
/// Invalidates sync_dirty. Removes couplings referencing `id`. O(E) worst case.
/// Required by `inelastic_concept_fusion` post-merge cleanup.
pub fn remove_oscillator(&mut self, id: NodeId) -> Result<(), GenesisError>
```

These methods do not exist in v0.1.0. Must be added before CRATE-004's `wormhole.rs` compiles.

---

### Synchrony utilities

```rust
pub fn synchrony_order_fast(network: &QuantumKuramotoNetwork) -> f64
pub fn synchrony_order(network: &QuantumKuramotoNetwork) -> f64
pub fn synchronized_cluster(network: &QuantumKuramotoNetwork, threshold: f64) -> Vec<NodeId>
```

`synchrony_order_fast`: amplitude-weighted. Feature `poly_trig` enables Cephes polynomials
(degree-9/10 Horner with Cody-Waite argument reduction). Not bit-exact. Do not use in Proof generation.

`synchronized_cluster`: nodes whose `primary_phase` deviation from mean is below `threshold`.

---

### `VFEMinimizer`

Active inference engine. Per-node Gaussian beliefs over all 16 G(1,3) blades.

#### `Belief`

```rust
pub struct Belief {
    pub mean_full:      [f64; 16],  // μ over all 16 G(1,3) blades
    pub precision_full: [f64; 16],  // diagonal precision ∈ [0, 1e6], init 1.0
    pub node_id:        NodeId,
}
```

`GRADE1_BLADE_INDICES: [usize; 4] = [1, 2, 4, 8]` — blade indices of e₀,e₁,e₂,e₃.

| Method | Contract |
|--------|----------|
| `mean() → [f64; 4]` | Grade-1 components from `mean_full[GRADE1_BLADE_INDICES]`. Backward compatible. |
| `fisher_trace() → f64` | Sum of all 16 `precision_full` values. |

#### `VFEMinimizer`

`VFE_BLADE_WEIGHTS: [f64; 16]` — mirrors `METRIC_WEIGHTS` for inferential-topological coherence.

| Method | Signature | Contract |
|--------|-----------|----------|
| `new() → Self` | | |
| `add_node(id, prior_mean)` | `(NodeId, [f64; 4])` | `mean_full` at GRADE1 blades = prior; zeros elsewhere. `precision_full = [1.0; 16]`. |
| `compute_vfe(id, obs)` | `(NodeId, Option<&[f64; 4]>) → f64` | Grade-1 subspace only. Kahan-Babuška. Backward compatible. |
| `compute_vfe_with_grad(id, obs)` | `(NodeId, Option<&SCV>) → (f64, [f64; 16])` | Full 16D VFE + gradient. For CRATE-004 Ricci flow signal. |
| `compute_vfe_with_grad_grade1(id, obs)` | `(NodeId, Option<&SCV>) → (f64, [f64; 4])` | Grade-1 components of 16D gradient. |
| `update(id, obs, dt)` | `(NodeId, &[f64; 4], f64)` | Grade-1 update only. Backward compatible. |
| `update_full(id, obs, dt)` | `(NodeId, &SCV, f64)` | 16D update. Touches all blades with finite obs coefficients. For CRATE-004. |
| `internal_drive() → Option<NodeId>` | `(&mut self)` | Max 16D VFE node weighted by `VFE_BLADE_WEIGHTS`. |
| `delta_g(id) → f64` | | `0.5·|ΔTr|`. AXIOMA-008 satiation gate. |
| `fisher(id) → Option<&FisherInfo>` | | Fisher scalar for node. |

**Note on `update` vs `update_full`:** `update()` modifies only 4 grade-1 blades. `update_full()`
modifies all 16. Using `update()` followed by `compute_vfe_with_grad()` is valid — the 12 non-grade-1
blades remain at prior (0.0). CRATE-004 must use `update_full` for post-collapse belief adjustments.

`FisherInfo::trace: f64` — scalar Fisher trace, init 1.0, floor 1e-12.
`FisherInfo::delta_g: f64` — change rate `≈ 0.5·|ΔTr|`.

#### CRATE-004 prerequisite APIs (to be added in Phase 5 prep)

```rust
/// Raw read of Belief for node `id`.
/// Required by `inelastic_concept_fusion` for precision-weighted centroid computation.
pub fn beliefs_raw(&self, id: NodeId) -> Result<&Belief, GenesisError>

/// Removes node `id` from all internal structures.
/// Required by `inelastic_concept_fusion` post-merge cleanup.
pub fn remove_node(&mut self, id: NodeId) -> Result<(), GenesisError>
```

These methods do not exist in v0.1.0. Must be added before CRATE-004's `wormhole.rs` compiles.

---

### `AttractorLandscape`

Internal structure: `HashMap<NodeId, f64>` (O(1) energy lookup) + `BTreeSet<AttractorEntry>`
(O(log N) energy-ordered iteration). **HashMap is intentional here** — not a hot-path structure.
It serves the consciousness layer bookkeeping (CRATE-006), not HNSW or Kuramoto.

`AttractorEntry` implements total order via `f64::total_cmp` + `NodeId` tiebreak,
and manual `PartialEq`/`Eq` via `to_bits()` to maintain `BTreeSet` invariants despite `f64::NaN`.

| Method | Contract |
|--------|----------|
| `register(id, energy)` | Upsert. O(log N). Idempotent for identical `(id, energy)` bits. |
| `descend(_current, vfe) → Option<NodeId>` | O(N_attractors) VFE evaluations. Single pass. `_current` reserved for CRATE-006 topological routing. |
| `ascending_energy() → impl Iterator<Item=NodeId>` | Deterministic BTreeSet iteration for CRATE-006 Ω construction. |
| `attractor_count() → usize` | O(1). |
| `energy_of(id) → Option<f64>` | O(1) via HashMap. |

---

### `CriticalityMonitor`

```rust
pub struct CriticalityReport {
    pub tau:          f64,
    pub ks_statistic: f64,
    pub p_value:      f64,   // Marsaglia series, 20 terms, convergence-checked
    pub is_critical:  bool,  // tau ∈ [SOC_TAU_MIN, SOC_TAU_MAX] && p_value > 0.05
}
```

| Method | Contract |
|--------|----------|
| `new(capacity)` | Ring buffer of `capacity` avalanche sizes. `NonZeroUsize` capacity. |
| `record_avalanche(size: u32)` | Overwrites oldest entry when full. |
| `tau_exponent() → Option<f64>` | MLE Clauset log-log regression. `None` if < 2 distinct sizes. |
| `tau_exponent_report() → Option<CriticalityReport>` | Full KS report. Allocates internally — not for hard real-time loops. |
| `needs_adjustment() → bool` | τ outside `[SOC_TAU_MIN, SOC_TAU_MAX]`. |
| `is_frozen() → bool` | All recent avalanches size 0. |
| `is_cognitively_viable(r_sync) → bool` | `!is_frozen() && r_sync ∈ [SOC_R_SYNC_MIN, SOC_R_SYNC_MAX]`. |

`kuramoto_critical_coupling(temperature) → f64` = `2√(2/π) · T` (analytic derivation).
`SOC_R_SYNC_MIN = 0.3`, `SOC_R_SYNC_MAX = 0.7`.

---

### AXIOMA-006 × AXIOMA-008 coupling pattern

After every `VFEMinimizer::update(id, obs, dt)`:
```rust
let trace = vfe.fisher(id).map(|f| f.trace).unwrap_or(1.0);
kuramoto.phases_mut()[idx].update_amplitude_from_fisher(trace);
```
Saturated nodes (trace → 0) → amplitude → 0 → weight in r_sync → 0.
High-VFE nodes (trace ≈ 1.0) → amplitude ≈ 1.0 → dominant weight in r_sync.
This coupling is the mechanism by which inferential state modulates collective coherence.

---

### Restrictions for upstream crates

- `step()` is O(N·K). Not re-entrant.
- `synchrony_order_fast` with `poly_trig` is not bit-exact. Do not use in Proof generation.
- `tau_exponent_report` allocates. Do not call in hard real-time loops.
- Kahan compensation in `compute_vfe` must not be double-applied by callers.
- `Witness` type in proof system is `SmallVec<[u8;512]>` — use the `Witness` alias.

---

### Prohibited imports

`genesis-evolution`, `genesis-consciousness`, `genesis-io`.

---

## Crates 004–006 — Specified, Not Yet Implemented

| Crate | Path | Phase |
|-------|------|-------|
| `genesis-evolution` (004) | `core/genesis-evolution` | Phase 5 |
| `genesis-consciousness` (006) | `core/genesis-consciousness` | Phase 6 |
| `genesis-io` (005) | `drivers/genesis-io` | Phase 7 |

Contracts for these crates will be added upon implementation.
Specification: `docs/04_IMPLEMENTATION_ROADMAP.md` + `docs/07_CRATE004_RICCI_SINKHORN_SPEC.md`.

### Phase 5 prep checklist (must complete before CRATE-004 builds)

The following APIs are documented in `07_CRATE004_RICCI_SINKHORN_SPEC.md` as
`"add when implementing CRATE-004"`. They must be added to CRATE-002/003 first:

| API | Crate | Purpose |
|-----|-------|---------|
| `ManifoldCollector::remove_node(NodeId) → Result<()>` | genesis-topology | Deregister fused node |
| `HnswGraph::remove_node(NodeId) → Result<()>` | genesis-topology | HNSW cleanup |
| `VFEMinimizer::beliefs_raw(NodeId) → Result<&Belief>` | genesis-dynamics | Raw belief read for centroid |
| `VFEMinimizer::remove_node(NodeId) → Result<()>` | genesis-dynamics | Belief deregister |
| `QuantumKuramotoNetwork::amplitude_norm(NodeId) → f64` | genesis-dynamics | Per-node amplitude read |
| `QuantumKuramotoNetwork::set_amplitude_all_grades(NodeId, f64)` | genesis-dynamics | Post-fusion amplitude set |
| `QuantumKuramotoNetwork::remove_oscillator(NodeId) → Result<()>` | genesis-dynamics | Oscillator deregister |

---

*Verified: `cargo test --workspace` — 385 tests, 0 failures, 0 warnings — v5.3.0-ci-clean (2026-03-06)*
*Contract revision: 4.2.0 (2026-03-09) — Arquitecto Fundacional*
