# IMPLEMENTATION_ROADMAP.md — GENESIS Cognitive Core

**Version:** 4.0.0  
**Based on:** AXIOMAS.md + LEY_FUNDACIONAL.md v1.2.0 + MACRO_ARCHITECTURE.md v4.0.0

---

## Construction protocol

1. **Phase isolation.** Do not begin Phase N until Phase N−1 passes 100% of its Verification Gate.
2. **Continuous verification.** Unit tests, integration tests, and benchmarks at every phase.
3. **No shortcuts.** No black-box libraries in core paths. No `HashMap`/`BTreeMap` in hot paths.
4. **Dependency tracing.** Strictly per MACRO_ARCHITECTURE §3.
5. **Living documentation.** Every module references its axiom and Hamiltonian term.

---

## Phase 0 — Foundational documentation ✅ COMPLETE

- ✅ LEY_FUNDACIONAL.md v1.2.0
- ✅ AXIOMAS.md v1.0.0
- ✅ MACRO_ARCHITECTURE.md v4.0.0
- ✅ IMPLEMENTATION_ROADMAP.md v4.0.0
- ✅ ENGINEERING_BLUEPRINT_V2.md v4.0.0
- ✅ GENESIS_PROOF_SPEC.md v2.0.0
- ✅ CLOUD_PLATFORM_ARCHITECTURE.md v1.1.0
- ✅ contratos_genesis.md v4.0.0

---

## Phase 1 — Mathematical substrate ✅ COMPLETE

**Crates:** genesis-types (CRATE-000), genesis-math (CRATE-001)

### genesis-types v0.1.0

- ✅ `NodeId` (u64, sentinel = `u64::MAX`)
- ✅ `SpikeEvent`, `SpikeComponents` (168 bytes, Copy, no heap)
- ✅ `DomainSignal<D>`, `DomainConsolidationSignal<State>`, `DomainResetSignal`
- ✅ `GenesisError` with `ErrorPayloadTier` severity
- ✅ `FisherEdgeMetric` (edge-keyed Fisher values for H_dualidad)
- ✅ Proof system: `AxiomID` (0–6), `AxiomSet`, `Proof` (BLAKE3), `AxiomGuard`, `WitnessBuilder`
- ✅ All variational constants including `LAMBDA_DIM_FIXED`, `LAMBDA_DIM_LOG`, `DELTA_DUALITY`, `KAPPA_REDUNDANCY`

### genesis-math v0.2.0

- ✅ `SparseCliffordVector` (160 bytes, align 32, `bytemuck::Pod`, `Copy`)
- ✅ `sparse_geometric_product` — scalar / AVX2 / AVX-512 / NEON dispatch
- ✅ `fast_metric_distance` — **grade-weighted** semantic metric (METRIC_WEIGHTS)
  - Grade 0 (scalar): weight 2.0
  - Grade 1 (vectors, semantic primary): weight 1.5
  - Grade 2 (bivectors, relations): weight 1.0
  - Grade 3 (trivectors): weight 0.5
  - Grade 4 (pseudoscalar): weight 0.3
- ✅ `CAYLEY_SIGN [[i8;16];16]` — compile-time, 256 bytes, `.rodata`
- ✅ Grade projectors: `grade_project`, `grade_project_ct<G>`, `even_grade`, `odd_grade`
- ✅ `CliffordBasis`, `CANONICAL_G13`, `CLIFFORD_NORM_WEIGHTS`
- ✅ `SparseDualVector`, `geometric_product_dual`
- ✅ Feature `deterministic_strict` — Kahan-compensated, bit-exact

**Verification Gate 1: ✅ 61 (genesis-types) + 129 (genesis-math) = 190 tests**

---

## Phase 2 — Topological fabric ✅ COMPLETE

**Crate:** genesis-topology (CRATE-002) v0.1.0

### Implemented

- ✅ `geometric_distance` — primary metric (delegates to `fast_metric_distance`)
- ✅ `bivector_interaction` — grade-2 product norm (not a metric)
- ✅ `fast_bivector_distance` — deprecated alias for `geometric_distance`
- ✅ `HnswGraph` — generation-counter visited tracking, `direct_index`, `FiniteDist`, `hnsw-f16`
- ✅ `IncrementalH1State` — PersistentUnionFind + IncrementalD2, O(1) `h1_is_zero()`
- ✅ `ManifoldCollector`:
  - `compute_edge_density() → f64` — `|E| / (N × log₂N)`
  - `compute_lambda2() → f64` — Lanczos + shifted power iteration
  - `h1_is_zero_fast() → bool` — O(1)
  - `find_affected_nodes()` — for DualityConsistency check
  - **`HyperbolicCoord` contract** — `set_hyperbolic_coord()`, `hyperbolic_coord()`, `hyperbolic_coord_count()`
- ✅ `CohomologyValidator::check_h1` — batch Z₂ elimination
- ✅ `RipsComplex::build` — Vietoris-Rips up to dim 2
- ✅ `CliffordHashTable` — LSH O(log N)
- ✅ `HyperbolicCoord` — Poincaré disk type with `new()` validation, `norm_sq()`, `hyperbolic_distance_to_origin()`

**`HyperbolicCoord` contract:**  
All `hyperbolic_coord()` calls return `None` until CRATE-004 calls `set_hyperbolic_coord()`.
Storage: sorted `Vec<(u64, HyperbolicCoord)>` — binary search, no HashMap.

**Verification Gate 2: ✅ 49 tests (1 stochastic `#[ignore]`)**

---

## Phase 3 — Proof system ✅ COMPLETE (integrated into Phase 1)

The proof system is implemented as part of `genesis-types`. No separate crate.

- ✅ `AxiomID` discriminants 0–6 including `DualityConsistency = 5`, `DimensionalAdmission = 6`
- ✅ `AxiomSet` bitmask (const-constructible)
- ✅ `Proof` with BLAKE3 (not SHA-256)
- ✅ `AxiomGuard::verify` — BLAKE3 integrity + witness replay
- ✅ `WitnessBuilder::check` — per-axiom frame append
- ✅ `EXPANSION_REQUIRED` includes both DualityConsistency + DimensionalAdmission
- ✅ `LAMBDA_DIM_FIXED = 0.05`, `LAMBDA_DIM_LOG = 0.01`

**Verification Gate 3: ✅ Included in Phase 1 count**

---

## Phase 4 — Dynamics and resonance ✅ COMPLETE

**Crate:** genesis-dynamics (CRATE-003) v0.1.0

### Implemented

**Oscillator and Kuramoto:**

- ✅ `QuantumOscillator` with `OscillatorState` (Active → Saturated → Pruned, unidirectional)
- ✅ **`amplitudes: [f64; 5]`** — per-grade inferential certainty, init 1.0
- ✅ **`complex_state(g) → (f64, f64)`** — ψ_{i,g} = A·e^{iφ}, for CRATE-004 cluster amplitude
- ✅ **`amplitude_norm() → f64`** — √(Σ A_g²/5) ∈ [0,1]
- ✅ **`update_amplitude_from_fisher(trace)`** — A[g] = clamp(trace/1.0, 0, 1)
- ✅ `QuantumKuramotoNetwork` — 5 phases/node, Euler-Maruyama, CSR coupling, LCG Box-Muller noise
- ✅ **Amplitude-weighted r_sync**: `r = (1/G) Σ_g |Σ_i A_{i,g}·e^{iφ}| / (Σ_i A_{i,g} + ε)`
  - With A=1.0 (default): identical to classical Kuramoto (backward compatible)
  - With learned nodes (A→0): saturated nodes fade from collective coherence
- ✅ `phase_diff_norm(i, j) → f64` — for H_compresión signal (CRATE-004 interface)
- ✅ `synchrony_order_fast`, `synchrony_order`, `synchronized_cluster`
- ✅ Feature `poly_trig` — Cephes minimax sin/cos, argument-reduction sign correction

**VFE and inference:**

- ✅ **`Belief` — 16D multivector state:**
  - `mean_full: [f64; 16]` — mean over all G(1,3) blades
  - `precision_full: [f64; 16]` — diagonal precision, init 1.0
  - `mean() → [f64; 4]` — grade-1 components, backward compatible
  - `fisher_trace() → f64` — sum of 16 precisions
- ✅ **`FisherInfo`** — `trace`, `delta_g` (unchanged structure, now covers 16 blades)
- ✅ **`VFEMinimizer`:**
  - `compute_vfe(id, &[f64;4])` — grade-1 VFE, backward compatible
  - `compute_vfe_with_grad(node, obs) → (f64, [f64;16])` — **full 16D gradient** for CRATE-004
  - `compute_vfe_with_grad_grade1(node, obs) → (f64, [f64;4])` — convenience wrapper
  - `update(id, &[f64;4], dt)` — grade-1 learning, backward compatible
  - `update_full(id, &SparseCliffordVector, dt)` — **16-blade learning** for CRATE-004
  - `internal_drive() → Option<NodeId>` — 16D VFE internal selection
  - `VFE_BLADE_WEIGHTS: [f64;16]` — mirrors METRIC_WEIGHTS for coherence
  - `GRADE1_BLADE_INDICES: [usize;4] = [1,2,4,8]`
- ✅ `AttractorLandscape` — sorted Vec, `descend` by minimum VFE
- ✅ `CriticalityMonitor` — ring buffer, log-log τ estimation, KS test

**AXIOMA-006 × AXIOMA-008 coupling:**  
After `VFEMinimizer::update()`, call `oscillator.update_amplitude_from_fisher(trace)`.
Saturated nodes (low trace) → low amplitude → suppressed weight in r_sync.
High-VFE nodes (trace≈1.0) → amplitude≈1.0 → dominant weight in r_sync.

**Verification Gate 4: ✅ 79 (genesis-dynamics) + 3 (doc) = 82 tests**

---

## Phase 5 — Learning and evolution 📋 PENDING

**Crate:** genesis-evolution (CRATE-004)

### Contracts provided by earlier phases

All CRATE-004 interfaces are already implemented and tested:

| Interface | Location | Purpose |
|-----------|----------|---------|
| `compute_vfe_with_grad() → [f64;16]` | genesis-dynamics | Full curvature signal for Ricci |
| `complex_state(g) → (re,im)` | genesis-dynamics | Cluster amplitude per grade |
| `phase_diff_norm(i,j)` | genesis-dynamics | H_compresión redundancy detection |
| `set_hyperbolic_coord()` | genesis-topology | Ricci hierarchy embedding |
| `HyperbolicCoord` | genesis-topology | Poincaré disk type |
| `FisherEdgeMetric` | genesis-types | H_dualidad edge Fisher values |
| `AxiomID::DimensionalAdmission` | genesis-types | Dimensional cost gate |
| `AxiomID::DualityConsistency` | genesis-types | Fisher update post-expansion |

### Modules to implement

**5.1 `ricci.rs`** — `DiscreteRicciFlow`  
- Ollivier-Ricci on semantic overlay (NOT on HnswGraph — avoids CRATE-004↔002 cycle)
- Wasserstein-1 via Sinkhorn (`SINKHORN_MAX_ITER = 1000`)
- On each step: reads `compute_vfe_with_grad() → [f64;16]`, calls `set_hyperbolic_coord()`

**5.2 `wormhole.rs`** — `WormholeCollapse`  
```rust
pub enum CollapseReason { HighCurvature, Redundancy }
```
`Redundancy` path activated by H_compresión signal from compression.rs.

**5.3 `gram_schmidt.rs`** — `GramSchmidtExpander`  
Mandatory admission check before any dimension creation:
```rust
let c_dim = LAMBDA_DIM_FIXED + LAMBDA_DIM_LOG * (n_nodes as f64).ln();
assert!(delta_h_total < c_dim);  // AxiomID::DimensionalAdmission
```

**5.4 `heat.rs`** — `HeatPruner::diffuse` — heat equation, preserves H¹

**5.5 `fisher.rs`** — `FisherGate` satiation gate wrapper

**5.6 `compression.rs`** *(H_compresión)*  
```rust
pub fn compute_h_compression(
    network:   &QuantumKuramotoNetwork,
    hnsw:      &HnswGraph,
    epsilon_r: f64,
    theta_r:   f64,
) -> (f64, Vec<(NodeId, NodeId)>);
// Uses phase_diff_norm() — zero additional coupling cost
```

### Verification Gate 5

1. Ricci flow preserves Minkowski signature
2. `WormholeCollapse` triggered by `CollapseReason::Redundancy`
3. `GramSchmidtExpander` checks `DimensionalAdmission` before expansion
4. `compute_h_compression → 0.0` when nodes are phase-differentiated
5. `FisherEdgeMetric::get(i,j)` is symmetric
6. `set_hyperbolic_coord()` called with valid `|coord| < 1` after Ricci step

---

## Phase 6 — Global observer 📋 PENDING

**Crate:** genesis-consciousness (CRATE-006)

### Modules to implement

**6.1 `observer.rs`**  
```rust
pub fn compute_omega_v11(
    r_sync: f64, k_avg: f64, delta_g: f64,
    h1_ok: bool, lambda2: f64, omega_dual: f64,
) -> f64 {
    let base = 0.25*r_sync + 0.20*k_avg.abs() + 0.20/(delta_g+1e-6)
             + 0.15*(if h1_ok {1.0} else {0.0}) + 0.10*lambda2 + 0.10*omega_dual;
    if !h1_ok { base * 0.01 } else { base }
}
```

**6.2 `decision.rs`**  
```rust
pub enum DecisionSignal {
    ExpandDimensionality,
    CollapseWormhole,
    Prune,
    Continue,
    ForceDualUpdate { nodes: Vec<NodeId> },
}
```

**6.3 `duality.rs`**  
```rust
// H_dualidad = δ · Σ_{(i,j)} ||g_{ij} - G^Fisher_{ij}||²
pub fn compute_h_duality(metric: &EdgeMetric, fisher: &FisherEdgeMetric) -> f64;
// Ω_dual = (1 - h_duality/h_duality_max).max(0.0)
pub fn compute_omega_dual(h_duality: f64, h_duality_max: f64) -> f64;
pub fn find_dual_violations(metric: &EdgeMetric, fisher: &FisherEdgeMetric, tol: f64) -> Vec<NodeId>;
```

### Verification Gate 6

1. `Ω_dual = 1.0` when `g = G^Fisher` everywhere
2. `ForceDualUpdate` emitted when `Ω_dual < 0.5`
3. `H¹ ≠ 0 → Ω × 0.01` coherence collapse
4. `compute_h_duality → 0.0` when structure and inference are isometric
5. `compute_omega_v11` with all-zero inputs is finite and non-negative

---

## Phase 7 — I/O gateway 📋 PENDING

**Crate:** genesis-io (CRATE-005)

Homeomorphic projection Π (sensory → G(1,3)) via GCNN with TDA persistence loss.
Adjoint Π* (G(1,3) → output). AXIOMA-017, AXIOMA-018, AXIOMA-019. Implement last.

---

## Stop and resume protocol

If a blocker arises in any phase:
1. Document the exact module, function, and failing test.
2. Deliver a status report.
3. Wait for "Continue" before proceeding.

---

## Phase summary

| Phase | Crate | Status | Tests |
|-------|-------|--------|-------|
| 0 | Documentation | ✅ Complete | — |
| 1 | genesis-types (+ proof) + genesis-math | ✅ Complete | **190** |
| 2 | genesis-topology + HyperbolicCoord | ✅ Complete | **49** |
| 3 | Proof system (in genesis-types) | ✅ Complete | (in Phase 1) |
| 4 | genesis-dynamics + amplitudes + Belief 16D | ✅ Complete | **82** |
| 5 | genesis-evolution | 📋 Pending | — |
| 6 | genesis-consciousness | 📋 Pending | — |
| 7 | genesis-io | 📋 Pending | — |

**Current total: 384 tests, 0 failures.**

Implemented crates expose all contracts needed by CRATE-004 and CRATE-006.
No blocking dependencies remain for Phase 5 or 6.

---

*Updated: 2026-03-06 | Based on: LEY_FUNDACIONAL.md v1.2.0*
