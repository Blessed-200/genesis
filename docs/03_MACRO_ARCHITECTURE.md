# MACRO_ARCHITECTURE.md — GENESIS Cognitive Core

**Rev:** 4.0.0  
**Status:** CANONICAL  
**Owner:** Principal Software Architect

---

## 0. Variational Framework (LEY_FUNDACIONAL v1.2.0)

$$H_{\text{total}} = \sum_{i=1}^7 \alpha_i H_i$$

### Hamiltonian components

| Term | Description | Implementing crate | Status |
|------|-------------|-------------------|--------|
| $H_{\text{estructura}}$ | Geometric energy (grade-weighted metric, Ricci) | genesis-math, genesis-evolution | math ✅, evolution Phase 5 |
| $H_{\text{dinámica}}$ | Phase synchronisation (Kuramoto + amplitudes) | genesis-dynamics | ✅ |
| $H_{\text{información}}$ | Active inference (VFE 16D + Fisher) | genesis-dynamics | ✅ |
| $H_{\text{teleología}}$ | Discovery pressure (GramSchmidt) | genesis-consciousness | Phase 6 |
| $H_{\text{restricción}}$ | Hard penalties (H¹, λ₂, proof, density) | genesis-topology, genesis-types | ✅ |
| $H_{\text{dualidad}}$ | Structure↔inference symmetry | genesis-consciousness | Phase 6 |
| $H_{\text{compresión}}$ | Representational redundancy | genesis-evolution | Phase 5 |

### Global observable Ω

$$\Omega = \alpha_1 r_{\text{sync}} + \alpha_2 |K_{\text{avg}}| + \alpha_3 \frac{1}{\Delta G + \epsilon} + \alpha_4 \mathbb{1}_{H^1=0} + \alpha_5 \lambda_2 + \alpha_6 \Omega_{\text{dual}}$$

Computed by genesis-consciousness (Phase 6). Currently implemented components:
- $r_{\text{sync}}$ — amplitude-weighted: `synchrony_order` in genesis-dynamics ✅  
- $\mathbb{1}_{H^1=0}$ — `h1_is_zero_fast` in genesis-topology ✅  
- $\lambda_2$ — `compute_lambda2` in genesis-topology ✅  
- $\Delta G$ — `delta_g` in genesis-dynamics ✅  

### Mutation guard

All structural mutations generate a BLAKE3-hashed `Proof` before execution.
`AxiomID::DualityConsistency` (Fisher consistency post-expansion) and
`AxiomID::DimensionalAdmission` (ΔH_total < C_dim(N)) required for dimensional expansion.

---

## 1. Workspace layout

```
genesis/
│
├── Cargo.toml
├── Cargo.lock
│
├── core/
│   ├── genesis-math/                    ← CRATE-001 ✅
│   │   └── src/
│   │       ├── multivector.rs           ← SparseCliffordVector, METRIC_WEIGHTS (grade-weighted)
│   │       ├── product.rs               ← sparse_geometric_product, SIMD dispatch
│   │       ├── basis.rs                 ← CliffordBasis, CANONICAL_G13
│   │       ├── sign.rs                  ← CAYLEY_SIGN [[i8;16];16], compute_clifford_sign
│   │       ├── grade.rs                 ← grade_project, CLIFFORD_NORM_WEIGHTS
│   │       └── dual.rs                  ← SparseDualVector
│   │
│   ├── genesis-topology/                ← CRATE-002 ✅
│   │   └── src/
│   │       ├── geodesic.rs              ← geometric_distance (primary), bivector_interaction
│   │       ├── hnsw.rs                  ← HnswGraph (hnsw-f16 feature)
│   │       ├── lsh.rs                   ← CliffordHashTable
│   │       ├── rips.rs                  ← RipsComplex
│   │       ├── cohomology.rs            ← CohomologyValidator (batch Z₂)
│   │       ├── incremental_cohomology.rs← IncrementalH1State (O(1) path)
│   │       └── manifold.rs              ← ManifoldCollector, HyperbolicCoord
│   │
│   ├── genesis-dynamics/                ← CRATE-003 ✅
│   │   └── src/
│   │       ├── oscillator.rs            ← QuantumOscillator (phases + amplitudes)
│   │       ├── kuramoto.rs              ← QuantumKuramotoNetwork, amplitude-weighted r_sync
│   │       ├── synchrony.rs             ← synchrony_order_fast (amplitude-weighted)
│   │       ├── free_energy.rs           ← VFEMinimizer, Belief (16D), FisherInfo
│   │       ├── attractor.rs             ← AttractorLandscape
│   │       └── criticality.rs           ← CriticalityMonitor
│   │
│   ├── genesis-evolution/               ← CRATE-004 Phase 5
│   │   └── src/
│   │       ├── ricci.rs                 ← DiscreteRicciFlow (Ollivier-Ricci + Sinkhorn)
│   │       ├── wormhole.rs              ← WormholeCollapse {HighCurvature, Redundancy}
│   │       ├── gram_schmidt.rs          ← GramSchmidtExpander + DimensionalAdmission
│   │       ├── heat.rs                  ← HeatPruner
│   │       ├── fisher.rs                ← FisherGate
│   │       └── compression.rs           ← compute_h_compression (H_compresión signal)
│   │
│   └── genesis-consciousness/           ← CRATE-006 Phase 6
│       └── src/
│           ├── observer.rs              ← GlobalObserver, compute_omega_v11
│           ├── decision.rs              ← DecisionSignal {ForceDualUpdate, ...}
│           └── duality.rs               ← compute_h_duality, compute_omega_dual
│
├── drivers/
│   └── genesis-io/                      ← CRATE-005 Phase 7
│
└── shared/
    └── genesis-types/                   ← CRATE-000 ✅
        └── src/
            ├── constants.rs             ← all variational parameters
            ├── error.rs                 ← GenesisError
            ├── signal.rs                ← NodeId, SpikeEvent, DomainSignal
            └── proof.rs                 ← AxiomID (0-6), Proof, AxiomGuard, WitnessBuilder
```

---

## 2. Crate definitions

### CRATE-000 — `genesis-types` ✅

| Field | Value |
|-------|-------|
| **Axiom** | Cross-cutting |
| **Key exports** | `NodeId`, `GenesisError`, `FisherEdgeMetric`, `AxiomID` (0-6), `Proof`, `AxiomGuard`, `WitnessBuilder`, all variational constants |
| **H term** | — (primitives) |

**Constants (complete list):**

```rust
// Cognitive physics
pub const COGNITIVE_PLANCK_CONSTANT: f64 = 1e-12;

// H_restricción parameters
pub const LAMBDA2_MIN:              f64 = 0.1;
pub const PROOF_PENALTY:            f64 = 1000.0;
pub const DENSITY_PENALTY:          f64 = 50.0;

// H_dualidad
pub const DELTA_DUALITY:            f64 = 0.01;

// H_compresión
pub const KAPPA_REDUNDANCY:         f64 = 0.05;
pub const REDUNDANCY_RADIUS:        f64 = 0.1;
pub const PHASE_DISTINCTION_THRESHOLD: f64 = 0.3;

// Dimensional admission (LEY_FUNDACIONAL §3.8)
pub const LAMBDA_DIM_FIXED:         f64 = 0.05;
pub const LAMBDA_DIM_LOG:           f64 = 0.01;

// Proof system
pub const PROOF_MAX_AGE_NS:         u64 = 5_000_000_000;
```

---

### CRATE-001 — `genesis-math` ✅

| Field | Value |
|-------|-------|
| **Axiom** | AXIOMA-001, AXIOMA-002 |
| **Key exports** | `SparseCliffordVector`, `fast_metric_distance`, `METRIC_WEIGHTS`, `CAYLEY_SIGN`, grade projectors |
| **H term** | $H_{\text{estructura}}$ (implicit) |

**`METRIC_WEIGHTS` — grade-differentiated semantic metric:**

Blade weights by grade: 0→2.0, 1→1.5, 2→1.0, 3→0.5, 4→0.3.
HNSW clusters form by semantic direction (grade 1) before global orientation (grade 4).
Consistent with `VFE_BLADE_WEIGHTS` in genesis-dynamics for inferential-topological coherence.

---

### CRATE-002 — `genesis-topology` ✅

| Field | Value |
|-------|-------|
| **Axiom** | AXIOMA-007, AXIOMA-009, AXIOMA-010, AXIOMA-013 |
| **Key exports** | `geometric_distance`, `HnswGraph`, `ManifoldCollector`, `HyperbolicCoord`, `IncrementalH1State`, `CohomologyValidator`, `CliffordHashTable`, `RipsComplex` |
| **H term** | $H_{\text{restricción}}$ (H¹, λ₂, edge density) |
| **Ω contribution** | λ₂, H¹, edge_density_ratio |

**`HyperbolicCoord` contract:**  
Present in `ManifoldCollector` as `Option<HyperbolicCoord>` per node (binary-searched
sorted Vec, no HashMap). Returns `None` until `DiscreteRicciFlow` (CRATE-004) populates
coordinates via `set_hyperbolic_coord()`. Encodes Ricci hierarchy in Poincaré disk.

---

### CRATE-003 — `genesis-dynamics` ✅

| Field | Value |
|-------|-------|
| **Axiom** | AXIOMA-003, AXIOMA-004, AXIOMA-005, AXIOMA-006, AXIOMA-008 |
| **Key exports** | `QuantumOscillator` (+ amplitudes), `QuantumKuramotoNetwork`, `VFEMinimizer` (Belief 16D), `FisherInfo`, `AttractorLandscape`, `CriticalityMonitor` |
| **H term** | $H_{\text{dinámica}}$, $H_{\text{información}}$ |
| **Ω contribution** | $r_{\text{sync}}$ (amplitude-weighted), $\Delta G$ |

**QuantumOscillator amplitudes:**  
`amplitudes: [f64; 5]`, init 1.0. Coupled to Fisher via `update_amplitude_from_fisher()`.
`complex_state(g) → (f64, f64)`: exposes ψ_{i,g} = A·e^{iφ} for CRATE-004.
`r_sync` is amplitude-weighted: saturated nodes fade from collective coherence.

**Belief 16D:**  
`mean_full: [f64; 16]`, `precision_full: [f64; 16]`.
`compute_vfe_with_grad() → (f64, [f64; 16])`: full gradient signal for CRATE-004.
`update_full(&SparseCliffordVector)`: 16-blade learning for post-collapse adjustments.
`mean() → [f64; 4]` and `compute_vfe(, &[f64;4])`: fully backward compatible.

---

### CRATE-004 — `genesis-evolution` (Phase 5)

| Field | Value |
|-------|-------|
| **Axiom** | AXIOMA-008, AXIOMA-014, AXIOMA-015, AXIOMA-016 |
| **Planned modules** | ricci, wormhole, gram_schmidt, heat, fisher, compression |
| **H term** | $H_{\text{estructura}}$ (Ricci), $H_{\text{información}}$ (Fisher gate), $H_{\text{compresión}}$ |
| **Ω contribution** | K_avg, ΔG |

**Contracts ready for CRATE-004:**  
`VFEMinimizer::compute_vfe_with_grad() → [f64;16]` ✅  
`QuantumOscillator::complex_state(g)` ✅  
`ManifoldCollector::set_hyperbolic_coord()` ✅  
`FisherEdgeMetric` in genesis-types ✅  
`phase_diff_norm(i,j)` in genesis-dynamics ✅  

---

### CRATE-006 — `genesis-consciousness` (Phase 6)

| Field | Value |
|-------|-------|
| **Axiom** | Observador global + gradient evolution |
| **Planned modules** | observer, decision, duality |
| **H term** | $H_{\text{teleología}}$, $H_{\text{dualidad}}$ |
| **Ω contribution** | Full Ω computation including Ω_dual |

```rust
// Planned API:
pub fn compute_omega_v11(r_sync, k_avg, delta_g, h1_ok, lambda2, omega_dual) -> f64;

pub enum DecisionSignal {
    ExpandDimensionality,
    CollapseWormhole,
    Prune,
    Continue,
    ForceDualUpdate { nodes: Vec<NodeId> },
}
```

---

### CRATE-005 — `genesis-io` (Phase 7)

Homeomorphic projection Π (sensory → G(1,3)) via GCNN with TDA persistence loss,
and adjoint Π* (G(1,3) → output). AXIOMA-017, AXIOMA-018, AXIOMA-019.

---

## 3. Dependency graph

```
genesis-io [CRATE-005]
    │
    ├── genesis-dynamics      [003]
    ├── genesis-evolution     [004]
    └── genesis-consciousness [006]
                │
                ├── genesis-topology  [002]
                │           │
                │           └── genesis-math [001]
                │                       │
                │                       └── genesis-types [000]
                └── (all of the above)
```

No circular dependencies. `compression.rs` (CRATE-004) accesses genesis-topology (HNSW)
and genesis-dynamics (phases) — both are upstream. `duality.rs` (CRATE-006) accesses
genesis-evolution (Fisher metric) and genesis-dynamics — both upstream.

---

## 4. Interface contracts between crates

### genesis-math → genesis-topology
```rust
pub fn fast_metric_distance(a: &SparseCliffordVector, b: &SparseCliffordVector) -> f64;
pub(crate) const METRIC_WEIGHTS: [f64; 16];  // grade-weighted
```

### genesis-dynamics → genesis-evolution (CRATE-004)
```rust
// phase signal for H_compresión
pub fn phase_diff_norm(&self, i: usize, j: usize) -> f64;
// full gradient for Ricci flow
pub fn compute_vfe_with_grad(
    &self, node: NodeId, obs: Option<&SparseCliffordVector>,
) -> (f64, [f64; 16]);
// cluster amplitude for Ricci curvature weighting
pub fn complex_state(&self, g: usize) -> (f64, f64);  // on QuantumOscillator
```

### genesis-evolution → genesis-topology (CRATE-004 writes)
```rust
pub fn set_hyperbolic_coord(&mut self, id: NodeId, coord: HyperbolicCoord);
```

### genesis-types → all crates
```rust
pub struct FisherEdgeMetric { ... }  // edge-keyed Fisher values for H_dualidad
pub enum AxiomID { ..., DualityConsistency = 5, DimensionalAdmission = 6 }
pub const LAMBDA_DIM_FIXED: f64;
pub const LAMBDA_DIM_LOG:   f64;
```

---

## 5. Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "shared/genesis-types",
    "core/genesis-math",
    "core/genesis-topology",
    "core/genesis-dynamics",
]

[workspace.dependencies]
genesis-types    = { path = "shared/genesis-types" }
genesis-math     = { path = "core/genesis-math" }
genesis-topology = { path = "core/genesis-topology" }
genesis-dynamics = { path = "core/genesis-dynamics" }

thiserror         = "1.0"
crossbeam         = "0.8"
rayon             = "1.10"
tokio             = { version = "1.38", features = ["full"] }
serde             = { version = "1.0", features = ["derive"], optional = true }
blake3            = "=1.5.4"
num-complex       = "0.4"
bytemuck          = "1.14"
static_assertions = "1.1"
proptest          = "1.4"
criterion         = { version = "0.5", features = ["html_reports"] }
```

---

**END OF MACRO_ARCHITECTURE.md v4.0.0**

Updated: 2026-03-06 | Based on: LEY_FUNDACIONAL.md v1.2.0 | 384 tests, 0 failures
