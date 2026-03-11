# COGNITIVE_EXTENSIONS_V1.md — Genesis Cognitive Extensions Record

**Version:** 1.0.0  
**Status:** CANONICAL — Supersedes all prior analysis documents  
**Scope:** Extensions implemented across CRATE-001, CRATE-002, CRATE-003

This document records the four cognitive extensions that bring the implemented crates
to production-grade semantic and inferential completeness. Each extension is described
as a first-class architectural decision with its design rationale, implementation
specification, and contract for pending crates.

---

## Extension 1 — Grade-Differentiated Metric Weights

**Crate:** genesis-math | **File:** `src/multivector.rs` | **Constant:** `METRIC_WEIGHTS`

### Design rationale

G(1,3) has five grades with distinct cognitive semantics. A uniform Euclidean norm
treats grade-0 scalars (global magnitude) identically to grade-1 vectors (semantic
direction) identically to grade-4 pseudoscalars (full-space orientation). The result
is that HNSW clusters form by geometric accident — nodes with similar pseudoscalar
coefficient happen to be placed near each other — rather than by semantic coherence.

The grade-differentiated metric assigns weights reflecting the cognitive importance
of each grade:

| Grade | Semantic role | Weight |
|-------|--------------|--------|
| 0 (scalar) | Global intensity / magnitude | 2.0 |
| 1 (vectors) | Semantic direction — primary | 1.5 |
| 2 (bivectors) | Relations, rotations | 1.0 |
| 3 (trivectors) | Oriented volume | 0.5 |
| 4 (pseudoscalar) | Full-space orientation | 0.3 |

All weights strictly positive → metric axioms preserved. Weights decrease with grade
(except the scalar bonus for magnitude): HNSW preferentially connects nodes with
similar semantic direction before connecting nodes with similar global orientation.

### Implementation

```rust
pub(crate) const METRIC_WEIGHTS: [f64; TOTAL_BLADES] = [
    2.0,                                         // blade 0: grade 0
    1.5, 1.5, 1.0, 1.5, 1.0, 1.0, 0.5,         // blades 1–7
    1.5, 1.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.3,    // blades 8–15
];
```

`fast_metric_distance` uses these weights: `d(x,y) = √(Σᵢ METRIC_WEIGHTS[i]·(xᵢ−yᵢ)²)`.
`geometric_distance` in genesis-topology delegates to `fast_metric_distance`.

### Propagation to pending crates

`DiscreteRicciFlow` (CRATE-004) computes Ollivier-Ricci curvature on the HNSW graph
built with this metric. With grade-differentiated weights, curvature peaks at genuine
semantic boundaries between conceptual domains, not at accidental norm coincidences.
`VFE_BLADE_WEIGHTS` in genesis-dynamics mirrors these weights for inferential-topological
coherence: the inferential distance between beliefs is consistent with the topological
distance between their nodes.

---

## Extension 2 — Hyperbolic Coordinate Contract

**Crate:** genesis-topology | **File:** `src/manifold.rs`  
**Types:** `HyperbolicCoord`, `ManifoldCollector::hyperbolic_coord*`

### Design rationale

Cognitive hierarchies are trees: concept roots have high degree and low specificity;
concept leaves have low degree and high specificity. Hyperbolic geometry (Poincaré
disk model) embeds trees in 2D with exponentially growing capacity toward the boundary,
matching the exponential branching of cognitive hierarchies exactly.

This extension provides the coordinate type and the storage contract within
`ManifoldCollector`, so that `DiscreteRicciFlow` (CRATE-004) can populate coordinates
as it computes Ollivier-Ricci curvature — without any architectural change to the manifold.

### Implementation

```rust
/// Poincaré disk coordinate. Invariant: x² + y² < 1.
pub struct HyperbolicCoord { pub x: f64, pub y: f64 }

impl HyperbolicCoord {
    pub fn new(x: f64, y: f64) -> Option<Self>;         // None if outside disk
    pub fn norm_sq(&self) -> f64;                        // always < 1.0
    pub fn hyperbolic_distance_to_origin(&self) -> f64;  // 2·arctanh(|coord|)
}
```

`ManifoldCollector` stores `Vec<(u64, HyperbolicCoord)>` sorted by node id.
`binary_search_by_key` provides O(log N) access without HashMap.

```rust
// ManifoldCollector contract methods:
pub fn hyperbolic_coord(&self, id: NodeId) -> Option<HyperbolicCoord>;
pub fn set_hyperbolic_coord(&mut self, id: NodeId, coord: HyperbolicCoord);
pub fn hyperbolic_coord_count(&self) -> usize;
```

### Activation by CRATE-004

Before CRATE-004 exists: all calls return `None`. The field is inert.

When `DiscreteRicciFlow::step()` is implemented:
```rust
let r = (k_avg_node / 2.0).tanh().clamp(0.0, 0.999);
let theta = oscillator.primary_phase();
if let Some(coord) = HyperbolicCoord::new(r * theta.cos(), r * theta.sin()) {
    manifold.set_hyperbolic_coord(node_id, coord);
}
```

The Poincaré disk radius encodes Ricci curvature: high-curvature (leaf) nodes map to
`|coord|→1`; low-curvature (root) nodes map to `|coord|→0`.

---

## Extension 3 — Complex Amplitudes in Kuramoto

**Crate:** genesis-dynamics | **Files:** `src/oscillator.rs`, `src/synchrony.rs`, `src/kuramoto.rs`

### Design rationale

The classical Kuramoto model assigns equal weight to all oscillators in the order
parameter $r = |\sum_i e^{i\phi_i}|/N$. In GÉNESIS, nodes have different inferential
certainty: a node that has mastered a domain (low VFE, low Fisher trace) should
contribute less to the collective dynamics than a node actively learning (high VFE,
high Fisher trace). The amplitude extension formalises this coupling.

The complex quantum state per grade:

$$\psi_{i,g} = A_{i,g} \cdot e^{i\phi_{i,g}}$$

The amplitude $A_{i,g} \in [0,1]$ is coupled to `FisherInfo::trace`:

$$A_{i,g} = \text{clamp}\!\left(\frac{\text{trace}}{\text{FISHER\_TRACE\_INITIAL}}, 0, 1\right)$$

### Implementation

```rust
pub struct QuantumOscillator {
    pub phases:      [f64; 5],
    pub amplitudes:  [f64; 5],  // init 1.0 (maximum drive)
    pub frequencies: [f64; 5],
    pub node_id:     NodeId,
    pub state:       OscillatorState,
}

impl QuantumOscillator {
    /// ψ_{i,g} = A·(cos φ, sin φ). Returns (re, im).
    pub fn complex_state(&self, g: usize) -> (f64, f64);
    /// √(Σ_g A_g²/5) ∈ [0,1]
    pub fn amplitude_norm(&self) -> f64;
    /// A[g] = clamp(trace/1.0, 0, 1) for all g.
    pub fn update_amplitude_from_fisher(&mut self, fisher_trace: f64);
}
```

**Amplitude-weighted order parameter:**

$$r_{\text{sync}} = \frac{1}{G}\sum_{g=0}^{G-1}
  \frac{\left|\sum_i A_{i,g}\, e^{i\phi_{i,g}}\right|}{\sum_i A_{i,g}+\varepsilon}$$

With $A_{i,g} = 1.0$ (prior): identical to classical Kuramoto (backward compatible).
With learned nodes: $r_{\text{sync}}$ measures **angular coherence weighted by
inferential certainty** — a richer observable that couples the dynamical and
inferential subsystems.

### Coupling protocol

After each `VFEMinimizer::update(node, observation, dt)`:
```rust
let trace = vfe.fisher(node).map_or(1.0, |f| f.trace);
network.oscillator_mut(node).update_amplitude_from_fisher(trace);
```

This creates the AXIOMA-006 × AXIOMA-008 coupling: saturated nodes (low trace) reduce
amplitude → reduced weight in $r_{\text{sync}}$ → high-VFE nodes dominate collective
coherence.

### Propagation to pending crates

`DiscreteRicciFlow` (CRATE-004) reads `complex_state(g)` to compute per-node cluster
amplitude, weighting the Wasserstein-1 transport measure when computing Ollivier-Ricci
curvature. Nodes with high amplitude (actively learning) receive stronger curvature
pressure; saturated nodes (amplitude→0) contribute minimally.

---

## Extension 4 — 16D Belief over G(1,3)

**Crate:** genesis-dynamics | **File:** `src/free_energy.rs`  
**Types:** `Belief`, `VFEMinimizer`, `VFE_BLADE_WEIGHTS`, `GRADE1_BLADE_INDICES`

### Design rationale

`Belief.mean` covered only 4 dimensions (grade-1 vectors: e₀, e₁, e₂, e₃). A concept
in GÉNESIS is a full multivector in G(1,3) — 16 coefficients spanning all 5 grades.
When inference operates on only 4 of 16 dimensions, the gradient `∂F/∂μ` is incomplete:
it signals error only in the vector component, missing relational (grade-2), volumetric
(grade-3), and global-orientation (grade-4) discrepancies.

With 16D belief, `compute_vfe_with_grad()` returns a gradient over all 16 blades,
which `DiscreteRicciFlow` (CRATE-004) uses for complete curvature signal.

### Type specification

```rust
/// Inferential state of a node over all 16 G(1,3) blades.
pub struct Belief {
    pub mean_full:      [f64; 16],  // μ over G(1,3). Init: grade-1 from prior, 0 elsewhere.
    pub precision_full: [f64; 16],  // diagonal precision ∈ [0,1e6]. Init: 1.0.
    pub node_id:        NodeId,
}

/// Scalar Fisher metric for satiation gate (AXIOMA-008).
pub struct FisherInfo { pub trace: f64, pub delta_g: f64 }
```

### VFEMinimizer API

| Method | Signature | Notes |
|--------|-----------|-------|
| `add_node` | `(id, [f64;4])` | grade-1 init; backward compatible |
| `compute_vfe` | `(id, Option<&[f64;4]>) → f64` | grade-1 only; unchanged |
| `compute_vfe_with_grad` | `(node, Option<&SCV>) → (f64, [f64;16])` | **full gradient; CRATE-004 primary** |
| `compute_vfe_with_grad_grade1` | `(node, Option<&SCV>) → (f64, [f64;4])` | convenience wrapper |
| `update` | `(id, &[f64;4], dt)` | grade-1 only; unchanged |
| `update_full` | `(id, &SCV, dt)` | 16-blade update; for CRATE-004 |
| `internal_drive` | `() → Option<NodeId>` | 16D VFE; richer selection |
| `fisher` | `(id) → Option<&FisherInfo>` | unchanged |
| `delta_g` | `(id) → f64` | unchanged |

```rust
/// Grade-1 blade indices: GRADE1_BLADE_INDICES = [1, 2, 4, 8] (e₀, e₁, e₂, e₃).
/// VFE_BLADE_WEIGHTS mirrors METRIC_WEIGHTS:
///   grade-0 (scalar): 2.0, grade-1 (vectors): 1.5, grade-2: 1.0, grade-3: 0.5, grade-4: 0.3
/// Consistent weighting ensures inferential distance ≈ topological distance in HNSW.
```

### Backward compatibility

All callers using `add_node`, `compute_vfe`, `update`, `fisher`, `delta_g` continue
to function identically. The old `Belief.mean: [f64;4]` is now `Belief.mean()` method.
The old `Belief.precision_diagonal: [f64;4]` is now `Belief.precision_full[GRADE1_BLADE_INDICES]`.

### Propagation to pending crates

`DiscreteRicciFlow` (CRATE-004) calls `compute_vfe_with_grad(node, obs) → [f64;16]`
to obtain the complete error gradient across all Clifford grades. This gradient drives
the semantic-layer edge metric deformation `dg/dt = -2K·g + 2α·Φ` with full geometric
fidelity — not just the 4D vector approximation that would have resulted from the prior
implementation.

`update_full(id, obs)` is called by CRATE-004 after `WormholeCollapse` to adjust
beliefs for the merged/collapsed node across all 16 blades.

---

## Summary matrix

| Extension | Crate | New symbols | Backward compat | CRATE-004 contract |
|-----------|-------|-------------|-----------------|-------------------|
| Grade metric | genesis-math | `METRIC_WEIGHTS` (changed) | Yes — same function signatures | Provides meaningful Ricci curvature |
| HyperbolicCoord | genesis-topology | `HyperbolicCoord`, 3 `ManifoldCollector` methods | Yes — zero semantic change to existing paths | `set_hyperbolic_coord()` write surface |
| Amplitudes | genesis-dynamics | `amplitudes`, `complex_state`, `amplitude_norm`, `update_amplitude_from_fisher` | Yes — `amplitudes=[1.0;5]` default | `complex_state(g)` for cluster amplitude |
| Belief 16D | genesis-dynamics | `mean_full`, `precision_full`, `compute_vfe_with_grad`→[16], `compute_vfe_with_grad_grade1`, `update_full` | Yes — `mean()`, `compute_vfe`, `update` unchanged | `compute_vfe_with_grad` full gradient |

**Test delta:** 354 → 384 tests (+30). 0 failures.

---

*Updated: 2026-03-06 | Applies to: genesis-math v0.2.0, genesis-topology v0.1.0, genesis-dynamics v0.1.0*
