# CRATE-004 Technical Specification — Ricci Flow, Micro-Sinkhorn, and Inelastic Fusion

**Version:** 1.0.0  
**Status:** SPECIFICATION — Implement in Phase 5  
**Crate:** `genesis-evolution` (CRATE-004)

This document specifies the three performance-critical modules of CRATE-004 whose
interfaces and mathematical structure are determined by the implemented crates.
All types referenced here exist and are tested in CRATE-000 through CRATE-003.

---

## Module 1: `genesis-evolution::sinkhorn`

### Problem: Wasserstein-1 for Ollivier-Ricci curvature

Ollivier-Ricci curvature on edge (i,j) is:

```
K_ij = 1 - W₁(μᵢ, μⱼ) / d(i,j)
```

where W₁ is the Wasserstein-1 distance between neighbourhood distributions
μᵢ, μⱼ (uniform over HNSW neighbours of i and j). Classical Sinkhorn is O(N³)
globally — prohibited at scale.

**Key observation from HNSW geometry:** HNSW uses `M=16` (layer ≥1) and `M0=32`
(layer 0). The neighbourhood distributions μᵢ, μⱼ are supported on at most
`MAX_LOCAL_DEGREE = 32` nodes. The cost matrix is therefore 32×32.

### Micro-Sinkhorn L1: Stateless, Stack-Allocated, Log-Domain

```rust
use genesis_types::constants::{SINKHORN_L1_MAX_ITER, SINKHORN_LOG_EPSILON, SINKHORN_MAX_LOCAL_DEGREE};

/// Cost matrix for local optimal transport. Fits in L1 cache: 32×32×4 = 4096 bytes.
/// Stack-allocated. Zero heap. Stateless — safe for rayon parallel execution per edge.
#[repr(C, align(64))]
struct CostMatrix {
    data: [f32; SINKHORN_MAX_LOCAL_DEGREE * SINKHORN_MAX_LOCAL_DEGREE],
    n_i: usize,  // support size of μᵢ (≤ SINKHORN_MAX_LOCAL_DEGREE)
    n_j: usize,  // support size of μⱼ (≤ SINKHORN_MAX_LOCAL_DEGREE)
}

/// Compute Ollivier-Ricci curvature for a single HNSW edge.
///
/// Runs entirely in L1 cache. No heap allocation. Thread-safe (stateless).
/// Called by rayon parallel iterator over all edges per Ricci step.
///
/// # Algorithm (log-domain Sinkhorn)
///
/// Instead of iterating u ← r / (K·v), v ← c / (Kᵀ·u) in primal domain
/// (where K_uv = exp(-C_uv/ε) risks underflow), the log-domain formulation:
///
/// ```text
/// f ← ε · LogSumExp((-C + g1ᵀ) / ε)   [softmin over columns]
/// g ← ε · LogSumExp((-Cᵀ + f1ᵀ) / ε)  [softmin over rows]
/// ```
///
/// where `LogSumExp(x) = max(x) + ln(Σ exp(xᵢ - max(x)))` is numerically stable.
///
/// Convergence: 12–15 iterations for ε = SINKHORN_LOG_EPSILON on 32×32 matrices.
///
/// # Parameters
/// `coords_i` — G(1,3) grade-1 coordinates of μᵢ neighbours (from VFEMinimizer::mean())
/// `coords_j` — G(1,3) grade-1 coordinates of μⱼ neighbours
/// `d_ij`     — edge distance from fast_metric_distance (already computed in Ricci step)
/// `epsilon`  — SINKHORN_LOG_EPSILON (regularisation)
///
/// # Returns
/// `K_ij = 1.0 - W₁ / d_ij`. Positive = positive curvature (converging, dense region).
///         Negative = negative curvature (diverging, tree-like region).
///
/// AX-ID: AXIOMA-015, H_estructura (LEY_FUNDACIONAL §3.1)
pub fn compute_ollivier_ricci_l1(
    coords_i: &[[f32; 4]],
    coords_j: &[[f32; 4]],
    d_ij: f32,
    epsilon: f32,
) -> f32 {
    debug_assert!(coords_i.len() <= SINKHORN_MAX_LOCAL_DEGREE);
    debug_assert!(coords_j.len() <= SINKHORN_MAX_LOCAL_DEGREE);

    let n = coords_i.len();
    let m = coords_j.len();
    if n == 0 || m == 0 || d_ij < f32::MIN_POSITIVE {
        return 0.0;
    }

    // Cost matrix C[u][v] = ||x_u - y_v||₂² (grade-1 Minkowski, f32 precision).
    // 32×32 = 4096 bytes — guaranteed L1 residence.
    let mut c = [0.0f32; SINKHORN_MAX_LOCAL_DEGREE * SINKHORN_MAX_LOCAL_DEGREE];
    for u in 0..n {
        for v in 0..m {
            let mut sq = 0.0f32;
            for k in 0..4 {
                let d = coords_i[u][k] - coords_j[v][k];
                sq += d * d;
            }
            c[u * SINKHORN_MAX_LOCAL_DEGREE + v] = sq.sqrt();
        }
    }

    // Log-domain Sinkhorn dual potentials (f: n, g: m). Init zeros.
    let mut f = [0.0f32; SINKHORN_MAX_LOCAL_DEGREE];
    let mut g = [0.0f32; SINKHORN_MAX_LOCAL_DEGREE];
    let inv_eps = 1.0 / epsilon;

    for _ in 0..SINKHORN_L1_MAX_ITER {
        // Update f[u] = -ε · LogSumExp_v( (-C[u,v] + g[v]) / ε )
        // Uniform marginals: log(1/m) = -ln(m)
        let ln_m = (m as f32).ln();
        for u in 0..n {
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..m {
                let x = (-c[u * SINKHORN_MAX_LOCAL_DEGREE + v] + g[v]) * inv_eps;
                if x > max_val { max_val = x; }
            }
            let mut lse = 0.0f32;
            for v in 0..m {
                let x = (-c[u * SINKHORN_MAX_LOCAL_DEGREE + v] + g[v]) * inv_eps;
                lse += (x - max_val).exp();
            }
            f[u] = -epsilon * (max_val + lse.ln() - ln_m);
        }

        // Update g[v] = -ε · LogSumExp_u( (-C[u,v] + f[u]) / ε )
        let ln_n = (n as f32).ln();
        for v in 0..m {
            let mut max_val = f32::NEG_INFINITY;
            for u in 0..n {
                let x = (-c[u * SINKHORN_MAX_LOCAL_DEGREE + v] + f[u]) * inv_eps;
                if x > max_val { max_val = x; }
            }
            let mut lse = 0.0f32;
            for u in 0..n {
                let x = (-c[u * SINKHORN_MAX_LOCAL_DEGREE + v] + f[u]) * inv_eps;
                lse += (x - max_val).exp();
            }
            g[v] = -epsilon * (max_val + lse.ln() - ln_n);
        }
    }

    // W₁ ≈ Σ_u f[u]/n + Σ_v g[v]/m  (dual objective of entropic OT)
    let w1: f32 = f[..n].iter().sum::<f32>() / n as f32
                + g[..m].iter().sum::<f32>() / m as f32;

    1.0 - w1 / d_ij
}
```

**Integration with DiscreteRicciFlow:**

```rust
// In DiscreteRicciFlow::step(), called by rayon over edge set:
use rayon::prelude::*;

edge_set.par_iter().for_each(|&(i, j)| {
    let coords_i = manifold.hnsw_neighbor_coords(i);  // [f32;4] per neighbor
    let coords_j = manifold.hnsw_neighbor_coords(j);
    let d_ij = fast_metric_distance(&vecs[i], &vecs[j]) as f32;
    let kappa = compute_ollivier_ricci_l1(&coords_i, &coords_j, d_ij, SINKHORN_LOG_EPSILON);
    // kappa used to update log-weight in module 2
});
```

---

## Module 2: `genesis-evolution::ricci`

### Problem: Edge weight collapse in Ricci flow

Standard Ricci flow `dw/dt = -κ·w` can produce `w < 0` with large `dt` or large
negative κ (negative curvature). Negative weights violate the metric axioms and
produce NaN in HNSW distance computation.

### Log-parametrised Ricci Flow + Inelastic Fusion

```rust
/// Log-parametrised edge weight: l_ij = ln(w_ij) → w_ij = exp(l_ij) > 0 always.
///
/// Ricci flow in log-domain: dl/dt = -κ_ij (additive, no collapse).
/// w_ij can only reach 0 asymptotically — never exactly in finite steps.
///
/// When exp(l_ij) < RICCI_PREEMPTIVE_THRESHOLD, preemptive fusion fires.
///
/// AX-ID: AXIOMA-015, H_estructura (LEY_FUNDACIONAL §3.1)
pub struct RicciEdge {
    pub log_weight: f64,   // l_ij = ln(w_ij). Init: ln(initial_weight).
    pub kappa:      f64,   // Ollivier-Ricci curvature, updated per step.
}

impl RicciEdge {
    pub fn weight(&self) -> f64 { self.log_weight.exp() }

    /// Euler step in log-domain. dt > 0. kappa from micro-Sinkhorn.
    pub fn step(&mut self, dt: f64) {
        self.log_weight -= dt * self.kappa;
        // w_ij = exp(l_ij) > 0 by construction. No clamp needed.
    }

    pub fn needs_fusion(&self) -> bool {
        use genesis_types::constants::RICCI_PREEMPTIVE_THRESHOLD;
        self.weight() < RICCI_PREEMPTIVE_THRESHOLD
    }
}
```

### Inelastic Concept Fusion — Physics-Correct Merge

When `w_ij < RICCI_PREEMPTIVE_THRESHOLD`:

```rust
/// Inelastic collision between two GÉNESIS concepts.
///
/// Conserved quantities:
///   - Total precision (mass): Πk = Πi + Πj
///   - Weighted Clifford centroid: Ck = (Πi·Ci + Πj·Cj) / Πk
///   - Hyperbolic position (Poincaré disk, Lorentz-normalised)
///
/// After fusion: node_j is removed from all data structures.
/// node_i inherits the merged state. VFEMinimizer::update_full() is called
/// with the centroid as a SparseCliffordVector observation.
///
/// Called by DiscreteRicciFlow when edge weight < RICCI_PREEMPTIVE_THRESHOLD.
/// Corresponds to WormholeCollapse::CollapseReason::HighCurvature.
///
/// AX-ID: AXIOMA-015, AXIOMA-016, H_estructura (LEY_FUNDACIONAL §3.1)
pub fn inelastic_concept_fusion(
    node_i: NodeId,
    node_j: NodeId,
    vfe:      &mut VFEMinimizer,
    manifold: &mut ManifoldCollector,
    kuramoto: &mut QuantumKuramotoNetwork,
) -> Result<(), GenesisError> {
    let belief_i = vfe.beliefs_raw(node_i)?;
    let belief_j = vfe.beliefs_raw(node_j)?;

    // Precision sums (mass conservation)
    let prec_i: f64 = belief_i.precision_full.iter().sum();
    let prec_j: f64 = belief_j.precision_full.iter().sum();
    let prec_k = prec_i + prec_j;

    // Weighted Clifford centroid (semantic conservation)
    let centroid: [f64; 16] = core::array::from_fn(|b| {
        (prec_i * belief_i.mean_full[b] + prec_j * belief_j.mean_full[b]) / prec_k
    });

    // Update node_i with centroid observation (uses update_full, precision-weighted)
    let centroid_vec = SparseCliffordVector::from_dense(&centroid)?;
    vfe.update_full(node_i, &centroid_vec, 1.0); // dt=1.0: full adoption

    // Hyperbolic position merge (Poincaré disk, Lorentz normalisation)
    if let (Some(h_i), Some(h_j)) = (
        manifold.hyperbolic_coord(node_i),
        manifold.hyperbolic_coord(node_j),
    ) {
        let w_i = prec_i / prec_k;
        let w_j = prec_j / prec_k;
        let x_k = w_i * h_i.x + w_j * h_j.x;
        let y_k = w_i * h_i.y + w_j * h_j.y;
        // Lorentz-normalise: project back inside unit disk if needed
        let norm = (x_k * x_k + y_k * y_k).sqrt();
        let (x_k, y_k) = if norm >= 1.0 {
            let s = 0.999 / norm;
            (x_k * s, y_k * s)
        } else {
            (x_k, y_k)
        };
        if let Some(coord) = HyperbolicCoord::new(x_k, y_k) {
            manifold.set_hyperbolic_coord(node_i, coord);
        }
    }

    // Merge Kuramoto amplitudes (precision-weighted average)
    let amp_i = kuramoto.amplitude_norm(node_i);
    let amp_j = kuramoto.amplitude_norm(node_j);
    let amp_k = (prec_i * amp_i + prec_j * amp_j) / prec_k;
    kuramoto.set_amplitude_all_grades(node_i, amp_k.clamp(0.0, 1.0));

    // Remove node_j from all structures
    manifold.remove_node(node_j)?;
    kuramoto.remove_oscillator(node_j)?;
    vfe.remove_node(node_j)?;

    Ok(())
}
```

**APIs needed from implemented crates** (add when implementing CRATE-004):
- `VFEMinimizer::beliefs_raw(id) → Result<&Belief>` — raw read for fusion
- `VFEMinimizer::remove_node(id)` — deregister after fusion
- `QuantumKuramotoNetwork::amplitude_norm(id)` — per-node amplitude read
- `QuantumKuramotoNetwork::set_amplitude_all_grades(id, a)` — set post-fusion
- `QuantumKuramotoNetwork::remove_oscillator(id)` — deregister
- `ManifoldCollector::remove_node(id)` — deregister from HNSW + H¹ state

---

## Module 3: `genesis-evolution::compression` (H_compresión)

Uses `phase_diff_norm(i, j)` already exposed by `QuantumKuramotoNetwork` (CRATE-003).
Specification unchanged from `IMPLEMENTATION_ROADMAP.md` Phase 5.6.
No new APIs required from CRATE-003.

---

## Integration Order Within CRATE-004

```
1. sinkhorn.rs     — compute_ollivier_ricci_l1 (no dependencies on other CRATE-004 modules)
2. ricci.rs        — RicciEdge + DiscreteRicciFlow (uses sinkhorn)
3. wormhole.rs     — inelastic_concept_fusion (uses ricci threshold)
4. compression.rs  — compute_h_compression (uses phase_diff_norm from CRATE-003)
5. gram_schmidt.rs — GramSchmidtExpander with DimensionalAdmission
6. heat.rs         — HeatPruner (uses manifold)
7. fisher.rs       — FisherGate satiation wrapper
```

---

## Constants already in `genesis-types`

All constants for this module are defined and tested in `genesis-types::constants`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `SINKHORN_MAX_LOCAL_DEGREE` | 32 | Max cost matrix dimension |
| `SINKHORN_LOG_EPSILON` | 5e-2 | Log-domain regularisation |
| `SINKHORN_L1_MAX_ITER` | 15 | Micro-Sinkhorn iterations |
| `RICCI_PREEMPTIVE_THRESHOLD` | 1e-4 | Fusion trigger |
| `SINKHORN_MAX_ITER` | 1000 | Global Sinkhorn (not used here) |
| `SINKHORN_REGULARISATION` | 5e-2 | Same as LOG_EPSILON |

---

*Updated: 2026-03-06 | Implements: LEY_FUNDACIONAL §3.1 (H_estructura) + §3.7 (H_compresión)*
