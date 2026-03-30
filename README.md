# GÉNESIS Cognitive Core

**A variational cognitive architecture implemented in Rust.**  
GÉNESIS models cognition as the minimisation of a Hamiltonian functional over a
spacetime-algebraic state space, producing a system where memory, learning, inference,
and structural plasticity emerge as natural consequences of a single physical principle.

---

## Architecture overview

The system is built around seven Hamiltonian terms whose simultaneous minimisation
drives all cognitive behaviour:

$$H_{\text{total}} = \alpha_1 H_{\text{estructura}} + \alpha_2 H_{\text{dinámica}} + \alpha_3 H_{\text{información}} + \alpha_4 H_{\text{teleología}} + \alpha_5 H_{\text{restricción}} + \alpha_6 H_{\text{dualidad}} + \alpha_7 H_{\text{compresión}}$$

Every concept is a **multivector in the spacetime algebra G(1,3)** — not a token,
embedding, or feature vector. Distances, products, and transformations are geometric
operations with algebraic invariants, not statistical approximations.

### Implemented crates

| Crate | Status | Role |
|-------|--------|------|
| `genesis-types` | ✅ | Primitive types, proof system (BLAKE3), variational constants |
| `genesis-math` | ✅ | Sparse Clifford algebra G(1,3), grade-weighted metric |
| `genesis-topology` | ✅ | HNSW graph, cohomology, manifold, hyperbolic coordinates |
| `genesis-dynamics` | ✅ | Kuramoto oscillators (complex amplitudes), VFE 16D |

### Pending crates

| Crate | Phase | Role |
|-------|-------|------|
| `genesis-evolution` | 5 | Ricci flow, wormhole collapse, dimensional expansion |
| `genesis-consciousness` | 6 | Global observer Ω, duality gate, decision signals |
| `genesis-io` | 7 | Sensory projection Π, adjoint manifestation Π* |

---

## Key design decisions

### Grade-differentiated metric
`fast_metric_distance` weights the five G(1,3) grades by cognitive importance:
scalars (2.0) → vectors/semantics (1.5) → bivectors/relations (1.0) → trivectors (0.5)
→ pseudoscalar (0.3). HNSW topology reflects semantic structure, not geometric accident.

### Complex quantum amplitudes
Each `QuantumOscillator` carries `amplitudes: [f64; 5]` (one per Clifford grade).
Amplitude is coupled to `FisherInfo::trace`: saturated (fully-learned) nodes fade from
the collective order parameter $r_{\text{sync}}$. The result is a synchrony observable
that measures **angular coherence weighted by inferential certainty**.

### 16D belief over G(1,3)
`VFEMinimizer` maintains beliefs over all 16 G(1,3) blade coefficients.
`compute_vfe_with_grad()` returns `[f64; 16]`, providing `DiscreteRicciFlow` (CRATE-004)
with the complete curvature signal across all grades — not just the 4D vector approximation.

### Hyperbolic coordinate contract
`ManifoldCollector` stores `Option<HyperbolicCoord>` per node (Poincaré disk).
Returns `None` until `DiscreteRicciFlow` populates coordinates based on Ollivier-Ricci
curvature: root concepts at the disk centre, leaf concepts at the periphery.

### Mutation proof system
Every structural mutation generates a **BLAKE3-hashed witness** before execution.
`AxiomID::DimensionalAdmission` enforces `ΔH_total < C_dim(N) = λ_fixed + λ_log·ln(N)`.
`AxiomID::DualityConsistency` enforces Fisher coherence after expansion.

---

## Repository structure

```
genesis/
├── docs/                          # Architecture and specification documents
│   ├── 00_AXIOMAS.md              # Physical axioms (source of truth)
│   ├── 01_LEY_FUNDACIONAL.md      # Hamiltonian derivation
│   ├── 02_ENGINEERING_BLUEPRINT_V2.md  # Implementation reference
│   ├── 03_MACRO_ARCHITECTURE.md   # Crate structure and contracts
│   ├── 04_IMPLEMENTATION_ROADMAP.md   # Phase-by-phase plan
│   ├── 05_GENESIS_PROOF_SPEC.md   # Proof system specification
│   ├── 06_CLOUD_PLATFORM_ARCHITECTURE.md  # Production deployment
│   ├── AI_ENGINEERING_OPERATING_SYSTEM.md # AI execution/review governance
│   ├── ENGINEERING_OWNERSHIP_MATRIX.md    # Owners, mandatory reviewers, escalation
│   ├── TRI_AGENT_HANDOFF_PROTOCOL.md      # Codex + CodeRabbit + lead handoff
│   ├── STALE_ARTIFACT_POLICY.md           # Stale file classification/removal policy
│   └── ANALISIS_FORENSE_EXTENSIONES.md    # Cognitive extensions record
│
├── shared/
│   └── genesis-types/             # CRATE-000: primitives and proof system
│
├── core/
│   ├── genesis-math/              # CRATE-001: G(1,3) sparse algebra
│   ├── genesis-topology/          # CRATE-002: HNSW, cohomology, manifold
│   └── genesis-dynamics/          # CRATE-003: Kuramoto, VFE, attractors
│
├── Cargo.toml                     # Workspace manifest
├── .coderabbit.yaml               # Enterprise AI review policy
├── PLANS.md                       # Mandatory execution plans for complex changes
├── AGENTS.md                      # Guidance for AI coding agents
└── contratos_genesis.md           # Full API contracts reference
```

---

## Building and testing

**Requirements:** Rust stable ≥ 1.75.0

```bash
# Build all implemented crates
cargo build --workspace

# Run all tests (384 tests, ~25s due to topology stochastic tests)
cargo test --workspace

# Type-check without building
cargo check --workspace

# Run benchmarks (genesis-types hash bench, genesis-dynamics dynamics bench)
cargo bench -p genesis-types
cargo bench -p genesis-dynamics
```

**Expected output:**
```
test result: ok. 119 passed; 0 failed   (genesis-types)
test result: ok. 129 passed; 0 failed   (genesis-math)
test result: ok. 49 passed; 0 failed    (genesis-topology)
test result: ok. 79 passed; 0 failed    (genesis-dynamics)
```

One test is marked `#[ignore]` in genesis-topology (stochastic Lanczos convergence
with high variance — run explicitly with `cargo test -- --ignored` when needed).

---

## Invariants

These constraints are enforced at runtime and must never be violated:

| Invariant | Enforcement |
|-----------|-------------|
| Minkowski signature (+,−,−,−) | `GenesisError::SignatureViolation` |
| H¹(M, F) = 0 | `IncrementalH1State` + `CohomologyValidator` |
| λ₂ ≥ 0.1 (algebraic connectivity) | `AxiomID::AlgebraicConnectivity` in Proof |
| No mutation without Proof | `AxiomID::ProofGuard` → `H_restricción = ∞` |
| ΔH_total < C_dim(N) for new dimension | `AxiomID::DimensionalAdmission` |
| Fisher updated after expansion | `AxiomID::DualityConsistency` |
| `COGNITIVE_PLANCK_CONSTANT = 1e-12` immutable | Compile-time constant |

---

## Absolute prohibitions

- **No `HashMap` or `BTreeMap` in hot paths** — cache-hostile, unpredictable latency
- **No Delaunay triangulation** — O(N²), prohibited at scale
- **No SHA-256 in the proof system** — BLAKE3 only (`blake3 = "=1.5.4"`)
- **No uniform metric weights** — grade-differentiated `METRIC_WEIGHTS` is mandatory
- **No cross-entropy, MSE, or external loss functions** — VFE is the only learning signal
- **No global clock** — time is a geometric dimension in G(1,3), not a loop parameter
- **No forced phase collapse** — decoherence must emerge from Lindblad noise

---

## Documentation

Full mathematical derivation and implementation contracts are in `docs/`.
`contratos_genesis.md` contains the complete API reference for all implemented crates.
`AGENTS.md` contains guidance for AI coding agents working in this repository.

---

## Foundational references

- **Variational Free Energy / Active Inference:** Friston et al. (2010–2022)
- **Geometric Algebra / Spacetime Algebra:** Hestenes, *Space-Time Algebra* (1966/2015)
- **Kuramoto model:** Kuramoto (1984); Strogatz (2000)
- **Ollivier-Ricci flow:** Ollivier (2009); Lin et al. (2011)
- **HNSW:** Malkov & Yashunin (2018)
- **Poincaré embeddings:** Nickel & Kiela (2017)
- **Johnson-Lindenstrauss lemma:** Johnson & Lindenstrauss (1984)

---

*GÉNESIS Cognitive Core — variational architecture, geometric algebra, production Rust.*
