# AGENTS.md — GÉNESIS Cognitive Core

> This file is read by OpenAI Codex, Amp, and other AI coding agents before any work.
> Read it completely before touching any file in this repository.

---

## What this project is

GÉNESIS is a **variational cognitive architecture in Rust**. All cognition emerges from
minimising a Hamiltonian functional over a spacetime-algebraic state space G(1,3).
Every concept is a sparse multivector in Clifford algebra — not a token or embedding.

This is not a neural network, not a transformer, not a statistical language model.
The mathematical foundations are in `docs/00_AXIOMAS.md` and `docs/01_LEY_FUNDACIONAL.md`.
Read them before proposing architectural changes.

---

## Repository state

### Implemented and passing (384 tests, 0 failures)

| Crate | Path | Tests |
|-------|------|-------|
| `genesis-types` | `shared/genesis-types/` | 119 |
| `genesis-math` | `core/genesis-math/` | 129 |
| `genesis-topology` | `core/genesis-topology/` | 49 |
| `genesis-dynamics` | `core/genesis-dynamics/` | 79 |

### Pending (do not create these crates without explicit instruction)

- `genesis-evolution` (Phase 5) — Ricci flow, wormhole collapse, compression
- `genesis-consciousness` (Phase 6) — global observer Ω, duality
- `genesis-io` (Phase 7) — sensory projection and manifestation

---

## Verification commands — run these, in this order

```bash
# 1. Type-check all crates (fast, < 5s)
cargo check --workspace

# 2. Run full test suite (required before any PR)
cargo test --workspace

# 3. Check for warnings (treat as errors for new code)
cargo check --workspace 2>&1 | grep "^warning:"
```

**Expected output for step 2:**
```
test result: ok. N passed; 0 failed   (for every crate)
```

The topology suite takes ~20s (stochastic Lanczos). One test is `#[ignore]` — this is
intentional. Do not un-ignore it without confirming numerical stability.

---

## Coding standards

### Module documentation
Every module, struct, and public function must include an `AX-ID` reference:
```rust
/// Short description.
///
/// Longer explanation if needed.
///
/// AX-ID: AXIOMA-006, H_dinámica (LEY_FUNDACIONAL §3.2)
pub fn my_function(...) {}
```

Valid axiom references: `AXIOMA-001` through `AXIOMA-019`, plus Hamiltonian terms
`H_estructura`, `H_dinámica`, `H_información`, `H_teleología`, `H_restricción`,
`H_dualidad`, `H_compresión`.

### Test requirements
- Every new public function needs at least one test
- Tests go in `#[cfg(test)] mod tests` at the bottom of the same file
- Name tests descriptively: `fn amplitude_zero_gives_rsync_zero()`
- Use `assert!((a - b).abs() < 1e-12)` for floating-point, never `assert_eq!` on f64

### Error handling
Use `GenesisError` from `genesis-types`. No `unwrap()` in non-test code.
Return `Result<T, GenesisError>` for fallible operations.

---

## Absolute prohibitions

Violating any of these is an architectural error, not a style issue:

| Prohibition | Rationale |
|-------------|-----------|
| `HashMap` or `BTreeMap` in hot paths | Cache-hostile. Use sorted `Vec` + binary search. |
| Delaunay triangulation | O(N²), never acceptable at scale |
| `sha2` crate | BLAKE3 is the project hash. `sha2` is removed from workspace. |
| Uniform `METRIC_WEIGHTS = [1.0; 16]` | Grade-differentiated weights are mandatory |
| `cross-entropy`, `MSE`, external loss | VFE is the only learning signal (AXIOMA-003) |
| Global clock / time-step loop | Time is a G(1,3) dimension, not a loop parameter |
| Circular crate dependencies | CRATE-004 may import CRATE-002/003, not the reverse |
| Direct modification of `HnswGraph` from Ricci flow | Use the semantic overlay layer |

---

## Critical implementation details

### Hash algorithm: BLAKE3, not SHA-256
The proof system in `genesis-types::proof` uses **BLAKE3 exclusively**.
`blake3 = "=1.5.4"` is a direct dependency of `genesis-types`.
**Do not add `sha2` to any Cargo.toml.** If you see `sha2` anywhere, remove it.

```rust
// CORRECT — in genesis-types/src/proof.rs:
use blake3;
let hash = *blake3::hash(data).as_bytes();

// WRONG — never do this:
use sha2::{Sha256, Digest};
```

### Metric weights: grade-differentiated
`METRIC_WEIGHTS` in `genesis-math/src/multivector.rs` is not uniform:
```
Grade 0 (scalar):       2.0
Grade 1 (vectors):      1.5  ← semantic direction, primary
Grade 2 (bivectors):    1.0
Grade 3 (trivectors):   0.5
Grade 4 (pseudoscalar): 0.3
```
Do not change these to uniform `1.0`. Do not add a feature flag for uniform weights.

### Oscillator amplitudes: default 1.0
`QuantumOscillator.amplitudes: [f64; 5]` initialises to `[1.0; 5]`.
With `amplitude = 1.0`, behaviour is **identical to classical Kuramoto**.
The amplitude-weighted `r_sync` is backward compatible by construction.
After `VFEMinimizer::update()`, call `oscillator.update_amplitude_from_fisher(trace)`.

### Belief is 16D
`Belief.mean_full: [f64; 16]` covers all G(1,3) blades.
`VFEMinimizer::compute_vfe_with_grad()` returns `(f64, [f64; 16])` — not 4D.
For legacy callers needing 4D: use `compute_vfe_with_grad_grade1()`.
`add_node(id, [f64; 4])` is still the constructor — backward compatible.

### HyperbolicCoord is a contract, not yet active
`ManifoldCollector::hyperbolic_coord(id)` always returns `None` until
`DiscreteRicciFlow` (CRATE-004) calls `set_hyperbolic_coord()`.
Do not pre-populate coordinates or write synthetic values for tests.

### NodeId sentinel
`NodeId::MAX_VALID = u64::MAX - 1`. Value `u64::MAX` is the sentinel (None).
Always construct with `NodeId::try_new(n)` — panics if `n > MAX_VALID`.

### Proof generation
`WitnessBuilder` + `AxiomGuard` in `genesis-types::proof`.
Structural mutations need `AxiomID::STRUCTURAL_REQUIRED`.
Dimensional expansions need `AxiomID::EXPANSION_REQUIRED` (includes `DualityConsistency` + `DimensionalAdmission`).

---

## Workspace dependency rules

When adding a dependency:
1. Add to `[workspace.dependencies]` in root `Cargo.toml`
2. Reference with `dep = { workspace = true }` in the crate's `Cargo.toml`
3. Exception: `blake3` is a direct dependency of `genesis-types` only (by design, see the comment in `shared/genesis-types/Cargo.toml`)

Current workspace deps: `thiserror`, `bytemuck`, `proptest`, `num-complex`, `smallvec`, `memoffset`, `criterion`, `tempfile`.

---

## File organisation

```
docs/                     ← architecture docs (read before changing behaviour)
  00_AXIOMAS.md           ← physical axioms, source of truth
  01_LEY_FUNDACIONAL.md   ← Hamiltonian derivation
  02_ENGINEERING_BLUEPRINT_V2.md  ← implementation reference
  03_MACRO_ARCHITECTURE.md        ← crate structure
  04_IMPLEMENTATION_ROADMAP.md    ← phase plan
  05_GENESIS_PROOF_SPEC.md        ← proof system
  ANALISIS_FORENSE_EXTENSIONES.md ← cognitive extensions record
contratos_genesis.md      ← full API contracts (all implemented crates)
shared/genesis-types/src/
  constants.rs            ← all variational parameters (edit here, not in crate files)
  proof.rs                ← AxiomID, Proof, AxiomGuard, WitnessBuilder
  signal.rs               ← NodeId, SpikeEvent, DomainSignal
  error.rs                ← GenesisError
core/genesis-math/src/
  multivector.rs          ← SparseCliffordVector, METRIC_WEIGHTS, fast_metric_distance
core/genesis-topology/src/
  manifold.rs             ← ManifoldCollector, HyperbolicCoord
core/genesis-dynamics/src/
  oscillator.rs           ← QuantumOscillator (phases + amplitudes)
  free_energy.rs          ← VFEMinimizer, Belief (16D), FisherInfo
  synchrony.rs            ← synchrony_order_fast (amplitude-weighted)
```

---

## Pull request guidelines

- Title format: `[CRATE-00N] Short description`
- Body must include: which axioms are affected, which Hamiltonian terms change, test count before/after
- Run `cargo test --workspace` and paste the summary in the PR body
- Do not open a PR if any test fails
- Do not open a PR if `cargo check --workspace` produces new `warning:` lines in code you modified

---

## When in doubt

1. Read `docs/00_AXIOMAS.md` — physical axioms cannot be violated
2. Read `docs/01_LEY_FUNDACIONAL.md` — Hamiltonian derivation, all 7 terms
3. Read `docs/02_ENGINEERING_BLUEPRINT_V2.md` — implementation patterns
4. Read `contratos_genesis.md` — full API with examples
5. Ask before proposing changes to `METRIC_WEIGHTS`, `BLAKE3`, or any `AX-ID` annotation

---

## SoaBatch4 — estado actual
- Kernel batch_distance_4: CORRECTO (test pasa, 1e-10 tolerancia)  
- Speedup medido: 0.85x (más lento que scalar)
- Causa: transpose-on-the-fly en hot loop domina el costo
- Solución pendiente: pre-construir SoA en inserción (persistent layout)
- NO revertir — la infraestructura es correcta y necesaria para persistent SoA

# ExecPlans

-For complex fixes or cross-module refactors, first produce and update ./PLANS.md.
Do not implement until the plan maps root causes to concrete file-level actions and validation steps.

---

## Optimization hierarchy (MANDATORY)

All implementations must follow this strict priority order:

1. Algorithmic optimality (O-notation, asymptotics)
2. Memory layout (cache locality, contiguous storage, SoA vs AoS)
3. Branch elimination (branchless logic preferred)
4. Vectorization (SIMD, unrolling where beneficial)
5. Allocation minimization (stack > arena > heap)
6. Instruction-level efficiency (fused ops, intrinsics when justified)

A solution that is correct but suboptimal in any higher tier MUST be rejected.

Agents must actively search for:
- better asymptotic algorithms
- better data layouts
- opportunities for precomputation

---

## Hot path rules

Hot paths MUST be identified explicitly.

A function is considered hot if:
- it is inside any loop over N elements
- it is called per node / per edge / per timestep
- it participates in VFE minimization or synchrony

For hot paths:

- No heap allocation
- No trait object dispatch
- No recursion
- No HashMap / BTreeMap
- Prefer:
  - slices (`&[T]`)
  - fixed-size arrays
  - SmallVec (only if bounded)
  - manual inlining (`#[inline(always)]` when justified)

Agents must annotate hot paths in comments:
```rust
// HOT PATH: O(N), called per iteration of VFE minimization
---

## 3. 🧬 IDIOMATICITY ≠ PERFORMANCE (clave)

Ahora mismo no lo separas. Añade:

```md
---

## Idiomatic Rust vs Performance

Idiomatic Rust is NOT sufficient.

Agents must:

- Prefer idiomatic constructs ONLY if they do not degrade performance
- Replace iterator chains with loops if:
  - it removes bounds checks
  - it improves vectorization
  - it avoids temporaries

Example:

// REJECT (alloc + iterator overhead)
vec.iter().map(...).collect()

// PREFERRED (hot path)
for i in 0..n { ... }

---

## Numerical stability and precision

All floating-point code must consider:

- catastrophic cancellation
- accumulation error
- normalization stability

Prefer:

- Kahan summation when summing many values
- fused multiply-add (FMA) patterns where possible
- stable formulations over naive equations

Agents must justify any non-trivial numeric transformation.

---

## Algorithmic superiority requirement

Before modifying any function, agents must ask:

"Is there a fundamentally better algorithm for this?"

This includes:

- replacing O(N²) with O(N log N) or O(N)
- using spatial indexing instead of brute force
- using algebraic identities to eliminate operations
- exploiting structure of G(1,3) (sparsity, symmetry, XOR properties)

If a superior algorithm exists, it MUST be implemented,
even if it increases local code complexity.
---

## Data layout constraints

Data layout is part of the architecture.

Agents must evaluate:

- AoS vs SoA
- alignment (cache line awareness)
- contiguous memory guarantees

Rules:

- Prefer SoA for batch operations
- Precompute layouts at insertion time (not in hot loops)
- Avoid transpose-on-the-fly (see SoaBatch4 note)

All layout decisions must be justified in comments.

---

## Forbidden inefficiencies (expanded)

Agents must detect and eliminate:

- redundant recomputation inside loops
- unnecessary cloning or copying
- temporary allocations in hot paths
- dynamic dispatch where static is possible
- bounds checks inside tight loops (use unsafe ONLY if proven safe)

Code that contains these must be rewritten, not patched.

---

## Performance validation

Any non-trivial change must include:

- reasoning about complexity (before/after)
- expected cache behavior
- allocation count impact

For critical paths:

- include micro-benchmark (criterion) if applicable

Claims like "faster" without justification are invalid.

---

## Inlining policy

Use:

- #[inline] for small functions
- #[inline(always)] ONLY for hot-path primitives

Do not over-inline large functions (code bloat risk).

---

## Unsafe usage

Unsafe is allowed ONLY if:

- it removes bounds checks or branching in hot paths
- it is proven memory-safe
- it is documented with invariants

Every unsafe block must include:

// SAFETY: explanation of why this is valid



*GÉNESIS Cognitive Core | AGENTS.md v4.0.0 | 384 tests, 0 failures*

## Documentation Language Integrity (CRITICAL RULE)

All documentation (Rustdoc, inline comments, module docs) MUST be written in strict, professional, native-level technical English.

### Absolute requirements

- NEVER mix languages (no Spanish, no hybrid "Spanglish")
- NEVER perform literal translation
- ALWAYS rewrite sentences fully when normalizing language
- ALWAYS preserve domain precision (mathematics, physics, inference systems)

### Forbidden patterns

The following are strictly prohibited:

- Mixed-language constructs ("por blade", "of un node", etc.)
- Broken grammar from translation
- Word-by-word translation artifacts
- Informal or conversational tone

### Required standard

Documentation must read as if written by:
- a senior systems engineer OR
- a mathematical physics researcher

Target qualities:
- precise
- unambiguous
- consistent terminology
- publication-grade clarity

### Semantic preservation rule

- Documentation changes MUST NOT alter meaning
- Mathematical intent MUST remain identical
- AX-ID anchors MUST remain unchanged

### Consistency enforcement

When editing any file:
- Normalize terminology globally (not locally)
- Ensure consistency across the entire module
- Prefer rewriting over patching

### Self-validation (mandatory before commit)

The agent MUST verify:

1. No non-English words remain
2. No hybrid grammar exists
3. Terminology is consistent across the file
4. Comments are understandable in isolation

If any condition fails → fix before completing task

### Priority

This rule has HIGHER priority than stylistic preferences.
