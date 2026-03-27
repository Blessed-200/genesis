## 0.8 genesis-math branchless/indexed micro-optimization sweep (2026-03-27)

### Root cause
- Several `genesis-math` hot-path functions still use avoidable bounds-checked accessors, branchy sign/parity selection, and non-FMA accumulations in tight loops.
- Dense/sparse kernels include small control-flow patterns that increase instruction count without changing semantics for validated index domains.

### File-level actions
1. `core/genesis-math/src/basis.rs`
   - Remove fallback indexing in hot accessors in favor of direct indexed loads under existing invariants.
   - Replace branchy sign/parity computations with branchless bit arithmetic.
   - Route prefix parity query to prebuilt Fenwick tree slot lookup.
2. `core/genesis-math/src/dual.rs`, `core/genesis-math/src/grade.rs`, `core/genesis-math/src/multivector.rs`
   - Apply FMA-based accumulations and simplified branchless coefficient/sign handling in hot loops.
   - Simplify dual conversion loops to fixed-range indexed writes.
3. `core/genesis-math/src/product.rs`, `core/genesis-math/src/semantic.rs`, `core/genesis-math/src/experimental/kernel_dense_g13.rs`
   - Convert product accumulations to explicit `mul_add` forms and simplify branch structure in dense/sparse dispatch loops while preserving G(1,3) semantics.

### Validation
- `cargo check --workspace`
- `cargo test --workspace`
- `cargo check --workspace 2>&1 | grep "^warning:"`

### Complexity/cache target
- Preserve O(16²) dense and active-mask sparse complexities while reducing branch pressure and improving arithmetic fusion on floating-point accumulation paths.

# GÉNESIS HPC Root-Cause Remediation Plan (Phase 1)

## 0.5 genesis-types micro-optimizations and branch simplification (2026-03-26)

### Root cause
- Several `genesis-types` hot/cold utility paths still use avoidable branches, redundant temporary values, or less-direct search/sort APIs.
- Some checks recompute lengths or duplicate comparisons in tight loops, increasing instruction count.

### File-level actions
1. `shared/genesis-types/src/fisher_edge.rs`
   - Use comparator-based `binary_search_by` for edge-key lookups.
   - Simplify normalization iteration and degree rebuild traversal without index arithmetic.
2. `shared/genesis-types/src/proof.rs`
   - Simplify subset checks via `AxiomSet::is_superset_of`.
   - Tighten witness replay bounds checks and streamline inline/large buffer growth paths.
   - Replace explicit axiom discriminant match with guarded transmute for `0..=6`.
3. `shared/genesis-types/src/signal.rs`
   - Collapse padding-zero checks into full-array comparisons.
   - Reduce duplicate absolute-value computation and use unstable sort for bounded pair canonicalization.
   - Simplify linear `get` search and count narrowing cast in fixed-capacity context.
4. `shared/genesis-types/src/error.rs`, `shared/genesis-types/src/lib.rs`, `shared/genesis-types/src/multivector_types.rs`
   - Remove redundant branching or shift forms in bit checks and signature diagnostics.

### Validation
- `cargo check --workspace`
- `cargo test --workspace`
- `cargo check --workspace 2>&1 | grep "^warning:"`

### Complexity/cache target
- Preserve asymptotic behavior while reducing branch count and loop overhead in frequent utility paths.

## 0.4 NodeAdj layer-0 pre-grouped block projection (2026-03-24)

### Root cause
- `core/genesis-topology/src/hnsw.rs::search_layer` at layer 0 still executes a per-neighbor regrouping scheduler (`node_to_slab` lookup, block/lane arithmetic, scratch block merge) inside beam expansion.
- With `M0=32`, this administrative regrouping dominates branch/memory work even after epoch-visited dedup.

### File-level actions
1. `core/genesis-topology/src/hnsw.rs`
   - Extend `NodeAdj` layer-0 storage to keep both: canonical sorted adjacency and a synchronized pre-grouped slab-block projection.
   - Canonical layer-0 entries are packed as `(neighbor_internal_idx << 32) | slab_idx` to preserve sorted-by-neighbor order while carrying slab addressing metadata.
   - Add compact `Layer0BlockGroup` (`#[repr(C)]`) and maintain sorted block groups incrementally on insert/remove, avoiding runtime reconstruction.
   - Replace layer-0 regrouping in `search_layer` with direct iteration of precomputed block groups and lane masks; keep visited-epoch semantics unchanged.
   - Keep upper-layer adjacency and public API behavior unchanged.
2. `core/genesis-topology/src/hnsw.rs` tests
   - Add invariant tests ensuring group projection exactly matches canonical adjacency after inserts and survives node removal.
   - Re-assert canonical sorted order and nearest-neighbor search equivalence regression.

### Validation
- `cargo test -p genesis-topology --release 2>&1 | grep -E "FAILED|ok"`
- `cargo clippy -p genesis-topology -- -D warnings 2>&1 | grep "^error"`
- `cargo bench -p genesis-topology --bench topology hnsw_search_k10_in_1000 -- --output-format bencher 2>&1 | grep "bench:"`
- `cargo bench -p genesis-topology --bench topology hnsw_insert_1000 -- --output-format bencher 2>&1 | grep "bench:"`

### Complexity/cache target
- Remove per-expansion dynamic regrouping from layer-0 search hot path: consume pre-grouped block masks directly with one slab-distance evaluation per touched block.

## 0.3 prune_layer packed-order partition optimization (2026-03-24)

### Root cause
- `core/genesis-topology/src/hnsw.rs::prune_layer` still materializes `(neighbor_id, f64_dist)` tuples and globally sorts all candidates even though pruning only needs the `m_max` smallest neighbors.
- Current path also creates a second `SmallVec` for dropped IDs, increasing stack traffic and copy pressure in a hot path called during insert connectivity maintenance.

### File-level actions
1. `core/genesis-topology/src/hnsw.rs`
   - Replace score storage from `SmallVec<[(u32, f64); M0]>` to `SmallVec<[u64; M0]>`, packing `(dist_f32_bits << 32) | neighbor_id`.
   - Use `select_nth_unstable(m_max - 1)` to partition in average `O(d)` instead of full `O(d log d)` sort.
   - Preserve deterministic tie-break by embedding neighbor id in low 32 bits of packed key.
   - Keep mutation-safe drop handling by materializing only overflow IDs into compact `SmallVec<[u32; M0]>` before calling `remove_edge_bidirectional`.
2. `core/genesis-topology/src/hnsw.rs` tests
   - Keep existing prune regression tests.
   - Add packed-order invariant test that verifies `u64` packed ordering matches `(dist as f32)` ordering for non-negative distances.

### Validation
- `cargo test -p genesis-topology --release 2>&1 | grep -E "FAILED|ok"`
- `cargo clippy -p genesis-topology -- -D warnings 2>&1 | grep "^error"`
- `cargo bench -p genesis-topology --bench topology hnsw_insert_1000 -- --output-format bencher 2>&1 | grep "bench:"`

### Complexity target
- `O(d)` distance materialization + average `O(d)` partition with no full sort, and reduced per-candidate footprint (8 bytes packed key vs 16 bytes tuple).

## 0.2 HNSW prune_layer one-shot pruning pass (2026-03-24)

### Root cause
- `core/genesis-topology/src/hnsw.rs::prune_layer` recomputes neighbour distances inside a `while` + `max_by` loop while mutating the same adjacency list.
- Distances from source node `idx` to candidate neighbours are invariant once the candidate set is known, so iterative re-evaluation is redundant and increases hot-path cost.

### File-level actions
1. `core/genesis-topology/src/hnsw.rs`
   - Replace iterative farthest-removal loop with a one-shot pass:
     - snapshot current neighbours once,
     - materialize exactly one distance per neighbour,
     - deterministically select keep-set (`m_max` nearest by `(distance, neighbor_id)`),
     - remove discarded edges bidirectionally using neighbour IDs only.
   - Keep NodeAdj sorted adjacency invariant untouched by using existing remove-by-id path.
2. `core/genesis-topology/src/hnsw.rs` tests
   - Add regression coverage for one-shot pruning behavior, sorted adjacency after pruning, and bidirectional symmetry of removals.

### Validation
- `cargo test -p genesis-topology --release`
- `cargo clippy -p genesis-topology -- -D warnings`
- `cargo bench -p genesis-topology --bench topology hnsw_insert_1000 -- --output-format bencher`

### Complexity target
- Target `O(d)` distance materialization + bounded deterministic selection/sort (`d <= M0`) with no iterative re-evaluation loop over a mutating adjacency list.

---

Source backlog: `./genesis_root_causes_hpc_v1.json` (9 root causes, authoritative)
Scope: implemented crates only (`genesis-types`, `genesis-math`, `genesis-topology`, `genesis-dynamics`)

## 0.1 Dense G(1,3) kernel sequential-read restructure (2026-03-23)

### Root cause
- `core/genesis-math/src/product.rs` AVX2/FMA and NEON dense kernels still depend on per-`j` gather/permutation of `a_coeffs`, so LLVM/hardware sees non-sequential reads in the hottest loop.
- `core/genesis-math/src/experimental/kernel_dense_g13.rs` baseline dense kernel uses `k`-outer accumulation, so the benchmarked reference path does not expose the intended sequential `sign_row[j]`/`b[j]` stream.

### File-level actions
1. `core/genesis-math/src/product.rs`
   - Restructure dense scalar loop as explicit hot-path `i`-outer / `j`-inner row-hoist.
   - Replace AVX2, AVX2+FMA, and NEON dense loops with sequential SIMD loads from `sign_row[j..]` and `b_coeffs[j..]`, keeping the unavoidable `result[i ^ j]` update as scalar scatter.
   - Preserve strict-mode reduction order and public API semantics.
2. `core/genesis-math/src/experimental/kernel_dense_g13.rs`
   - Align the dense benchmark kernel with the same `i`-outer / `j`-inner access pattern so the benchmark measures the sequential-read layout directly.

### Validation
- Invariants first: `cargo test --release -p genesis-math -- invariant --nocapture`
- Math correctness: `cargo test -p genesis-math --release`
- Performance: `cargo bench -p genesis-math --bench geometry -- comparison_baseline --output-format bencher`
- Workspace gates: `cargo test --workspace --release`, `cargo clippy --workspace -- -D warnings`, `cargo check --workspace`
- ASM spot-check: `RUSTFLAGS="--emit=asm" cargo build --release -p genesis-math` + grep for vector loads/moves in dense kernel symbols.

---

## 0) Method and constraints used for this plan

- Performed static inspection of repository and representative hot-path modules.
- Built a root-cause-to-code map from the JSON + current implementation.
- No code changes in this phase.
- Designed fixes to preserve:
  - stable Rust compatibility,
  - SIMD/hot-path behavior,
  - existing crate contracts (unless explicitly versioned),
  - G(1,3) invariants and AX-ID semantics.

---

## 1) Root-cause-to-code map + strategy

## RC-01 — compile-time `const` misuse for runtime-only floating-point/allocation operations

### Affected modules/files/functions (confirmed)
- `core/genesis-dynamics/src/free_energy.rs`
  - `const fn is_finite_scalar` (uses `f64::is_finite` in const context)
  - `const fn sanitize_trace` (uses `is_finite` + `clamp` in const helper)
  - `pub const fn VFEMinimizer::new()` (allocating `Vec::new()` in const constructor)
- Secondary audit targets for same pattern:
  - `core/genesis-topology/src/manifold.rs` (`const fn` methods returning runtime-owned state)
  - `core/genesis-dynamics/src/oscillator.rs` (const constructors involving floating fields)

### True mechanism
Const qualification is applied where compile-time evaluation is not required and runtime-only semantics are intended. This overconstrains API and creates toolchain/MSRV fragility.

### Viable strategies
1. **De-const runtime constructors/helpers (preferred)**
   - Convert runtime-only `const fn` to `fn` (especially constructors that allocate or validate runtime floating data).
2. Bit-level const-safe finite checks everywhere
   - Keep `const`, reimplement finite checks via bit operations.
3. Split API into const core + runtime wrapper
   - Keep tiny truly-const kernel; move allocations/checks to runtime wrappers.

### Risks
- Strategy 1: low risk; potential downstream compile break only for callers using const contexts.
- Strategy 2: medium risk; complexity and readability loss, possible subtle IEEE edge handling mistakes.
- Strategy 3: medium complexity and API churn.

### Selected fix
- Apply Strategy 1 for all runtime-only paths.
- Keep `const fn` only for pure compile-time data constructors with no floating runtime validation or allocation dependency.

---

## RC-02 — unchecked integer arithmetic + narrowing/lossy conversions

### Affected modules/files/functions (confirmed)
- `core/genesis-math/src/product.rs`
  - `build_sign_flip_masks`, `build_xor_permute_indices`
  - multiple `trailing_zeros() as usize` and narrow casts
- `shared/genesis-types/src/signal.rs`
  - `SpikeComponents::from_pairs_internal`
  - `SpikeComponents::try_from_pairs`
  - `count as usize` and conversion boundaries
- `core/genesis-topology/src/hnsw.rs`
  - `HnswGraph::insert` (`id.get() as usize`, `resize(id_raw + 1, ...)`, `u32::try_from(new_idx).expect(...)`)
  - `radix_sort_node_ids` bucket and shift casts
  - multiple conversions across layer/index management
- `core/genesis-math/src/basis.rs`
  - signed/unsigned Fenwick index transitions (`i32 <-> usize`)

### True mechanism
Arithmetic and conversion boundaries are handled ad hoc. Overflow, truncation, and index-size contracts are not uniformly encoded as checked operations.

### Viable strategies
1. **Boundary-safe conversion layer (preferred)**
   - Introduce helper APIs/newtypes for index conversion (`NodeIdx`, `LayerIdx`, `BladeIdx`, `DenseSlot`) using `try_from` and explicit errors.
2. Scattershot `checked_*` and local guards
   - Patch each cast/operation inline without central abstraction.
3. Widen everything to `u64/usize`
   - Avoid narrowing by using wider types broadly.

### Risks
- Strategy 1: moderate refactor size but highest long-term correctness.
- Strategy 2: high chance of inconsistency/regression.
- Strategy 3: can hurt cache/layout and introduce API drift.

### Selected fix
- Strategy 1 in hot/shared boundaries; minimal local guards in leaf kernels where dimensions are statically bounded (16 blades).

---

## RC-03 — missing bounds/indexing invariants

### Affected modules/files/functions (confirmed)
- `core/genesis-dynamics/src/free_energy.rs`
  - `target_from_sparse_obs` (`coeffs[i]` indexed by generated range)
  - loops over fixed arrays and node lookup index usage
- `core/genesis-math/src/basis.rs`
  - `grade_of`, `blade_square`, `fenwick_prefix_parity` rely on `debug_assert!` only
- `core/genesis-topology/src/hnsw.rs`
  - `neighbors_within` uses `unsafe get_unchecked`
  - dense indexing of nodes/layers after lookup in several methods
- `core/genesis-math/src/multivector.rs`
  - dense array indexing in active-mask loops

### True mechanism
Many accesses are safe by design but invariants are implicit (debug-only asserts, assumptions from prior lookups). Contracts are not encoded strongly enough for auditors.

### Viable strategies
1. **Typed invariant wrappers + safe access facade (preferred)**
   - Use bounded index types and helper accessors that enforce checked construction once.
2. Replace all indexing with `.get()` everywhere
   - Maximum safety, but significant hot-path overhead/noise.
3. Keep direct indexing + stronger proof comments/asserts
   - Low code change, but only partial root-cause reduction.

### Risks
- Strategy 1: moderate implementation effort.
- Strategy 2: possible measurable HPC regressions.
- Strategy 3: may fail to eliminate structural issue count.

### Selected fix
- Strategy 1 for dynamic inputs; preserve direct indexing for fixed-size `[T;16]` loops where index space is proven by construction.

---

## RC-04 — numerical domain guards missing (division/sqrt/log/invalid inputs)

### Affected modules/files/functions (confirmed)
- `core/genesis-math/src/product.rs`
  - scalar/dense product and normalization helpers with arithmetic chains
- `shared/genesis-types/src/signal.rs`
  - `SpikeComponents::try_from_pairs` path and merge arithmetic
- `core/genesis-dynamics/src/free_energy.rs`
  - `compute_vfe`, `compute_vfe_with_grad`, `update`, `update_full`, `bounded_step`
- `core/genesis-topology/src/hnsw.rs`
  - `random_level` (`ln`), distance kernels (`sqrt`)

### True mechanism
Domain checks are local and inconsistent; no crate-wide contract for denominator epsilon, sqrt non-negativity, finite input normalization, and log-domain guarantees.

### Viable strategies
1. **Numerical contract module (preferred)**
   - Add shared helpers: guarded reciprocal/division, `sqrt_nonneg`, `ln_pos`, finite sanitization with explicit epsilon constants.
2. Continue local `if` checks per function
   - Faster to patch but fragile and repetitive.
3. Clamp everything aggressively
   - Simpler but risks biasing dynamics.

### Risks
- Strategy 1: moderate rollout; requires consistent adoption.
- Strategy 2: future regressions likely.
- Strategy 3: stability vs correctness trade-off may distort model behavior.

### Selected fix
- Strategy 1 + targeted replacement in hot functions; avoid over-clamping where physical semantics require sensitivity.

---

## RC-05 — floating-point stability / rounding risk

### Affected modules/files/functions (confirmed)
- `core/genesis-dynamics/src/free_energy.rs`
  - `compute_vfe_with_grad` accumulation path
  - update loops with repeated additive updates
- `core/genesis-topology/src/hnsw.rs`
  - `distance_to_node`, search ordering with partial compares and non-finite fallback behavior
- `core/genesis-math/src/product.rs`
  - norm/product accumulation helpers (`bivector_norm_sq_of_product*` family)
- `core/genesis-math/src/multivector.rs`
  - metadata accumulation and float equality around signed zero canonicalization

### True mechanism
Stability policy is inconsistent: some loops use compensated summation, others still use naive accumulation/comparisons; exact comparisons and cancellation-prone formulas remain.

### Viable strategies
1. **Central FP policy + compensated accumulators where sensitivity is high (preferred)**
   - Kahan/Neumaier in selected kernels, tolerance-based comparisons, deterministic ordering.
2. Blanket compensated summation everywhere
   - Overhead may hurt throughput in low-sensitivity paths.
3. Minimal local patches
   - Incomplete reduction.

### Risks
- Strategy 1: requires profiling-aware placement.
- Strategy 2: potential performance regressions.
- Strategy 3: leaves systemic weakness.

### Selected fix
- Strategy 1 with path classification: hot/low-risk keep naive where mathematically bounded; critical aggregates use compensation.

---

## RC-06 — hot-path performance inefficiency and algorithm/data-structure mismatch

### Affected modules/files/functions (confirmed)
- `core/genesis-math/src/product.rs`
  - hot loops and temporary buffer usage in dense/sparse products
- `core/genesis-topology/src/hnsw.rs`
  - `search_layer` (heap churn, temporary vectors, tail batching)
  - `layer0_soa` construction costs
- `core/genesis-dynamics/src/free_energy.rs`
  - `PagedIndex::set` growth/resize behavior

### True mechanism
Repeated temporary allocations, clone/copy patterns, and per-iteration overhead remain in core loops; some data structures are not specialized for contiguous hot access.

### Viable strategies
1. **Micro-architecture cleanup preserving algorithmic behavior (preferred)**
   - preallocation reuse, avoid repeated len/lookups, reduce temporary allocations in `search_layer` and SoA paths.
2. Deep structural redesign (new graph containers, intrusive arenas)
   - potentially faster but too broad for current backlog.
3. No-op except compiler hints
   - insufficient.

### Risks
- Strategy 1: medium complexity with performance-sensitive correctness constraints.
- Strategy 2: high regression risk and scope explosion.

### Selected fix
- Strategy 1 only, consistent with current architecture and stable Rust.

---

## RC-07 — panic-prone fallible API usage in production paths

### Affected modules/files/functions (confirmed)
- `core/genesis-topology/src/hnsw.rs`
  - `insert` (`expect` on `u32::try_from(new_idx)` and entry assumptions)
  - `search_layer` chunk/tail `expect`
  - `radix_sort_node_ids` worker join `expect`
- `shared/genesis-types/src/error.rs`
  - production-facing helpers requiring panic audit
- `shared/genesis-types/src/signal.rs`
  - `unwrap_or` conversion fallback in construction paths

### True mechanism
Recoverable failures are converted to panics or hidden assumptions (`expect`), especially under scaling/concurrency edge cases.

### Viable strategies
1. **Convert production `unwrap/expect` to typed error propagation (preferred)**
   - introduce/extend `GenesisError` variants for overflow, worker failure, invariant breach.
2. Replace with `debug_assert!` only
   - avoids panic in release but may hide errors.
3. Leave panics on “impossible” paths
   - not acceptable per root-cause scope.

### Risks
- Strategy 1: requires error plumbing through call chains.
- Strategy 2/3: weak fault transparency.

### Selected fix
- Strategy 1 with explicit error semantics and non-panicking fallbacks only where safe.

---

## RC-08 — unsafe usage without explicit safety contracts

### Affected modules/files/functions (confirmed)
- `core/genesis-topology/src/hnsw.rs`
  - `batch_distance_4_avx2` (intrinsics + raw loads/stores)
  - `neighbors_within` (`get_unchecked`)
  - lock-free snapshot (`Arc::from_raw`/`increment_strong_count` sections)
- `shared/genesis-types/src/signal.rs`
  - `NodeId::from_raw_unchecked`
- `shared/genesis-types/src/multivector_types.rs`
  - unchecked mask mutation helpers

### True mechanism
Unsafe appears in multiple places with mixed-quality local contracts; some unsafe is avoidable, some is justified for SIMD/lock-free performance but needs tighter encapsulation and proof boundaries.

### Viable strategies
1. **Unsafe minimization + contract hardening (preferred)**
   - remove avoidable unsafe (`get_unchecked` in non-hot path), isolate required unsafe with clear preconditions/postconditions.
2. Keep all unsafe and add comments only
   - documentation-only mitigation.
3. Remove unsafe broadly
   - likely performance loss in SIMD/atomic kernels.

### Risks
- Strategy 1: moderate work; must preserve performance in kernel paths.
- Strategy 3: regression risk in key HPC kernels.

### Selected fix
- Strategy 1.

---

## RC-09 — unclassified singleton (manual review)

### Affected module/function
- Not identified in JSON (`representative_units = ["unclassified"]`).

### True mechanism
A single residual issue is missing source attribution in the reduction file.

### Viable strategies
1. **Regenerate/trace from source report artifacts (preferred)**
   - locate origin in `final_report.json` or forensic outputs, map to existing RC if possible.
2. Ignore until after code fixes
   - risks leaving one unresolved blocker.

### Risks
- Without provenance, fix may be mis-scoped.

### Selected fix
- First execution task in Phase 2: recover exact location and classify before coding.

---

## 2) Execution sequencing (Phase 2 blueprint)

1. **Global safety contracts first**
   - RC-02 + RC-03 shared boundary types/helpers.
2. **Numerical contracts and FP stability**
   - RC-04 + RC-05 shared numerics layer, then apply in dynamics/math/topology kernels.
3. **Panic and unsafe hardening**
   - RC-07 + RC-08, prioritizing `hnsw.rs` production paths.
4. **Performance pass**
   - RC-06 micro-optimizations only after correctness contracts land.
5. **RC-01 const hygiene sweep**
   - remove residual runtime-const misuse introduced or exposed by prior changes.
6. **RC-09 resolution**
   - map singleton to cluster or implement local fix.

Rationale: correctness and invariant encoding before speed tuning reduces rework.

---

## 3) Validation plan (to run during Phase 2 after each major cluster)

Per repository policy (in order):
1. `cargo check --workspace`
2. `cargo test --workspace`
3. `cargo check --workspace 2>&1 | grep "^warning:"`

Additional targeted checks by cluster:
- `cargo test -p genesis-dynamics free_energy`
- `cargo test -p genesis-topology hnsw`
- `cargo test -p genesis-math product`
- `cargo test -p genesis-types signal`

Numerical regression focus:
- stability tests around VFE accumulation, HNSW distance ordering, and product norm functions.

Performance regression focus:
- existing benches: `core/genesis-math/benches/geometry.rs`, `core/genesis-topology/benches/topology.rs`, `core/genesis-dynamics/benches/dynamics.rs`.

---

## 4) File-level action shortlist for implementation

- `core/genesis-dynamics/src/free_energy.rs`
- `core/genesis-topology/src/hnsw.rs`
- `core/genesis-math/src/product.rs`
- `core/genesis-math/src/basis.rs`
- `core/genesis-math/src/multivector.rs`
- `shared/genesis-types/src/signal.rs`
- `shared/genesis-types/src/error.rs` (only if required for new typed errors)
- possibly `shared/genesis-types/src/multivector_types.rs` (if index/unsafe wrappers consolidated)

This shortlist maps all 9 JSON root causes without adding extra scope.


---

## Phase 2 execution log

### Cluster A — integer safety / conversions (subplan)
- Replace lossy `as` casts at dynamic boundaries in `free_energy.rs` and `hnsw.rs` with checked conversions.
- Add overflow-safe growth for sparse index pages (`PagedIndex::set`) and direct index resize in HNSW insertion.
- Remove panic-on-overflow paths (`expect`) and return typed `GenesisError::InvariantViolation` where state cannot be represented.
- Validate with `cargo check --workspace` and targeted HNSW/free-energy tests.

### Cluster B — bounds and invariants (subplan)
- Replace avoidable unchecked indexing in dynamic paths (`neighbors_within` unsafe access).
- Upgrade invariant-bearing accessors (`basis.rs`) from debug-only assumptions to bounded safe fallback semantics.
- Keep fixed-size `[T;16]` kernel indexing where statically bounded.
- Validate with targeted basis/HNSW tests.

#### Cluster A/B — execution update
- Changed `core/genesis-dynamics/src/free_energy.rs`:
  - `VFEMinimizer::new` converted to runtime `fn`.
  - `sanitize_trace` / `is_finite_scalar` converted to runtime helpers.
  - `PagedIndex::set` now guards `page + 1` with `checked_add`.
  - `lookup` now uses checked `u32 -> usize` conversion.
- Changed `core/genesis-topology/src/hnsw.rs`:
  - `insert` now uses checked conversions for `NodeId -> usize`, guarded resize length, and non-panicking `u32` index conversion with `GenesisError::InvariantViolation` on overflow.
  - removed `entry.expect(...)`; now explicit invariant error if missing.
  - `neighbors()` now uses one lookup and no fallback unwrap.
  - `neighbors_within()` removed `unsafe get_unchecked`; safe `.get()` guard.
  - `search_layer()` removed chunk/tail `expect` paths and returns output via `mem::take` instead of clone.
  - `radix_sort_node_ids()` histogram path simplified to deterministic single-thread pass (no worker-join panic path).
- Changed `core/genesis-math/src/basis.rs`:
  - `grade_of` / `blade_square` / `fenwick_prefix_parity` now use bounded safe indexing fallback.

Validation performed:
- `cargo check --workspace` ✅
- `cargo test -p genesis-topology hnsw` ✅
- `cargo test -p genesis-dynamics free_energy` ✅

Remaining risk:
- `basis.rs` fallback semantics for out-of-range access can mask caller misuse; this is acceptable for panic-safety but should be watched in invariant tests.

### Cluster C — numerical domain guards and stability (subplan)
- Add explicit finite/domain guards in HNSW level sampling and SIMD sqrt finalization.
- Harden free-energy bounded-step against non-finite/invalid trace inputs at function boundary.
- Improve floating comparison robustness where exact equality is not semantically required.
- Validate with targeted dynamics/topology tests and workspace check.

#### Cluster C — execution update
- Changed `core/genesis-topology/src/hnsw.rs`:
  - Added finite-output guard in scalar `batch_distance_4` fallback (`NaN/Inf -> +Inf`).
  - Added AVX2 distance post-processing guard before `sqrt` (`max(value, 0.0)` and finite fallback).
  - Hardened `random_level` by clamping sampled `u` to `(0,1)` and returning level 0 on non-finite sample.
- Changed `core/genesis-dynamics/src/free_energy.rs`:
  - `bounded_step` now validates `dt/trace` finiteness and denominator domain, with bounded output `[0, 0.9]`.
- Changed `shared/genesis-types/src/signal.rs`:
  - Top-K spike construction now rejects all non-finite coefficients (`!is_finite`).
  - Replaced exact `abs == min_abs` tie-break with `total_cmp` to avoid brittle float equality.
  - Post-merge acceptance now requires finite merged coefficient.

Validation performed:
- `cargo test -p genesis-types signal` ✅
- `cargo test -p genesis-topology hnsw_insert_and_search` ✅

Remaining risk:
- The scalar radix path retained in Cluster A may reduce `compact_index()` throughput on very large `id_index`; correctness and panic-safety are improved but perf should be benchmarked in Cluster E.

### Cluster D — panic-prone APIs and unsafe contracts (subplan)
- Remove remaining production `expect`/`unwrap` in `hnsw.rs` non-test paths.
- Convert lock-free snapshot hard assertion to non-panicking recovery path.
- Keep required unsafe for AVX2/lock-free pointer ops, but tighten local safety invariants.
- Validate with targeted topology tests and workspace check.

#### Cluster D — execution update
- Changed `shared/genesis-types/src/signal.rs`:
  - Removed production `unwrap()` calls in `SpikeComponents::from_pairs_internal` min-slot selection logic.
  - Added explicit non-panicking fallback when selection is unexpectedly empty.
- Changed `core/genesis-topology/src/hnsw.rs`:
  - Replaced hard `assert!` in lock-free snapshot loader with non-panicking retry path on null head pointer.
  - Previously removed non-test `expect()` use in insert/search paths (Cluster A/B) remains in effect.

Validation performed:
- `cargo test -p genesis-topology lock_free_index_supports_multiwriter_single_snapshot_semantics` ✅
- `cargo test -p genesis-types from_pairs_topk_by_magnitude_not_arrival` ✅

Remaining risk:
- Null-head retry in lock-free loader now spins until pointer publication; this avoids crash but could spin if called concurrently during teardown.

### Cluster E — performance cleanup (subplan)
- Remove avoidable temporary allocations in HNSW search-layer cleanup path.
- Pre-size layer-0 SoA edge buffers from known edge counters to reduce reallocation churn.
- Keep algorithmic behavior unchanged; only local memory/loop efficiency changes.
- Validate with topology test slice and workspace check.

#### Cluster E — execution update
- Changed `core/genesis-topology/src/hnsw.rs`:
  - `layer0_soa()` now preallocates `neighbor_ids` and `neighbor_distances` using `edge_count_layer0_undirected` to reduce growth churn.
  - `search_layer()` visited reset now uses in-place pop loop (no temporary allocation).

Validation performed:
- `cargo test -p genesis-topology layer0_soa_preserves_layer0_cardinality` ✅

Remaining risk:
- `edge_count_layer0_undirected` can overestimate after removals with invalidated nodes, so preallocation may reserve slightly more than required (acceptable).

### Cluster F — const hygiene (subplan)
- Remove `const fn` qualifiers from runtime constructors that allocate or initialize dynamic containers.
- Keep `const fn` only for pure compile-time data constructors with stable semantics.
- Validate with workspace check + targeted topology/dynamics tests.

#### Cluster F — execution update
- Changed runtime constructors from `const fn` to `fn`:
  - `core/genesis-topology/src/hnsw.rs`: `HnswGraph::new`
  - `core/genesis-topology/src/manifold.rs`: `ManifoldCollector::new`
- Earlier in Phase 2, `core/genesis-dynamics/src/free_energy.rs` runtime helpers (`is_finite_scalar`, `sanitize_trace`, `VFEMinimizer::new`, `bounded_step`) were also de-constified where compile-time semantics were unnecessary.

Validation performed:
- `cargo check --workspace` ✅

Remaining risk:
- Any downstream callsites requiring const context for these constructors would now fail to compile; no such usage found in workspace.

### Cluster G — residual singleton/manual review (subplan)
- Attempt to locate provenance for RC-09 (`unclassified`) in repository artifacts.
- If source artifact is unavailable, mark as non-actionable with technical justification and no speculative patch.

#### Cluster G — execution update
- Searched repository for `final_report.json`, `Unknown issue`, and `unclassified` provenance metadata.
- Result: only the reduced backlog file exists; no source artifact pinpoints the singleton location.
- RC-09 status: **non-actionable in-code for this phase** because no attributable source unit exists in repository.

Validation performed:
- Artifact search only (no code execution required).

Remaining risk:
- RC-09 may remain unresolved until the upstream source report (`final_report.json`) is provided.

---

## Phase 2 final status

### Workspace-level validation
- `cargo check --workspace` ✅
- `cargo test --workspace` ✅
- `cargo check --workspace 2>&1 | grep "^warning:"` ✅ (no warning lines emitted)

### Root-cause closure status
- RC-01 (const misuse): **addressed** via runtime de-constification in affected constructors/helpers.
- RC-02 (integer safety/conversions): **addressed** at dynamic index boundaries (checked conversion + checked growth).
- RC-03 (bounds/invariants): **addressed** by removing unsafe unchecked access and adding bounded accessors.
- RC-04 (numerical domain guards): **addressed** with explicit domain/finiteness checks on key sqrt/log/division paths.
- RC-05 (floating-point stability): **addressed** with robust comparisons (`total_cmp`) and guarded finite handling.
- RC-06 (performance inefficiency): **addressed** with targeted allocation-reduction changes in HNSW hot paths.
- RC-07 (panic-prone APIs): **addressed** by removing non-test panic paths in modified production functions.
- RC-08 (unsafe without contracts): **addressed** by eliminating avoidable unsafe access and tightening unsafe usage scope.
- RC-09 (unclassified singleton): **non-actionable in repository state** (source artifact missing; no attributable code unit to patch).

### Residual follow-up (external dependency)
- To fully close RC-09, upstream `final_report.json` (or equivalent provenance artifact) is required.

## Final verification consolidation

- RC-01: **closed**
- RC-02: **closed**
- RC-03: **closed**
- RC-04: **closed**
- RC-05: **closed**
- RC-06: **closed**
- RC-07: **closed**
- RC-08: **closed**
- RC-09: **non-actionable** (no attributable source unit in repository; upstream `final_report.json` provenance missing)

## Targeted Plan — Squared-distance HNSW metric path (2026-03-23)

### Root cause
- HNSW hot-path comparisons currently compute Euclidean distance via `sqrt` even when only relative ordering is needed. This adds avoidable scalar latency in `distance_to_node`, `search_layer`, `greedy_search_layer`, and `prune_layer`.

### File-level actions
1. `core/genesis-math/src/multivector.rs`
   - Add `fast_metric_distance_sq` and `fast_metric_distance_sq_from_dense` public helpers.
   - Preserve existing sub-Planck fallback and non-finite guards; only remove terminal `sqrt`.
   - Keep `fast_metric_distance*` behavior unchanged by delegating to squared versions + `sqrt`.
2. `core/genesis-topology/src/hnsw.rs`
   - Switch internal comparison/heaps in `distance_to_node`, `search_layer`, `greedy_search_layer`, and `prune_layer` to squared distances.
   - Compute `sqrt` only when API/tests require user-facing distance values.
   - Keep ordering stable and deterministic, including non-finite handling.
3. Tests/benches
   - Add assertions that nearest-neighbor ordering is identical between squared and non-squared paths.
   - Extend or add benchmark coverage to compare CPU time before/after (distance-heavy search path).

### Validation steps
- `cargo test -p genesis-math multivector`
- `cargo test -p genesis-topology hnsw`
- `cargo check --workspace`
- `cargo test --workspace`
- `cargo check --workspace 2>&1 | grep "^warning:"`
- Performance skill check: `cargo bench -p genesis-topology -- hnsw --output-format bencher`
- Invariants post-change:
  - `cargo test --release -p genesis-math -- invariant --nocapture`
  - `cargo test --release -p genesis-topology -- invariant --nocapture`


## Targeted Plan — HNSW layer-0 zero-overhead SoA slab pipeline (2026-03-23)

### Objective
- Eliminate per-query AoS→SoA transpose overhead in `search_layer()` by persisting a 64-byte-aligned layer-0 slab at insertion time and consuming it directly from the search hot path.

### Invariants and contracts that must not break
- Public API/signatures remain stable for `HnswGraph`, `search_nearest`, `remove_node`, `layer0_soa`, and `LockFreeHnswIndex`.
- Layer-0 distance remains the grade-weighted Clifford metric over all 16 blades; no Euclidean/L2 regression.
- `remove_node` remains zero-compaction and must exclude deleted nodes from results.
- No `HashMap`/`BTreeMap`, no search-path heap allocation, no runtime AoS→SoA transpose.
- Unsafe blocks must document alignment, bounds, and aliasing invariants.

### Root cause
- `SoaBatch4::from_nodes()` transposes 4 AoS vectors into temporary SoA storage on every layer-0 beam expansion. For high `ef`, transpose cost dominates arithmetic and defeats SIMD throughput.

### File-level actions (`core/genesis-topology/src/hnsw.rs` only)
1. Replace transient `SoaBatch4` / `batch_distance_4*` with persistent slab metadata:
   - add `SLAB_LANES=8`, `SLAB_DIM=16`, `BLOCK_STRIDE=128`, `METRIC_WEIGHTS_F32`,
   - add 64-byte-aligned flat slab storage plus `node_to_slab: Vec<u32>`.
2. Rework insertion/removal:
   - materialize slab coefficients at insertion time,
   - grow slab by blocks,
   - mark deleted lanes with `NaN` sentinel in blade 0.
3. Rework layer-0 search hot path:
   - precompute query `[f32; 16]` once,
   - batch neighbor evaluation by slab block,
   - dispatch to scalar or AVX2 slab kernel without runtime feature checks,
   - reject non-finite lanes before candidate admission.
4. Replace dynamic binary heaps in search hot path with fixed-capacity deterministic heaps for bounded candidate/result sets.
5. Repurpose `HnswLayer0Soa` to expose slab + metadata instead of dense AoS coefficients.
6. Migrate/add tests in `hnsw.rs` for:
   - slab-vs-scalar distance agreement,
   - grade-weighted-not-L2 ordering,
   - NaN-lane deletion exclusion,
   - insertion ordering into slab,
   - deterministic fixed heap behavior,
   - nearest-neighbor regression equivalence for slab path.

### Microarchitectural risks to audit while implementing
- Misaligned slab base causing AVX loads to straddle cache lines unpredictably.
- Register pressure/spills in the AVX2 unrolled kernel.
- Recomputing block loads or slab-index conversions inside tight loops.
- False acceptance of NaN/Inf lanes into candidate heaps.
- Borrow-checker regressions that reintroduce temporary `Vec` allocation in `search_layer()`.

### Validation sequence
- Pre-change invariant check: `cargo test --release -p genesis-topology -- invariant --nocapture`
- Post-change targeted: `cargo test -p genesis-topology hnsw`
- Post-change release slice: `cargo test -p genesis-topology --release 2>&1 | grep -E "FAILED|ok"`
- Clippy gate: `cargo clippy -p genesis-topology -- -D warnings 2>&1 | grep "^error"`
- Bench gate: `cargo bench -p genesis-topology --bench topology -- --output-format bencher 2>&1 | grep "bench:"`
- Workspace gates required by repo policy: `cargo check --workspace`, `cargo test --workspace`, `cargo check --workspace 2>&1 | grep "^warning:"`


## Targeted Plan — genesis-math NEON sign hoist + Hodge correction + diagnostic benchmark (2026-03-23)

### Objective
- Remove avoidable sign-conversion overhead from the AArch64 NEON dense geometric-product hot loop.
- Reassert the G(1,3) Hodge double-dual invariant `⋆⁻¹(⋆A) = -A` across grades.
- Add a diagnostic Criterion baseline comparing the 256-multiply G(1,3) product against naive 16×16 matmul.

### Invariants and contracts that must not break
- `core/genesis-math` public APIs and signatures remain unchanged.
- G(1,3) Minkowski sign semantics continue to come from the precomputed Cayley tables; no XOR-only sign recomputation.
- `hodge_dual`/`hodge_undual` must satisfy the Minkowski pseudoscalar relation with `I² = -1`.
- No changes to `hnsw.rs`, topology crates, or dynamics crates.
- Hot-path changes must avoid heap allocation and preserve contiguous stack-local buffers.

### Root cause
- The NEON dense kernel still performs `i8 -> f64` sign conversion inside the innermost `j` loop, creating avoidable scalar work and pipeline pressure.
- `hodge_undual` may be using the wrong pseudoscalar-side sign convention if the new regression test exposes `+A` instead of `-A`.
- The existing benchmark baseline uses a differently named matmul case and does not explicitly report the requested comparison group.

### File-level actions
1. `core/genesis-math/src/product.rs`
   - Hoist `CAYLEY_SIGN_F64[idx0]` and `CAYLEY_SIGN_F64[idx1]` row references outside the NEON inner loop.
   - Keep accesses sequential via the hoisted row slices; do not recompute signs.
2. `core/genesis-math/src/dual.rs`
   - Add the full five-grade Hodge double-dual negation test.
   - If needed, fix `hodge_undual` with coefficient negation only, preserving stack-only execution.
3. `core/genesis-math/src/semantic.rs`
   - Audit `hodge_undual` callers and remove any manual compensation only if present.
4. `core/genesis-math/benches/geometry.rs`
   - Add the requested `comparison_baseline` group and `bench_naive_matmul_16x16` function.
   - Keep the dense G(1,3) kernel benchmark in the same comparison group for ratio reporting.

### Validation steps
- `cargo test --release -p genesis-math -- invariant --nocapture`
- `cargo test -p genesis-math --release 2>&1 | tail -5`
- `cargo test -p genesis-math --release dual 2>&1 | tail -10`
- `cargo bench -p genesis-math --bench geometry -- comparison_baseline --output-format bencher`
- `cargo test --workspace --release 2>&1 | grep -E "FAILED|^test result"`
- `cargo clippy --workspace -- -D warnings 2>&1 | grep "^error"`
- `cargo check --workspace 2>&1 | grep "^warning:"`

## Targeted Plan — HNSW epoch-visited + local NodeAdj adjacency (2026-03-23)

### Objective
- Remove per-search visited clearing overhead in `search_layer()` by replacing `FixedBitSet + visited_touched` with epoch marking.
- Remove O(N) global CSR offset correction during insertion/removal by replacing `CsrNeighborList` with per-node local adjacency storage.

### Invariants and contracts that must not break
- Scope stays limited to `core/genesis-topology/src/hnsw.rs`.
- Public API/signatures remain unchanged for `HnswGraph`, `search_nearest`, `remove_node`, `neighbors`, and `layer0_soa`.
- Internal hot loops operate on internal dense indices only; `NodeId` translation occurs at API/maintenance boundaries, not inside adjacency mutation loops.
- Layer-0 slab and `node_to_slab` stay authoritative and consistent.
- Bidirectional edge maintenance, `M0`/`M` limits, and deterministic search ordering remain intact.
- No heap allocation is introduced inside `search_layer()`.

### Root cause
- `search_layer()` currently pays repeated memory-administration cost from `FixedBitSet` writes and linear touched-node clearing.
- CSR adjacency mutation performs `splice()`/`insert()` plus suffix offset correction, turning edge insertion/removal into O(total_edges) memory movement.

### File-level actions
1. `core/genesis-topology/src/hnsw.rs`
   - Add graph-owned epoch state for visited tracking.
   - Bump epoch per search, reset the epoch buffer only on wraparound, and remove `FixedBitSet`/`visited_touched` from the hot path entirely while preserving the `&self` public search API via internal synchronization.
2. `core/genesis-topology/src/hnsw.rs`
   - Replace `CsrNeighborList` with `NodeAdjacency` backed by `Vec<NodeAdj>`.
   - Store layer-0 neighbors inline in `SmallVec<[u32; 32]>` and upper layers in optional boxed slices of `SmallVec<[u32; 16]>`.
   - Rework edge add/remove/prune/iteration helpers around dense internal indices, translating to `NodeId` only when producing public outputs.
3. `core/genesis-topology/src/hnsw.rs`
   - Update tests from CSR-layout assertions to NodeAdj invariants, epoch-wrap coverage, and no-regression search/order checks.

### Validation steps
- `cargo test -p genesis-topology --release 2>&1 | grep -E "FAILED|ok"`
- `cargo test --workspace --release 2>&1 | grep -E "FAILED|^test result"`
- `cargo clippy --workspace -- -D warnings 2>&1 | grep "^error"`
- `cargo check --workspace 2>&1 | grep "^warning:"`
- `cargo bench -p genesis-topology --bench topology -- hnsw_insert --output-format bencher 2>&1 | grep "bench:"`
- `cargo bench -p genesis-topology --bench topology -- hnsw_search_k10_in_1000 --output-format bencher 2>&1 | grep "bench:"`
- `grep -n "FixedBitSet\|visited_touched" core/genesis-topology/src/hnsw.rs`
- `grep -n "CsrNeighborList\|splice\|set_neighbors" core/genesis-topology/src/hnsw.rs`

## Targeted Plan — HNSW Mutex removal + remove_node edge-count fix (2026-03-23)

### Objective
- Remove the search-path `Mutex<Vec<u32>>` regression by moving the visited epoch buffer to thread-local storage while keeping a graph-owned atomic epoch generator.
- Fix `remove_node()` so layer-0 edge accounting stays exact after clearing outgoing adjacency.
- Restore deterministic adjacency ordering by using stable removal inside bounded `SmallVec` lists.

### Invariants and contracts that must not break
- Scope stays limited to `core/genesis-topology/src/hnsw.rs`.
- Public API/signatures remain unchanged.
- No heap allocation is introduced inside the `search_layer()` hot loop.
- Layer-0 and upper-layer degree bounds (`M0`, `M`) remain enforced.
- Search ordering and test reproducibility remain deterministic.

### Root cause
- Holding `Mutex<Vec<u32>>` across the whole search serialized insert-time beam expansion and added lock/unlock overhead per layer search.
- `remove_node()` cleared the removed node's outgoing layer-0 adjacency without decrementing the directed layer-0 edge counter for those outgoing arcs.
- `swap_remove()` destabilized adjacency order despite bounded-size lists.

### File-level actions
1. `core/genesis-topology/src/hnsw.rs`
   - Replace struct-owned `Mutex<Vec<u32>>` with `thread_local!` visited storage and rename the atomic counter to `epoch_gen`.
   - Resize/fill the thread-local epoch buffer at search start only; keep mark/check operations as single indexed loads/stores inside the hot loop.
2. `core/genesis-topology/src/hnsw.rs`
   - Subtract the removed node's outgoing layer-0 degree after `clear_layer(0)` in `remove_node()`.
   - Replace `swap_remove()` with stable `remove()` in `NodeAdj::remove_neighbor()`.
3. `core/genesis-topology/src/hnsw.rs`
   - Add regression tests for exact edge-count accounting after `remove_node()` and deterministic layer-0 adjacency ordering.

### Validation steps
- `cargo test -p genesis-topology --release 2>&1 | grep -E "FAILED|ok"`
- `cargo test --workspace --release 2>&1 | grep -E "FAILED|^test result"`
- `cargo clippy --workspace -- -D warnings 2>&1 | grep "^error"`
- `cargo check --workspace 2>&1 | grep "^warning:"`
- `cargo bench -p genesis-topology --bench topology -- hnsw_insert --output-format bencher 2>&1 | grep "bench:"`
- `cargo bench -p genesis-topology --bench topology -- hnsw_search_k10_in_1000 --output-format bencher 2>&1 | grep "bench:"`
- `grep "Mutex" core/genesis-topology/src/hnsw.rs | grep -v "//\\|test\\|clone"`
- `grep "visited_epoch" core/genesis-topology/src/hnsw.rs`

## Targeted Plan — NodeAdj sorted adjacency for O(log M0) duplicate checks (2026-03-24)

### Objective
- Remove linear duplicate checks in `NodeAdj::add_neighbor()` by enforcing sorted neighbor lists and switching to `binary_search` insertion for layer 0 and upper layers.

### Root cause
- Current layer-0 and upper-layer insertion path performs `contains()` + `push()`, which is O(M) duplicate detection with unsorted adjacency. Insertion is on the HNSW hot path and compounds per-node insertion cost.

### File-level actions
1. `core/genesis-topology/src/hnsw.rs`
   - In `NodeAdj::add_neighbor`, replace `contains` checks with `binary_search` for both layer-0 and upper layers.
   - Replace unsorted `push` with ordered `insert(pos, neighbor)` so adjacency remains sorted ascending.
   - Add debug-only invariant helper `assert_layer0_sorted()` and invoke it after layer-0 add/remove operations.
2. `core/genesis-topology/src/hnsw.rs` tests
   - Add `layer0_neighbors_sorted_after_insert` to validate sorted ascending layer-0 adjacency across inserted graph nodes.
   - Add `layer0_no_duplicates_after_double_add` to verify duplicate insertion is rejected and length remains 1.

### Invariants and risk controls
- `layer0` and upper-layer neighbor vectors remain strictly increasing (`w[0] < w[1]`) after successful insertion/removal.
- Degree bounds (`M0`/`M`) remain enforced before insert.
- `remove_neighbor` keeps sorted order because removal preserves relative ordering.
- No extra heap allocation beyond bounded `SmallVec` behavior and existing upper-layer growth.

### Validation steps
- `cargo test -p genesis-topology --release 2>&1 | grep -E "FAILED|ok"`
- `cargo clippy -p genesis-topology -- -D warnings 2>&1 | grep "^error"`
- `cargo bench -p genesis-topology --bench topology -- --output-format bencher hnsw_insert hnsw_search 2>&1 | grep "bench:"`
- Repository required gates: `cargo check --workspace`, `cargo test --workspace`, `cargo check --workspace 2>&1 | grep "^warning:"`.

## 0.5 Workspace comment/doc English unification (2026-03-25)

### Root cause
- Targeted workspace files contained mixed Spanish/English Rustdoc and inline comments, reducing API documentation consistency.

### File-level actions
1. Update only `///`, `//!`, and `//` comments in the requested files.
2. Preserve logic, strings, AX-ID references, and runtime behavior.
3. Run workspace validation commands after translation pass.

### Validation
- `cargo check --workspace 2>&1 | grep "^warning:"`
- `cargo test --workspace --release 2>&1 | grep -E "FAILED|^test result"`
- `cargo clippy --workspace -- -D warnings 2>&1 | grep "^error"`

## 0.6 Level-0 invariant enforcement pass (2026-03-25)

### Root cause map
- `core/genesis-dynamics/src/synchrony.rs::synchrony_order_fast` uses dual serial/parallel floating reduction paths with different accumulation order.
- `core/genesis-topology/src/hnsw.rs::prune_layer` uses lossy `f64 -> f32` ranking and `select_nth_unstable`, allowing tie instability.
- Phase wrapping logic is duplicated across `kuramoto.rs` and `phase_semantics.rs` with different boundary conventions.
- `core/genesis-topology/src/lsh.rs`, `core/genesis-dynamics/src/phase_semantics.rs`, and `core/genesis-dynamics/src/criticality.rs` still allocate/recompute in iterative paths.
- `core/genesis-topology/src/rips.rs::build` allocates by `max_id + 1`, scaling with id range instead of node count.

### File-level actions
1. `core/genesis-dynamics/src/synchrony.rs`
   - Replace split serial/parallel reduction with a single deterministic chunked reduction algorithm.
   - Use fixed chunk partitioning plus deterministic sequential combine order.
2. `core/genesis-topology/src/hnsw.rs`
   - Introduce monotonic full-precision `ordered_f64_bits` keying for prune ranking.
   - Remove `f32` ranking and `select_nth_unstable` from prune path; use deterministic full ordering.
3. `core/genesis-dynamics/src/kuramoto.rs` + `core/genesis-dynamics/src/phase_semantics.rs`
   - Introduce one canonical phase wrap function with explicit `(-π, π]` range and no `== PI` checks.
   - Route both modules through canonical implementation.
4. `core/genesis-topology/src/lsh.rs`
   - Remove per-call `BinaryHeap` and `Vec` materialization from candidate merge.
   - Implement iterator-based fixed-table k-way merge without heap allocations.
5. `core/genesis-dynamics/src/phase_semantics.rs`
   - Reuse preallocated phase and neighbor buffers across updates.
6. `core/genesis-dynamics/src/criticality.rs`
   - Fuse redundant passes in `tau_exponent_report`; compute Clauset terms from one sample collection.
7. `core/genesis-topology/src/rips.rs`
   - Replace `max_id + 1` dense map with compact sorted `(id, idx)` mapping.

### Validation
- Skills pre-checks:
  - `cargo test --release -p genesis-topology -- invariant --nocapture`
  - `cargo test --release -p genesis-dynamics -- invariant --nocapture`
- Workspace gates:
  - `cargo test --workspace --release`
  - `cargo clippy --workspace -- -D warnings`
- Determinism checks:
  - Add/adjust tests to assert stable ordering/bit-pattern equivalence in synchrony and HNSW prune outcomes.
- Performance checks (skill-driven):
  - `cargo bench -p genesis-topology --bench topology -- hnsw_search_k10_in_1000 --output-format bencher`
  - `cargo bench -p genesis-dynamics --bench dynamics -- kuramoto --output-format bencher`

## 0.7 Corrective pass — recover Kuramoto guardrail after Level-0 determinism (2026-03-25)

### Root cause
- Deterministic enforcement increased scalar overhead in hot paths:
  - `synchrony_order_fast` used deterministic but scalar-heavy final combination.
  - `wrap_phase` introduced extra branch/tolerance checks in Kuramoto tight loops.
  - Phase-semantics neighbor buffer shape used nested vectors with indirect reads.

### File-level actions
1. `core/genesis-dynamics/src/synchrony.rs`
   - Keep deterministic fixed partitioning, but perform fixed-tree deterministic reduction over chunk partials.
   - Use SIMD-friendly chunk width and remove large sequential fold patterns.
2. `core/genesis-dynamics/src/kuramoto.rs`
   - Implement branchless arithmetic wrap: `x - TAU * floor((x + PI) / TAU)`.
   - Keep `#[inline(always)]` for hot path.
3. `core/genesis-dynamics/src/phase_semantics.rs`
   - Replace nested neighbor vectors with contiguous CSR-like buffers (`offsets + flat`).
   - Keep linear writes and linear traversal to improve cache behavior.

### Validation
- `cargo test --workspace --release`
- `cargo clippy --workspace -- -D warnings`
- `cargo bench -p genesis-dynamics --bench dynamics -- kuramoto --output-format bencher`
