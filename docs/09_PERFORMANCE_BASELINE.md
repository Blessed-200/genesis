# 09 — Performance Baseline (Branch post-change snapshot)

This document records reproducible baseline measurements for this branch snapshot
after recent topology-path updates in `core/genesis-topology/src/hnsw.rs`.

## Scope

Tracked workloads:

- `hnsw_insert_1000`
- `hnsw_search_k10_in_1000`
- `rips_build_10k`
- `manifold_compute_lambda2`
- `synchrony_order_fast_1000_nodes`
- `cohomology_h1_check_10k` (cold-start)
- `rank_by_gaussian_elimination_throughput`

## Reference hardware and software

- Date (UTC): 2026-04-07
- OS/runtime: Linux on KVM virtualized host
- CPU: Intel(R) Xeon(R) Platinum 8370C @ 2.80GHz
- vCPU allocation: 3 CPUs, 1 thread/core
- Relevant CPU flags: `avx2`, `fma`, `avx512f`, `avx512dq`, `avx512bw`, `avx512vl`
- Rust profile: `cargo run --release` / `cargo bench` (`opt-level=3`)

Raw hardware capture command:

```bash
lscpu
```

## Methodology

### Sampling protocol

For baseline probe runs:

- Warm-up iterations per workload: `5`
- Sample count per workload: `30`
- Statistic family: empirical percentiles over wall-clock nanoseconds (`Instant::now()`)
- Reported quantiles: p50, p95, p99

### Statistical confidence approach

Percentiles are reported as non-parametric order statistics from independent repeated runs.
For regression tracking, confidence is assessed by comparing percentile envelopes between runs
(on identical hardware class and pinning) rather than assuming Gaussian latency.

### Cold-start requirement for cohomology

`CohomologyValidator::invalidate_cache()` is called before each measured iteration so that the
cache key/result are cleared prior to `check_h1`, forcing cold-start execution.

## Baseline results

| Workload | p50 | p95 | p99 | Samples | Notes |
|---|---:|---:|---:|---:|---|
| `hnsw_insert_1000` | 50,292,495 ns | 53,134,316 ns | 53,326,516 ns | 30 | Deterministic fixture vectors |
| `hnsw_search_k10_in_1000` | 12,034 ns | 12,086 ns | 12,253 ns | 30 | Query = fixture id 500 |
| `rips_build_10k` | 27,864,230 ns | 28,654,267 ns | 28,895,003 ns | 30 | `epsilon=0.5` |
| `manifold_compute_lambda2` | 171,401,807 ns | 175,001,206 ns | 175,251,002 ns | 30 | 5k-node manifold fixture |
| `synchrony_order_fast_1000_nodes` | 39,365 ns | 47,905 ns | 66,328 ns | 30 | 1000-node sparse ring (`k=8`) |
| `cohomology_h1_check_10k` (cold-start) | N/A | N/A | N/A | N/A | Timed out at 300 s before completing first sample in cold-start mode |
| `rank_by_gaussian_elimination_throughput` | 296,688 ns | 324,303 ns | 364,339 ns | 30 | Matrix params `(128,256)` |

## Reproduction commands

```bash
# Mandatory validation before collecting baseline
cargo check --workspace
cargo test --workspace
if cargo check --workspace 2>&1 | grep -q "^warning:"; then
  echo "Warnings found"
  exit 1
fi

# Criterion subset for topology (reference names)
cargo bench -p genesis-topology --bench topology -- \
  '(hnsw_insert_1000|hnsw_search_k10_in_1000|rips_build_10k|manifold_compute_lambda2_p50_p95_p99|rank_by_gaussian_elimination_throughput)'

# Cold-start cohomology guard run (in-repo, reproducible)
/usr/bin/timeout 300s cargo bench -p genesis-topology --bench topology -- \
  cohomology_h1_check_10k --sample-size 10 --warm-up-time 0.1 --measurement-time 0.2
```

## Interpretation

- The new cold-start benchmark path is now structurally enforced by cache invalidation hooks.
- On this VM profile, `cohomology_h1_check_10k` cold-start is currently beyond a practical
  single-sample window (`>300 s`). This baseline is intentionally recorded as an unresolved
  throughput bottleneck prior to optimization work.
