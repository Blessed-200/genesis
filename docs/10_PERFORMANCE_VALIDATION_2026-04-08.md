# 10 — Performance Validation Report (2026-04-08)

This report compares the current branch against the baseline documented in
`docs/09_PERFORMANCE_BASELINE.md`.

## Method parity

For metrics used in the delta table, the run protocol matches the baseline document:

- Criterion warm-up time: `5s`
- Criterion sample size: `30`
- Release profile (`cargo bench`)
- Same host class and same local runner for all measurements in this report

The cohomology cold-start probe intentionally uses the baseline guard command
(`sample-size=10`, `warm-up-time=0.1`, `measurement-time=0.2`) because it is a
timeout smoke check, not a percentile table source.

### Commands executed

```bash
cargo bench -p genesis-topology --bench topology -- \
  '(hnsw_insert_1000|hnsw_search_k10_in_1000|rips_build_10k|manifold_compute_lambda2_p50_p95_p99|rank_by_gaussian_elimination_throughput)' \
  --sample-size 30 --warm-up-time 5

/usr/bin/timeout 300s cargo bench -p genesis-topology --bench topology -- \
  cohomology_h1_check_10k --sample-size 10 --warm-up-time 0.1 --measurement-time 0.2

cargo bench -p genesis-dynamics --bench dynamics -- synchrony_order_fast_1000_nodes
```

## Comparison summary

Percent deltas are computed as `(current - baseline) / baseline * 100`.
Negative values are improvements (lower latency). Baseline and current columns
explicitly declare the estimator used for each value.

| Metric | Baseline (p50 from 09) | Current (Criterion estimate median) | Delta | Status |
|---|---:|---:|---:|---|
| `hnsw_insert_1000` | 50.292 ms (p50) | 66.273 ms (criterion median) | +31.77% | Regression |
| `hnsw_search_k10_in_1000` | 12.034 µs (p50) | 25.580 µs (criterion median) | +112.56% | Regression |
| `rips_build_10k` | 27.864 ms (p50) | 21.382 ms (criterion median) | -23.26% | Improvement |
| `manifold_compute_lambda2_p50_p95_p99` | 171.402 ms (p50) | 64.988 ms (criterion median) | -62.08% | Improvement |
| `rank_by_gaussian_elimination_throughput` | 296.688 µs (p50) | 317.800 µs (criterion median) | +7.12% | Regression |
| `cohomology_h1_check_10k` cold-start | timed out at 300s | timed out at 300s | N/A | Unresolved bottleneck |
| `synchrony_order_fast_1000_nodes` | 39.365 µs (p50) | N/A | N/A | Blocked by benchmark guardrail panic |

## Latency target check

- There is no separate numeric target table in `docs/09_PERFORMANCE_BASELINE.md`; baseline p50 values are treated here as the practical target envelope.
- **Not all targets are satisfied**:
  - `hnsw_insert_1000`, `hnsw_search_k10_in_1000`, and `rank_by_gaussian_elimination_throughput` regressed.
  - `cohomology_h1_check_10k` remains above the 300s timeout guard.
  - `synchrony_order_fast_1000_nodes` could not be measured due to an early panic in `bench/dynamics.rs` guardrail (`kuramoto_step_1000_nodes`).

## Investigation flags

1. `hnsw_search_k10_in_1000` regression (+112.56%) requires profiling of layer-0 scan and entry-point selection.
2. `hnsw_insert_1000` regression (+31.77%) suggests insertion-path overhead drift (candidate expansion and prune costs).
3. `rank_by_gaussian_elimination_throughput` regression (+7.12%) is mild but persistent; inspect row-pivot and word-scan loop scheduling.
4. `cohomology_h1_check_10k` cold-start remains unresolved and should stay on the optimization queue.
