# 09 — Performance Baseline (Measurement-First Infrastructure)

This document records the initial benchmark baseline after introducing:

- `iai-callgrind` hot-path harnesses in `genesis-topology` and `genesis-dynamics`
- expanded Criterion suites for topology and dynamics hot paths
- PGO orchestration scripts (`scripts/pgo_build.sh`, `scripts/bench_extreme.sh --pgo`)

Date captured: **2026-03-28** (UTC).

## Benchmark commands used

```bash
cargo bench -p genesis-topology --bench hnsw_hotpaths -- --noplot --sample-size 10
cargo bench -p genesis-dynamics --bench vfe_hotpaths -- --noplot --sample-size 10
```

IAI callgrind harnesses were validated as buildable via:

```bash
cargo bench -p genesis-topology --bench iai_hotpaths --no-run
cargo bench -p genesis-dynamics --bench iai_hotpaths --no-run
```

## Baseline measurements

### `genesis-topology` — `hnsw_hotpaths`

| Benchmark | Baseline time |
|---|---:|
| `enforce_density_limit/1000` | 4.88–4.95 ns |
| `enforce_density_limit/10000` | 5.10–5.28 ns |
| `enforce_density_limit/100000` | 5.05–5.23 ns |
| `incremental_d2_xor_columns/density_1pct/65536` | 793.64–799.62 µs |
| `incremental_d2_xor_columns/density_10pct/65536` | 3.49–3.57 ms |
| `incremental_d2_xor_columns/density_50pct/65536` | 17.86–18.01 ms |

### `genesis-dynamics` — `vfe_hotpaths`

| Benchmark | Baseline time |
|---|---:|
| `compute_vfe_with_grad/10000` | 84.01–84.82 µs |
| `compute_vfe_with_grad/100000` | 84.46–89.45 µs |
| `compute_vfe_with_grad/1000000` | 84.29–85.02 µs |

## Notes for regression tracking

- Use this file as the reference floor before algorithmic or layout optimizations.
- Re-run with identical command-line settings to keep comparisons valid.
- For PGO validation, use `scripts/pgo_build.sh` (or `scripts/bench_extreme.sh --pgo`) and store resulting tuned binary metrics separately from this non-PGO baseline.
