# Performance Feature Flags

This document describes optional performance-oriented feature flags and their tradeoffs for production deployment planning.

## Safety Classification

- **Proof-safe:** feature does not alter numerical semantics in ways that violate proof assumptions.
- **Non-proof-safe:** feature may change floating-point behavior or representation in ways that are not bit-exact or proof-compatible.

## Feature Matrix

| Feature | Crate | Classification | Benefit | Tradeoff |
|---|---|---|---|---|
| `avx512` | `genesis-math` | Proof-safe (hardware-dependent) | Higher SIMD throughput on AVX-512-capable CPUs | Reduced portability; possible frequency downclock on some microarchitectures |
| `poly_trig` | `genesis-dynamics` | **Non-proof-safe** | Faster trigonometric evaluation in hot loops | Not bit-exact; approximation error; unsuitable for proof-critical paths |
| `hnsw-f16` | `genesis-topology` | **Non-proof-safe** | Lower memory footprint and improved cache residency | Precision loss vs `f32`/`f64`; potential recall/quality degradation |

## Detailed Notes

### `genesis-math/avx512`

Use when deploying on homogeneous AVX-512 hardware and when throughput dominates over strict cross-host portability.

- Expected effect: wider vector lanes for arithmetic kernels.
- Risk: some CPUs reduce turbo frequencies with heavy AVX-512 utilization.
- Recommendation: benchmark against AVX2 targets on your production SKU.

### `genesis-dynamics/poly_trig`

Enables polynomial trigonometric approximations for faster synchrony and oscillator-related math.

- Expected effect: reduced trig latency in hot paths.
- Risk: approximation drift; behavior is not bit-exact with standard library trig.
- **Do not use** for proof-generation or proof-verification workloads.

### `genesis-topology/hnsw-f16`

Uses half-precision storage paths in topology/HNSW contexts to improve memory efficiency.

- Expected effect: lower bandwidth and memory pressure, especially at scale.
- Risk: quantization error can reduce neighborhood fidelity and downstream quality.
- Recommendation: validate recall/quality metrics before production activation.

## Deployment Profiles

### 1) Proof-Critical Build (Conservative)

Use only proof-safe features.

```bash
cargo build --workspace --release --features "genesis-math/avx512"
```

If AVX-512 is not guaranteed, omit the feature:

```bash
cargo build --workspace --release
```

### 2) Throughput-Optimized, Non-Proof Build

Enable aggressive performance features when proof compatibility is not required.

```bash
cargo build --workspace --release \
  --features "genesis-math/avx512 genesis-dynamics/poly_trig genesis-topology/hnsw-f16"
```

### 3) Memory-Constrained Topology Deployment

Favor memory reduction while keeping other behavior conservative.

```bash
cargo build -p genesis-topology --release --features "hnsw-f16"
```

## Recommendation Checklist

Before enabling non-proof-safe flags in production:

1. Confirm the workload is not proof-critical.
2. Run regression and quality benchmarks against a proof-safe baseline.
3. Document the selected feature set in release notes for traceability.
