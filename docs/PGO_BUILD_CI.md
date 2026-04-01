# PGO Build Integration for Production Releases

This document defines a reproducible Profile-Guided Optimization (PGO) workflow for tagged production releases in the GÉNESIS workspace.

## Goals

- Increase real-world throughput by training the optimizer with representative execution traces.
- Keep release builds deterministic and auditable.
- Isolate PGO artifacts from regular developer workflows.

## Profiles Used

The root workspace defines two dedicated profiles:

- `release-pgo-gen`: builds instrumented binaries for profile data generation.
- `release-pgo-use`: builds optimized binaries using merged profile data.

Both profiles inherit from `release` to preserve consistent optimization defaults.

## End-to-End Workflow

1. **Build instrumented binaries** using `release-pgo-gen`.
2. **Run representative training workloads** to generate one or more `.profraw` files.
3. **Merge raw profiles** into a single `.profdata` file.
4. **Rebuild with profile use** via `release-pgo-use` and pass the merged profile.
5. **Run validation** (`cargo test --workspace`) before artifact publication.

## Reference Commands

> The commands below are examples. Keep workload inputs representative of production behavior.

```bash
# 1) Instrumented build
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
  cargo build --workspace --profile release-pgo-gen

# 2) Execute representative workloads (examples)
# Replace with real binaries and realistic datasets.
./target/release-pgo-gen/your_binary --scenario production_like_a
./target/release-pgo-gen/your_binary --scenario production_like_b

# 3) Merge profile artifacts
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw

# 4) Optimized build using trained profile
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
  cargo build --workspace --profile release-pgo-use

# 5) Safety validation
cargo test --workspace
```

## CI Integration Outline (Tagged Releases)

Use a release-only job (for example, tags matching `v*`) that:

1. Installs the pinned Rust toolchain.
2. Builds with `release-pgo-gen`.
3. Runs a deterministic training workload suite.
4. Merges `.profraw` outputs into `merged.profdata`.
5. Rebuilds with `release-pgo-use`.
6. Executes workspace tests.
7. Publishes artifacts if all checks pass.

### Minimal GitHub Actions Sketch

```yaml
name: release-pgo

on:
  push:
    tags:
      - 'v*'

jobs:
  pgo-release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Build instrumented
        run: |
          RUSTFLAGS="-Cprofile-generate=${{ runner.temp }}/pgo" \
            cargo build --workspace --profile release-pgo-gen

      - name: Run training workload
        run: |
          # Replace with reproducible production-like invocations.
          ./target/release-pgo-gen/your_binary --scenario ci_training

      - name: Merge profile data
        run: |
          llvm-profdata merge \
            -o ${{ runner.temp }}/pgo/merged.profdata \
            ${{ runner.temp }}/pgo/*.profraw

      - name: Build optimized
        run: |
          RUSTFLAGS="-Cprofile-use=${{ runner.temp }}/pgo/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
            cargo build --workspace --profile release-pgo-use

      - name: Validate
        run: cargo test --workspace
```

## Operational Notes

- Keep training inputs versioned and deterministic where possible.
- Regenerate PGO data when workload characteristics materially change.
- If profile coverage is poor, LLVM may warn about missing functions; retain warnings in CI logs.
