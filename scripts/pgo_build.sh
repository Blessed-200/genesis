#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PGO_DIR="${PGO_DIR:-$ROOT_DIR/target/pgo-data}"
RAW_DIR="$PGO_DIR/raw"
MERGED_PROFILE="$PGO_DIR/merged.profdata"

mkdir -p "$RAW_DIR"
rm -f "$RAW_DIR"/*.profraw "$MERGED_PROFILE"

if ! command -v llvm-profdata >/dev/null 2>&1; then
  echo "error: llvm-profdata is required for PGO merge" >&2
  exit 1
fi

echo "[1/3] instrumented build (release-pgo-gen)"
RUSTFLAGS="-Cprofile-generate=$RAW_DIR" \
  cargo build --workspace --profile release-pgo-gen

echo "[2/3] benchmark execution for profile capture"
RUSTFLAGS="-Cprofile-generate=$RAW_DIR" \
  cargo bench -p genesis-topology --bench hnsw_hotpaths -- --noplot
RUSTFLAGS="-Cprofile-generate=$RAW_DIR" \
  cargo bench -p genesis-topology --bench iai_hotpaths
RUSTFLAGS="-Cprofile-generate=$RAW_DIR" \
  cargo bench -p genesis-dynamics --bench vfe_hotpaths -- --noplot
RUSTFLAGS="-Cprofile-generate=$RAW_DIR" \
  cargo bench -p genesis-dynamics --bench iai_hotpaths

echo "[2.5/3] merge raw profiles"
llvm-profdata merge -output="$MERGED_PROFILE" "$RAW_DIR"/*.profraw

echo "[3/3] optimized rebuild (release-pgo-use)"
RUSTFLAGS="-Cprofile-use=$MERGED_PROFILE -Cllvm-args=-pgo-warn-missing-function" \
  cargo build --workspace --profile release-pgo-use

echo "PGO build completed: $MERGED_PROFILE"
