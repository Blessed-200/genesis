#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/3] criterion bench"
cargo bench -p genesis-math --bench geometry -- --noplot

echo "[2/3] perf counters (if available)"
if command -v perf >/dev/null 2>&1; then
  perf stat -e cycles,instructions,branches,branch-misses,cache-misses \
    cargo bench -p genesis-math --bench geometry -- --profile-time=1 >/tmp/genesis_perf.txt 2>&1 || true
  echo "perf output: /tmp/genesis_perf.txt"
else
  echo "perf not found; skipping"
fi

echo "[3/3] done"
