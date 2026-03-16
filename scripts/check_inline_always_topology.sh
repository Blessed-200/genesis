#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/core/genesis-topology/src"
ALLOWLIST_FILE="$ROOT_DIR/.ci/inline_always_allowlist_topology.txt"

if [[ ! -f "$ALLOWLIST_FILE" ]]; then
  echo "❌ Missing allowlist: $ALLOWLIST_FILE"
  exit 1
fi

mapfile -t hits < <(rg -n '^[[:space:]]*#\[inline\(always\)\]' "$TARGET_DIR" || true)
if [[ ${#hits[@]} -eq 0 ]]; then
  echo "✅ No #[inline(always)] found in core/genesis-topology/src"
  exit 0
fi

mapfile -t allowed_raw < "$ALLOWLIST_FILE"
declare -A allowed=()
for line in "${allowed_raw[@]}"; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  allowed["$line"]=1
done

violations=()
for hit in "${hits[@]}"; do
  location="$(echo "$hit" | cut -d: -f1-2)"
  rel="${location#${ROOT_DIR}/}"
  if [[ -z "${allowed[$rel]:-}" ]]; then
    violations+=("$rel")
  fi
done

if [[ ${#violations[@]} -gt 0 ]]; then
  echo "❌ Policy violation: #[inline(always)] is forbidden in core/genesis-topology/src unless allowlisted."
  echo
  echo "Found non-allowlisted occurrences:"
  printf ' - %s\n' "${violations[@]}"
  echo
  echo "How to fix:"
  echo "  1) Prefer removing #[inline(always)] or downgrade to #[inline]."
  echo "  2) If exception is mandatory, document rationale near code and update allowlist:"
  echo "     $ALLOWLIST_FILE"
  echo "  3) Use this exception template in PR/commit notes:"
  echo "     - Benchmark: <command + before/after ns/op>"
  echo "     - Architectural reason: <hot path + AX-ID/H-term>"
  echo "     - Risk: <portability/regression/maintainability + mitigation>"
  exit 1
fi

echo "✅ #[inline(always)] usage matches reviewed allowlist"
