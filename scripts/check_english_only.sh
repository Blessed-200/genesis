#!/usr/bin/env bash
set -euo pipefail

# Detect mixed-language Spanish technical terminology in Rust comments/rustdoc.
# Scope is intentionally limited to core/shared source trees.
SPANISH_TERMS='función|funciones|algoritmo|algoritmos|estructura|estructuras|método|métodos|parámetro|parámetros|retorna|retornar|propiedad|propiedades|valor|valores|tipo|tipos|cero|ejemplo|ejemplos|nota|notas|ver|como|para|cuando|donde|entre|sobre|bajo|dentro|fuera|dirigida|dirigido|antisimetría|reproducibilidad|térmica|pendiente|infinito|siempre|fresco|contiguo|reconstruye|comportamiento|fronteras'
COMMENT_PATTERN="(///|//!|//|/\\*|\\*)[^\\n]*\\b(${SPANISH_TERMS})\\b"

echo "Checking Rust comments/rustdoc for non-English technical wording..."

base_ref=""
if [[ -n "${GITHUB_BASE_REF:-}" ]] && git rev-parse --verify --quiet "origin/${GITHUB_BASE_REF}" >/dev/null; then
  base_ref="$(git merge-base "origin/${GITHUB_BASE_REF}" HEAD)"
elif git rev-parse --verify --quiet HEAD~1 >/dev/null; then
  base_ref="HEAD~1"
fi

files=()
# Local/dev mode: prefer currently modified files for fast pre-commit feedback.
if ! git diff --quiet -- 'core/**/*.rs' 'shared/**/*.rs'; then
  while IFS= read -r path; do
    [[ -n "${path}" && -f "${path}" ]] && files+=("${path}")
  done < <(git diff --name-only HEAD -- 'core/**/*.rs' 'shared/**/*.rs')
elif [[ -n "${base_ref}" ]]; then
  while IFS= read -r path; do
    [[ -n "${path}" && -f "${path}" ]] && files+=("${path}")
  done < <(git diff --name-only "${base_ref}...HEAD" -- 'core/**/*.rs' 'shared/**/*.rs')
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No changed core/shared Rust sources detected relative to baseline; skipping."
  exit 0
fi

matches="$(rg -n --pcre2 -i "${COMMENT_PATTERN}" "${files[@]}" || true)"

if [[ -n "${matches}" ]]; then
  echo
  echo "English-only documentation violations detected (file:line):"
  echo "${matches}"
  echo
  echo "Fix all listed comment/rustdoc lines before merging."
  exit 1
fi

echo "English-only documentation check passed."
