# GÉNESIS Skill Adaptation Matrix

This document adapts externally installed skills to the GÉNESIS repository operating model.

## Scope and precedence

1. `AGENTS.md` at repository root is authoritative for this codebase.
2. Skill instructions are applied only when they do not conflict with repository rules.
3. For `genesis-math`, `genesis-topology`, and `genesis-dynamics`, physical invariants and Hamiltonian contracts take precedence over generic optimization advice.

## Compatibility note with OpenAI Codex AGENTS guidance

Reference reviewed: `https://github.com/openai/codex/blob/main/AGENTS.md`.

Adoption policy in this repo:
- Keep generic quality conventions where compatible (clear APIs, exhaustive matching, narrow module scope).
- Do **not** import project-specific codex-rs conventions that are unrelated to GÉNESIS workspace structure.
- Preserve GÉNESIS-specific mandatory checks and AX-ID annotation policy.

## Installed skills and GÉNESIS-specific adaptations

### 1) `github-actions-docs`
- Apply only to `.github/workflows/*` and CI documentation updates.
- Must preserve current CI gates: invariants, workspace tests, warning scan.
- Any new workflow step that affects Rust code quality must keep deterministic failure conditions.

### 2) `systematic-debugging`
- Mandatory debugging sequence for runtime defects:
  1. Reproduce with a failing test.
  2. Isolate root cause in one crate/module.
  3. Add regression coverage.
  4. Re-run invariant tests for touched crates.
- Do not add logging noise in hot paths unless explicitly guarded.

### 3) `requesting-code-review`
- Review requests must include:
  - affected AXIOMAs,
  - affected Hamiltonian terms,
  - benchmark deltas for hot paths,
  - warning-scan status.
- Reject review-ready state if any required command is missing.

### 4) `executing-plans`
- For complex/cross-module work, update `PLANS.md` before implementation.
- Every plan section must include root cause, file-level actions, and validation commands.

### 5) `code-reviewer`
- Review focus order:
  1. invariants and axiom compliance,
  2. asymptotic complexity,
  3. memory layout/cache locality,
  4. numerical stability,
  5. style/docs.
- Flag regressions when benchmark evidence is absent for hot-path edits.

### 6) `code-quality`
- Enforce non-negotiables:
  - no `unwrap()` in non-test paths for fallible operations,
  - no `HashMap`/`BTreeMap` in hot paths,
  - no `sha2` introduction,
  - no mixed-language docs/comments.

### 7) `rust-mcp-server-generator`
- If used, generated MCP server code must integrate via workspace dependencies and follow crate-boundary rules.
- Generated code must be validated against clippy/warnings and must not bypass `GenesisError` conventions where fallibility is exposed.

### 8) `rust-best-practices`
- Apply idiomatic Rust only when it does not reduce hot-path performance.
- Prefer explicit loops over iterator chains in critical numeric kernels when they improve branch predictability and vectorization.

### 9) `rust-async-patterns`
- Async patterns are allowed only outside deterministic numeric kernels.
- No async abstraction should leak into VFE, synchrony, or HNSW inner loops.

### 10) `rust-engineer`
- Use as a synthesis layer for architecture changes and maintainability hardening.
- All generated docs/comments must remain professional technical English with preserved AX-ID semantics.

## Standard validation profile after skill-driven changes

Run in this order:

1. `cargo check --workspace`
2. `cargo test --workspace`
3. `cargo check --workspace 2>&1 | grep "^warning:"`

If hot paths are modified, also run relevant criterion benchmark(s) and include before/after evidence.
