# AI Engineering Operating System

This document defines the enterprise operating protocol for AI-assisted development in GÉNESIS.
It aligns Codex execution, CodeRabbit review, and human lead approval into one deterministic quality system.

## 1. Scope and objective

The objective is to maximize engineering throughput without sacrificing:

- mathematical correctness,
- physical/architectural invariants,
- hot-path performance,
- documentation precision.

The system is optimized for high-assurance Rust HPC workflows.

## 2. Role model

### 2.1 Human lead (final authority)

- Owns architectural intent and acceptance criteria.
- Approves or rejects trade-offs involving invariant risk.
- Decides merge readiness.

### 2.2 Codex (implementation agent)

- Executes bounded tasks with explicit file-level changes.
- Produces reproducible command logs and test evidence.
- Must avoid speculative refactors outside the approved scope.

### 2.3 CodeRabbit (review gate)

- Enforces policy-level checks in pull requests.
- Detects regressions and invariant mismatches.
- Must provide high-signal, actionable findings.

## 3. Quality gate stack (ordered)

Each gate is mandatory and ordered by priority:

1. **Algorithmic complexity:** reject avoidable asymptotic regressions.
2. **Data layout and memory locality:** reject avoidable cache-hostile transformations.
3. **Numerical correctness/stability:** reject unstable or non-robust formulations.
4. **Concurrency safety:** reject race-prone and lock-contention regressions in hot paths.
5. **API and contract integrity:** reject breaking changes not explicitly approved.
6. **Documentation integrity:** reject non-English or ambiguous technical docs.

## 4. Pull request protocol

A pull request is merge-eligible only if all statements below are true:

1. Code compiles and tests pass in workspace.
2. No new warnings are introduced in modified code.
3. CodeRabbit review is resolved with no unresolved blocking comments.
4. All unsafe blocks introduced/changed include explicit safety invariants.
5. Public API changes include updated contract and AX-ID references.

Every PR description must include a **Failure Attribution** block:

- `introduced_by_this_diff`: yes/no
- `scope`: changed crate/module vs external
- `classification`: `scope-related` | `baseline-preexisting` | `external/tooling`
- `evidence`: command output snippet or file/line pointer

## 5. Codex execution protocol

For every non-trivial change, Codex must:

1. Capture root cause.
2. Specify exact file-level actions.
3. Specify validation commands.
4. Execute only approved scope.
5. Report before/after behavior and residual risk.

For cross-module or architecture-adjacent changes, update `PLANS.md` first.

If any command fails outside the requested change scope, Codex must still:

1. diagnose and report root cause,
2. classify failure attribution,
3. either fix immediately (if low-risk/high-leverage) or open a concrete follow-up task with owner and acceptance criteria.

## 6. CodeRabbit policy design principles

The `.coderabbit.yaml` policy should follow these rules:

- Prefer correctness and invariants over style-only comments.
- Flag only actionable findings with direct code impact.
- Treat stale assumptions as policy defects and update promptly.
- Keep strict local knowledge (repo-scoped learnings/issues).
- Use path-specific rules to reduce false positives.

## 7. Learning hygiene

When a CodeRabbit finding is incorrect:

1. Reply with a concise correction citing the exact invariant.
2. Record the correction in local learning scope.
3. If repeated, update `.coderabbit.yaml` rather than repeating manual replies.

Never encode mathematically incorrect statements in learnings.

## 8. Performance doctrine for AI-generated code

Any claim of performance gain must include:

- complexity delta (before/after),
- memory layout implications,
- allocation behavior impact,
- benchmark evidence when touching hot paths.

A stylistically idiomatic change that degrades performance is rejected.

When guardrails fail, the report must include:

1. threshold/value ratio,
2. first-order suspected cause (`algorithmic`, `memory layout/cache`, `allocation`, `branch/vectorization`),
3. minimal reproduction command,
4. remediation plan and expected impact.

## 9. Documentation doctrine

All technical documentation and inline comments must be:

- English-only,
- precise and unambiguous,
- consistent with active contracts and axioms,
- suitable for publication-grade engineering review.

## 10. Escalation policy

Escalate to human lead review immediately when a change affects:

- metric semantics (`METRIC_WEIGHTS` profile),
- proof hash algorithm,
- VFE dimensionality contracts,
- topology consistency gates,
- lock-free correctness in indexing/search paths.

These areas are architecture-critical and require explicit approval.

## 11. Out-of-scope failure triage

All non-scope failures are mandatory triage items, not optional notes.

### 11.1 Classes

- **scope-related:** caused by current diff.
- **baseline-preexisting:** already broken in branch baseline.
- **external/tooling:** infrastructure, toolchain, network, or CI host instability.

### 11.2 Decision rules

- scope-related: fix before merge.
- baseline-preexisting: open remediation task and explicitly gate merge risk.
- external/tooling: rerun with evidence; if persistent, open infra task.

### 11.3 Reporting template

```text
Failure Attribution
- Command:
- Impacted crate/module:
- Classification:
- Why:
- Decision:
```
