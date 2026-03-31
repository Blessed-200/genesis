# Tri-Agent Handoff Protocol

This protocol standardizes collaboration between Codex, CodeRabbit, and human lead review.

## Actors

- **Codex**: implementation and technical execution.
- **CodeRabbit**: policy and review gate.
- **Human lead**: final architectural authority.

## Required artifacts per PR

1. **Codex execution record**
   - root cause,
   - files changed,
   - commands run,
   - failure attribution.
2. **CodeRabbit finding map**
   - accepted findings,
   - rejected findings with evidence,
   - deferred findings with follow-up task IDs.
3. **Lead decision checklist**
   - invariant status,
   - performance-risk status,
   - merge/no-merge rationale.

## Finding-state model

Every CodeRabbit finding must be classified as:

- `accepted` (fixed in PR),
- `rejected-with-proof` (false positive with direct evidence),
- `deferred` (valid but intentionally postponed with task + owner + deadline).

Unclassified findings are not allowed at merge time.

## Merge checklist

A PR is merge-ready only when:

1. all blocking findings are resolved or waived by lead,
2. no unresolved scope-related correctness/performance blocker remains,
3. failure attribution is complete,
4. follow-up tasks for deferred items exist and are linked.

## Escalation

Escalate immediately to human lead when a finding touches:

- metric semantics,
- proof-hash/witness contracts,
- VFE dimensionality,
- lock-free topology correctness,
- benchmark guardrail regressions > 1.25× threshold.

## Quality standard

No placeholder explanations, vague TODOs, or unsupported performance claims are accepted.
All decisions must be reproducible via commands or file-level references.
