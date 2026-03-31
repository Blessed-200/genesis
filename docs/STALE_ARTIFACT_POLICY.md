# Stale Artifact Deprecation and Removal Policy

This policy defines how to identify, classify, and remove stale repository artifacts safely.

## Goals

- Keep repository structure legible and professional.
- Remove dead artifacts without losing high-value historical context.
- Prevent accidental deletion of active architectural references.

## Artifact classes

1. **Active specification**
   - Current, authoritative documents used by implementation and review.
2. **Historical reference**
   - Useful for architectural lineage or forensic context.
3. **Forensic artifact**
   - Generated investigation outputs and one-off reports.
4. **Dead artifact**
   - Obsolete, unreferenced, and not needed for compliance or traceability.

## Classification process

1. Build inventory for `docs/`, `files/`, `forensic_reports/`, scripts, and CI logs.
2. For each item, capture:
   - owner,
   - last meaningful update,
   - inbound references (`rg`-based),
   - class decision.
3. Any item without owner defaults to **escalation required**.

## Removal rules

- Remove only artifacts marked **dead**.
- For removed files, include:
  - rationale,
  - references proving non-use,
  - rollback plan.
- For high-noise but valuable historical files, move to `docs/archive/` with concise index.

## Required output for cleanup PR

- `inventory.csv` (artifact, class, owner, disposition)
- changelog section: removed/moved/kept summary
- risk note: why removals are safe

## Non-negotiable constraints

- Never delete active contracts (`contratos_genesis.md`, active axioms/blueprint docs).
- Never remove benchmark guardrail artifacts without replacing traceability.
- Never remove proof-system specifications without owner approval.
