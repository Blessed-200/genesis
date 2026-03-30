# Engineering Ownership Matrix

This document defines review ownership and escalation responsibility for architecture-critical domains.

## Ownership model

| Domain | Primary owner | Required reviewer(s) | Escalation trigger |
|---|---|---|---|
| `shared/genesis-types` | Core contracts owner | Dynamics + Topology owner | Any API/ABI contract change |
| `core/genesis-math` | Math/HPC owner | Topology owner | Metric semantics or numerical kernel changes |
| `core/genesis-topology` | Topology owner | Math owner | HNSW lock-free/ordering changes |
| `core/genesis-dynamics` | Dynamics owner | Math owner | VFE/Kuramoto semantics changes |
| CI + governance (`.github`, `.coderabbit.yaml`, `docs/*`) | Platform owner | At least one crate owner | Any quality-gate policy change |

## Mandatory reviewer matrix

The following changes require explicit approval from listed owners:

1. `METRIC_WEIGHTS` profile changes:
   - Math/HPC owner (mandatory)
   - Topology owner (mandatory)
2. Proof hash algorithm or witness contract changes:
   - Core contracts owner (mandatory)
3. VFE gradient dimensionality or belief storage changes:
   - Dynamics owner (mandatory)
   - Math owner (mandatory)
4. HNSW search/prune lock-free path changes:
   - Topology owner (mandatory)
   - Platform owner (mandatory)

## SLA for blocking findings

- First response target: 24 hours.
- Resolution target: 72 hours for scope-related blockers.
- If unresolved after SLA, escalate to human lead with explicit risk memo.

## Merge authority

Human lead has final merge authority and may override with a documented waiver.
Waivers must include:

- invariant impact analysis,
- rollback strategy,
- follow-up task with owner and deadline.
