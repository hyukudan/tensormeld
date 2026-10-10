# ADR-0048 — Legal-unit planning is advisory resident-capacity search only

Status: **accepted**  
Date: 2026-10-10

## Context

TensorMeld now has explicit tensor movability evidence and adapter-declared ordered legal
model units with indivisible tensor groups and cut boundaries. Those contracts describe
where tensors may legally live, but they do not provide per-unit compute timings,
communication costs, persistent-state migration, workspace, or staging behavior.

Treating legal units as executable layers would overstate the evidence.

## Decision

TensorMeld adds a bounded advisory planner over `legal-model-units-v1`.

The planner:

- reuses the existing static selection/policy resolver;
- assigns each complete legal unit to one currently policy-eligible allowed device;
- permits an owner change only after an explicit `cut_after=true` boundary;
- accounts each unit's hard-resident plus reclaimable/file-backed tensor bytes against
  the owning device's physical pool;
- preserves shared-pool accounting by pool identity rather than logical device count;
- enforces required devices/nodes, local-GPU constraints and configured min/max owners;
- uses the existing candidate/deadline search budgets;
- reports `SEARCH_INCOMPLETE` rather than infeasibility when a budget expires.

Ranking is intentionally limited to resident-capacity heuristics: fewer devices, fewer
nodes, then lower maximum static pool utilization. It is **not** a latency or throughput
ranking.

Every result remains:

```text
qualified = false
executable = false
objective = resident_capacity_only
```

## Consequences

- TensorMeld can explore finer legal placements without inventing illegal cuts.
- Reclaimable/file-backed tensor bytes are still counted in full.
- The planner cannot authorize native execution.
- Whole-block placement remains the validated executable baseline.
- A future native adapter must provide per-unit state/workspace/transfer/compute evidence
  before these placements can enter an executable path.
