# ADR-0013 — Runtime availability snapshots are advisory, not reservations

Status: **accepted**  
Date: 2026-10-01

## Context

Static installation policy can describe which resources may participate and the maximum
budget an owner is willing to expose. It cannot establish whether a device is currently
ready or how much of a physical pool is actually available at admission time.

Conversely, a free-memory snapshot is transient. Treating it as an allocation guarantee
would create race conditions and unsafe overcommit.

## Decision

TensorMeld defines an advisory `tensormeld/runtime-observation-v1` snapshot and a
`runtime-select` resolution step.

The snapshot is bound to an exact installation-config fingerprint and may report:

- known configured devices with backend identity and state;
- known physical pools with currently available bytes;
- explicitly false `qualified` and `executable` flags.

Runtime resolution intersects the observation with the already-resolved static owner
policy. For an observed pool:

```text
dynamic_budget = max(0, available_bytes - owner_safety_headroom)
runtime_budget = min(dynamic_budget, static_budget)  # when static budget is known
```

A compute device is runtime-eligible only when it is statically eligible, observed
`ready`, and its physical pool has an observed runtime budget.

Required devices/nodes and minimum participation counts produce structured unmet
requirements rather than silently becoming optional.

The result always states:

```text
reservation_created = false
qualified = false
executable = false
```

## Consequences

- Missing runtime observations are not interpreted as zero-cost or infinite resources.
- A runtime snapshot cannot enlarge an owner's static budget.
- Runtime observations with unknown identities, backend mismatches, impossible available
  memory, wrong config identity, or self-promoted qualification are rejected.
- A fresh observation must be repeated during real admission and converted into an actual
  reservation/lease before execution can be authorized.
- Coordinator availability and model/operator qualification remain separate gates.
