# ADR-0050 — Directional path evidence uses bounded conservative payload buckets

Status: **accepted**  
Date: 2026-10-10

## Context

Legal-unit cost evidence now exposes exact boundary payload sizes. Converting those bytes
into transfer time requires path evidence. Nominal link rate, host ping, symmetric-link
assumptions or adding two cables together are not sufficient.

## Decision

TensorMeld introduces `tensormeld/directional-path-evidence-v1`.

Each explicit directed device-to-device path binds:

- current TensorMeld config identity;
- exact runtime identities for every endpoint device;
- source and target device;
- transport identifier;
- physical resource-sharing group;
- a strictly increasing table of payload buckets.

Each bucket states:

```text
max_payload_bytes
upper_bound_us
```

`upper_bound_us` must be nondecreasing as payload coverage grows.

Lookup semantics are conservative:

- use the first bucket that covers the requested payload;
- if several explicit paths cover it, choose the single lowest upper bound;
- do not interpolate;
- do not extrapolate beyond the largest bucket;
- do not sum parallel paths or infer multirail bandwidth;
- same-device or zero-byte movement costs zero and needs no path record.

Every result remains `qualified=false` and `executable=false`.

## Consequences

- transfer estimates remain auditable and directional;
- asymmetric paths are first-class;
- topology/runtime identity changes invalidate retained path evidence;
- boundary bytes can now be combined with path evidence in a future performance-aware
  advisory planner without fabricating bandwidth;
- explicit multirail qualification, overlap and contention remain later work.
