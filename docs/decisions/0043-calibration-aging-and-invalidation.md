# ADR-0043 — Persist measured placement calibrations with explicit aging and invalidation

Status: **accepted**  
Date: 2026-10-08

## Context

Placement calibration can reorder planner candidates using native measurements, but a
measurement remains meaningful only for the exact environment in which it was taken.
Driver/runtime/package/topology changes can invalidate it even when the model and planner
candidate are unchanged.

Reusing a calibration indefinitely would turn historical data into an unbounded performance
claim.

## Decision

TensorMeld persists each placement calibration as one immutable local JSON record named by
the calibration fingerprint.

A persisted record contains:

- the validated placement-calibration record;
- a timezone-aware recording timestamp;
- an explicit bounded maximum age.

Persistence is idempotent only when an existing fingerprint file is byte-identical.
Conflicting content under the same fingerprint fails closed.

Before a persisted calibration can influence measured preference, TensorMeld:

1. rejects records from the future;
2. rejects records older than their configured maximum age;
3. re-parses the calibration against the current config, planning input, retained candidate,
   llama.cpp build package and runtime identities;
4. therefore invalidates changes in worker artifact, OS/driver/runtime/device/topology
   identity already covered by `RuntimeIdentity`;
5. recomputes the derived normalized objective and checks it against the persisted value;
6. optionally requires genuine `native-target` provenance.

Expired or incompatible records remain auditable but are excluded from the preference input.
They never mutate planner candidates and never imply qualification or executability.

## Consequences

- calibration persistence is safe across process restarts without treating measurements as
  evergreen;
- package/runtime/topology changes require fresh measurements;
- fixture records can exercise persistence logic but cannot override native-mode ranking;
- the store remains local and simple; a future database/index may replace the directory
  representation without changing applicability semantics;
- real RTX/Strix calibration still requires execution on those target hosts.
