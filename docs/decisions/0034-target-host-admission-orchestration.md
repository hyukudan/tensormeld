# ADR-0034 — Execution authorization requires reserve plus a distinct launch observation

Status: **accepted**  
Date: 2026-10-04

## Context

TensorMeld now has exact native qualification and runtime-manifest collection contracts.
The remaining control-plane step is converting those artifacts into an
AcceptedExecutionBundle without weakening local resource admission.

A reservation snapshot cannot also prove launch-time availability because capacity may
change between preparation and launch.

## Decision

TensorMeld introduces `tensormeld/target-host-admission-v1`.

The current llama.cpp local path requires exactly one compute node. The orchestrator:

1. validates the qualification handoff and native runtime collection;
2. requires `admission_ready_inputs=true` and a `native-adapter` manifest;
3. requires config/model/adapter/worker/plan/manifest identities to match;
4. reserves exact physical-pool preparation peaks under the explicit compute-node
   authority;
5. requires a second admission snapshot with a different observation ID;
6. performs launch recheck against that fresh observation;
7. constructs AcceptedExecutionBundle from the exact E3 v2, backend readiness, runtime
   manifest and launch-admitted lease.

If launch recheck fails, or bundle construction fails after reservation, TensorMeld
releases the lease before returning failure.

Successful orchestration authorizes execution but does not start inference.

## Consequences

- Reservation-time availability cannot be reused as launch-time evidence.
- Failed orchestration does not intentionally leave an active lease behind.
- AcceptedExecutionBundle becomes the exact boundary between admission and native worker
  startup.
- Real inference remains a separate later action with lifecycle/release responsibilities.
