# ADR-0026 — Retained E2 reuse requires an exact stable runtime identity

Status: **accepted**  
Date: 2026-10-03

## Context

Retained backend-readiness evidence v1 bound config, probe, device mapping and test
artifacts but deliberately could not restore runtime `ready`. It lacked stable identity
for the worker build, operating system, driver/runtime stack, physical device and topology.

Rerunning an isolated backend self-test after every process restart is conservative but
unnecessary when the complete invalidation identity can be proven unchanged.

## Decision

TensorMeld introduces `tensormeld/runtime-identity-v1` and
`tensormeld/backend-readiness-evidence-v2`.

The stable runtime identity binds:

- node ID;
- worker artifact SHA-256 and build ID;
- OS name/version;
- driver ID/version;
- runtime ID/version;
- TensorMeld device ID;
- stable physical-device ID;
- topology SHA-256.

Transient observations such as timestamps and free memory are excluded.

A native self-test result now also records the exact node ID. Retaining E2 v2 evidence
requires the supplied runtime identity to match that self-test node/device.

Later applicability requires all existing retained backend identities, a fresh
`observed` binding, and the complete current runtime-identity fingerprint to match
exactly. Only then may TensorMeld reuse the narrow backend-ready fact and promote that
single device to `ready` in a copied runtime observation.

The result still keeps `reservation_created=false`, `qualified=false` and
`executable=false`.

Legacy v1 retained evidence lacks this invalidation identity and therefore cannot satisfy
the v2 readiness-reuse gate.

## Consequences

- Worker rebuild, OS, driver, runtime, physical-device or topology changes invalidate
  retained backend readiness.
- Exact identity reuse avoids treating every process restart as new hardware evidence.
- Backend readiness remains strictly weaker than model/operator qualification.
- Resource admission and executable-plan authorization remain independent later gates.
