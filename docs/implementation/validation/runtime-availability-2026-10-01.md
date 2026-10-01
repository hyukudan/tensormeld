# Runtime availability validation — 2026-10-01

Implementation slice: advisory runtime observation and policy/budget intersection.

## Scope

This slice imports a bounded runtime snapshot tied to the exact installation config,
filters statically eligible devices by observed readiness, and intersects observed
physical-pool availability with owner caps and safety headroom.

It creates no reservation, does not establish coordinator availability, and does not
qualify model/operator execution.

## GitHub Actions

Workflow run 36877709141 for commit
`e65de15150fd9d4576a0d10ed15c0f3239de5cfe` completed successfully.

Portable Windows/Linux jobs and optional dependency jobs passed. These are software
contract tests only, not GPU or distributed-inference qualification.

## Safety properties exercised

- config fingerprint must match;
- observations cannot self-declare qualified/executable;
- unknown devices and backend mismatches fail;
- impossible available-byte reports fail;
- runtime budget never exceeds static owner policy;
- missing/offline required resources produce structured unmet requirements;
- missing pool observations do not become infinite capacity;
- output cannot overwrite runtime/config inputs;
- duplicate JSON keys are rejected.

## Result

The runtime observation gate is accepted as an advisory pre-admission component.
A fresh launch-time observation and an actual resource reservation/lease remain required
before any execution can be authorized.
