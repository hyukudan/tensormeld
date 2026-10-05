# Placement calibration validation — 2026-10-05

Implementation slice: measured placement calibration records and bounded coarse→refine
candidate selection.

## Calibration identity

Portable tests validate exact binding to config/planning/profile/model/package/candidate
and a common stable runtime environment. Every runtime identity must use the exact
llama-server artifact from the build package.

Candidate compute devices may be a subset of the runtime environment so one-device and
multi-device plans remain comparable within one sweep.

Physical-pool peak observations may reference only known pools on nodes used by the
candidate and cannot exceed reported physical capacity.

## Workload objective

Prefill and decode are measured independently and normalized to the exact target context
and max-output workload using integer arithmetic.

## Search

Coarse sampling operates only on existing planner candidates. Refinement exposes nearby
unmeasured candidates around the best applicable measured candidate. The helper is bounded
and explicitly does not claim global optimality.

Fixture measurements can exercise ranking/search semantics but are excluded when native
ranking is required.

No hardware benchmark or performance claim is produced by hosted CI.

GitHub Actions result: pending for the implementation PR.
