# Measured planner preference validation — 2026-10-05

Implementation slice: evidence-backed recommendation overlay over immutable planner-v2
candidates.

## Invariants

The overlay validates:
- planner-v2 schema/provenance;
- retained candidate uniqueness;
- planner best is exactly one retained candidate;
- calibration config/planning/profile identity;
- calibration candidate exists in the retained set;
- calibration fingerprints.

It never mutates planner result or candidate content.

## Recommendation behavior

If applicable native calibration exists, the lowest measured workload objective becomes
the measured recommendation. Otherwise the planner's synthetic best remains recommended.

Fixture calibration is ignored when native measurements are required.

## Portable coverage

Tests demonstrate measured override of synthetic ordering, exact preservation of original
planner output, fixture/native provenance separation, unretained-candidate rejection,
planning-identity mismatch rejection and tampered planner-best rejection.

No hardware benchmark or performance claim is created by hosted CI.

GitHub Actions result: PR #27, workflow `Portable Python tests`, run #263 (37364578939) ultimately passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. Several jobs were cancelled by runner availability and re-run individually; the completed attempts were green. Hosted CI validates recommendation-overlay semantics only and does not establish target-hardware performance.
