# Target-host qualification orchestrator validation — 2026-10-04

Implementation slice: fail-closed assembly of the native qualification evidence chain.

## Chain checks

The orchestrator verifies:
- Config/PlanningInput/ModelManifest identity;
- freshly recomputed probe/binding result;
- probe artifact == trial llama-cli artifact;
- recomputed pre-E3 placement == supplied placement;
- exact retained E2 + RuntimeIdentity for every candidate compute device;
- combined device-ready runtime observation;
- native-subprocess E3 trial/reference correctness;
- E3 v2 applicability to the same plan and runtime identities.

## Handoff

A successful handoff includes exact identities required by the future native runtime
manifest:
- config/profile/model;
- adapter capability fingerprint and engine revision;
- worker artifact;
- compute devices/nodes;
- target profile workload;
- E3-tested workload and whether it exactly matches the profile;
- requirement for native operator and physical-pool memory measurement.

The handoff remains:
- runtime_manifest_required=true;
- reservation_created=false;
- launch_authorized=false;
- executable=false.

## Portable coverage

Tests use native-shaped fixture artifacts and exercise successful assembly plus stale
binding, changed runtime identity, fixture trial provenance, probe/trial artifact mismatch
and incomplete E2 coverage.

No subprocess is launched by the orchestrator. No physical GPU, real llama.cpp model
correctness, native memory measurement or distributed inference is established.

GitHub Actions result: pending for the implementation PR.
