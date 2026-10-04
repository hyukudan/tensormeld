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

GitHub Actions result: PR #17, workflow `Portable Python tests`, run #182 (37225661220) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. Hosted CI validates cross-artifact qualification-chain semantics using native-shaped fixture records only; it does not establish real target-host E2/E3, native memory measurement or GPU/model execution.
