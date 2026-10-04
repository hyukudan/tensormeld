# Native runtime-manifest collector validation — 2026-10-04

Implementation slice: handoff-bound native operator/memory measurement collection into
RuntimeModelManifest.

## Inputs

The collector requires an intact target-host qualification handoff plus:

- an independent runtime-operator requirements contract;
- a runtime measurement contract containing per-device observed operators and one record
  per measured physical pool.

Both are bound to the same handoff. Device observations carry exact runtime-identity
fingerprints from the qualification chain.

## Invariants

The collector validates:
- config/model/adapter/engine/worker identity;
- exact profile workload;
- exact candidate devices and nodes;
- runtime identity per device;
- independent operator requirements;
- compute-device physical-pool coverage;
- no duplicate physical pool records;
- no pools from unused nodes.

The existing RuntimeModelManifest parser then enforces device/backend identity, physical
capacity, steady/preparation peak relationships and complete/incomplete operator coverage.

Native manifest provenance requires both measurement and requirements sources to be
native. Mixed or fixture provenance produces a fixture manifest.

## Output

The collection record contains the runtime-manifest fingerprint and reports whether the
inputs are admission-ready, but always keeps:
- reservation_created=false;
- launch_authorized=false;
- executable=false.

Portable tests use fixture/mixed provenance. No native operator probe, memory measurement,
physical GPU or model inference is executed by this validation.

GitHub Actions result: pending for the implementation PR.
