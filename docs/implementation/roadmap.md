# Implementation roadmap

## M0.2 — General contracts and policy

Status: **advanced**.

Implemented: heterogeneous registry/policy, bounded synthetic whole-block planning,
adapter representability and advisory runtime availability. Runtime observations remain
snapshots, not reservations.

## M1 — Agent and enrollment

Implemented:

- cross-platform local agent skeleton without a remote listener;
- explicit enrollment/node/key identity supplied at runtime;
- authenticated versioned capability envelope with replay rejection;
- drain/disable/revoke lifecycle and fixed allowlisted operations;
- host-owned physical-pool lease authority over the existing local admission primitive.

Next:

- mutually authenticated encrypted private-LAN channel;
- loopback/private-LAN integration tests through that channel;
- OS-protected key/certificate storage integration and rotation/revocation persistence.

Exit gate: two machines can establish a secure control relationship without exposing an
arbitrary execution surface.

## M2 — Evidence, topology and model manifests

Status: **in progress**.

Implemented:

- exact model/checkpoint identity;
- qualification evidence/applicability contracts;
- advisory runtime physical-pool observations;
- pinned llama.cpp trusted-local no-model probe;
- explicit approved engine-device ↔ TensorMeld-device binding;
- one explicit memory reporter per physical pool;
- observed-but-not-ready native device state;
- pinned upstream `test-backend-ops` readiness contract with target-execution proof;
- narrow observed → ready promotion after strict E2 backend self-test evidence;
- native-only retained E2 backend-readiness record with exact identity fingerprints;
- conservative retained-E2 applicability that still requires a live runtime recheck;
- exact runtime model/operator/memory manifest contract;
- per-device operator declarations plus one memory record per physical pool;
- fixture/native-adapter provenance separation and non-executable CLI validation;
- atomic in-process physical-pool leases for exact manifest preparation peaks;
- launch-time recheck with explicit telemetry/reflected-lease accounting and release.

Next:

- live worker/driver/topology identity and invalidation rules for retained evidence;
- real target-host backend-readiness/runtime-manifest records;
- directional path profiling and evidence provenance.

No real native GPU backend has yet been qualified by TensorMeld; portable fixtures do not
satisfy the real E2/E3 gates.

## M3 — First executable distributed inference

- revision-pinned native adapter implementation;
- real target-host llama.cpp capability probe and device binding;
- real backend self-test and live adapter qualification;
- runtime memory/admission manifest for exact model/workload;
- atomic/leased resource admission and launch-time recheck;
- secure enrolled agent and authenticated private transport;
- whole-block executable adapter across one or more nodes;
- immutable accepted plan;
- stable local streaming API;
- cancellation and deterministic release;
- reference correctness suite;
- same-model comparisons: local vs companion vs distributed.

Exit gate: a model larger than the entrypoint GPU can execute through a verified plan.

## M4 — Planner quality and product UX

- objective-specific scoring;
- plan comparison/explanations;
- profile presets;
- model/session queue;
- local web/desktop UI using the same control API;
- cache/catalog management.

## M5 — Advanced heterogeneous execution

- expert-aware placement;
- tensor/operator strategies;
- prefill/decode phase placement;
- compute/communication overlap;
- multirail qualification;
- optimized Strix Halo worker integration;
- speculative/MTP strategies where measurable and semantically safe.
