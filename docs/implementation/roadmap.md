# Implementation roadmap

## M0.2 — General contracts and policy

Status: **advanced**.

Implemented: heterogeneous registry/policy, bounded synthetic whole-block planning,
adapter representability and advisory runtime availability. Runtime observations remain
snapshots, not reservations.

## M1 — Agent and enrollment

- cross-platform agent skeleton;
- authenticated pairing and node identity;
- signed/versioned capability envelope;
- health, drain, disable and worker lifecycle;
- local owner policy is authoritative;
- loopback and private-LAN integration tests.

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
- observed-but-not-ready native device state.

Next:

- backend self-test/qualification that can promote an observed bound device to ready;
- real target-host device/backend qualification records;
- runtime model/operator/memory observations;
- directional path profiling and evidence provenance;
- invalidation rules tied to worker/driver/topology changes.

No native GPU binary has been qualified by TensorMeld yet.

## M3 — First executable distributed inference

- revision-pinned native adapter implementation;
- real target-host llama.cpp capability probe;
- explicit engine-device identity binding;
- backend self-test and live adapter qualification;
- runtime memory/admission manifest for exact model/workload;
- atomic/leased resource admission and launch-time recheck;
- whole-block placement across one or more nodes;
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
