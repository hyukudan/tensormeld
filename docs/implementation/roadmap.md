# Implementation roadmap

## M0.2 — General contracts and policy

Status: **in progress; configuration, policy, synthetic bounded planning and static adapter representability implemented**.

Deliverables:

- v2 installation configuration parser;
- node/device/physical-pool/resource-policy separation;
- profile and execution-mode contracts;
- legal candidate resolver;
- structured policy errors;
- bounded planner interface replacing the legacy three-device assumption;
- tests for 1, 2, 3, 4, 8 and 16 simulated nodes;
- versioned native-adapter capability envelope;
- exact whole-block candidate representability gate with structured rejection;
- no network listener required.

The optional GGUF directory adapter is an early M2 preparation, not a complete model
manifest. Static adapter capabilities are declarations, not live qualification evidence.
Remaining M0.2/M2 work includes trusted capability/availability input and real backend
manifest/admission integration.

Exit gate: the control plane can represent and validate heterogeneous installations and
can distinguish planner feasibility from adapter representability without knowledge of
one particular model or hardware family.

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

- model manifest format;
- device/backend qualification records;
- memory-pool runtime observations;
- directional path profiling and evidence provenance;
- invalidation rules tied to worker/driver/topology changes;
- bounded background/explicit diagnostics.

Exit gate: planning inputs can be traced to measurements or explicit unknowns.

## M3 — First executable distributed inference

- revision-pinned native adapter implementation;
- live adapter qualification against a concrete engine/build;
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
