# ADR-0018 — Runtime manifests bind exact workload identity and physical-pool memory

Status: **accepted**  
Date: 2026-10-02

## Context

Synthetic planning inputs intentionally estimate placement costs and memory. They are
not execution manifests. Before live admission, TensorMeld needs an adapter-produced
description of what an exact model/workload requires at runtime.

Memory cannot be attached independently to every logical device because multiple
devices may alias one physical pool, particularly unified/shared memory systems.

Operator support also cannot be inferred from backend names or a successful isolated
ADD self-test.

## Decision

TensorMeld introduces `tensormeld/runtime-model-manifest-v1`.

A runtime manifest is bound exactly to:

- TensorMeld configuration fingerprint;
- model-manifest fingerprint;
- selected profile and its exact workload;
- adapter ID and capability fingerprint;
- engine revision and worker-artifact SHA-256;
- explicit worker device/node/backend identities.

The manifest declares required operators and the operators exposed for every included
device. Complete declaration coverage is reported separately and never upgrades the
manifest to E3 qualification.

Runtime memory is declared once per **physical pool**. Each pool record contains
resident bytes, state bytes, peak workspace and preparation peak. Preparation peak must
cover the steady peak and must not exceed known physical capacity.

The contract supports `fixture` and `native-adapter` provenance. Fixture provenance
exists for portable tests and cannot be presented as hardware evidence.

Parsing or validating a manifest always leaves
`reservation_created=false`, `qualified=false` and `executable=false`.

## Consequences

- Shared/unified pools are not multiplied by logical-device count.
- Model/workload/config/adapter drift invalidates the manifest.
- Backend labels do not imply operator coverage.
- Complete operator declaration is still weaker than model correctness qualification.
- Live admission must intersect these exact peaks with current pool observations and
  owner policy using an atomic lease; that is the next slice.
