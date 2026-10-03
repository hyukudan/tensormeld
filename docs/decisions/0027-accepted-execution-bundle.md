# ADR-0027 — Execution authorization exists only as an immutable accepted bundle

Status: **accepted**  
Date: 2026-10-03

## Context

TensorMeld now has planner candidates, adapter representability, E2 backend readiness,
E3 model qualification, runtime model/operator/memory manifests and launch-admitted
resource leases. None of those artifacts alone may authorize execution.

The next milestone needs an executable whole-block lifecycle without weakening any of
those existing boundaries.

## Decision

TensorMeld introduces `tensormeld/accepted-execution-bundle-v1`.

The bundle is created only after TensorMeld:

1. recomputes the planner candidate SHA-256 from config, planning input, profile and the
   complete candidate contents;
2. reruns exact adapter representability;
3. requires runtime-manifest devices/nodes to exactly match the plan;
4. requires complete declared operator coverage;
5. requires applicable E3-or-higher model qualification for the exact adapter, model,
   config, devices and workload;
6. requires the E3 worker artifact to equal the runtime-manifest worker artifact;
7. requires one exact current backend-ready result per compute device;
8. requires one launch-admitted lease per compute node;
9. requires those readiness/admission results to match the exact config and runtime
   manifest identities.

The accepted bundle is immutable and fingerprints all gate identities, segments and
lease/readiness evidence.

This is the first TensorMeld artifact that may state `execution_authorized=true`.

A bounded `ReferenceWholeBlockSession` executes the exact accepted segments through an
injected backend interface and implements prepared/running/completed/cancelled/failed/
released lifecycle semantics.

The built-in validation path uses only deterministic fixture backends and always reports
`real_model_inference=false`.

## Consequences

- Planner candidates, manifests, E2/E3 evidence and admission records retain their
  original non-executable semantics.
- Old/tampered plan hashes cannot be reused after candidate mutation.
- Readiness from another config or leases from another runtime manifest cannot be mixed
  into an accepted execution bundle.
- The native whole-block backend can replace the fixture backend without changing the
  authorization gates.
- Real model inference and distributed execution remain unclaimed until a native backend
  passes correctness and hardware evidence gates.
