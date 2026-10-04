# ADR-0032 — Target-host qualification is assembled from validated artifacts, not automated shell steps

Status: **accepted**  
Date: 2026-10-04

## Context

TensorMeld now has separate contracts for native probe/binding, backend E2, stable runtime
identity, pre-E3 llama.cpp placement, deterministic native trial, E3 correctness,
runtime-model manifests and admission.

Running these manually on a target host creates a new risk: artifacts from different
configurations, plans, binaries, bindings or runtime identities could accidentally be
combined even though each individual artifact is valid.

An orchestration layer is needed, but it must not become a privileged shell runner or
silently manufacture hardware evidence.

## Decision

TensorMeld introduces `tensormeld/target-host-qualification-handoff-v1`.

The assembler performs no subprocess execution or hardware discovery. It accepts
already-produced artifacts and independently validates their identities.

It:
1. recomputes the current llama.cpp bound result from Config + probe + explicit binding;
2. requires the trial's llama-cli artifact to equal the probed artifact;
3. recomputes pre-E3 placement from Config, PlanningInput, candidate, adapter, GGUF index
   and current binding;
4. requires exact RuntimeIdentity and retained native E2 evidence for every compute device;
5. revalidates each retained E2 and combines the resulting device-ready states into one
   runtime observation;
6. reruns E3 evaluation from the native trial and reference contract;
7. requires emitted E3 v2 to apply to the same candidate and runtime identities;
8. emits exact identity requirements for a future native RuntimeModelManifest.

The handoff does not create a RuntimeModelManifest because operator coverage and physical
pool memory must come from the native adapter. It does not reserve memory, authorize
launch or create an AcceptedExecutionBundle.

The E3-tested workload and target profile workload are both recorded. A mismatch remains
visible and blocks later exact applicability; the orchestrator does not widen E3 evidence.

## Consequences

- Valid individual artifacts cannot be mixed across stale bindings/plans/runtimes without
  failing the chain.
- Target-host automation can call existing native steps explicitly and then feed their
  results into one deterministic validator.
- No generic shell or arbitrary command surface is introduced.
- The next real-hardware step is collecting native runtime operator/memory measurements
  for the exact handoff tuple and then entering admission.
