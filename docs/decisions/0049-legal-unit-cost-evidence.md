# ADR-0049 — Legal-unit cost evidence is explicit and identity-bound

Status: **accepted**  
Date: 2026-10-10

## Context

Legal model units and cut boundaries establish where TensorMeld may legally place complete
tensor groups. Resident-capacity planning can therefore answer whether a legal assignment
fits static pool budgets, but it cannot estimate whether a split is worthwhile.

Per-unit compute time, persistent state, workspace, staging and inter-owner payload size
cannot be inferred safely from tensor size, tensor names, backend labels or model family.

## Decision

TensorMeld introduces `tensormeld/legal-unit-costs-v1`.

The profile is bound exactly to:

- TensorMeld config fingerprint;
- legal-model-units fingerprint;
- tensor-movability fingerprint;
- runtime-model-manifest fingerprint;
- therefore the exact model, adapter/build and runtime identities already carried by
  those upstream evidence contracts.

For every legal unit the profile requires:

- exact sequence and unit ID;
- `boundary_output_bytes` describing payload size after the unit;
- an explicit device profile for **every** legal allowed device;
- positive measured/declared `compute_us`;
- physical-pool maps for persistent state, workspace peak and staging peak.

All memory maps may reference only physical pools local to the profiled device node.
The final legal unit must declare zero boundary output because no following unit exists.

Aggregation semantics are fixed for future planning:

- persistent state is additive across assigned units;
- workspace peak is the maximum per device/pool across assigned units;
- staging peak is the maximum per device/pool across assigned units;
- boundary payload is a byte count only and needs independent directional path evidence
  before transfer time can be computed.

The contract remains `qualified=false` and `executable=false`.

## Consequences

- performance-aware legal-unit ranking can be added without inventing costs;
- incomplete device profiling is rejected rather than silently extrapolated;
- cross-node pool references and model/adapter/runtime drift fail closed;
- transfer payloads do not imply bandwidth or latency;
- whole-block remains the validated executable baseline until a native adapter proves
  the finer-grained path end to end.
