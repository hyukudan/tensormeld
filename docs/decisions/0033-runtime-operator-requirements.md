# ADR-0033 — Runtime operator requirements are independent from runtime observations

Status: **accepted**  
Date: 2026-10-04

## Context

The target-host qualification handoff identifies one exact config/model/plan/runtime
tuple, but admission still needs a RuntimeModelManifest containing operator coverage and
physical-pool memory peaks.

If one measurement artifact were allowed to declare both "required operators" and
"observed operators", it could trivially shrink the required set and self-report complete
coverage. Shared/unified memory also must remain represented once per physical pool, not
once per logical device.

## Decision

TensorMeld introduces:

- `tensormeld/runtime-operator-requirements-v1`;
- `tensormeld/native-runtime-measurement-v1`;
- `tensormeld/native-runtime-manifest-collector-v1`.

Operator requirements are a separate artifact bound to the exact qualification handoff,
model, candidate plan and placement. Runtime measurement reports only observed operators
per device and physical-pool memory values.

Runtime measurement is bound to:
- target-host handoff SHA;
- config/model/adapter/engine identities;
- exact worker artifact;
- exact profile workload;
- exact per-device runtime-identity SHA.

The collector requires every handoff compute device and at least every compute device's
physical pool. Additional pools are allowed only on used nodes. Duplicate pool records are
rejected.

The collector builds the existing RuntimeModelManifest and delegates final workload,
capacity, preparation-peak and operator-coverage invariants to its canonical parser.

RuntimeModelManifest provenance becomes `native-adapter` only when both the runtime
measurement and operator-requirement source are native. Any mixed/fixture source produces
fixture provenance.

The collector never reserves memory, authorizes launch or makes the workload executable.

## Consequences

- Operator coverage cannot be self-certified by redefining the requirement set.
- Shared/unified memory remains single-counted at physical-pool granularity.
- Fixture/native provenance cannot be upgraded by mixing one native label with fixture
  evidence.
- The output can be passed directly to LocalAdmissionController once a fresh runtime
  observation exists.
