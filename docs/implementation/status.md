# Current implementation status

Version: **0.2.0a2** — TensorMeld. Specification baseline: 0.2.0.

The canonical repository is public for engineering transparency. This does not change
the pre-alpha status or qualify any execution path.

## Implemented

- strict heterogeneous installation/policy contracts and bounded synthetic planner;
- exact static adapter representability gate;
- exact GGUF-backed model identity and qualification-evidence applicability contracts;
- advisory runtime-observation and conservative runtime-pool budget resolution;
- optional psutil host inventory and upstream GGUF catalog reuse;
- trusted-local pinned llama.cpp no-model probe with artifact/source identity;
- explicit approved one-to-one llama.cpp engine-device ↔ TensorMeld-device binding;
- backend identity taken from TensorMeld config, never inferred from engine labels;
- optional single memory reporter per physical pool to avoid shared-memory double counting;
- bound native devices enter runtime observations as `observed`, not `ready`;
- strict pinned llama.cpp `test-backend-ops` readiness adapter for one bound device;
- positive target-backend execution proof is required in addition to exit code;
- successful narrow E2 readiness can promote only that bound device to runtime `ready`;
- native-only retained E2 backend-readiness records with deterministic fingerprints;
- exact config/probe/binding/test-artifact/device/backend applicability checks for retained E2;
- retained E2 evidence never restores `ready` without a live runtime recheck;
- strict runtime model/operator/memory manifests tied to exact config/model/profile/workload/adapter identities;
- runtime memory is represented once per physical pool with resident/state/workspace/preparation peaks;
- operator coverage is explicit per device and cannot self-promote qualification/execution;
- CLI validation for runtime manifests without creating resource reservations;
- in-process atomic physical-pool leases for exact runtime manifests;
- launch-time re-admission requires a newly identified runtime observation;
- active leases can be explicitly marked reflected in telemetry to avoid double subtraction;
- deterministic local lease release and concurrent admission serialization;
- unmapped engine devices remain visible but are never auto-bound;
- dependency/license register and alternate pinned native-engine candidate;
- legacy analytical planner and bounded loopback diagnostics retained.

## Tested here / in portable CI

Contract tests cover policy, planner, adapter representability, exact model/evidence
identity, runtime availability, llama.cpp probe parsing/safety, explicit device binding,
strict backend-self-test parsing/promotion behavior, retained-E2 identity rules and
runtime-manifest identity/pool/operator invariants.

GitHub Actions separately validates portable Windows/Linux Python behavior and the real
optional GGUF dependency integration. Hosted CI is not GPU qualification.

The llama.cpp probe, binding, backend-self-test and retained-evidence portable tests use
harmless fixtures or synthetic records where appropriate. Runtime-manifest tests use
fixture provenance and do not claim native runtime measurements. Injected-runner
self-test results remain rejected from retained hardware evidence.

No real llama.cpp CUDA/HIP `test-backend-ops` execution or real native runtime-model
manifest has been recorded by TensorMeld in this development environment.

## Not tested / not implemented

- real target-host llama.cpp probe/binding/backend self test on CUDA/HIP hardware;
- real native-adapter model/operator/memory manifest capture on target hardware;
- live worker/driver/topology identity needed to safely reuse retained E2 readiness;
- native Windows GPU inference;
- remote enrollment/agent and authenticated private-LAN transport;
- native CUDA/HIP/distributed inference;
- retained hardware/backend E2-E5 evidence from real target devices;
- expert/tensor/phase placement, multirail, GUI/API.

Runtime manifests are evidence inputs, not reservations. The local admission controller can
now reserve their exact preparation peaks atomically within one process and recheck before
launch, but this is not yet a distributed/agent lease service. Explicit reflected-lease IDs
separate telemetry that already includes an allocation from pending logical reservations so
committed bytes are not necessarily subtracted twice.

No measured RTX, Strix or model speed is claimed.
