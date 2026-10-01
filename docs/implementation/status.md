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
- unmapped engine devices remain visible but are never auto-bound;
- dependency/license register and alternate pinned native-engine candidate;
- legacy analytical planner and bounded loopback diagnostics retained.

## Tested here / in portable CI

Contract tests cover policy, planner, adapter representability, exact model/evidence
identity, runtime availability, llama.cpp probe parsing/safety, explicit device binding,
strict backend-self-test parsing/promotion behavior and retained-E2 identity rules.

GitHub Actions separately validates portable Windows/Linux Python behavior and the real
optional GGUF dependency integration. Hosted CI is not GPU qualification.

The llama.cpp probe, binding, backend-self-test and retained-evidence portable tests use
harmless fixtures or synthetic records where appropriate. Injected-runner self-test
results are explicitly rejected from retained hardware evidence.

No real llama.cpp CUDA/HIP `test-backend-ops` execution has been recorded by TensorMeld
in this development environment.

## Not tested / not implemented

- real target-host llama.cpp probe/binding/backend self test on CUDA/HIP hardware;
- live worker/driver/topology identity needed to safely reuse retained E2 readiness;
- native Windows GPU inference;
- remote enrollment/agent and authenticated private-LAN transport;
- active memory reservations/leases and launch-time re-admission;
- model-specific runtime memory/operator manifest from a real engine;
- native CUDA/HIP/distributed inference;
- retained hardware/backend E2-E5 evidence from real target devices;
- expert/tensor/phase placement, multirail, GUI/API.

The native probe and binding remain E1-style identity/observation stages. The self-test
contract can produce narrow E2 backend-readiness evidence only when the real pinned
native target executes on the bound backend. Retained E2 records preserve that narrow
fact, but cannot by themselves re-promote a later runtime observation because
worker/driver/topology invalidation identity is not implemented yet.

No measured RTX, Strix or model speed is claimed.
