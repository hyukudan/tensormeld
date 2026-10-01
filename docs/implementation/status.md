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
- unmapped engine devices remain visible but are never auto-bound;
- dependency/license register and alternate pinned native-engine candidate;
- legacy analytical planner and bounded loopback diagnostics retained.

## Tested here / in portable CI

Contract tests cover policy, planner, adapter representability, exact model/evidence
identity, runtime availability, llama.cpp probe parsing/safety and explicit device
binding/CLI behavior.

GitHub Actions separately validates portable Windows/Linux Python behavior and the real
optional GGUF dependency integration. Hosted CI is not GPU qualification.

The llama.cpp native-probe/binding tests currently use harmless stored fixtures/injected
runners. No real llama.cpp CUDA/HIP binary has been compiled or executed in this
development environment.

## Not tested / not implemented

- real target-host llama.cpp probe and approved device binding on CUDA/HIP hardware;
- backend self-test that promotes an observed device to runtime-ready;
- native Windows GPU inference;
- remote enrollment/agent and authenticated private-LAN transport;
- active memory reservations/leases and launch-time re-admission;
- model-specific runtime memory/operator manifest from a real engine;
- native CUDA/HIP/distributed inference;
- hardware/backend E2-E5 evidence on real target devices;
- expert/tensor/phase placement, multirail, GUI/API.

The native probe and binding are E1-style identity/observation stages. They do not load
a model, qualify operators or authorize execution. Free memory is transient and remains
non-reserved. No measured RTX, Strix or model speed is claimed.
