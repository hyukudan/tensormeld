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
- trusted-local pinned llama.cpp no-model probe:
  - full native artifact SHA-256;
  - bounded `--version` / `--list-devices` execution;
  - pinned source-revision verification;
  - bounded parser for engine-local device/memory observations;
  - no automatic mapping from engine names to TensorMeld IDs;
- dependency/license register and alternate pinned native-engine candidate;
- legacy analytical planner and bounded loopback diagnostics retained.

## Tested here / in portable CI

Contract tests cover policy, planner, adapter representability, exact model/evidence
identity, runtime availability and llama.cpp probe parsing/safety.

GitHub Actions separately validates portable Windows/Linux Python behavior and the real
optional GGUF dependency integration. Hosted CI is not GPU qualification.

The llama.cpp native-probe tests currently use an injected harmless runner. No real
llama.cpp CUDA/HIP binary has been compiled or executed in this development environment.

## Not tested / not implemented

- live mapping of engine-local device names to enrolled TensorMeld device identities;
- real llama.cpp artifact probe on target CUDA/HIP hardware;
- native Windows GPU inference;
- remote enrollment/agent and authenticated private-LAN transport;
- active memory reservations/leases and launch-time re-admission;
- model-specific runtime memory/operator manifest from a real engine;
- native CUDA/HIP/distributed inference;
- hardware/backend E2-E5 evidence on real target devices;
- expert/tensor/phase placement, multirail, GUI/API.

The native probe is E1-style observation at most. It does not load a model, start a
listener, qualify operators or authorize execution. Free memory is transient and remains
non-reserved. No measured RTX, Strix or model speed is claimed.
