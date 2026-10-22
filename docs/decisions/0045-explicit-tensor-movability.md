# ADR-0045 — Tensor movability is explicit evidence, never a name/size heuristic

Status: **accepted**  
Date: 2026-10-08

## Context

Predictive pool memory distinguishes hard and reclaimable pressure, but it does not explain
which exact tensors can legally change owner or remain file-backed.

Tensor names such as `blk.N.*`, backend labels, tensor size, or model family are not
sufficient evidence of movability. Engines can have tied tensors, mandatory local
operators, repacked buffers, backend-specific restrictions and hidden ownership rules.

## Decision

TensorMeld introduces `tensormeld/tensor-movability-v1`.

The profile is bound exactly to:

- the model-manifest fingerprint;
- the complete GGUF tensor-index fingerprint, which is recomputed before use;
- adapter-capability fingerprint;
- llama.cpp build-package fingerprint;
- exact current runtime identities.

Every tensor in the complete GGUF index must appear exactly once with its exact indexed byte
size and an explicit storage and movement classification.

Storage classes:

- `hard_resident`;
- `reclaimable_file_backed`.

Movement classes:

- `pinned`: exactly one allowed device;
- `owner_local`: one or more allowed devices, all on one node;
- `placement_movable`: one or more explicitly allowed current devices.

Allowed devices must be exposed by the current config/adapter and covered by current runtime
identities. Runtime identities must use the exact llama-server artifact from the declared
build package.

Optional alias groups are explicit evidence. A group must contain at least two tensors and
all members must share the same storage/movement policy.

The profile never self-promotes qualification or executability and does not change current
planner ownership or runtime admission.

## Consequences

- tensor movement cannot be inferred from regexes, names, sizes or backend families;
- stale driver/runtime/device/topology or package identity invalidates the profile;
- tampering with GGUF index contents while retaining an old fingerprint is detected;
- complete tensor coverage is required, so unknown tensors fail closed;
- future planners can consume this evidence to construct legal finer-grained candidates,
  but only after dedicated correctness and memory-admission gates are implemented;
- fixture profiles validate contract behavior only; real target-host movability remains
  unqualified until measured/verified on those hosts.
