# ADR-0028 — Native model workers sit behind a fixed subprocess protocol

Status: **accepted**  
Date: 2026-10-03

## Context

TensorMeld has an accepted execution bundle and an in-process reference whole-block
backend. The next step needs a real process boundary without exposing arbitrary command
execution or inventing placement flags that the selected inference engine does not
actually support.

The pinned `llama.cpp` source provides backend, multi-device and RPC mechanisms.
The reviewed `llama-halo-hybrid` fork also demonstrates model-specific tensor override
and multi-host layouts. Neither is treated as a stable generic ABI equivalent to
TensorMeld's architecture-neutral `unit_ids + whole-block segments` contract.

## Decision

TensorMeld introduces `tensormeld/native-whole-block-worker-v1`.

A local subprocess backend may launch only artifacts whose SHA-256 identities are
approved before execution. It supports either:

- one native executable, or
- one approved launcher plus one separately approved worker program.

The argv shape is closed and generated entirely by TensorMeld:

`[launcher, program?, "--tensormeld-worker-v1"]`

There is no shell and no caller/peer supplied command line.

All dynamic segment data travels as one bounded canonical JSON request over stdin. The
request binds the accepted bundle, adapter ID, engine revision, worker artifact,
launcher/program artifacts, exact device and unit IDs, segment fingerprint and payload.

The worker response must bind the exact request SHA-256, segment, bundle, worker and
engine revision before any returned bytes are accepted.

Until a model-aware shim has passed real correctness qualification, this protocol requires
`real_model_inference=false` and rejects a worker that attempts to self-promote it.

The accepted execution bundle now also records the exact engine revision, and an
execution session rejects a backend bound to a different accepted-bundle fingerprint.

## Consequences

- A remote or local caller cannot turn the worker interface into arbitrary process launch.
- Replacing a worker program changes the SHA and invalidates the accepted worker identity.
- Engine revision is carried through E3/adapter/bundle/worker boundaries.
- llama.cpp-family integration can be added as a narrow shim without changing the
  orchestration gates.
- Current real subprocess tests prove process/protocol behavior only, not model inference.
