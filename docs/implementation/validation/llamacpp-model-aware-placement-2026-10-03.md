# Pinned llama.cpp model-aware placement validation — 2026-10-03

Implementation slice: exact static translation from accepted TensorMeld whole-block
ownership to pinned llama.cpp tensor placement controls.

## Upstream contract used

At pinned llama.cpp revision `552f18f912a32ea86edf82e2b76431cb7131538d`,
TensorMeld relies only on the documented/local interfaces:

- `--device` for device selection;
- `--override-tensor <tensor name pattern>=<buffer type>`;
- transformer tensor names under the `blk.N.*` namespace;
- `--fit off` to prevent automatic placement adjustment.

TensorMeld does not use `--tensor-split` as a substitute for exact ownership.

## Exactness gates

The shim requires:

- accepted bundle adapter/revision/fingerprint identity;
- one compute node only;
- current native binding SHA matching the placement binding;
- exact engine-device names matching current resolved mappings;
- primary local buffer type equal to the approved engine device;
- no RPC device/buffer;
- complete GGUF index identity matching the ModelManifest;
- contiguous real GGUF block indices starting at zero;
- exact equality between real GGUF blocks and accepted bundle `blk.N` units;
- exact non-overlapping segment coverage.

The generated regex patterns are internal and anchored. No user-provided regex or command
line is accepted.

## Portable coverage

Tests cover successful local two-device block placement, deterministic translation,
non-block/gapped/reordered units, overlapping or incomplete segment coverage, stale
native binding, changed engine mapping, partial placement binding, remote RPC rejection,
non-primary buffers, incomplete real GGUF block coverage, index mismatch and malformed
`blk.*` tensor namespaces.

No GGUF is loaded by llama.cpp in this validation. No real model correctness, CUDA/HIP
kernel execution, KV/compute placement, performance or distributed inference is claimed.

GitHub Actions result: pending for the implementation PR.
