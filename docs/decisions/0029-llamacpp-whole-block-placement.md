# ADR-0029 — The first llama.cpp shim translates only complete local blk.N placement

Status: **accepted**  
Date: 2026-10-03

## Context

TensorMeld now has a gated execution bundle and a strict subprocess worker protocol. The
pinned llama.cpp revision exposes real device selection and tensor-buffer override
controls, but its CLI does not provide a generic TensorMeld-style "execute arbitrary
unit range" ABI.

The upstream interface does expose:

- `--device` for engine device names;
- `--override-tensor <pattern>=<buffer type>`;
- model tensors named under `blk.N...` for transformer blocks;
- RPC devices, which are intentionally not used by this first shim because they would
  bypass TensorMeld's authenticated control-plane design.

## Decision

TensorMeld introduces `tensormeld/llamacpp-placement-binding-v1` and
`tensormeld/llamacpp-whole-block-placement-v1`.

The first shim is deliberately local-only and accepts exactly one compute node.

A placement binding is tied to:

- the exact adapter ID/fingerprint;
- the pinned llama.cpp source revision;
- the current native device-binding SHA-256;
- each TensorMeld device's exact engine device name;
- that device's primary buffer type.

Remote RPC devices and non-primary buffer types are rejected.

The translator also requires the complete GGUF tensor index used by the ModelManifest.
It extracts every `blk.N.*` tensor namespace and requires contiguous blocks from zero.
The accepted execution bundle's unit IDs must cover exactly that real block set.

TensorMeld generates one anchored override per block:

`^blk\.N\..*=<approved primary buffer>`

No caller-supplied regex is accepted. The generated placement fragment disables
auto-fit and selects only the approved local engine devices.

This translation remains static placement evidence and always reports
`real_model_inference=false`.

## Consequences

- TensorMeld no longer approximates whole-block ownership with tensor-split ratios.
- Missing/extra real GGUF blocks invalidate translation.
- A stale or different native engine-device mapping invalidates translation.
- llama.cpp RPC cannot silently bypass TensorMeld's mTLS/control boundaries.
- The next step is a pinned worker that consumes this placement spec and an approved
  local GGUF, followed by real full-model correctness qualification.
