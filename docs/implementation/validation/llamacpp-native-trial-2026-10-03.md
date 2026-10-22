# llama.cpp pre-E3 native trial foundation validation — 2026-10-03

Implementation slice: break the E3 bootstrap cycle and define a strict native trial.

## Pre-E3 qualification placement

A planner candidate is independently hash-validated and rerun through exact adapter
representability. The placement layer then applies the existing pinned llama.cpp, native
binding and complete GGUF block-coverage checks. The result is non-qualified and
non-executable.

This path shares canonical plan identity validation with the post-E3 execution bundle.

## Native trial specification

The initial trial accepts one local GGUF only. TensorMeld verifies the file name, size and
SHA-256 against ModelManifest and verifies the local llama-cli artifact SHA-256.

The generated command is a closed argv containing the approved model path, exact placement
fragment and bounded deterministic generation controls. No arbitrary extra argv or llama.cpp
RPC endpoint is accepted.

The default runner uses `shell=False`, bounded stdout/stderr and removes inherited
`LLAMA_ARG_*` variables. Injected runners cannot claim `native-subprocess` provenance.

## Portable coverage

CI runs a real fixture subprocess through the closed argv and verifies the pre-E3
planner→placement→trial path without constructing an AcceptedExecutionBundle. Tests also
cover GGUF tampering, multi-file rejection, prompt/context bounds, environment scrubbing
and provenance mislabeling.

No real llama.cpp GGUF is loaded in hosted CI. No CUDA/HIP backend, model correctness,
performance or distributed inference is established.

GitHub Actions result: PR #15, workflow `Portable Python tests`, run #168 (37135609787) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. Hosted CI validates the pre-E3 qualification-placement, closed argv, fixture subprocess, environment-scrubbing and provenance boundaries only; no real llama.cpp GGUF execution occurred.
