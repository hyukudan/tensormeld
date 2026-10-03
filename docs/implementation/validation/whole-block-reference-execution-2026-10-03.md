# Whole-block executable reference path validation — 2026-10-03

Implementation slice: first executable reference adapter behind all existing gates.

## Accepted execution bundle

The bundle recomputes exact planner integrity and adapter representability, validates an
E3+ model qualification record, requires worker-artifact identity with the runtime
manifest, requires exact backend-ready coverage for every compute device and exact
launch-admitted coverage for every compute node.

Launch-admission results now propagate node/config/runtime-manifest identity. Backend
readiness applicability propagates config/device/runtime-identity identity. The execution
bundle rejects identity mixing.

## Reference execution

`ReferenceWholeBlockSession` executes the immutable segment list through a bounded
backend interface. Portable tests use a deterministic bytes-transform fixture backend,
verify exact segment/unit order, cancellation before execution and idempotent release.

The bundle may state `execution_authorized=true` because all software gates have passed,
but the reference run always states `real_model_inference=false`.

No GGUF model tensors, CUDA/HIP kernels, native worker, performance benchmark or
distributed inference is executed by this validation.

GitHub Actions result: PR #12, workflow `Portable Python tests`, run #127 (37120543892) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. This validates the accepted-bundle gates and deterministic reference execution lifecycle only; `real_model_inference=false` remains mandatory for this reference backend.
