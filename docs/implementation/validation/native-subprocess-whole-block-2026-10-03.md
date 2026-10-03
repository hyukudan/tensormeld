# Native subprocess whole-block foundation validation — 2026-10-03

Implementation slice: first revision-pinned whole-block worker process boundary behind
the AcceptedExecutionBundle.

## Contract

`NativeSubprocessWholeBlockBackend` verifies a trusted-local launcher and optional
program by exact SHA-256. The launched worker artifact must equal the worker artifact
already bound into the accepted bundle and its engine revision must match the bundle.

The command line is fixed, uses `shell=False`, and accepts no user/peer supplied argv.
Each segment request is bounded JSON over stdin and includes the exact accepted-bundle,
worker, launcher/program, device/unit and segment identities. Returned output is accepted
only after request/segment/bundle/worker/revision identities match.

The protocol currently requires `real_model_inference=false`.

## Real subprocess coverage

Portable Windows/Linux tests launch a real subprocess. Python is used only as the hashed
test launcher; a separate hashed fixture program implements the protocol and performs a
deterministic bytes transform.

Tests also reject artifact hash changes, wrong engine revision, backend/session bundle
mismatch, non-zero process exit, response identity tampering and a worker claiming real
model inference.

## OSS interface review

The existing registry entries for pinned `llama.cpp@552f18f...` and
`llama-halo-hybrid@f072119...` were re-reviewed. Their placement/RPC capabilities are
not treated as a generic architecture-neutral whole-block ABI. A model-aware shim remains
required before a llama.cpp-family worker can implement this protocol.

No GGUF tensor execution, real model correctness, CUDA/HIP kernel execution, performance
measurement or distributed inference is established by this validation.

GitHub Actions result: PR #13, workflow `Portable Python tests`, run #139 (37121130652) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. The new native-worker tests execute a real subprocess on Windows/Linux, but the worker is still a deterministic fixture and `real_model_inference=false` remains mandatory.
