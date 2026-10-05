# Managed llama-server process validation — 2026-10-05

Implementation slice: TensorMeld-owned persistent llama-server child-process lifecycle.

## Portable process coverage

Windows/Linux tests launch a real Python fixture server through an approved launcher and
program artifact. The fixture accepts only the closed llama-server-shaped argv and serves
a loopback /health endpoint.

Tests validate:
- exact bundle/model/GGUF/placement launch binding;
- loopback-only host and one server slot;
- removal of inherited LLAMA_ARG_* overrides;
- readiness over HTTP;
- bounded stderr log tail;
- early-exit diagnostics;
- idempotent stop;
- POSIX terminate→kill fallback for a process that ignores SIGTERM.

No inference request is issued.

## Identity boundary

The server artifact is recorded separately from the current llama-cli E3 artifact.
Portable process success does not upgrade E3, runtime manifest provenance or
real_model_inference.

GitHub Actions result: pending for the implementation PR.
