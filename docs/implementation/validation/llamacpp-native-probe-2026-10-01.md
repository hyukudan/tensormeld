# Pinned llama.cpp native-probe validation — 2026-10-01

Implementation slice: trusted-local, no-model probe for the pinned llama.cpp revision.

## Scope

The probe hashes one explicitly trusted local executable and invokes only `--version`
and `--list-devices` with `shell=False`, no stdin, bounded timeout and bounded output.

It verifies the reported source commit against the pinned revision and parses only
engine-local device names/descriptions plus total/free memory. It does not load a model,
start RPC/server listeners, infer CUDA/HIP identity, map engine devices to TensorMeld IDs,
or authorize execution.

## GitHub Actions

Workflow run 36878373117 for commit
`995f8a228516a36caf211d42748648d5a7a3c193` completed successfully.

Portable Windows/Linux and optional dependency jobs passed. The native-probe tests use
an injected harmless runner; no real llama.cpp CUDA/HIP artifact was compiled or executed
as part of this validation.

## Result

The probe contract is accepted as a bounded E1-style observation mechanism. Real target
hardware use still requires explicit engine-device binding, live binary observation and
later model/operator qualification.
