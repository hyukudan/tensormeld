# Retained backend E2 evidence validation — 2026-10-01

Implementation slice: persist narrow native backend-readiness evidence without turning a
historical pass into execution authorization.

## Contract validated

The new `backend_evidence` module accepts only the existing
`tensormeld/llamacpp-backend-self-test-v1` output with:

- `execution_source=native-subprocess`;
- pinned llama.cpp revision;
- exact test/probe/binding/config identities;
- a ready target device on the expected configured backend;
- positive supported/passed result counts;
- no model load, listener, reservation, broad qualification or executable flag.

It emits a deterministic, fingerprinted
`tensormeld/backend-readiness-evidence-v1` record.

Portable fixtures marked `injected-runner` are explicitly rejected from retention.

## Applicability behavior

A retained record can be checked against a fresh observed binding only when the exact
config, probe artifact, binding fingerprint, test artifact, TensorMeld device,
engine-local device and backend identities still match.

A successful applicability check does **not** promote the device to `ready`. It returns
`requires_live_runtime_recheck=true`, `runtime_ready=false`,
`reservation_created=false`, `qualified=false` and `executable=false`.

This is intentional because worker/driver/topology invalidation identity is not yet
implemented.

## Portable validation

A focused local isolated harness exercised the new pure retention/applicability logic:
5 tests passed for native-record creation, injected-fixture rejection, tamper rejection,
stale binding rejection and bounded JSON loading.

Repository CI is recorded separately when available. The isolated harness did not
execute a GPU backend.

## Hardware status

No real CUDA/HIP backend, Windows GPU, Strix Halo, model inference, performance test or
distributed inference was executed for this validation record.
