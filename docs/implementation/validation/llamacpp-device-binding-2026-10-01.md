# Explicit llama.cpp device-binding validation — 2026-10-01

Implementation slice: approved engine-device to TensorMeld-device binding.

## Scope

The binding is tied to exact config and native artifact identities and maps one engine
device name to one TensorMeld device ID. Backend identity is taken from the approved
TensorMeld config, never inferred from engine labels.

An optional memory reporter can populate availability for the configured physical pool;
at most one reporter is permitted per pool.

Bound devices become runtime state `observed`, not `ready`.

## GitHub Actions

Workflow run 36879001563 for commit
`dcc960c5ddaaa69bada78560dd5d318a7bcc0404` completed successfully.

The run covers portable Windows/Linux software tests. No real CUDA/HIP engine binary or
GPU was used.

## Result

The explicit binding and CLI flow are accepted as an identity bridge between the native
probe and TensorMeld runtime-observation format. Backend self-test/qualification is still
required before runtime readiness.
