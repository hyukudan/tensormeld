# ADR-0015 — Native engine device identities require explicit binding

Status: **accepted**  
Date: 2026-10-01

## Context

A native engine exposes its own device identifiers, such as `CUDA0` or `ROCm0`.
Those names are runtime-local labels. They are not stable TensorMeld identities and must
not be treated as evidence of backend family, ownership node, or physical-memory aliasing.

Automatic name-based binding could attach observations to the wrong device after driver,
device-order, topology or engine changes.

## Decision

TensorMeld requires an explicit approved binding document between:

- one exact installation config fingerprint;
- one exact probed native artifact SHA-256;
- one node identity;
- one or more one-to-one engine-device ↔ TensorMeld-device mappings.

The binding must use `approval = "explicit"`.

Backend identity comes from the approved TensorMeld configuration, not from parsing the
engine device name or description.

A mapping may optionally nominate a device as the sole memory reporter for its configured
physical pool. At most one engine device may report availability for a physical pool.
No automatic summation is allowed for multiple devices that alias the same pool.

A successful binding converts native observations into
`tensormeld/runtime-observation-v1` with device state:

```text
observed
```

not `ready`.

The runtime selector accepts `observed` as a valid snapshot state but excludes it from
runtime-ready candidates.

## Consequences

- Engine device ordering cannot silently rewrite TensorMeld identity.
- CUDA/HIP/backend identity is never inferred from labels like CUDA0/ROCm0.
- Unified/shared memory is protected from double counting by explicit single-reporter
  semantics.
- Unmapped engine devices are reported but never auto-bound.
- A backend self-test or stronger qualification step is required before an observed
  device can become runtime-ready.
