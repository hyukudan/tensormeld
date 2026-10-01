# ADR-0014 — Native engine probing is explicit, bounded and no-model

Status: **accepted**  
Date: 2026-10-01

## Context

TensorMeld needs evidence about an actual native inference-engine build before it can
map engine-local devices, qualify operators, or prepare execution. Reimplementing
backend discovery would duplicate functionality already exposed by llama.cpp.

At the same time, invoking a native executable is a stronger trust boundary than
parsing TensorMeld JSON. A discovery step must not accidentally load a model, start a
listener, accept remote input, or turn transient free-memory output into an allocation.

## Decision

The first native-engine probe targets the pinned llama.cpp source revision recorded in
the third-party registry.

For an explicitly trusted local binary, TensorMeld:

1. resolves a regular local file;
2. computes its full SHA-256 and optionally requires an expected artifact hash;
3. runs only `--version` and `--list-devices`;
4. uses `shell=False`, no stdin, a bounded timeout and bounded captured output;
5. verifies that the reported source commit is the pinned revision or an unambiguous
   prefix of it;
6. parses only the documented device-list shape for engine-local names, descriptions,
   total memory and free memory;
7. rejects unexpected output instead of guessing.

The probe does not infer CUDA/HIP from a device name and does not map engine-local device
names to TensorMeld device identities automatically.

The result always records:

```text
model_loaded = false
listener_started = false
device_identity_mapping = "unresolved"
qualified = false
executable = false
```

## Consequences

- TensorMeld reuses upstream device discovery instead of duplicating GPU-backend
  enumeration logic.
- Binary/source identity is explicit and can be tied to later qualification evidence.
- Free memory remains a transient observation, not a reservation.
- A successful probe is E1-style build/device observation at most; it is not model,
  operator, backend, network or performance qualification.
- Future adapters may implement different probe contracts, but they must preserve the
  same bounded/trust principles and may not silently execute arbitrary commands.
