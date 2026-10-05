# First E4-authorized persistent completion validation — 2026-10-05

Implementation slice: execute exactly one E4-qualified completion contract through a
persistent admitted server.

## Runtime gates

Each call verifies:
- authorized binding fingerprint and hash fields;
- E4 spec fingerprint;
- exact admitted-server/package/bundle/server-spec identities;
- ready/live child process;
- active launched lease with matching lease/runtime-manifest fingerprints.

The same runtime checks run after the HTTP response.

## Output invariant

The deterministic completion content SHA-256 must equal the E4 expected output SHA stored
in the authorized binding. Any drift is rejected.

## Portable coverage

A real fixture HTTP process runs under a real LocalAdmissionController lease. Tests cover:
- successful completion while keeping the lease active;
- release only after server stop;
- output drift rejection;
- stale/released lease rejection;
- stopped-server rejection;
- tampered authorized binding;
- mismatched E4 spec.

Fixture execution remains real_model_inference=false. No real llama.cpp binary, GGUF
inference or GPU executes in hosted CI.

GitHub Actions result: pending for the implementation PR.
