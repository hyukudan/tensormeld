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

GitHub Actions result: PR #25, workflow `Portable Python tests`, run #253 (37358461514) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. Portable CI executed a real fixture HTTP completion under a real LocalAdmissionController lease and validated lease persistence, stale-state rejection and E4 output-drift checking; it did not execute llama.cpp or GPU inference.
