# llama.cpp build/package identity validation — 2026-10-05

Implementation slice: exact sibling-artifact build identity plus admitted server lease
bridge.

## Package contract

Portable tests validate:
- distinct llama-cli and llama-server artifact hashes;
- identical build/version/commit/compiler/target metadata;
- deterministic backend-library identity set;
- rejection of build mismatch and duplicate backend-library artifact identities;
- package fingerprint integrity on reuse.

## Admitted server bridge

A real fixture server is attached to a real launch-admitted LocalAdmissionController
lease only when:
- package llama-cli matches the bundle worker artifact;
- package llama-server matches the server launch spec;
- model/bundle/revision/spec fingerprints match;
- the lease remains active/launched and matches the runtime manifest.

The lease remains active during server readiness and is released only after confirmed
process stop. Early process failure also performs process normalization before lease
release.

The binding explicitly keeps server semantic equivalence and inference-request
authorization false.

No real llama.cpp binary, GGUF inference or GPU executes in this validation.

GitHub Actions result: pending for the implementation PR.
