# llama-cli ↔ llama-server E4 validation — 2026-10-05

Implementation slice: exact deterministic CLI/server equivalence plus admitted request gate.

## Portable end-to-end comparison

CI launches:
- a real fixture CLI subprocess;
- a real managed HTTP server fixture on loopback.

Both consume the same GGUF path, model identity, placement override, prompt and
deterministic generation settings. The server fixture exposes the pinned llama-server
/completion-shaped contract.

The evaluator compares CLI stdout SHA-256 with server completion-content SHA-256.

Because both are fixture processes, equal output produces fixture equivalence evidence,
not E4 qualification.

## Native authorization contract

Separate portable contract tests construct native-shaped evidence to validate the
production gate. Authorization requires:
- intact E4 spec and evidence fingerprints;
- native CLI/server provenance;
- exact equal output hashes;
- exact package/model/AcceptedExecutionBundle/pre-post-placement/trial/server-spec
  identities;
- exact admitted server package/bundle/server-spec identities.

The resulting authorized binding can report inference_request_authorized=true while still
reporting real_model_inference=false because no production request has yet executed.

No real llama.cpp binary, GGUF inference or GPU is executed in hosted CI.

GitHub Actions result: PR #24, workflow `Portable Python tests`, run #243 (37356485327) passed all 7 jobs after one earlier run exposed an E4 spec canonicalization mismatch between numeric `0` and `0.0` for temperature. The contract now canonicalizes temperature as float zero consistently. Linux and Windows Python 3.11/3.13, optional-adapter integration on both operating systems, and the dedicated real-mTLS-loopback job all passed. Hosted CI validated real fixture CLI/server equivalence only; it did not establish native llama.cpp E4 or GPU inference.
