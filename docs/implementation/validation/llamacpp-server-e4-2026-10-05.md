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

GitHub Actions result: pending for the implementation PR.
