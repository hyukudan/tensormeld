# Changelog

## Unreleased

- Replica router: opt-in `prefill_offload` sends a Qwen prompt to an RTX prefill service for the replica the router holds (tokenize on the replica, threshold and queue priced from a `tensormeld/ttft-curve-v1` receipt, state imported by the replica, original request forwarded unchanged, fallback to the replica's own prefill on any failure), optional lossless handoff codec and cooperative local rows, per-request records in `/api/status` and counters in `/metrics`. `examples/hetero/qwen38-ttft-receipt.json` carries the curves measured on 2026-10-24.
- Add `plan-hetero --mode dense+hot --placement FILE`: a versioned hot-expert placement file (`tensormeld/hot-expert-placement-v1`) puts dense parts and the hottest routed experts on one parts-host, further experts on a peer GPU or host cache and every expert on remote expert servers; the candidate charges memory from the file, prices remote layers once on top of the local walk and reports the file's expected numbers per row pattern. The model-parts descriptor gains an optional `moe` geometry.
- Add `deploy-pipeline`: a two-stage layer pipeline plan becomes per-process manifests (`tensormeld/deployment-manifest-v1`) bound to approved binaries, engine instances and group modes, launched locally or over the operator's SSH, watched by `READY`/`/health`, routed through the replica router, and switched to the fallback with a state export/import on retirement of the FRONT device (ADR-0056). `plan-hetero` candidates gain `sets` and a pipeline `layout`; the router gains administrative drain, hold and admin routes.
- Add heterogeneous part placement: `tensormeld/model-parts-v1`, `tensormeld/hetero-hardware-v1` and `tensormeld/engine-describe-v1` contracts, `plan-hetero` (ranked whole/replica/pipeline/part-split/prefill-decode candidates with cost breakdowns, `prefill-only`/`preemptible` availability and fallback) and a bounded `link-probe` (ADR-0055).
- Add `serve-replicas`: a bounded least-loaded HTTP router over whole-model engine replicas with health-gated rotation, streaming relay and status/metrics endpoints (ADR-0054).
- Replica router: optional prefix/session affinity that yields to load, per-request prefill placement (nodes | rtx | auto with a token crossover, prefix-cache hits kept local, fallback on busy or failing services) over a generic tokenize/import/prefill-service contract, and `bench-replicas`, a concurrent multi-turn chat load client (ADR-0057).
- Add exact adapter representability, model identity and qualification-evidence contracts.
- Add advisory runtime availability/budget intersection without treating free memory as
  a reservation.
- Add a trusted-local probe for the pinned llama.cpp revision with artifact/source
  identity and bounded `--version` / `--list-devices` execution.
- Add explicit approved one-to-one engine-device ↔ TensorMeld-device binding.
- Never infer backend identity from engine labels.
- Allow at most one explicit engine memory reporter per physical pool.
- Convert bound devices to runtime state `observed`, not `ready`; backend self-test
  remains required.
- Keep all pre-execution stages separate from live session admission.
- Persist placement calibrations as immutable fingerprint records with bounded age and revalidation against current build/runtime/device/topology identities before measured-preference reuse.
- Add predictive physical-pool memory classes that distinguish hard-resident/state/workspace/staging from reclaimable file-backed pressure while reconciling exactly to existing runtime admission peaks.
- Add bounded advisory legal-unit placement search that honors explicit cut boundaries and physical-pool resident capacity without enabling finer-grained native execution.
- Add exact per-legal-unit compute/state/workspace/staging and boundary-payload evidence bound to current legal/model/adapter/runtime identities.
- Add exact directional transfer-path evidence with conservative payload upper-bound buckets, runtime identity binding and no interpolation, extrapolation or automatic multirail aggregation.
- Add bounded performance-aware legal-unit advisory ranking that combines explicit compute/memory evidence and conservative directional transfer costs as an ordered-chain upper bound, without claiming token latency or execution.
- Add scenario-specific generation-phase evidence for prefill, one decode step, sampling, logits movement and token feedback, bound to exact context and runtime identities.
- Add backward-compatible generation-phase v2 with distinct prefill/decode inter-unit boundary payloads; v1 remains valid but cannot drive phase-transfer estimates.
- Add complete explicit per-tensor storage/movability evidence bound to exact GGUF index, adapter/build and runtime identities; no name/size/backend heuristic can mark a tensor movable.
- Derive overlapping per-device and de-duplicated physical-pool movability byte envelopes without treating legal eligibility as current ownership or measured resident memory.
- Add adapter-declared ordered indivisible legal model units and explicit cut boundaries; complete tensor coverage and alias-group indivisibility are mandatory and unit device sets may only narrow tensor-level legality.

## 0.2.0a2 — 2026-10-01

- Adopt TensorMeld and preserve the existing Git history.
- Add explicit schema migration and prevent CLI report output from overwriting inputs.
- Tighten selection, manual ownership, CPU opt-in, count limits and pool headroom.
- Add bounded synthetic multi-pool whole-block planner with feedback and coordinator costs.
- Integrate optional psutil; add upstream gguf catalog adapter and separate integration gate.
- Record third-party versions/licenses, native engine candidates, ADRs and test status.
- Publish the repository early for engineering transparency while explicitly retaining pre-alpha status.
- Do not claim GPU inference or native Windows validation.
