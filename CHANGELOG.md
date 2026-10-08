# Changelog

## Unreleased

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
- Add complete explicit per-tensor storage/movability evidence bound to exact GGUF index, adapter/build and runtime identities; no name/size/backend heuristic can mark a tensor movable.
- Derive overlapping per-device and de-duplicated physical-pool movability byte envelopes without treating legal eligibility as current ownership or measured resident memory.

## 0.2.0a2 — 2026-10-01

- Adopt TensorMeld and preserve the existing Git history.
- Add explicit schema migration and prevent CLI report output from overwriting inputs.
- Tighten selection, manual ownership, CPU opt-in, count limits and pool headroom.
- Add bounded synthetic multi-pool whole-block planner with feedback and coordinator costs.
- Integrate optional psutil; add upstream gguf catalog adapter and separate integration gate.
- Record third-party versions/licenses, native engine candidates, ADRs and test status.
- Publish the repository early for engineering transparency while explicitly retaining pre-alpha status.
- Do not claim GPU inference or native Windows validation.
