# Changelog

## Unreleased

- Add exact adapter representability, model identity and qualification-evidence contracts.
- Add advisory runtime availability/budget intersection without treating free memory as
  a reservation.
- Add a trusted-local probe for the pinned llama.cpp revision:
  - artifact SHA-256 verification;
  - bounded `--version` and `--list-devices` subprocess calls;
  - source-revision verification;
  - conservative engine-local device/memory parsing;
  - no model loading, listener or automatic TensorMeld identity mapping.
- Keep planner, representability, evidence, runtime observation and native probe stages
  separate from live executable-session admission.

## 0.2.0a2 — 2026-10-01

- Adopt TensorMeld and preserve the existing Git history.
- Add explicit schema migration and prevent CLI report output from overwriting inputs.
- Tighten selection, manual ownership, CPU opt-in, count limits and pool headroom.
- Add bounded synthetic multi-pool whole-block planner with feedback and coordinator costs.
- Integrate optional psutil; add upstream gguf catalog adapter and separate integration gate.
- Record third-party versions/licenses, native engine candidates, ADRs and test status.
- Publish the repository early for engineering transparency while explicitly retaining pre-alpha status.
- Do not claim GPU inference or native Windows validation.
