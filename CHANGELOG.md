# Changelog

## Unreleased

- Add a versioned native-adapter capability envelope and exact whole-block
  representability gate.
- Add exact GGUF-backed model identity and E0-E5 qualification-evidence applicability.
- Add advisory runtime observations bound to config identity.
- Intersect observed available physical-pool memory with owner caps/headroom without
  treating free memory as a reservation.
- Filter runtime compute candidates by observed ready state and report unmet required
  nodes/devices/minimums structurally.
- Keep planner, representability, evidence and runtime-observation stages separate from
  live executable-session admission.

## 0.2.0a2 — 2026-10-01

- Adopt TensorMeld and preserve the existing Git history.
- Add explicit schema migration and prevent CLI report output from overwriting inputs.
- Tighten selection, manual ownership, CPU opt-in, count limits and pool headroom.
- Add bounded synthetic multi-pool whole-block planner with feedback and coordinator costs.
- Integrate optional psutil; add upstream gguf catalog adapter and separate integration gate.
- Record third-party versions/licenses, native engine candidates, ADRs and test status.
- Publish the repository early for engineering transparency while explicitly retaining pre-alpha status.
- Do not claim GPU inference or native Windows validation.


## 0.2.0a1 — 2026-10-01

- establish hardware/model-neutral repository architecture;
- add v2 installation configuration contract;
- support explicit nodes, multiple devices per node and physical memory pools;
- add owner resource policies and compute/coordinator selection policy;
- add local-only, companion-only, distributed and automatic execution profiles;
- add policy-only candidate resolution CLI;
- add accepted configuration examples for low-VRAM, companion-only and multi-node cases;
- retain the v1 analytical planner as a bounded regression/reference tool;
- add architecture documents and six accepted ADRs;
- expand tests to 78, including simulated 1/2/3/4/8/16-node contracts.

No distributed GPU inference, secure remote agent or model backend adapter is implemented
in this version.
