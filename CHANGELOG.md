# Changelog

## Unreleased

- Add a versioned native-adapter capability envelope.
- Add an exact whole-block representability gate between planner output and future live
  adapter qualification.
- Reject unsupported device ownership, explicit ranges, route modes, backends,
  coordinators and adapter limits with structured reason codes.
- Keep representable plans explicitly non-qualified and non-executable.
- Add an exact GGUF-backed model identity manifest requiring complete shard coverage and
  full SHA-256 identities rather than treating tensor payload size as runtime memory.
- Add qualification-evidence records bound to adapter/build/model/config/devices/workload
  with explicit E0-E5 applicability and invalidation checks.
- Keep applicable evidence non-executable until live admission/session preparation exists.

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
