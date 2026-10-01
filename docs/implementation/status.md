# Current implementation status

Version: **0.2.0a2** — TensorMeld. Specification baseline: 0.2.0.

The canonical repository is public for engineering transparency. This does not change
the pre-alpha status or qualify any execution path.

## Implemented

- renamed package, CLI, maintained documentation, examples and publication default;
- explicit non-overwriting migration from the previous v2 working-title schema;
- strict v2 installation configuration and policy-only resource selection;
- optional allowlist semantics, exact manual compute set, required-owner constraints;
- distributed minimum device count, CPU opt-in, any-local-GPU requirement;
- conservative intersection of static owner cap and reported capacity minus headroom;
- bounded synthetic v2 whole-block placement, 1..N configured devices;
- multi-pool resident/workspace accounting, independent coordinator overhead;
- explicit directional routes, token feedback, budgeted work and top-k output;
- optional psutil inventory reuse, with read-only fallbacks;
- optional GGUF catalog adapter using the upstream package;
- dependency/license register, revision-pinned native engine candidates;
- legacy analytical planner and bounded loopback diagnostics retained.

## Tested here

Local Linux/Python 3.13 unit/contract tests, exact small-case planner oracle, budgeted
simulated 1/2/3/4/8/16-node scenarios, real psutil inventory, archive/install smoke tests
as recorded in the release validation. GGUF adapter tests use injected readers.

## Not tested / not implemented

- real upstream gguf integration: package unavailable here; one explicit test skip;
- native Windows execution; GitHub Actions results are reported separately and do not constitute GPU qualification;
- remote enrollment/agent, secure transport and active memory reservations;
- verified model-specific execution manifests and hardware/backend qualification;
- native worker compilation or CUDA/HIP/distributed inference;
- balanced/throughput v2 scoring, expert/tensor/phase placement, multirail, GUI/API.

The v2 planner is advisory and synthetic. It cannot authorize a worker or upgrade itself
to qualified. A tensor file index is not runtime memory admission. No measured RTX,
Strix or model speed is claimed. M0.2 is advanced, not a claim that M1/M2/M3 are done.
