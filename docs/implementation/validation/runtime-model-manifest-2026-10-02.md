# Runtime model/operator/memory manifest validation — 2026-10-02

Implementation slice: exact runtime manifests as the bridge between model identity and
future live resource admission.

## Contract

`tensormeld/runtime-model-manifest-v1` binds an imported manifest to the exact config,
model manifest, profile workload, adapter capability fingerprint, engine revision and
worker artifact.

Worker devices carry explicit operator declarations. Required operators are checked
against every included device, but complete declaration coverage does not self-qualify
the workload.

Memory is represented exactly once for each physical pool. Each record contains resident,
state, workspace peak and preparation peak bytes. Duplicate physical pools are rejected;
preparation peak must cover the steady peak and respect known physical capacity.

## Portable coverage

The new portable tests cover exact tuple acceptance, duplicate physical-pool rejection,
identity drift, incomplete operator coverage, peak/capacity invariants, fixture
provenance and duplicate-key rejection.

A CLI command, `validate-runtime-manifest`, loads the existing config/model/adapter
contracts and returns the non-executable validation summary.

## Evidence boundary

The test records use fixture provenance. They are software contract tests, not native
adapter measurements. A future `native-adapter` record still requires live admission
and E3 model correctness evidence before execution.

No CUDA/HIP backend, Windows GPU, Strix Halo, model inference, performance test or
distributed inference was executed for this validation record.

GitHub Actions result: PR #3, workflow `Portable Python tests`, run #68 (36935749574) passed all 6 jobs: Linux and Windows on Python 3.11/3.13 plus optional-adapter integration jobs on both operating systems. The first run (#66) exposed two fixture-test defects; both were corrected before the successful run. Hosted CI remains portable software evidence, not GPU qualification.
