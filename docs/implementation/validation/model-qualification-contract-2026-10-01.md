# Model identity and qualification-evidence validation — 2026-10-01

Implementation slice: exact GGUF-backed model identity plus qualification-evidence
identity/applicability.

## Scope

This slice introduces contracts only. It does not hash model shards itself, execute a
model, generate hardware evidence, reserve memory, or launch a worker.

The model manifest requires complete GGUF shard coverage and caller-supplied full
SHA-256 for every shard. Tensor payload bytes are retained as file/index facts and are
not interpreted as runtime RAM/VRAM requirements.

Qualification evidence binds adapter capabilities, engine revision, worker artifact,
model manifest, config, device identities, workload, E0-E5 level, timestamp and named
tests. Applicability is exact and invalidates on identity changes.

## Local validation

Available Linux/Python 3.13 environment:

- 168 tests discovered;
- 167 passed;
- 1 optional upstream-GGUF integration test skipped locally;
- 0 failures/errors.

## GitHub Actions

Workflow run 36866547619 for commit
`f2e54b06c41ec6bbb76b5088f817965f61ebbb24` completed successfully.

All six jobs passed:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows.

Hosted CI validates portable software contracts only. It is not CUDA, ROCm, Strix Halo,
network, large-model, or distributed-inference qualification.

## Result

Exact model identity and evidence applicability are accepted as M2-preparatory contracts.

An applicable evidence record may report `qualified=true` only for its exact recorded
scope. It remains `executable=false`: live admission, secure preparation and runtime
resource/state checks are still required.
