# 05 — Platforms, compatibility and validation

## Support is multidimensional

Support means a qualified tuple of host OS, backend artifact, driver/runtime,
device architecture, model revision/encoding, execution mode and transport.
Neither an operating-system logo nor detected executable proves this tuple works.

| Component/profile | Initial target | Evidence required |
|---|---|---|
| CLI, schema, planner | Native Windows and Linux | Unit/CLI tests on both |
| Desktop GPU worker | Native Windows + CUDA | Build, allocation, operators, same-model execution |
| Desktop GPU worker | Native Linux + CUDA | Same gates, independently recorded |
| First APU worker | Native Linux + HIP | Usable pool, kernels, state, sustained workloads |
| Windows APU worker | Later qualification profile | Do not infer from Linux or SDK listing |
| Alternative GPU backend | Optional adapter | Explicit operator/encoding/performance gates |
| Advanced direct-link transport | Optional platform capability | Kernel/driver availability and exact path tests |

Windows is not required to run a Linux subsystem. Linux companions are not Windows
subsystem guests in the initial deployment. Optional virtualization is a separate
future profile and cannot substitute for validating native Windows operation.

Do not hard-code a single evergreen driver version. Store the qualified set, exact
binary hashes and reproducible build recipe. New SDK releases do not automatically
replace a working profile. Driver/toolchain installation is outside agent privileges.

## Layered evidence

E0: static configuration only. E1: hardware/runtime observed. E2: isolated allocation
and operator self-tests. E3: complete model correctness for a defined workload.
E4: repeatable performance and memory qualification. E5: sustained mixed-platform
operation and recovery testing. The UI must retain these levels separately.

M0 provides E0 planning and some E1 observations, plus host-side socket tests. It
must not claim E2–E5. A Windows CI definition is not a Windows CI pass.

## Correctness gates

Use tiny deterministic fixtures to validate transport and partition mechanics, then
representative real model fixtures. Compare reference and distributed runs using
the same checkpoint, tokenizer, chat template, quantization, context and sampler.
Greedy-token comparisons are useful but near-tied logits can differ across numeric
paths. Record tensor/logit tolerances, error distributions, state transitions and
quality checks; never call semantically changed execution an exact speedup.

Test prompt ingestion, token generation, long contexts, cache reuse, cancellation,
repeated load/unload and restart behavior. Architectures with recurrent state require
state-specific reset/rollback tests. Unsupported operators must fail explicitly.
Qualification includes tied embeddings, output/sampling placement and token feedback.

A future MTP path must validate draft acceptance, rejected-token rollback, committed
state and tokens actually emitted. It is not part of the initial reference.

## Memory gates

Measure preparation and steady-state peaks, context growth and workspace changes.
Exercise pool pressure and avoid unsafe overcommit. Validate CPU/iGPU aliasing and
multiple workers sharing a physical pool. A full model file plus repacked shard
may coexist during load even when only the shard is needed for decode.

User-reserved headroom is configurable and applies before admission. Benchmark
profiles must state whether display and background workloads were active. Passing
one short load does not establish a sustainable memory budget.

## Performance gates

Start with concurrency one and supported short/medium/long contexts: 2K/8K/32K are
candidate fixtures, not mandatory allocations on every 12 GB GPU. Use fixed output
lengths and sampling seeds. Add larger contexts only when state/memory rules permit.
Compare local GPU(s), explicitly allowed CPU offload, companion-only and eligible
multi-node subsets on the same workload. Reject workloads that cannot fit safely.

Record time to first token, prefill rate, per-token p50/p95, useful committed decode
rate, load time, CPU use, per-pool peaks, transfer bytes and waits per boundary,
GPU idle time, temperature and power observations. Do not mix aggregate multi-request
throughput with single-stream latency.

Warm/cold cache and MTP on/off are separate rows. Store raw samples, commands, source
revisions, worker hashes, topology, routes and trial order. Repeat runs and assess
variance before making a performance claim. A low-level benchmark is a diagnostic,
not the product's success criterion.

## Security/recovery gates

Reject unpaired peers, wrong certificates, revoked identities, malformed/oversized
frames, mismatched model/plan/epoch, out-of-order state messages and invalid local
paths. A cancelled or interrupted request must release resources safely. A restored
link must not silently restore lost GPU state. Backpressure prevents a fast sender
from unboundedly buffering data at a slower receiver.

## Build and CI policy

Portable unit tests run on Linux and Windows with a minimum and a recent Python
version. Native CUDA/HIP builds and real GPU tests are independent jobs on suitable
hardware. Ordinary hosted CI must not claim hardware validation. CI defaults to
read-only repository permissions and pinned action commits.

Automated runs must not spend scarce GPU time or download large checkpoints without
explicit configuration. Performance baselines are protected from synthetic fixtures.
A merge must update the implementation ledger when capability status changes.


## Low-VRAM, topology and practical desktop gates

There is no general performance extrapolation from a 96 GB accelerator to a 12 GB
consumer GPU. Memory caps on a large GPU validate constrained admission, not smaller
GPU throughput, kernel coverage, host-RAM pressure or display interference. Label
virtual/constrained budgets separately from real hardware observations.

The acceptance matrix includes a physically low-VRAM dense-model case, a large
sparse/MoE case, two local GPUs, a control-only PC, variable node counts, companion
only, a slow/contended link, shared host/iGPU memory and unavailable required nodes.
Use fixture identities rather than a particular current model brand as a test API.
A representative dense model and a representative MoE architecture are independently
qualified; a family name cannot stand in for operators or state behavior.

Run foreground desktop workloads during resource-preservation tests. Recheck runtime
budgets and allocation feasibility in the worker process; external telemetry may not
represent the worker's OS budget. Observe CPU threads, pinned memory, page-cache
pressure and storage requirements as well as VRAM. Unknown telemetry is not a zero.

Test node drain, disable, certificate revocation, network change, sleep/wake and
process restart. Active requests fail or complete under documented semantics;
new hardware is used only after a new admitted plan. Post-token failure cannot
silently retry and concatenate an unrelated continuation.

## CI versus hardware evidence

New schema/planner tests must run with 1/2/3/4/8/16-node synthetic fixtures before
claiming configurable scale. Four registered nodes are not four executing nodes.
Each release publishes maximum tested registry, planner and execution dimensions,
with backend/model/platform tuple and test type. Larger software-only tests cannot
satisfy physical multi-node qualification. CI cannot upgrade evidence labels.

The new matrix is a required future suite. It is NOT among the existing 54 M0 tests.
See spec 11 for test IDs and the review validation report for this delivery only.
