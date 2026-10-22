# 12 — Specification review, decisions and migration

Review date: 2026-10-01. Input: supplied specification/archive v0.1.0.
Output: specification 0.2.0. The original review froze runtime 0.1.0; implementation
subsequently began as 0.2.0a1 with a separate v2 configuration parser and policy resolver.

## Findings and resolution

| Finding in v0.1.0 | Why it matters | Revised contract |
|---|---|---|
| Product defined by a large MoE workload and one/two companions | Excludes the user's 12 GB/dense/general-node intent | Spec 01: model/hardware-agnostic scope and scenarios |
| AGENTS requires a large GLM/DeepSeek target except tiny correctness tests | Future contributors would reject legitimate smaller use cases | Revised AGENTS and co-equal dense/MoE acceptance |
| Parser hard-caps devices/coordinators at 3, pools at 8, links at 32, stages at 128 | A fourth accelerator already fails even on fewer hosts | Preserve v1; new independent v2 inventory/limits |
| No explicit Node entity; coordinator is a Device ID | CPU-only orchestration and multiple GPUs/host are awkward | Node/device/worker/process-role separation |
| Exhaustive subsets/permutations/cuts | Removing the numeric limit does not scale the algorithm | Bounded production search + small exact oracle |
| One pool reference per device | GPU execution also consumes host/staging memory | Per-worker/per-placement resource demand vectors |
| Model accounting is user-supplied stages | Cannot establish real low-VRAM/long-context feasibility | Exact manifest, legal units and load/state peaks |
| M0 considers decode only, concurrency one | Cannot predict first-token delay or multi-client throughput | Separate phases, queue and optional bounded concurrency |
| Fixed baseline list culminates in all three | Implies a fixed installation and local GPU participation | Generic subsets, local/companion-only modes |
| Limits and policy absent for required/excluded nodes | Count alone does not specify usable placements | Auto/manual sets with exact count/filter semantics |
| Config versus telemetry/profile provenance insufficiently separated | User input could be mistaken for verified hardware | Desired/observed/profile/intent/plan separation |
| Resource headroom lacks enforceable local ownership/lifecycle | A desktop owner needs to reclaim its GPU safely | Atomic local leases, drain, resource guard and live rechecks |
| General safety text but few concrete useful-feature gates | Easy to build a dashboard without a usable system | 25 prioritized feature groups and 62 future tests |
| Roadmap goes directly from M0 to paired agents | Would encode current cardinality/identity defects in protocol | New M0.2 contract/planner milestone first |

The prior spec already handled physical-pool aliasing, actual RPC routes, cyclic
feedback, numerical correctness, secure transport and evidence separation well.
Those contracts are retained and extended, not replaced with automatic-pooling claims.

## New decisions before coding

ADR-011: general local-first inference; no model-family, capacity or companion-count gate.
ADR-012: distinct node/device/worker/pool/role identities and node/process coordinator.
ADR-013: variable registered scale and bounded search; limits/evidence published separately.
ADR-014: workload semantics and local owner constraints cannot be silently relaxed.
ADR-015: local-only, companion-only and distributed are distinct first-class modes.
ADR-016: topology changes affect new plans; live migration/failover is not implicit.
ADR-017: resource guard, bounded queue and cache lifecycle precede advanced kernels.
ADR-018 (historical): the review phase changed docs only; v1 fixtures remain an analytical oracle.

## Migration contract

The review phase did not change executable code. That phase is complete. Runtime
0.2.0a1 now contains a separate `tensormeld/v2` parser and policy-only candidate
resolver while retaining the legacy v1 analytical planner for regression comparisons.

Future v1 conversion tooling must still be explicit. It
must resolve nodes/pools/roles explicitly, preserve bounded validation and refuse
ambiguous upgrades. A v1 measured label stays an untrusted input declaration; conversion
cannot create a certificate, runtime capability or performance qualification.
V1 regression fixtures remain available to compare small-case planner behavior.

Do not change 3 to a larger number and declare configurable nodes complete. First
pass membership, role, local resource, quality/precision and search-budget invariants.
A model importer cannot assume a v1 'stage' is a universally valid backend cut.

## Outstanding empirical choices (not specification blockers)

Which pinned engine/build can lower exact blocks across native Windows CUDA and
Linux HIP; the actual safe budgets and GPU-ready link costs; the effective small-
cluster planning budget; the tested scale per adapter; and energy metrics remain
measurement tasks. These unknowns do not justify hard-coding today's hardware.

No particular model size/speed, third-party benchmark or dual-link gain is promised.
The first target fixtures can be chosen by exact revision after backend qualification.
Remote repository creation is an operational concern separate from this specification.
