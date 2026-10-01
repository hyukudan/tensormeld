# 06 — Implementation roadmap and exit gates

Specification 0.2.0. M0.2 implementation has started: the v2 configuration parser and
policy-only candidate resolver are executable. Priorities are P0 (first useful release),
P1 (product usability), P2 (advanced).
Milestones are dependencies, not dates or promises of unattended work.

## M0 — Historical analytical prototype (runtime 0.1.0)

Keep the bounded v1 scenario parser, three-device serial planner, inventory and
loopback test as a regression baseline. It is not the production node/configuration
schema. Its exact contract remains in spec 07. Its limits must not be renamed away.

## M0.2 — Generalized contracts BEFORE LAN implementation (P0)

Implement v2 registry/config/observations/intent separation with explicit nodes,
devices, multi-pool demands, host-local resource policy and node/process coordinators.
Add variable-length membership, automatic/manual subset constraints and schema
version rejection. Freeze v1 fixtures; conversion must be explicit and preserve
synthetic provenance. Introduce bounded heuristic planning with a small exact oracle.

Exit: software-only 1/2/3/4/8/16-node fixtures, multiple devices in one node, control
host without a GPU, insufficient indivisible units, no hidden quality/context
changes, search timeout versus proven infeasibility, stable IDs and configuration
precedence. Limits and evidence scope are emitted. Nothing claims GPU execution.

## M1 — Authenticated agents, local leases and real links (P0)

Manual enrollment, revocation, bounded encrypted messages, version negotiation,
read-only capability exchange, node-local policy and atomic resource leases. Add
bounded authenticated host-path probes and per-interface directed observations.
GPU-path probes follow a native worker. No driver/network changes are automatic.

Exit: native Windows/Linux secure enrollment and rejection tests; authoritative
local budgets; expired coordinator fencing; online versus eligible state; explicit
routes and shared-link measurements. A larger registry does not trigger unbounded
probe traffic. No insecure legacy RPC is published on a LAN.

## M2 — Pinned adapter, model truth and single-node modes (P0)

Prove a pinned native engine can express required partitions before building a
broad engine abstraction. Validate independent native CUDA/HIP builds. Inventory
real models and all state/load peaks without running model-supplied code. Establish
local-only and companion-only modes; CPU fallback is opt-in and traced.

Exit: tiny dense and MoE fixtures execute correctly where supported, a useful model
has real inventory, and plan lowering is exact or explicitly rejected. Multiple
backends visible on a GPU are not double-counted. A successful build is not a
mixed-vendor correctness pass. Publish adapter scale/coordinator restrictions.

## M3 — First useful distributed release (P0)

Execute one model across two hosts, then three where available, retaining stable
client endpoint and exact workload semantics. Add node-local verified caches,
streaming text API, bounded admission/queue, request cancellation, drain and clean
failure. The reference placement is complete legal blocks, not expert migrations.

Exit: real native Windows/Linux evidence; low-VRAM dense use case and large sparse/
MoE capacity case separately qualified; same-workload local/CPU/companion/distributed
comparisons; preparation peaks, output-state correctness, load/unload, sleep/wake
and network failure checks. Hardware not tested stays explicitly unqualified.

## M4 — Product usability and bounded concurrency (P1)

Add a simple dashboard, named per-model resource profiles, explainable comparison,
node status, installer/service UX, checkpoint cache quotas, idle unload, warm/cold
metrics, bounded concurrent requests and fairness, and compatible client fixtures.
Add optional discovery after manual pairing. Keep advanced controls visible without
requiring manual layer mathematics for ordinary use.

Exit: desktop resources can be reclaimed predictably, cancellation releases bounded
buffers, multiple sessions cannot over-admit a shared pool, and diagnostics export
without prompts/secrets. Runtime/API behavior, not screenshots, establishes support.

## M5 — Advanced execution and integration (P2)

Compare expert-bank placement, within-layer expert parallelism, tensor parallelism,
prefill/decode separation, speculative decoding/MTP and qualified precision variants
against the coarse reference. Extend task types (embeddings, reranking, multimodal)
and adapters separately. Multi-model routing/replicas remain distinct from splitting.

Exit: each optional capability preserves declared numerical semantics and improves
or enables the target workload with raw evidence. No unsupported feature is silently
ignored. Energy optimization requires sufficient measured power coverage.

## M6 — Optional transport acceleration (P2)

Qualify one advanced path, then multi-link distribution/striping and optional direct
buffers. Preserve authenticated portable fallback, explicit ownership and physical
bottleneck accounting. Replanning after link failure is a new plan unless live
transport continuity has been independently qualified.

Exit: end-to-end model benefit, bounded memory and correct recovery are demonstrated.
A working bandwidth benchmark or a second cable alone is not completion.

## Immediate implementation sequence AFTER spec agreement

1. Implement M0.2 schemas/policy invariants and the acceptance fixtures, not just a
   larger array bound in v1.
2. Establish the adapter's real placement/scale/route restrictions with an isolated
   local spike; feed its capability contract into the planner.
3. Implement M1 enrollment/local leases and bounded measurements against those contracts.
4. Validate M2 local/companion-only paths, then the smallest genuine M3 split.

Do not advance expert kernels, a large GUI or additional transports at the expense
of these gates. The current implementation has started M0.2 but does not yet start an agent or worker.
