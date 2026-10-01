# 01 — Product specification

Specification: 0.2.0-draft. Runtime remains the unchanged 0.1.0 M0 prototype.

## Definition and intended users

TensorMeld is a local-first Windows/Linux inference manager. It helps a
user run a selected language model on an appropriate subset of enrolled computers
and their devices, when local capacity, responsiveness or availability is inadequate.
The user's normal PC remains a convenient entrypoint; its GPU need not participate.

A 12 GB GPU plus one companion is a first-class product case, not a reduced edition
of a 96 GB workstation product. The design MUST NOT require a particular VRAM size,
a specific model family, an MoE architecture, two companions, or a GPU in the control
host. Advertised capacity is not evidence that a particular workload fits.

Dense and sparse/MoE language models are co-equal workload classes. Architecture,
checkpoint revision, tensor encodings, tokenizer, state layout and backend coverage
determine compatibility. Small, medium and large checkpoints are all valid targets.
Multimodal components and adapters are extensibility requirements, not initial
execution promises. Strix Halo is the first companion qualification profile, not
a device type hard-coded into the generic scheduler or protocol.

The distinguishing requirement remains execution of ONE model across selected
resources when useful. Routing whole models to different hosts is also valuable,
but MUST NOT be represented as splitting a single model across their memories.
Remote resources never become transparent OS-level VRAM or coherent shared RAM.

## Product scenarios

| Scenario | Required behavior |
|---|---|
| Everyday PC with a 12 GB GPU and one APU companion | Compare local GPU, explicit local CPU offload, companion-only and distributed execution of the same workload |
| PC with a 24/32 GB GPU and multiple companions | Choose a useful subset, not every enrolled host |
| Workstation with a 96 GB GPU and two companions | Support very large checkpoints through the same model/placement contract |
| PC with two local GPUs and one companion | Distinguish two compute nodes from three accelerators and their separate/shared pools |
| PC with no eligible GPU | Retain control/API and use a companion alone; local CPU computation is an explicitly enabled option |
| Model weights fit locally but the requested context does not | Evaluate state-inclusive placement; do not silently reduce context or change KV precision |
| PC GPU is needed for another application | Exclude it from new placements or apply a lower resource policy; drain active work safely |
| Four, eight or more enrolled computers | Accept variable-length inventories and bounded planning; expose implementation/qualification limits explicitly |

These scenarios are acceptance targets, not tested hardware claims.

## Scale and launch envelope

The data model supports a variable number of nodes and multiple devices per node.
There are zero or more companions and at least one compute owner in a running plan.
A local-only setup, companion-only setup and multi-host setup share one contract.

The first physical qualification proceeds through one, two and three compute hosts.
This is an evidence envelope, NOT a product maximum or protocol field count.
Larger simulated inventories are required before implementing the LAN agent.
A build MUST declare registry, planner, adapter and qualified-scale limits separately.
It MUST NOT silently ignore nodes beyond a limit or claim unbounded scalability.

Native Windows and Linux are control/agent targets. Initial GPU-worker validation
covers native Windows/Linux CUDA and native Linux HIP independently. Windows HIP,
other vendors and other operating systems are additional qualification profiles.
The workstation MUST work without USB4. Ordinary IP networking is the first path.
Fast links between companions are optional and do not accelerate unrelated paths.

## Execution modes versus optimization objectives

Execution mode describes WHERE a request runs:

- Local-only: compute on devices of the entrypoint host.
- Companion-only: one or more companion hosts compute; the PC may only serve clients.
- Distributed: one model spans selected devices/hosts through a qualified adapter.
- Multi-model routing/replicas: independent executions; a later feature, never VRAM pooling.

Optimization objective describes WHY a plan is preferred: capacity feasibility,
interactive latency (first token and subsequent tokens reported separately), or
throughput at stated concurrency and latency constraints. Resource-saving policies
can limit participation or preserve desktop resources without claiming measured
energy savings. Energy optimization is unavailable until energy data is qualified.

Compare the same checkpoint, encoding and workload for speed claims. A changed model,
quantization or context is an alternative workload requiring approval, not an
optimization silently applied to the original request. No metric based on model
size alone may claim an improvement in output quality.

Only serial C1 decode estimates exist in the current M0 runtime. None of the newly
specified modes, configuration controls or objectives is thereby implemented.

## Core user journey

Install the controller and a companion agent; enroll explicitly; inspect resources;
choose resource-sharing policy; import an exact checkpoint; request context/output
limits; inspect compatibility and placement alternatives; approve; run; inspect
measured results; release resources. Manual addressing works before discovery.

Begin with CLI and a stable local API. A small web dashboard follows successful
execution, then optional native packaging. One endpoint remains stable when the
execution coordinator changes. No backend-specific internal UI API is a public
contract. Simple controls and advanced controls act on the same configuration.

The user sees selected and excluded nodes with reasons, per-pool memory, model and
state precision, cold-load versus warm-run behavior, and estimated versus observed
metrics. The product may recommend using no companion, no local GPU, or fewer nodes.

## Core invariants

PR-01: nodes, devices, workers, physical pools and roles are separate identities.
PR-02: no GPU capacity, node count or model-family gate defines product eligibility.
PR-03: declared user constraints and each host owner's local policy are enforced.
PR-04: load-time and steady-state memory are admitted across ALL affected pools.
PR-05: model graph order and numerical/state semantics are preserved by placement.
PR-06: actual adapter routes and capabilities override idealized topology drawings.
PR-07: estimates, measurements and qualification are distinct, with provenance.
PR-08: adding a node cannot force its use, weaken safety or change the workload.
PR-09: node/driver/model changes cannot silently modify an active committed session.
PR-10: the first useful release supports both a low-VRAM dense case and a high-capacity
       sparse/MoE case, with feasibility and performance evidence kept separate.

## Non-goals and scope control

The initial release does not implement transparent memory pooling, NPU support,
WAN/public federation, training/fine-tuning, image/video generation, live state
migration, elastic resharding of a running request, autonomous tool execution,
speculation/MTP, high availability or high-concurrency cluster infrastructure.
External APIs, optional models and desktop packaging must not displace a working
secure model-partition path. Advanced execution requires its own qualification.

## First usable release acceptance

Qualify controller/agent operation on native Windows/Linux, exact placement on a
pinned GPU backend, local-only and companion-only execution, and one distributed
model. Then qualify the second companion. A tiny fixture checks correctness; a
workload exceeding the selected local budget checks capacity usefulness.

Physical low-VRAM evidence is required before claiming support/performance for a
12 GB GPU. Imposing a 12 GB software budget on a larger GPU only tests admission.
A larger sparse/MoE workload is a separate useful-capacity gate. Failure to run a
particular checkpoint is reported by architecture/encoding/budget, not by brand.

A release checklist MUST distinguish specified, implemented, simulated, observed,
qualified, executing and failed. Read the implementation ledger, not this spec,
for the capabilities currently delivered.
