# 02 — Architecture and boundaries

## Role separation

**Desktop/front door:** owns the user's session, model catalog, pairing controls,
API endpoint, telemetry display and launch approval. It may run on Windows or Linux.

**Control service:** manages node identities, configuration, immutable plan records
and lifecycle. Control messages are small; the service is not implicitly a tensor
relay. M0 provides an offline CLI subset, not a listening control service.

**Execution coordinator:** owns the model execution graph and per-request ordering.
Its placement is a plan decision and may differ from the desktop. Coordinator
selection must reflect actual backend forwarding behavior, not a presumed mesh.

**Node agent:** reports capabilities, starts only approved worker artifacts, handles
local caches and resource limits, and reports health. It may not accept arbitrary
shell commands. A service manager integration is a packaging concern, not a
prerequisite for core correctness.

**Native worker:** performs model operators using one qualified local backend. CUDA
and HIP need not share a process or toolchain. A backend's buffer pointer never
crosses the network as an address that another backend can dereference.

**Transport:** sends bounded tensors or model shards with explicit ownership,
completion and failure semantics. OS-specific transports are optional plugins.

## Code boundaries

The initial control/planning layer uses Python 3.11+ and standard-library-only
runtime dependencies. This minimizes bootstrap friction and enables Linux/Windows
unit tests before hardware arrives. Python must not implement the steady-state
per-layer tensor path or bounce GPU activations through JSON.

Initial native inference workers should be adapters around a revision-pinned
existing C/C++ execution engine. A subprocess boundary avoids forcing all GPU SDKs
into the same executable. The adapter must either faithfully lower a plan to the
engine or reject it. It must not silently approximate a requested block placement
using memory-ratio flags. Custom native transport or worker code is introduced
only when measurement demonstrates the missing capability.

A future optimized APU worker is another adapter, not a dependency of the generic
manager. Changes to its kernels must not require CUDA. The manager must remain
able to use a slower reference worker as a correctness baseline.

The initial UI follows the M3 execution gate; richer UX is M4. A web frontend or
desktop shell MUST use the same local control API as the CLI. It cannot implement
its own admission rules or bypass node-owner policy.

## Proposed module layout

`inventory`: read-only OS, device, physical-pool and interface observations.

`profiles`: workload-bound allocation/compute/transfer observations with timestamps,
binary hashes, confidence and invalidation rules.

`model_manifest`: tensor and state inventory; architecture-specific semantics.

`planner`: feasible device subsets, coarse partitions, memory admission, route cost,
ranked candidates, rejected-candidate explanations.

`adapters`: describe, qualify, prepare, start, cancel, release, collect trace.

`agents`: authenticated discovery/pairing and bounded lifecycle control.

`transports`: platform-neutral operation and buffer ownership contracts.

`api`: local request admission, stream events, cancellation, stable client endpoint.

`bench`: correctness fixtures, reproducible workload execution and evidence bundles.

## Backend capability contract

A worker advertisement must include protocol version and engine commit/build ID,
OS/runtime/driver identifiers, device identities, physical-pool mappings, supported
model architecture versions, exact weight/activation encodings, operators, state
layouts, alignment constraints, maximum frame size, cancellation behavior and
available transfer mechanisms.

Capabilities are explicit states: unknown, observed, self-tested, workload-qualified,
unavailable. Advertised support alone is not qualification. A worker may expose
model execution without supporting arbitrary operator RPC. A native binary named
like a supported engine is not sufficient evidence of compatibility.

A proposed adapter interface is:

```text
describe() -> Capabilities
qualify(model_manifest, workload) -> QualificationReport
validate_plan(plan) -> AcceptedPlan | StructuredRejection
prepare(plan_id, shards, resource_budget) -> PreparedSession
start(session_id, request) -> EventStream
cancel(session_id, request_id) -> CancellationStatus
release(session_id) -> ReleaseStatus
collect_trace(session_id) -> TraceBundle
```

`validate_plan` must explain unsupported partition boundaries or quantization
crossings. Shards reference content hashes and allowed local paths, never arbitrary
commands or URLs executed by the peer. A software-only qualification is not a
hardware-specific qualification.

## Execution lifecycle

A planned session advances through CREATED, VALIDATED, RESOURCES_RESERVED, LOADING,
READY, RUNNING, DRAINING and CLOSED, or a terminal FAILED state. Model load, node
restart and repeated cancellation must be idempotent at the control boundary.
Resources are released on failed preparation. Partial preparation on one node
must not publish READY to the client.

A request has an immutable model/plan identity, generation epoch, context budget,
sampler contract and cancellation identity. A token is committed only after its
required state updates are accepted by the coordinator. Transport completion is
not model commitment. A connection loss creates a new epoch; old tensor messages
cannot advance the new request.

## Control placement versus tensor routing

The plan must bind each transfer to the route actually implemented by the adapter.
Direct peer-to-peer is not inferred from physical connectivity. A coordinator that
copies a remote tensor between two workers creates two hops, plus staging copies.
The profiler must capture those hops. Co-located device transfers still need a
qualified local copy path; they are not automatically free.

When two companions have a fast direct path and the desktop has a slow path, the
planner should compare a companion-hosted coordinator against desktop coordination.
Moving the coordinator must not move the user-facing endpoint. A hub topology can
be a deliberate backend restriction rather than a physical-network fact.

## Model delivery and caching

The desktop does not need to stream the checkpoint anew for every run. Each worker
may retain permitted, content-addressed shards. The coordinator's own model loader
may still require a metadata file, mapping or even the full checkpoint; the adapter
must declare that requirement instead of promising zero extra disk/RAM usage.

Cache keys include tensor bytes, shape, encoding and packing version. Node-local
repacking is allowed only with a declared conversion and separately budgeted peak
memory. Cache corruption fails closed. Never silently re-quantize the model to make
a previously approved placement fit.


## General resource identity model

`Installation` owns a bounded registry of `Node` records. A node is one agent trust
identity/OS instance, NOT one GPU. A `Device` belongs to a node and references its
physical memory domains and available backends. A `Worker` is a process/artifact
instance that may use one or multiple local devices if its adapter supports this.
A `Pool` is a physical allocation constraint; a worker may consume several pools.

Logical device views MUST resolve to the same physical device identity. Discovering
one GPU through two APIs does not create two GPUs. CPU and integrated-GPU views of
shared RAM reference one physical pool. Distinct discrete GPU pools remain distinct.
A device can require both VRAM and pinned host RAM. Model allocations are resource
vectors across pools, not a single `device.pool` byte total.

Roles include front door, control owner, execution coordinator, compute worker,
transport relay and model cache. One node can hold several roles or none of the
compute roles. Execution coordination is a NODE/PROCESS placement, not a required
GPU identity. The adapter declares coordinator CPU/RAM/runtime dependencies and
whether its graph implementation imposes a particular host or device requirement.

Node IDs survive IP address changes; device IDs survive backend enumeration-order
changes where the platform provides stable identity. A reinstalled/rekeyed agent
requires authorized identity recovery, not trust based on a matching display name.

## Configuration, observations, intent and plan are distinct

Desired configuration contains user policy and references to enrolled resources.
Observations contain read-only discovery results with timestamps and provenance.
Profiles contain measured workload/path costs bound to their test environment.
Request intent fixes model identity, quality/context contract and requested features.
An execution plan binds the admitted subset, allocations, routes and exact workers.
Editing configuration MUST NOT turn an estimate into a measured profile.

The service supports explicit configuration revisions and optimistic concurrency
checks. A request is bound to a snapshot; new config does not mutate it in place.
Local host ceilings always constrain installation-wide policy. A peer cannot raise
another host's resource budget, authorize a new binary or bypass its participation
switch. Unknown required features or incompatible schema/protocol versions fail
before worker allocation. See specs 08 and 09.

## Membership and ownership

Registry state includes enrolled, online, eligible, draining, offline and revoked;
connectivity and eligibility are independent. New nodes become candidates for new
plans, not automatically owners in active model graphs. Disable/drain/revoke actions
have different lifecycle effects and explicit client-visible outcomes.

One control owner serializes installation configuration. Node agents arbitrate local
resource leases atomically, including competing model sessions. A coordinator failure
is not solved by a second UI process claiming the same session. Controller leases,
fencing identities and expiry prevent stale owners from allocating or committing work.
Multi-controller high availability is deferred.
