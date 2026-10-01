# Execution and backend adapters

The generic manager does not embed every GPU runtime into one process. Each native
worker adapter is responsible for one validated execution engine/build and exposes a
small versioned contract to the control plane.

A target adapter contract is:

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

Adapters may wrap an existing native engine through a subprocess or local IPC boundary.
They must not translate an exact placement request into an unrelated approximate flag
without reporting that change.

## Static representability gate

Before live qualification or worker preparation, TensorMeld performs a narrower static
check: can this adapter represent the planner candidate *exactly*?

The versioned `tensormeld/adapter-capabilities-v1` envelope describes explicit device
bindings, backend labels, coordinator support, route modes, whole-block range pinning,
mixed-backend/remote-compute support and bounded node/device/segment limits.

The corresponding representability report is either `REPRESENTABLE` or `REJECTED`
with structured reason codes. A successful report remains:

```text
qualified = false
executable = false
```

It is not a runtime handshake, operator/model qualification, memory lease, transport
proof or worker launch. Adapter self-description alone cannot upgrade a plan.

Malformed or tampered candidates are contract errors, not ordinary adapter rejections.
The candidate's unit IDs and contiguous segments must remain exactly those implied by
the planner ownership sequence.

This gate lets TensorMeld reuse existing inference engines without silently translating
an exact placement request into an approximate engine flag.

The first executable distributed baseline is whole-block placement. Expert/tensor/phase
strategies are added only when the adapter can represent and validate them faithfully.
