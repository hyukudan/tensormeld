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

The first executable distributed baseline is whole-block placement. Expert/tensor/phase
strategies are added only when the adapter can represent and validate them faithfully.
