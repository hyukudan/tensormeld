# Runtime optimization principles

Public documentation records project-owned contracts and implementation priorities.
Detailed comparative evaluations, source observations and third-party measurements
are retained privately outside Git. They do not qualify this runtime.

## Implementation priorities

- Observe the selected compute runtime and bind devices explicitly.
- Derive exact model bytes and semantics from approved checkpoint metadata.
- Calibrate placement with distinct prefill and decode workloads.
- Account once for each physical pool, including state, workspace and staging peaks.
- Own child-process lifecycle and keep leases charged until cleanup is confirmed.
- Bind cache/state reuse to model, numerical plan, layout and runtime identity.
- Compare packed execution, residency and transport alternatives by complete useful work.
- Keep optimization modes separately qualified and reject silent semantic changes.

These principles are independent of any external project's implementation or
performance. Refer to accepted ADRs and the implementation ledger for current scope.
