# ADR-0035 — Native session owns the admitted lease until execution is terminal

Status: **accepted**  
Date: 2026-10-04

## Context

Target-host admission can now produce an AcceptedExecutionBundle and a launch-admitted
lease. Starting worker execution introduces a new resource-lifecycle boundary: the lease
must not be released while backend work may still be using the reserved resources.

The existing whole-block subprocess interface is synchronous and does not expose a
portable process-kill handle to the session layer.

## Decision

TensorMeld introduces `tensormeld/native-admitted-session-v1`.

A session can be created only when:
- TargetHostAdmissionResult is intact and execution-authorized;
- backend is bound to the exact AcceptedExecutionBundle;
- the referenced lease is still active in state `launched`;
- lease SHA and runtime-manifest SHA match the admission/bundle.

The managed session owns lease cleanup.

Terminal completion and backend failure release the lease. Cancellation before execution
releases immediately without invoking backend work.

If cancellation arrives while a backend call is already in flight, TensorMeld records a
cancel request but does not release the lease immediately. The existing backend is allowed
to reach the next safe terminal boundary, after which the session releases the lease.

Explicit release is idempotent and cannot directly release a session marked running.

### Concurrent lifecycle operations

The native session serializes run ownership, cancellation and explicit release under
one lifecycle lock. Calls into the reference lifecycle acquire locks in native-to-reference
order. Backend execution does not hold the native lifecycle lock. A cancellation or
release decision made before execution cannot subsequently release resources owned by a
concurrent run.

`CANCEL_REQUESTED` acknowledges cooperative cancellation while the run still owns its
lease; it reports `cancelled=false` and `lease_released=false`. Ownership persists until
the synchronous reference run returns or raises, including cancellation during startup.
Cancellation that prevents all backend work returns a terminal `CANCELLED` run result
with `inference_started=false`.

Cancellation after a terminal outcome preserves that outcome. It returns
`ALREADY_COMPLETED`, `ALREADY_FAILED`, `ALREADY_CANCELLED` or `ALREADY_RELEASED`, according
to the native state. Only `ALREADY_CANCELLED` reports `cancelled=true`. Each response
reports `lease_released` from actual lease cleanup. A rejected reference release is
propagated and cannot authorize lease cleanup.

Terminal state publication and lease cleanup are serialized with lifecycle operations.
Deterministic portable tests cover concurrent startup, cancellation, explicit release,
backend failure and cleanup. They provide software lifecycle evidence, not hardware or
real model inference qualification.

## Consequences

- Resources are not deliberately made available while synchronous backend work may still
  be active.
- Current cancellation is cooperative at whole-block boundaries, not a hard subprocess
  termination primitive.
- A future real llama.cpp backend may add a process handle/termination protocol while
  preserving this lease-ownership rule.
- Successful session execution still does not imply real llama.cpp/GPU evidence in
  portable CI.
