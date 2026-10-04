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

## Consequences

- Resources are not deliberately made available while synchronous backend work may still
  be active.
- Current cancellation is cooperative at whole-block boundaries, not a hard subprocess
  termination primitive.
- A future real llama.cpp backend may add a process handle/termination protocol while
  preserving this lease-ownership rule.
- Successful session execution still does not imply real llama.cpp/GPU evidence in
  portable CI.
