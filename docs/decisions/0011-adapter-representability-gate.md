# ADR-0011 — Separate planner proposals from adapter representability

Status: **accepted**  
Date: 2026-10-01

## Context

The v2 planner produces advisory whole-block candidates. Those candidates are deliberately
marked `executable=false` and `qualified=false`.

A native engine may expose flags that approximate device splitting, but an approximate
translation is not enough. TensorMeld must know whether the engine can represent the
planner's ownership, unit ranges, coordinator choice and route mode without silently
changing them.

## Decision

TensorMeld introduces a distinct static **adapter representability** gate.

An adapter publishes a bounded, versioned capability envelope that identifies:

- adapter and engine revision;
- exposed TensorMeld device identities, nodes and backend labels;
- supported placement strategies;
- whether exact device ownership can be guaranteed;
- whether explicit contiguous unit ranges can be pinned;
- whether mixed backends and remote compute are representable;
- coordinator constraints;
- route modes;
- explicit node/device/segment limits.

The control plane validates a planner candidate against that envelope and returns either
`REPRESENTABLE` or `REJECTED` with structured reason codes.

A representability result **must remain** `qualified=false` and `executable=false`.
It does not launch a worker, reserve memory, prove operator/model compatibility, establish
secure transport, or prove that the capability declaration is truthful on the current host.

Malformed/tampered candidates are validation errors rather than ordinary adapter
rejections.

## Consequences

- Planner output cannot become executable merely because an adapter advertises support.
- Existing engines may be reused, but only when their concrete controls can preserve the
  requested placement contract.
- Approximate flags must be reported as unsupported for exact placement instead of being
  silently substituted.
- Future live qualification and worker preparation can build on this static gate without
  changing the meaning of planner results.
