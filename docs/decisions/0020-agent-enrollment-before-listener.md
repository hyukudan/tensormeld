# ADR-0020 — Enrollment precedes any remote agent listener

Status: **accepted**  
Date: 2026-10-02

## Context

TensorMeld now has local runtime manifests and host-owned lease primitives. Exposing them
directly over a LAN would violate the security model: a local network is not trusted,
peer identity must be explicit, replay must be rejected and arbitrary shell/backend
execution must never become a control-plane feature.

The encrypted transport boundary is not implemented yet.

## Decision

TensorMeld introduces a local `HostAgent` with no remote listener.

Enrollment is explicit and runtime-supplied. It binds an enrollment ID, node ID and key
ID to a runtime-only shared secret. Capability envelopes are versioned, bound to the
exact config fingerprint and agent instance, sequence-numbered and authenticated with
HMAC-SHA256 from Python's standard library. Verification uses constant-time comparison
and optional monotonic replay rejection.

The shared secret is never included in an envelope, repository configuration or output.

The agent exposes only fixed methods for description, lifecycle and local lease control.
It cannot execute peer-supplied shell fragments, arbitrary commands, executables or paths.

A host agent can reserve/recheck only the physical pools owned by its enrolled node,
preserving node-owner authority for future multi-host admission.

Lifecycle states are `enabled`, `draining`, `disabled` and `revoked`. Draining
blocks new reservations while allowing existing leases to be released. Disable/revoke
fails while active leases remain.

## Consequences

- Enrollment and authenticated capability identity can be tested before networking.
- No unauthenticated LAN surface is introduced.
- HMAC envelopes authenticate metadata but do not encrypt traffic; they are not the
  future LAN transport security boundary.
- Mutual authenticated encryption, certificate/key storage, rotation and persistent
  revocation remain the next transport/security slice.
- Host-owned resource authority is preserved without claiming distributed admission is
  complete.
