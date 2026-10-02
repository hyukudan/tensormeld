# Secure agent/enrollment validation — 2026-10-02

Implementation slice: local enrolled host agent before any remote listener.

## Contract

The new host-agent skeleton uses explicit enrollment ID, node ID and key ID plus a
runtime-only secret. Versioned capability envelopes bind config identity, agent instance,
monotonic sequence, lifecycle and an allowlist of operations.

Authentication uses Python standard-library HMAC-SHA256 and constant-time signature
comparison. Replay guards reject a repeated or stale sequence for the same enrolled agent
instance.

The agent exposes no remote listener and no peer-supplied shell/executable API.

Local reservation and launch recheck calls are scoped to physical pools owned by the
enrolled node, extending the existing admission primitive without giving a remote peer
authority over another host's budgets.

## Portable coverage

Tests cover valid authentication, replay rejection, tampering/wrong-secret failure,
secret non-serialization, signed monotonic descriptions, host-owned pool scoping,
drain/disable lifecycle and explicit rejection of arbitrary peer execution.

These are portable control-plane tests. They do not prove encrypted LAN transport,
certificate storage, CUDA/HIP or Windows GPU execution, model correctness, performance
or distributed inference.

GitHub Actions result: pending for the implementation PR.
