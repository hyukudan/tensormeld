# ADR-0024 — Remote agent control is validated end-to-end before LAN expansion

Status: **accepted**  
Date: 2026-10-02

## Context

TensorMeld separately validated a real mTLS loopback channel and bounded remote-agent
dispatch. Before adding multi-machine endpoint handling, those layers need one integrated
test showing that authenticated transport, request correlation, local object resolution
and host-owned lease creation work together.

## Decision

The real-mTLS CI job now exercises RemoteAgentClient and RemoteAgentDispatcher over the
actual mutually authenticated loopback TLS socket.

The integration performs:

1. real ephemeral-certificate mTLS handshake;
2. enrolled peer certificate pinning;
3. remote `health` request/response correlation;
4. remote `reserve` request carrying only lease ID, runtime-manifest SHA-256 and
   observation ID;
5. server-local resolution of the manifest and snapshot;
6. host-owned lease creation in the server HostAgent.

No runtime manifest or observation object crosses the control channel.

## Consequences

- The remote control stack is validated as one composed path rather than isolated layers.
- The next network step can focus on private-LAN endpoint/process integration rather than
  basic dispatch semantics.
- Loopback success is still not evidence of cross-machine reachability, tensor transport,
  GPU execution or distributed inference.
