# ADR-0023 — Remote control references host-owned objects by identity

Status: **accepted**  
Date: 2026-10-02

## Context

TensorMeld now has an authenticated private control channel and a local HostAgent. Remote
control must not turn the channel into a generic object-loading or execution API.

In particular, a peer must not be able to send arbitrary runtime manifests, snapshots,
paths, commands or executables and ask the host to trust them.

## Decision

Remote endpoint records are explicit and bind node ID, enrollment ID, private/loopback IP
literal, port and exact peer-certificate SHA-256. Public, hostname, unspecified,
multicast and link-local endpoint targets are rejected.

Remote HostAgent requests are bounded request/response records with unique request IDs.
Only the existing agent operation allowlist is dispatchable.

For `reserve` and `launch_recheck`, the peer supplies only:

- lease ID;
- exact runtime-manifest SHA-256;
- observation ID.

The host resolves those identities through a local object registry populated by trusted
local control-plane code. Unknown references fail closed. Remote requests cannot carry
paths, commands, executable names, model bytes, tensor payloads or replacement manifest
objects.

Responses correlate the request ID and operation before returning a result.

## Consequences

- Remote control authority remains narrower than local object registration authority.
- Network peers cannot inject replacement resource manifests or observation payloads.
- Endpoint selection cannot silently route control to a public address.
- A future multi-machine integration can reuse the same dispatch contract over mTLS.
- Tensor transport and executable worker dispatch remain separate later milestones.
