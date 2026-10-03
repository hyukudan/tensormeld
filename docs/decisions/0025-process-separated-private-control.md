# ADR-0025 — Private control is process-separated before multi-host rollout

Status: **accepted**  
Date: 2026-10-03

## Context

The private control stack has real mTLS and end-to-end remote dispatch, but earlier
integration kept client and server in one process. Before moving to real machines,
TensorMeld needs to prove that endpoint binding, TLS setup, host-local object authority
and lease state survive an actual process boundary.

## Decision

TensorMeld adds explicit private endpoint client/server helpers.

The server binds only an already-validated PrivateEndpoint and requires:

- its local enrollment ↔ certificate binding;
- expected peer enrollment ↔ certificate binding;
- mutual TLS;
- exact certificate pinning;
- the existing connection epoch/replay rules;
- bounded request count per integration server instance.

The client validates the same endpoint/enrollment/certificate relationship before
connecting.

CI launches the server HostAgent in a separately spawned OS process. The server process
constructs and registers its own runtime manifest and admission snapshot. The client
process performs remote health and reserve calls using only object identities.

## Consequences

- Host-local object authority is now validated across a real process boundary.
- The next networking step can focus on actual private-LAN host reachability.
- The helper still exposes only bounded control operations; native workers remain outside
  the remote protocol.
- Loopback process separation is not multi-machine, GPU, performance, tensor-transfer or
  distributed-inference evidence.
