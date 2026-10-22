# ADR-0054 — Whole-model replicas are routed, not split

Status: **accepted**  
Date: 2026-10-19

## Context

When a model fits on one node, the cheapest multi-node gain is aggregate throughput:
two nodes serving two requests at once. Each node then runs a complete engine server
with its own copy of the model, and no tensor crosses nodes.

TensorMeld had no inference front end. The first executable multi-node surface does
not need placement search, tensor transport or a new numerical identity: a request
that runs wholly on one replica returns exactly what that replica returns alone.

## Decision

TensorMeld adds `tensormeld/replica-router-v1`, a bounded HTTP router over
OpenAI-compatible engine servers (`tensormeld serve-replicas CONFIG`):

- replicas are numeric loopback/private endpoints from the config; a remote node's
  server stays bound to its own loopback and is reached through an authenticated
  tunnel, never through an unauthenticated LAN listener;
- the router itself binds loopback by default; a private listener requires a client
  API key;
- each replica declares `max_inflight` (the engine's own concurrent-request
  capacity); the least-loaded healthy replica takes each generation request, ties go
  to the least recently assigned; when all are full, requests wait in one bounded
  queue with a deadline, then get 503;
- a replica is routable only after its `/health` answers 200 and leaves rotation
  after consecutive failed probes or a refused connection;
- a request is retried on another replica only when its connection failed before any
  byte was sent; after that, failures are reported, never replayed;
- responses, including server-sent event streams, are relayed as they arrive; a
  client disconnect closes the upstream connection so the engine can cancel;
- client credentials are not forwarded; each replica has its own optional key;
- `/health`, `/api/status` (`tensormeld/replica-router-status-v1`) and `/metrics`
  describe the router and its replicas.

The router launches nothing. Launching and supervising engine processes, memory
admission per node and mode selection are separate contracts built on top of it.

## Consequences

- aggregate throughput scales with replicas without any execution-qualification
  claim: exactness is the replica's own;
- a single request is not faster; split execution remains a separate engine-group
  contract;
- prefix-affinity routing, process supervision and drain are follow-ups;
- the compatibility matrix of the public API is that of the replicas; the router does
  not reinterpret request or response fields.
