# ADR-0057 — Session affinity and per-request prefill placement in the replica router

Status: **accepted**
Date: 2026-10-21

## Context

ADR-0054 routes whole requests to whole-model replicas by load. Two things were left out:

- an engine with a prefix state cache only saves prefill work when a conversation's next
  turn reaches the replica that already holds its prefix; least-loaded routing alternates
  turns between replicas;
- heterogeneous setups can prefill elsewhere (a resident prefill service on a discrete
  GPU) and hand the request state to a decode replica. Where to prefill depends on the
  prompt length, on cache hits and on what is present and healthy.

## Decision

The router gains two optional, configured behaviours:

1. **Prefix affinity** (`affinity.mode = "prefix"`): requests carry message-prefix
   digests (or an explicit session id); the router prefers the replica recorded for the
   most specific known key, but only among replicas with a free slot and within
   `max_extra_inflight` of the least loaded. It never waits for a busy replica.
2. **Cost-based prefill placement** (`prefill.policy = nodes | auto | rtx`). Under `auto`
   every request's candidates are scored with measured profiles. Locally on a replica:
   slot wait + prefill of the rows its cache cannot restore, plus the pause imposed on its
   active decodes. Offloaded: the service's pending seconds + its prefill of the whole prompt
   + handoff, against the decode replica's slot wait, plus the import pause. The lowest
   wins; an offload must beat local by a noise margin, and short prompts on idle replicas
   stay local. Offloads queue on the service (a busy service is a cost), hold no decode slot
   while waiting, and fall back to a local prefill on any failure. Without services the
   same cost picks the replica: cache- and work-aware routing.

Both are engine-neutral contracts (tokenize route, import route, prefill service
routes), so every model with those routes uses the same router.

## Consequences

- Affinity changes placement only; a request still runs whole on one replica and returns
  what that replica returns alone.
- Profiles are configuration: measured curves keyed by GPU and model (seed profiles for the
  DeepSeek RTX service ship with the router); the router predicts, it does not measure.
- The cache view is a mirror of the router's own traffic at the engine's checkpoint block;
  it can be stale, which costs a misplaced request, never a wrong answer.
