# ADR-0001 — Separate control, placement and execution

**Status:** Accepted
**Date:** 2026-10-01

## Decision

The user-facing controller, placement planner and execution coordinator are separate
roles. They may run on the same node but must not be modeled as the same identity.

## Why

The best place to expose an API is not necessarily the best place to relay tensors.
A desktop connected by slower Ethernet can remain the front door while a companion node
coordinates a faster internal topology.

## Consequences

- coordinator placement becomes a plan decision;
- the stable client endpoint does not move when execution placement changes;
- transport routing must follow actual backend forwarding behavior;
- controller availability and compute availability can fail independently.
