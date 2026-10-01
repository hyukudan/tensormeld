# ADR-0002 — Model nodes, devices and physical pools separately

**Status:** Accepted
**Date:** 2026-10-01

## Decision

A node owns devices; devices reference physical memory pools; owner policies constrain
pools. Device memory labels are never blindly summed.

## Why

Unified-memory APUs expose CPU and GPU access to the same physical capacity. Multiple
GPUs may also have independent pools in one node. Correct admission therefore requires
pool identity rather than a flat list of advertised VRAM values.

## Consequences

- shared pools are accounted once;
- device count and memory capacity are independent dimensions;
- the planner can later reason about CPU/offload paths without inventing extra memory.
