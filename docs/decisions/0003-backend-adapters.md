# ADR-0003 — Use backend adapters instead of one universal GPU process

**Status:** Accepted
**Date:** 2026-10-01

## Decision

CUDA, HIP and other execution engines are integrated through versioned native worker
adapters. The control plane does not require all GPU SDKs in one process.

## Why

Heterogeneous driver/runtime stacks have different build and compatibility constraints.
A subprocess/IPC boundary permits independent qualification and rollback.

## Consequences

- each adapter advertises exact capabilities;
- an adapter must accept or reject the exact plan;
- optimized hardware-specific workers remain optional;
- Python stays outside the steady-state tensor hot path.
