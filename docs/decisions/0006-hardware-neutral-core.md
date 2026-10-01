# ADR-0006 — Strix Halo is a qualification profile, not a product dependency

**Status:** Accepted
**Date:** 2026-10-01

## Decision

The core contracts contain no Strix-Halo-specific requirement. Strix Halo is the first
large-unified-memory companion profile we intend to qualify.

## Why

Users may have a 12 GB desktop GPU and one helper, a high-VRAM workstation and several
helpers, or completely different future accelerators. Model size and hardware brand are
inputs to planning, not product identity.

## Consequences

- hardware-specific optimizations live in adapters/profiles;
- dense and MoE models are both first-class;
- tests include small-VRAM and multi-node scenarios;
- the manager remains useful even when only one remote machine executes the model.
