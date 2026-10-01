# ADR-0004 — Bounded planning search with explicit incompleteness

**Status:** Accepted
**Date:** 2026-10-01

## Decision

Planning has explicit candidate and time budgets. Exhaustive search may be used for
small cases but is not the universal algorithm.

## Why

The combination of nodes, devices, partition boundaries, routes and strategies grows
combinatorially. A configurable node count cannot be implemented safely by merely
raising a constant in an exhaustive planner.

## Consequences

- `SEARCH_INCOMPLETE` is distinct from `NO_FEASIBLE_PLAN`;
- deterministic heuristics may seed candidate construction;
- exact small-case search remains valuable as a correctness oracle;
- search budgets are persisted with plan evidence.
