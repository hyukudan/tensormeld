# ADR-0005 — Whole-block placement is the first executable distributed baseline

**Status:** Accepted
**Date:** 2026-10-01

## Decision

The first executable distributed adapter targets contiguous/whole model block ownership
before expert-level, tensor-parallel or phase-specialized placement.

## Why

Whole-block ownership minimizes synchronization surfaces and makes memory/state ownership
and correctness easier to validate. It is also useful for capacity expansion over
moderate network links.

## Consequences

- advanced strategies remain first-class future features, not rejected ideas;
- the baseline provides a measured comparison for every more complex strategy;
- models whose architecture cannot expose legal block boundaries require another
  validated strategy rather than an approximation.
