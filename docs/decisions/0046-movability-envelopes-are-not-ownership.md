# ADR-0046 — Movability envelopes describe legal potential, not measured ownership

Status: **accepted**  
Date: 2026-10-08

## Context

Tensor movability evidence states which devices may legally own each exact tensor, while
predictive memory describes physical-pool memory classes for the currently observed runtime
manifest.

Those are different facts. A tensor that is legal on two devices is not resident twice, and
eligibility bytes are not a runtime memory measurement.

Shared-memory systems add another trap: two logical devices can reference the same physical
pool. Summing each device's eligible bytes would double count the same legal tensor set.

## Decision

TensorMeld derives a read-only `tensormeld/movability-envelope-v1` from:

- exact runtime model manifest;
- exact predictive memory profile;
- exact tensor movability evidence;
- installation device-to-physical-pool mapping.

For every device, the envelope reports overlapping eligible tensor bytes by storage class and
the subset that is pinned to that exact device.

For every physical pool, TensorMeld computes the **union** of tensors legal on at least one
device backed by that pool. A tensor is counted once per physical pool even when multiple
logical devices sharing that pool may own it.

The envelope does not assign a current owner, does not change planner candidates, and does not
attempt to reconcile eligibility totals with measured resident memory. Runtime overhead,
repacking, aliases and inactive legal alternatives make those quantities intentionally
different.

## Consequences

- per-device eligible totals may overlap and must not be summed as physical memory;
- physical-pool union totals avoid duplicate accounting within shared/unified pools;
- pinned bytes provide a lower-bound ownership constraint, not total resident demand;
- measured runtime memory and legal placement potential remain separate evidence surfaces;
- planner integration requires a later legal-unit construction step and cannot treat these
  envelopes as executable placements;
- no performance or capacity gain is implied by a larger legal envelope.
