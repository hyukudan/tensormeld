# ADR-0047 — Legal model units and cuts are adapter-declared evidence

Status: **accepted**  
Date: 2026-10-08

## Context

Tensor-level movability states where individual tensors may legally live, but a runtime may still require groups of tensors to move and execute together. Model dependency order and legal cut points cannot be reconstructed safely from tensor names, numeric layer suffixes or model-family assumptions.

## Decision

TensorMeld introduces `tensormeld/legal-model-units-v1`.

The evidence producer explicitly declares:

- a contiguous sequence number for every legal unit;
- the complete set of tensors in each indivisible unit;
- the unit's allowed devices;
- whether a cut is legal immediately after that unit.

Every tensor from the bound tensor-movability profile must belong to exactly one unit. Unit-level allowed devices may only **narrow** the intersection of the tensor-level legal device sets of all member tensors.

Tied/alias-group tensors must remain within one indivisible unit.

The final unit cannot declare `cut_after=true`. Unit IDs and sequence positions are unique, and sequence positions must cover `0..N-1` exactly.

The legal-unit profile inherits and must exactly match tensor-movability provenance. It cannot self-promote qualification or executability.

## Consequences

- TensorMeld does not infer dependency order or legal cuts from tensor names or model family.
- A group with no common legal device fails closed.
- Adapter evidence may conservatively narrow legal devices for a unit but cannot enlarge the underlying tensor-level set.
- Alias/tied tensors cannot be separated by a planner cut.
- These units are future planner inputs only. Whole-block execution remains the validated baseline until a native adapter proves finer-grained execution semantics and correctness.
