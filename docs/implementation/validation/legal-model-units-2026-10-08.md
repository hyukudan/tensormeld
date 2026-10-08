# Legal model units validation — 2026-10-08

Implementation slice: adapter-declared ordered indivisible legal model units and explicit
cut-after boundaries over exact tensor-movability evidence.

## Scope

Every tensor must belong to exactly one unit. Unit device sets may only narrow the
intersection of tensor-level legal devices. Alias/tied groups must stay inside one unit.
Execution order and legal cuts are explicit evidence and are never inferred from tensor
names or model family.

## CI

Pull request #32 workflow run 37782002306 completed with all seven jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

## Result

PR #32 merged as 355f759e876a01ffafe2ca4194e6b788f24c5264.

This remains portable contract/fixture validation. It does not enable tensor-level
execution, change planner ownership, or claim native target-host cut correctness.
Whole-block execution remains the validated baseline.
