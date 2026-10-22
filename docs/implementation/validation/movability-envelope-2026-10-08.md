# Movability envelope validation — 2026-10-08

Implementation slice: auditable per-device/per-physical-pool tensor eligibility envelopes.

## Scope

The envelope combines exact runtime-manifest identity, predictive memory identity and
tensor movability evidence without treating legal placement potential as current ownership
or measured resident memory.

Per-device eligible-byte totals may overlap. Per-pool unions count each tensor once for
that physical pool even when multiple logical devices share it.

## CI

Pull request #31 workflow run 37781215312 completed with all seven jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

The shared-pool fixture verifies that one movable 150-byte tensor eligible on two logical
devices contributes 300 overlapping device-eligible bytes but only 150 bytes to the
physical-pool union.

## Result

PR #31 merged as a2c548008f0b768952c1f333d37031b40bcbec45.

This remains portable contract/fixture validation. It does not measure resident memory,
assign current tensor owners, create planner candidates or claim target-host performance.
