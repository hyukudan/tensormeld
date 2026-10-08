# Predictive memory classes validation — 2026-10-08

Implementation slice: exact physical-pool memory classification layered over
RuntimeModelManifest v1.

## Scope

The predictive profile separates hard-resident, reclaimable/file-backed, persistent
state, workspace and staging classes. It does not change admission semantics.

For every physical pool the classification must reconstruct the existing runtime
manifest exactly. Worst-case physical bytes therefore remain equal to the existing
preparation peak.

## CI

Pull request #29 final workflow run 37779687471 completed with all seven jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

An earlier PR run exposed a CLI fixture that reordered coordinator identities while
reconstructing an adapter capability record. Commit
32cda257a1625b79138c39a7cf32ea7063383bf6 preserved the original coordinator order;
the final run passed.

## Result

PR #29 merged as 08eb3052155f1f4935e21a08a2bfaf0cf6710062.

This remains portable contract/fixture validation. No real RTX, Strix Halo, CUDA/HIP
memory classification or capacity/performance claim was produced.
