# Performance-aware legal-unit planner validation — 2026-10-10

Implementation slice: bounded advisory ranking over legal model units using exact
per-unit compute/memory evidence and conservative directional transfer evidence.

## Scope

The planner computes an ordered unit-chain upper bound from:

- explicit per-unit device compute_us;
- explicit directed boundary-transfer upper bounds for owner changes;
- tensor resident bytes counted in full, including reclaimable/file-backed bytes;
- additive persistent state;
- per-device/pool workspace and staging maxima, conservatively summed for shared pools.

It enforces legal cut boundaries, current static selection policy and physical-pool
budgets. Missing path coverage rejects only candidates that require that transfer.

The objective is explicitly not token latency, TTFT or throughput. Sampling, token
feedback, phase-specific prefill/decode behavior, overlap, contention and queueing are
outside this slice.

## CI

Pull request #36 completed with all seven portable jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

Tests include:
- a faster helper winning only when compute savings exceed transfer cost;
- a slower/expensive transfer keeping work local;
- full state/workspace/staging pool accounting;
- memory forcing a legal split;
- a required split becoming infeasible when its directed path is missing;
- bounded/deterministic search and exact runtime-environment identity checks.

## Result

PR #36 merged as c06466ba9264cba1b2069784eddde0c8c6ac29e6.

All results remain qualified=false and executable=false. No real RTX/Strix/CUDA/HIP
performance measurement or finer-grained native execution is claimed.
