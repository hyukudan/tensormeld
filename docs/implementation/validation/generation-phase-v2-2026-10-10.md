# Generation phase v2 validation — 2026-10-10

Implementation slice: backward-compatible phase-specific inter-unit boundary payloads.

## Scope

Generation-phase v2 adds distinct prefill and decode boundary payload bytes per legal unit.
The final unit must close both payloads at zero. v1 remains parseable and retains its
original canonical fingerprint behavior, but is explicitly marked insufficient for
phase-transfer modeling.

## CI

Pull request #38 completed with all seven portable jobs green:
- Linux / Python 3.11 and 3.13;
- Windows / Python 3.11 and 3.13;
- optional adapter dependencies on Linux and Windows;
- real mTLS loopback integration.

## Result

PR #38 merged as 039db69674f2df273afbfc022b7656bf5c6de2ef.

This enables sound future phase-transfer accounting, not TTFT/token-latency/throughput
or native-execution claims.
