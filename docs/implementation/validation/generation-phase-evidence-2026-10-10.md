# Generation phase evidence validation — 2026-10-10

Implementation slice: exact scenario-specific prefill/decode-step and sampling evidence.

## Scope

The contract separates:
- prefill compute for an explicit prefill token count;
- one decode-step compute at an explicit context position;
- sampling compute;
- logits payload to the sampler;
- token-feedback payload from the sampler.

Every legal unit/device pair requires explicit phase costs. Sampling devices must belong
to the exact runtime manifest. Evidence is bound to exact config, legal-unit, cost,
movability/model/adapter and runtime-manifest identities.

## CI

Pull request #37 completed with all seven portable jobs green:
- Linux / Python 3.11 and 3.13;
- Windows / Python 3.11 and 3.13;
- optional adapter dependencies on Linux and Windows;
- real mTLS loopback integration.

## Result

PR #37 merged as b5267f642898d3858ee1d5d35aabdd9ea399fed8.

This contract alone is not TTFT, token latency, throughput, qualification or native
execution evidence.
