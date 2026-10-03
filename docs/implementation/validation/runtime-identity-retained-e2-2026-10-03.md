# Live runtime identity / retained E2 v2 validation — 2026-10-03

Implementation slice: stable runtime identity and exact invalidation of retained backend
readiness evidence.

## Contract

`tensormeld/runtime-identity-v1` fingerprints stable worker, OS, driver/runtime,
physical-device and topology identity for one TensorMeld device.

Native backend self-test results now carry the bound node ID. Retained evidence schema v2
embeds the exact runtime identity and includes its fingerprint in the invalidation keys.

Applicability requires the previous config/probe/binding/test/device/backend identities,
a fresh observed binding, matching node identity and an exact current runtime-identity
fingerprint. A match promotes only the target device from `observed` to `ready` in a
copied runtime observation. It does not create a lease, qualify a model or make the
workload executable.

## Portable coverage

Tests cover deterministic identity construction, serialized fingerprint verification,
duplicate-key rejection, native-only retention, node/device continuity, stale binding,
tampering, driver identity change and exact-match runtime-ready promotion.

All worker/driver/runtime/topology values in portable tests are fixtures. No real GPU,
CUDA/HIP driver, Windows GPU backend, performance or distributed inference is qualified.

GitHub Actions result: PR #11, workflow `Portable Python tests`, run #116 (37119396744) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. Hosted CI validates fixture-based runtime-identity/invalidation semantics only; it does not establish any real GPU, driver/runtime or topology identity.
