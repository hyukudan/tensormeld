# ADR-0017 — Retained E2 evidence is native-only and cannot restore readiness alone

Status: **accepted**  
Date: 2026-10-01

## Context

The pinned llama.cpp backend self-test can produce a narrow E2 fact: one explicitly
bound backend initialized and executed supported `ADD` correctness cases.

Portable tests use injected runners and intentionally produce the same structural
self-test result so parser and promotion rules can be validated without GPU hardware.
Those fixtures must never be persisted or later interpreted as target-host evidence.

A successful native self-test is also not timeless. TensorMeld does not yet have a
stable live identity covering worker build, driver/runtime and relevant topology state.
Reusing an old pass to restore `ready` after those inputs change would therefore be
unsafe.

## Decision

TensorMeld may retain a deterministic backend-readiness evidence record only when the
self-test result states `execution_source=native-subprocess`.

The retained record binds:

- pinned llama.cpp source revision;
- exact `test-backend-ops` artifact SHA-256;
- exact probe artifact SHA-256;
- exact approved binding fingerprint;
- exact TensorMeld config fingerprint;
- TensorMeld device ID;
- engine-local device name;
- backend identity from TensorMeld config;
- operation and successful result counts;
- canonical fingerprint of the complete originating self-test result.

Injected-runner results are rejected at the retention boundary.

Applicability against a later fresh binding requires every retained identity above to
match. Even when they match, the result remains `runtime_ready=false`,
`reservation_created=false`, `qualified=false` and `executable=false`.

Until live worker/driver/topology identity is implemented, the native self-test must run
again before a later observation can be promoted to `ready`.

## Consequences

- Fixture/CI portability cannot accidentally become hardware qualification evidence.
- Retained E2 evidence is tamper-evident and narrowly scoped.
- A config, binding, probe artifact, test artifact, device mapping or backend change
  invalidates applicability.
- Current retained evidence is useful for provenance/audit but is deliberately
  insufficient for execution admission.
- Runtime model/operator/memory manifests and live invalidation identity remain separate
  later gates.
