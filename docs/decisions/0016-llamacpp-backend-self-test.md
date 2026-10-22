# ADR-0016 — Backend readiness requires positive native execution evidence

Status: **accepted**  
Date: 2026-10-01

## Context

The pinned llama.cpp `test-backend-ops` target can select one backend with `-b`.
At revision `552f18f912a32ea86edf82e2b76431cb7131538d`, a missing selected backend can still
produce exit code 0 because non-matching devices are counted as skipped successes.
Exit status alone therefore cannot prove that the intended backend initialized or ran.

TensorMeld also must not invent a replacement GPU test kernel when upstream already
provides a bounded operator correctness harness.

## Decision

TensorMeld reuses the pinned upstream `test-backend-ops` target for a narrow E2
backend-readiness self-test.

For one explicitly bound TensorMeld device, the adapter:

1. requires explicit trust and an exact SHA-256 for the local `test-backend-ops` artifact;
2. revalidates the exact installation fingerprint, binding fingerprint, probe artifact,
   configured backend and engine-device mapping;
3. accepts only a fresh bound runtime state of `observed`;
4. executes a fixed command equivalent to
   `test-backend-ops test -b <bound-engine-device> -o ADD --output sql -j 1`;
5. uses no shell or stdin and bounds runtime and captured output;
6. parses, but never executes, the pinned tool's SQL-format records;
7. requires result rows for the exact target backend, `ADD`, and `test` mode;
8. requires the reported llama.cpp source commit to match the pinned revision;
9. requires at least one supported target row that passed with no error;
10. rejects any failed supported target row even when the process exits 0.

Only after those checks may the corresponding runtime observation move from `observed`
to `ready`. The result still states `reservation_created=false`, `qualified=false` and
`executable=false`.

## Consequences

- A successful process exit without target-backend result rows is rejected.
- Fixture/injected-runner tests validate parsing and promotion rules only; they are not
  CUDA, HIP, Windows GPU, performance or hardware qualification.
- Real E2 evidence requires running the pinned native target on the explicitly bound
  backend and retaining its exact artifact/source/config/binding identities.
- Runtime `ready` is narrower than model/operator qualification and does not create a
  memory lease or authorize inference.
- Model-specific operator/memory manifests and launch-time admission remain later gates.
