# llama.cpp backend self-test contract validation — 2026-10-01

Implementation slice: strict backend-readiness evidence using the pinned upstream
`test-backend-ops` target.

## Upstream contract reviewed

Source revision: `552f18f912a32ea86edf82e2b76431cb7131538d`.

At that revision, `tests/test-backend-ops.cpp` loads native backends, accepts `-b` for an
exact backend-device filter, initializes the selected backend and executes correctness
tests against a CPU reference in `test` mode.

A successful process exit alone is insufficient evidence: non-selected devices are
counted as skipped successes, so a filter that matches no backend can still reach exit 0.
The TensorMeld adapter therefore requires positive target-backend result rows.

The upstream SQL printer includes `build_commit`, backend name, operation, mode,
supported/pass flags and error text. TensorMeld parses this text without executing SQL.

## Portable validation

The new tests use a harmless local fixture file plus an injected runner. They cover exact
artifact identity, config/binding/probe continuity, fixed command construction, rejection
of exit 0 without target execution, source revision matching, bounded parsing, observed
to ready promotion, and CLI output safety.

These are portable contract fixtures, not real backend execution evidence.

## Hardware status

No real CUDA/HIP `test-backend-ops` binary or GPU was executed for this validation
record. No Windows GPU, Strix Halo, performance or distributed-inference claim is made.

GitHub Actions result: pending for the implementation commit.
