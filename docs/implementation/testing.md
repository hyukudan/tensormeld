# Testing strategy

Tests are layered by claim strength.

## Contract tests

Pure tests for malformed input, unknown references, conflicting policies, shared memory
pools, required resources and deterministic resolution. These run on Windows and Linux.

## Planner simulation tests

Synthetic models/topologies exercise search bounds, failure explanations and resource
accounting. They prove algorithm behavior, not hardware performance.

## Runtime observation tests

Runtime snapshots test config identity binding, backend identity, ready/offline/draining
states, physical-pool available-byte bounds, owner headroom intersection, required-resource
failure and CLI/output safety.

These tests prove conservative control-plane behavior only. They do not prove freshness,
create memory reservations, or establish GPU availability at launch time.

## Native backend readiness tests

Portable tests inject bounded fixture output from the pinned llama.cpp
`test-backend-ops` contract. They verify artifact/config/binding checks, exact command
construction, strict SQL-record parsing, rejection of exit-0-without-target-execution,
source-revision matching and observed → ready promotion.

An injected runner is fixture evidence only. Real E2 evidence requires the actual pinned
native `test-backend-ops` artifact to execute on the explicitly bound backend. Runtime
`ready` remains distinct from model/operator qualification, memory reservation and
execution authorization.

## Agent/transport integration tests

Use real sockets/processes with bounded payloads and authenticated test identities.
They prove protocol behavior, not GPU-path performance.

## Backend qualification tests

Bind exact worker build, OS/runtime/driver, model revision, encodings and workload.
Compare outputs against a reference contract and exercise load/unload/cancel/restart.

## Performance qualification

Record prompt processing, decode latency distribution, time to first token, memory peaks,
transferred bytes, synchronization waits and sustained behavior. A faster microbenchmark
cannot override failed correctness or resource gates.

## 0.2.0a2 additions

Tests cover v2 whole-block memory/cost search, manual and empty-list semantics, explicit
migration, required owners, host/coordinator overhead, shared pools, token feedback,
directed paths, multirail non-aggregation, search exhaustion, exact adapter
representability, model/evidence invalidation and advisory runtime availability.

GGUF tests using injected readers are adapter-contract tests. The separately named
`GGUFUpstreamIntegrationTests` requires the real optional package; a skip is explicitly
not a pass. The optional-dependencies CI job installs the package and requires imports
before running this integration suite.

Hosted Windows/Linux CI validates portable software behavior only. Native GPU
qualification remains a separate hardware job class.
