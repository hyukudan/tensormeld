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

## Retained E2 evidence tests

Portable tests build deterministic records from synthetic self-test dictionaries and
exercise exact config/probe/binding/test-artifact/device/backend applicability.

The retention boundary is intentionally stricter than the self-test parser:
`execution_source=injected-runner` is rejected. Only a self-test result marked
`native-subprocess` can become a retained E2 record.

These tests validate serialization, fingerprints and invalidation behavior only. They do
not prove that a native GPU backend actually ran. A retained record does not restore
runtime `ready`; live worker/driver/topology invalidation identity and a runtime recheck
remain required.

## Runtime model/operator/memory manifest tests

Portable fixtures validate exact binding to configuration, model manifest, profile
workload, adapter capability fingerprint, engine revision and worker artifact identity.

Operator requirements are explicit and compared against each declared worker device.
Incomplete coverage remains a valid observation but never self-qualifies.

Memory is reported at the physical-pool level, not once per logical device. Duplicate
pool records are rejected. Resident bytes, state bytes and peak workspace form the
steady peak; preparation peak must cover it and cannot exceed a known physical capacity.

Fixture provenance is contract evidence only. A native-adapter manifest still needs live
admission and E3 correctness evidence before execution.

## Local reservation/admission tests

Portable tests exercise atomic in-process leases across all physical pools required by an
exact runtime manifest. Tests cover concurrent contenders, deterministic release, launch-time
recheck using a newly identified observation, rejection under reduced availability, and
shared-pool accounting inherited from the manifest.

Admission snapshots may explicitly list active lease IDs whose allocations are already
reflected in reported available bytes. Such leases are not subtracted again; active leases
not listed as reflected are charged conservatively. This accounting rule is local process
control-plane behavior only and is not a distributed lock or proof of GPU allocation.

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
representability, model/evidence invalidation, advisory runtime availability and exact
runtime-manifest invariants.

GGUF tests using injected readers are adapter-contract tests. The separately named
`GGUFUpstreamIntegrationTests` requires the real optional package; a skip is explicitly
not a pass. The optional-dependencies CI job installs the package and requires imports
before running this integration suite.

Hosted Windows/Linux CI validates portable software behavior only. Native GPU
qualification remains a separate hardware job class.


## Agent/enrollment tests

Portable tests validate explicit enrollment identity, authenticated capability envelopes,
constant-time HMAC verification, monotonic replay rejection, secret non-serialization,
host-owned pool scoping, drain/disable behavior and the absence of a peer-supplied
execution/shell API.

The agent module opens no remote listener. HMAC-SHA256 uses Python's standard-library
implementation; this slice is an authenticated envelope and host-authority primitive, not
encrypted LAN transport. Mutual authenticated encryption remains a separate gate.


## Private TLS control-channel tests

Portable tests validate TLS policy construction, required client-certificate verification,
dedicated ALPN, exact peer-certificate SHA-256 pinning, bounded control framing,
connection-epoch checks, monotonic sequence enforcement and rejection of non-allowlisted
operations.

The socket tests use an injected TLS-socket fixture and therefore prove the control-channel
contract, not a real certificate handshake or encrypted LAN path. Real provisioned mTLS
integration remains a separate test gate. Python `ssl`/OpenSSL is reused; TensorMeld does
not implement TLS cryptography itself.


## Real mTLS loopback integration

A dedicated Linux CI job generates an ephemeral CA plus server/client certificates with
OpenSSL at runtime, stores them only in the job workspace, performs a real mutual TLS
handshake over 127.0.0.1, verifies certificate pinning against enrolled identities, and
exchanges bounded allowlisted control frames through `PrivateControlChannel`.

No certificate or private key fixture is committed to Git. Windows continues to validate
the portable TLS/control contract but does not run this OpenSSL-generated integration job.
Loopback mTLS success is not private-LAN performance or distributed-inference evidence.


## Private endpoint and remote-agent dispatch tests

Portable tests validate that remote endpoints are explicit IP literals and are restricted
to private or loopback address space. Public, hostname, unspecified, multicast and
link-local targets fail closed.

Remote dispatch accepts only the existing HostAgent operation allowlist. Reservation and
launch-recheck requests carry only a lease ID, runtime-manifest SHA-256 and observation
ID; the actual manifest/snapshot must already exist in a host-owned local registry.
Unknown local references, duplicate request IDs and unexpected argument fields are
rejected.

These tests validate control-plane semantics only. They do not create a real two-machine
LAN deployment, transport tensor payloads, or prove distributed inference.


## End-to-end remote dispatch over real mTLS

The dedicated real-mTLS Linux job now also constructs a HostAgent, pre-registers an exact
runtime manifest and admission snapshot on the server, then performs RemoteAgentClient
calls through the real mutually authenticated TLS socket. The test validates a remote
`health` call and a remote `reserve` request that resolves only server-local object
identities and creates the lease on the server-side HostAgent.

This is still loopback-only integration. It does not establish multi-machine LAN
reachability, tensor transport, GPU execution or distributed inference.


## Separate-process private endpoint integration

The dedicated Linux mTLS job now also starts the enrolled HostAgent server in a spawned
OS process and connects from the parent process through an explicit `PrivateEndpoint`.
The processes perform a real mutual TLS handshake, enrolled certificate pinning, remote
health, and a host-local reserve operation.

The manifest and admission snapshot are created and registered inside the server process.
The client transmits only their identities. CI still binds the endpoint to 127.0.0.1, so
this is process-isolation evidence rather than a physical multi-machine LAN test.


## Live runtime identity / retained E2 invalidation tests

A stable runtime identity binds worker artifact/build, host OS identity, driver/runtime
identity, TensorMeld device identity, stable physical-device identity and a topology
fingerprint. Transient values such as timestamps and free memory are deliberately excluded
from the identity fingerprint.

Retained backend-readiness evidence is now schema v2 and embeds that exact runtime
identity. Portable tests verify deterministic identity fingerprints, duplicate-key/tamper
rejection, node/device continuity, driver/topology invalidation and exact-match reuse.

When a fresh bound device is still `observed` and all retained backend identities plus
the current runtime identity match exactly, validation returns a copied runtime observation
with only that device promoted to `ready`. It still returns
`reservation_created=false`, `qualified=false` and `executable=false`.

These tests use synthetic worker/driver/runtime strings. They do not prove any actual
CUDA/HIP driver, GPU, OS-specific backend or performance qualification.
