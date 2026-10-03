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


## Accepted execution bundle and reference whole-block adapter tests

The first executable reference path recomputes the planner candidate hash using the same
canonical identity as the planner, reruns exact adapter representability, requires an
applicable E3+ model qualification record, checks the qualification worker artifact
against the runtime manifest, requires one current backend-ready proof per compute device,
and one launch-admitted lease per compute node.

Launch-admission outputs now carry node, config and runtime-manifest identities; retained
backend-readiness applicability exposes config/device/runtime-identity fingerprints. The
execution bundle rejects cross-config or cross-manifest mixing.

Portable tests exercise an actual reference session that iterates the immutable whole-block
segments in order through a deterministic in-process backend, plus cancellation and
deterministic release. Negative tests cover plan tampering, wrong E3 worker identity,
missing device readiness and missing node admission.

This proves orchestration/lifecycle semantics only. The reference run explicitly reports
`real_model_inference=false`; it does not execute GGUF tensors, CUDA/HIP kernels or
distributed model inference.


## Revision-pinned subprocess whole-block worker tests

The first native-process foundation validates the exact launcher/program artifacts before
launch, requires the accepted bundle's worker artifact and engine revision to match the
process actually invoked, and constructs a closed argv shape with `shell=False`.

Segment execution sends one bounded canonical JSON request over stdin. The request binds
worker protocol, adapter/revision, accepted-bundle SHA, exact device/unit segment,
launcher/program artifact fingerprints and a bounded base64 payload. The response must
echo the exact request/segment/bundle/worker identities before its output is accepted.

Portable CI executes a real subprocess on both Windows and Linux: the local Python
interpreter is the hashed launcher and a separate hashed fixture program implements the
worker protocol. Negative tests cover artifact mismatch, engine-revision mismatch,
bundle mismatch, non-zero exit, response tampering and a worker attempting to self-claim
`real_model_inference=true`.

This proves the local process boundary and protocol, not GGUF/model inference. The fixture
worker only transforms bytes and every accepted response remains
`real_model_inference=false`.


## Pinned llama.cpp model-aware placement shim tests

The initial model-aware shim is intentionally narrower than llama.cpp itself. It accepts
only single-node accepted bundles whose unit IDs are exactly contiguous `blk.N`
transformer blocks.

The shim consumes a complete `gguf-index-v1` already produced by the pinned upstream
GGUF reader. The index fingerprint, architecture and tensor count must match the exact
ModelManifest, and every real tensor name under `blk.*` is inspected to derive the
complete block set. The accepted bundle must cover that block set exactly: missing,
extra, reordered, duplicate or malformed block namespaces fail closed.

Placement binding separately records the current native device-binding SHA, exact
llama.cpp engine device name and primary buffer type for every compute device. The
translator verifies those engine names against the fresh bound result. The first shim
allows only one compute node and rejects llama.cpp RPC devices and non-primary buffer
types so it cannot bypass TensorMeld's authenticated remote-control boundary.

For each real block, TensorMeld generates its own anchored
`^blk\.N\..*=<buffer>` override. User regexes never enter the translator. The
generated placement fragment disables auto-fit and uses the exact approved local engine
device list.

These tests validate deterministic static placement translation only. They do not load a
GGUF in llama.cpp, prove KV/compute placement, execute CUDA/HIP kernels, or establish
model correctness/performance.


## Pre-E3 llama.cpp qualification placement and native trial tests

TensorMeld now has a pre-E3 placement path specifically to avoid a qualification
bootstrap cycle. The path starts from an exact planner candidate, recomputes its canonical
plan hash, reruns adapter representability, verifies the current native device binding and
complete GGUF block coverage, then produces the same closed llama.cpp placement fragment
without requiring an AcceptedExecutionBundle.

The native trial layer accepts only one exact local GGUF file in this first increment.
Its file name, size and SHA-256 must match the ModelManifest. The llama-cli artifact is
also pre-approved by SHA-256.

The generated argv is closed and includes the exact GGUF path and placement fragment plus
bounded deterministic trial controls: context size, predict count, fixed prompt argument,
seed 0, temperature 0, simple IO, single-turn, no prompt display, no timings and color off.
No caller-supplied extra argv or remote RPC is supported.

The default subprocess runner strips inherited `LLAMA_ARG_*` variables so environment
configuration cannot silently override the generated command line. Portable CI uses a
fixture CLI process to verify the real subprocess boundary. An injected runner cannot be
labeled `native-subprocess`.

A successful native process with stdout is at most trial evidence (reported as E2.5 here);
it remains `qualified=false`, `real_model_inference=false` and
`executable=false` until a separate native E3 correctness evaluator exists.
