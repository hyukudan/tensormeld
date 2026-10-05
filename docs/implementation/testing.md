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


## Native llama.cpp E3 correctness evaluator tests

The E3 evaluator consumes a pre-E3 native trial specification, its retained trial result,
an approved deterministic reference contract and stable runtime identities for every
placement device.

Only `execution_source=native-subprocess`, exit code zero and the exact retained trial
identities are accepted. Fixture/injected trial provenance fails closed. The reference
contract binds the exact trial-spec SHA, llama-cli SHA, ModelManifest SHA, placement SHA,
expected raw stdout SHA-256 and exact tested workload.

QualificationEvidence v2 extends the existing evidence contract with:
- candidate-plan SHA-256;
- placement SHA-256;
- trial-spec SHA-256;
- correctness-contract SHA-256;
- ordered runtime-identity SHA-256 values.

V1 parsing remains supported for historical/general evidence, but AcceptedExecutionBundle
now requires v2 for executable E3 authorization and compares the E3 plan/runtime identities
against the exact candidate and current per-device backend-readiness identities.

Portable tests intentionally use native-shaped synthetic records rather than claiming a
native target-host run. They cover exact E3 emission, plan/runtime applicability,
fixture/injected provenance rejection, wrong stdout contract, worker/device identity
changes, trial/spec/placement tampering and workload mismatch.

A passing portable E3 test proves evaluator semantics only. It is not evidence that
llama.cpp, a GGUF model, CUDA/HIP kernels or a real GPU executed in hosted CI.


## Target-host qualification-chain tests

The target-host orchestrator launches nothing. It receives already-produced probe,
binding, retained backend E2, runtime identity, pre-E3 placement, native-trial and
correctness-reference artifacts and recomputes/validates their relationships.

The supplied bound result is compared against a freshly recomputed
`bind_llamacpp_probe` result, and the pre-E3 placement is independently regenerated
from Config, PlanningInput, planner candidate, adapter, current native binding and complete
GGUF index.

Every compute device must have exact retained E2 evidence, expected test artifact identity
and current RuntimeIdentity. Each E2 record is revalidated and its narrow
`observed -> ready` promotion is combined into one runtime observation.

The orchestrator reruns E3 correctness evaluation from the retained native trial and
reference contract rather than trusting a supplied "qualified" flag. Its output is a
deterministic handoff containing exact runtime-manifest identity requirements and a
combined ready observation, but always keeps reservation, launch authorization and
execution false.

The handoff explicitly compares the E3-tested workload against the target profile workload.
A narrower E3 trial is reported as a mismatch; it is never silently widened into profile
qualification.

Portable tests use native-shaped fixtures only and cover stale bound records, changed
runtime identity, fixture trial provenance, probe/trial binary mismatch and incomplete E2
coverage. This is orchestration-contract evidence, not real target-host qualification.


## Native runtime-manifest collector tests

The collector consumes the deterministic target-host qualification handoff and two
separate evidence inputs:

1. runtime observations: per-device operator observations plus one memory record per
   physical pool;
2. operator requirements: the independent required-operator set for the exact
   handoff/model/plan/placement.

Separating requirements from observations prevents a measurement from declaring a small
required set (for example only ADD) and then self-reporting complete coverage.

Both contracts are fingerprint-bound to the intact target-host handoff. Device
measurements additionally carry the exact runtime-identity SHA expected for that device.
The collector verifies device node/backend identity against Config and requires every
compute device's physical pool to appear in the memory measurement. Duplicate pool
records fail closed so shared/unified memory cannot be counted once per logical device.

The existing RuntimeModelManifest parser remains the final canonical validator for
profile workload, physical-pool capacity bounds, preparation peak >= steady peak, device
identity and operator coverage.

Manifest provenance is promoted to `native-adapter` only when both the runtime
measurement and operator-requirement contracts are native. Mixed or fixture provenance
produces a fixture manifest and can never be admission-ready native input.

Portable tests cover fixture manifest collection, missing compute pools, stale runtime
identity, worker mismatch, independent/stale operator requirements, incomplete operator
coverage, workload mismatch, duplicate pools, tampered handoff fingerprints and mixed
provenance. These remain software-contract tests only.


## Target-host admission orchestrator tests

The admission orchestrator consumes an intact admission-ready native runtime-manifest
collection plus the exact qualification handoff/candidate/E3/readiness tuple.

The current llama.cpp path is single-node, so the orchestrator requires exactly one
compute node and passes that node as the explicit local admission authority.

The orchestration sequence is fixed:

1. validate collection/handoff/manifest/plan identities;
2. reserve exact manifest physical-pool preparation peaks;
3. require a second admission snapshot with a different observation ID;
4. launch-recheck the existing lease;
5. build AcceptedExecutionBundle from the same E3 v2, current readiness, manifest and
   launch-admitted lease.

If launch recheck rejects, or AcceptedExecutionBundle construction fails after
reservation, the lease is released deterministically.

Portable tests cover successful reserve→fresh-recheck→bundle construction, rejection of
reused observation IDs, rollback after launch shortfall, non-native/non-ready collection,
tampered collection fingerprints and stale candidate-plan identity.

The resulting bundle may be execution-authorized, but the orchestrator always reports
`inference_started=false` and `real_model_inference=false`; it never launches a model.


## Lease-bound native admitted session tests

The managed native session sits after TargetHostAdmissionResult and before any model-facing
API. It validates that the admission result is intact and execution-authorized, the
backend belongs to the exact AcceptedExecutionBundle, and the referenced lease is still
active in state `launched` with matching lease/runtime-manifest identity.

The session executes synchronously through the existing bounded whole-block backend
interface. It owns lease cleanup:
- completion releases the lease;
- backend failure releases the lease;
- cancellation before run releases immediately without invoking the backend;
- cancellation while a backend call is already in flight becomes `CANCEL_REQUESTED`
  and defers lease release until the run reaches a terminal boundary;
- explicit release is idempotent.

The in-flight rule is important because the current subprocess backend does not yet expose
a portable process-kill handle through the whole-block interface; releasing memory while
a worker could still be running would be unsafe.

Portable tests use a deterministic backend plus a real LocalAdmissionController and a
real TargetHostAdmissionResult produced through the admission orchestrator. The separate
native-worker suite already covers the real fixture subprocess boundary. No real llama.cpp
or GPU process is launched by these lifecycle tests.


## Managed llama-server process lifecycle tests

TensorMeld now owns a persistent server-process primitive separately from the one-shot
llama-cli E3 trial path.

The launch spec is bound to the pinned llama.cpp revision, approved llama-server artifact,
optional approved launcher artifact used only for portable fixture execution, exact
ModelManifest and approved GGUF, exact post-E3 placement translation, and loopback host
with one explicit server slot.

No caller-supplied argv or environment extension is accepted. Inherited LLAMA_ARG_*
variables are removed before launch.

Portable tests start a real Python HTTP fixture process on Windows/Linux, wait for /health,
verify bounded stderr diagnostics, detect early process exit, stop cleanly, and on POSIX
prove terminate→kill fallback with a stubborn process.

This is process-lifecycle evidence only. The fixture does not execute llama.cpp or a GGUF
model.

A deliberate identity boundary remains: native E3 currently binds llama-cli, while the
persistent server is a distinct llama-server artifact. TensorMeld will not treat them as
the same worker merely because they came from the same source tree. A future build/package
identity must bind both artifacts before persistent server execution can inherit E3 and
lease authorization.


## llama.cpp build/package identity and admitted-server bridge tests

TensorMeld now represents a llama.cpp build package separately from individual executable
identity. The package requires two distinct artifact hashes for llama-cli and llama-server,
identical observed build metadata, the pinned source revision, and a deterministic set of
backend-library artifact hashes.

Package identity does not assert request-semantic equivalence. It only proves that the
qualified CLI artifact and persistent server artifact are sibling artifacts in the exact
recorded build package.

The admitted-server bridge then requires:
- intact TargetHostAdmissionResult;
- exact AcceptedExecutionBundle;
- package llama-cli SHA == the bundle's E3-qualified worker artifact;
- server launch spec SHA == package llama-server SHA;
- exact model/source revision identity;
- live launched lease with matching lease/runtime-manifest fingerprint.

Portable tests start a real fixture HTTP server under a real launch-admitted
LocalAdmissionController lease. The lease stays active while the server is running and is
released only after process stop is confirmed. Early server exit also cleans up the lease.

The bridge deliberately reports both
`server_semantic_equivalence_qualified=false` and
`inference_request_authorized=false`. Same-build provenance is not an E4 correctness
claim.


## llama-cli ↔ llama-server E4 equivalence tests

The E4 gate compares one deterministic request through the one-shot CLI path and the
same-build persistent server path.

The E4 spec binds:
- exact llama.cpp build package;
- ModelManifest;
- AcceptedExecutionBundle;
- pre-E3 qualification placement and post-E3 execution placement;
- exact semantic equality of block ownership, buffer types, override-tensor value and
  placement argv;
- CLI trial spec and persistent server launch spec;
- prompt, context, output length, seed 0 and temperature 0;
- closed non-streaming server body with cache_prompt disabled.

The server result is fingerprinted and bounded. The evaluator compares the SHA-256 of the
CLI stdout bytes against the SHA-256 of the server's completion content bytes.

Portable CI launches a real CLI fixture subprocess and a real HTTP server fixture process.
They intentionally receive fixture provenance. Exact output equality therefore proves the
comparison protocol but does not produce native E4 qualification.

Request authorization requires E4 evidence with:
- cli_execution_source=native-subprocess;
- server_execution_source=native-server-subprocess;
- exact output equality;
- intact E4 spec/evidence fingerprints;
- package/model/bundle/pre/post-placement/trial/server-spec identities matching the
  admitted server binding.

Only then can a derived binding report
`server_semantic_equivalence_qualified=true` and
`inference_request_authorized=true`. The gate still reports
`real_model_inference=false` until an actual request is executed through that authorized
binding.
