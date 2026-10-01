# 04 — Topology, transport and security

## Network is a graph, not one advertised speed

Represent device endpoints, host interfaces, physical ports, directed links,
shared bottleneck groups, forwarding nodes and route policies. A physical USB4
cable, a logical stream and an IP interface are different resources. Multiple
interfaces on one host may share a PCIe or controller bottleneck.

The first supported topology is a workstation linked to a companion using normal
IP networking. The next adds a second companion and a faster companion-to-companion
link. A second direct companion link is optional. There is no requirement that the
workstation have USB4 or that one vendor's GPU-direct mechanism span all devices.

Nominal port rate, negotiated rate, useful host throughput and GPU-visible tensor
throughput must be separate observations. Routes must record relays and both local
GPU staging copies. Missing links do not imply an automatic direct connection.

The reference network path is portable TCP/IP. A secure remote path may wrap a
legacy local-only worker using an authenticated encrypted tunnel. GPU tensor
throughput must be measured through that exact wrapper, not through an unsecured
bypass. Advanced transports are optional capabilities; their absence must not
prevent ordinary Ethernet operation.

## Dual-link policy

Start with one qualified link. The second link may carry independent transfers or
striped large payloads only when the implementation supports it and concurrent
measurements demonstrate improvement. Small latency-sensitive transfers should
usually remain whole until data supports striping.

A sender must identify physical-link IDs, sharing groups, sequence, offsets and
reassembly bounds. Single-link and dual-link modes have distinct profiles. A second
link must not silently double memory slot counts or change correctness semantics.
The system must distinguish link failure from worker/state loss.

The M0 solver chooses one declared directed link per hop. It deliberately does not
implement multirail aggregation, contention scheduling or automatic path search.

## Buffer and completion contract

A future native transport interface must describe prepare-send, prepare-receive,
submit, poll/wait, cancel and health, with explicit buffer ownership. Required
completion distinctions are local submission, local buffer reusable, receive
complete, remote acknowledgement and semantic request commit. These are not
interchangeable.

Tensor metadata binds protocol version, peer identity, connection epoch, model and
plan hashes, request ID, generation/speculation round, stage/boundary ID, tensor
role, dtype, dimensions, payload size and sequence. Validate bounds, overflow,
expected shape and ownership before allocating or exposing a payload to a GPU.
Serialized pointers or platform-native struct layout must never define the wire ABI.

Host buffers may be copied, registered or GPU-visible according to qualified
capabilities. Direct streaming is not proof of zero-copy or GPU-direct support.
External-DMA visibility and GPU synchronization must be proven for the exact buffer
kind and drivers. A copy-based fallback remains available.

Weight transfer is a separate, resumable operation with shard hashes and disk/RAM
quotas. It must not block decode unexpectedly. A plan may run only when every
required weight shard and persistent-state allocation is prepared.

## Trust model

A local LAN is not automatically trusted. Risks include unsolicited pairing,
malicious protocol frames, unauthorized model extraction, prompt/log exposure,
arbitrary command execution through launch parameters, model path traversal,
cache poisoning, unbounded memory use and stale request replay.

The production agent must use explicit enrollment and mutually authenticated,
encrypted connections. Pairing creates node identities with revocation and rotation.
Auto-discovery only advertises reachability. It never grants execution permission.
A token in a plaintext connection is not adequate pairing security.

The first implementation may use manually provisioned certificates rather than a
complex discovery UI. Keys stay in OS-protected local storage, never source control.
An authenticated channel alone does not authorize arbitrary backend binaries.

Legacy inference RPC must bind to loopback and sit behind an approved authenticated
tunnel/proxy. The application must not expose it on all interfaces because a firewall
might exist. An optional lab escape hatch, if ever added, must be visibly unsafe and
cannot satisfy qualification. No unsafe remote listener exists in M0.

## Least authority

Agents run as ordinary users. Only approved worker hashes and allowlisted operations
are launchable. A peer cannot supply arbitrary command lines, shell fragments,
executables or paths outside configured model/cache roots. Model formats must be
parsed defensively; do not load arbitrary pickle or remote model code.

The product does not automatically alter BIOS, kernel modules, drivers, firewall,
network bonding or system power limits. Diagnostics report required setup and ask
for explicit administrator action outside the inference protocol.

Default HTTP binding is loopback. Remote API access requires explicit configuration
and authentication. Logs redact prompts by default and never include pairing keys.
Metrics are local; external telemetry is opt-in. Model bytes and user prompts remain
within the explicitly enrolled installation unless the user configures otherwise.

## Failure behavior

A worker restart invalidates its runtime state. The coordinator must fail affected
requests cleanly unless a checkpoint/replay mechanism has actually been implemented
and qualified. It must not promise seamless failover based on a redundant cable.
A transport reconnect begins a new epoch. Bounded retries cannot commit duplicate
tokens. Cancelling an in-flight tensor does not free a buffer until ownership returns.

## M0 diagnostic boundary

The included diagnostic opens a short-lived TCP listener only on IPv4 loopback,
uses bounded frames, verifies exact echo data and closes the listener after the
run. It accepts no remote address or commands. It is for protocol smoke testing,
not LAN throughput or production security. Remote profiling is intentionally a
separate milestone gated on authentication and resource controls.


## Multiple nodes, interfaces and shared resources

Use directed topology records, not fixed A/B/desktop fields. Host interfaces,
physical links and backend-reachable device paths are distinct entities. A switch
uplink, USB router or host memory bus may constrain several paths. Enforce bounded
per-peer connections/queues, fairness and total staging capacity as node count grows.
Discovery/profiling concurrency is bounded; adding N nodes does not launch unlimited
pairwise probes or stall active inference.

Manual endpoints support IPv4 and IPv6; address changes require authenticated
identity confirmation. The user can pin an interface/path or disable Wi-Fi for
bulk traffic. Wi-Fi is not forbidden by the model, but its measured availability,
latency tails and bandwidth govern admission; no stable wired performance is implied.
Automatic interface choice cannot route model data over a public or unenrolled peer.
Control and bulk traffic can have distinct paths without distinct trust policies.

Direct/relay/routed transfer capability is declared by the adapter. Physical
reachability never implies arbitrary tensor relaying. Relays consume CPU, buffers
and bandwidth even without model layers; they count toward participant resources
but not compute-node counts. Coordinator-to-worker and worker-to-worker traces must
identify their actual endpoint processes, copies and synchronization boundaries.

## Privacy, local API and update surface

Enrolled nodes that compute on data can access prompts, activations or weights as
required by that computation. Encryption in transit does not hide them from a
malicious authorized node. Per-model allowed-node policy and cache/export restrictions
are required; this is not confidential computing between mutually distrusting owners.

The local API checks origin/host and authentication requirements to resist browser
cross-origin and DNS-rebinding attacks. CORS is not authorization. Administrative
operations, model inference and node lifecycle have distinct scopes. Tool-calling
output is data for the client; it does not authorize the agent to execute shell code.

Approved releases identify source revision, artifact hash, license notices and update
channel. Hashes alone do not establish a publisher: artifact authorization/signature
verification and approved provenance are required for automatic installation.
Updates are staged and version compatibility is checked; active sessions are not
hot-swapped. Package signing/OS service setup is a release gate, not an M0 feature.
No installer silently changes drivers, clocks, firmware or privileged network settings.

Prompt/session retention is off by default. Diagnostic exports redact prompts,
credentials and optional host identifiers; the user previews the export. Cache and
session deletion policies include backups where managed, but must not promise secure
erasure from arbitrary SSDs. Secrets are never included in example configuration.
