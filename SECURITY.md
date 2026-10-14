# Security status

TensorMeld is a pre-alpha control-plane prototype, not a qualified production security
boundary. The local `HostAgent` implements explicit enrollment, authenticated capability
envelopes and a fixed operation allowlist; it opens no listener itself. Separate private
endpoint helpers implement bounded control dispatch over mutual TLS with CA verification,
enrolled certificate pinning, dedicated ALPN and epoch/sequence replay rejection.

Real loopback and separate-process tests exercise this control stack. They do not establish
a secure multi-machine deployment, GPU execution or distributed model inference. Persistent
OS-protected secret storage, rotation/revocation and deployment hardening remain separate
gates; see the [current status](docs/implementation/status.md) and
[security contract](docs/specification/04-network-security.md).

The short-lived `bench-loopback` diagnostic binds exclusively to IPv4 loopback. It is
separate from authenticated control and must not be repurposed as a network service.

Untrusted scenario JSON has strict keys, finite numeric bounds, unique identifiers,
size limits and no executable expressions. Local inventory uses fixed read-only
commands with shell=False and timeouts. Hardware/model names do not become commands.

Private-LAN deployments must meet [the network/security contract](docs/specification/04-network-security.md)
with explicit endpoint and enrollment configuration before testing. Legacy
inference RPC requires an authenticated private wrapper; never publish it directly.
Do not commit model weights, prompts, diagnostic inventories, credentials or TLS keys.
Use the public repository issue tracker only for non-secret findings. Never post credentials,
private inventories, prompts, model data, network details that should remain private, or other secrets.

## Optional local GGUF inspection

`inspect-gguf` is a trusted-local-file development command, not a sandbox. Upstream
metadata parsing precedes our index-validation limits; a malicious file could consume
resources. Do not expose this path through a remote agent until process isolation and
resource limits are implemented. Index digests are not full checkpoint-content digests.

Third-party versions and source notices are recorded under `third_party/`. A reviewed
root license is not a transitive security audit. Experimental native RPC is not a safe
LAN boundary simply because its code is open source.
