# Security status

M0.2 remains offline except for a short-lived diagnostic listener bound exclusively to
127.0.0.1. There is no remote agent, execution service or authentication implementation.
Do not expose or repurpose the diagnostic as a network service.

Untrusted scenario JSON has strict keys, finite numeric bounds, unique identifiers,
size limits and no executable expressions. Local inventory uses fixed read-only
commands with shell=False and timeouts. Hardware/model names do not become commands.

Future agents must meet docs/specification/04-network-security.md before LAN testing. Legacy
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
