# Private endpoint and remote-agent dispatch validation — 2026-10-02

Implementation slice: endpoint policy plus bounded remote HostAgent method dispatch.

## Contract

Private endpoints bind node/enrollment identity, IP literal, port and certificate
fingerprint. Only private or loopback addresses are accepted.

Remote calls use unique request IDs and the existing allowlisted HostAgent operations.
Reserve/launch-recheck calls refer to locally registered runtime manifests and admission
snapshots by exact identity. Unknown references and unexpected request fields are rejected.

No remote request can supply shell commands, executable paths, model files, runtime
manifest objects, observation objects or tensor payloads.

## Portable coverage

Tests cover private endpoint acceptance/rejection, enrollment/certificate binding,
local-object-only reserve dispatch, unexpected-field rejection, duplicate request IDs and
bounded lifecycle calls.

These tests do not establish a real multi-machine private LAN or distributed inference.

GitHub Actions result: PR #8, workflow `Portable Python tests`, run #94 (37033679451) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. This validates the bounded remote-control contract and preserves the previously validated real loopback mTLS path; it is not multi-machine LAN or distributed-inference evidence.
