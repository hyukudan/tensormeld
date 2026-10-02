# Private TLS control-channel validation — 2026-10-02

Implementation slice: authenticated/encrypted control-channel foundation.

## Contract

The new transport module builds strict client/server TLS contexts using Python
`ssl`/OpenSSL. Policy requires CA verification, a local certificate/key pair,
server-side client-certificate verification, TLS 1.2+ and dedicated ALPN.

A connected TLS socket is accepted only when it exposes an allowed TLS version, a
negotiated cipher, the expected ALPN and a peer certificate whose DER SHA-256 exactly
matches the enrolled/pinned value.

Control frames are canonical JSON capped at 64 KiB and bind protocol, connection epoch,
monotonic sequence and one existing allowlisted agent operation. Replay, wrong epoch,
oversize framing and non-allowlisted operations are rejected.

## Portable coverage

Portable tests use injected TLS-socket fixtures and mocked context constructors. They
exercise TLS policy construction, certificate pinning, ALPN/version enforcement, frame
bounds, send sequencing and receive replay/epoch/operation validation.

These tests do **not** perform a real TLS certificate handshake or private-LAN exchange.
No encryption performance, CUDA/HIP, Windows GPU execution, tensor transport or
distributed inference is claimed.

GitHub Actions result: pending for the implementation PR.
