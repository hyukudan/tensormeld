# ADR-0021 — Private control transport reuses TLS and pins enrolled peers

Status: **accepted**  
Date: 2026-10-02

## Context

TensorMeld has an enrolled local agent and host-owned lease authority, but no remote
security boundary. A LAN is not trusted. Remote control needs encryption, mutual
authentication, bounded framing and replay/stale-connection rejection before any agent
operation can cross a network.

TensorMeld must not implement its own TLS cryptography.

## Decision

The private control transport uses Python's `ssl` module backed by OpenSSL.

Client and server context builders require:

- CA verification;
- a local certificate and private key supplied at runtime;
- server-side client-certificate verification;
- TLS 1.2 or newer;
- dedicated ALPN `tensormeld-control/1`.

After the TLS handshake, TensorMeld additionally pins the exact DER certificate SHA-256
of the expected enrolled peer. A channel is rejected when TLS version, cipher, ALPN,
certificate presence or fingerprint does not match.

The application framing layer carries at most 64 KiB canonical JSON records. Every frame
contains the protocol ID, connection epoch, monotonic sequence, one allowlisted agent
operation and an object body. Foreign epochs, stale/out-of-order sequences and
non-allowlisted operations fail closed.

This slice operates on an already-authenticated TLS socket. It does not yet create a
certificate-provisioned listener or claim a real mTLS handshake was exercised.

## Consequences

- TLS cryptography and certificate validation are delegated to maintained platform/OpenSSL
  implementations rather than reimplemented.
- Peer identity is stricter than ordinary CA validity because an exact certificate
  fingerprint is pinned after the handshake.
- A reconnect requires a new connection epoch, preventing old control frames from
  advancing the new connection.
- Tensor payload transport, public listeners and arbitrary execution are outside this
  protocol.
- Real provisioned-certificate handshake integration is the next security increment.
