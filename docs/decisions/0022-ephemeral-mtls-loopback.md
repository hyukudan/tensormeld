# ADR-0022 — Real mTLS integration uses ephemeral runtime certificates

Status: **accepted**  
Date: 2026-10-02

## Decision

TensorMeld validates the private control channel with a real mutual TLS handshake on
loopback using certificates generated ephemerally during Linux CI by OpenSSL.

No test CA, certificate or private key is committed to the repository. The integration
creates a one-job CA, server certificate for localhost and client certificate, performs
mutual certificate verification, checks ALPN/TLS policy, pins the exact peer certificate
SHA-256 and binds that fingerprint to the expected enrolled node identity.

Only bounded allowlisted control frames are exchanged after the authenticated handshake.

## Consequences

- CI now distinguishes mocked TLS contract coverage from a real TLS handshake.
- Repository history contains no reusable private key fixture.
- The result proves loopback mTLS mechanics, not private-LAN reachability or performance.
- Windows remains portable contract coverage for this slice.
- Private-LAN endpoint policy and remote agent method dispatch remain later increments.
