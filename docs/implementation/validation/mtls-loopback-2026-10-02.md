# Real mTLS loopback validation — 2026-10-02

A dedicated Linux CI job generates an ephemeral CA, localhost server certificate and
client certificate using OpenSSL. It then starts a TCP listener only on 127.0.0.1,
performs a real Python ssl/OpenSSL mutual TLS handshake, verifies both certificate chains,
checks the pinned peer certificate fingerprints against enrolled node identities and
exchanges bounded allowlisted control frames in both directions.

No certificate or private key is stored in Git. The job workspace is ephemeral.

This is real encrypted/authenticated loopback evidence. It is not evidence of a private
LAN deployment, network performance, tensor transport, GPU execution or distributed
inference.

GitHub Actions result: pending for the implementation PR.
