# Separate-process private endpoint validation — 2026-10-03

The real-mTLS CI integration starts the HostAgent server in a spawned OS process and uses
an explicit PrivateEndpoint from the parent/client process.

The server process independently builds its HostAgent, runtime manifest, admission
snapshot and local object registry. Client and server perform real mutual TLS,
certificate/enrollment binding and bounded remote dispatch. The client issues health and
reserve; the resulting lease exists only inside the server process.

The endpoint remains 127.0.0.1 in CI. This validates process separation and endpoint
semantics, not a physical multi-machine private LAN.

GitHub Actions result: pending for the implementation PR.
