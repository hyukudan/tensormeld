# End-to-end mTLS remote-agent dispatch validation — 2026-10-02

The dedicated real-mTLS integration now drives RemoteAgentClient/RemoteAgentDispatcher
through the actual ephemeral-certificate mutually authenticated TLS socket.

The server constructs a HostAgent and locally registers the exact runtime manifest and
admission snapshot. The client performs a remote health call and then a reserve request
containing only the lease ID, manifest SHA-256 and observation ID. The server resolves
those references locally and creates the lease in its own HostAgent admission controller.

The integration verifies request/response correlation, enrolled certificate identity,
bounded allowlisted operations and host-local object authority in one path.

This remains loopback-only evidence. No two-machine private LAN, tensor transfer, GPU
execution, performance claim or distributed inference is established.

GitHub Actions result: pending for the implementation PR.
