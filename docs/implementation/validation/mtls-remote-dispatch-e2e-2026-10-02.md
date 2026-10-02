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

GitHub Actions result: PR #9, workflow `Portable Python tests`, run #99 (37051535623) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job exercising end-to-end RemoteAgentClient/Dispatcher health and reserve calls. The first run (#97) exposed only an unittest module-discovery path issue; the dedicated job was switched to unittest discovery before the successful run. Evidence remains loopback-only.
