from __future__ import annotations

import hashlib
import os
from pathlib import Path
import socket
import ssl
import threading
import unittest

from tensormeld.agent import Enrollment
from tensormeld.private_transport import (
    PrivateControlChannel,
    bind_enrollment_to_peer_certificate,
    build_client_context,
    build_server_context,
    tls_material,
    validate_tls_peer,
)


def cert_sha256(path: Path) -> str:
    pem = path.read_text(encoding="utf-8")
    der = ssl.PEM_cert_to_DER_cert(pem)
    return hashlib.sha256(der).hexdigest()


@unittest.skipUnless(
    os.environ.get("TENSORMELD_REAL_MTLS_TEST") == "1",
    "real mTLS integration requires ephemeral provisioned certificates",
)
class RealMTLSLoopbackIntegrationTests(unittest.TestCase):
    def test_real_mutual_tls_handshake_and_control_exchange(self):
        ca = Path(os.environ["TENSORMELD_MTLS_CA"])
        server_cert = Path(os.environ["TENSORMELD_MTLS_SERVER_CERT"])
        server_key = Path(os.environ["TENSORMELD_MTLS_SERVER_KEY"])
        client_cert = Path(os.environ["TENSORMELD_MTLS_CLIENT_CERT"])
        client_key = Path(os.environ["TENSORMELD_MTLS_CLIENT_KEY"])

        server_ctx = build_server_context(
            tls_material(ca_file=ca, cert_file=server_cert, key_file=server_key)
        )
        client_ctx = build_client_context(
            tls_material(ca_file=ca, cert_file=client_cert, key_file=client_key)
        )

        server_enrollment = Enrollment("server-enrollment", "server-node", "server-key")
        client_enrollment = Enrollment("client-enrollment", "client-node", "client-key")
        server_binding = bind_enrollment_to_peer_certificate(
            server_enrollment, certificate_sha256=cert_sha256(server_cert)
        )
        client_binding = bind_enrollment_to_peer_certificate(
            client_enrollment, certificate_sha256=cert_sha256(client_cert)
        )

        errors: list[BaseException] = []
        received: list[dict] = []

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(10)
            port = listener.getsockname()[1]

            def server() -> None:
                try:
                    raw, _ = listener.accept()
                    with raw:
                        raw.settimeout(5)
                        with server_ctx.wrap_socket(raw, server_side=True) as tls_sock:
                            validate_tls_peer(
                                tls_sock,
                                expected_peer_certificate_sha256=client_binding.certificate_sha256,
                                expected_enrollment=client_enrollment,
                                peer_binding=client_binding,
                            )
                            channel = PrivateControlChannel(
                                tls_sock,
                                expected_peer_certificate_sha256=client_binding.certificate_sha256,
                                connection_epoch="integration-epoch",
                            )
                            received.append(channel.receive())
                            channel.send("health", {"server": "ok"})
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=server, daemon=True)
            thread.start()
            with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
                raw.settimeout(5)
                with client_ctx.wrap_socket(raw, server_hostname="localhost") as tls_sock:
                    validate_tls_peer(
                        tls_sock,
                        expected_peer_certificate_sha256=server_binding.certificate_sha256,
                        expected_enrollment=server_enrollment,
                        peer_binding=server_binding,
                    )
                    channel = PrivateControlChannel(
                        tls_sock,
                        expected_peer_certificate_sha256=server_binding.certificate_sha256,
                        connection_epoch="integration-epoch",
                    )
                    channel.send("describe", {"client": "ok"})
                    response = channel.receive()
                    self.assertEqual(response["operation"], "health")
                    self.assertEqual(response["body"], {"server": "ok"})

            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors, errors)
            self.assertEqual(received[0]["operation"], "describe")
            self.assertEqual(received[0]["body"], {"client": "ok"})


if __name__ == "__main__":
    unittest.main()
