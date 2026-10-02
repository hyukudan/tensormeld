from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from tensormeld.private_transport import (
    CONTROL_ALPN,
    CONTROL_PROTOCOL,
    MAX_CONTROL_FRAME,
    PrivateControlChannel,
    TLSMaterial,
    build_client_context,
    build_server_context,
    receive_control_frame,
    send_control_frame,
    validate_tls_peer,
)
from tensormeld.schema import ValidationError


class FakeTLSSocket:
    def __init__(self, cert=b"peer-cert", incoming=b""):
        self.cert = cert
        self.incoming = bytearray(incoming)
        self.sent = bytearray()

    def sendall(self, data):
        self.sent.extend(data)

    def recv(self, size):
        if not self.incoming:
            return b""
        data = bytes(self.incoming[:size])
        del self.incoming[:size]
        return data

    def version(self):
        return "TLSv1.3"

    def cipher(self):
        return ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)

    def getpeercert(self, binary_form=False):
        return self.cert if binary_form else {"subject": ()}

    def selected_alpn_protocol(self):
        return CONTROL_ALPN


def encoded_frame(sequence=0, epoch="epoch", operation="health", body=None):
    payload = json.dumps({
        "protocol": CONTROL_PROTOCOL,
        "epoch": epoch,
        "sequence": sequence,
        "operation": operation,
        "body": body or {},
    }, sort_keys=True, separators=(",", ":")).encode()
    return struct.pack("!I", len(payload)) + payload


class PrivateTransportTests(unittest.TestCase):
    def test_peer_certificate_is_exactly_pinned(self):
        sock = FakeTLSSocket()
        sha = hashlib.sha256(sock.cert).hexdigest()
        result = validate_tls_peer(
            sock, expected_peer_certificate_sha256=sha
        )
        self.assertEqual(result["peer_certificate_sha256"], sha)
        with self.assertRaises(ValidationError):
            validate_tls_peer(
                sock, expected_peer_certificate_sha256="0" * 64
            )

    def test_alpn_and_tls_version_are_mandatory(self):
        sock = FakeTLSSocket()
        sha = hashlib.sha256(sock.cert).hexdigest()
        sock.selected_alpn_protocol = lambda: None
        with self.assertRaises(ValidationError):
            validate_tls_peer(sock, expected_peer_certificate_sha256=sha)
        sock.selected_alpn_protocol = lambda: CONTROL_ALPN
        sock.version = lambda: "TLSv1.1"
        with self.assertRaises(ValidationError):
            validate_tls_peer(sock, expected_peer_certificate_sha256=sha)

    def test_bounded_control_frame_roundtrip(self):
        sender = FakeTLSSocket()
        send_control_frame(sender, {
            "protocol": CONTROL_PROTOCOL,
            "epoch": "epoch",
            "sequence": 0,
            "operation": "health",
            "body": {"ok": True},
        })
        receiver = FakeTLSSocket(incoming=bytes(sender.sent))
        frame = receive_control_frame(receiver)
        self.assertEqual(frame["body"], {"ok": True})

    def test_oversized_frame_is_rejected_before_payload_read(self):
        incoming = struct.pack("!I", MAX_CONTROL_FRAME + 1)
        with self.assertRaises(ValidationError):
            receive_control_frame(FakeTLSSocket(incoming=incoming))

    def test_channel_rejects_replay_wrong_epoch_and_non_allowlisted_operation(self):
        cert = b"peer"
        sha = hashlib.sha256(cert).hexdigest()
        sock = FakeTLSSocket(
            cert=cert,
            incoming=encoded_frame(0) + encoded_frame(0),
        )
        channel = PrivateControlChannel(
            sock,
            expected_peer_certificate_sha256=sha,
            connection_epoch="epoch",
        )
        self.assertEqual(channel.receive()["sequence"], 0)
        with self.assertRaises(ValidationError):
            channel.receive()

        sock2 = FakeTLSSocket(cert=cert, incoming=encoded_frame(0, epoch="old"))
        channel2 = PrivateControlChannel(
            sock2,
            expected_peer_certificate_sha256=sha,
            connection_epoch="new",
        )
        with self.assertRaises(ValidationError):
            channel2.receive()

        sock3 = FakeTLSSocket(cert=cert, incoming=encoded_frame(0, operation="shell"))
        channel3 = PrivateControlChannel(
            sock3,
            expected_peer_certificate_sha256=sha,
            connection_epoch="epoch",
        )
        with self.assertRaises(ValidationError):
            channel3.receive()

    def test_send_sequence_is_monotonic_and_allowlisted(self):
        sock = FakeTLSSocket(cert=b"peer")
        channel = PrivateControlChannel(
            sock,
            expected_peer_certificate_sha256=hashlib.sha256(b"peer").hexdigest(),
            connection_epoch="epoch",
        )
        self.assertEqual(channel.send("health", {}), 0)
        self.assertEqual(channel.send("describe", {}), 1)
        with self.assertRaises(ValidationError):
            channel.send("shell", {})

    def test_tls_context_builders_require_mutual_auth_policy(self):
        fake_client = Mock()
        fake_server = Mock()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("ca.pem", "cert.pem", "key.pem"):
                (root / name).write_text("fixture", encoding="utf-8")
            material = TLSMaterial(root / "ca.pem", root / "cert.pem", root / "key.pem")

            with patch("tensormeld.private_transport.ssl.create_default_context", return_value=fake_client):
                context = build_client_context(material)
                self.assertIs(context, fake_client)
                self.assertEqual(fake_client.verify_mode, __import__("ssl").CERT_REQUIRED)
                fake_client.load_cert_chain.assert_called_once()
                fake_client.set_alpn_protocols.assert_called_once_with([CONTROL_ALPN])

            with patch("tensormeld.private_transport.ssl.SSLContext", return_value=fake_server):
                context = build_server_context(material)
                self.assertIs(context, fake_server)
                self.assertEqual(fake_server.verify_mode, __import__("ssl").CERT_REQUIRED)
                fake_server.load_verify_locations.assert_called_once()
                fake_server.load_cert_chain.assert_called_once()
                fake_server.set_alpn_protocols.assert_called_once_with([CONTROL_ALPN])


if __name__ == "__main__":
    unittest.main()
