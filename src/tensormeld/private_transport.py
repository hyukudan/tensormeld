"""Authenticated private control-channel foundation over Python ssl/OpenSSL.

This module does not create a public listener and does not transport tensor payloads.
TLS contexts require CA verification and client certificates. A connected TLS socket is
accepted only after TLS version, ALPN and exact peer-certificate SHA-256 pinning checks.

Control frames are bounded JSON records with connection epoch + monotonic sequence.
No shell, arbitrary executable, or peer-supplied command line is part of this protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import ssl
import struct
from typing import Any, Protocol

from .agent import ALLOWED_OPERATIONS, Enrollment
from .schema import ValidationError, text

CONTROL_PROTOCOL = "tensormeld/private-control-v1"
CONTROL_ALPN = "tensormeld-control/1"
MAX_CONTROL_FRAME = 64 * 1024
_HEADER = struct.Struct("!I")
_TLS_VERSIONS = {"TLSv1.2", "TLSv1.3"}


@dataclass(frozen=True)
class TLSMaterial:
    ca_file: Path
    cert_file: Path
    key_file: Path


@dataclass(frozen=True)
class EnrolledPeerTLSBinding:
    enrollment_id: str
    node_id: str
    certificate_sha256: str


def bind_enrollment_to_peer_certificate(
    enrollment: Enrollment,
    *,
    certificate_sha256: str,
) -> EnrolledPeerTLSBinding:
    return EnrolledPeerTLSBinding(
        enrollment_id=enrollment.enrollment_id,
        node_id=enrollment.node_id,
        certificate_sha256=_sha256_hex(certificate_sha256, "certificate_sha256"),
    )


class TLSSocketLike(Protocol):
    def sendall(self, data: bytes) -> None: ...
    def recv(self, size: int) -> bytes: ...
    def version(self) -> str | None: ...
    def cipher(self) -> tuple[str, str, int] | None: ...
    def getpeercert(self, binary_form: bool = False) -> Any: ...
    def selected_alpn_protocol(self) -> str | None: ...


def _existing_file(path: str | Path, where: str) -> Path:
    p = Path(path).expanduser().resolve(strict=True)
    if not p.is_file():
        raise ValidationError(f"{where}: expected regular file")
    return p


def tls_material(
    *,
    ca_file: str | Path,
    cert_file: str | Path,
    key_file: str | Path,
) -> TLSMaterial:
    return TLSMaterial(
        _existing_file(ca_file, "ca_file"),
        _existing_file(cert_file, "cert_file"),
        _existing_file(key_file, "key_file"),
    )


def build_client_context(material: TLSMaterial) -> ssl.SSLContext:
    context = ssl.create_default_context(
        purpose=ssl.Purpose.SERVER_AUTH,
        cafile=str(material.ca_file),
    )
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_cert_chain(
        certfile=str(material.cert_file),
        keyfile=str(material.key_file),
    )
    context.set_alpn_protocols([CONTROL_ALPN])
    return context


def build_server_context(material: TLSMaterial) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(material.ca_file))
    context.load_cert_chain(
        certfile=str(material.cert_file),
        keyfile=str(material.key_file),
    )
    context.set_alpn_protocols([CONTROL_ALPN])
    return context


def _sha256_hex(value: str, where: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValidationError(f"{where}: expected SHA-256 hex")
    return normalized


def validate_tls_peer(
    sock: TLSSocketLike,
    *,
    expected_peer_certificate_sha256: str,
    expected_enrollment: Enrollment | None = None,
    peer_binding: EnrolledPeerTLSBinding | None = None,
) -> dict[str, Any]:
    expected = _sha256_hex(
        expected_peer_certificate_sha256,
        "expected_peer_certificate_sha256",
    )
    version = sock.version()
    if version not in _TLS_VERSIONS:
        raise ValidationError("control channel requires TLS 1.2 or TLS 1.3")
    cipher = sock.cipher()
    if not cipher or not cipher[0]:
        raise ValidationError("TLS control channel has no negotiated cipher")
    if sock.selected_alpn_protocol() != CONTROL_ALPN:
        raise ValidationError("TLS control channel ALPN mismatch")
    der = sock.getpeercert(binary_form=True)
    if not isinstance(der, (bytes, bytearray)) or not der:
        raise ValidationError("TLS peer did not present a certificate")
    observed = hashlib.sha256(bytes(der)).hexdigest()
    if observed != expected:
        raise ValidationError("TLS peer certificate fingerprint mismatch")
    if expected_enrollment is not None or peer_binding is not None:
        if expected_enrollment is None or peer_binding is None:
            raise ValidationError("TLS enrollment validation requires both enrollment and binding")
        if (
            peer_binding.enrollment_id != expected_enrollment.enrollment_id
            or peer_binding.node_id != expected_enrollment.node_id
            or peer_binding.certificate_sha256 != observed
        ):
            raise ValidationError("TLS peer certificate is not bound to expected enrollment")
    return {
        "transport": "tls",
        "tls_version": version,
        "cipher": cipher[0],
        "alpn": CONTROL_ALPN,
        "peer_certificate_sha256": observed,
        "mutual_auth_required_by_context": True,
    }


def _recv_exact(sock: TLSSocketLike, size: int) -> bytes:
    if not 0 <= size <= MAX_CONTROL_FRAME:
        raise ValidationError("control receive size exceeds frame bound")
    out = bytearray()
    while len(out) < size:
        chunk = sock.recv(min(16384, size - len(out)))
        if not chunk:
            raise EOFError("TLS peer closed before complete control frame")
        out.extend(chunk)
    return bytes(out)


def send_control_frame(sock: TLSSocketLike, frame: dict[str, Any]) -> None:
    try:
        payload = json.dumps(
            frame,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("control frame is not canonical JSON") from exc
    if not 0 < len(payload) <= MAX_CONTROL_FRAME:
        raise ValidationError("control frame exceeds bounded size")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def receive_control_frame(sock: TLSSocketLike) -> dict[str, Any]:
    header = _recv_exact(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    if not 0 < size <= MAX_CONTROL_FRAME:
        raise ValidationError("invalid control frame length")
    raw = _recv_exact(sock, size)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError("invalid control frame JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError("control frame must be an object")
    return value


class ControlReplayGuard:
    def __init__(self, connection_epoch: str) -> None:
        self.connection_epoch = text(connection_epoch, "connection_epoch")
        self.last_sequence = -1

    def validate(self, frame: dict[str, Any]) -> dict[str, Any]:
        expected = {"protocol", "epoch", "sequence", "operation", "body"}
        if set(frame) != expected:
            raise ValidationError("control frame has unexpected or missing fields")
        if frame["protocol"] != CONTROL_PROTOCOL:
            raise ValidationError("control protocol mismatch")
        if frame["epoch"] != self.connection_epoch:
            raise ValidationError("stale or foreign control connection epoch")
        sequence = frame["sequence"]
        if type(sequence) is not int or sequence < 0:
            raise ValidationError("control sequence must be a non-negative integer")
        if sequence <= self.last_sequence:
            raise ValidationError("control frame replayed or out of order")
        operation = text(frame["operation"], "operation")
        if operation not in ALLOWED_OPERATIONS:
            raise ValidationError("control operation is not allowlisted")
        if not isinstance(frame["body"], dict):
            raise ValidationError("control frame body must be an object")
        self.last_sequence = sequence
        return frame


class PrivateControlChannel:
    """Bounded control framing over an already authenticated TLS socket."""

    def __init__(
        self,
        sock: TLSSocketLike,
        *,
        expected_peer_certificate_sha256: str,
        connection_epoch: str,
    ) -> None:
        self.sock = sock
        self.tls = validate_tls_peer(
            sock,
            expected_peer_certificate_sha256=expected_peer_certificate_sha256,
        )
        self.epoch = text(connection_epoch, "connection_epoch")
        self._send_sequence = 0
        self._receive_guard = ControlReplayGuard(self.epoch)

    def send(self, operation: str, body: dict[str, Any]) -> int:
        operation = text(operation, "operation")
        if operation not in ALLOWED_OPERATIONS:
            raise ValidationError("control operation is not allowlisted")
        if not isinstance(body, dict):
            raise ValidationError("control body must be an object")
        sequence = self._send_sequence
        send_control_frame(
            self.sock,
            {
                "protocol": CONTROL_PROTOCOL,
                "epoch": self.epoch,
                "sequence": sequence,
                "operation": operation,
                "body": body,
            },
        )
        self._send_sequence += 1
        return sequence

    def receive(self) -> dict[str, Any]:
        return self._receive_guard.validate(receive_control_frame(self.sock))
