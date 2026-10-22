"""Explicit private endpoint policy and bounded remote HostAgent dispatch.

Remote peers never supply executable paths, command lines, manifests, snapshots or model
payloads. Reservation calls refer only to locally registered immutable objects by ID/hash.
This module sits above PrivateControlChannel and inherits its mTLS, certificate pinning,
epoch and replay guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
import ssl
from typing import Any

from .agent import Enrollment, HostAgent
from .private_transport import (
    EnrolledPeerTLSBinding,
    PrivateControlChannel,
    _sha256_hex,
    validate_tls_peer,
)
from .runtime_model_manifest import RuntimeModelManifest
from .schema import ValidationError, number, record, text

REMOTE_REQUEST_SCHEMA = "tensormeld/remote-agent-request-v1"
REMOTE_RESPONSE_SCHEMA = "tensormeld/remote-agent-response-v1"


@dataclass(frozen=True)
class PrivateEndpoint:
    node_id: str
    enrollment_id: str
    address: str
    port: int
    certificate_sha256: str


def private_endpoint(
    *,
    node_id: str,
    enrollment_id: str,
    address: str,
    port: int,
    certificate_sha256: str,
) -> PrivateEndpoint:
    node_id = text(node_id, "node_id")
    enrollment_id = text(enrollment_id, "enrollment_id")
    address = text(address, "address")
    try:
        ip = ipaddress.ip_address(address)
    except ValueError as exc:
        raise ValidationError("private endpoint address must be an IP literal") from exc
    if ip.is_unspecified or ip.is_multicast or ip.is_link_local:
        raise ValidationError("private endpoint address class is not allowed")
    if not (ip.is_private or ip.is_loopback):
        raise ValidationError("private endpoint must be private or loopback")
    port = int(number(port, "port", 1, True))
    if port > 65535:
        raise ValidationError("port exceeds 65535")
    return PrivateEndpoint(
        node_id,
        enrollment_id,
        str(ip),
        port,
        _sha256_hex(certificate_sha256, "certificate_sha256"),
    )


def validate_endpoint_binding(
    endpoint: PrivateEndpoint,
    *,
    enrollment: Enrollment,
    binding: EnrolledPeerTLSBinding,
) -> None:
    if endpoint.node_id != enrollment.node_id:
        raise ValidationError("endpoint node does not match enrollment")
    if endpoint.enrollment_id != enrollment.enrollment_id:
        raise ValidationError("endpoint enrollment identity mismatch")
    if binding.node_id != enrollment.node_id or binding.enrollment_id != enrollment.enrollment_id:
        raise ValidationError("TLS binding does not match enrollment")
    if binding.certificate_sha256 != endpoint.certificate_sha256:
        raise ValidationError("endpoint certificate does not match TLS enrollment binding")


class LocalObjectRegistry:
    """Host-owned references used by remote reservation requests."""

    def __init__(self) -> None:
        self._manifests: dict[str, RuntimeModelManifest] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}

    def register_manifest(self, manifest: RuntimeModelManifest) -> None:
        self._manifests[manifest.fingerprint] = manifest

    def register_snapshot(self, observation_id: str, snapshot: dict[str, Any]) -> None:
        observation_id = text(observation_id, "observation_id")
        if snapshot.get("observation_id") != observation_id:
            raise ValidationError("snapshot observation_id mismatch")
        self._snapshots[observation_id] = snapshot

    def manifest(self, fingerprint: str) -> RuntimeModelManifest:
        value = self._manifests.get(_sha256_hex(fingerprint, "runtime_manifest_sha256"))
        if value is None:
            raise ValidationError("runtime manifest is not registered locally")
        return value

    def snapshot(self, observation_id: str) -> dict[str, Any]:
        observation_id = text(observation_id, "observation_id")
        value = self._snapshots.get(observation_id)
        if value is None:
            raise ValidationError("runtime observation is not registered locally")
        return value


class RemoteAgentDispatcher:
    def __init__(self, agent: HostAgent, registry: LocalObjectRegistry) -> None:
        self.agent = agent
        self.registry = registry
        self._seen_request_ids: set[str] = set()

    def _request(self, operation: str, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        root = record(
            body,
            "remote request",
            {"request_schema", "request_id", "args"},
        )
        if root["request_schema"] != REMOTE_REQUEST_SCHEMA:
            raise ValidationError(f"request_schema: expected {REMOTE_REQUEST_SCHEMA}")
        request_id = text(root["request_id"], "request_id")
        if request_id in self._seen_request_ids:
            raise ValidationError("duplicate remote request_id")
        if not isinstance(root["args"], dict):
            raise ValidationError("remote request args must be an object")
        self._seen_request_ids.add(request_id)
        return request_id, root["args"]

    def dispatch(self, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        request_id, args = self._request(operation, body)
        if operation == "describe":
            if args:
                raise ValidationError("describe takes no arguments")
            result = self.agent.describe()
        elif operation == "health":
            if args:
                raise ValidationError("health takes no arguments")
            result = self.agent.health()
        elif operation == "drain":
            if args:
                raise ValidationError("drain takes no arguments")
            result = self.agent.drain()
        elif operation == "disable":
            if args:
                raise ValidationError("disable takes no arguments")
            result = self.agent.disable()
        elif operation == "release":
            a = record(args, "release args", {"lease_id"})
            result = self.agent.release(text(a["lease_id"], "lease_id"))
        elif operation in {"reserve", "launch_recheck"}:
            a = record(
                args,
                f"{operation} args",
                {"lease_id", "runtime_manifest_sha256", "observation_id"},
            )
            lease_id = text(a["lease_id"], "lease_id")
            manifest = self.registry.manifest(a["runtime_manifest_sha256"])
            snapshot = self.registry.snapshot(a["observation_id"])
            if operation == "reserve":
                result = self.agent.reserve(
                    lease_id=lease_id, manifest=manifest, snapshot=snapshot
                )
            else:
                result = self.agent.launch_recheck(
                    lease_id=lease_id, manifest=manifest, snapshot=snapshot
                )
        else:
            raise ValidationError("remote operation is not dispatchable")
        return {
            "response_schema": REMOTE_RESPONSE_SCHEMA,
            "request_id": request_id,
            "ok": True,
            "result": result,
        }

    def serve_one(self, channel: PrivateControlChannel) -> None:
        frame = channel.receive()
        response = self.dispatch(frame["operation"], frame["body"])
        channel.send(frame["operation"], response)


class RemoteAgentClient:
    def __init__(self, channel: PrivateControlChannel) -> None:
        self.channel = channel
        self._request_counter = 0

    def call(self, operation: str, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValidationError("remote args must be an object")
        request_id = f"request-{self._request_counter}"
        self._request_counter += 1
        self.channel.send(
            operation,
            {
                "request_schema": REMOTE_REQUEST_SCHEMA,
                "request_id": request_id,
                "args": args,
            },
        )
        frame = self.channel.receive()
        if frame["operation"] != operation:
            raise ValidationError("remote response operation mismatch")
        response = frame["body"]
        expected = {"response_schema", "request_id", "ok", "result"}
        if not isinstance(response, dict) or set(response) != expected:
            raise ValidationError("remote response schema is invalid")
        if response["response_schema"] != REMOTE_RESPONSE_SCHEMA:
            raise ValidationError("remote response schema mismatch")
        if response["request_id"] != request_id or response["ok"] is not True:
            raise ValidationError("remote response correlation failed")
        if not isinstance(response["result"], dict):
            raise ValidationError("remote response result must be an object")
        return response["result"]



class RemoteAgentConnection:
    def __init__(self, tls_socket: ssl.SSLSocket, client: RemoteAgentClient) -> None:
        self.tls_socket = tls_socket
        self.client = client

    def close(self) -> None:
        self.tls_socket.close()

    def __enter__(self) -> RemoteAgentClient:
        return self.client

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def connect_private_endpoint(
    endpoint: PrivateEndpoint,
    *,
    context: ssl.SSLContext,
    expected_enrollment: Enrollment,
    peer_binding: EnrolledPeerTLSBinding,
    connection_epoch: str,
    timeout_s: float = 5.0,
) -> RemoteAgentConnection:
    validate_endpoint_binding(
        endpoint, enrollment=expected_enrollment, binding=peer_binding
    )
    if not 0 < timeout_s <= 60:
        raise ValidationError("timeout_s must be in (0, 60]")
    raw = socket.create_connection((endpoint.address, endpoint.port), timeout=timeout_s)
    raw.settimeout(timeout_s)
    try:
        tls_sock = context.wrap_socket(raw, server_hostname=endpoint.address)
        validate_tls_peer(
            tls_sock,
            expected_peer_certificate_sha256=endpoint.certificate_sha256,
            expected_enrollment=expected_enrollment,
            peer_binding=peer_binding,
        )
        channel = PrivateControlChannel(
            tls_sock,
            expected_peer_certificate_sha256=endpoint.certificate_sha256,
            connection_epoch=connection_epoch,
        )
        return RemoteAgentConnection(tls_sock, RemoteAgentClient(channel))
    except BaseException:
        raw.close()
        raise


def serve_private_endpoint_once(
    endpoint: PrivateEndpoint,
    *,
    context: ssl.SSLContext,
    agent: HostAgent,
    registry: LocalObjectRegistry,
    expected_peer_enrollment: Enrollment,
    peer_binding: EnrolledPeerTLSBinding,
    local_binding: EnrolledPeerTLSBinding,
    connection_epoch: str,
    request_count: int = 1,
    timeout_s: float = 5.0,
) -> None:
    validate_endpoint_binding(
        endpoint,
        enrollment=agent.enrollment,
        binding=local_binding,
    )
    if type(request_count) is not int or not 1 <= request_count <= 32:
        raise ValidationError("request_count must be an integer in 1..32")
    if not 0 < timeout_s <= 60:
        raise ValidationError("timeout_s must be in (0, 60]")

    dispatcher = RemoteAgentDispatcher(agent, registry)
    with socket.socket(socket.AF_INET6 if ":" in endpoint.address else socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((endpoint.address, endpoint.port))
        listener.listen(1)
        listener.settimeout(timeout_s)
        raw, _ = listener.accept()
        with raw:
            raw.settimeout(timeout_s)
            with context.wrap_socket(raw, server_side=True) as tls_sock:
                validate_tls_peer(
                    tls_sock,
                    expected_peer_certificate_sha256=peer_binding.certificate_sha256,
                    expected_enrollment=expected_peer_enrollment,
                    peer_binding=peer_binding,
                )
                channel = PrivateControlChannel(
                    tls_sock,
                    expected_peer_certificate_sha256=peer_binding.certificate_sha256,
                    connection_epoch=connection_epoch,
                )
                for _ in range(request_count):
                    dispatcher.serve_one(channel)
