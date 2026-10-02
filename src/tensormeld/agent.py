"""Local enrolled host-agent skeleton with authenticated capability envelopes.

This module intentionally exposes no remote listener. Enrollment material is supplied at
runtime, secrets are never serialized, and the only state-changing operations are fixed
allowlisted methods. HMAC-SHA256 uses Python's standard-library implementation; this is an
authentication envelope, not encrypted transport.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
from typing import Any

from .admission import LocalAdmissionController
from .config_v2 import Config
from .runtime_model_manifest import RuntimeModelManifest
from .schema import ValidationError, text

AGENT_PROTOCOL = "tensormeld/agent-control-v1"
CAPABILITY_SCHEMA = "tensormeld/agent-capability-envelope-v1"
LIFECYCLE_STATES = {"enabled", "draining", "disabled", "revoked"}
ALLOWED_OPERATIONS = frozenset({
    "describe", "health", "reserve", "launch_recheck", "release", "drain", "disable"
})
MAX_SECRET_BYTES = 128


@dataclass(frozen=True)
class Enrollment:
    enrollment_id: str
    node_id: str
    key_id: str


class ReplayGuard:
    def __init__(self) -> None:
        self._last: dict[tuple[str, str], int] = {}

    def accept(self, enrollment_id: str, instance_id: str, sequence: int) -> None:
        key = (enrollment_id, instance_id)
        previous = self._last.get(key, -1)
        if sequence <= previous:
            raise ValidationError("capability envelope is replayed or stale")
        self._last[key] = sequence


def enrollment_from_runtime(
    *,
    enrollment_id: str,
    node_id: str,
    key_id: str,
    shared_secret: bytes,
) -> tuple[Enrollment, bytes]:
    enrollment_id = text(enrollment_id, "enrollment_id")
    node_id = text(node_id, "node_id")
    key_id = text(key_id, "key_id")
    if not isinstance(shared_secret, bytes) or not 32 <= len(shared_secret) <= MAX_SECRET_BYTES:
        raise ValidationError("shared_secret must be 32..128 runtime-only bytes")
    return Enrollment(enrollment_id, node_id, key_id), shared_secret


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sign_capability_envelope(
    *,
    enrollment: Enrollment,
    shared_secret: bytes,
    config: Config,
    instance_id: str,
    sequence: int,
    lifecycle: str,
    operations: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    instance_id = text(instance_id, "instance_id")
    if type(sequence) is not int or sequence < 0:
        raise ValidationError("sequence must be a non-negative integer")
    if lifecycle not in LIFECYCLE_STATES:
        raise ValidationError("unsupported lifecycle state")
    if enrollment.node_id not in {n.id for n in config.nodes}:
        raise ValidationError("enrollment node is absent from config")
    ops = tuple(sorted(set(operations)))
    if not ops or not set(ops) <= ALLOWED_OPERATIONS:
        raise ValidationError("capability envelope contains unsupported operations")
    payload = {
        "capability_schema": CAPABILITY_SCHEMA,
        "protocol": AGENT_PROTOCOL,
        "enrollment_id": enrollment.enrollment_id,
        "node_id": enrollment.node_id,
        "key_id": enrollment.key_id,
        "config_sha256": config.fingerprint,
        "instance_id": instance_id,
        "sequence": sequence,
        "lifecycle": lifecycle,
        "operations": list(ops),
    }
    signature = hmac.new(shared_secret, _canonical(payload), hashlib.sha256).hexdigest()
    return {**payload, "auth": {"algorithm": "HMAC-SHA256", "signature": signature}}


def verify_capability_envelope(
    envelope: dict[str, Any],
    *,
    enrollment: Enrollment,
    shared_secret: bytes,
    config: Config,
    replay_guard: ReplayGuard | None = None,
) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ValidationError("capability envelope must be an object")
    expected = {
        "capability_schema", "protocol", "enrollment_id", "node_id", "key_id",
        "config_sha256", "instance_id", "sequence", "lifecycle", "operations", "auth",
    }
    if set(envelope) != expected:
        raise ValidationError("capability envelope has unexpected or missing fields")
    auth = envelope["auth"]
    if not isinstance(auth, dict) or set(auth) != {"algorithm", "signature"}:
        raise ValidationError("capability envelope auth record is invalid")
    if auth["algorithm"] != "HMAC-SHA256":
        raise ValidationError("unsupported capability envelope authentication")
    payload = {k: envelope[k] for k in envelope if k != "auth"}
    if payload["capability_schema"] != CAPABILITY_SCHEMA or payload["protocol"] != AGENT_PROTOCOL:
        raise ValidationError("capability envelope protocol/schema mismatch")
    if (
        payload["enrollment_id"] != enrollment.enrollment_id
        or payload["node_id"] != enrollment.node_id
        or payload["key_id"] != enrollment.key_id
    ):
        raise ValidationError("capability envelope enrollment identity mismatch")
    if payload["config_sha256"] != config.fingerprint:
        raise ValidationError("capability envelope config identity mismatch")
    if payload["lifecycle"] not in LIFECYCLE_STATES:
        raise ValidationError("capability envelope lifecycle is invalid")
    ops = payload["operations"]
    if not isinstance(ops, list) or not ops or len(ops) != len(set(ops)) or not set(ops) <= ALLOWED_OPERATIONS:
        raise ValidationError("capability envelope operations are invalid")
    sequence = payload["sequence"]
    if type(sequence) is not int or sequence < 0:
        raise ValidationError("capability envelope sequence is invalid")
    supplied = auth["signature"]
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ValidationError("capability envelope signature is invalid")
    expected_sig = hmac.new(shared_secret, _canonical(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(supplied, expected_sig):
        raise ValidationError("capability envelope authentication failed")
    if replay_guard is not None:
        replay_guard.accept(enrollment.enrollment_id, payload["instance_id"], sequence)
    return payload


class HostAgent:
    """Local host authority. No network listener or arbitrary execution surface."""

    def __init__(
        self,
        *,
        config: Config,
        enrollment: Enrollment,
        shared_secret: bytes,
        instance_id: str | None = None,
    ) -> None:
        if enrollment.node_id not in {n.id for n in config.nodes}:
            raise ValidationError("agent enrollment node is absent from config")
        self.config = config
        self.enrollment = enrollment
        self._secret = shared_secret
        self.instance_id = instance_id or secrets.token_hex(16)
        self.lifecycle = "enabled"
        self.sequence = 0
        self.admission = LocalAdmissionController()

    def health(self) -> dict[str, Any]:
        return {
            "status": "HEALTHY" if self.lifecycle == "enabled" else self.lifecycle.upper(),
            "node_id": self.enrollment.node_id,
            "instance_id": self.instance_id,
            "lifecycle": self.lifecycle,
            "active_lease_ids": [lease.lease_id for lease in self.admission.active_leases()],
            "qualified": False,
            "executable": False,
        }

    def describe(self) -> dict[str, Any]:
        self.sequence += 1
        return sign_capability_envelope(
            enrollment=self.enrollment,
            shared_secret=self._secret,
            config=self.config,
            instance_id=self.instance_id,
            sequence=self.sequence,
            lifecycle=self.lifecycle,
            operations=sorted(ALLOWED_OPERATIONS),
        )

    def drain(self) -> dict[str, Any]:
        if self.lifecycle in {"disabled", "revoked"}:
            raise ValidationError("disabled/revoked agent cannot enter draining")
        self.lifecycle = "draining"
        return {"status": "DRAINING", "node_id": self.enrollment.node_id}

    def disable(self) -> dict[str, Any]:
        if self.admission.active_leases():
            raise ValidationError("cannot disable agent with active leases; drain/release first")
        self.lifecycle = "disabled"
        return {"status": "DISABLED", "node_id": self.enrollment.node_id}

    def revoke_local(self) -> dict[str, Any]:
        if self.admission.active_leases():
            raise ValidationError("cannot revoke local identity with active leases")
        self.lifecycle = "revoked"
        return {"status": "REVOKED", "node_id": self.enrollment.node_id}

    def reserve(
        self,
        *,
        lease_id: str,
        manifest: RuntimeModelManifest,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if self.lifecycle != "enabled":
            raise ValidationError("agent is not accepting new reservations")
        return self.admission.reserve(
            lease_id=lease_id,
            config=self.config,
            manifest=manifest,
            snapshot=snapshot,
            owned_node_id=self.enrollment.node_id,
        )

    def launch_recheck(
        self,
        *,
        lease_id: str,
        manifest: RuntimeModelManifest,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if self.lifecycle not in {"enabled", "draining"}:
            raise ValidationError("agent cannot launch in current lifecycle state")
        return self.admission.launch_recheck(
            lease_id=lease_id,
            config=self.config,
            manifest=manifest,
            snapshot=snapshot,
            owned_node_id=self.enrollment.node_id,
        )

    def release(self, lease_id: str) -> dict[str, Any]:
        return self.admission.release(lease_id)

    def execute_peer_request(self, *_: Any, **__: Any) -> None:
        raise ValidationError("agent exposes no peer-supplied execution or shell API")
