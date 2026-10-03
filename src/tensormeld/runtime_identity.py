"""Stable live runtime identity for retained backend-readiness evidence.

The identity intentionally excludes transient values such as timestamps and free memory.
It binds the narrow backend-ready fact to the worker build, host OS, driver/runtime stack,
physical device identity and relevant topology fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, record, text
from .model_manifest import _sha256

RUNTIME_IDENTITY_SCHEMA = "tensormeld/runtime-identity-v1"


@dataclass(frozen=True)
class RuntimeIdentity:
    node_id: str
    worker_artifact_sha256: str
    worker_build_id: str
    os_name: str
    os_version: str
    driver_id: str
    driver_version: str
    runtime_id: str
    runtime_version: str
    tensormeld_device_id: str
    physical_device_id: str
    topology_sha256: str
    identity_sha256: str

    @classmethod
    def parse(cls, data: Any) -> "RuntimeIdentity":
        r = record(data, "runtime identity", {
            "runtime_identity_schema",
            "node_id",
            "worker_artifact_sha256",
            "worker_build_id",
            "os_name",
            "os_version",
            "driver_id",
            "driver_version",
            "runtime_id",
            "runtime_version",
            "tensormeld_device_id",
            "physical_device_id",
            "topology_sha256",
        })
        if r["runtime_identity_schema"] != RUNTIME_IDENTITY_SCHEMA:
            raise ValidationError(
                f"runtime_identity_schema: expected {RUNTIME_IDENTITY_SCHEMA}"
            )
        canonical = {
            "runtime_identity_schema": RUNTIME_IDENTITY_SCHEMA,
            "node_id": text(r["node_id"], "node_id"),
            "worker_artifact_sha256": _sha256(
                r["worker_artifact_sha256"], "worker_artifact_sha256"
            ),
            "worker_build_id": text(r["worker_build_id"], "worker_build_id"),
            "os_name": text(r["os_name"], "os_name"),
            "os_version": text(r["os_version"], "os_version"),
            "driver_id": text(r["driver_id"], "driver_id"),
            "driver_version": text(r["driver_version"], "driver_version"),
            "runtime_id": text(r["runtime_id"], "runtime_id"),
            "runtime_version": text(r["runtime_version"], "runtime_version"),
            "tensormeld_device_id": text(
                r["tensormeld_device_id"], "tensormeld_device_id"
            ),
            "physical_device_id": text(
                r["physical_device_id"], "physical_device_id"
            ),
            "topology_sha256": _sha256(r["topology_sha256"], "topology_sha256"),
        }
        digest = hashlib.sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            canonical["node_id"],
            canonical["worker_artifact_sha256"],
            canonical["worker_build_id"],
            canonical["os_name"],
            canonical["os_version"],
            canonical["driver_id"],
            canonical["driver_version"],
            canonical["runtime_id"],
            canonical["runtime_version"],
            canonical["tensormeld_device_id"],
            canonical["physical_device_id"],
            canonical["topology_sha256"],
            digest,
        )

    def as_record(self) -> dict[str, str]:
        return {
            "runtime_identity_schema": RUNTIME_IDENTITY_SCHEMA,
            "node_id": self.node_id,
            "worker_artifact_sha256": self.worker_artifact_sha256,
            "worker_build_id": self.worker_build_id,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "driver_id": self.driver_id,
            "driver_version": self.driver_version,
            "runtime_id": self.runtime_id,
            "runtime_version": self.runtime_version,
            "tensormeld_device_id": self.tensormeld_device_id,
            "physical_device_id": self.physical_device_id,
            "topology_sha256": self.topology_sha256,
            "identity_sha256": self.identity_sha256,
        }


def load_runtime_identity(path: str | Path) -> RuntimeIdentity:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("runtime identity exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid runtime identity JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("runtime identity: expected object")
    supplied = value.pop("identity_sha256", None)
    identity = RuntimeIdentity.parse(value)
    if supplied is not None and supplied != identity.identity_sha256:
        raise ValidationError("runtime identity fingerprint mismatch")
    return identity
