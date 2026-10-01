"""Explicit mapping between native-engine device names and TensorMeld identities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .config_v2 import Config
from .llamacpp_probe import LLAMACPP_PINNED_COMMIT
from .schema import ValidationError, items, record, text, unique

MAX_BINDINGS = 128


@dataclass(frozen=True)
class DeviceBinding:
    engine_device_name: str
    tensormeld_device_id: str
    memory_reporter: bool


@dataclass(frozen=True)
class LlamaCppBinding:
    config_sha256: str
    node_id: str
    artifact_sha256: str
    approval: str
    mappings: tuple[DeviceBinding, ...]
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> "LlamaCppBinding":
        r = record(data, "llama.cpp device binding", {
            "binding_schema", "config_sha256", "node_id", "artifact_sha256",
            "approval", "mappings",
        })
        if r["binding_schema"] != "tensormeld/llamacpp-device-binding-v1":
            raise ValidationError(
                "binding_schema: expected tensormeld/llamacpp-device-binding-v1"
            )
        if r["approval"] != "explicit":
            raise ValidationError("approval: explicit approval is required")
        mappings = []
        for i, raw in enumerate(items(r["mappings"], "mappings", MAX_BINDINGS, 1)):
            m = record(
                raw, f"mappings[{i}]",
                {"engine_device_name", "tensormeld_device_id", "memory_reporter"},
            )
            if type(m["memory_reporter"]) is not bool:
                raise ValidationError(f"mappings[{i}].memory_reporter: expected boolean")
            mappings.append(DeviceBinding(
                text(m["engine_device_name"], f"mappings[{i}].engine_device_name"),
                text(m["tensormeld_device_id"], f"mappings[{i}].tensormeld_device_id"),
                m["memory_reporter"],
            ))
        unique([m.engine_device_name for m in mappings], "mappings.engine_device_name")
        unique([m.tensormeld_device_id for m in mappings], "mappings.tensormeld_device_id")
        canonical = {
            "binding_schema": r["binding_schema"],
            "config_sha256": text(r["config_sha256"], "config_sha256"),
            "node_id": text(r["node_id"], "node_id"),
            "artifact_sha256": text(r["artifact_sha256"], "artifact_sha256").lower(),
            "approval": "explicit",
            "mappings": [m.__dict__ for m in mappings],
        }
        if len(canonical["artifact_sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in canonical["artifact_sha256"]
        ):
            raise ValidationError("artifact_sha256: expected SHA-256 hex")
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            canonical["config_sha256"], canonical["node_id"],
            canonical["artifact_sha256"], "explicit", tuple(mappings), digest,
        )


def bind_llamacpp_probe(
    config: Config, probe: dict[str, Any], binding: LlamaCppBinding,
) -> dict[str, Any]:
    if probe.get("probe_schema") != "tensormeld/llamacpp-probe-v1":
        raise ValidationError("probe_schema: expected tensormeld/llamacpp-probe-v1")
    if probe.get("engine") != "llama.cpp":
        raise ValidationError("probe.engine: expected llama.cpp")
    if probe.get("pinned_source_revision") != LLAMACPP_PINNED_COMMIT:
        raise ValidationError("probe source revision does not match pinned llama.cpp revision")
    if probe.get("qualified") is not False or probe.get("executable") is not False:
        raise ValidationError("probe cannot self-promote qualification/execution")
    if binding.config_sha256 != config.fingerprint:
        raise ValidationError("binding config_sha256 does not match current config")
    if binding.artifact_sha256 != probe.get("artifact_sha256"):
        raise ValidationError("binding artifact_sha256 does not match probe")
    nodes = {n.id: n for n in config.nodes}
    devices = {d.id: d for d in config.devices}
    pools = {p.id: p for p in config.pools}
    if binding.node_id not in nodes:
        raise ValidationError("binding node_id is unknown")

    observed_engine = {}
    for raw in probe.get("devices", []):
        if not isinstance(raw, dict):
            raise ValidationError("probe.devices: expected objects")
        name = raw.get("engine_device_name")
        if not isinstance(name, str) or not name:
            raise ValidationError("probe device has invalid engine name")
        if name in observed_engine:
            raise ValidationError("probe contains duplicate engine device names")
        observed_engine[name] = raw

    runtime_devices = {}
    runtime_pools = {}
    reporter_by_pool = {}
    resolved = []
    for mapping in binding.mappings:
        cfg = devices.get(mapping.tensormeld_device_id)
        if cfg is None:
            raise ValidationError(
                f"binding references unknown TensorMeld device {mapping.tensormeld_device_id}"
            )
        if cfg.node != binding.node_id:
            raise ValidationError(
                f"TensorMeld device {cfg.id} is not owned by binding node {binding.node_id}"
            )
        raw = observed_engine.get(mapping.engine_device_name)
        if raw is None:
            raise ValidationError(
                f"engine device {mapping.engine_device_name} is absent from probe"
            )
        runtime_devices[cfg.id] = {"backend": cfg.backend, "state": "observed"}
        item = {
            "engine_device_name": mapping.engine_device_name,
            "tensormeld_device_id": cfg.id,
            "backend_from_config": cfg.backend,
            "pool_ref": cfg.pool,
            "memory_reporter": mapping.memory_reporter,
            "engine_description": raw.get("description"),
        }
        if mapping.memory_reporter:
            if cfg.pool in reporter_by_pool:
                raise ValidationError(
                    f"multiple memory reporters selected for physical pool {cfg.pool}"
                )
            free_bytes = raw.get("free_bytes")
            total_bytes = raw.get("total_bytes")
            if type(free_bytes) is not int or type(total_bytes) is not int:
                raise ValidationError("probe memory observation must be integer bytes")
            if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
                raise ValidationError("probe memory observation is invalid")
            capacity = pools[cfg.pool].reported_capacity_bytes
            if capacity is not None and free_bytes > capacity:
                raise ValidationError(
                    f"engine free-memory report exceeds configured capacity for {cfg.pool}"
                )
            runtime_pools[cfg.pool] = {"available_bytes": free_bytes}
            reporter_by_pool[cfg.pool] = cfg.id
        resolved.append(item)

    return {
        "result_schema": "tensormeld/llamacpp-device-binding-result-v1",
        "config_sha256": config.fingerprint,
        "probe_artifact_sha256": probe["artifact_sha256"],
        "binding_sha256": binding.fingerprint,
        "node_id": binding.node_id,
        "resolved_mappings": resolved,
        "unmapped_engine_devices": sorted(set(observed_engine) - {
            m.engine_device_name for m in binding.mappings
        }),
        "runtime_observation": {
            "runtime_observation_schema": "tensormeld/runtime-observation-v1",
            "config_sha256": config.fingerprint,
            "devices": runtime_devices,
            "pools": runtime_pools,
            "qualified": False,
            "executable": False,
        },
        "qualified": False,
        "executable": False,
        "warnings": [
            "Mapped devices are observed, not ready; a backend self-test/qualification is still required.",
            "Backend identity comes from approved TensorMeld config, not from engine device names.",
            "Only explicitly selected memory reporters populate physical-pool availability.",
            "Native engine free memory remains transient and is not a reservation.",
        ],
    }
