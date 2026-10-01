"""Strict runtime model/operator/memory manifest contract.

The manifest is imported adapter evidence for one exact TensorMeld configuration,
model checkpoint, profile workload and adapter build. Memory is represented once per
physical pool so shared/unified memory is never added once per logical device.

Parsing a manifest does not reserve resources, qualify a workload or authorize
execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .adapter_contract import AdapterCapabilities
from .config_v2 import Config, Profile
from .model_manifest import ModelManifest, _sha256
from .schema import (
    MAX_INPUT_BYTES,
    ValidationError,
    _no_duplicates,
    items,
    number,
    record,
    text,
    unique,
)

RUNTIME_MANIFEST_SCHEMA = "tensormeld/runtime-model-manifest-v1"
MAX_RUNTIME_DEVICES = 128
MAX_RUNTIME_POOLS = 192
MAX_OPERATORS = 512
PROVENANCE_VALUES = {"fixture", "native-adapter"}


@dataclass(frozen=True)
class RuntimeDeviceManifest:
    id: str
    node: str
    backend: str
    operators: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalPoolMemory:
    pool: str
    node: str
    resident_bytes: int
    state_bytes: int
    workspace_peak_bytes: int
    preparation_peak_bytes: int

    @property
    def steady_peak_bytes(self) -> int:
        return self.resident_bytes + self.state_bytes + self.workspace_peak_bytes


@dataclass(frozen=True)
class RuntimeModelManifest:
    provenance: str
    config_sha256: str
    profile: str
    model_manifest_sha256: str
    adapter_id: str
    adapter_capabilities_sha256: str
    engine_revision: str
    worker_artifact_sha256: str
    workload: dict[str, Any]
    required_operators: tuple[str, ...]
    devices: tuple[RuntimeDeviceManifest, ...]
    pools: tuple[PhysicalPoolMemory, ...]
    operator_coverage_complete: bool
    fingerprint: str


def _profile(config: Config, profile_name: str | None) -> Profile:
    name = profile_name or config.installation.default_profile
    profile = config.profile_map.get(name)
    if profile is None:
        raise ValidationError(f"runtime manifest profile {name!r} is unknown")
    return profile


def _operator_list(value: Any, where: str) -> tuple[str, ...]:
    ops = tuple(
        text(item, f"{where}[]")
        for item in items(value, where, MAX_OPERATORS, 1)
    )
    unique(list(ops), where)
    return tuple(sorted(ops))


def _expected_workload(profile: Profile) -> dict[str, Any]:
    return {
        "task": profile.workload.task,
        "context_tokens": profile.workload.context_tokens,
        "max_output_tokens": profile.workload.max_output_tokens,
        "concurrency": profile.workload.max_active_requests,
    }


def parse_runtime_model_manifest(
    data: Any,
    *,
    config: Config,
    model: ModelManifest,
    adapter: AdapterCapabilities,
    profile_name: str | None = None,
) -> RuntimeModelManifest:
    r = record(data, "runtime model manifest", {
        "runtime_manifest_schema",
        "provenance",
        "config_sha256",
        "profile",
        "model_manifest_sha256",
        "adapter_id",
        "adapter_capabilities_sha256",
        "engine_revision",
        "worker_artifact_sha256",
        "workload",
        "required_operators",
        "devices",
        "physical_pool_memory",
        "reservation_created",
        "qualified",
        "executable",
    })
    if r["runtime_manifest_schema"] != RUNTIME_MANIFEST_SCHEMA:
        raise ValidationError(
            f"runtime_manifest_schema: expected {RUNTIME_MANIFEST_SCHEMA}"
        )
    provenance = text(r["provenance"], "provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError("provenance: expected fixture or native-adapter")
    if r["reservation_created"] is not False:
        raise ValidationError("runtime manifest cannot create a reservation")
    if r["qualified"] is not False or r["executable"] is not False:
        raise ValidationError("runtime manifest cannot self-promote qualification/execution")

    profile = _profile(config, profile_name)
    if r["profile"] != profile.name:
        raise ValidationError("runtime manifest profile does not match requested profile")
    if profile.model_manifest_ref != model.manifest_sha256:
        raise ValidationError(
            "profile.model_manifest_ref must equal the exact model manifest SHA-256"
        )
    if r["config_sha256"] != config.fingerprint:
        raise ValidationError("runtime manifest config identity mismatch")
    if r["model_manifest_sha256"] != model.manifest_sha256:
        raise ValidationError("runtime manifest model identity mismatch")
    if r["adapter_id"] != adapter.adapter_id:
        raise ValidationError("runtime manifest adapter identity mismatch")
    if r["adapter_capabilities_sha256"] != adapter.fingerprint:
        raise ValidationError("runtime manifest adapter capability identity mismatch")
    if r["engine_revision"] != adapter.engine_revision:
        raise ValidationError("runtime manifest engine revision mismatch")
    worker_artifact_sha256 = _sha256(
        r["worker_artifact_sha256"], "worker_artifact_sha256"
    )

    w = record(
        r["workload"],
        "workload",
        {"task", "context_tokens", "max_output_tokens", "concurrency"},
    )
    workload = {
        "task": text(w["task"], "workload.task"),
        "context_tokens": int(number(
            w["context_tokens"], "workload.context_tokens", 1, True
        )),
        "max_output_tokens": int(number(
            w["max_output_tokens"], "workload.max_output_tokens", 1, True
        )),
        "concurrency": int(number(
            w["concurrency"], "workload.concurrency", 1, True
        )),
    }
    if workload != _expected_workload(profile):
        raise ValidationError("runtime manifest workload does not exactly match profile")

    required_operators = _operator_list(r["required_operators"], "required_operators")
    config_devices = {d.id: d for d in config.devices}
    adapter_devices = {d.id: d for d in adapter.devices}
    devices: list[RuntimeDeviceManifest] = []
    for i, raw in enumerate(items(r["devices"], "devices", MAX_RUNTIME_DEVICES, 1)):
        d = record(
            raw,
            f"devices[{i}]",
            {"id", "node", "backend", "operators"},
        )
        device_id = text(d["id"], f"devices[{i}].id")
        cfg = config_devices.get(device_id)
        exposed = adapter_devices.get(device_id)
        if cfg is None:
            raise ValidationError(f"devices[{i}]: unknown config device {device_id}")
        if exposed is None:
            raise ValidationError(f"devices[{i}]: adapter does not expose {device_id}")
        node = text(d["node"], f"devices[{i}].node")
        backend = text(d["backend"], f"devices[{i}].backend")
        if node != cfg.node or exposed.node != cfg.node:
            raise ValidationError(f"devices[{i}]: node identity mismatch")
        if backend != cfg.backend or exposed.backend != cfg.backend:
            raise ValidationError(f"devices[{i}]: backend identity mismatch")
        devices.append(RuntimeDeviceManifest(
            device_id,
            node,
            backend,
            _operator_list(d["operators"], f"devices[{i}].operators"),
        ))
    unique([d.id for d in devices], "devices")

    used_nodes = {d.node for d in devices}
    pools_by_id = {p.id: p for p in config.pools}
    pool_records: list[PhysicalPoolMemory] = []
    for i, raw in enumerate(items(
        r["physical_pool_memory"],
        "physical_pool_memory",
        MAX_RUNTIME_POOLS,
        1,
    )):
        p = record(
            raw,
            f"physical_pool_memory[{i}]",
            {
                "pool_ref",
                "node",
                "resident_bytes",
                "state_bytes",
                "workspace_peak_bytes",
                "preparation_peak_bytes",
            },
        )
        pool_id = text(p["pool_ref"], f"physical_pool_memory[{i}].pool_ref")
        cfg_pool = pools_by_id.get(pool_id)
        if cfg_pool is None:
            raise ValidationError(
                f"physical_pool_memory[{i}]: unknown physical pool {pool_id}"
            )
        node = text(p["node"], f"physical_pool_memory[{i}].node")
        if node != cfg_pool.node:
            raise ValidationError(
                f"physical_pool_memory[{i}]: pool/node identity mismatch"
            )
        if node not in used_nodes:
            raise ValidationError(
                f"physical_pool_memory[{i}]: pool belongs to an unused worker node"
            )
        resident = int(number(
            p["resident_bytes"],
            f"physical_pool_memory[{i}].resident_bytes",
            0,
            True,
        ))
        state = int(number(
            p["state_bytes"],
            f"physical_pool_memory[{i}].state_bytes",
            0,
            True,
        ))
        workspace = int(number(
            p["workspace_peak_bytes"],
            f"physical_pool_memory[{i}].workspace_peak_bytes",
            0,
            True,
        ))
        preparation = int(number(
            p["preparation_peak_bytes"],
            f"physical_pool_memory[{i}].preparation_peak_bytes",
            0,
            True,
        ))
        steady = resident + state + workspace
        if preparation < steady:
            raise ValidationError(
                f"physical_pool_memory[{i}]: preparation peak is below steady peak"
            )
        if (
            cfg_pool.reported_capacity_bytes is not None
            and preparation > cfg_pool.reported_capacity_bytes
        ):
            raise ValidationError(
                f"physical_pool_memory[{i}]: preparation peak exceeds reported capacity"
            )
        pool_records.append(PhysicalPoolMemory(
            pool_id, node, resident, state, workspace, preparation
        ))
    unique([p.pool for p in pool_records], "physical_pool_memory.pool_ref")

    covered = set(required_operators)
    for device in devices:
        covered &= set(device.operators)
    coverage_complete = covered == set(required_operators)

    canonical = {
        "runtime_manifest_schema": RUNTIME_MANIFEST_SCHEMA,
        "provenance": provenance,
        "config_sha256": config.fingerprint,
        "profile": profile.name,
        "model_manifest_sha256": model.manifest_sha256,
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": worker_artifact_sha256,
        "workload": workload,
        "required_operators": list(required_operators),
        "devices": [
            {
                "id": d.id,
                "node": d.node,
                "backend": d.backend,
                "operators": list(d.operators),
            }
            for d in sorted(devices, key=lambda item: item.id)
        ],
        "physical_pool_memory": [
            {
                "pool_ref": p.pool,
                "node": p.node,
                "resident_bytes": p.resident_bytes,
                "state_bytes": p.state_bytes,
                "workspace_peak_bytes": p.workspace_peak_bytes,
                "preparation_peak_bytes": p.preparation_peak_bytes,
            }
            for p in sorted(pool_records, key=lambda item: item.pool)
        ],
        "operator_coverage_complete": coverage_complete,
        "reservation_created": False,
        "qualified": False,
        "executable": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return RuntimeModelManifest(
        provenance,
        config.fingerprint,
        profile.name,
        model.manifest_sha256,
        adapter.adapter_id,
        adapter.fingerprint,
        adapter.engine_revision,
        worker_artifact_sha256,
        workload,
        required_operators,
        tuple(sorted(devices, key=lambda item: item.id)),
        tuple(sorted(pool_records, key=lambda item: item.pool)),
        coverage_complete,
        fingerprint,
    )


def runtime_manifest_summary(manifest: RuntimeModelManifest) -> dict[str, Any]:
    return {
        "result_schema": "tensormeld/runtime-model-manifest-validation-v1",
        "runtime_manifest_sha256": manifest.fingerprint,
        "provenance": manifest.provenance,
        "config_sha256": manifest.config_sha256,
        "profile": manifest.profile,
        "model_manifest_sha256": manifest.model_manifest_sha256,
        "adapter_id": manifest.adapter_id,
        "adapter_capabilities_sha256": manifest.adapter_capabilities_sha256,
        "engine_revision": manifest.engine_revision,
        "worker_artifact_sha256": manifest.worker_artifact_sha256,
        "workload": manifest.workload,
        "required_operators": list(manifest.required_operators),
        "operator_coverage_complete": manifest.operator_coverage_complete,
        "devices": [
            {
                "id": d.id,
                "node": d.node,
                "backend": d.backend,
                "operators": list(d.operators),
            }
            for d in manifest.devices
        ],
        "physical_pool_memory": [
            {
                "pool_ref": p.pool,
                "node": p.node,
                "resident_bytes": p.resident_bytes,
                "state_bytes": p.state_bytes,
                "workspace_peak_bytes": p.workspace_peak_bytes,
                "steady_peak_bytes": p.steady_peak_bytes,
                "preparation_peak_bytes": p.preparation_peak_bytes,
            }
            for p in manifest.pools
        ],
        "reservation_created": False,
        "qualified": False,
        "executable": False,
        "warnings": [
            "Memory is an adapter observation/manifest, not a live reservation.",
            "Each physical pool appears once; shared/unified pools must not be duplicated per device.",
            "Operator coverage is declared for this exact model/workload tuple but is not E3 correctness qualification.",
            "Fixture provenance is portable contract evidence only, never hardware qualification.",
        ],
    }


def load_runtime_model_manifest(
    path: str | Path,
    *,
    config: Config,
    model: ModelManifest,
    adapter: AdapterCapabilities,
    profile_name: str | None = None,
) -> RuntimeModelManifest:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("runtime model manifest exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid runtime model manifest JSON: {exc}") from exc
    return parse_runtime_model_manifest(
        value,
        config=config,
        model=model,
        adapter=adapter,
        profile_name=profile_name,
    )
