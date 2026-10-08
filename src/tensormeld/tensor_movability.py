"""Exact per-tensor storage and movability classification.

This is an evidence contract, not a heuristic. TensorMeld never infers movability from
tensor names, backend labels, tensor size, or model family. Every tensor in the exact
complete GGUF index must be classified explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .adapter_contract import AdapterCapabilities
from .config_v2 import Config
from .llamacpp_package import LlamaCppBuildPackage, validate_llamacpp_package_identity
from .model_manifest import ModelManifest
from .runtime_identity import RuntimeIdentity
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

MOVABILITY_SCHEMA = "tensormeld/tensor-movability-v1"
MAX_TENSORS = 100_000
MAX_RUNTIME_IDENTITIES = 128
MAX_ALLOWED_DEVICES = 128
STORAGE_CLASSES = {"hard_resident", "reclaimable_file_backed"}
MOVEMENT_CLASSES = {"pinned", "owner_local", "placement_movable"}
PROVENANCE_VALUES = {"fixture", "native-adapter"}


@dataclass(frozen=True)
class TensorMovability:
    name: str
    n_bytes: int
    storage_class: str
    movement_class: str
    allowed_devices: tuple[str, ...]
    alias_group: str | None


@dataclass(frozen=True)
class TensorMovabilityProfile:
    provenance: str
    model_manifest_sha256: str
    tensor_index_sha256: str
    adapter_capabilities_sha256: str
    package_sha256: str
    runtime_identity_sha256: tuple[str, ...]
    tensors: tuple[TensorMovability, ...]
    fingerprint: str


def _flatten_index(index: Any, model: ModelManifest) -> dict[str, int]:
    if not isinstance(index, dict):
        raise ValidationError("GGUF index: expected object")
    if index.get("index_schema") != "tensormeld/gguf-index-v1":
        raise ValidationError("GGUF index schema mismatch")
    if index.get("complete_shard_set") is not True:
        raise ValidationError("tensor movability requires a complete GGUF shard set")
    if index.get("index_sha256") != model.tensor_index_sha256:
        raise ValidationError("GGUF tensor index identity does not match model manifest")
    if index.get("tensor_count") != model.tensor_count:
        raise ValidationError("GGUF tensor count does not match model manifest")
    if index.get("tensor_payload_bytes") != model.tensor_payload_bytes:
        raise ValidationError("GGUF tensor payload bytes do not match model manifest")

    model_files = {f.name: f.size_bytes for f in model.files}
    files = index.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("GGUF index files: expected non-empty list")
    seen_files: dict[str, int] = {}
    tensors: dict[str, int] = {}
    for i, file_record in enumerate(files):
        if not isinstance(file_record, dict):
            raise ValidationError(f"GGUF index files[{i}]: expected object")
        name = file_record.get("file_name")
        size = file_record.get("file_size_bytes")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"GGUF index files[{i}].file_name invalid")
        if type(size) is not int or size < 1:
            raise ValidationError(f"GGUF index files[{i}].file_size_bytes invalid")
        if name in seen_files:
            raise ValidationError("GGUF index contains duplicate file name")
        seen_files[name] = size
        raw_tensors = file_record.get("tensors")
        if not isinstance(raw_tensors, list):
            raise ValidationError(f"GGUF index files[{i}].tensors: expected list")
        for j, tensor in enumerate(raw_tensors):
            if not isinstance(tensor, dict):
                raise ValidationError(
                    f"GGUF index files[{i}].tensors[{j}]: expected object"
                )
            tensor_name = tensor.get("name")
            n_bytes = tensor.get("n_bytes")
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValidationError("GGUF tensor name invalid")
            if type(n_bytes) is not int or n_bytes < 0:
                raise ValidationError("GGUF tensor n_bytes invalid")
            if tensor_name in tensors:
                raise ValidationError("GGUF index contains duplicate tensor name")
            tensors[tensor_name] = n_bytes
            if len(tensors) > MAX_TENSORS:
                raise ValidationError("GGUF tensor count exceeds movability limit")
    if seen_files != model_files:
        raise ValidationError("GGUF index file names/sizes do not match model manifest")
    if len(tensors) != model.tensor_count:
        raise ValidationError("flattened GGUF tensor count mismatch")
    if sum(tensors.values()) != model.tensor_payload_bytes:
        raise ValidationError("flattened GGUF tensor payload mismatch")
    return tensors


def parse_tensor_movability_profile(
    data: Any,
    *,
    config: Config,
    model: ModelManifest,
    tensor_index: dict[str, Any],
    adapter: AdapterCapabilities,
    package: LlamaCppBuildPackage,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
) -> TensorMovabilityProfile:
    root = record(
        data,
        "tensor movability profile",
        {
            "tensor_movability_schema",
            "provenance",
            "model_manifest_sha256",
            "tensor_index_sha256",
            "adapter_capabilities_sha256",
            "package_sha256",
            "runtime_identity_sha256",
            "tensors",
            "qualified",
            "executable",
        },
    )
    if root["tensor_movability_schema"] != MOVABILITY_SCHEMA:
        raise ValidationError(
            f"tensor_movability_schema: expected {MOVABILITY_SCHEMA}"
        )
    provenance = text(root["provenance"], "provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError("provenance: expected fixture or native-adapter")
    if root["model_manifest_sha256"] != model.manifest_sha256:
        raise ValidationError("tensor movability model identity mismatch")
    if root["tensor_index_sha256"] != model.tensor_index_sha256:
        raise ValidationError("tensor movability tensor-index identity mismatch")
    if root["adapter_capabilities_sha256"] != adapter.fingerprint:
        raise ValidationError("tensor movability adapter capability identity mismatch")
    validate_llamacpp_package_identity(package)
    if root["package_sha256"] != package.fingerprint:
        raise ValidationError("tensor movability package identity mismatch")
    if root["qualified"] is not False or root["executable"] is not False:
        raise ValidationError("tensor movability profile cannot self-promote")

    exact_tensors = _flatten_index(tensor_index, model)
    config_devices = {d.id: d for d in config.devices}
    adapter_devices = {d.id: d for d in adapter.devices}

    identities = tuple(runtime_identities)
    if not identities:
        raise ValidationError("tensor movability requires runtime identities")
    by_device = {identity.tensormeld_device_id: identity for identity in identities}
    if len(by_device) != len(identities):
        raise ValidationError("duplicate runtime identity device")
    for device_id, identity in by_device.items():
        cfg = config_devices.get(device_id)
        exposed = adapter_devices.get(device_id)
        if cfg is None or exposed is None:
            raise ValidationError(
                f"runtime identity device {device_id!r} is not config/adapter exposed"
            )
        if identity.node_id != cfg.node or exposed.node != cfg.node:
            raise ValidationError(f"runtime identity node mismatch for {device_id}")
        if exposed.backend != cfg.backend:
            raise ValidationError(f"adapter backend mismatch for {device_id}")
        if identity.worker_artifact_sha256 != package.llama_server_sha256:
            raise ValidationError(
                "tensor movability runtime identity must use package llama-server artifact"
            )

    expected_runtime = tuple(
        by_device[device].identity_sha256 for device in sorted(by_device)
    )
    supplied_runtime = tuple(
        text(value, "runtime_identity_sha256[]")
        for value in items(
            root["runtime_identity_sha256"],
            "runtime_identity_sha256",
            MAX_RUNTIME_IDENTITIES,
            1,
        )
    )
    if supplied_runtime != expected_runtime:
        raise ValidationError("tensor movability runtime identity mismatch")

    classified: list[TensorMovability] = []
    for i, raw in enumerate(items(root["tensors"], "tensors", MAX_TENSORS, 1)):
        t = record(
            raw,
            f"tensors[{i}]",
            {
                "name",
                "n_bytes",
                "storage_class",
                "movement_class",
                "allowed_devices",
                "alias_group",
            },
        )
        name = text(t["name"], f"tensors[{i}].name")
        if name not in exact_tensors:
            raise ValidationError(f"tensors[{i}]: unknown GGUF tensor {name!r}")
        n_bytes = int(number(t["n_bytes"], f"tensors[{i}].n_bytes", 0, True))
        if n_bytes != exact_tensors[name]:
            raise ValidationError(f"tensors[{i}]: byte size differs from exact GGUF index")
        storage = text(t["storage_class"], f"tensors[{i}].storage_class")
        if storage not in STORAGE_CLASSES:
            raise ValidationError(f"tensors[{i}]: unsupported storage class")
        movement = text(t["movement_class"], f"tensors[{i}].movement_class")
        if movement not in MOVEMENT_CLASSES:
            raise ValidationError(f"tensors[{i}]: unsupported movement class")
        allowed = tuple(
            text(value, f"tensors[{i}].allowed_devices[]")
            for value in items(
                t["allowed_devices"],
                f"tensors[{i}].allowed_devices",
                MAX_ALLOWED_DEVICES,
                1,
            )
        )
        unique(list(allowed), f"tensors[{i}].allowed_devices")
        if not set(allowed) <= set(by_device):
            raise ValidationError(
                f"tensors[{i}]: allowed devices must have current runtime identities"
            )
        if movement == "pinned" and len(allowed) != 1:
            raise ValidationError(
                f"tensors[{i}]: pinned movement requires exactly one allowed device"
            )
        if movement == "owner_local":
            nodes = {config_devices[device].node for device in allowed}
            if len(nodes) != 1:
                raise ValidationError(
                    f"tensors[{i}]: owner_local devices must belong to one node"
                )
        alias = t["alias_group"]
        if alias is not None:
            alias = text(alias, f"tensors[{i}].alias_group")
        classified.append(
            TensorMovability(name, n_bytes, storage, movement, allowed, alias)
        )

    unique([t.name for t in classified], "tensors.name")
    classified_names = {t.name for t in classified}
    if classified_names != set(exact_tensors):
        missing = len(set(exact_tensors) - classified_names)
        extra = len(classified_names - set(exact_tensors))
        raise ValidationError(
            f"tensor movability must classify the complete tensor index "
            f"(missing={missing}, extra={extra})"
        )

    alias_groups: dict[str, list[TensorMovability]] = {}
    for tensor in classified:
        if tensor.alias_group is not None:
            alias_groups.setdefault(tensor.alias_group, []).append(tensor)
    for group, members in alias_groups.items():
        if len(members) < 2:
            raise ValidationError(
                f"alias_group {group!r} must contain at least two tensors"
            )
        if len({m.storage_class for m in members}) != 1:
            raise ValidationError(
                f"alias_group {group!r} must use one storage class"
            )
        if len({(m.movement_class, m.allowed_devices) for m in members}) != 1:
            raise ValidationError(
                f"alias_group {group!r} must share one movement policy"
            )

    canonical = {
        "tensor_movability_schema": MOVABILITY_SCHEMA,
        "provenance": provenance,
        "model_manifest_sha256": model.manifest_sha256,
        "tensor_index_sha256": model.tensor_index_sha256,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "package_sha256": package.fingerprint,
        "runtime_identity_sha256": list(expected_runtime),
        "tensors": [
            {
                "name": t.name,
                "n_bytes": t.n_bytes,
                "storage_class": t.storage_class,
                "movement_class": t.movement_class,
                "allowed_devices": list(t.allowed_devices),
                "alias_group": t.alias_group,
            }
            for t in sorted(classified, key=lambda item: item.name)
        ],
        "qualified": False,
        "executable": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return TensorMovabilityProfile(
        provenance,
        model.manifest_sha256,
        model.tensor_index_sha256,
        adapter.fingerprint,
        package.fingerprint,
        expected_runtime,
        tuple(sorted(classified, key=lambda item: item.name)),
        fingerprint,
    )


def tensor_movability_summary(profile: TensorMovabilityProfile) -> dict[str, Any]:
    counts = {
        movement: sum(t.movement_class == movement for t in profile.tensors)
        for movement in sorted(MOVEMENT_CLASSES)
    }
    bytes_by_storage = {
        storage: sum(
            t.n_bytes for t in profile.tensors if t.storage_class == storage
        )
        for storage in sorted(STORAGE_CLASSES)
    }
    return {
        "result_schema": "tensormeld/tensor-movability-validation-v1",
        "tensor_movability_sha256": profile.fingerprint,
        "provenance": profile.provenance,
        "model_manifest_sha256": profile.model_manifest_sha256,
        "tensor_index_sha256": profile.tensor_index_sha256,
        "adapter_capabilities_sha256": profile.adapter_capabilities_sha256,
        "package_sha256": profile.package_sha256,
        "runtime_identity_sha256": list(profile.runtime_identity_sha256),
        "tensor_count": len(profile.tensors),
        "movement_counts": counts,
        "bytes_by_storage_class": bytes_by_storage,
        "qualified": False,
        "executable": False,
        "warnings": [
            "Movability is explicit evidence; TensorMeld did not infer it from tensor names, sizes, backend labels or model family.",
            "This profile does not alter current runtime admission or planner ownership.",
            "Reclaimable/file-backed classification does not make bytes free or guarantee OS reclaim behavior.",
            "Fixture provenance is contract evidence only, not target-host movability measurement.",
        ],
    }


def load_tensor_movability_profile(
    path: str | Path,
    *,
    config: Config,
    model: ModelManifest,
    tensor_index: dict[str, Any],
    adapter: AdapterCapabilities,
    package: LlamaCppBuildPackage,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
) -> TensorMovabilityProfile:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("tensor movability profile exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid tensor movability JSON: {exc}") from exc
    return parse_tensor_movability_profile(
        value,
        config=config,
        model=model,
        tensor_index=tensor_index,
        adapter=adapter,
        package=package,
        runtime_identities=runtime_identities,
    )
