"""Exact directional path evidence with conservative payload buckets.

A path profile is explicit evidence, not a link-speed heuristic. Cost lookup uses the
first declared bucket whose max_payload_bytes covers the payload. TensorMeld never
interpolates, extrapolates, or sums parallel paths in this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config_v2 import Config
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

PATH_EVIDENCE_SCHEMA = "tensormeld/directional-path-evidence-v1"
MAX_PATHS = 4096
MAX_BUCKETS = 128
MAX_RUNTIME_IDENTITIES = 128
PROVENANCE_VALUES = {"fixture", "native-target"}


@dataclass(frozen=True)
class PayloadBucket:
    max_payload_bytes: int
    upper_bound_us: int


@dataclass(frozen=True)
class DirectionalPath:
    id: str
    source_device: str
    target_device: str
    transport: str
    physical_group: str
    buckets: tuple[PayloadBucket, ...]


@dataclass(frozen=True)
class DirectionalPathEvidence:
    provenance: str
    config_sha256: str
    runtime_identity_sha256: tuple[str, ...]
    paths: tuple[DirectionalPath, ...]
    fingerprint: str


def _runtime_map(
    config: Config,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
) -> dict[str, RuntimeIdentity]:
    identities = tuple(runtime_identities)
    if not identities:
        raise ValidationError("directional path evidence requires runtime identities")
    by_device = {identity.tensormeld_device_id: identity for identity in identities}
    if len(by_device) != len(identities):
        raise ValidationError("duplicate runtime identity device")
    cfg_devices = {device.id: device for device in config.devices}
    for device_id, identity in by_device.items():
        cfg = cfg_devices.get(device_id)
        if cfg is None:
            raise ValidationError(
                f"runtime identity references unknown config device {device_id}"
            )
        if identity.node_id != cfg.node:
            raise ValidationError(
                f"runtime identity node mismatch for {device_id}"
            )
    return by_device


def parse_directional_path_evidence(
    data: Any,
    *,
    config: Config,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
) -> DirectionalPathEvidence:
    root = record(
        data,
        "directional path evidence",
        {
            "path_evidence_schema",
            "provenance",
            "config_sha256",
            "runtime_identity_sha256",
            "paths",
            "qualified",
            "executable",
        },
    )
    if root["path_evidence_schema"] != PATH_EVIDENCE_SCHEMA:
        raise ValidationError(
            f"path_evidence_schema: expected {PATH_EVIDENCE_SCHEMA}"
        )
    provenance = text(root["provenance"], "provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError("provenance: expected fixture or native-target")
    if root["config_sha256"] != config.fingerprint:
        raise ValidationError("directional path evidence config identity mismatch")
    if root["qualified"] is not False or root["executable"] is not False:
        raise ValidationError("directional path evidence cannot self-promote")

    runtime_by_device = _runtime_map(config, runtime_identities)
    cfg_devices = {device.id: device for device in config.devices}
    parsed: list[DirectionalPath] = []
    endpoint_devices: set[str] = set()

    for i, raw in enumerate(items(root["paths"], "paths", MAX_PATHS, 1)):
        p = record(
            raw,
            f"paths[{i}]",
            {
                "id",
                "source_device",
                "target_device",
                "transport",
                "physical_group",
                "buckets",
            },
        )
        path_id = text(p["id"], f"paths[{i}].id")
        source = text(p["source_device"], f"paths[{i}].source_device")
        target = text(p["target_device"], f"paths[{i}].target_device")
        if source == target:
            raise ValidationError(f"paths[{i}]: source and target must differ")
        if source not in cfg_devices or target not in cfg_devices:
            raise ValidationError(f"paths[{i}]: unknown config device endpoint")
        if source not in runtime_by_device or target not in runtime_by_device:
            raise ValidationError(
                f"paths[{i}]: both endpoints require current runtime identities"
            )
        endpoint_devices |= {source, target}

        buckets: list[PayloadBucket] = []
        previous_payload = 0
        previous_cost = 0
        for j, raw_bucket in enumerate(
            items(p["buckets"], f"paths[{i}].buckets", MAX_BUCKETS, 1)
        ):
            b = record(
                raw_bucket,
                f"paths[{i}].buckets[{j}]",
                {"max_payload_bytes", "upper_bound_us"},
            )
            payload = int(number(
                b["max_payload_bytes"],
                f"paths[{i}].buckets[{j}].max_payload_bytes",
                1,
                True,
            ))
            cost = int(number(
                b["upper_bound_us"],
                f"paths[{i}].buckets[{j}].upper_bound_us",
                1,
                True,
            ))
            if payload <= previous_payload:
                raise ValidationError(
                    f"paths[{i}].buckets: max_payload_bytes must strictly increase"
                )
            if cost < previous_cost:
                raise ValidationError(
                    f"paths[{i}].buckets: upper_bound_us must be nondecreasing"
                )
            previous_payload = payload
            previous_cost = cost
            buckets.append(PayloadBucket(payload, cost))

        parsed.append(DirectionalPath(
            path_id,
            source,
            target,
            text(p["transport"], f"paths[{i}].transport"),
            text(p["physical_group"], f"paths[{i}].physical_group"),
            tuple(buckets),
        ))

    unique([path.id for path in parsed], "paths.id")

    expected_runtime = tuple(
        runtime_by_device[device].identity_sha256
        for device in sorted(endpoint_devices)
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
        raise ValidationError(
            "directional path evidence runtime identity set/order mismatch"
        )

    canonical = {
        "path_evidence_schema": PATH_EVIDENCE_SCHEMA,
        "provenance": provenance,
        "config_sha256": config.fingerprint,
        "runtime_identity_sha256": list(expected_runtime),
        "paths": [
            {
                "id": path.id,
                "source_device": path.source_device,
                "target_device": path.target_device,
                "transport": path.transport,
                "physical_group": path.physical_group,
                "buckets": [
                    {
                        "max_payload_bytes": bucket.max_payload_bytes,
                        "upper_bound_us": bucket.upper_bound_us,
                    }
                    for bucket in path.buckets
                ],
            }
            for path in sorted(parsed, key=lambda item: item.id)
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
    return DirectionalPathEvidence(
        provenance,
        config.fingerprint,
        expected_runtime,
        tuple(sorted(parsed, key=lambda item: item.id)),
        fingerprint,
    )


def lookup_transfer_upper_bound(
    evidence: DirectionalPathEvidence,
    *,
    source_device: str,
    target_device: str,
    payload_bytes: int,
) -> dict[str, Any] | None:
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise ValidationError("payload_bytes must be a nonnegative integer")
    if source_device == target_device:
        return {
            "source_device": source_device,
            "target_device": target_device,
            "payload_bytes": payload_bytes,
            "path_id": None,
            "transport": "local",
            "physical_group": None,
            "upper_bound_us": 0,
        }
    if payload_bytes == 0:
        return {
            "source_device": source_device,
            "target_device": target_device,
            "payload_bytes": 0,
            "path_id": None,
            "transport": "none",
            "physical_group": None,
            "upper_bound_us": 0,
        }

    choices: list[tuple[int, str, DirectionalPath]] = []
    for path in evidence.paths:
        if (
            path.source_device != source_device
            or path.target_device != target_device
        ):
            continue
        bucket = next(
            (
                item for item in path.buckets
                if payload_bytes <= item.max_payload_bytes
            ),
            None,
        )
        if bucket is None:
            continue
        choices.append((bucket.upper_bound_us, path.id, path))

    if not choices:
        return None
    cost, _, path = min(choices, key=lambda item: (item[0], item[1]))
    return {
        "source_device": source_device,
        "target_device": target_device,
        "payload_bytes": payload_bytes,
        "path_id": path.id,
        "transport": path.transport,
        "physical_group": path.physical_group,
        "upper_bound_us": cost,
    }


def directional_path_evidence_summary(
    evidence: DirectionalPathEvidence,
) -> dict[str, Any]:
    return {
        "result_schema": "tensormeld/directional-path-evidence-validation-v1",
        "directional_path_evidence_sha256": evidence.fingerprint,
        "provenance": evidence.provenance,
        "config_sha256": evidence.config_sha256,
        "runtime_identity_sha256": list(evidence.runtime_identity_sha256),
        "path_count": len(evidence.paths),
        "paths": [
            {
                "id": path.id,
                "source_device": path.source_device,
                "target_device": path.target_device,
                "transport": path.transport,
                "physical_group": path.physical_group,
                "buckets": [bucket.__dict__ for bucket in path.buckets],
            }
            for path in evidence.paths
        ],
        "qualified": False,
        "executable": False,
        "warnings": [
            "Directional path evidence is explicit; nominal link speed is not used as effective bandwidth.",
            "Lookup uses conservative declared upper-bound buckets only; no interpolation or extrapolation.",
            "Parallel paths are alternatives and are never summed as multirail bandwidth by this contract.",
            "Path evidence does not authorize finer-grained native execution.",
        ],
    }


def load_directional_path_evidence(
    path: str | Path,
    *,
    config: Config,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
) -> DirectionalPathEvidence:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("directional path evidence exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid directional path evidence JSON: {exc}") from exc
    return parse_directional_path_evidence(
        value,
        config=config,
        runtime_identities=runtime_identities,
    )
