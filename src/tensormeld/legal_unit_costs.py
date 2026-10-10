"""Exact per-legal-unit compute/memory/boundary evidence.

This contract is a prerequisite for performance-aware legal-unit planning. It does not
rank candidates or authorize execution. Every legal device for every legal unit must have
an explicit profile; TensorMeld does not infer costs from tensor size, names, model family,
or backend labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config_v2 import Config
from .legal_model_units import LegalModelUnitsProfile
from .runtime_model_manifest import RuntimeModelManifest
from .tensor_movability import TensorMovabilityProfile
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

LEGAL_UNIT_COSTS_SCHEMA = "tensormeld/legal-unit-costs-v1"
MAX_UNITS = 100_000
MAX_DEVICE_PROFILES = 128
MAX_POOLS_PER_CLASS = 192
PROVENANCE_VALUES = {"fixture", "native-adapter"}


@dataclass(frozen=True)
class UnitDeviceCost:
    device: str
    compute_us: int
    persistent_state: tuple[tuple[str, int], ...]
    workspace_peak: tuple[tuple[str, int], ...]
    staging_peak: tuple[tuple[str, int], ...]

    def memory_map(self, field: str) -> dict[str, int]:
        if field == "persistent_state":
            return dict(self.persistent_state)
        if field == "workspace_peak":
            return dict(self.workspace_peak)
        if field == "staging_peak":
            return dict(self.staging_peak)
        raise KeyError(field)


@dataclass(frozen=True)
class LegalUnitCost:
    unit_id: str
    sequence: int
    boundary_output_bytes: int
    device_profiles: tuple[UnitDeviceCost, ...]


@dataclass(frozen=True)
class LegalUnitCostsProfile:
    provenance: str
    config_sha256: str
    legal_model_units_sha256: str
    runtime_manifest_sha256: str
    units: tuple[LegalUnitCost, ...]
    fingerprint: str


def _pool_map(
    value: Any,
    *,
    where: str,
    device_node: str,
    config: Config,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, dict) or len(value) > MAX_POOLS_PER_CLASS:
        raise ValidationError(f"{where}: expected bounded physical-pool byte map")
    pools = {pool.id: pool for pool in config.pools}
    result: list[tuple[str, int]] = []
    for pool_id, raw_bytes in value.items():
        pool_id = text(pool_id, f"{where}.pool_ref")
        pool = pools.get(pool_id)
        if pool is None:
            raise ValidationError(f"{where}: unknown physical pool {pool_id}")
        if pool.node != device_node:
            raise ValidationError(
                f"{where}: physical pool {pool_id} is not local to device node {device_node}"
            )
        amount = int(number(raw_bytes, f"{where}.{pool_id}", 0, True))
        if pool.reported_capacity_bytes is not None and amount > pool.reported_capacity_bytes:
            raise ValidationError(
                f"{where}: bytes exceed reported capacity for {pool_id}"
            )
        result.append((pool_id, amount))
    result.sort()
    return tuple(result)


def parse_legal_unit_costs(
    data: Any,
    *,
    config: Config,
    legal_units: LegalModelUnitsProfile,
    runtime_manifest: RuntimeModelManifest,
    movability: TensorMovabilityProfile,
) -> LegalUnitCostsProfile:
    root = record(
        data,
        "legal unit costs",
        {
            "legal_unit_costs_schema",
            "provenance",
            "config_sha256",
            "legal_model_units_sha256",
            "runtime_manifest_sha256",
            "units",
            "qualified",
            "executable",
        },
    )
    if root["legal_unit_costs_schema"] != LEGAL_UNIT_COSTS_SCHEMA:
        raise ValidationError(
            f"legal_unit_costs_schema: expected {LEGAL_UNIT_COSTS_SCHEMA}"
        )
    provenance = text(root["provenance"], "provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError("provenance: expected fixture or native-adapter")
    if provenance != legal_units.provenance:
        raise ValidationError("cost provenance must match legal-unit provenance")
    if provenance != runtime_manifest.provenance:
        raise ValidationError("cost provenance must match runtime-manifest provenance")
    if root["config_sha256"] != config.fingerprint:
        raise ValidationError("legal-unit costs config identity mismatch")
    if root["legal_model_units_sha256"] != legal_units.fingerprint:
        raise ValidationError("legal-unit costs legal-unit identity mismatch")
    if root["runtime_manifest_sha256"] != runtime_manifest.fingerprint:
        raise ValidationError("legal-unit costs runtime-manifest identity mismatch")
    if legal_units.tensor_movability_sha256 != movability.fingerprint:
        raise ValidationError("legal-unit costs tensor-movability identity mismatch")
    if movability.model_manifest_sha256 != runtime_manifest.model_manifest_sha256:
        raise ValidationError("legal-unit costs model identity mismatch")
    if (
        movability.adapter_capabilities_sha256
        != runtime_manifest.adapter_capabilities_sha256
    ):
        raise ValidationError("legal-unit costs adapter capability identity mismatch")
    if root["qualified"] is not False or root["executable"] is not False:
        raise ValidationError("legal-unit costs cannot self-promote")

    legal_by_id = {unit.id: unit for unit in legal_units.units}
    config_devices = {device.id: device for device in config.devices}
    runtime_devices = {device.id: device for device in runtime_manifest.devices}
    parsed: list[LegalUnitCost] = []

    for i, raw in enumerate(items(root["units"], "units", MAX_UNITS, 1)):
        unit_raw = record(
            raw,
            f"units[{i}]",
            {"id", "sequence", "boundary_output_bytes", "device_profiles"},
        )
        unit_id = text(unit_raw["id"], f"units[{i}].id")
        legal = legal_by_id.get(unit_id)
        if legal is None:
            raise ValidationError(f"units[{i}]: unknown legal unit {unit_id}")
        sequence = int(number(unit_raw["sequence"], f"units[{i}].sequence", 0, True))
        if sequence != legal.sequence:
            raise ValidationError(f"units[{i}]: sequence mismatch")
        boundary = int(number(
            unit_raw["boundary_output_bytes"],
            f"units[{i}].boundary_output_bytes",
            0,
            True,
        ))
        if legal.sequence == len(legal_units.units) - 1 and boundary != 0:
            raise ValidationError("last legal unit boundary_output_bytes must be zero")

        profiles_raw = unit_raw["device_profiles"]
        if not isinstance(profiles_raw, dict) or not 1 <= len(profiles_raw) <= MAX_DEVICE_PROFILES:
            raise ValidationError(f"units[{i}].device_profiles: expected bounded object")
        if set(profiles_raw) != set(legal.allowed_devices):
            raise ValidationError(
                f"units[{i}].device_profiles must cover exact legal allowed_devices"
            )

        device_profiles: list[UnitDeviceCost] = []
        for device_id in sorted(profiles_raw):
            cfg_device = config_devices.get(device_id)
            runtime_device = runtime_devices.get(device_id)
            if cfg_device is None or runtime_device is None:
                raise ValidationError(
                    f"units[{i}].device_profiles: device {device_id} missing from config/runtime manifest"
                )
            if (
                runtime_device.node != cfg_device.node
                or runtime_device.backend != cfg_device.backend
            ):
                raise ValidationError(
                    f"units[{i}].device_profiles: config/runtime identity mismatch for {device_id}"
                )
            profile_raw = record(
                profiles_raw[device_id],
                f"units[{i}].device_profiles.{device_id}",
                {
                    "compute_us",
                    "persistent_state_bytes",
                    "workspace_peak_bytes",
                    "staging_peak_bytes",
                },
            )
            compute_us = int(number(
                profile_raw["compute_us"],
                f"units[{i}].device_profiles.{device_id}.compute_us",
                1,
                True,
            ))
            device_profiles.append(UnitDeviceCost(
                device_id,
                compute_us,
                _pool_map(
                    profile_raw["persistent_state_bytes"],
                    where=f"units[{i}].device_profiles.{device_id}.persistent_state_bytes",
                    device_node=cfg_device.node,
                    config=config,
                ),
                _pool_map(
                    profile_raw["workspace_peak_bytes"],
                    where=f"units[{i}].device_profiles.{device_id}.workspace_peak_bytes",
                    device_node=cfg_device.node,
                    config=config,
                ),
                _pool_map(
                    profile_raw["staging_peak_bytes"],
                    where=f"units[{i}].device_profiles.{device_id}.staging_peak_bytes",
                    device_node=cfg_device.node,
                    config=config,
                ),
            ))

        parsed.append(LegalUnitCost(
            unit_id,
            sequence,
            boundary,
            tuple(device_profiles),
        ))

    unique([unit.unit_id for unit in parsed], "units.id")
    if set(unit.unit_id for unit in parsed) != set(legal_by_id):
        raise ValidationError("legal-unit costs must cover every legal unit exactly once")
    parsed.sort(key=lambda item: item.sequence)

    canonical = {
        "legal_unit_costs_schema": LEGAL_UNIT_COSTS_SCHEMA,
        "provenance": provenance,
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal_units.fingerprint,
        "runtime_manifest_sha256": runtime_manifest.fingerprint,
        "units": [
            {
                "id": unit.unit_id,
                "sequence": unit.sequence,
                "boundary_output_bytes": unit.boundary_output_bytes,
                "device_profiles": {
                    profile.device: {
                        "compute_us": profile.compute_us,
                        "persistent_state_bytes": dict(profile.persistent_state),
                        "workspace_peak_bytes": dict(profile.workspace_peak),
                        "staging_peak_bytes": dict(profile.staging_peak),
                    }
                    for profile in unit.device_profiles
                },
            }
            for unit in parsed
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
    return LegalUnitCostsProfile(
        provenance,
        config.fingerprint,
        legal_units.fingerprint,
        runtime_manifest.fingerprint,
        tuple(parsed),
        fingerprint,
    )


def legal_unit_costs_summary(profile: LegalUnitCostsProfile) -> dict[str, Any]:
    return {
        "result_schema": "tensormeld/legal-unit-costs-validation-v1",
        "legal_unit_costs_sha256": profile.fingerprint,
        "config_sha256": profile.config_sha256,
        "legal_model_units_sha256": profile.legal_model_units_sha256,
        "runtime_manifest_sha256": profile.runtime_manifest_sha256,
        "provenance": profile.provenance,
        "unit_count": len(profile.units),
        "units": [
            {
                "id": unit.unit_id,
                "sequence": unit.sequence,
                "boundary_output_bytes": unit.boundary_output_bytes,
                "device_profiles": [
                    {
                        "device": p.device,
                        "compute_us": p.compute_us,
                        "persistent_state_bytes": dict(p.persistent_state),
                        "workspace_peak_bytes": dict(p.workspace_peak),
                        "staging_peak_bytes": dict(p.staging_peak),
                    }
                    for p in unit.device_profiles
                ],
            }
            for unit in profile.units
        ],
        "qualified": False,
        "executable": False,
        "warnings": [
            "Costs are explicit adapter evidence; TensorMeld did not infer them from tensor size, names, model family or backend labels.",
            "persistent_state_bytes are additive across assigned legal units.",
            "workspace_peak_bytes and staging_peak_bytes are intended as per-device/pool maxima across assigned legal units.",
            "boundary_output_bytes describes payload size only; transfer time still requires explicit directional path evidence.",
            "This contract does not authorize finer-grained native execution or performance claims.",
        ],
    }


def load_legal_unit_costs(
    path: str | Path,
    *,
    config: Config,
    legal_units: LegalModelUnitsProfile,
    runtime_manifest: RuntimeModelManifest,
    movability: TensorMovabilityProfile,
) -> LegalUnitCostsProfile:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("legal-unit costs exceed 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid legal-unit costs JSON: {exc}") from exc
    return parse_legal_unit_costs(
        value,
        config=config,
        legal_units=legal_units,
        runtime_manifest=runtime_manifest,
        movability=movability,
    )
