"""Explicit ordered legal model units and cut boundaries.

The adapter/evidence producer declares execution order and indivisible tensor groups.
TensorMeld does not infer units or cut points from tensor names, layer numbering or model
family. This contract is advisory evidence only and does not enable planner execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

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
from .tensor_movability import TensorMovabilityProfile

LEGAL_UNITS_SCHEMA = "tensormeld/legal-model-units-v1"
MAX_UNITS = 100_000
MAX_TENSORS_PER_UNIT = 100_000
MAX_DEVICES_PER_UNIT = 128
PROVENANCE_VALUES = {"fixture", "native-adapter"}


@dataclass(frozen=True)
class LegalModelUnit:
    id: str
    sequence: int
    tensors: tuple[str, ...]
    allowed_devices: tuple[str, ...]
    cut_after: bool
    hard_resident_bytes: int
    reclaimable_file_backed_bytes: int


@dataclass(frozen=True)
class LegalModelUnitsProfile:
    provenance: str
    tensor_movability_sha256: str
    units: tuple[LegalModelUnit, ...]
    fingerprint: str


def parse_legal_model_units(
    data: Any,
    *,
    movability: TensorMovabilityProfile,
) -> LegalModelUnitsProfile:
    root = record(
        data,
        "legal model units",
        {
            "legal_units_schema",
            "provenance",
            "tensor_movability_sha256",
            "units",
            "qualified",
            "executable",
        },
    )
    if root["legal_units_schema"] != LEGAL_UNITS_SCHEMA:
        raise ValidationError(f"legal_units_schema: expected {LEGAL_UNITS_SCHEMA}")
    provenance = text(root["provenance"], "provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError("provenance: expected fixture or native-adapter")
    if provenance != movability.provenance:
        raise ValidationError(
            "legal-unit provenance must match tensor movability provenance"
        )
    if root["tensor_movability_sha256"] != movability.fingerprint:
        raise ValidationError("legal units tensor-movability identity mismatch")
    if root["qualified"] is not False or root["executable"] is not False:
        raise ValidationError("legal model units cannot self-promote")

    tensor_map = {tensor.name: tensor for tensor in movability.tensors}
    raw_units = items(root["units"], "units", MAX_UNITS, 1)
    parsed: list[LegalModelUnit] = []
    tensor_owner: dict[str, str] = {}

    for i, raw in enumerate(raw_units):
        unit = record(
            raw,
            f"units[{i}]",
            {"id", "sequence", "tensors", "allowed_devices", "cut_after"},
        )
        unit_id = text(unit["id"], f"units[{i}].id")
        sequence = int(number(unit["sequence"], f"units[{i}].sequence", 0, True))
        tensor_names = tuple(
            text(value, f"units[{i}].tensors[]")
            for value in items(
                unit["tensors"],
                f"units[{i}].tensors",
                MAX_TENSORS_PER_UNIT,
                1,
            )
        )
        unique(list(tensor_names), f"units[{i}].tensors")
        unknown = [name for name in tensor_names if name not in tensor_map]
        if unknown:
            raise ValidationError(
                f"units[{i}]: unknown tensor(s): {', '.join(sorted(unknown)[:3])}"
            )
        for name in tensor_names:
            previous = tensor_owner.get(name)
            if previous is not None:
                raise ValidationError(
                    f"tensor {name!r} appears in both {previous!r} and {unit_id!r}"
                )
            tensor_owner[name] = unit_id

        allowed = tuple(
            text(value, f"units[{i}].allowed_devices[]")
            for value in items(
                unit["allowed_devices"],
                f"units[{i}].allowed_devices",
                MAX_DEVICES_PER_UNIT,
                1,
            )
        )
        unique(list(allowed), f"units[{i}].allowed_devices")
        legal_intersection = set(tensor_map[tensor_names[0]].allowed_devices)
        for name in tensor_names[1:]:
            legal_intersection &= set(tensor_map[name].allowed_devices)
        if not legal_intersection:
            raise ValidationError(
                f"units[{i}]: member tensors have no common legal device"
            )
        if not set(allowed) <= legal_intersection:
            raise ValidationError(
                f"units[{i}]: allowed_devices exceed tensor-level legal intersection"
            )
        if type(unit["cut_after"]) is not bool:
            raise ValidationError(f"units[{i}].cut_after: expected boolean")

        hard = sum(
            tensor_map[name].n_bytes
            for name in tensor_names
            if tensor_map[name].storage_class == "hard_resident"
        )
        reclaimable = sum(
            tensor_map[name].n_bytes
            for name in tensor_names
            if tensor_map[name].storage_class == "reclaimable_file_backed"
        )
        parsed.append(
            LegalModelUnit(
                unit_id,
                sequence,
                tuple(sorted(tensor_names)),
                tuple(sorted(allowed)),
                unit["cut_after"],
                hard,
                reclaimable,
            )
        )

    unique([unit.id for unit in parsed], "units.id")
    unique([unit.sequence for unit in parsed], "units.sequence")
    expected_sequence = list(range(len(parsed)))
    actual_sequence = sorted(unit.sequence for unit in parsed)
    if actual_sequence != expected_sequence:
        raise ValidationError("unit sequence must be contiguous 0..N-1")

    if set(tensor_owner) != set(tensor_map):
        missing = set(tensor_map) - set(tensor_owner)
        raise ValidationError(
            f"legal units must cover every tensor exactly once (missing={len(missing)})"
        )

    parsed.sort(key=lambda unit: unit.sequence)
    if parsed[-1].cut_after:
        raise ValidationError("last legal unit cannot declare cut_after=true")

    unit_by_tensor = {
        tensor_name: unit.id
        for unit in parsed
        for tensor_name in unit.tensors
    }
    alias_groups: dict[str, set[str]] = {}
    for tensor in movability.tensors:
        if tensor.alias_group is not None:
            alias_groups.setdefault(tensor.alias_group, set()).add(tensor.name)
    for group, members in alias_groups.items():
        owners = {unit_by_tensor[name] for name in members}
        if len(owners) != 1:
            raise ValidationError(
                f"alias_group {group!r} must remain inside one indivisible legal unit"
            )

    canonical = {
        "legal_units_schema": LEGAL_UNITS_SCHEMA,
        "provenance": provenance,
        "tensor_movability_sha256": movability.fingerprint,
        "units": [
            {
                "id": unit.id,
                "sequence": unit.sequence,
                "tensors": list(unit.tensors),
                "allowed_devices": list(unit.allowed_devices),
                "cut_after": unit.cut_after,
                "hard_resident_bytes": unit.hard_resident_bytes,
                "reclaimable_file_backed_bytes":
                    unit.reclaimable_file_backed_bytes,
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
    return LegalModelUnitsProfile(
        provenance,
        movability.fingerprint,
        tuple(parsed),
        fingerprint,
    )


def legal_model_units_summary(
    profile: LegalModelUnitsProfile,
) -> dict[str, Any]:
    return {
        "result_schema": "tensormeld/legal-model-units-validation-v1",
        "legal_model_units_sha256": profile.fingerprint,
        "tensor_movability_sha256": profile.tensor_movability_sha256,
        "provenance": profile.provenance,
        "unit_count": len(profile.units),
        "cut_after_sequences": [
            unit.sequence for unit in profile.units if unit.cut_after
        ],
        "units": [
            {
                "id": unit.id,
                "sequence": unit.sequence,
                "tensor_count": len(unit.tensors),
                "allowed_devices": list(unit.allowed_devices),
                "cut_after": unit.cut_after,
                "hard_resident_bytes": unit.hard_resident_bytes,
                "reclaimable_file_backed_bytes":
                    unit.reclaimable_file_backed_bytes,
            }
            for unit in profile.units
        ],
        "qualified": False,
        "executable": False,
        "warnings": [
            "Unit order, grouping and cuts are explicit evidence; TensorMeld did not infer them from tensor names or model family.",
            "Allowed devices may only narrow the tensor-level legal intersection.",
            "This contract does not enable fine-grained planner or runtime execution.",
            "Whole-block placement remains the executable baseline until a native adapter validates finer-grained semantics.",
        ],
    }


def load_legal_model_units(
    path: str | Path,
    *,
    movability: TensorMovabilityProfile,
) -> LegalModelUnitsProfile:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("legal model units exceed 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid legal model units JSON: {exc}") from exc
    return parse_legal_model_units(value, movability=movability)
