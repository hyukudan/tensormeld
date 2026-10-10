"""Exact generation-phase and sampling evidence for legal model units.

This contract separates prefill, one decode step, sampling, logits movement and token
feedback. It is scenario-specific evidence: the prefill token count and decode context
position are explicit and cannot be reused silently for another context.

Parsing does not rank placements, authorize execution, or claim TTFT/token latency.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config_v2 import Config
from .legal_model_units import LegalModelUnitsProfile
from .legal_unit_costs import LegalUnitCostsProfile
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

PHASE_EVIDENCE_SCHEMA_V1 = "tensormeld/generation-phase-evidence-v1"
PHASE_EVIDENCE_SCHEMA_V2 = "tensormeld/generation-phase-evidence-v2"
PHASE_EVIDENCE_SCHEMAS = {PHASE_EVIDENCE_SCHEMA_V1, PHASE_EVIDENCE_SCHEMA_V2}
MAX_UNITS = 100_000
MAX_DEVICE_PROFILES = 128
MAX_SAMPLERS = 128
PROVENANCE_VALUES = {"fixture", "native-adapter"}


@dataclass(frozen=True)
class UnitPhaseCost:
    device: str
    prefill_us: int
    decode_step_us: int


@dataclass(frozen=True)
class UnitPhaseEvidence:
    unit_id: str
    sequence: int
    device_profiles: tuple[UnitPhaseCost, ...]
    prefill_boundary_output_bytes: int | None = None
    decode_boundary_output_bytes: int | None = None


@dataclass(frozen=True)
class SamplingProfile:
    device: str
    sampling_us: int
    logits_payload_bytes: int
    feedback_payload_bytes: int


@dataclass(frozen=True)
class GenerationPhaseEvidence:
    schema: str
    provenance: str
    config_sha256: str
    legal_model_units_sha256: str
    legal_unit_costs_sha256: str
    runtime_manifest_sha256: str
    prefill_tokens: int
    decode_context_tokens: int
    concurrency: int
    units: tuple[UnitPhaseEvidence, ...]
    sampling_profiles: tuple[SamplingProfile, ...]
    fingerprint: str


def parse_generation_phase_evidence(
    data: Any,
    *,
    config: Config,
    legal_units: LegalModelUnitsProfile,
    costs: LegalUnitCostsProfile,
    runtime_manifest: RuntimeModelManifest,
    movability: TensorMovabilityProfile,
) -> GenerationPhaseEvidence:
    root = record(
        data,
        "generation phase evidence",
        {
            "generation_phase_schema",
            "provenance",
            "config_sha256",
            "legal_model_units_sha256",
            "legal_unit_costs_sha256",
            "runtime_manifest_sha256",
            "phase_workload",
            "units",
            "sampling_profiles",
            "qualified",
            "executable",
        },
    )
    phase_schema = text(root["generation_phase_schema"], "generation_phase_schema")
    if phase_schema not in PHASE_EVIDENCE_SCHEMAS:
        raise ValidationError(
            "generation_phase_schema: expected v1 or v2 generation phase evidence"
        )
    provenance = text(root["provenance"], "provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError("provenance: expected fixture or native-adapter")
    if provenance != legal_units.provenance:
        raise ValidationError("phase provenance must match legal-unit provenance")
    if provenance != costs.provenance:
        raise ValidationError("phase provenance must match legal-unit cost provenance")
    if provenance != runtime_manifest.provenance:
        raise ValidationError("phase provenance must match runtime-manifest provenance")
    if root["config_sha256"] != config.fingerprint:
        raise ValidationError("generation phase config identity mismatch")
    if root["legal_model_units_sha256"] != legal_units.fingerprint:
        raise ValidationError("generation phase legal-unit identity mismatch")
    if root["legal_unit_costs_sha256"] != costs.fingerprint:
        raise ValidationError("generation phase legal-unit cost identity mismatch")
    if root["runtime_manifest_sha256"] != runtime_manifest.fingerprint:
        raise ValidationError("generation phase runtime-manifest identity mismatch")
    if legal_units.tensor_movability_sha256 != movability.fingerprint:
        raise ValidationError("generation phase tensor-movability identity mismatch")
    if costs.legal_model_units_sha256 != legal_units.fingerprint:
        raise ValidationError("generation phase cost/legal-unit identity mismatch")
    if costs.runtime_manifest_sha256 != runtime_manifest.fingerprint:
        raise ValidationError("generation phase cost/runtime identity mismatch")
    if movability.model_manifest_sha256 != runtime_manifest.model_manifest_sha256:
        raise ValidationError("generation phase model identity mismatch")
    if (
        movability.adapter_capabilities_sha256
        != runtime_manifest.adapter_capabilities_sha256
    ):
        raise ValidationError("generation phase adapter capability identity mismatch")
    if root["qualified"] is not False or root["executable"] is not False:
        raise ValidationError("generation phase evidence cannot self-promote")

    workload_raw = record(
        root["phase_workload"],
        "phase_workload",
        {"prefill_tokens", "decode_context_tokens", "concurrency"},
    )
    prefill_tokens = int(number(
        workload_raw["prefill_tokens"],
        "phase_workload.prefill_tokens",
        1,
        True,
    ))
    decode_context_tokens = int(number(
        workload_raw["decode_context_tokens"],
        "phase_workload.decode_context_tokens",
        1,
        True,
    ))
    concurrency = int(number(
        workload_raw["concurrency"],
        "phase_workload.concurrency",
        1,
        True,
    ))
    profile = config.profile_map.get(runtime_manifest.profile)
    if profile is None:
        raise ValidationError("generation phase runtime profile is unknown")
    if concurrency != runtime_manifest.workload["concurrency"]:
        raise ValidationError("generation phase concurrency differs from runtime manifest")
    if prefill_tokens > profile.workload.context_tokens:
        raise ValidationError("prefill_tokens exceed configured context")
    if decode_context_tokens > profile.workload.context_tokens:
        raise ValidationError("decode_context_tokens exceed configured context")
    if prefill_tokens + profile.workload.max_output_tokens > profile.workload.context_tokens:
        raise ValidationError(
            "prefill_tokens plus configured max output exceed configured context"
        )

    legal_by_id = {unit.id: unit for unit in legal_units.units}
    parsed_units: list[UnitPhaseEvidence] = []
    for i, raw in enumerate(items(root["units"], "units", MAX_UNITS, 1)):
        unit_fields = {"id", "sequence", "device_profiles"}
        if phase_schema == PHASE_EVIDENCE_SCHEMA_V2:
            unit_fields |= {
                "prefill_boundary_output_bytes",
                "decode_boundary_output_bytes",
            }
        unit_raw = record(
            raw,
            f"units[{i}]",
            unit_fields,
        )
        unit_id = text(unit_raw["id"], f"units[{i}].id")
        legal = legal_by_id.get(unit_id)
        if legal is None:
            raise ValidationError(f"units[{i}]: unknown legal unit {unit_id}")
        sequence = int(number(
            unit_raw["sequence"],
            f"units[{i}].sequence",
            0,
            True,
        ))
        if sequence != legal.sequence:
            raise ValidationError(f"units[{i}]: sequence mismatch")
        profiles_raw = unit_raw["device_profiles"]
        if (
            not isinstance(profiles_raw, dict)
            or not 1 <= len(profiles_raw) <= MAX_DEVICE_PROFILES
        ):
            raise ValidationError(
                f"units[{i}].device_profiles: expected bounded object"
            )
        if set(profiles_raw) != set(legal.allowed_devices):
            raise ValidationError(
                f"units[{i}].device_profiles must cover exact legal allowed_devices"
            )
        profiles: list[UnitPhaseCost] = []
        for device_id in sorted(profiles_raw):
            p = record(
                profiles_raw[device_id],
                f"units[{i}].device_profiles.{device_id}",
                {"prefill_us", "decode_step_us"},
            )
            profiles.append(UnitPhaseCost(
                device_id,
                int(number(
                    p["prefill_us"],
                    f"units[{i}].device_profiles.{device_id}.prefill_us",
                    1,
                    True,
                )),
                int(number(
                    p["decode_step_us"],
                    f"units[{i}].device_profiles.{device_id}.decode_step_us",
                    1,
                    True,
                )),
            ))
        prefill_boundary = None
        decode_boundary = None
        if phase_schema == PHASE_EVIDENCE_SCHEMA_V2:
            prefill_boundary = int(number(
                unit_raw["prefill_boundary_output_bytes"],
                f"units[{i}].prefill_boundary_output_bytes",
                0,
                True,
            ))
            decode_boundary = int(number(
                unit_raw["decode_boundary_output_bytes"],
                f"units[{i}].decode_boundary_output_bytes",
                0,
                True,
            ))
            if sequence == len(legal_units.units) - 1 and (
                prefill_boundary != 0 or decode_boundary != 0
            ):
                raise ValidationError(
                    "last legal unit phase boundary payloads must both be zero"
                )
        parsed_units.append(UnitPhaseEvidence(
            unit_id,
            sequence,
            tuple(profiles),
            prefill_boundary,
            decode_boundary,
        ))
    unique([unit.unit_id for unit in parsed_units], "units.id")
    if set(unit.unit_id for unit in parsed_units) != set(legal_by_id):
        raise ValidationError(
            "generation phase evidence must cover every legal unit exactly once"
        )
    parsed_units.sort(key=lambda item: item.sequence)

    runtime_devices = {device.id: device for device in runtime_manifest.devices}
    sampler_raw = root["sampling_profiles"]
    if (
        not isinstance(sampler_raw, dict)
        or not 1 <= len(sampler_raw) <= MAX_SAMPLERS
    ):
        raise ValidationError("sampling_profiles: expected bounded non-empty object")
    samplers: list[SamplingProfile] = []
    for device_id in sorted(sampler_raw):
        if device_id not in runtime_devices:
            raise ValidationError(
                f"sampling_profiles: device {device_id} is not in runtime manifest"
            )
        p = record(
            sampler_raw[device_id],
            f"sampling_profiles.{device_id}",
            {"sampling_us", "logits_payload_bytes", "feedback_payload_bytes"},
        )
        samplers.append(SamplingProfile(
            device_id,
            int(number(
                p["sampling_us"],
                f"sampling_profiles.{device_id}.sampling_us",
                1,
                True,
            )),
            int(number(
                p["logits_payload_bytes"],
                f"sampling_profiles.{device_id}.logits_payload_bytes",
                0,
                True,
            )),
            int(number(
                p["feedback_payload_bytes"],
                f"sampling_profiles.{device_id}.feedback_payload_bytes",
                0,
                True,
            )),
        ))

    canonical_units = []
    for unit in parsed_units:
        item = {
            "id": unit.unit_id,
            "sequence": unit.sequence,
            "device_profiles": {
                p.device: {
                    "prefill_us": p.prefill_us,
                    "decode_step_us": p.decode_step_us,
                }
                for p in unit.device_profiles
            },
        }
        if phase_schema == PHASE_EVIDENCE_SCHEMA_V2:
            item["prefill_boundary_output_bytes"] = (
                unit.prefill_boundary_output_bytes
            )
            item["decode_boundary_output_bytes"] = (
                unit.decode_boundary_output_bytes
            )
        canonical_units.append(item)

    canonical = {
        "generation_phase_schema": phase_schema,
        "provenance": provenance,
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal_units.fingerprint,
        "legal_unit_costs_sha256": costs.fingerprint,
        "runtime_manifest_sha256": runtime_manifest.fingerprint,
        "phase_workload": {
            "prefill_tokens": prefill_tokens,
            "decode_context_tokens": decode_context_tokens,
            "concurrency": concurrency,
        },
        "units": canonical_units,
        "sampling_profiles": {
            p.device: {
                "sampling_us": p.sampling_us,
                "logits_payload_bytes": p.logits_payload_bytes,
                "feedback_payload_bytes": p.feedback_payload_bytes,
            }
            for p in samplers
        },
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
    return GenerationPhaseEvidence(
        phase_schema,
        provenance,
        config.fingerprint,
        legal_units.fingerprint,
        costs.fingerprint,
        runtime_manifest.fingerprint,
        prefill_tokens,
        decode_context_tokens,
        concurrency,
        tuple(parsed_units),
        tuple(samplers),
        fingerprint,
    )


def generation_phase_summary(
    evidence: GenerationPhaseEvidence,
) -> dict[str, Any]:
    return {
        "result_schema": "tensormeld/generation-phase-evidence-validation-v1",
        "generation_phase_schema": evidence.schema,
        "generation_phase_evidence_sha256": evidence.fingerprint,
        "provenance": evidence.provenance,
        "config_sha256": evidence.config_sha256,
        "legal_model_units_sha256": evidence.legal_model_units_sha256,
        "legal_unit_costs_sha256": evidence.legal_unit_costs_sha256,
        "runtime_manifest_sha256": evidence.runtime_manifest_sha256,
        "phase_workload": {
            "prefill_tokens": evidence.prefill_tokens,
            "decode_context_tokens": evidence.decode_context_tokens,
            "concurrency": evidence.concurrency,
        },
        "unit_count": len(evidence.units),
        "sampling_devices": [profile.device for profile in evidence.sampling_profiles],
        "phase_boundary_payloads_complete": (
            evidence.schema == PHASE_EVIDENCE_SCHEMA_V2
        ),
        "qualified": False,
        "executable": False,
        "warnings": [
            "Prefill and decode-step costs apply only to the explicit phase workload/context position.",
            "No phase cost is inferred from tensor size, model family, backend label or generic legal-unit compute cost.",
            "Sampling profiles state sampling compute plus logits/feedback byte payloads; transfer time still requires directional path evidence.",
            "v1 has no phase-specific inter-unit boundary payloads; only v2 can support phase-transfer modeling.",
            "This evidence alone is not a TTFT, token-latency, throughput or native-execution claim.",
        ],
    }


def load_generation_phase_evidence(
    path: str | Path,
    *,
    config: Config,
    legal_units: LegalModelUnitsProfile,
    costs: LegalUnitCostsProfile,
    runtime_manifest: RuntimeModelManifest,
    movability: TensorMovabilityProfile,
) -> GenerationPhaseEvidence:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("generation phase evidence exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid generation phase evidence JSON: {exc}") from exc
    return parse_generation_phase_evidence(
        value,
        config=config,
        legal_units=legal_units,
        costs=costs,
        runtime_manifest=runtime_manifest,
        movability=movability,
    )
