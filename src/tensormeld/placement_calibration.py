"""Measured placement calibration over existing TensorMeld planner candidates.

Calibration never creates plans and never upgrades correctness/executability. It measures
already-hashed planner candidates and records separate prefill/decode costs under one exact
model/build/runtime/workload identity.

The normalized objective uses integer arithmetic:
  ceil(prefill_us * target_prefill_tokens / measured_prefill_tokens)
+ ceil(decode_us * target_decode_tokens / measured_decode_tokens)

This is a workload-specific comparison, not a universal throughput claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .config_v2 import Config
from .llamacpp_package import (
    LlamaCppBuildPackage,
    validate_llamacpp_package_identity,
)
from .planning_contract import PlanningInput
from .plan_identity import validate_candidate_hash
from .runtime_identity import RuntimeIdentity
from .schema import ValidationError, items, number, record, text, unique

CALIBRATION_SCHEMA = "tensormeld/placement-calibration-v1"
SEARCH_SCHEMA = "tensormeld/placement-calibration-search-v1"
MAX_RUNTIME_IDENTITIES = 128
MAX_POOL_PEAKS = 192


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _ceil_scaled(elapsed_us: int, target_tokens: int, measured_tokens: int) -> int:
    return (elapsed_us * target_tokens + measured_tokens - 1) // measured_tokens


@dataclass(frozen=True)
class PlacementCalibration:
    source: str
    config_sha256: str
    planning_input_sha256: str
    profile: str
    model_manifest_sha256: str
    package_sha256: str
    candidate_plan_sha256: str
    runtime_identity_sha256: tuple[str, ...]
    target_prefill_tokens: int
    target_decode_tokens: int
    measured_prefill_tokens: int
    measured_decode_tokens: int
    prefill_elapsed_us: int
    decode_elapsed_us: int
    objective_us: int
    physical_pool_peak_bytes: tuple[tuple[str, int], ...]
    fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {
            "calibration_schema": CALIBRATION_SCHEMA,
            "measurement_source": self.source,
            "config_sha256": self.config_sha256,
            "planning_input_sha256": self.planning_input_sha256,
            "profile": self.profile,
            "model_manifest_sha256": self.model_manifest_sha256,
            "package_sha256": self.package_sha256,
            "candidate_plan_sha256": self.candidate_plan_sha256,
            "runtime_identity_sha256": list(self.runtime_identity_sha256),
            "target_workload": {
                "prefill_tokens": self.target_prefill_tokens,
                "decode_tokens": self.target_decode_tokens,
            },
            "measurement": {
                "prefill_tokens": self.measured_prefill_tokens,
                "decode_tokens": self.measured_decode_tokens,
                "prefill_elapsed_us": self.prefill_elapsed_us,
                "decode_elapsed_us": self.decode_elapsed_us,
            },
            "objective_us": self.objective_us,
            "physical_pool_peak_bytes": [
                {"pool_ref": pool, "peak_bytes": peak}
                for pool, peak in self.physical_pool_peak_bytes
            ],
            "qualified": False,
            "executable": False,
            "fingerprint": self.fingerprint,
        }


def parse_placement_calibration(
    data: Any,
    *,
    config: Config,
    planning: PlanningInput,
    candidate: dict[str, Any],
    package: LlamaCppBuildPackage,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
    profile_name: str | None = None,
) -> PlacementCalibration:
    r = record(data, "placement calibration", {
        "calibration_schema",
        "measurement_source",
        "config_sha256",
        "planning_input_sha256",
        "profile",
        "model_manifest_sha256",
        "package_sha256",
        "candidate_plan_sha256",
        "runtime_identity_sha256",
        "target_workload",
        "measurement",
        "physical_pool_peak_bytes",
        "qualified",
        "executable",
    })
    if r["calibration_schema"] != CALIBRATION_SCHEMA:
        raise ValidationError(f"calibration_schema: expected {CALIBRATION_SCHEMA}")
    source = text(r["measurement_source"], "measurement_source")
    if source not in {"native-target", "fixture"}:
        raise ValidationError("measurement_source: expected native-target or fixture")
    if r["qualified"] is not False or r["executable"] is not False:
        raise ValidationError("placement calibration cannot self-promote")

    profile = config.profile_map.get(profile_name or config.installation.default_profile)
    if profile is None:
        raise ValidationError("placement calibration profile is unknown")
    plan_sha = validate_candidate_hash(config, planning, profile.name, candidate)
    validate_llamacpp_package_identity(package)

    checks = {
        "config_sha256": config.fingerprint,
        "planning_input_sha256": planning.fingerprint,
        "profile": profile.name,
        "model_manifest_sha256": planning.manifest_ref,
        "package_sha256": package.fingerprint,
        "candidate_plan_sha256": plan_sha,
    }
    for field, expected in checks.items():
        if r[field] != expected:
            raise ValidationError(f"placement calibration {field} mismatch")

    identities = tuple(runtime_identities)
    if not identities:
        raise ValidationError("placement calibration requires runtime identities")
    by_device = {identity.tensormeld_device_id: identity for identity in identities}
    if len(by_device) != len(identities):
        raise ValidationError("duplicate runtime identity device")
    candidate_devices = tuple(candidate.get("compute_devices", ()))
    if set(by_device) != set(candidate_devices):
        raise ValidationError(
            "placement calibration runtime identities must cover exact candidate devices"
        )
    for device in candidate_devices:
        if by_device[device].worker_artifact_sha256 != package.llama_server_sha256:
            raise ValidationError(
                "placement calibration runtime identity must use package llama-server artifact"
            )
    expected_runtime = tuple(
        by_device[device].identity_sha256 for device in candidate_devices
    )
    supplied_runtime = tuple(
        text(value, "runtime_identity_sha256[]")
        for value in items(
            r["runtime_identity_sha256"],
            "runtime_identity_sha256",
            MAX_RUNTIME_IDENTITIES,
            1,
        )
    )
    if supplied_runtime != expected_runtime:
        raise ValidationError("placement calibration runtime identity mismatch")

    target = record(
        r["target_workload"],
        "target_workload",
        {"prefill_tokens", "decode_tokens"},
    )
    target_prefill = int(number(
        target["prefill_tokens"], "target_workload.prefill_tokens", 1, True
    ))
    target_decode = int(number(
        target["decode_tokens"], "target_workload.decode_tokens", 1, True
    ))
    if target_prefill != planning.context_tokens:
        raise ValidationError("target prefill tokens must equal planning context")
    if target_decode != planning.max_output_tokens:
        raise ValidationError("target decode tokens must equal planning max output")

    measurement = record(
        r["measurement"],
        "measurement",
        {
            "prefill_tokens",
            "decode_tokens",
            "prefill_elapsed_us",
            "decode_elapsed_us",
        },
    )
    measured_prefill = int(number(
        measurement["prefill_tokens"], "measurement.prefill_tokens", 1, True
    ))
    measured_decode = int(number(
        measurement["decode_tokens"], "measurement.decode_tokens", 1, True
    ))
    prefill_us = int(number(
        measurement["prefill_elapsed_us"], "measurement.prefill_elapsed_us", 1, True
    ))
    decode_us = int(number(
        measurement["decode_elapsed_us"], "measurement.decode_elapsed_us", 1, True
    ))
    objective_us = (
        _ceil_scaled(prefill_us, target_prefill, measured_prefill)
        + _ceil_scaled(decode_us, target_decode, measured_decode)
    )

    pool_peaks: list[tuple[str, int]] = []
    for i, raw in enumerate(items(
        r["physical_pool_peak_bytes"],
        "physical_pool_peak_bytes",
        MAX_POOL_PEAKS,
        0,
    )):
        item = record(raw, f"physical_pool_peak_bytes[{i}]", {"pool_ref", "peak_bytes"})
        pool_peaks.append((
            text(item["pool_ref"], f"physical_pool_peak_bytes[{i}].pool_ref"),
            int(number(
                item["peak_bytes"],
                f"physical_pool_peak_bytes[{i}].peak_bytes",
                0,
                True,
            )),
        ))
    unique([pool for pool, _ in pool_peaks], "physical_pool_peak_bytes.pool_ref")
    pools_by_id = {pool.id: pool for pool in config.pools}
    candidate_nodes = set(candidate.get("compute_nodes", ()))
    for pool_id, peak in pool_peaks:
        pool = pools_by_id.get(pool_id)
        if pool is None:
            raise ValidationError("placement calibration references unknown physical pool")
        if pool.node not in candidate_nodes:
            raise ValidationError(
                "placement calibration physical pool belongs to unused candidate node"
            )
        if (
            pool.reported_capacity_bytes is not None
            and peak > pool.reported_capacity_bytes
        ):
            raise ValidationError(
                "placement calibration physical pool peak exceeds reported capacity"
            )
    pool_peaks.sort()

    canonical = {
        "calibration_schema": CALIBRATION_SCHEMA,
        "measurement_source": source,
        "config_sha256": config.fingerprint,
        "planning_input_sha256": planning.fingerprint,
        "profile": profile.name,
        "model_manifest_sha256": planning.manifest_ref,
        "package_sha256": package.fingerprint,
        "candidate_plan_sha256": plan_sha,
        "runtime_identity_sha256": list(expected_runtime),
        "target_workload": {
            "prefill_tokens": target_prefill,
            "decode_tokens": target_decode,
        },
        "measurement": {
            "prefill_tokens": measured_prefill,
            "decode_tokens": measured_decode,
            "prefill_elapsed_us": prefill_us,
            "decode_elapsed_us": decode_us,
        },
        "objective_us": objective_us,
        "physical_pool_peak_bytes": [
            {"pool_ref": pool, "peak_bytes": peak}
            for pool, peak in pool_peaks
        ],
        "qualified": False,
        "executable": False,
    }
    return PlacementCalibration(
        source,
        config.fingerprint,
        planning.fingerprint,
        profile.name,
        planning.manifest_ref,
        package.fingerprint,
        plan_sha,
        expected_runtime,
        target_prefill,
        target_decode,
        measured_prefill,
        measured_decode,
        prefill_us,
        decode_us,
        objective_us,
        tuple(pool_peaks),
        _canonical_sha256(canonical),
    )


def validate_placement_calibration(
    calibration: PlacementCalibration,
) -> dict[str, Any]:
    if not isinstance(calibration, PlacementCalibration):
        raise ValidationError("expected PlacementCalibration")
    record_value = calibration.as_record()
    supplied = record_value.pop("fingerprint")
    if _canonical_sha256(record_value) != supplied:
        raise ValidationError("placement calibration fingerprint mismatch")
    if calibration.source not in {"native-target", "fixture"}:
        raise ValidationError("placement calibration source invalid")
    return {**record_value, "fingerprint": supplied}


def rank_applicable_calibrations(
    calibrations: list[PlacementCalibration] | tuple[PlacementCalibration, ...],
    *,
    require_native: bool = True,
) -> list[PlacementCalibration]:
    if not calibrations:
        return []
    for calibration in calibrations:
        validate_placement_calibration(calibration)
    first = calibrations[0]
    common = (
        first.config_sha256,
        first.planning_input_sha256,
        first.profile,
        first.model_manifest_sha256,
        first.package_sha256,
        first.runtime_identity_sha256,
        first.target_prefill_tokens,
        first.target_decode_tokens,
    )
    applicable: list[PlacementCalibration] = []
    seen_plans: set[str] = set()
    for calibration in calibrations:
        identity = (
            calibration.config_sha256,
            calibration.planning_input_sha256,
            calibration.profile,
            calibration.model_manifest_sha256,
            calibration.package_sha256,
            calibration.runtime_identity_sha256,
            calibration.target_prefill_tokens,
            calibration.target_decode_tokens,
        )
        if identity != common:
            raise ValidationError("calibration set mixes incompatible identities/workloads")
        if calibration.candidate_plan_sha256 in seen_plans:
            raise ValidationError("duplicate calibration for candidate plan")
        seen_plans.add(calibration.candidate_plan_sha256)
        if require_native and calibration.source != "native-target":
            continue
        applicable.append(calibration)
    return sorted(
        applicable,
        key=lambda item: (
            item.objective_us,
            item.prefill_elapsed_us,
            item.decode_elapsed_us,
            item.candidate_plan_sha256,
        ),
    )


def _coarse_indices(total: int, max_samples: int) -> tuple[int, ...]:
    if type(total) is not int or total < 1:
        raise ValidationError("candidate count must be positive")
    if type(max_samples) is not int or not 2 <= max_samples <= 20:
        raise ValidationError("max_samples must be within 2..20")
    if total <= max_samples:
        return tuple(range(total))
    # Integer evenly-spaced samples including both endpoints.
    indices = {
        (i * (total - 1)) // (max_samples - 1)
        for i in range(max_samples)
    }
    return tuple(sorted(indices))


def next_calibration_round(
    candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    measured_plan_sha256: set[str] | frozenset[str],
    ranked_measurements: list[PlacementCalibration] | tuple[PlacementCalibration, ...],
    max_coarse_samples: int = 5,
    refine_radius: int = 2,
) -> dict[str, Any]:
    if not candidates:
        raise ValidationError("calibration search requires candidates")
    if type(refine_radius) is not int or not 1 <= refine_radius <= 8:
        raise ValidationError("refine_radius must be within 1..8")
    plan_ids = []
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValidationError(f"candidates[{i}] must be an object")
        plan_id = candidate.get("plan_sha256")
        if not isinstance(plan_id, str) or len(plan_id) != 64:
            raise ValidationError(f"candidates[{i}] has invalid plan_sha256")
        plan_ids.append(plan_id)
    unique(plan_ids, "candidates.plan_sha256")

    measured = set(measured_plan_sha256)
    unknown = measured - set(plan_ids)
    if unknown:
        raise ValidationError("measured calibration references unknown candidate")

    coarse = _coarse_indices(len(candidates), max_coarse_samples)
    coarse_pending = [
        plan_ids[index] for index in coarse if plan_ids[index] not in measured
    ]
    if coarse_pending:
        phase = "coarse"
        selected = coarse_pending
        winner = None
    else:
        if not ranked_measurements:
            return {
                "search_schema": SEARCH_SCHEMA,
                "phase": "blocked",
                "selected_plan_sha256": [],
                "winner_plan_sha256": None,
                "complete": False,
                "warnings": [
                    "Coarse candidates are measured but no applicable calibration ranking is available."
                ],
            }
        for item in ranked_measurements:
            validate_placement_calibration(item)
        winner = ranked_measurements[0].candidate_plan_sha256
        if winner not in plan_ids:
            raise ValidationError("calibration winner is not in candidate list")
        index = plan_ids.index(winner)
        lo = max(0, index - refine_radius)
        hi = min(len(plan_ids), index + refine_radius + 1)
        selected = [
            plan_ids[i]
            for i in range(lo, hi)
            if plan_ids[i] not in measured
        ]
        phase = "refine" if selected else "complete"

    core = {
        "search_schema": SEARCH_SCHEMA,
        "phase": phase,
        "candidate_count": len(plan_ids),
        "coarse_indices": list(coarse),
        "selected_plan_sha256": selected,
        "winner_plan_sha256": winner,
        "complete": phase == "complete",
        "qualified": False,
        "executable": False,
        "warnings": [
            "Candidate ordering comes from TensorMeld planner output; calibration does not invent plans.",
            "Search is bounded coarse-to-local refinement and does not prove a global optimum.",
        ],
    }
    core["search_sha256"] = _canonical_sha256(core)
    return core
