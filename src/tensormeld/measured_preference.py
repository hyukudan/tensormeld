"""Measured preference overlay for planner-v2 results.

This module never modifies planner candidates. It takes an existing planner result and
optional placement calibrations, then exposes an explicit recommendation view:

- synthetic_best: planner-v2's own best candidate;
- measured_best: best applicable native-target calibration among retained candidates;
- recommended: measured_best when available, otherwise synthetic_best.

Calibration affects recommendation provenance only. It does not alter candidate hashes,
synthetic estimates, qualification, executable state, or planner search results.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from .placement_calibration import (
    PlacementCalibration,
    rank_applicable_calibrations,
    validate_placement_calibration,
)
from .schema import ValidationError

PREFERENCE_SCHEMA = "tensormeld/measured-planner-preference-v1"


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


def apply_measured_preference(
    planning_result: dict[str, Any],
    calibrations: list[PlacementCalibration] | tuple[PlacementCalibration, ...],
    *,
    require_native: bool = True,
) -> dict[str, Any]:
    if not isinstance(planning_result, dict):
        raise ValidationError("planning_result must be an object")
    required = {
        "result_schema",
        "status",
        "config_sha256",
        "planning_input_sha256",
        "profile",
        "objective",
        "provenance",
        "qualified",
        "executable",
        "best",
        "candidates",
        "search",
        "warnings",
    }
    if set(planning_result) != required:
        raise ValidationError("planning_result fields do not match planner-v2 schema")
    if planning_result["result_schema"] != "tensormeld/planning-result-v1":
        raise ValidationError("measured preference requires planner-v2 result")
    if planning_result["provenance"] != "synthetic":
        raise ValidationError("planner-v2 result provenance must remain synthetic")
    if planning_result["qualified"] is not False or planning_result["executable"] is not False:
        raise ValidationError("planner result cannot already be qualified/executable")

    candidates = planning_result["candidates"]
    if not isinstance(candidates, list):
        raise ValidationError("planning_result.candidates must be a list")
    candidate_by_plan: dict[str, dict[str, Any]] = {}
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValidationError(f"candidates[{i}] must be an object")
        plan_sha = candidate.get("plan_sha256")
        if not isinstance(plan_sha, str) or len(plan_sha) != 64:
            raise ValidationError(f"candidates[{i}].plan_sha256 invalid")
        if plan_sha in candidate_by_plan:
            raise ValidationError("duplicate retained candidate plan_sha256")
        if candidate.get("qualified") is not False or candidate.get("executable") is not False:
            raise ValidationError("retained candidate cannot self-promote")
        candidate_by_plan[plan_sha] = candidate

    synthetic_best = planning_result["best"]
    if synthetic_best is not None:
        if not isinstance(synthetic_best, dict):
            raise ValidationError("planning_result.best must be object or null")
        best_sha = synthetic_best.get("plan_sha256")
        if best_sha not in candidate_by_plan:
            raise ValidationError("planning_result.best is not a retained candidate")
        if synthetic_best != candidate_by_plan[best_sha]:
            raise ValidationError("planning_result.best differs from retained candidate")

    for calibration in calibrations:
        validate_placement_calibration(calibration)
        if calibration.config_sha256 != planning_result["config_sha256"]:
            raise ValidationError("calibration config identity differs from planner result")
        if calibration.planning_input_sha256 != planning_result["planning_input_sha256"]:
            raise ValidationError("calibration planning identity differs from planner result")
        if calibration.profile != planning_result["profile"]:
            raise ValidationError("calibration profile differs from planner result")
        if calibration.candidate_plan_sha256 not in candidate_by_plan:
            raise ValidationError("calibration references candidate not retained by planner")

    ranked = rank_applicable_calibrations(
        calibrations,
        require_native=require_native,
    )
    measured_best_calibration = ranked[0] if ranked else None
    measured_best = (
        candidate_by_plan[measured_best_calibration.candidate_plan_sha256]
        if measured_best_calibration is not None
        else None
    )
    recommended = measured_best if measured_best is not None else synthetic_best
    recommendation_source = (
        "native-calibration"
        if measured_best is not None and require_native
        else "calibration"
        if measured_best is not None
        else "synthetic"
    )

    synthetic_sha = synthetic_best.get("plan_sha256") if synthetic_best else None
    measured_sha = measured_best.get("plan_sha256") if measured_best else None
    recommended_sha = recommended.get("plan_sha256") if recommended else None

    core = {
        "preference_schema": PREFERENCE_SCHEMA,
        "planner_result_schema": planning_result["result_schema"],
        "config_sha256": planning_result["config_sha256"],
        "planning_input_sha256": planning_result["planning_input_sha256"],
        "profile": planning_result["profile"],
        "objective": planning_result["objective"],
        "synthetic_best_plan_sha256": synthetic_sha,
        "measured_best_plan_sha256": measured_sha,
        "recommended_plan_sha256": recommended_sha,
        "recommendation_source": recommendation_source,
        "measured_objective_us": (
            measured_best_calibration.objective_us
            if measured_best_calibration is not None
            else None
        ),
        "measured_calibration_sha256": (
            measured_best_calibration.fingerprint
            if measured_best_calibration is not None
            else None
        ),
        "native_measurement_required": require_native,
        "qualified": False,
        "executable": False,
        "warnings": [
            "Measured preference does not mutate planner candidates or their plan_sha256.",
            "Calibration changes recommendation ordering only; correctness/admission remain separate gates.",
            "Synthetic planner estimates remain preserved for audit/explanation.",
        ],
    }
    core["preference_sha256"] = _canonical_sha256(core)

    return {
        "preference": core,
        "synthetic_best": deepcopy(synthetic_best),
        "measured_best": deepcopy(measured_best),
        "recommended": deepcopy(recommended),
        "candidates": deepcopy(candidates),
    }
