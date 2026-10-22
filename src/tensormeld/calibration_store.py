"""Persistent placement-calibration records with explicit aging/invalidation.

The store is intentionally simple and local: one immutable JSON record per calibration
fingerprint. Loading a record never trusts its previous applicability. The calibration is
re-parsed against the current config/planning/candidate/package/runtime identities and its
age policy before it may influence measured preference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .config_v2 import Config
from .llamacpp_package import LlamaCppBuildPackage
from .placement_calibration import (
    PlacementCalibration,
    parse_placement_calibration,
    validate_placement_calibration,
)
from .planning_contract import PlanningInput
from .runtime_identity import RuntimeIdentity
from .schema import ValidationError, _no_duplicates, number, record, text

STORE_SCHEMA = "tensormeld/persisted-placement-calibration-v1"
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_AGE_SECONDS = 31 * 24 * 60 * 60


def _parse_time(value: Any, where: str) -> datetime:
    raw = text(value, where)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{where}: expected ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{where}: timezone is required")
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValidationError("timestamp timezone is required")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CalibrationRecord:
    recorded_at: datetime
    max_age_seconds: int
    calibration_record: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        value = self.calibration_record.get("fingerprint")
        if not isinstance(value, str) or len(value) != 64:
            raise ValidationError("persisted calibration fingerprint invalid")
        return value

    def as_record(self) -> dict[str, Any]:
        return {
            "store_schema": STORE_SCHEMA,
            "recorded_at": _format_time(self.recorded_at),
            "max_age_seconds": self.max_age_seconds,
            "calibration": self.calibration_record,
        }


def make_calibration_record(
    calibration: PlacementCalibration,
    *,
    recorded_at: datetime,
    max_age_seconds: int = 7 * 24 * 60 * 60,
) -> CalibrationRecord:
    validated = validate_placement_calibration(calibration)
    age = int(number(max_age_seconds, "max_age_seconds", 1, True))
    if age > MAX_AGE_SECONDS:
        raise ValidationError(
            f"max_age_seconds exceeds store limit of {MAX_AGE_SECONDS}"
        )
    return CalibrationRecord(
        recorded_at.astimezone(timezone.utc)
        if recorded_at.tzinfo is not None
        else _raise_naive(),
        age,
        validated,
    )


def _raise_naive():
    raise ValidationError("recorded_at timezone is required")


def parse_calibration_record(value: Any) -> CalibrationRecord:
    root = record(
        value,
        "persisted calibration",
        {"store_schema", "recorded_at", "max_age_seconds", "calibration"},
    )
    if root["store_schema"] != STORE_SCHEMA:
        raise ValidationError(f"store_schema: expected {STORE_SCHEMA}")
    age = int(number(root["max_age_seconds"], "max_age_seconds", 1, True))
    if age > MAX_AGE_SECONDS:
        raise ValidationError(
            f"max_age_seconds exceeds store limit of {MAX_AGE_SECONDS}"
        )
    if not isinstance(root["calibration"], dict):
        raise ValidationError("calibration: expected object")
    calibration = dict(root["calibration"])
    fp = calibration.get("fingerprint")
    if not isinstance(fp, str) or len(fp) != 64:
        raise ValidationError("calibration.fingerprint invalid")
    return CalibrationRecord(
        _parse_time(root["recorded_at"], "recorded_at"),
        age,
        calibration,
    )


def persist_calibration(
    directory: str | Path,
    record_value: CalibrationRecord,
) -> Path:
    record = parse_calibration_record(record_value.as_record())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{record.fingerprint}.json"
    payload = (
        json.dumps(
            record.as_record(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise ValidationError("persisted calibration record exceeds 2 MiB")

    if target.exists():
        current = target.read_bytes()
        if current == payload:
            return target
        raise ValidationError(
            "calibration fingerprint already exists with different persisted content"
        )

    temp = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return target


def load_calibration_records(directory: str | Path) -> list[CalibrationRecord]:
    directory = Path(directory)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValidationError("calibration store path is not a directory")
    records: list[CalibrationRecord] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        raw = path.read_bytes()
        if len(raw) > MAX_RECORD_BYTES:
            raise ValidationError(f"{path.name}: calibration record exceeds 2 MiB")
        try:
            parsed = json.loads(raw, object_pairs_hook=_no_duplicates)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValidationError(
                f"{path.name}: invalid persisted calibration JSON: {exc}"
            ) from exc
        item = parse_calibration_record(parsed)
        if path.name != f"{item.fingerprint}.json":
            raise ValidationError(
                f"{path.name}: filename does not match calibration fingerprint"
            )
        if item.fingerprint in seen:
            raise ValidationError("duplicate persisted calibration fingerprint")
        seen.add(item.fingerprint)
        records.append(item)
    return records


def applicable_persisted_calibrations(
    records: Iterable[CalibrationRecord],
    *,
    now: datetime,
    config: Config,
    planning: PlanningInput,
    candidates: Iterable[dict[str, Any]],
    package: LlamaCppBuildPackage,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
    profile_name: str | None = None,
    require_native: bool = True,
) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ValidationError("now timezone is required")
    now_utc = now.astimezone(timezone.utc)
    by_plan = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValidationError("candidate must be object")
        plan = candidate.get("plan_sha256")
        if not isinstance(plan, str) or len(plan) != 64:
            raise ValidationError("candidate plan_sha256 invalid")
        if plan in by_plan:
            raise ValidationError("duplicate candidate plan_sha256")
        by_plan[plan] = candidate

    applicable: list[PlacementCalibration] = []
    rejected: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()

    for persisted in records:
        if persisted.fingerprint in seen_fingerprints:
            raise ValidationError("duplicate calibration record in input")
        seen_fingerprints.add(persisted.fingerprint)
        age_seconds = int((now_utc - persisted.recorded_at).total_seconds())
        if age_seconds < 0:
            rejected.append({
                "fingerprint": persisted.fingerprint,
                "reason": "RECORDED_IN_FUTURE",
            })
            continue
        if age_seconds > persisted.max_age_seconds:
            rejected.append({
                "fingerprint": persisted.fingerprint,
                "reason": "EXPIRED",
                "age_seconds": age_seconds,
                "max_age_seconds": persisted.max_age_seconds,
            })
            continue

        raw = dict(persisted.calibration_record)
        supplied_fingerprint = raw.pop("fingerprint", None)
        supplied_objective = raw.pop("objective_us", None)
        plan = raw.get("candidate_plan_sha256")
        candidate = by_plan.get(plan)
        if candidate is None:
            rejected.append({
                "fingerprint": persisted.fingerprint,
                "reason": "CANDIDATE_NOT_RETAINED",
            })
            continue
        try:
            calibration = parse_placement_calibration(
                raw,
                config=config,
                planning=planning,
                candidate=candidate,
                package=package,
                runtime_identities=runtime_identities,
                profile_name=profile_name,
            )
        except ValidationError as exc:
            rejected.append({
                "fingerprint": persisted.fingerprint,
                "reason": "IDENTITY_OR_CONTEXT_MISMATCH",
                "detail": str(exc),
            })
            continue
        if calibration.fingerprint != supplied_fingerprint:
            rejected.append({
                "fingerprint": persisted.fingerprint,
                "reason": "CALIBRATION_FINGERPRINT_MISMATCH",
            })
            continue
        if calibration.objective_us != supplied_objective:
            rejected.append({
                "fingerprint": persisted.fingerprint,
                "reason": "CALIBRATION_OBJECTIVE_MISMATCH",
            })
            continue
        if require_native and calibration.source != "native-target":
            rejected.append({
                "fingerprint": persisted.fingerprint,
                "reason": "NON_NATIVE_MEASUREMENT",
            })
            continue
        applicable.append(calibration)

    return {
        "store_selection_schema": "tensormeld/calibration-store-selection-v1",
        "applicable": applicable,
        "rejected": rejected,
        "applicable_count": len(applicable),
        "rejected_count": len(rejected),
        "qualified": False,
        "executable": False,
    }
