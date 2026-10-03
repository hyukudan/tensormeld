"""Qualification evidence identity and applicability checks.

Evidence is imported/recorded data. Parsing it never performs qualification and never
changes a planner or representability result into an executable session.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .adapter_contract import AdapterCapabilities
from .model_manifest import ModelManifest, _sha256
from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, items, number, record, text, unique

MAX_EVIDENCE_DEVICES = 128
MAX_TESTS = 256
EVIDENCE_LEVELS = {"E0", "E1", "E2", "E3", "E4", "E5"}


def _timestamp(value: Any, where: str) -> str:
    s = text(value, where)
    try:
        parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{where}: expected ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{where}: timezone is required")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class QualificationEvidence:
    evidence_id: str
    level: str
    adapter_id: str
    adapter_capabilities_sha256: str
    engine_revision: str
    worker_artifact_sha256: str
    model_manifest_sha256: str
    config_sha256: str
    device_ids: tuple[str, ...]
    workload: dict[str, int]
    result: str
    observed_at: str
    tests: tuple[str, ...]
    candidate_plan_sha256: str | None
    placement_sha256: str | None
    trial_spec_sha256: str | None
    correctness_contract_sha256: str | None
    runtime_identity_sha256: tuple[str, ...] | None
    evidence_sha256: str

    @classmethod
    def parse(cls, data: Any) -> "QualificationEvidence":
        if not isinstance(data, dict):
            raise ValidationError("qualification evidence: expected object")
        schema = data.get("qualification_schema")
        base_fields = {
            "qualification_schema", "evidence_id", "level", "adapter_id",
            "adapter_capabilities_sha256", "engine_revision", "worker_artifact_sha256",
            "model_manifest_sha256", "config_sha256", "device_ids", "workload",
            "result", "observed_at", "tests",
        }
        v2_fields = base_fields | {
            "candidate_plan_sha256", "placement_sha256", "trial_spec_sha256",
            "correctness_contract_sha256", "runtime_identity_sha256",
        }
        if schema == "tensormeld/qualification-evidence-v1":
            r = record(data, "qualification evidence", base_fields)
            candidate_plan_sha = placement_sha = trial_sha = correctness_sha = None
            runtime_ids = None
        elif schema == "tensormeld/qualification-evidence-v2":
            r = record(data, "qualification evidence", v2_fields)
            candidate_plan_sha = _sha256(
                r["candidate_plan_sha256"], "candidate_plan_sha256"
            )
            placement_sha = _sha256(r["placement_sha256"], "placement_sha256")
            trial_sha = _sha256(r["trial_spec_sha256"], "trial_spec_sha256")
            correctness_sha = _sha256(
                r["correctness_contract_sha256"], "correctness_contract_sha256"
            )
            runtime_ids = tuple(
                _sha256(x, "runtime_identity_sha256[]")
                for x in items(
                    r["runtime_identity_sha256"],
                    "runtime_identity_sha256",
                    MAX_EVIDENCE_DEVICES,
                    1,
                )
            )
            unique(list(runtime_ids), "runtime_identity_sha256")
        else:
            raise ValidationError(
                "qualification_schema: expected tensormeld/qualification-evidence-v1 or v2"
            )
        level = text(r["level"], "level")
        if level not in EVIDENCE_LEVELS:
            raise ValidationError("level: expected E0..E5")
        result = text(r["result"], "result")
        if result not in {"passed", "failed"}:
            raise ValidationError("result: expected passed or failed")
        device_ids = tuple(
            text(x, "device_ids[]")
            for x in items(r["device_ids"], "device_ids", MAX_EVIDENCE_DEVICES, 1)
        )
        unique(list(device_ids), "device_ids")
        tests = tuple(
            text(x, "tests[]") for x in items(r["tests"], "tests", MAX_TESTS, 1)
        )
        unique(list(tests), "tests")
        w = record(
            r["workload"], "workload",
            {"context_tokens", "max_output_tokens", "concurrency"},
        )
        workload = {
            "context_tokens": int(
                number(w["context_tokens"], "workload.context_tokens", 1, True)
            ),
            "max_output_tokens": int(
                number(w["max_output_tokens"], "workload.max_output_tokens", 1, True)
            ),
            "concurrency": int(
                number(w["concurrency"], "workload.concurrency", 1, True)
            ),
        }
        if workload["max_output_tokens"] > workload["context_tokens"]:
            raise ValidationError("workload.max_output_tokens exceeds context_tokens")
        canonical = {
            "qualification_schema": schema,
            "evidence_id": text(r["evidence_id"], "evidence_id"),
            "level": level,
            "adapter_id": text(r["adapter_id"], "adapter_id"),
            "adapter_capabilities_sha256": _sha256(
                r["adapter_capabilities_sha256"], "adapter_capabilities_sha256"
            ),
            "engine_revision": text(r["engine_revision"], "engine_revision"),
            "worker_artifact_sha256": _sha256(
                r["worker_artifact_sha256"], "worker_artifact_sha256"
            ),
            "model_manifest_sha256": _sha256(
                r["model_manifest_sha256"], "model_manifest_sha256"
            ),
            "config_sha256": _sha256(r["config_sha256"], "config_sha256"),
            "device_ids": list(device_ids),
            "workload": workload,
            "result": result,
            "observed_at": _timestamp(r["observed_at"], "observed_at"),
            "tests": list(tests),
        }
        if schema.endswith("-v2"):
            canonical.update({
                "candidate_plan_sha256": candidate_plan_sha,
                "placement_sha256": placement_sha,
                "trial_spec_sha256": trial_sha,
                "correctness_contract_sha256": correctness_sha,
                "runtime_identity_sha256": list(runtime_ids or ()),
            })
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            canonical["evidence_id"], level, canonical["adapter_id"],
            canonical["adapter_capabilities_sha256"], canonical["engine_revision"],
            canonical["worker_artifact_sha256"], canonical["model_manifest_sha256"],
            canonical["config_sha256"], device_ids, workload, result,
            canonical["observed_at"], tests, candidate_plan_sha, placement_sha, trial_sha,
            correctness_sha, runtime_ids, digest,
        )


def evidence_applies(
    evidence: QualificationEvidence, *, adapter: AdapterCapabilities,
    model: ModelManifest, config_sha256: str,
    device_ids: list[str] | tuple[str, ...], context_tokens: int,
    max_output_tokens: int, concurrency: int, minimum_level: str = "E3",
    candidate_plan_sha256: str | None = None,
    runtime_identity_sha256: list[str] | tuple[str, ...] | None = None,
    require_v2: bool = False,
) -> dict[str, Any]:
    if minimum_level not in EVIDENCE_LEVELS:
        raise ValidationError("minimum_level: expected E0..E5")
    reasons = []
    order = {f"E{i}": i for i in range(6)}
    checks = [
        (evidence.result == "passed", "EVIDENCE_FAILED"),
        (evidence.adapter_id == adapter.adapter_id, "ADAPTER_ID_MISMATCH"),
        (
            evidence.adapter_capabilities_sha256 == adapter.fingerprint,
            "ADAPTER_CAPABILITIES_MISMATCH",
        ),
        (evidence.engine_revision == adapter.engine_revision, "ENGINE_REVISION_MISMATCH"),
        (
            evidence.model_manifest_sha256 == model.manifest_sha256,
            "MODEL_MANIFEST_MISMATCH",
        ),
        (evidence.config_sha256 == config_sha256, "CONFIG_MISMATCH"),
        (tuple(evidence.device_ids) == tuple(device_ids), "DEVICE_SET_MISMATCH"),
        (
            evidence.workload == {
                "context_tokens": context_tokens,
                "max_output_tokens": max_output_tokens,
                "concurrency": concurrency,
            },
            "WORKLOAD_MISMATCH",
        ),
        (
            order[evidence.level] >= order[minimum_level],
            "EVIDENCE_LEVEL_TOO_LOW",
        ),
    ]
    for ok, code in checks:
        if not ok:
            reasons.append(code)
    if require_v2:
        if evidence.candidate_plan_sha256 is None:
            reasons.append("EVIDENCE_V2_REQUIRED")
        if candidate_plan_sha256 is None:
            reasons.append("EXPECTED_PLAN_ID_REQUIRED")
        elif evidence.candidate_plan_sha256 != candidate_plan_sha256:
            reasons.append("PLAN_ID_MISMATCH")
        if runtime_identity_sha256 is None:
            reasons.append("EXPECTED_RUNTIME_ID_REQUIRED")
        elif evidence.runtime_identity_sha256 != tuple(runtime_identity_sha256):
            reasons.append("RUNTIME_IDENTITY_MISMATCH")
    return {
        "result_schema": "tensormeld/qualification-applicability-v1",
        "evidence_id": evidence.evidence_id,
        "evidence_sha256": evidence.evidence_sha256,
        "applies": not reasons,
        "reasons": reasons,
        "qualified": not reasons,
        "executable": False,
        "minimum_level": minimum_level,
    }


def load_qualification_evidence(path: str | Path) -> QualificationEvidence:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("qualification evidence exceeds 2 MiB")
    try:
        return QualificationEvidence.parse(
            json.loads(raw, object_pairs_hook=_no_duplicates)
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid qualification evidence JSON: {exc}") from exc
