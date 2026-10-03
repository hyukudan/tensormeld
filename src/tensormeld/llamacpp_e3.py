"""Native llama.cpp E3 correctness evaluation.

E3 is emitted only from a genuine native-subprocess trial whose exact output hash matches
an approved deterministic reference contract and whose adapter/model/placement/workload
and stable runtime identities all match exactly.

Portable tests may construct native-shaped records, but such tests remain contract tests
and are not target-host evidence by themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from .adapter_contract import AdapterCapabilities
from .llamacpp_native_trial import LlamaCppNativeTrialSpec, TRIAL_SCHEMA
from .llamacpp_placement import LlamaCppQualificationPlacement
from .model_manifest import ModelManifest, _sha256
from .qualification import QualificationEvidence
from .runtime_identity import RuntimeIdentity
from .schema import ValidationError, items, number, record, text, unique

REFERENCE_SCHEMA = "tensormeld/llamacpp-e3-reference-v1"
EVALUATION_SCHEMA = "tensormeld/llamacpp-e3-evaluation-v1"


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


@dataclass(frozen=True)
class LlamaCppE3Reference:
    reference_id: str
    trial_spec_sha256: str
    llama_cli_sha256: str
    model_manifest_sha256: str
    placement_sha256: str
    expected_stdout_sha256: str
    workload: dict[str, int]
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> "LlamaCppE3Reference":
        r = record(data, "llama.cpp E3 reference", {
            "reference_schema",
            "reference_id",
            "trial_spec_sha256",
            "llama_cli_sha256",
            "model_manifest_sha256",
            "placement_sha256",
            "expected_stdout_sha256",
            "workload",
        })
        if r["reference_schema"] != REFERENCE_SCHEMA:
            raise ValidationError(f"reference_schema: expected {REFERENCE_SCHEMA}")
        workload_raw = record(
            r["workload"],
            "workload",
            {"context_tokens", "max_output_tokens", "concurrency"},
        )
        workload = {
            "context_tokens": int(number(
                workload_raw["context_tokens"], "workload.context_tokens", 1, True
            )),
            "max_output_tokens": int(number(
                workload_raw["max_output_tokens"], "workload.max_output_tokens", 1, True
            )),
            "concurrency": int(number(
                workload_raw["concurrency"], "workload.concurrency", 1, True
            )),
        }
        if workload["max_output_tokens"] > workload["context_tokens"]:
            raise ValidationError("reference workload output exceeds context")
        canonical = {
            "reference_schema": REFERENCE_SCHEMA,
            "reference_id": text(r["reference_id"], "reference_id"),
            "trial_spec_sha256": _sha256(r["trial_spec_sha256"], "trial_spec_sha256"),
            "llama_cli_sha256": _sha256(r["llama_cli_sha256"], "llama_cli_sha256"),
            "model_manifest_sha256": _sha256(
                r["model_manifest_sha256"], "model_manifest_sha256"
            ),
            "placement_sha256": _sha256(r["placement_sha256"], "placement_sha256"),
            "expected_stdout_sha256": _sha256(
                r["expected_stdout_sha256"], "expected_stdout_sha256"
            ),
            "workload": workload,
        }
        return cls(
            canonical["reference_id"],
            canonical["trial_spec_sha256"],
            canonical["llama_cli_sha256"],
            canonical["model_manifest_sha256"],
            canonical["placement_sha256"],
            canonical["expected_stdout_sha256"],
            workload,
            _canonical_sha256(canonical),
        )


def _validate_trial_result(
    trial_result: dict[str, Any],
    *,
    spec: LlamaCppNativeTrialSpec,
) -> None:
    if not isinstance(trial_result, dict):
        raise ValidationError("trial result must be an object")
    required = {
        "trial_schema": TRIAL_SCHEMA,
        "source_revision": spec.source_revision,
        "spec_sha256": spec.spec_sha256,
        "llama_cli_sha256": spec.llama_cli_sha256,
        "model_manifest_sha256": spec.model_manifest_sha256,
        "gguf_sha256": spec.gguf_sha256,
        "config_sha256": spec.config_sha256,
        "planning_input_sha256": spec.planning_input_sha256,
        "candidate_plan_sha256": spec.candidate_plan_sha256,
        "placement_sha256": spec.placement_sha256,
        "execution_source": "native-subprocess",
        "exit_code": 0,
        "process_succeeded": True,
        "evidence_level": "E2.5",
        "qualified": False,
        "real_model_inference": False,
        "executable": False,
    }
    for key, expected in required.items():
        if trial_result.get(key) != expected:
            raise ValidationError(f"native E3 trial result mismatch at {key}")
    _sha256(trial_result.get("stdout_sha256"), "trial.stdout_sha256")
    _sha256(trial_result.get("stderr_sha256"), "trial.stderr_sha256")
    if type(trial_result.get("stdout_bytes")) is not int or trial_result["stdout_bytes"] <= 0:
        raise ValidationError("native E3 trial requires non-empty stdout")
    if trial_result.get("stdout_nonempty") is not True:
        raise ValidationError("native E3 trial stdout_nonempty must be true")


def evaluate_llamacpp_native_e3(
    *,
    adapter: AdapterCapabilities,
    model: ModelManifest,
    placement: LlamaCppQualificationPlacement,
    spec: LlamaCppNativeTrialSpec,
    trial_result: dict[str, Any],
    reference: LlamaCppE3Reference,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
    observed_at: str | None = None,
) -> tuple[dict[str, Any], QualificationEvidence]:
    if adapter.engine != "llama.cpp":
        raise ValidationError("native E3 evaluator requires llama.cpp adapter")
    if adapter.engine_revision != spec.source_revision:
        raise ValidationError("native E3 adapter revision mismatch")
    if model.manifest_sha256 != spec.model_manifest_sha256:
        raise ValidationError("native E3 model identity mismatch")
    if placement.fingerprint != spec.placement_sha256:
        raise ValidationError("native E3 placement identity mismatch")
    if placement.config_sha256 != spec.config_sha256:
        raise ValidationError("native E3 config identity mismatch")
    _validate_trial_result(trial_result, spec=spec)

    checks = {
        "trial_spec_sha256": (reference.trial_spec_sha256, spec.spec_sha256),
        "llama_cli_sha256": (reference.llama_cli_sha256, spec.llama_cli_sha256),
        "model_manifest_sha256": (
            reference.model_manifest_sha256, model.manifest_sha256
        ),
        "placement_sha256": (reference.placement_sha256, placement.fingerprint),
        "stdout_sha256": (
            reference.expected_stdout_sha256, trial_result["stdout_sha256"]
        ),
    }
    for label, (expected, actual) in checks.items():
        if expected != actual:
            raise ValidationError(f"native E3 reference mismatch at {label}")

    expected_workload = {
        "context_tokens": spec.context_tokens,
        "max_output_tokens": spec.predict_tokens,
        "concurrency": 1,
    }
    if reference.workload != expected_workload:
        raise ValidationError("native E3 reference workload mismatch")

    device_ids = tuple(dict.fromkeys(d for _, d, _ in placement.block_owners))
    identities = tuple(runtime_identities)
    if not identities:
        raise ValidationError("native E3 requires runtime identities")
    unique([identity.tensormeld_device_id for identity in identities], "runtime devices")
    by_device = {identity.tensormeld_device_id: identity for identity in identities}
    if set(by_device) != set(device_ids):
        raise ValidationError("native E3 runtime identities must cover exact placement devices")
    ordered_identity_sha: list[str] = []
    for device_id in device_ids:
        identity = by_device[device_id]
        if identity.worker_artifact_sha256 != spec.llama_cli_sha256:
            raise ValidationError("native E3 runtime worker artifact mismatch")
        ordered_identity_sha.append(identity.identity_sha256)

    if observed_at is None:
        observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        # QualificationEvidence.parse performs strict timestamp validation.
        observed_at = text(observed_at, "observed_at")

    evidence_id = (
        "llamacpp-e3:"
        + _canonical_sha256({
            "reference": reference.fingerprint,
            "trial": spec.spec_sha256,
            "runtime": ordered_identity_sha,
        })[:24]
    )
    raw_evidence = {
        "qualification_schema": "tensormeld/qualification-evidence-v2",
        "evidence_id": evidence_id,
        "level": "E3",
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": spec.llama_cli_sha256,
        "model_manifest_sha256": model.manifest_sha256,
        "config_sha256": placement.config_sha256,
        "device_ids": list(device_ids),
        "workload": expected_workload,
        "result": "passed",
        "observed_at": observed_at,
        "tests": [
            "native-subprocess",
            "deterministic-exact-stdout-sha256",
            "exact-placement",
            "exact-runtime-identity",
        ],
        "placement_sha256": placement.fingerprint,
        "trial_spec_sha256": spec.spec_sha256,
        "correctness_contract_sha256": reference.fingerprint,
        "runtime_identity_sha256": ordered_identity_sha,
    }
    evidence = QualificationEvidence.parse(raw_evidence)
    evaluation = {
        "evaluation_schema": EVALUATION_SCHEMA,
        "reference_sha256": reference.fingerprint,
        "trial_spec_sha256": spec.spec_sha256,
        "placement_sha256": placement.fingerprint,
        "runtime_identity_sha256": ordered_identity_sha,
        "stdout_sha256": trial_result["stdout_sha256"],
        "correct": True,
        "level": "E3",
        "qualification_evidence_sha256": evidence.evidence_sha256,
        "qualified": True,
        "real_model_inference": True,
        "executable": False,
    }
    evaluation["evaluation_sha256"] = _canonical_sha256(evaluation)
    return evaluation, evidence
