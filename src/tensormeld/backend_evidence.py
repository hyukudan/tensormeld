"""Retained, identity-bound backend-readiness evidence.

A retained record proves only that a native backend self-test ran successfully for an
exact set of TensorMeld/llama.cpp identities. Version 2 also binds a stable live runtime
identity covering worker build, OS, driver/runtime, physical device and topology.

An exact current identity match may restore only the narrow runtime backend-ready fact.
It still does not reserve memory, qualify a model, or authorize inference.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .config_v2 import Config
from .llamacpp_probe import LLAMACPP_PINNED_COMMIT
from .llamacpp_selftest import (
    BOUND_RESULT_SCHEMA,
    SELF_TEST_OPERATION,
    SELF_TEST_SCHEMA,
    UPSTREAM_TARGET,
)
from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates
from .runtime_identity import RuntimeIdentity

EVIDENCE_SCHEMA = "tensormeld/backend-readiness-evidence-v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INVALIDATION_KEYS = (
    "config_sha256",
    "probe_artifact_sha256",
    "binding_sha256",
    "test_artifact_sha256",
    "pinned_source_revision",
    "tensormeld_device_id",
    "engine_device_name",
    "backend",
    "runtime_identity_sha256",
)


def _sha256_hex(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{where}: expected SHA-256 hex")
    normalized = value.lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValidationError(f"{where}: expected SHA-256 hex")
    return normalized


def _nonempty_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValidationError(
            f"{where}: expected nonempty text up to 256 characters"
        )
    return value


def _positive_int(value: Any, where: str) -> int:
    if type(value) is not int or value < 1:
        raise ValidationError(f"{where}: expected positive integer")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("evidence contains non-canonical JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _validate_native_self_test(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValidationError("backend self-test result: expected object")
    if result.get("self_test_schema") != SELF_TEST_SCHEMA:
        raise ValidationError(
            f"backend self-test result: expected {SELF_TEST_SCHEMA}"
        )
    if (
        result.get("engine") != "llama.cpp"
        or result.get("upstream_target") != UPSTREAM_TARGET
    ):
        raise ValidationError("backend self-test result: unexpected engine/target")
    if result.get("pinned_source_revision") != LLAMACPP_PINNED_COMMIT:
        raise ValidationError(
            "backend self-test result: pinned source revision mismatch"
        )
    if result.get("execution_source") != "native-subprocess":
        raise ValidationError(
            "only native-subprocess backend self-tests may be retained as E2 evidence"
        )
    if result.get("evidence_level") != "E2":
        raise ValidationError("backend self-test result: expected E2 evidence")
    if result.get("operation") != SELF_TEST_OPERATION:
        raise ValidationError("backend self-test result: unexpected operation")
    for key in ("backend_initialized", "backend_executed"):
        if result.get(key) is not True:
            raise ValidationError(f"backend self-test result: {key} must be true")
    for key in (
        "model_loaded",
        "listener_started",
        "reservation_created",
        "qualified",
        "executable",
    ):
        if result.get(key) is not False:
            raise ValidationError(f"backend self-test result: {key} must remain false")

    result_rows = _positive_int(result.get("result_rows"), "result_rows")
    supported_rows = _positive_int(result.get("supported_rows"), "supported_rows")
    passed_rows = _positive_int(result.get("passed_rows"), "passed_rows")
    if not passed_rows <= supported_rows <= result_rows:
        raise ValidationError("backend self-test row counts are inconsistent")

    observed = result.get("observed_source_revisions")
    if not isinstance(observed, list) or not observed:
        raise ValidationError(
            "backend self-test result: missing observed source revisions"
        )
    for item in observed:
        if (
            not isinstance(item, str)
            or len(item) < 7
            or not LLAMACPP_PINNED_COMMIT.startswith(item.lower())
        ):
            raise ValidationError(
                "backend self-test result: observed source revision mismatch"
            )

    config_sha = _sha256_hex(result.get("config_sha256"), "config_sha256")
    node_id = _nonempty_text(result.get("node_id"), "node_id")
    test_sha = _sha256_hex(
        result.get("test_artifact_sha256"), "test_artifact_sha256"
    )
    probe_sha = _sha256_hex(
        result.get("probe_artifact_sha256"), "probe_artifact_sha256"
    )
    binding_sha = _sha256_hex(result.get("binding_sha256"), "binding_sha256")
    device_id = _nonempty_text(
        result.get("tensormeld_device_id"), "tensormeld_device_id"
    )
    engine_name = _nonempty_text(
        result.get("engine_device_name"), "engine_device_name"
    )
    backend = _nonempty_text(
        result.get("backend_from_config"), "backend_from_config"
    )

    observation = result.get("runtime_observation")
    if not isinstance(observation, dict):
        raise ValidationError(
            "backend self-test result: missing runtime observation"
        )
    if (
        observation.get("runtime_observation_schema")
        != "tensormeld/runtime-observation-v1"
    ):
        raise ValidationError(
            "backend self-test result: unexpected runtime observation schema"
        )
    if observation.get("config_sha256") != config_sha:
        raise ValidationError(
            "backend self-test result: runtime config identity mismatch"
        )
    if (
        observation.get("qualified") is not False
        or observation.get("executable") is not False
    ):
        raise ValidationError(
            "backend self-test result: runtime observation cannot self-promote"
        )
    runtime_devices = observation.get("devices")
    runtime_device = (
        runtime_devices.get(device_id)
        if isinstance(runtime_devices, dict)
        else None
    )
    if not isinstance(runtime_device, dict):
        raise ValidationError(
            "backend self-test result: target runtime device is missing"
        )
    if (
        runtime_device.get("state") != "ready"
        or runtime_device.get("backend") != backend
    ):
        raise ValidationError(
            "backend self-test result: target device is not ready on expected backend"
        )

    return {
        "config_sha256": config_sha,
        "node_id": node_id,
        "test_artifact_sha256": test_sha,
        "probe_artifact_sha256": probe_sha,
        "binding_sha256": binding_sha,
        "tensormeld_device_id": device_id,
        "engine_device_name": engine_name,
        "backend": backend,
        "result_rows": result_rows,
        "supported_rows": supported_rows,
        "passed_rows": passed_rows,
        "observed_source_revisions": sorted(
            {item.lower() for item in observed}
        ),
    }


def retain_llamacpp_backend_evidence(
    self_test_result: dict[str, Any],
    *,
    runtime_identity: RuntimeIdentity,
) -> dict[str, Any]:
    """Create a deterministic retained record from one genuine native self-test result."""
    checked = _validate_native_self_test(self_test_result)
    if runtime_identity.node_id != checked["node_id"]:
        raise ValidationError(
            "runtime identity applies to a different node than the native self-test"
        )
    if runtime_identity.tensormeld_device_id != checked["tensormeld_device_id"]:
        raise ValidationError(
            "runtime identity applies to a different TensorMeld device"
        )
    payload = {
        "evidence_schema": EVIDENCE_SCHEMA,
        "engine": "llama.cpp",
        "upstream_target": UPSTREAM_TARGET,
        "pinned_source_revision": LLAMACPP_PINNED_COMMIT,
        "observed_source_revisions": checked["observed_source_revisions"],
        "test_artifact_sha256": checked["test_artifact_sha256"],
        "probe_artifact_sha256": checked["probe_artifact_sha256"],
        "binding_sha256": checked["binding_sha256"],
        "config_sha256": checked["config_sha256"],
        "node_id": checked["node_id"],
        "tensormeld_device_id": checked["tensormeld_device_id"],
        "engine_device_name": checked["engine_device_name"],
        "backend": checked["backend"],
        "runtime_identity": runtime_identity.as_record(),
        "runtime_identity_sha256": runtime_identity.identity_sha256,
        "operation": SELF_TEST_OPERATION,
        "result_rows": checked["result_rows"],
        "supported_rows": checked["supported_rows"],
        "passed_rows": checked["passed_rows"],
        "retained_self_test_sha256": _canonical_sha256(self_test_result),
        "native_execution_recorded": True,
        "backend_readiness_proven": True,
        "qualification_scope": "backend-readiness-only",
        "invalidation_keys": list(_INVALIDATION_KEYS),
        "requires_live_runtime_recheck": True,
        "reservation_created": False,
        "qualified": False,
        "executable": False,
    }
    return {**payload, "evidence_sha256": _canonical_sha256(payload)}


def _validate_evidence_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("backend readiness evidence: expected object")
    expected_keys = {
        "evidence_schema",
        "engine",
        "upstream_target",
        "pinned_source_revision",
        "observed_source_revisions",
        "test_artifact_sha256",
        "probe_artifact_sha256",
        "binding_sha256",
        "config_sha256",
        "node_id",
        "node_id",
        "tensormeld_device_id",
        "engine_device_name",
        "backend",
        "runtime_identity",
        "runtime_identity_sha256",
        "operation",
        "result_rows",
        "supported_rows",
        "passed_rows",
        "retained_self_test_sha256",
        "native_execution_recorded",
        "backend_readiness_proven",
        "qualification_scope",
        "invalidation_keys",
        "requires_live_runtime_recheck",
        "reservation_created",
        "qualified",
        "executable",
        "evidence_sha256",
    }
    if set(value) != expected_keys:
        raise ValidationError(
            "backend readiness evidence: unexpected or missing fields"
        )
    if value["evidence_schema"] != EVIDENCE_SCHEMA:
        raise ValidationError(
            f"backend readiness evidence: expected {EVIDENCE_SCHEMA}"
        )
    if (
        value["engine"] != "llama.cpp"
        or value["upstream_target"] != UPSTREAM_TARGET
    ):
        raise ValidationError(
            "backend readiness evidence: unexpected engine/target"
        )
    if value["pinned_source_revision"] != LLAMACPP_PINNED_COMMIT:
        raise ValidationError(
            "backend readiness evidence: source revision mismatch"
        )
    if value["operation"] != SELF_TEST_OPERATION:
        raise ValidationError("backend readiness evidence: operation mismatch")
    for key in (
        "native_execution_recorded",
        "backend_readiness_proven",
        "requires_live_runtime_recheck",
    ):
        if value[key] is not True:
            raise ValidationError(
                f"backend readiness evidence: {key} must be true"
            )
    for key in ("reservation_created", "qualified", "executable"):
        if value[key] is not False:
            raise ValidationError(
                f"backend readiness evidence: {key} must remain false"
            )
    if value["qualification_scope"] != "backend-readiness-only":
        raise ValidationError(
            "backend readiness evidence: qualification scope mismatch"
        )
    if value["invalidation_keys"] != list(_INVALIDATION_KEYS):
        raise ValidationError(
            "backend readiness evidence: invalidation key contract mismatch"
        )

    for key in (
        "test_artifact_sha256",
        "probe_artifact_sha256",
        "binding_sha256",
        "config_sha256",
        "runtime_identity_sha256",
        "retained_self_test_sha256",
        "evidence_sha256",
    ):
        _sha256_hex(value[key], key)
    for key in (
        "tensormeld_device_id",
        "engine_device_name",
        "backend",
    ):
        _nonempty_text(value[key], key)
    runtime_identity_raw = value["runtime_identity"]
    if not isinstance(runtime_identity_raw, dict):
        raise ValidationError("backend readiness evidence: runtime_identity must be an object")
    runtime_identity_payload = dict(runtime_identity_raw)
    recorded_runtime_identity_sha = runtime_identity_payload.pop("identity_sha256", None)
    retained_runtime_identity = RuntimeIdentity.parse(runtime_identity_payload)
    if (
        recorded_runtime_identity_sha != retained_runtime_identity.identity_sha256
        or value["runtime_identity_sha256"] != retained_runtime_identity.identity_sha256
    ):
        raise ValidationError(
            "backend readiness evidence: runtime identity fingerprint mismatch"
        )
    if retained_runtime_identity.node_id != value["node_id"]:
        raise ValidationError(
            "backend readiness evidence: runtime identity node mismatch"
        )
    if retained_runtime_identity.tensormeld_device_id != value["tensormeld_device_id"]:
        raise ValidationError(
            "backend readiness evidence: runtime identity device mismatch"
        )
    result_rows = _positive_int(value["result_rows"], "result_rows")
    supported_rows = _positive_int(value["supported_rows"], "supported_rows")
    passed_rows = _positive_int(value["passed_rows"], "passed_rows")
    if not passed_rows <= supported_rows <= result_rows:
        raise ValidationError(
            "backend readiness evidence: inconsistent row counts"
        )
    observed = value["observed_source_revisions"]
    if not isinstance(observed, list) or not observed:
        raise ValidationError(
            "backend readiness evidence: missing observed source revisions"
        )
    for item in observed:
        if (
            not isinstance(item, str)
            or len(item) < 7
            or not LLAMACPP_PINNED_COMMIT.startswith(item)
        ):
            raise ValidationError(
                "backend readiness evidence: observed source revision mismatch"
            )

    payload = dict(value)
    recorded = payload.pop("evidence_sha256")
    if _canonical_sha256(payload) != recorded:
        raise ValidationError(
            "backend readiness evidence: fingerprint mismatch"
        )
    return value


def validate_llamacpp_backend_evidence(
    evidence: dict[str, Any],
    *,
    config: Config,
    bound_result: dict[str, Any],
    expected_test_artifact_sha256: str,
    tensormeld_device_id: str,
    current_runtime_identity: RuntimeIdentity,
) -> dict[str, Any]:
    """Validate exact retained identities including stable live runtime identity."""
    value = _validate_evidence_record(evidence)
    if value["config_sha256"] != config.fingerprint:
        raise ValidationError(
            "retained E2 evidence does not apply to current config"
        )
    if value["node_id"] != current_runtime_identity.node_id:
        raise ValidationError(
            "retained E2 evidence applies to a different node"
        )
    if value["tensormeld_device_id"] != tensormeld_device_id:
        raise ValidationError(
            "retained E2 evidence applies to a different TensorMeld device"
        )
    if value["test_artifact_sha256"] != _sha256_hex(
        expected_test_artifact_sha256,
        "expected_test_artifact_sha256",
    ):
        raise ValidationError(
            "retained E2 evidence test artifact identity mismatch"
        )
    if bound_result.get("result_schema") != BOUND_RESULT_SCHEMA:
        raise ValidationError(
            f"bound result: expected {BOUND_RESULT_SCHEMA}"
        )
    if bound_result.get("config_sha256") != config.fingerprint:
        raise ValidationError(
            "bound result config identity does not match current config"
        )
    if (
        bound_result.get("probe_artifact_sha256")
        != value["probe_artifact_sha256"]
    ):
        raise ValidationError(
            "retained E2 evidence probe artifact identity mismatch"
        )
    if bound_result.get("binding_sha256") != value["binding_sha256"]:
        raise ValidationError(
            "retained E2 evidence binding identity mismatch"
        )
    mappings = bound_result.get("resolved_mappings")
    if not isinstance(mappings, list):
        raise ValidationError(
            "bound result resolved_mappings must be a list"
        )
    matches = [
        item
        for item in mappings
        if isinstance(item, dict)
        and item.get("tensormeld_device_id") == tensormeld_device_id
    ]
    if len(matches) != 1:
        raise ValidationError(
            "target device must have exactly one current resolved mapping"
        )
    mapping = matches[0]
    if mapping.get("engine_device_name") != value["engine_device_name"]:
        raise ValidationError(
            "retained E2 evidence engine-device identity mismatch"
        )
    if mapping.get("backend_from_config") != value["backend"]:
        raise ValidationError(
            "retained E2 evidence backend identity mismatch"
        )

    current = bound_result.get("runtime_observation")
    current_devices = (
        current.get("devices")
        if isinstance(current, dict)
        else None
    )
    current_device = (
        current_devices.get(tensormeld_device_id)
        if isinstance(current_devices, dict)
        else None
    )
    if (
        not isinstance(current_device, dict)
        or current_device.get("state") != "observed"
    ):
        raise ValidationError(
            "retained evidence applicability requires a fresh observed binding state"
        )
    if current_device.get("backend") != value["backend"]:
        raise ValidationError(
            "fresh runtime observation backend mismatch"
        )

    retained_identity_record = value["runtime_identity"]
    retained_identity_payload = dict(retained_identity_record)
    retained_identity_payload.pop("identity_sha256", None)
    retained_identity = RuntimeIdentity.parse(retained_identity_payload)
    if bound_result.get("node_id") != current_runtime_identity.node_id:
        raise ValidationError(
            "current runtime identity node does not match bound result"
        )
    if current_runtime_identity.tensormeld_device_id != tensormeld_device_id:
        raise ValidationError(
            "current runtime identity applies to a different TensorMeld device"
        )
    if retained_identity.identity_sha256 != current_runtime_identity.identity_sha256:
        raise ValidationError(
            "retained E2 evidence live runtime identity mismatch"
        )

    promoted_observation = deepcopy(current)
    promoted_observation["devices"][tensormeld_device_id]["state"] = "ready"
    promoted_observation["qualified"] = False
    promoted_observation["executable"] = False

    return {
        "evidence_schema": EVIDENCE_SCHEMA,
        "evidence_sha256": value["evidence_sha256"],
        "identity_applicable": True,
        "backend_readiness_recorded": True,
        "tensormeld_device_id": tensormeld_device_id,
        "runtime_identity_sha256": current_runtime_identity.identity_sha256,
        "requires_live_runtime_recheck": False,
        "runtime_ready": True,
        "runtime_observation": promoted_observation,
        "reservation_created": False,
        "qualified": False,
        "executable": False,
        "reason": (
            "Exact retained backend and stable live runtime identities match. "
            "Only the narrow backend-ready fact is reusable; model qualification "
            "and execution admission remain separate."
        ),
    }


def load_backend_readiness_evidence(
    path: str | Path,
) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError(
            "backend readiness evidence exceeds 2 MiB"
        )
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(
            f"invalid backend readiness evidence JSON: {exc}"
        ) from exc
    return _validate_evidence_record(value)
