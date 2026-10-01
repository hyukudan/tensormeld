"""Bounded backend-readiness self-test for the pinned llama.cpp test-backend-ops target.

This adapter proves a narrow fact: the explicitly bound backend was initialized and
executed at least one upstream ADD correctness case. It does not load a model, qualify
all operators, create a resource reservation, or authorize inference.
"""
from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence

from .config_v2 import Config
from .llamacpp_probe import LLAMACPP_PINNED_COMMIT, MAX_BINARY_BYTES
from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates

SELF_TEST_SCHEMA = "tensormeld/llamacpp-backend-self-test-v1"
BOUND_RESULT_SCHEMA = "tensormeld/llamacpp-device-binding-result-v1"
UPSTREAM_TARGET = "test-backend-ops"
SELF_TEST_OPERATION = "ADD"
DEFAULT_TIMEOUT_S = 30.0
MAX_OUTPUT_BYTES = 2 * 1024 * 1024

_SQL_FIELDS = (
    "test_time",
    "build_commit",
    "backend_name",
    "op_name",
    "op_params",
    "test_mode",
    "supported",
    "passed",
    "error_message",
    "time_us",
    "flops",
    "bandwidth_gb_s",
    "memory_kb",
    "n_runs",
    "device_description",
    "backend_reg_name",
)
_SQL_TYPES = {
    "supported": "INTEGER",
    "passed": "INTEGER",
    "memory_kb": "INTEGER",
    "n_runs": "INTEGER",
    "time_us": "REAL",
    "flops": "REAL",
    "bandwidth_gb_s": "REAL",
}
_SQL_HEADER = (
    "CREATE TABLE IF NOT EXISTS test_backend_ops (",
    *tuple(
        f"  {field} {_SQL_TYPES.get(field, 'TEXT')}{',' if i < len(_SQL_FIELDS) - 1 else ''}"
        for i, field in enumerate(_SQL_FIELDS)
    ),
    ");",
)
_INSERT_PREFIX = (
    "INSERT INTO test_backend_ops (" + ", ".join(_SQL_FIELDS) + ") VALUES ("
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _hash_file(path: Path) -> str:
    size = path.stat().st_size
    if size <= 0 or size > MAX_BINARY_BYTES:
        raise ValidationError(
            "test-backend-ops binary size is outside the trusted-local limit"
        )
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_runner(argv: Sequence[str], timeout: float) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            list(argv),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=None,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(
            f"llama.cpp backend self-test timed out after {timeout:g}s"
        ) from exc
    return completed.returncode, completed.stdout, completed.stderr


def _bounded_decode(stdout: bytes, stderr: bytes) -> tuple[str, str]:
    if len(stdout) + len(stderr) > MAX_OUTPUT_BYTES:
        raise ValidationError(
            f"backend self-test output exceeds {MAX_OUTPUT_BYTES} bytes"
        )
    try:
        return stdout.decode("utf-8"), stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("backend self-test output is not UTF-8") from exc


def _parse_sql_rows(stdout: str) -> list[dict[str, str]]:
    """Parse only the pinned tool's SQL printer; never execute its SQL text."""
    lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]
    if len(lines) < len(_SQL_HEADER) + 1:
        raise ValidationError("test-backend-ops SQL output has no test result rows")
    if tuple(lines[: len(_SQL_HEADER)]) != _SQL_HEADER:
        raise ValidationError("test-backend-ops SQL header does not match pinned contract")

    rows: list[dict[str, str]] = []
    for line in lines[len(_SQL_HEADER) :]:
        if not line.startswith(_INSERT_PREFIX) or not line.endswith(");"):
            raise ValidationError(
                "test-backend-ops SQL output contains an unexpected record"
            )
        payload = line[len(_INSERT_PREFIX) : -2]
        try:
            parsed = list(
                csv.reader(
                    io.StringIO(payload),
                    delimiter=",",
                    quotechar="'",
                    skipinitialspace=True,
                    strict=True,
                )
            )
        except csv.Error as exc:
            raise ValidationError(
                "test-backend-ops SQL result row is malformed"
            ) from exc
        if len(parsed) != 1 or len(parsed[0]) != len(_SQL_FIELDS):
            raise ValidationError(
                "test-backend-ops SQL result row has an unexpected field count"
            )
        rows.append(dict(zip(_SQL_FIELDS, parsed[0])))
    return rows


def _validate_commit(value: str) -> str:
    commit = value.lower()
    if not _COMMIT_RE.fullmatch(commit) or not LLAMACPP_PINNED_COMMIT.startswith(commit):
        raise ValidationError(
            "test-backend-ops result does not identify the pinned llama.cpp revision"
        )
    return commit


def _validate_bound_target(
    config: Config, bound_result: dict[str, Any], tensormeld_device_id: str
) -> tuple[str, str, dict[str, Any]]:
    if bound_result.get("result_schema") != BOUND_RESULT_SCHEMA:
        raise ValidationError(f"bound result: expected {BOUND_RESULT_SCHEMA}")
    if bound_result.get("config_sha256") != config.fingerprint:
        raise ValidationError("bound result config identity does not match current config")
    if bound_result.get("qualified") is not False or bound_result.get("executable") is not False:
        raise ValidationError("bound result cannot self-promote qualification/execution")

    probe_sha = bound_result.get("probe_artifact_sha256")
    binding_sha = bound_result.get("binding_sha256")
    if not isinstance(probe_sha, str) or not _SHA256_RE.fullmatch(probe_sha):
        raise ValidationError("bound result has invalid probe artifact identity")
    if not isinstance(binding_sha, str) or not _SHA256_RE.fullmatch(binding_sha):
        raise ValidationError("bound result has invalid binding identity")

    devices = {d.id: d for d in config.devices}
    device = devices.get(tensormeld_device_id)
    if device is None:
        raise ValidationError(f"unknown TensorMeld device {tensormeld_device_id}")

    mappings = bound_result.get("resolved_mappings")
    if not isinstance(mappings, list):
        raise ValidationError("bound result resolved_mappings must be a list")
    matches = [
        item for item in mappings
        if isinstance(item, dict)
        and item.get("tensormeld_device_id") == tensormeld_device_id
    ]
    if len(matches) != 1:
        raise ValidationError(
            "target TensorMeld device must have exactly one resolved native mapping"
        )
    mapping = matches[0]
    engine_name = mapping.get("engine_device_name")
    if not isinstance(engine_name, str) or not engine_name.strip():
        raise ValidationError("resolved native mapping has no engine device name")
    if mapping.get("backend_from_config") != device.backend:
        raise ValidationError("resolved native mapping backend no longer matches config")

    observation = bound_result.get("runtime_observation")
    if not isinstance(observation, dict):
        raise ValidationError("bound result has no runtime observation")
    if observation.get("runtime_observation_schema") != "tensormeld/runtime-observation-v1":
        raise ValidationError("bound result has an unexpected runtime observation schema")
    if observation.get("config_sha256") != config.fingerprint:
        raise ValidationError("bound runtime observation config identity mismatch")
    if observation.get("qualified") is not False or observation.get("executable") is not False:
        raise ValidationError("bound runtime observation cannot self-promote")
    runtime_devices = observation.get("devices")
    if not isinstance(runtime_devices, dict):
        raise ValidationError("bound runtime observation devices must be an object")
    runtime_device = runtime_devices.get(tensormeld_device_id)
    if not isinstance(runtime_device, dict):
        raise ValidationError("target device is absent from bound runtime observation")
    if runtime_device.get("backend") != device.backend:
        raise ValidationError("bound runtime observation backend mismatch")
    if runtime_device.get("state") != "observed":
        raise ValidationError(
            "backend self-test accepts only a freshly observed, not already-ready, device"
        )

    return engine_name, device.backend, observation


def self_test_llamacpp_backend(
    test_binary: str | Path,
    *,
    config: Config,
    bound_result: dict[str, Any],
    tensormeld_device_id: str,
    trusted_local_binary: bool = False,
    expected_artifact_sha256: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner: Callable[[Sequence[str], float], tuple[int, bytes, bytes]] | None = None,
) -> dict[str, Any]:
    if not trusted_local_binary:
        raise ValidationError(
            "backend self-test requires explicit trusted_local_binary approval"
        )
    if expected_artifact_sha256 is None:
        raise ValidationError(
            "backend self-test requires an expected test-backend-ops SHA-256"
        )
    expected = expected_artifact_sha256.lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValidationError("expected test-backend-ops SHA-256 must be 64 hex characters")
    if not (0.1 <= timeout_s <= 120.0):
        raise ValidationError("timeout_s must be within 0.1..120 seconds")

    path = Path(test_binary).resolve(strict=True)
    if not path.is_file():
        raise ValidationError("test-backend-ops path must be a regular file")
    artifact_sha256 = _hash_file(path)
    if artifact_sha256 != expected:
        raise ValidationError(
            "test-backend-ops SHA-256 does not match expected artifact identity"
        )

    engine_name, backend_from_config, observation = _validate_bound_target(
        config, bound_result, tensormeld_device_id
    )

    argv = (
        str(path),
        "test",
        "-b",
        engine_name,
        "-o",
        SELF_TEST_OPERATION,
        "--output",
        "sql",
        "-j",
        "1",
    )
    run = runner or _default_runner
    rc, stdout_raw, stderr_raw = run(argv, timeout_s)
    stdout, _stderr = _bounded_decode(stdout_raw, stderr_raw)
    if rc != 0:
        raise ValidationError(
            f"test-backend-ops failed with exit code {rc}"
        )

    rows = _parse_sql_rows(stdout)
    target_rows = [
        row for row in rows
        if row["backend_name"] == engine_name
        and row["op_name"] == SELF_TEST_OPERATION
        and row["test_mode"] == "test"
    ]
    if not target_rows:
        raise ValidationError(
            "test-backend-ops returned success without proving target-backend execution"
        )

    observed_commits = {_validate_commit(row["build_commit"]) for row in target_rows}
    supported_rows = [row for row in target_rows if row["supported"] == "1"]
    passed_rows = [
        row for row in supported_rows
        if row["passed"] == "1" and row["error_message"] == ""
    ]
    invalid_flags = [
        row for row in target_rows
        if row["supported"] not in {"0", "1"} or row["passed"] not in {"0", "1"}
    ]
    failed_supported = [
        row for row in supported_rows
        if row["passed"] != "1" or row["error_message"] != ""
    ]
    if invalid_flags:
        raise ValidationError("test-backend-ops emitted invalid support/pass flags")
    if failed_supported:
        raise ValidationError("target backend reported a failed supported ADD test")
    if not passed_rows:
        raise ValidationError(
            "target backend initialized but no supported ADD test was executed successfully"
        )

    promoted = deepcopy(observation)
    promoted["devices"][tensormeld_device_id]["state"] = "ready"
    promoted["qualified"] = False
    promoted["executable"] = False

    return {
        "self_test_schema": SELF_TEST_SCHEMA,
        "engine": "llama.cpp",
        "upstream_target": UPSTREAM_TARGET,
        "pinned_source_revision": LLAMACPP_PINNED_COMMIT,
        "observed_source_revisions": sorted(observed_commits),
        "test_artifact_sha256": artifact_sha256,
        "probe_artifact_sha256": bound_result["probe_artifact_sha256"],
        "binding_sha256": bound_result["binding_sha256"],
        "config_sha256": config.fingerprint,
        "tensormeld_device_id": tensormeld_device_id,
        "engine_device_name": engine_name,
        "backend_from_config": backend_from_config,
        "operation": SELF_TEST_OPERATION,
        "result_rows": len(target_rows),
        "supported_rows": len(supported_rows),
        "passed_rows": len(passed_rows),
        "backend_initialized": True,
        "backend_executed": True,
        "evidence_level": "E2",
        "model_loaded": False,
        "listener_started": False,
        "reservation_created": False,
        "execution_source": "injected-runner" if runner is not None else "native-subprocess",
        "runtime_observation": promoted,
        "qualified": False,
        "executable": False,
        "warnings": [
            "Injected-runner results are portable fixture evidence only and must not be retained as hardware evidence."
            if runner is not None
            else "The native subprocess executed the pinned self-test contract; broader hardware qualification remains separate.",
            "E2 here is narrow backend-readiness evidence, not model/operator coverage.",
            "Runtime ready does not create a resource lease or authorize execution.",
            "Portable fixture tests validate this contract but are not hardware evidence.",
            "No CUDA/HIP, Windows GPU, performance, or distributed-inference claim follows from this record.",
        ],
    }


def load_llamacpp_bound_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("llama.cpp bound result exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid llama.cpp bound-result JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("llama.cpp bound result: expected object")
    if value.get("result_schema") != BOUND_RESULT_SCHEMA:
        raise ValidationError(f"llama.cpp bound result: expected {BOUND_RESULT_SCHEMA}")
    return value
