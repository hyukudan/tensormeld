"""Revision-pinned local subprocess foundation for whole-block execution.

This adapter launches only pre-approved local artifacts with exact SHA-256 identities and
a fixed argv shape. Segment/device/unit/payload data is sent as one bounded JSON request
over stdin; peers/users cannot supply shell fragments, executable paths or arbitrary argv.

The current protocol foundation does not claim model inference. A worker response must
state real_model_inference=false until a model backend is separately correctness-qualified.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

from .schema import ValidationError, text
from .whole_block_execution import AcceptedExecutionBundle, MAX_INPUT_BYTES

WORKER_PROTOCOL = "tensormeld/native-whole-block-worker-v1"
DEFAULT_TIMEOUT_S = 30.0
MAX_WORKER_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


def _sha256_hex(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{where}: expected SHA-256 hex")
    value = value.lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValidationError(f"{where}: expected SHA-256 hex")
    return value


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


def _hash_file(path: Path, where: str) -> str:
    if not path.is_file():
        raise ValidationError(f"{where}: expected regular file")
    size = path.stat().st_size
    if not 0 < size <= MAX_ARTIFACT_BYTES:
        raise ValidationError(f"{where}: artifact size outside trusted-local bounds")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkerArtifact:
    path: Path
    sha256: str


def approved_worker_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    where: str = "worker artifact",
) -> WorkerArtifact:
    resolved = Path(path).resolve(strict=True)
    expected = _sha256_hex(expected_sha256, f"{where}.expected_sha256")
    observed = _hash_file(resolved, where)
    if observed != expected:
        raise ValidationError(f"{where}: SHA-256 identity mismatch")
    return WorkerArtifact(resolved, observed)


Runner = Callable[[Sequence[str], bytes, float], tuple[int, bytes, bytes]]


def _default_runner(
    argv: Sequence[str],
    stdin: bytes,
    timeout: float,
) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            list(argv),
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=timeout,
            check=False,
            env=None,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(
            f"native whole-block worker timed out after {timeout:g}s"
        ) from exc
    return completed.returncode, completed.stdout, completed.stderr


class NativeSubprocessWholeBlockBackend:
    """Strict local worker protocol behind one accepted execution bundle."""

    def __init__(
        self,
        *,
        bundle: AcceptedExecutionBundle,
        adapter_id: str,
        engine_revision: str,
        worker_artifact_sha256: str,
        executable: WorkerArtifact,
        program: WorkerArtifact | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner: Runner | None = None,
    ) -> None:
        adapter_id = text(adapter_id, "adapter_id")
        engine_revision = text(engine_revision, "engine_revision")
        worker_sha = _sha256_hex(
            worker_artifact_sha256, "worker_artifact_sha256"
        )
        if bundle.adapter_id != adapter_id:
            raise ValidationError("worker adapter_id does not match accepted bundle")
        if bundle.engine_revision != engine_revision:
            raise ValidationError("worker engine revision does not match accepted bundle")
        if bundle.worker_artifact_sha256 != worker_sha:
            raise ValidationError(
                "worker artifact identity does not match accepted bundle"
            )
        launched_worker_sha = (
            program.sha256 if program is not None else executable.sha256
        )
        if launched_worker_sha != worker_sha:
            raise ValidationError(
                "approved launched worker artifact does not match bundle worker identity"
            )
        if not 0.1 <= timeout_s <= 120:
            raise ValidationError("timeout_s must be within 0.1..120 seconds")
        self.bundle = bundle
        self.bundle_sha256 = bundle.bundle_sha256
        self.adapter_id = adapter_id
        self.engine_revision = engine_revision
        self.worker_artifact_sha256 = worker_sha
        self.executable = executable
        self.program = program
        self.timeout_s = timeout_s
        self.runner = runner or _default_runner

    def _request(
        self,
        *,
        device_id: str,
        unit_ids: tuple[str, ...],
        payload: bytes,
    ) -> dict[str, Any]:
        if device_id not in self.bundle.compute_devices:
            raise ValidationError("segment device is absent from accepted bundle")
        if not unit_ids:
            raise ValidationError("native worker segment has no units")
        if not isinstance(payload, bytes) or len(payload) > MAX_INPUT_BYTES:
            raise ValidationError("native worker payload must be bytes up to 1 MiB")
        segment = {
            "device_id": device_id,
            "unit_ids": list(unit_ids),
        }
        request = {
            "worker_protocol": WORKER_PROTOCOL,
            "adapter_id": self.adapter_id,
            "engine_revision": self.engine_revision,
            "worker_artifact_sha256": self.worker_artifact_sha256,
            "bundle_sha256": self.bundle.bundle_sha256,
            "segment_sha256": _canonical_sha256({
                "bundle_sha256": self.bundle.bundle_sha256,
                **segment,
            }),
            **segment,
            "payload_b64": base64.b64encode(payload).decode("ascii"),
            "launcher_artifact_sha256": self.executable.sha256,
            "program_artifact_sha256": (
                self.program.sha256 if self.program is not None else None
            ),
            "real_model_inference": False,
        }
        return request

    def execute_segment(
        self,
        *,
        device_id: str,
        unit_ids: tuple[str, ...],
        payload: bytes,
    ) -> bytes:
        request = self._request(
            device_id=device_id,
            unit_ids=unit_ids,
            payload=payload,
        )
        request_sha = _canonical_sha256(request)
        encoded = (
            json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_WORKER_OUTPUT_BYTES:
            raise ValidationError("native worker request exceeds bounded protocol size")

        argv = (
            (str(self.executable.path), str(self.program.path), "--tensormeld-worker-v1")
            if self.program is not None
            else (str(self.executable.path), "--tensormeld-worker-v1")
        )
        rc, stdout, stderr = self.runner(argv, encoded, self.timeout_s)
        if len(stdout) + len(stderr) > MAX_WORKER_OUTPUT_BYTES:
            raise ValidationError("native worker output exceeds bounded protocol size")
        if rc != 0:
            raise ValidationError(f"native whole-block worker failed with exit code {rc}")
        try:
            response = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValidationError("native worker returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise ValidationError("native worker response must be an object")
        expected_keys = {
            "worker_protocol",
            "adapter_id",
            "engine_revision",
            "worker_artifact_sha256",
            "bundle_sha256",
            "request_sha256",
            "segment_sha256",
            "status",
            "output_b64",
            "real_model_inference",
        }
        if set(response) != expected_keys:
            raise ValidationError("native worker response fields do not match protocol")
        checks = (
            (response["worker_protocol"] == WORKER_PROTOCOL, "worker protocol"),
            (response["adapter_id"] == self.adapter_id, "adapter identity"),
            (response["engine_revision"] == self.engine_revision, "engine revision"),
            (
                response["worker_artifact_sha256"] == self.worker_artifact_sha256,
                "worker artifact identity",
            ),
            (response["bundle_sha256"] == self.bundle.bundle_sha256, "bundle identity"),
            (response["request_sha256"] == request_sha, "request identity"),
            (
                response["segment_sha256"] == request["segment_sha256"],
                "segment identity",
            ),
            (response["status"] == "ok", "worker status"),
            (
                response["real_model_inference"] is False,
                "real-model-inference boundary",
            ),
        )
        for ok, label in checks:
            if not ok:
                raise ValidationError(f"native worker {label} mismatch")
        try:
            output = base64.b64decode(
                response["output_b64"],
                validate=True,
            )
        except (ValueError, TypeError) as exc:
            raise ValidationError("native worker output_b64 is invalid") from exc
        if len(output) > MAX_INPUT_BYTES:
            raise ValidationError("native worker decoded output exceeds 1 MiB")
        return output
