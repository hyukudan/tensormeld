"""Trusted-local, no-model probe for one pinned llama.cpp revision.

Only --version and --list-devices are executed. The output is bounded and parsed into
observations. Engine-local device names are not automatically mapped to TensorMeld IDs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
from typing import Callable, Sequence

from .schema import ValidationError

LLAMACPP_PINNED_COMMIT = "552f18f912a32ea86edf82e2b76431cb7131538d"
MAX_BINARY_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 64 * 1024
MAX_DEVICES = 128
DEFAULT_TIMEOUT_S = 5.0

_VERSION_RE = re.compile(
    r"^version:\s*(?P<version>.+?)\s+\(build\s+(?P<build>\d+),\s+commit\s+(?P<commit>[0-9a-fA-F]{7,40})\)\s*$"
)
_BUILT_RE = re.compile(r"^built with\s+(?P<compiler>.+?)\s+for\s+(?P<target>.+?)\s*$")
_DEVICE_RE = re.compile(
    r"^\s{2}(?P<name>[^:]+):\s+(?P<description>.+?)\s+"
    r"\((?P<total>\d+)\s+MiB,\s+(?P<free>\d+)\s+MiB free\)\s*$"
)


def _hash_file(path: Path) -> str:
    size = path.stat().st_size
    if size <= 0 or size > MAX_BINARY_BYTES:
        raise ValidationError("llama.cpp binary size is outside the trusted-local probe limit")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
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
        raise ValidationError(f"llama.cpp probe timed out after {timeout:g}s") from exc
    return completed.returncode, completed.stdout, completed.stderr


def _bounded_text(stdout: bytes, stderr: bytes, where: str) -> str:
    if len(stdout) + len(stderr) > MAX_OUTPUT_BYTES:
        raise ValidationError(f"{where}: output exceeds {MAX_OUTPUT_BYTES} bytes")
    try:
        # llama.cpp historically prints build info to stderr for common args while
        # device lists are stdout; preserve both without assuming one stream.
        return (stdout + (b"\n" if stdout and stderr else b"") + stderr).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(f"{where}: output is not UTF-8") from exc


def _parse_version(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    version = next((_VERSION_RE.match(line) for line in lines if _VERSION_RE.match(line)), None)
    built = next((_BUILT_RE.match(line) for line in lines if _BUILT_RE.match(line)), None)
    if version is None or built is None:
        raise ValidationError("llama.cpp --version output does not match the pinned contract")
    commit = version.group("commit").lower()
    if len(commit) < 7 or not LLAMACPP_PINNED_COMMIT.startswith(commit):
        raise ValidationError(
            f"llama.cpp source revision mismatch: observed {commit}, expected pinned "
            f"{LLAMACPP_PINNED_COMMIT}"
        )
    return {
        "version": version.group("version"),
        "build": int(version.group("build")),
        "commit": commit,
        "compiler": built.group("compiler"),
        "target": built.group("target"),
    }


def _parse_devices(text: str) -> list[dict]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Available devices:")
    except StopIteration as exc:
        raise ValidationError("llama.cpp --list-devices output has no Available devices header") from exc
    payload = lines[start + 1 :]
    if payload == ["  (none)"] or payload == ["(none)"]:
        return []
    devices = []
    for line in payload:
        match = _DEVICE_RE.match(line)
        if match is None:
            raise ValidationError(f"unrecognized llama.cpp device line: {line!r}")
        total_mib = int(match.group("total"))
        free_mib = int(match.group("free"))
        if total_mib <= 0 or free_mib > total_mib:
            raise ValidationError("invalid memory values in llama.cpp device list")
        devices.append({
            "engine_device_name": match.group("name").strip(),
            "description": match.group("description").strip(),
            "total_bytes": total_mib * 1024 * 1024,
            "free_bytes": free_mib * 1024 * 1024,
        })
        if len(devices) > MAX_DEVICES:
            raise ValidationError("llama.cpp device list exceeds probe limit")
    names = [d["engine_device_name"] for d in devices]
    if len(names) != len(set(names)):
        raise ValidationError("llama.cpp device list contains duplicate names")
    return devices


def probe_llamacpp(
    binary: str | Path,
    *,
    trusted_local_binary: bool = False,
    expected_artifact_sha256: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    runner: Callable[[Sequence[str], float], tuple[int, bytes, bytes]] | None = None,
) -> dict:
    if not trusted_local_binary:
        raise ValidationError("native probe requires explicit trusted_local_binary approval")
    path = Path(binary).resolve(strict=True)
    if not path.is_file():
        raise ValidationError("llama.cpp binary path must be a regular file")
    if not (0.1 <= timeout_s <= 30.0):
        raise ValidationError("timeout_s must be within 0.1..30 seconds")
    artifact_sha256 = _hash_file(path)
    if expected_artifact_sha256 is not None:
        expected = expected_artifact_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValidationError("expected artifact SHA-256 must be 64 lowercase hex characters")
        if artifact_sha256 != expected:
            raise ValidationError("llama.cpp binary SHA-256 does not match expected artifact identity")

    run = runner or _default_runner
    version_rc, version_out, version_err = run((str(path), "--version"), timeout_s)
    if version_rc != 0:
        raise ValidationError(f"llama.cpp --version failed with exit code {version_rc}")
    version = _parse_version(_bounded_text(version_out, version_err, "--version"))

    devices_rc, devices_out, devices_err = run((str(path), "--list-devices"), timeout_s)
    if devices_rc != 0:
        raise ValidationError(f"llama.cpp --list-devices failed with exit code {devices_rc}")
    devices = _parse_devices(_bounded_text(devices_out, devices_err, "--list-devices"))

    return {
        "probe_schema": "tensormeld/llamacpp-probe-v1",
        "engine": "llama.cpp",
        "pinned_source_revision": LLAMACPP_PINNED_COMMIT,
        "artifact_sha256": artifact_sha256,
        "binary_name": path.name,
        "build": version,
        "devices": devices,
        "device_identity_mapping": "unresolved",
        "model_loaded": False,
        "listener_started": False,
        "qualified": False,
        "executable": False,
        "warnings": [
            "Engine-local device names are observations, not TensorMeld device identities.",
            "Reported free memory is transient and is not a reservation.",
            "No model was loaded and no operator/model correctness was tested.",
            "This probe does not establish CUDA/HIP or distributed-inference qualification.",
        ],
    }
