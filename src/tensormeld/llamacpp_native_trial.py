"""Strict revision-pinned llama.cpp native trial runner.

This module binds an approved local llama-cli artifact, one exact local GGUF file, the
pre-E3 qualification placement and the deterministic llama.cpp placement translation into a
closed subprocess argv.

The runner can execute a real local process, but its result is only trial evidence.
Success never self-promotes model qualification or `real_model_inference`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

from .llamacpp_placement import LlamaCppQualificationPlacement
from .llamacpp_probe import LLAMACPP_PINNED_COMMIT
from .model_manifest import ModelManifest
from .native_worker import WorkerArtifact, approved_worker_artifact
from .schema import ValidationError, text

TRIAL_SCHEMA = "tensormeld/llamacpp-native-trial-v1"
MAX_PROMPT_CHARS = 4096
MAX_STDIO_BYTES = 4 * 1024 * 1024
MAX_CTX = 1_048_576
MAX_PREDICT = 256


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ApprovedGGUF:
    path: Path
    file_name: str
    sha256: str


def approve_single_file_gguf(
    model: ModelManifest,
    path: str | Path,
) -> ApprovedGGUF:
    if len(model.files) != 1:
        raise ValidationError(
            "initial llama.cpp native trial supports exactly one GGUF file"
        )
    expected = model.files[0]
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValidationError("GGUF path must be a regular file")
    if resolved.name != expected.name:
        raise ValidationError("GGUF file name does not match ModelManifest")
    if resolved.stat().st_size != expected.size_bytes:
        raise ValidationError("GGUF file size does not match ModelManifest")
    observed = _hash_file(resolved)
    if observed != expected.sha256:
        raise ValidationError("GGUF SHA-256 does not match ModelManifest")
    return ApprovedGGUF(resolved, expected.name, observed)


@dataclass(frozen=True)
class LlamaCppNativeTrialSpec:
    source_revision: str
    llama_cli_sha256: str
    model_manifest_sha256: str
    gguf_sha256: str
    config_sha256: str
    planning_input_sha256: str
    candidate_plan_sha256: str
    placement_sha256: str
    prompt: str
    context_tokens: int
    predict_tokens: int
    argv: tuple[str, ...]
    spec_sha256: str


def build_llamacpp_native_trial_spec(
    *,
    model: ModelManifest,
    placement: LlamaCppQualificationPlacement,
    llama_cli: WorkerArtifact,
    gguf: ApprovedGGUF,
    prompt: str,
    context_tokens: int,
    predict_tokens: int = 1,
) -> LlamaCppNativeTrialSpec:
    if placement.source_revision != LLAMACPP_PINNED_COMMIT:
        raise ValidationError("qualification placement source revision mismatch")
    if model.manifest_sha256 != placement.model_manifest_sha256:
        raise ValidationError("model manifest does not match qualification placement")
    if gguf.sha256 != model.files[0].sha256:
        raise ValidationError("approved GGUF identity does not match model manifest")
    prompt = text(prompt, "prompt")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValidationError("prompt exceeds 4096-character native trial bound")
    if type(context_tokens) is not int or not 1 <= context_tokens <= MAX_CTX:
        raise ValidationError("context_tokens outside native trial bound")
    if type(predict_tokens) is not int or not 1 <= predict_tokens <= MAX_PREDICT:
        raise ValidationError("predict_tokens outside native trial bound")

    argv = (
        str(llama_cli.path),
        "-m",
        str(gguf.path),
        *placement.argv_fragment,
        "--ctx-size",
        str(context_tokens),
        "--n-predict",
        str(predict_tokens),
        "--prompt",
        prompt,
        "--seed",
        "0",
        "--temp",
        "0",
        "--simple-io",
        "--single-turn",
        "--no-display-prompt",
        "--no-show-timings",
        "--color",
        "off",
    )
    canonical = {
        "trial_schema": TRIAL_SCHEMA,
        "source_revision": LLAMACPP_PINNED_COMMIT,
        "llama_cli_sha256": llama_cli.sha256,
        "model_manifest_sha256": model.manifest_sha256,
        "gguf_sha256": gguf.sha256,
        "config_sha256": placement.config_sha256,
        "planning_input_sha256": placement.planning_input_sha256,
        "candidate_plan_sha256": placement.candidate_plan_sha256,
        "placement_sha256": placement.fingerprint,
        "prompt": prompt,
        "context_tokens": context_tokens,
        "predict_tokens": predict_tokens,
        "argv": list(argv),
    }
    spec_sha = hashlib.sha256(
        __import__("json").dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return LlamaCppNativeTrialSpec(
        LLAMACPP_PINNED_COMMIT,
        llama_cli.sha256,
        model.manifest_sha256,
        gguf.sha256,
        placement.config_sha256,
        placement.planning_input_sha256,
        placement.candidate_plan_sha256,
        placement.fingerprint,
        prompt,
        context_tokens,
        predict_tokens,
        argv,
        spec_sha,
    )


Runner = Callable[[Sequence[str], float], tuple[int, bytes, bytes]]


def _default_runner(argv: Sequence[str], timeout_s: float) -> tuple[int, bytes, bytes]:
    try:
        result = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            timeout=timeout_s,
            check=False,
            env={
                key: value
                for key, value in __import__("os").environ.items()
                if not key.startswith("LLAMA_ARG_")
            },
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(
            f"llama.cpp native trial timed out after {timeout_s:g}s"
        ) from exc
    return result.returncode, result.stdout, result.stderr


def run_llamacpp_native_trial(
    spec: LlamaCppNativeTrialSpec,
    *,
    timeout_s: float = 120.0,
    runner: Runner | None = None,
    execution_source: str = "native-subprocess",
) -> dict[str, Any]:
    if not 0.1 <= timeout_s <= 600:
        raise ValidationError("timeout_s must be within 0.1..600 seconds")
    execution_source = text(execution_source, "execution_source")
    if execution_source not in {"native-subprocess", "fixture-subprocess", "injected-runner"}:
        raise ValidationError("unsupported execution_source")
    if runner is not None and execution_source == "native-subprocess":
        raise ValidationError(
            "injected runner cannot be labeled as native-subprocess evidence"
        )
    actual_runner = runner or _default_runner
    rc, stdout, stderr = actual_runner(spec.argv, timeout_s)
    if len(stdout) + len(stderr) > MAX_STDIO_BYTES:
        raise ValidationError("llama.cpp native trial output exceeds 4 MiB")
    result = {
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
        "execution_source": execution_source,
        "exit_code": rc,
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_sha256": _sha256_bytes(stderr),
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "process_succeeded": rc == 0,
        "stdout_nonempty": rc == 0 and bool(stdout),
        "evidence_level": "E2.5" if execution_source == "native-subprocess" and rc == 0 else "fixture",
        "qualified": False,
        "real_model_inference": False,
        "executable": False,
        "warnings": [
            "A successful native trial is not E3 model correctness qualification.",
            "stdout presence alone is not proof of semantically correct model output.",
            "Injected runners cannot be labeled as native-subprocess evidence.",
        ],
    }
    return result


__all__ = [
    "ApprovedGGUF",
    "LlamaCppNativeTrialSpec",
    "approve_single_file_gguf",
    "approved_worker_artifact",
    "build_llamacpp_native_trial_spec",
    "run_llamacpp_native_trial",
]
