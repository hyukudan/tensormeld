"""Owned llama-server child-process lifecycle.

This module owns a revision-pinned llama-server process as a separately approved artifact.
It intentionally does not pretend that llama-cli (currently used by native E3) and
llama-server are the same executable identity. The managed server therefore remains a
pre-integration process primitive until a build/package identity binds both artifacts.

The launch contract is closed:
- exact approved server artifact;
- exact approved GGUF;
- exact post-E3 placement translation;
- loopback-only host;
- one explicit bounded port and context;
- one server slot;
- no caller-provided argv or environment extensions.

No inference request is issued by this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from threading import Lock, Thread
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from .llamacpp_native_trial import ApprovedGGUF
from .llamacpp_placement import LlamaCppPlacementTranslation
from .llamacpp_probe import LLAMACPP_PINNED_COMMIT
from .model_manifest import ModelManifest
from .native_worker import WorkerArtifact
from .schema import ValidationError
from .whole_block_execution import AcceptedExecutionBundle

SERVER_SPEC_SCHEMA = "tensormeld/llamacpp-server-launch-v1"
SERVER_RESULT_SCHEMA = "tensormeld/llamacpp-managed-server-v1"
MAX_LOG_LINE_CHARS = 8192
DEFAULT_LOG_LINES = 80


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
class LlamaCppServerLaunchSpec:
    source_revision: str
    server_artifact_sha256: str
    launcher_artifact_sha256: str | None
    model_manifest_sha256: str
    gguf_sha256: str
    accepted_bundle_sha256: str
    placement_sha256: str
    host: str
    port: int
    context_tokens: int
    argv: tuple[str, ...]
    spec_sha256: str


def build_llamacpp_server_launch_spec(
    *,
    bundle: AcceptedExecutionBundle,
    placement: LlamaCppPlacementTranslation,
    model: ModelManifest,
    llama_server: WorkerArtifact,
    gguf: ApprovedGGUF,
    launcher: WorkerArtifact | None = None,
    context_tokens: int,
    port: int,
) -> LlamaCppServerLaunchSpec:
    if bundle.engine_revision != LLAMACPP_PINNED_COMMIT:
        raise ValidationError("accepted bundle is not pinned to required llama.cpp revision")
    if placement.source_revision != LLAMACPP_PINNED_COMMIT:
        raise ValidationError("llama.cpp server placement revision mismatch")
    if placement.accepted_bundle_sha256 != bundle.bundle_sha256:
        raise ValidationError("llama.cpp server placement belongs to another bundle")
    if placement.fingerprint == "":
        raise ValidationError("llama.cpp server placement has no fingerprint")
    if model.manifest_sha256 != bundle.model_manifest_sha256:
        raise ValidationError("llama.cpp server model manifest does not match bundle")
    if len(model.files) != 1:
        raise ValidationError("initial managed llama-server supports one GGUF file")
    if (
        gguf.file_name != model.files[0].name
        or gguf.sha256 != model.files[0].sha256
    ):
        raise ValidationError("approved GGUF does not match ModelManifest")
    if not 1 <= int(context_tokens) <= 1_048_576:
        raise ValidationError("context_tokens outside supported server bound")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValidationError("port must be an unprivileged TCP port")
    host = "127.0.0.1"

    prefix = (
        (str(launcher.path), str(llama_server.path))
        if launcher is not None
        else (str(llama_server.path),)
    )
    argv = (
        *prefix,
        "-m",
        str(gguf.path),
        *placement.argv_fragment,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(context_tokens),
        "--parallel",
        "1",
        "--no-webui",
        "--jinja",
    )
    canonical = {
        "server_spec_schema": SERVER_SPEC_SCHEMA,
        "source_revision": LLAMACPP_PINNED_COMMIT,
        "server_artifact_sha256": llama_server.sha256,
        "launcher_artifact_sha256": (
            launcher.sha256 if launcher is not None else None
        ),
        "model_manifest_sha256": bundle.model_manifest_sha256,
        "gguf_sha256": gguf.sha256,
        "accepted_bundle_sha256": bundle.bundle_sha256,
        "placement_sha256": placement.fingerprint,
        "host": host,
        "port": port,
        "context_tokens": int(context_tokens),
        "argv": list(argv),
        "real_model_inference": False,
    }
    return LlamaCppServerLaunchSpec(
        LLAMACPP_PINNED_COMMIT,
        llama_server.sha256,
        launcher.sha256 if launcher is not None else None,
        bundle.model_manifest_sha256,
        gguf.sha256,
        bundle.bundle_sha256,
        placement.fingerprint,
        host,
        port,
        int(context_tokens),
        argv,
        _canonical_sha256(canonical),
    )


class ManagedLlamaCppServer:
    """Own one local llama-server child through readiness and shutdown."""

    def __init__(
        self,
        spec: LlamaCppServerLaunchSpec,
        *,
        log_lines: int = DEFAULT_LOG_LINES,
    ) -> None:
        if not 1 <= int(log_lines) <= 512:
            raise ValidationError("log_lines must be within 1..512")
        self.spec = spec
        self._log = deque(maxlen=int(log_lines))
        self._log_lock = Lock()
        self._child: subprocess.Popen[bytes] | None = None
        self._stderr_thread: Thread | None = None
        self.state = "prepared"
        self.ready_seconds: float | None = None

    @property
    def pid(self) -> int | None:
        return None if self._child is None else self._child.pid

    def log_tail(self) -> tuple[str, ...]:
        with self._log_lock:
            return tuple(self._log)

    def _drain_stderr(self, pipe) -> None:
        try:
            for raw in iter(pipe.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if len(line) > MAX_LOG_LINE_CHARS:
                    line = line[:MAX_LOG_LINE_CHARS] + "…"
                with self._log_lock:
                    self._log.append(line)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def start(self) -> dict[str, Any]:
        if self.state != "prepared":
            raise ValidationError(f"llama-server cannot start from state {self.state}")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("LLAMA_ARG_")
        }
        try:
            child = subprocess.Popen(
                list(self.spec.argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                shell=False,
                env=env,
            )
        except OSError as exc:
            self.state = "failed"
            raise ValidationError("failed to start approved llama-server artifact") from exc
        if child.stderr is None:
            child.kill()
            child.wait()
            self.state = "failed"
            raise ValidationError("llama-server stderr pipe was not created")
        self._child = child
        self._stderr_thread = Thread(
            target=self._drain_stderr,
            args=(child.stderr,),
            name="tensormeld-llama-server-stderr",
            daemon=True,
        )
        self._stderr_thread.start()
        self.state = "starting"
        return {
            "server_result_schema": SERVER_RESULT_SCHEMA,
            "status": "STARTED",
            "spec_sha256": self.spec.spec_sha256,
            "server_artifact_sha256": self.spec.server_artifact_sha256,
            "launcher_artifact_sha256": self.spec.launcher_artifact_sha256,
            "pid": child.pid,
            "endpoint": f"http://{self.spec.host}:{self.spec.port}",
            "ready": False,
            "real_model_inference": False,
        }

    def wait_ready(
        self,
        *,
        timeout_s: float = 600.0,
        poll_s: float = 0.1,
    ) -> dict[str, Any]:
        if self.state not in {"starting", "ready"}:
            raise ValidationError(f"llama-server readiness invalid from state {self.state}")
        if self.state == "ready":
            return self._ready_record()
        if not 0.1 <= timeout_s <= 1800:
            raise ValidationError("timeout_s must be within 0.1..1800 seconds")
        if not 0.01 <= poll_s <= 5:
            raise ValidationError("poll_s must be within 0.01..5 seconds")
        child = self._child
        assert child is not None
        started = time.monotonic()
        health_url = f"http://{self.spec.host}:{self.spec.port}/health"
        while time.monotonic() - started < timeout_s:
            code = child.poll()
            if code is not None:
                self.state = "failed"
                raise ValidationError(
                    f"llama-server exited before readiness with code {code}; "
                    f"log_tail={list(self.log_tail())!r}"
                )
            try:
                with urlrequest.urlopen(health_url, timeout=min(1.0, poll_s + 0.25)) as response:
                    if 200 <= int(response.status) < 300:
                        self.ready_seconds = time.monotonic() - started
                        self.state = "ready"
                        return self._ready_record()
            except (urlerror.URLError, TimeoutError, OSError):
                pass
            time.sleep(poll_s)
        tail = list(self.log_tail())
        self.state = "failed"
        try:
            self.stop()
        except Exception:
            pass
        raise ValidationError(
            f"llama-server did not become ready within {timeout_s:g}s; "
            f"log_tail={tail!r}"
        )

    def _ready_record(self) -> dict[str, Any]:
        child = self._child
        assert child is not None
        core = {
            "server_result_schema": SERVER_RESULT_SCHEMA,
            "status": "READY",
            "spec_sha256": self.spec.spec_sha256,
            "server_artifact_sha256": self.spec.server_artifact_sha256,
            "launcher_artifact_sha256": self.spec.launcher_artifact_sha256,
            "pid": child.pid,
            "endpoint": f"http://{self.spec.host}:{self.spec.port}",
            "ready_seconds": self.ready_seconds,
            "ready": True,
            "real_model_inference": False,
        }
        core["result_sha256"] = _canonical_sha256(core)
        return core

    def stop(
        self,
        *,
        terminate_timeout_s: float = 5.0,
        kill_timeout_s: float = 5.0,
    ) -> dict[str, Any]:
        if not 0.1 <= terminate_timeout_s <= 60:
            raise ValidationError("terminate_timeout_s must be within 0.1..60 seconds")
        if not 0.1 <= kill_timeout_s <= 60:
            raise ValidationError("kill_timeout_s must be within 0.1..60 seconds")
        child = self._child
        if child is None:
            self.state = "stopped"
            return {
                "server_result_schema": SERVER_RESULT_SCHEMA,
                "status": "ALREADY_STOPPED",
                "spec_sha256": self.spec.spec_sha256,
                "terminated": True,
                "killed": False,
            }
        killed = False
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=terminate_timeout_s)
            except subprocess.TimeoutExpired:
                killed = True
                child.kill()
                try:
                    child.wait(timeout=kill_timeout_s)
                except subprocess.TimeoutExpired as exc:
                    self.state = "failed"
                    raise ValidationError(
                        "llama-server did not exit after terminate/kill"
                    ) from exc
        code = child.returncode
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
        self._child = None
        self.state = "stopped"
        core = {
            "server_result_schema": SERVER_RESULT_SCHEMA,
            "status": "STOPPED",
            "spec_sha256": self.spec.spec_sha256,
            "server_artifact_sha256": self.spec.server_artifact_sha256,
            "launcher_artifact_sha256": self.spec.launcher_artifact_sha256,
            "exit_code": code,
            "terminated": True,
            "killed": killed,
            "log_tail": list(self.log_tail()),
            "real_model_inference": False,
        }
        core["result_sha256"] = _canonical_sha256(core)
        return core

    def __enter__(self) -> "ManagedLlamaCppServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop()
        except Exception:
            pass
