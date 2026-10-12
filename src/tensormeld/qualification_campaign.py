"""Reproducible private campaigns over closed local qualification stages.

Campaign completion is orchestration evidence, not hardware/model qualification.
No stage accepts shell commands, arbitrary argv, peer endpoints or serving requests.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
from threading import Event, Lock, Thread
import time
import unittest
from uuid import uuid4

from .adapter_contract import load_adapter_capabilities
from .config_v2 import load_config
from .llamacpp_selftest import load_llamacpp_bound_result
from .model_manifest import load_model_manifest, _sha256
from .runtime_identity import load_runtime_identity
from .runtime_model_manifest import load_runtime_model_manifest
from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, items, number, record, text, unique

SCHEMA = "tensormeld/qualification-campaign-v1"
REPO = Path(__file__).resolve().parents[2]
KINDS = frozenset({"validate_config", "host_probe", "runtime_probe", "backend_self_test",
                   "validate_runtime_manifest", "portable_tests", "native_platform_probe",
                   "native_projection_lab", "native_graph_fixture"})
EXPERIMENTS = frozenset({"native_platform_probe", "native_projection_lab", "native_graph_fixture"})
SUITES = {"configuration": "test_config_v2.py", "numerical": "test_numerical_profile.py",
          "request_envelope": "test_request_envelope.py", "native_session": "test_native_session.py",
          "c1_serving": "test_llamacpp_c1_serving.py"}
# Preserve documented runtime/device selections, DLL/library discovery and local
# temporary directories. Unrelated credentials, Python hooks, loader injection,
# and LLAMA_ARG_* overrides do not enter the stage environment.
ENVIRONMENT_KEYS = frozenset({"SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "TEMP", "TMP", "HOME",
    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
    "COMMONPROGRAMFILES", "COMSPEC", "LANG", "LC_ALL", "LC_CTYPE", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    "HIP_PATH", "ROCM_PATH", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HSA_ENABLE_SDMA", "AMD_LOG_LEVEL", "OMP_NUM_THREADS",
    "OMP_WAIT_POLICY"})


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(value, where: str) -> Path:
    value = record(value, where, {"path", "sha256"})
    path = Path(text(value["path"], where + ".path"))
    if not path.is_absolute():
        raise ValidationError(f"{where}: artifact path must be absolute")
    path = path.resolve(strict=True)
    if not path.is_file() or _file_sha(path) != _sha256(value["sha256"], where + ".sha256"):
        raise ValidationError(f"{where}: artifact identity mismatch")
    return path


def _private(path: Path) -> Path:
    if not path.is_absolute():
        raise ValidationError("results_directory must be absolute")
    resolved = path.resolve()
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValidationError("campaign results must remain outside Git checkouts")
    if resolved == REPO or REPO in resolved.parents:
        raise ValidationError("campaign results must remain outside the package checkout")
    return resolved


def code_identity() -> dict:
    paths = list((REPO / "src" / "tensormeld").glob("*.py"))
    paths += list((REPO / "tests").rglob("*.py"))
    paths += list((REPO / "tests").rglob("*.json"))
    paths += list((REPO / "docs" / "examples").glob("*.json"))
    files = {str(path.relative_to(REPO)).replace("\\", "/"): _file_sha(path)
             for path in sorted(paths)}
    return {"source_sha256": _digest(files), "source_files": files,
            "python_artifact_sha256": _file_sha(Path(sys.executable)),
            "python_version": platform.python_version()}


@dataclass(frozen=True)
class QualificationCampaign:
    canonical_json: str
    fingerprint: str
    binding_json: str

    @property
    def config(self) -> dict:
        return json.loads(self.canonical_json)

    @property
    def binding(self) -> dict:
        return json.loads(self.binding_json)

    @classmethod
    def parse(cls, data) -> "QualificationCampaign":
        root = record(data, "campaign", {"campaign_schema", "mode", "host", "artifacts",
                                          "results_directory", "stages"})
        if root["campaign_schema"] != SCHEMA or type(root["mode"]) is not str or root["mode"] not in {"fixture", "native"}:
            raise ValidationError("unsupported campaign schema or provenance mode")
        host = record(root["host"], "host", {"node_id", "hostname", "platform"})
        for key in host:
            text(host[key], "host." + key)
        if host["hostname"] != socket.gethostname() or host["platform"] != platform.system():
            raise ValidationError("campaign is bound to another local host/platform")
        _private(Path(text(root["results_directory"], "results_directory")))
        artifacts = record(root["artifacts"], "artifacts", {
            "config", "model_manifest", "adapter_capabilities", "runtime_manifest", "runtime_identities",
        })
        config = load_config(_artifact(artifacts["config"], "artifacts.config"))
        model = load_model_manifest(_artifact(artifacts["model_manifest"], "artifacts.model_manifest"))
        adapter = load_adapter_capabilities(_artifact(artifacts["adapter_capabilities"], "artifacts.adapter_capabilities"))
        manifest = load_runtime_model_manifest(_artifact(artifacts["runtime_manifest"], "artifacts.runtime_manifest"),
                                               config=config, model=model, adapter=adapter)
        identities = [load_runtime_identity(_artifact(value, "runtime identity")) for value in
                      items(artifacts["runtime_identities"], "runtime_identities", 128, 1)]
        unique([value.tensormeld_device_id for value in identities], "runtime identity devices")
        if host["node_id"] not in {node.id for node in config.nodes}:
            raise ValidationError("host node is absent from config")
        if {value.tensormeld_device_id for value in identities} != {device.id for device in manifest.devices}:
            raise ValidationError("runtime identities must cover exact runtime-manifest devices")
        devices = {device.id: device for device in manifest.devices}
        for value in identities:
            if (value.node_id != devices[value.tensormeld_device_id].node
                    or value.worker_artifact_sha256 != manifest.worker_artifact_sha256):
                raise ValidationError("runtime identity worker/node differs from runtime manifest")
        if root["mode"] == "native" and manifest.provenance != "native-adapter":
            raise ValidationError("fixture runtime manifest cannot enter a native campaign")
        stages = items(root["stages"], "stages", 32, 1)
        for stage in stages:
            stage = record(stage, "stage", {"id", "kind", "enabled", "timeout_s", "max_output_bytes", "parameters"})
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", text(stage["id"], "stage.id")):
                raise ValidationError("stage id must be a plain bounded identifier")
            if type(stage["kind"]) is not str or stage["kind"] not in KINDS or type(stage["enabled"]) is not bool:
                raise ValidationError("unknown stage kind or non-boolean enable flag")
            timeout = number(stage["timeout_s"], "stage.timeout_s", 0.1)
            if timeout > 300:
                raise ValidationError("stage timeout exceeds 300 seconds")
            limit = number(stage["max_output_bytes"], "stage.max_output_bytes", 1, True)
            if limit > 4 * 1024 * 1024:
                raise ValidationError("stage output limit exceeds 4 MiB")
            if not stage["enabled"]:
                record(stage["parameters"], "disabled parameters", set())
                continue
            required = {"portable_tests": {"suite"}, "runtime_probe": {"binary"},
                        "backend_self_test": {"binary", "bound_result", "device_id"}}.get(stage["kind"], set())
            if stage["kind"] in EXPERIMENTS:
                required = {"binary", "source_record"}
            params = record(stage["parameters"], "stage.parameters", required)
            if stage["kind"] == "portable_tests" and text(params["suite"], "suite") not in SUITES:
                raise ValidationError("portable test suite is not allowlisted")
            if stage["kind"] in {"runtime_probe", "backend_self_test"} | EXPERIMENTS:
                _artifact(params["binary"], "stage.binary")
                if root["mode"] != "native":
                    raise ValidationError("fixture campaigns cannot enable native artifact stages")
            if stage["kind"] in EXPERIMENTS:
                _artifact(params["source_record"], "stage.source_record")
            if stage["kind"] == "runtime_probe":
                if params["binary"]["sha256"] != manifest.worker_artifact_sha256:
                    raise ValidationError("runtime probe artifact differs from bound runtime worker")
            if stage["kind"] == "backend_self_test":
                if timeout > 120:
                    raise ValidationError("backend self-test timeout exceeds its 120-second contract")
                bound_result = load_llamacpp_bound_result(_artifact(params["bound_result"], "stage.bound_result"))
                if (bound_result.get("node_id") != host["node_id"]
                        or bound_result.get("probe_artifact_sha256") != manifest.worker_artifact_sha256):
                    raise ValidationError("backend bound result differs from configured host/runtime worker")
                device = text(params["device_id"], "device_id")
                if device not in devices or devices[device].node != host["node_id"]:
                    raise ValidationError("backend stage must target an explicitly bound local-host device")
        unique([stage["id"] for stage in stages], "stage IDs")
        binding = {"config_sha256": config.fingerprint, "model_manifest_sha256": model.manifest_sha256,
                   "adapter_capabilities_sha256": adapter.fingerprint, "runtime_manifest_sha256": manifest.fingerprint,
                   "runtime_manifest_provenance": manifest.provenance,
                   "runtime_identity_sha256": sorted(value.identity_sha256 for value in identities),
                   "worker_artifact_sha256": manifest.worker_artifact_sha256}
        encoded = json.dumps(root, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return cls(encoded, _digest(root), json.dumps(binding, sort_keys=True))


def load_campaign(path: str | Path) -> QualificationCampaign:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("campaign exceeds 2 MiB")
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError("invalid campaign JSON") from exc
    return QualificationCampaign.parse(data)


class _WindowsJob:
    """Contain the gated local child before it can run its stage or spawn workers."""
    def __init__(self, process):
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        kernel.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION consists of BASIC_LIMIT_INFORMATION,
        # six uint64 IO counters, then four SIZE_T memory limits/counters.
        class Basic(ctypes.Structure):
            _fields_ = [("user", ctypes.c_longlong), ("job", ctypes.c_longlong),
                        ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                        ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", wintypes.DWORD), ("schedule", wintypes.DWORD)]
        class Limits(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_ulonglong * 6), ("memory", ctypes.c_size_t * 4)]
        self.kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if (not self.handle or not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                or not kernel.AssignProcessToJobObject(self.handle, int(process._handle))):
            self.close()
            raise ValidationError("stage process containment could not be established")

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _run_process(argv: list[str], *, timeout_s: float, output_limit: int,
                 gated: bool = False, env: dict | None = None) -> dict:
    """Internal closed-stage runner; configuration never supplies argv directly."""
    process = subprocess.Popen(argv, cwd=REPO, shell=False, stdin=subprocess.PIPE if gated else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                               start_new_session=os.name != "nt" and gated)
    job = None
    buffers = [bytearray(), bytearray()]
    limit_hit = Event()
    lock = Lock()
    observed = 0

    def read(index, pipe):
        nonlocal observed
        try:
            while chunk := pipe.read(4096):
                with lock:
                    before = observed
                    observed += len(chunk)
                    keep = max(0, output_limit - before)
                    buffers[index].extend(chunk[:keep])
                    if observed > output_limit:
                        limit_hit.set()
        finally:
            pipe.close()

    readers = [Thread(target=read, args=(index, pipe), daemon=True)
               for index, pipe in enumerate((process.stdout, process.stderr))]
    started = time.monotonic()
    reason = None
    try:
        if gated and os.name == "nt":
            job = _WindowsJob(process)
        for reader in readers:
            reader.start()
        if gated:
            process.stdin.write(b"1")
            process.stdin.close()
        while process.poll() is None:
            if limit_hit.is_set():
                reason = "OUTPUT_LIMIT"
                break
            if time.monotonic() - started >= timeout_s:
                reason = "TIMEOUT"
                break
            try:
                process.wait(timeout=min(0.05, max(0.001, timeout_s - (time.monotonic() - started))))
            except subprocess.TimeoutExpired:
                pass
    finally:
        if job is not None:
            job.close()  # also stops descendants after an ordinary stage exit
        elif gated and os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=5)
        for pipe, reader in zip((process.stdout, process.stderr), readers):
            if reader.ident is None:
                pipe.close()
    if any(reader.is_alive() for reader in readers):
        raise ValidationError("stage output pipes remain owned after process cleanup")
    if limit_hit.is_set():
        reason = "OUTPUT_LIMIT"
    return {"returncode": process.returncode, "stdout": bytes(buffers[0]), "stderr": bytes(buffers[1]),
            "failure_reason": reason, "timed_out": reason == "TIMEOUT", "output_limit_exceeded": limit_hit.is_set(),
            "output_bytes_observed": observed, "child_exit_confirmed": process.poll() is not None,
            "process_tree_stop_requested": gated, "pipes_closed": True,
            "elapsed_s": time.monotonic() - started}


def _stage_argv(campaign: QualificationCampaign, stage: dict) -> list[str]:
    config = campaign.config
    artifacts, params = config["artifacts"], stage["parameters"]
    paths = {key: value["path"] for key, value in artifacts.items() if key != "runtime_identities"}
    if stage["kind"] == "validate_config":
        return ["validate-config", paths["config"]]
    if stage["kind"] == "host_probe":
        return ["probe"]
    if stage["kind"] == "validate_runtime_manifest":
        return ["validate-runtime-manifest", paths["config"], paths["model_manifest"],
                paths["adapter_capabilities"], paths["runtime_manifest"]]
    if stage["kind"] == "runtime_probe":
        return ["probe-llamacpp", params["binary"]["path"], "--trusted-local-binary", "--expected-sha256",
                params["binary"]["sha256"], "--timeout", str(min(stage["timeout_s"], 30))]
    if stage["kind"] == "backend_self_test":
        return ["self-test-llamacpp-backend", paths["config"], params["bound_result"]["path"],
                params["binary"]["path"], params["device_id"], "--trusted-local-binary", "--expected-sha256",
                params["binary"]["sha256"], "--timeout", str(stage["timeout_s"])]
    raise ValidationError("stage has no CLI translation")


def _experiment_argv(stage: dict) -> list[str]:
    if stage["kind"] not in EXPERIMENTS:
        raise ValidationError("not a native experiment stage")
    return [stage["parameters"]["binary"]["path"]] + (["--run"] if stage["kind"] == "native_projection_lab" else [])


def _execution_source(kind: str) -> str:
    return ("portable-fixture-tests" if kind == "portable_tests" else
            "trusted-local-experiment" if kind in EXPERIMENTS else "local-cli")


def _install_parent_death_guard(parent_pid: int) -> None:
    """Install containment in the new-session worker, never in a preexec_fn."""
    if os.name == "nt":
        return  # The campaign parent's kill-on-close Job owns this process.
    if sys.platform != "linux":
        raise ValidationError("campaign process containment supports only Windows and Linux")
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise ValidationError("campaign worker does not own its isolated process group")
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
    libc.prctl.restype = ctypes.c_int

    def stop_group(_signum=None, _frame=None):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, stop_group)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise ValidationError("campaign parent-death guard could not be established")
    # Parent death before/while installing the guard must also fail before the gate.
    if os.getppid() != parent_pid:
        stop_group()


def _inputs_unchanged(source: Path, campaign: QualificationCampaign, identity: dict) -> bool:
    try:
        current = load_campaign(source)
        return current.fingerprint == campaign.fingerprint and code_identity() == identity
    except (OSError, ValidationError):
        return False


def _worker(campaign_path: Path, stage_id: str, expected: str, expected_code: str,
            parent_pid: int | None = None) -> int:
    if parent_pid is not None:
        _install_parent_death_guard(parent_pid)
    if sys.stdin.buffer.read(1) != b"1":
        raise ValidationError("stage execution gate was not granted")
    campaign = load_campaign(campaign_path)
    if campaign.fingerprint != expected or code_identity()["source_sha256"] != expected_code:
        raise ValidationError("campaign or source identity changed before stage execution")
    stage = next((value for value in campaign.config["stages"] if value["id"] == stage_id), None)
    if stage is None or not stage["enabled"]:
        raise ValidationError("stage is absent or disabled")
    native_runs = []
    print(json.dumps({"stage_started_schema": "tensormeld/campaign-stage-started-v1",
                      "stage_id": stage_id, "kind": stage["kind"]}), flush=True)

    def native_runner(argv, timeout):
        expected_artifact = stage["parameters"]["binary"]
        path = _artifact(expected_artifact, "native invocation binary")
        if Path(argv[0]).resolve(strict=True) != path:
            raise ValidationError("native invocation differs from configured artifact")
        before_sha = _file_sha(path)
        if before_sha != expected_artifact["sha256"]:
            raise ValidationError("native binary identity changed before execution")
        # Leave space in the bounded outer protocol for base64 raw evidence,
        # CLI summaries and metadata. The stage's outer cap still applies.
        outcome = _run_process(list(argv), timeout_s=timeout, output_limit=max(1, stage["max_output_bytes"] // 4))
        after_sha = _file_sha(path)
        native_runs.append({"argv_sha256": _digest(list(argv)), "expected_artifact_sha256": expected_artifact["sha256"],
                            "artifact_sha256_before": before_sha, "artifact_sha256_after": after_sha,
                            "effective_environment_sha256": _digest(dict(os.environ)),
                            "returncode": outcome["returncode"], "stdout_sha256": hashlib.sha256(outcome["stdout"]).hexdigest(),
                            "stderr_sha256": hashlib.sha256(outcome["stderr"]).hexdigest(),
                            "timed_out": outcome["timed_out"], "output_limit_exceeded": outcome["output_limit_exceeded"],
                            "output_bytes_observed": outcome["output_bytes_observed"],
                            "child_exit_confirmed": outcome["child_exit_confirmed"],
                            "stdout_base64": base64.b64encode(outcome["stdout"]).decode("ascii"),
                            "stderr_base64": base64.b64encode(outcome["stderr"]).decode("ascii")})
        if before_sha != expected_artifact["sha256"] or after_sha != expected_artifact["sha256"]:
            raise ValidationError("native binary identity changed during execution")
        if outcome["failure_reason"]:
            raise ValidationError("native stage " + outcome["failure_reason"])
        return outcome["returncode"], outcome["stdout"], outcome["stderr"]

    output = io.StringIO()
    all_skipped = False
    if stage["kind"] in EXPERIMENTS:
        params = stage["parameters"]
        try:
            exit_code, stdout, _ = native_runner(_experiment_argv(stage), stage["timeout_s"])
            code = 0 if exit_code == 0 and stdout.strip() else 1
            try:
                detail = json.loads(stdout, object_pairs_hook=_no_duplicates)
            except (UnicodeError, json.JSONDecodeError, ValidationError):
                detail = {"non_json_output_sha256": hashlib.sha256(stdout).hexdigest()}
        except ValidationError:
            code, detail = 1, None
    elif stage["kind"] == "portable_tests":
        suite = unittest.TestLoader().discover(str(REPO / "tests"), pattern=SUITES[stage["parameters"]["suite"]])
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        code = 0 if result.wasSuccessful() and result.testsRun > 0 else 1
        detail = {"tests_executed": result.testsRun, "tests_skipped": len(result.skipped),
                  "tests_failed": len(result.failures) + len(result.errors)}
        all_skipped = result.testsRun > 0 and len(result.skipped) == result.testsRun
    else:
        from . import cli, llamacpp_probe, llamacpp_selftest
        # Reuse the same CLI/native proof paths, replacing only their unchecked
        # pipe collection with actual bounded subprocess capture. No injected
        # fixture runner is presented to the native self-test API.
        llamacpp_probe._default_runner = native_runner
        llamacpp_selftest._default_runner = native_runner
        with redirect_stdout(output):
            code = cli.main(_stage_argv(campaign, stage))
        try:
            detail = json.loads(output.getvalue())
        except json.JSONDecodeError:
            detail = None
    payload = {"stage_result_schema": "tensormeld/campaign-stage-result-v1", "stage_id": stage_id,
               "kind": stage["kind"], "exit_code": code, "detail": detail, "native_invocations": native_runs,
               "execution_source": _execution_source(stage["kind"]),
               "campaign_mode": campaign.config["mode"], "qualified": False, "executable": False}
    payload["all_tests_skipped"] = all_skipped
    payload["portable_test_output"] = output.getvalue() if stage["kind"] == "portable_tests" else None
    payload["effective_environment_sha256"] = _digest(dict(os.environ))
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return code


def run_qualification_campaign(path: str | Path) -> dict:
    source = Path(path).resolve(strict=True)
    campaign = load_campaign(source)
    if os.name != "nt" and sys.platform != "linux":
        raise ValidationError("campaign process containment supports only Windows and Linux")
    identity = code_identity()
    root = _private(Path(campaign.config["results_directory"])) / ("campaign-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    (root / "campaign-input.json").write_text(campaign.canonical_json + "\n", encoding="utf-8")
    results = []
    blocked = False
    env = {key: value for key, value in os.environ.items() if key.upper() in ENVIRONMENT_KEYS}
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "src"), str(REPO / "tests")))
    env["PYTHONNOUSERSITE"] = "1"
    environment_identity = {"boundary": "configured-child-environment", "sha256": _digest(env),
                            "variables": dict(sorted(env.items()))}
    for stage in campaign.config["stages"]:
        row = {"id": stage["id"], "kind": stage["kind"], "executed": False,
               "mode": campaign.config["mode"], "qualified": False, "executable": False}
        if not stage["enabled"]:
            row.update(outcome="SKIPPED", reason="DISABLED_BY_CONFIGURATION")
        elif blocked:
            row.update(outcome="NOT_RUN", reason="EARLIER_STAGE_FAILED")
        else:
            argv = [sys.executable, "-m", "tensormeld.qualification_campaign", "--worker-config", str(source),
                    "--stage", stage["id"], "--expected", campaign.fingerprint,
                    "--expected-code", identity["source_sha256"], "--parent-pid", str(os.getpid())]
            try:
                if not _inputs_unchanged(source, campaign, identity):
                    row.update(outcome="FAILED", reason="INPUT_IDENTITY_CHANGED_BEFORE_STAGE")
                    blocked = True
                    results.append(row)
                    continue
                captured = _run_process(argv, timeout_s=stage["timeout_s"], output_limit=stage["max_output_bytes"],
                                        gated=True, env=env)
                row.update({key: value for key, value in captured.items() if key not in {"stdout", "stderr"}})
                for name in ("stdout", "stderr"):
                    raw = captured[name]
                    (root / (stage["id"] + "." + name)).write_bytes(raw)
                    row[name + "_sha256"] = hashlib.sha256(raw).hexdigest()
                    row[name + "_retained_bytes"] = len(raw)
                lines = captured["stdout"].splitlines()
                parsed = []
                for line in lines:
                    try:
                        parsed.append(json.loads(line, object_pairs_hook=_no_duplicates))
                    except (UnicodeError, json.JSONDecodeError, ValidationError):
                        parsed.append(None)
                first = parsed[0] if parsed else None
                started_record = first if (isinstance(first, dict)
                    and first == {"stage_started_schema": "tensormeld/campaign-stage-started-v1",
                                  "stage_id": stage["id"], "kind": stage["kind"]}) else None
                payload = parsed[1] if len(parsed) == 2 and started_record is not None else None
                valid = (isinstance(payload, dict) and payload.get("stage_result_schema") == "tensormeld/campaign-stage-result-v1"
                         and payload.get("stage_id") == stage["id"] and payload.get("kind") == stage["kind"]
                         and payload.get("campaign_mode") == campaign.config["mode"]
                         and payload.get("execution_source") == _execution_source(stage["kind"])
                         and type(payload.get("exit_code")) is int and payload["exit_code"] == captured["returncode"]
                         and type(payload.get("all_tests_skipped")) is bool
                         and (not payload["all_tests_skipped"] or stage["kind"] == "portable_tests")
                         and isinstance(payload.get("native_invocations"), list)
                         and all(isinstance(value, dict) for value in payload["native_invocations"])
                         and payload.get("qualified") is False and payload.get("executable") is False)
                row.update(process_executed=True, executed=started_record is not None,
                           outcome="PASSED" if captured["returncode"] == 0
                           and captured["failure_reason"] is None and valid else "FAILED", stage_result=payload)
                if valid and row["outcome"] == "PASSED" and payload.get("all_tests_skipped") is True:
                    row.update(outcome="SKIPPED", reason="ALL_TESTS_SKIPPED")
                if valid:
                    invocations = payload.get("native_invocations", [])
                    row["timed_out"] = row["timed_out"] or any(value.get("timed_out") is True for value in invocations)
                    row["output_limit_exceeded"] = row["output_limit_exceeded"] or any(value.get("output_limit_exceeded") is True for value in invocations)
                    for index, invocation in enumerate(invocations):
                        for name in ("stdout", "stderr"):
                            raw = base64.b64decode(invocation.pop(name + "_base64"), validate=True)
                            if hashlib.sha256(raw).hexdigest() != invocation[name + "_sha256"]:
                                raise ValidationError("native output hash mismatch")
                            filename = f"{stage['id']}.native-{index}.{name}"
                            (root / filename).write_bytes(raw)
                            invocation[name + "_file"] = filename
                            invocation[name + "_retained_bytes"] = len(raw)
                after_unchanged = _inputs_unchanged(source, campaign, identity)
                row["input_identity_unchanged_after_stage"] = after_unchanged
                if not after_unchanged:
                    row.update(outcome="FAILED", reason="INPUT_IDENTITY_CHANGED_DURING_STAGE")
                if row["outcome"] == "FAILED":
                    native_reason = "TIMEOUT" if row["timed_out"] else "OUTPUT_LIMIT" if row["output_limit_exceeded"] else None
                    row.setdefault("reason", captured["failure_reason"] or native_reason or "STAGE_FAILED_OR_INVALID_OUTPUT")
                    blocked = True
            except (OSError, ValidationError, subprocess.SubprocessError, KeyError, ValueError):
                row.update(outcome="FAILED", reason="STAGE_PROCESS_OR_CLEANUP_FAILED")
                row["input_identity_unchanged_after_stage"] = _inputs_unchanged(source, campaign, identity)
                blocked = True
        results.append(row)
    report = {"result_schema": "tensormeld/qualification-campaign-result-v1",
              "campaign_sha256": campaign.fingerprint, "binding": campaign.binding, "code_identity": identity,
              "stage_environment": environment_identity,
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "results_directory": str(root),
              "stages": results, "status": "FAILED" if blocked else "CAMPAIGN_RECORDED",
              "native_qualified": False, "hardware_readiness_established": False,
              "inference_request_authorized": False, "executable": False}
    report["result_sha256"] = _digest(report)
    (root / "campaign.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Closed local qualification stages; private evidence only")
    parser.add_argument("campaign", type=Path, nargs="?")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage", help=argparse.SUPPRESS)
    parser.add_argument("--expected", help=argparse.SUPPRESS)
    parser.add_argument("--expected-code", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker_config:
            if args.parent_pid is None or args.parent_pid <= 0:
                raise ValidationError("worker parent identity is required")
            return _worker(args.worker_config, args.stage, args.expected, args.expected_code, args.parent_pid)
        if args.campaign is None:
            raise ValidationError("campaign configuration is required")
        report = run_qualification_campaign(args.campaign)
        print(json.dumps({"status": report["status"], "results_directory": report["results_directory"],
                          "result_sha256": report["result_sha256"], "native_qualified": False}))
        return 1 if report["status"] == "FAILED" else 0
    except (ValidationError, OSError) as exc:
        print("campaign error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
