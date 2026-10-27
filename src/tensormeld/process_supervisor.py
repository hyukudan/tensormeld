"""Supervision of engine processes that a deployment manifest names.

A supervised process is one approved engine binary started with a closed argv on one node.
TensorMeld builds the argv itself from a validated manifest; nothing on the far side
supplies an executable, an argument or a shell fragment.

Two launchers:

* ``LocalLauncher`` starts the process on this machine (no shell, minimal environment).
* ``SshLauncher`` starts it on another node through the operator's own SSH client
  (``BatchMode``, so it never prompts). The remote side runs a fixed POSIX ``sh`` wrapper
  that ``cd``s to the node work directory, starts the program under ``env -i`` with only
  the manifest's environment, prints ``TM-PID <pid>``, and terminates the program when the
  SSH channel's stdin closes (a dead-man switch: if the controller dies, the remote
  process stops) or on SIGTERM/SIGHUP. It prints ``TM-EXIT <code>`` when the program ends.

Readiness is a line on the process's stdout, in either engine form:

* ``READY {json}`` (pipeline roles), or
* ``{"event": "READY", ...}`` (engine-group server events; ``POISONED`` marks a failure).

Secrets (session tokens) are written to a file only the owner can read: locally with an
exclusive ``0600`` create inside a ``0700`` directory, remotely with ``umask 077`` and the
secret sent over the SSH channel's stdin, never on a command line.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
from typing import Any

from .schema import ValidationError

PROCESS_RESULT_SCHEMA = "tensormeld/supervised-process-v1"
MAX_LINE_CHARS = 64 * 1024
MAX_READY_JSON_BYTES = 64 * 1024
DEFAULT_LOG_LINES = 200
REMOTE_PATH = "/usr/local/bin:/usr/bin:/bin"
ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
# Names a manifest may never set: they change what code the dynamic loader or an
# interpreter runs, which would bypass the approved-binary check.
FORBIDDEN_ENV = frozenset({"LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES", "PYTHONPATH", "PYTHONSTARTUP",
                           "PYTHONHOME", "BASH_ENV", "ENV", "PATH"})


def check_env(env: dict[str, str], where: str) -> dict[str, str]:
    if not isinstance(env, dict) or len(env) > 32:
        raise ValidationError(f"{where}: expected an object of at most 32 variables")
    out = {}
    for key, value in env.items():
        if not isinstance(key, str) or not ENV_NAME.match(key) or key in FORBIDDEN_ENV:
            raise ValidationError(f"{where}: variable {key!r} is not allowed")
        if not isinstance(value, str) or len(value) > 1024 or any(c in value for c in "\0\n\r"):
            raise ValidationError(f"{where}.{key}: expected a single-line string of at most 1024 characters")
        out[key] = value
    return dict(sorted(out.items()))


@dataclass(frozen=True)
class LaunchSpec:
    """What to run on a node: ``prefix`` (an interpreter, tests only) + ``argv`` (binary first)."""

    process_id: str
    node: str
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    work_dir: str = "."
    prefix: tuple[str, ...] = ()

    @property
    def command(self) -> tuple[str, ...]:
        return (*self.prefix, *self.argv)


def _local_base_env() -> dict[str, str]:
    # Windows needs SYSTEMROOT for sockets and the CRT; PATH lets an interpreter prefix resolve DLLs.
    return {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if k in os.environ}


class LocalLauncher:
    kind = "local"

    def __init__(self, node: str):
        self.node = node

    def spawn(self, spec: LaunchSpec) -> subprocess.Popen:
        env = {**_local_base_env(), **spec.env}
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        Path(spec.work_dir).mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(list(spec.command), cwd=spec.work_dir, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, **kwargs)

    def put_secret(self, path: str, data: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)

    def remove(self, path: str) -> None:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
        try:
            Path(path).parent.rmdir()
        except OSError:
            pass

    def sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def run_bounded(self, command: tuple[str, ...], *, timeout_s: float, max_bytes: int) -> bytes:
        proc = subprocess.run(list(command), capture_output=True, timeout=timeout_s, env=_local_base_env(),
                              stdin=subprocess.DEVNULL, check=False)
        if proc.returncode != 0:
            raise ValidationError(f"{command[0]} exited {proc.returncode}: {proc.stderr[-400:].decode(errors='replace')}")
        if len(proc.stdout) > max_bytes:
            raise ValidationError(f"{command[0]}: output above {max_bytes} bytes")
        return proc.stdout

    def signal(self, child: subprocess.Popen, remote_pid: int | None, kill: bool) -> None:
        if child.poll() is None:
            child.kill() if kill else child.terminate()


def remote_wrapper(spec: LaunchSpec) -> str:
    """The fixed POSIX sh program the remote side runs for one process."""
    env = " ".join(shlex.quote(f"{k}={v}") for k, v in sorted(spec.env.items()))
    command = " ".join(shlex.quote(a) for a in spec.command)
    # A non-interactive sh gives an asynchronous list /dev/null as stdin unless it is redirected
    # explicitly, so the dead-man switch reads the channel through fd 3 (otherwise it fires at once).
    return "\n".join([
        f"mkdir -p {shlex.quote(spec.work_dir)} && cd {shlex.quote(spec.work_dir)} || exit 97",
        "exec 3<&0",
        f"env -i PATH={REMOTE_PATH} {env} {command} </dev/null 3<&- &",
        "child=$!",
        'echo "TM-PID $child"',
        "trap 'kill -TERM $child 2>/dev/null' TERM INT HUP",
        "(cat <&3 >/dev/null; kill -TERM $child 2>/dev/null) &",
        "exec 3<&-",
        "code=0",
        "while kill -0 $child 2>/dev/null; do wait $child; code=$?; done",
        'echo "TM-EXIT $code"',
        "exit $code",
    ])


class SshLauncher:
    kind = "ssh"

    def __init__(self, node: str, target: str, *, ssh_command: tuple[str, ...] = ("ssh",),
                 connect_timeout_s: int = 10):
        if not re.match(r"^[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?$", target or ""):
            raise ValidationError(f"ssh target {target!r}: expected [user@]host")
        self.node = node
        self.target = target
        self.ssh = (*ssh_command, "-T", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(connect_timeout_s)}",
                    target, "--")

    def _remote(self, script: str) -> list[str]:
        return [*self.ssh, f"sh -c {shlex.quote(script)}"]

    def spawn(self, spec: LaunchSpec) -> subprocess.Popen:
        return subprocess.Popen(self._remote(remote_wrapper(spec)), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=False, env=_local_base_env())

    def _run(self, script: str, *, data: bytes | None = None, timeout_s: float = 30.0,
             max_bytes: int = 1 << 20) -> bytes:
        proc = subprocess.run(self._remote(script), input=data if data is not None else b"", capture_output=True,
                              timeout=timeout_s, env=_local_base_env(), check=False)
        if proc.returncode != 0:
            raise ValidationError(f"{self.target}: remote step exited {proc.returncode}: "
                                  f"{proc.stderr[-400:].decode(errors='replace')}")
        if len(proc.stdout) > max_bytes:
            raise ValidationError(f"{self.target}: output above {max_bytes} bytes")
        return proc.stdout

    def put_secret(self, path: str, data: bytes) -> None:
        d = shlex.quote(path.rsplit("/", 1)[0] or "/")
        p = shlex.quote(path)
        self._run(f"umask 077 && mkdir -p {d} && chmod 700 {d} && set -C && cat > {p}", data=data)

    def remove(self, path: str) -> None:
        d = shlex.quote(path.rsplit("/", 1)[0] or "/")
        self._run(f"rm -f {shlex.quote(path)}; rmdir {d} 2>/dev/null; true")

    def sha256(self, path: str) -> str:
        out = self._run(f"sha256sum -- {shlex.quote(path)}").decode(errors="replace").split()
        if not out or not re.match(r"^[0-9a-f]{64}$", out[0]):
            raise ValidationError(f"{self.target}: no sha256 for {path}")
        return out[0]

    def run_bounded(self, command: tuple[str, ...], *, timeout_s: float, max_bytes: int) -> bytes:
        argv = " ".join(shlex.quote(a) for a in command)
        return self._run(f"env -i PATH={REMOTE_PATH} {argv} </dev/null", timeout_s=timeout_s, max_bytes=max_bytes)

    def signal(self, child: subprocess.Popen, remote_pid: int | None, kill: bool) -> None:
        if not kill:
            # Closing the channel's stdin fires the wrapper's dead-man switch (SIGTERM to the program).
            try:
                if child.stdin is not None:
                    child.stdin.close()
            except OSError:
                pass
            return
        if remote_pid is not None:
            try:
                self._run(f"kill -KILL {int(remote_pid)} 2>/dev/null; true", timeout_s=15)
            except (ValidationError, subprocess.TimeoutExpired, OSError):
                pass
        if child.poll() is None:
            child.kill()


def parse_event(line: str) -> tuple[str, dict[str, Any]] | None:
    """``READY {json}`` / ``POISONED ...`` / ``STOPPED`` or a JSON event line; None otherwise."""
    text_line = line.strip()
    if text_line.startswith("{"):
        try:
            value = json.loads(text_line)
        except (json.JSONDecodeError, RecursionError):
            return None
        if isinstance(value, dict) and value.get("event") in ("READY", "POISONED", "STOPPED"):
            return str(value["event"]), value
        return None
    for name in ("READY", "POISONED", "STOPPED"):
        if text_line == name or text_line.startswith(name + " "):
            rest = text_line[len(name):].strip()
            if name == "READY" and rest:
                if len(rest) > MAX_READY_JSON_BYTES:
                    return "POISONED", {"reason": "READY payload above bound"}
                try:
                    value = json.loads(rest)
                except (json.JSONDecodeError, RecursionError):
                    return "POISONED", {"reason": "READY payload is not JSON"}
                if not isinstance(value, dict):
                    return "POISONED", {"reason": "READY payload is not an object"}
                return name, value
            return name, ({"reason": rest} if rest else {})
    return None


class ManagedProcess:
    """One supervised engine process: start, readiness, liveness, ordered stop."""

    def __init__(self, spec: LaunchSpec, launcher: Any, *, log_lines: int = DEFAULT_LOG_LINES):
        self.spec = spec
        self.launcher = launcher
        self._out: deque[str] = deque(maxlen=log_lines)
        self._err: deque[str] = deque(maxlen=log_lines)
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._child: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []
        self.ready: dict[str, Any] | None = None
        self.poisoned: str | None = None
        self.remote_pid: int | None = None
        self.remote_exit: int | None = None
        self.state = "prepared"
        self.started_at: float | None = None
        self.ready_seconds: float | None = None

    # --- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        if self.state != "prepared":
            raise ValidationError(f"{self.spec.process_id}: cannot start from state {self.state}")
        try:
            self._child = self.launcher.spawn(self.spec)
        except OSError as exc:
            self.state = "failed"
            raise ValidationError(f"{self.spec.process_id}: failed to start: {exc}") from exc
        self.started_at = time.monotonic()
        self.state = "starting"
        for name, pipe, sink in (("out", self._child.stdout, self._out), ("err", self._child.stderr, self._err)):
            t = threading.Thread(target=self._drain, args=(pipe, sink, name == "out"),
                                 name=f"tm-{self.spec.process_id}-{name}", daemon=True)
            t.start()
            self._threads.append(t)

    def _drain(self, pipe, sink: deque, events: bool) -> None:
        try:
            for raw in iter(pipe.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if len(line) > MAX_LINE_CHARS:
                    line = line[:MAX_LINE_CHARS] + "..."
                with self._changed:
                    sink.append(line)
                    if events:
                        self._on_line(line)
                    self._changed.notify_all()
        except (OSError, ValueError):
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass
            with self._changed:
                self._changed.notify_all()

    def _on_line(self, line: str) -> None:
        if line.startswith("TM-PID ") and self.remote_pid is None:
            try:
                self.remote_pid = int(line.split()[1])
            except (IndexError, ValueError):
                pass
            return
        if line.startswith("TM-EXIT "):
            try:
                self.remote_exit = int(line.split()[1])
            except (IndexError, ValueError):
                self.remote_exit = -1
            return
        event = parse_event(line)
        if event is None:
            return
        name, value = event
        if name == "READY" and self.ready is None and self.poisoned is None:
            self.ready = value
            self.ready_seconds = round(time.monotonic() - (self.started_at or time.monotonic()), 3)
        elif name == "POISONED" and self.poisoned is None:
            self.poisoned = str(value.get("reason", "poisoned"))[:400]

    @property
    def exit_code(self) -> int | None:
        if self._child is None:
            return None
        code = self._child.poll()
        if code is None:
            return None
        return self.remote_exit if self.remote_exit is not None else code

    @property
    def alive(self) -> bool:
        return (self._child is not None and self._child.poll() is None and self.remote_exit is None
                and self.poisoned is None)

    def failure(self) -> str | None:
        """Why the process is not usable (None while it runs and has not reported a failure)."""
        if self._child is None:
            return "not started"
        if self.poisoned is not None:
            return f"poisoned: {self.poisoned}"
        if self.remote_exit is not None or self._child.poll() is not None:
            return f"exited with code {self.exit_code}"
        return None

    def wait_ready(self, timeout_s: float, *, require_line: bool = True) -> dict[str, Any]:
        """Block until READY (or, when ``require_line`` is false, until the process has run
        ``timeout_s`` without failing); raise on exit, POISONED or timeout."""
        deadline = time.monotonic() + timeout_s
        with self._changed:
            while True:
                if self.ready is not None:
                    self.state = "ready"
                    return self.ready
                problem = self.failure()
                if problem is not None:
                    self.state = "failed"
                    raise ValidationError(f"{self.spec.process_id}: {problem} before READY; "
                                          f"stderr tail={list(self._err)[-8:]!r}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not require_line:
                        self.state = "running"
                        return {}
                    self.state = "failed"
                    raise ValidationError(f"{self.spec.process_id}: no READY within {timeout_s:g}s; "
                                          f"stdout tail={list(self._out)[-8:]!r}")
                self._changed.wait(min(remaining, 0.25))

    def stop(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Graceful stop (SIGTERM / dead-man switch), then kill after ``timeout_s``."""
        child = self._child
        killed = False
        if child is not None and child.poll() is None:
            self.launcher.signal(child, self.remote_pid, kill=False)
            try:
                child.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                killed = True
                self.launcher.signal(child, self.remote_pid, kill=True)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for t in self._threads:
            t.join(timeout=2)
        if child is not None and child.stdin is not None:
            try:
                child.stdin.close()
            except OSError:
                pass
        self.state = "stopped"
        return self.snapshot() | {"killed": killed}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": PROCESS_RESULT_SCHEMA,
                "process": self.spec.process_id,
                "node": self.spec.node,
                "launcher": self.launcher.kind,
                "state": self.state,
                "alive": self.alive,
                "pid": None if self._child is None else self._child.pid,
                "remote_pid": self.remote_pid,
                "exit_code": self.exit_code,
                "ready": self.ready,
                "ready_seconds": self.ready_seconds,
                "poisoned": self.poisoned,
                "stdout_tail": list(self._out)[-12:],
                "stderr_tail": list(self._err)[-12:],
            }


__all__ = [
    "LaunchSpec",
    "LocalLauncher",
    "ManagedProcess",
    "SshLauncher",
    "check_env",
    "parse_event",
    "remote_wrapper",
]
