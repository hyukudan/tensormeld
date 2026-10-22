"""Run a two-stage layer pipeline from its manifests: launch, health, HTTP route, switchover.

The controller owns the processes a ``Deployment`` names and nothing else:

1. **Pre-flight**: the approved binary's sha256 is checked on every node before anything runs.
2. **Secrets**: one random session token (32 bytes, hex) per deployment, written to each
   stage process's token file (``0600``); an HTTP key per fallback node when the fallback
   listens on a non-loopback address. Secrets never appear in a manifest, argv or log.
3. **Start order**: every TAIL rank starts; rank 0 must print ``READY`` (with the manifest's
   role and layer range when it reports them) and the other ranks must stay alive; then the
   FRONT starts, connects to the TAIL, prints ``READY`` and must answer ``GET /health`` with
   200 (and ``stage: connected`` when it reports a stage). A ``standby`` fallback starts too.
4. **Route**: clients talk to the FRONT's OpenAI-compatible HTTP API. With a router
   configured, TensorMeld's least-loaded router (``replica_router``) is the stable front end
   and proxies HTTP to the FRONT; it never sees activations or tokens other than the HTTP
   bytes it relays. Without a router the FRONT URL is reported and clients use it directly.
5. **Watch**: process liveness and FRONT health every ``health_interval_s``.
6. **Switchover to the fallback** (the device set of the plan without the preemptible
   device): on ``retire()`` (a planned retirement of the FRONT device) the router holds new
   requests, the FRONT's in-flight requests finish (up to ``drain_s``), the FRONT drains,
   the FRONT exports the request state (its layers plus the TAIL's, which the engines move
   over their own wire), FRONT and TAIL stop, the fallback starts (cold) or is already up
   (standby), the state is imported there and verified by sha256, and the router routes to
   the fallback. If a stage process dies instead, the state of its layers is gone: the
   controller either stops (``on_failure="stop"``) or starts the fallback empty
   (``on_failure="fallback"``) and reports ``state_lost: true``. It never claims a migration
   that did not happen.

State handoff HTTP routes (proposed to the engine as ``tensormeld/state-handoff-v1``):
``POST /admin/drain``; ``POST /admin/export-state`` -> ``application/octet-stream`` with
``X-State-Format`` and ``X-State-Sha256``; ``POST /admin/import-state`` (same headers) ->
``{"imported": true, "sha256": ...}``.
"""
from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Any

from .engine_describe import EngineDescribe, parse_engine_describe
from .pipeline_deploy import Deployment, Inventory, validate_manifest
from .process_supervisor import LaunchSpec, LocalLauncher, ManagedProcess, SshLauncher
from .replica_router import ReplicaRouter, ReplicaSpec, RouterConfig, _read_key
from .schema import ValidationError, _no_duplicates

CONTROLLER_STATUS_SCHEMA = "tensormeld/pipeline-controller-status-v1"
SWITCHOVER_SCHEMA = "tensormeld/pipeline-switchover-v1"
MAX_DESCRIBE_BYTES = 4 * 1024 * 1024


def launchers_for(inv: Inventory, *, ssh_command: tuple[str, ...] = ("ssh",)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in inv.nodes.values():
        out[node.id] = (LocalLauncher(node.id) if node.launcher == "local"
                        else SshLauncher(node.id, node.ssh_target or "", ssh_command=ssh_command))
    return out


def resolve_describes(inv: Inventory, launchers: dict[str, Any], *, probe: bool,
                      engine_nodes: dict[str, str] | None = None) -> dict[str, EngineDescribe | None]:
    """Recorded descriptions are parsed; ``probe`` engines run ``--describe-json`` (bounded) on
    the node that will run them (``engine_nodes``), checked against the approved sha256."""
    from .pipeline_deploy import load_recorded_describe

    out: dict[str, EngineDescribe | None] = {}
    for engine in inv.engines.values():
        if engine.describe_source == "file":
            out[engine.id] = load_recorded_describe(engine)
        elif engine.describe_source == "probe" and probe:
            node = (engine_nodes or {}).get(engine.id)
            if node is None:  # not bound to any stage: nothing will run it
                out[engine.id] = None
                continue
            launcher = launchers[node]
            actual = launcher.sha256(engine.binary)
            if actual != engine.sha256:
                raise ValidationError(f"engine {engine.id}: {engine.binary} on {node} is not the approved artifact")
            raw = launcher.run_bounded((*engine.launch_prefix, engine.binary, "--describe-json"),
                                       timeout_s=inv.timeouts["describe_s"], max_bytes=MAX_DESCRIBE_BYTES)
            try:
                data = json.loads(raw, object_pairs_hook=_no_duplicates)
            except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValidationError(f"engine {engine.id}: describe is not JSON: {exc}") from exc
            out[engine.id] = parse_engine_describe(data, expected_executable_sha256=engine.sha256)
        else:
            out[engine.id] = None
    return out


def engine_nodes(inv: Inventory, hw) -> dict[str, str]:
    """The node each engine is probed on: the node of its binding's first rank/device."""
    out = {}
    for b in (inv.front, inv.tail, inv.fallback):
        if b is None:
            continue
        devices = list(b.ranks) or [d for g in hw.groups if g.id == b.set for d in g.devices] or [b.set]
        out.setdefault(b.engine, hw.device(devices[0]).node)
    return out


def _http(method: str, address: str, path: str, *, body: bytes | None = None, headers: dict | None = None,
          timeout_s: float = 5.0, max_bytes: int = 1 << 20) -> tuple[int, dict[str, str], bytes]:
    host, port = address.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=timeout_s)
    try:
        h = {"Connection": "close", **(headers or {})}
        if body is not None or method == "POST":
            h["Content-Length"] = str(len(body or b""))
        conn.request(method, path, body=body, headers=h)
        resp = conn.getresponse()
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValidationError(f"{method} {path}: response above {max_bytes} bytes")
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    finally:
        conn.close()


class PipelineController:
    def __init__(self, deployment: Deployment, launchers: dict[str, Any], *, on_failure: str = "stop",
                 retire_file: Path | None = None, log: Any = None):
        if on_failure not in ("stop", "fallback"):
            raise ValidationError("on_failure: expected stop or fallback")
        for m in deployment.manifests:
            validate_manifest(m)
            if m["node"]["id"] not in launchers:
                raise ValidationError(f"no launcher for node {m['node']['id']}")
        self.d = deployment
        self.launchers = launchers
        self.on_failure = on_failure
        self.retire_file = retire_file
        self.log = log
        self.t = deployment.timeouts
        self.state = "prepared"
        self.reason: str | None = None
        self.events: list[dict[str, Any]] = []
        self.procs: dict[str, ManagedProcess] = {}
        self.router: ReplicaRouter | None = None
        self._secrets: list[tuple[str, str]] = []  # (node, path)
        self._fallback_key: str | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._switch_thread: threading.Thread | None = None
        self.switchover: dict[str, Any] | None = None
        self.front_failures = 0

    # --- helpers -----------------------------------------------------------------
    def _event(self, what: str, **info: Any) -> None:
        record = {"t": round(time.time(), 3), "event": what, **info}
        self.events.append(record)
        if len(self.events) > 500:
            del self.events[:100]
        if self.log is not None:
            print(f"tensormeld pipeline: {what} {json.dumps(info, sort_keys=True) if info else ''}".rstrip(),
                  file=self.log, flush=True)

    def _spec(self, m: dict) -> LaunchSpec:
        return LaunchSpec(m["process"]["id"], m["node"]["id"], tuple(m["argv"]), dict(m["env"]),
                          m["node"]["work_dir"], tuple(m["launch_prefix"]))

    def _put_secret(self, m: dict, path: str, value: str) -> None:
        self.launchers[m["node"]["id"]].put_secret(path, value.encode("ascii"))
        self._secrets.append((m["node"]["id"], path))

    def _launch(self, manifests: tuple[dict, ...]) -> list[ManagedProcess]:
        started = []
        for m in manifests:
            p = ManagedProcess(self._spec(m), self.launchers[m["node"]["id"]])
            self.procs[m["process"]["id"]] = p
            p.start()
            started.append(p)
            self._event("started", process=m["process"]["id"], node=m["node"]["id"])
        return started

    def _await_ready(self, manifests: tuple[dict, ...]) -> None:
        # READY lines first (the processes that report them), then the "alive" ranks.
        for m in sorted(manifests, key=lambda m: m["readiness"]["line"] != "READY"):
            p = self.procs[m["process"]["id"]]
            if m["readiness"]["line"] == "READY":
                ready = p.wait_ready(m["readiness"]["timeout_s"])
                if "layers" in ready and list(ready["layers"]) != m["layers"]:
                    raise ValidationError(f"{m['process']['id']}: READY layers {ready['layers']} != manifest {m['layers']}")
                if "role" in ready and ready["role"] not in (m["process"]["role"], m["process"]["position"]):
                    raise ValidationError(f"{m['process']['id']}: READY role {ready['role']!r} != {m['process']['role']}")
                self._event("ready", process=m["process"]["id"], seconds=p.ready_seconds)
            else:
                problem = p.failure()
                if problem is not None:
                    raise ValidationError(f"{m['process']['id']}: {problem}")

    def _http_address(self, manifests: tuple[dict, ...]) -> str:
        return manifests[0]["endpoints"]["http"]

    def _auth(self, which: str) -> dict[str, str]:
        if which == "fallback" and self._fallback_key:
            return {"Authorization": f"Bearer {self._fallback_key}"}
        return {}

    def health(self, which: str) -> tuple[bool, str]:
        manifests = self.d.front if which == "front" else self.d.fallback
        try:
            status, _, body = _http("GET", self._http_address(manifests), "/health",
                                    timeout_s=self.t["health_timeout_s"], max_bytes=64 * 1024)
        except (OSError, http.client.HTTPException, ValidationError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if status != 200:
            return False, f"health HTTP {status}"
        try:
            value = json.loads(body or b"{}")
        except json.JSONDecodeError:
            value = {}
        if isinstance(value, dict) and value.get("stage") not in (None, "connected"):
            return False, f"stage {value.get('stage')}"
        return True, ""

    def _await_health(self, which: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        manifests = self.d.front if which == "front" else self.d.fallback
        last = ""
        while time.monotonic() < deadline:
            for m in manifests:
                problem = self.procs[m["process"]["id"]].failure()
                if problem is not None:
                    raise ValidationError(f"{m['process']['id']}: {problem}")
            ok, last = self.health(which)
            if ok:
                self._event("healthy", stage=which)
                return
            time.sleep(min(0.2, self.t["health_interval_s"]))
        raise ValidationError(f"{which}: /health not ready within {timeout_s:g}s ({last})")

    # --- start -------------------------------------------------------------------
    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.state != "prepared":
                raise ValidationError(f"cannot start from state {self.state}")
            self.state = "starting"
        try:
            for m in self.d.manifests:
                launcher = self.launchers[m["node"]["id"]]
                actual = launcher.sha256(m["engine"]["binary"])
                if actual != m["engine"]["sha256"]:
                    raise ValidationError(f"{m['process']['id']}: {m['engine']['binary']} on {m['node']['id']} "
                                          "is not the approved artifact")
            self._event("preflight", binaries=len(self.d.manifests))
            token = secrets.token_hex(32)
            for m in (*self.d.tail, *self.d.front):
                self._put_secret(m, m["secrets"]["token_file"]["file"], token)
            if any("api_key_file" in m["secrets"] for m in self.d.fallback):
                self._fallback_key = secrets.token_hex(32)
                for m in self.d.fallback:
                    self._put_secret(m, m["secrets"]["api_key_file"]["file"], self._fallback_key)
            del token
            self._launch(self.d.tail)
            self._await_ready(self.d.tail)
            self._launch(self.d.front)
            self._await_ready(self.d.front)
            self._await_health("front", self.t["front_ready_s"])
            if self.d.fallback and self.d.fallback_start == "standby":
                self._launch(self.d.fallback)
                self._await_ready(self.d.fallback)
                self._await_health("fallback", self.t["fallback_ready_s"])
            self._start_router()
        except BaseException as exc:
            self._event("start-failed", reason=str(exc)[:400])
            self.reason = str(exc)
            self._teardown()
            self.state = "failed"
            raise
        self.state = "serving"
        self._event("serving", url=self.url())
        self._monitor = threading.Thread(target=self._watch, name="tm-pipeline-watch", daemon=True)
        self._monitor.start()
        return self.status()

    def _start_router(self) -> None:
        r = self.d.router
        if r is None:
            return
        front = self.d.front[0]["endpoints"]["http"].rsplit(":", 1)
        replicas = [ReplicaSpec("front", self.d.front[0]["node"]["id"], front[0], int(front[1]), r["max_inflight"], None)]
        if self.d.fallback:
            fb = self.d.fallback[0]["endpoints"]["http"].rsplit(":", 1)
            replicas.append(ReplicaSpec("fallback", self.d.fallback[0]["node"]["id"], fb[0], int(fb[1]),
                                        r["max_inflight"], self._fallback_key))
        config = RouterConfig(
            listen_address=r["address"], listen_port=r["port"],
            api_key=_read_key(r["api_key_file"], "router.api_key_file") if r.get("api_key_file") else None,
            replicas=tuple(replicas), health_interval_s=self.t["health_interval_s"],
            health_timeout_s=self.t["health_timeout_s"], failures_to_down=int(self.t["failures_to_down"]),
            connect_timeout_s=5.0, read_timeout_s=900.0, queue_timeout_s=r["queue_timeout_s"], max_waiting=64,
            max_connections=128, max_request_bytes=8 * 1024 * 1024)
        self.router = ReplicaRouter(config)
        if self.d.fallback:
            self.router.pool.set_admitting("fallback", False)
        self.router.server.status_extra = lambda: {"deployment": self.status()}
        self.router.server.admin_routes["/admin/pipeline/retire"] = self._retire_route
        self.router.start()
        host, port = self.router.address
        self._event("router", listen=f"{host}:{port}")

    def url(self) -> str:
        if self.router is not None:
            host, port = self.router.address
            return f"http://{host}:{port}"
        active = self.d.fallback if self.state == "fallback" else self.d.front
        return f"http://{active[0]['endpoints']['http']}"

    # --- watch ---------------------------------------------------------------------
    def _watch(self) -> None:
        while not self._stop.wait(self.t["health_interval_s"]):
            with self._lock:
                state = self.state
            if state == "serving" and self.retire_file is not None and self.retire_file.exists():
                self._event("retire-file", path=str(self.retire_file))
                self.retire(reason=f"retire file {self.retire_file.name}")
                continue
            if state == "serving":
                failed = [(pid, p.failure()) for pid, p in self.procs.items()
                          if pid.startswith(("front", "tail")) and p.failure() is not None]
                if not failed:
                    ok, why = self.health("front")
                    self.front_failures = 0 if ok else self.front_failures + 1
                    if self.front_failures >= self.t["failures_to_down"]:
                        failed = [("front", f"unhealthy: {why}")]
                if failed:
                    self._on_stage_failure(failed)
            elif state == "fallback":
                failed = [(pid, p.failure()) for pid, p in self.procs.items()
                          if pid.startswith("fallback") and p.failure() is not None]
                if failed:
                    self._fail(f"fallback process failed: {failed}")

    def _on_stage_failure(self, failed: list[tuple[str, str | None]]) -> None:
        reason = "; ".join(f"{pid}: {why}" for pid, why in failed)
        self._event("stage-failed", reason=reason)
        if self.on_failure == "fallback" and self.d.fallback:
            self._switch(reason=f"stage failure: {reason}", export=False)
        else:
            self._fail(f"stage failure: {reason}")

    def _fail(self, reason: str) -> None:
        with self._lock:
            if self.state in ("failed", "stopped"):
                return
            self.state = "failed"
            self.reason = reason
        self._event("failed", reason=reason)
        if self.router is not None:
            self.router.pool.hold(False)
            for spec in self.router.pool.specs():
                self.router.pool.set_admitting(spec.id, False)
        for p in self.procs.values():
            if p.alive:
                p.stop(self.t["stop_s"])

    # --- switchover ----------------------------------------------------------------
    def _retire_route(self) -> tuple[int, dict]:
        with self._lock:
            if self.state != "serving":
                return 409, {"error": {"message": f"deployment is {self.state}", "type": "conflict"}}
            if self._switch_thread is not None and self._switch_thread.is_alive():
                return 409, {"error": {"message": "switchover already running", "type": "conflict"}}
            self._switch_thread = threading.Thread(target=self.retire, kwargs={"reason": "admin retire"},
                                                   name="tm-pipeline-switch", daemon=True)
            self._switch_thread.start()
        return 202, {"switchover": "started", "status": "/api/status"}

    def retire(self, reason: str = "retire") -> dict[str, Any]:
        """Planned retirement of the FRONT device: move to the fallback with the request state."""
        return self._switch(reason=reason, export=True)

    def _switch(self, *, reason: str, export: bool) -> dict[str, Any]:
        with self._lock:
            if self.state != "serving":
                raise ValidationError(f"switchover needs a serving deployment (state {self.state})")
            if not self.d.fallback:
                raise ValidationError("the deployment has no fallback")
            self.state = "switching"
        t0 = time.monotonic()
        rec: dict[str, Any] = {"schema": SWITCHOVER_SCHEMA, "reason": reason, "steps": [],
                               "state_handoff": False, "state_lost": True}

        def step(name: str, **info: Any) -> None:
            rec["steps"].append({"step": name, "at_s": round(time.monotonic() - t0, 3), **info})
            self._event(f"switch:{name}", **info)

        try:
            pool = self.router.pool if self.router is not None else None
            if pool is not None:
                pool.hold(True)
                pool.set_admitting("front", False)
                drained = pool.wait_idle("front", self.t["drain_s"])
                step("router-drain", drained=drained, inflight=pool.inflight("front"))
            blob = None
            front_alive = all(self.procs[m["process"]["id"]].alive for m in self.d.front)
            tail_alive = all(self.procs[m["process"]["id"]].alive for m in self.d.tail)
            if export and self.d.state_handoff and front_alive and tail_alive:
                address = self._http_address(self.d.front)
                status, _, body = _http("POST", address, "/admin/drain", timeout_s=30)
                step("front-drain", http=status)
                if status != 200:
                    raise ValidationError(f"FRONT /admin/drain answered {status}")
                status, headers, blob = _http("POST", address, "/admin/export-state", timeout_s=self.t["export_s"],
                                              max_bytes=int(self.t["max_state_bytes"]))
                if status != 200:
                    raise ValidationError(f"FRONT /admin/export-state answered {status}: {blob[:200]!r}")
                digest = hashlib.sha256(blob).hexdigest()
                if headers.get("x-state-sha256", "").lower() != digest:
                    raise ValidationError("exported state does not match its X-State-Sha256")
                rec["state"] = {"bytes": len(blob), "sha256": digest, "format": headers.get("x-state-format")}
                step("export", bytes=len(blob), sha256=digest[:16])
            elif export and self.d.state_handoff:
                step("export-skipped", reason="a stage process is not alive")
            for m in (*self.d.front, *self.d.tail):
                p = self.procs[m["process"]["id"]]
                p.stop(self.t["stop_s"])
            step("stages-stopped")
            if self.d.fallback_start == "cold":
                self._launch(self.d.fallback)
                self._await_ready(self.d.fallback)
            self._await_health("fallback", self.t["fallback_ready_s"])
            step("fallback-ready", start=self.d.fallback_start)
            if blob is not None:
                status, _, body = _http("POST", self._http_address(self.d.fallback), "/admin/import-state", body=blob,
                                        headers={"Content-Type": "application/octet-stream",
                                                 "X-State-Format": rec["state"]["format"] or "",
                                                 "X-State-Sha256": rec["state"]["sha256"], **self._auth("fallback")},
                                        timeout_s=self.t["import_s"])
                answer = json.loads(body or b"{}") if status == 200 else {}
                if status != 200 or not answer.get("imported") or answer.get("sha256") != rec["state"]["sha256"]:
                    raise ValidationError(f"fallback /admin/import-state answered {status}: {body[:200]!r}")
                rec["state_handoff"] = True
                rec["state_lost"] = False
                step("import", sessions=answer.get("sessions"))
            if pool is not None:
                pool.set_admitting("fallback", True)
                self.router.monitor.check_once()
                pool.hold(False)
                step("routed", replica="fallback")
            rec["seconds"] = round(time.monotonic() - t0, 3)
            rec["status"] = "OK"
            with self._lock:
                self.state = "fallback"
            self._event("fallback", state_lost=rec["state_lost"], seconds=rec["seconds"])
        except BaseException as exc:
            rec["status"] = "FAILED"
            rec["error"] = str(exc)[:400]
            rec["seconds"] = round(time.monotonic() - t0, 3)
            self.switchover = rec
            with self._lock:
                self.state = "serving-failed-switch"
            self._fail(f"switchover failed: {exc}")
            return rec
        self.switchover = rec
        return rec

    # --- stop ----------------------------------------------------------------------
    def _teardown(self) -> list[dict]:
        out = []
        if self.router is not None:
            try:
                self.router.stop()
            except Exception:
                pass
        # FRONT first (it holds the HTTP clients), then the stages behind it, then the fallback.
        order = sorted(self.procs.values(),
                       key=lambda p: {"front": 0, "tail": 1, "fallback": 2}.get(p.spec.process_id.split("-")[0], 3))
        for p in order:
            out.append(p.stop(self.t["stop_s"]))
        for node, path in self._secrets:
            try:
                self.launchers[node].remove(path)
            except Exception:
                pass
        self._secrets.clear()
        return out

    def stop(self) -> dict[str, Any]:
        if self.state == "stopped":
            return {**self.status(), "final_state": "stopped", "processes": []}
        self._stop.set()
        if self._switch_thread is not None:
            self._switch_thread.join(timeout=self.t["fallback_ready_s"] + 60)
        if self._monitor is not None:
            self._monitor.join(timeout=self.t["health_interval_s"] + 5)
        processes = self._teardown()
        final = self.state
        self.state = "stopped"
        self._event("stopped")
        return {**self.status(), "final_state": final,
                "processes": [{k: p[k] for k in ("process", "node", "exit_code", "killed", "ready_seconds")}
                              for p in processes]}

    def status(self) -> dict[str, Any]:
        return {
            "schema": CONTROLLER_STATUS_SCHEMA,
            "deployment": self.d.id,
            "candidate": self.d.candidate,
            "state": self.state,
            "reason": self.reason,
            "url": self.url() if self.state in ("serving", "fallback", "switching") else None,
            "processes": {pid: {"alive": p.alive, "ready": p.ready is not None, "exit_code": p.exit_code,
                                "node": p.spec.node} for pid, p in self.procs.items()},
            "switchover": self.switchover,
            "events": self.events[-20:],
            "qualified": False,
        }


__all__ = [
    "CONTROLLER_STATUS_SCHEMA",
    "PipelineController",
    "engine_nodes",
    "launchers_for",
    "resolve_describes",
]
