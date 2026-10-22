"""Process manifests for a layer-pipeline placement (``tensormeld/deployment-manifest-v1``).

``plan-hetero`` ranks placements; it never executes one. This module turns one
``hetero-pipeline`` candidate (and its preemptible-device ``fallback``) into the exact set
of processes that would run it, each bound to one approved engine binary, one engine
instance and, for a device group, one of that engine's group modes:

* **inputs**: the plan, the model parts and hardware descriptors it was computed from
  (fingerprints must match), and an operator inventory (``tensormeld/pipeline-inventory-v1``)
  that names, per plan device set, the engine that runs it, where the nodes are, which
  ports to use and how to start processes there;
* **binding**: every stage set maps to exactly one inventory binding whose ranks cover
  exactly the set's devices; the engine's description (``tensormeld/engine-describe-v1``,
  from a recorded file or a probe of the approved binary) must declare the role, the
  group mode (world = number of devices, the transport) and every capability the stage's
  parts require. Capabilities are those of *this* engine (its own plus its group mode's),
  never a union over different binaries;
* **output**: one manifest per process: role, layer range, exact part ids, peers and
  ports, session-token file, binary path and approved sha256, node, devices, environment
  and the closed argv. A manifest carries its own sha256 and is re-derived on validation,
  so the argv cannot drift from the fields it is built from.

A manifest never contains a secret: tokens are generated at launch and delivered as files
(see ``process_supervisor``). Manifests are deployment intent, not execution
qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path
import re
from typing import Any

from .engine_describe import EngineDescribe, parse_engine_describe
from .hetero_hardware import Hardware
from .hetero_model import ModelParts
from .process_supervisor import check_env
from .schema import ValidationError, _no_duplicates, items, number, record, text, unique

INVENTORY_SCHEMA = "tensormeld/pipeline-inventory-v1"
MANIFEST_SCHEMA = "tensormeld/deployment-manifest-v1"
DEPLOYMENT_SCHEMA = "tensormeld/pipeline-deployment-v1"
PLAN_SCHEMA = "tensormeld/hetero-plan-v1"
MAX_INVENTORY_BYTES = 256 * 1024
MAX_PLAN_BYTES = 16 * 1024 * 1024

# Closed argv forms (the engine contract's command lines). A form is code, not configuration.
FORMS = {
    "pipeline-tail-v1": {"role": "pipeline-stage", "positions": ("tail",), "group": True},
    "pipeline-front-v1": {"role": "pipeline-stage", "positions": ("front",), "group": False},
    "group-serve-v1": {"role": "whole", "positions": ("fallback",), "group": True},
}
FRONT_BACKENDS = ("cuda", "hip", "host")
TIMEOUT_DEFAULTS = {
    "describe_s": 60.0, "tail_ready_s": 900.0, "front_ready_s": 900.0, "fallback_ready_s": 900.0,
    "health_interval_s": 2.0, "health_timeout_s": 2.0, "failures_to_down": 3, "stop_s": 30.0,
    "drain_s": 120.0, "export_s": 300.0, "import_s": 300.0, "max_state_bytes": 1 << 30,
}
TIMEOUT_BOUNDS = {
    "describe_s": (1, 600), "tail_ready_s": (1, 3600), "front_ready_s": (1, 3600), "fallback_ready_s": (1, 3600),
    "health_interval_s": (0.05, 600), "health_timeout_s": (0.05, 60), "failures_to_down": (1, 100),
    "stop_s": (0.5, 600), "drain_s": (0, 3600), "export_s": (1, 3600), "import_s": (1, 3600),
    "max_state_bytes": (1, 1 << 36),
}
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _id(value: Any, where: str) -> str:
    value = text(value, where)
    if not ID.match(value):
        raise ValidationError(f"{where}: expected an identifier [A-Za-z0-9._-], at most 64 characters")
    return value


def _sha(value: Any, where: str) -> str:
    value = text(value, where).lower()
    if not re.match(r"^[0-9a-f]{64}$", value):
        raise ValidationError(f"{where}: expected 64 hex characters")
    return value


def _ip(value: Any, where: str) -> str:
    value = text(value, where)
    try:
        ip = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValidationError(f"{where}: must be a numeric IP literal") from exc
    if ip.is_unspecified or ip.is_multicast or ip.is_link_local or not (ip.is_loopback or ip.is_private):
        raise ValidationError(f"{where}: must be a loopback or private address")
    return str(ip)


def _port(value: Any, where: str) -> int:
    port = int(number(value, where, 1024, True))
    if port > 65535:
        raise ValidationError(f"{where}: expected 1024..65535")
    return port


def _path(value: Any, where: str) -> str:
    value = text(value, where)
    if any(c in value for c in "\0\n\r"):
        raise ValidationError(f"{where}: invalid path")
    return value


def _load_json(path: Path, limit: int, what: str) -> Any:
    with Path(path).open("rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit:
        raise ValidationError(f"{what} exceeds {limit} bytes")
    try:
        return json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"{what}: invalid JSON: {exc}") from exc


# --- inventory -----------------------------------------------------------------------

@dataclass(frozen=True)
class InvNode:
    id: str
    address: str
    work_dir: str
    launcher: str  # "local" | "ssh"
    ssh_target: str | None


@dataclass(frozen=True)
class InvEngine:
    id: str
    binary: str
    sha256: str
    backends: tuple[str, ...]
    model_path: str
    argv_form: str
    env: dict[str, str]
    describe_source: str  # "file" | "probe" | "unchecked"
    describe_file: str | None
    unchecked_reason: str | None
    launch_prefix: tuple[str, ...]


@dataclass(frozen=True)
class InvBinding:
    set: str
    engine: str
    position: str  # "front" | "tail" | "fallback"
    ranks: tuple[str, ...]  # device id per rank (rank i runs on ranks[i])
    group_mode: str | None
    transport: str | None
    backend: str | None
    stage_port: int | None
    http_port: int | None
    http_address: str | None
    max_positions: int | None
    start: str  # fallback only: "cold" | "standby"


@dataclass(frozen=True)
class Inventory:
    nodes: dict[str, InvNode]
    engines: dict[str, InvEngine]
    front: InvBinding
    tail: InvBinding
    fallback: InvBinding | None
    router: dict[str, Any] | None
    timeouts: dict[str, float]
    fingerprint: str


def _binding(raw: Any, where: str, position: str) -> InvBinding:
    keys = {"set", "engine"}
    opt = {"ranks", "group_mode", "transport", "backend", "stage_port", "http_port", "http_address",
           "max_positions", "start"}
    r = record(raw, where, keys, opt)
    ranks = tuple(_id(d, f"{where}.ranks[]") for d in items(r.get("ranks", []), f"{where}.ranks", 16, 0))
    unique(list(ranks), f"{where}.ranks")
    start = r.get("start", "cold")
    if start not in ("cold", "standby") or (position != "fallback" and "start" in r):
        raise ValidationError(f"{where}.start: only a fallback binding has start cold|standby")
    if position == "tail" and "stage_port" not in r:
        raise ValidationError(f"{where}.stage_port: required for the tail stage")
    if position in ("front", "fallback") and "http_port" not in r:
        raise ValidationError(f"{where}.http_port: required")
    if position != "front" and "backend" in r:
        raise ValidationError(f"{where}.backend: only the front stage selects a backend")
    return InvBinding(
        set=_id(r["set"], f"{where}.set"), engine=_id(r["engine"], f"{where}.engine"), position=position,
        ranks=ranks,
        group_mode=_id(r["group_mode"], f"{where}.group_mode") if "group_mode" in r else None,
        transport=_id(r["transport"], f"{where}.transport") if "transport" in r else None,
        backend=text(r["backend"], f"{where}.backend") if "backend" in r else None,
        stage_port=_port(r["stage_port"], f"{where}.stage_port") if "stage_port" in r else None,
        http_port=_port(r["http_port"], f"{where}.http_port") if "http_port" in r else None,
        http_address=_ip(r["http_address"], f"{where}.http_address") if "http_address" in r else None,
        max_positions=int(number(r["max_positions"], f"{where}.max_positions", 1, True))
        if "max_positions" in r else None,
        start=start)


def parse_inventory(data: Any) -> Inventory:
    r = record(data, "inventory", {"schema", "nodes", "engines", "front", "tail"},
               {"fallback", "router", "timeouts", "notes"})
    if r["schema"] != INVENTORY_SCHEMA:
        raise ValidationError(f"inventory.schema: expected {INVENTORY_SCHEMA}")
    nodes = {}
    for i, raw in enumerate(items(r["nodes"], "inventory.nodes", 64)):
        w = f"inventory.nodes[{i}]"
        n = record(raw, w, {"id", "address", "work_dir", "launcher"})
        launcher = record(n["launcher"], f"{w}.launcher", {"kind"}, {"target"})
        if launcher["kind"] == "local":
            if "target" in launcher:
                raise ValidationError(f"{w}.launcher: a local launcher has no target")
            target = None
        elif launcher["kind"] == "ssh":
            target = text(launcher.get("target"), f"{w}.launcher.target")
            if not re.match(r"^[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?$", target):
                raise ValidationError(f"{w}.launcher.target: expected [user@]host")
        else:
            raise ValidationError(f"{w}.launcher.kind: expected local or ssh")
        node = InvNode(_id(n["id"], f"{w}.id"), _ip(n["address"], f"{w}.address"),
                       _path(n["work_dir"], f"{w}.work_dir"), launcher["kind"], target)
        if node.id in nodes:
            raise ValidationError(f"{w}.id: duplicate node {node.id}")
        nodes[node.id] = node
    engines = {}
    for i, raw in enumerate(items(r["engines"], "inventory.engines", 32)):
        w = f"inventory.engines[{i}]"
        e = record(raw, w, {"id", "binary", "sha256", "backends", "model_path", "argv_form", "describe"},
                   {"env", "launch_prefix"})
        if e["argv_form"] not in FORMS:
            raise ValidationError(f"{w}.argv_form: expected one of {sorted(FORMS)}")
        d = record(e["describe"], f"{w}.describe", set(), {"file", "probe", "unchecked"})
        if len(d) != 1:
            raise ValidationError(f"{w}.describe: exactly one of file, probe, unchecked")
        source = next(iter(d))
        if source == "probe" and d["probe"] is not True:
            raise ValidationError(f"{w}.describe.probe: expected true")
        backends = tuple(text(b, f"{w}.backends[]") for b in items(e["backends"], f"{w}.backends", 8))
        prefix = tuple(_path(a, f"{w}.launch_prefix[]") for a in items(e.get("launch_prefix", []),
                                                                         f"{w}.launch_prefix", 4, 0))
        eng = InvEngine(
            id=_id(e["id"], f"{w}.id"), binary=_path(e["binary"], f"{w}.binary"),
            sha256=_sha(e["sha256"], f"{w}.sha256"), backends=backends,
            model_path=_path(e["model_path"], f"{w}.model_path"), argv_form=e["argv_form"],
            env=check_env(e.get("env", {}), f"{w}.env"), describe_source=source,
            describe_file=_path(d["file"], f"{w}.describe.file") if source == "file" else None,
            unchecked_reason=text(d["unchecked"], f"{w}.describe.unchecked") if source == "unchecked" else None,
            launch_prefix=prefix)
        if eng.id in engines:
            raise ValidationError(f"{w}.id: duplicate engine {eng.id}")
        engines[eng.id] = eng
    front = _binding(r["front"], "inventory.front", "front")
    tail = _binding(r["tail"], "inventory.tail", "tail")
    fallback = _binding(r["fallback"], "inventory.fallback", "fallback") if "fallback" in r else None
    router = None
    if "router" in r:
        rr = record(r["router"], "inventory.router", {"listen"}, {"api_key_file", "max_inflight", "queue_timeout_s"})
        listen = record(rr["listen"], "inventory.router.listen", {"address", "port"})
        address = _ip(listen["address"], "inventory.router.listen.address")
        port = int(number(listen["port"], "inventory.router.listen.port", 0, True))
        if port > 65535:
            raise ValidationError("inventory.router.listen.port: exceeds 65535")
        if not ipaddress.ip_address(address).is_loopback and "api_key_file" not in rr:
            raise ValidationError("inventory.router: a private (non-loopback) listener requires api_key_file")
        router = {"address": address, "port": port,
                  "api_key_file": _path(rr["api_key_file"], "inventory.router.api_key_file")
                  if "api_key_file" in rr else None,
                  "max_inflight": int(number(rr.get("max_inflight", 4), "inventory.router.max_inflight", 1, True)),
                  "queue_timeout_s": float(number(rr.get("queue_timeout_s", 600), "inventory.router.queue_timeout_s"))}
    t = record(r.get("timeouts", {}), "inventory.timeouts", set(), set(TIMEOUT_DEFAULTS))
    timeouts = {}
    for key, default in TIMEOUT_DEFAULTS.items():
        value = number(t.get(key, default), f"inventory.timeouts.{key}")
        low, high = TIMEOUT_BOUNDS[key]
        if not low <= value <= high:
            raise ValidationError(f"inventory.timeouts.{key}: expected {low}..{high}")
        timeouts[key] = value
    return Inventory(nodes, engines, front, tail, fallback, router, timeouts, canonical_sha256(r))


def load_inventory(path: Path) -> Inventory:
    return parse_inventory(_load_json(path, MAX_INVENTORY_BYTES, "inventory"))


def load_plan(path: Path) -> dict:
    plan = _load_json(path, MAX_PLAN_BYTES, "plan")
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValidationError(f"plan.schema: expected {PLAN_SCHEMA}")
    return plan


def load_recorded_describe(engine: InvEngine) -> EngineDescribe:
    data = _load_json(Path(engine.describe_file), 4 * 1024 * 1024, f"describe file of {engine.id}")
    return parse_engine_describe(data, expected_executable_sha256=engine.sha256)


# --- plan selection ----------------------------------------------------------------

def select_candidate(plan: dict, name: str | None) -> dict:
    pool = [c for c in [plan.get("best"), *plan.get("candidates", []), *plan.get("best_by_mode", {}).values()] if c]
    if name is None:
        chosen = plan.get("best")
        if chosen is None:
            raise ValidationError("plan has no feasible placement")
    else:
        found = [c for c in pool if c.get("name") == name]
        if not found:
            raise ValidationError(f"plan has no candidate {name!r}")
        chosen = found[0]
    if chosen.get("mode") != "hetero-pipeline":
        raise ValidationError(f"candidate {chosen.get('name')!r} is {chosen.get('mode')!r}, not a hetero-pipeline")
    if not chosen.get("feasible"):
        raise ValidationError(f"candidate {chosen.get('name')!r} is not feasible")
    layout = chosen.get("layout")
    if not isinstance(layout, dict) or not isinstance(layout.get("stages"), list):
        raise ValidationError("candidate has no pipeline layout (re-run plan-hetero with this TensorMeld)")
    if len(layout["stages"]) != 2:
        raise ValidationError(f"only two-stage pipelines deploy (FRONT + TAIL); this one has {len(layout['stages'])}")
    return chosen


# --- argv forms ----------------------------------------------------------------------

def build_argv(form: str, f: dict[str, Any]) -> list[str]:
    """The engine contract's command line for ``form`` from manifest fields ``f``."""
    binary, model = f["binary"], f["model_path"]
    if form == "pipeline-tail-v1":
        argv = [binary, model, "--rank", str(f["rank"]), "--transport", f["transport"], "--role", "pipeline-stage",
                "--first-layer", str(f["layers"][0]), "--layers", str(f["layers"][1]),
                "--stage-listen", f["stage_listen"], "--stage-token-file", f["token_file"]]
    elif form == "pipeline-front-v1":
        argv = [binary, model, "--backend", f["backend"], "--layers", str(f["layers"][1]),
                "--stage-connect", f["stage_connect"], "--stage-token-file", f["token_file"], "--http", f["http"]]
    elif form == "group-serve-v1":
        host, port = f["http"].rsplit(":", 1)
        argv = [binary, model, "--rank", str(f["rank"]), "--transport", f["transport"],
                "--layers", str(f["layers"][1]), "--http-bind", host, "--http-port", port]
        if f.get("api_key_file"):
            argv += ["--http-api-key-file", f["api_key_file"]]
        if f.get("max_positions"):
            argv += ["--context", str(f["max_positions"])]
        return argv
    else:
        raise ValidationError(f"unknown argv form {form!r}")
    if f.get("max_positions"):
        argv += ["--max-positions", str(f["max_positions"])]
    return argv


# --- binding and manifests ---------------------------------------------------------

@dataclass(frozen=True)
class Deployment:
    id: str
    epoch: int
    plan_sha256: str
    candidate: str
    front: tuple[dict, ...]
    tail: tuple[dict, ...]
    fallback: tuple[dict, ...]
    fallback_start: str | None
    state_handoff: bool
    router: dict[str, Any] | None
    timeouts: dict[str, float]
    checks: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def manifests(self) -> tuple[dict, ...]:
        return (*self.tail, *self.front, *self.fallback)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": DEPLOYMENT_SCHEMA,
            "deployment": {"id": self.id, "epoch": self.epoch},
            "plan": {"sha256": self.plan_sha256, "candidate": self.candidate},
            "start_order": [m["process"]["id"] for m in (*self.tail, *self.front)]
            + ([m["process"]["id"] for m in self.fallback] if self.fallback_start == "standby" else []),
            "fallback": None if not self.fallback else {
                "start": self.fallback_start, "processes": [m["process"]["id"] for m in self.fallback],
                "state_handoff": self.state_handoff},
            "router": self.router,
            "timeouts": self.timeouts,
            "checks": list(self.checks),
            "warnings": list(self.warnings),
            "manifests": list(self.manifests),
            "qualified": False,
        }


def _part_requires(model: ModelParts, part_ids: list[str]) -> set[str]:
    out: set[str] = set()
    for pid in part_ids:
        p = model.part(pid)
        out |= set(p.requires)
    return out


def _check_describe(desc: EngineDescribe | None, engine: InvEngine, binding: InvBinding, role: str,
                    required: set[str], model_id: str, world: int, checks: list[str]) -> dict[str, Any]:
    """Validate one engine instance against what its stage needs; returns the manifest's describe record."""
    where = f"{binding.position} ({engine.id})"
    if desc is None:
        if engine.describe_source == "unchecked":
            return {"source": "unchecked", "reason": engine.unchecked_reason}
        return {"source": "probe", "status": "pending: checked at launch"}
    if desc.executable_sha256 != engine.sha256:
        raise ValidationError(f"{where}: describe is for another binary")
    if role not in desc.roles:
        raise ValidationError(f"{where}: engine does not declare the {role} role")
    if engine.argv_form == "pipeline-front-v1" and binding.backend and desc.backend not in engine.backends:
        raise ValidationError(f"{where}: describe backend {desc.backend!r} not in the engine's backends")
    have = set(desc.capabilities)
    if world > 1:
        modes = [g for g in desc.group_modes if g.id == binding.group_mode]
        if not modes:
            raise ValidationError(f"{where}: engine has no group mode {binding.group_mode!r}")
        mode = modes[0]
        if mode.world != world:
            raise ValidationError(f"{where}: group mode {mode.id} has world {mode.world}, the set has {world} devices")
        if binding.transport not in mode.transports:
            raise ValidationError(f"{where}: group mode {mode.id} does not list transport {binding.transport!r}")
        have |= set(mode.capabilities)
    missing = sorted(required - have)
    if missing:
        raise ValidationError(f"{where}: engine lacks capabilities {missing} for its parts")
    if not any(m.model_id == model_id for m in desc.models):
        raise ValidationError(f"{where}: engine does not describe model {model_id!r}")
    checks.append(f"{where}: role {role}, {len(required)} capabilities"
                  + (f", group mode {binding.group_mode} world {world}" if world > 1 else ""))
    return {"source": engine.describe_source, "fingerprint": desc.fingerprint, "revision": desc.revision,
            "backend": desc.backend, "state_export": desc.state_export, "state_import": desc.state_import,
            "state_format": desc.state_format or None}


def build_deployment(plan: dict, model: ModelParts, hw: Hardware, inv: Inventory, *,
                     candidate: str | None = None, describes: dict[str, EngineDescribe | None] | None = None,
                     deployment_id: str = "dry-run", epoch: int = 1) -> Deployment:
    describes = describes or {}
    _id(deployment_id, "deployment_id")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValidationError(f"plan.schema: expected {PLAN_SCHEMA}")
    if plan.get("model_fingerprint") != model.fingerprint or plan.get("model") != model.model_id:
        raise ValidationError("plan was computed for another model descriptor")
    if plan.get("hardware_fingerprint") != hw.fingerprint:
        raise ValidationError("plan was computed for another hardware descriptor (or other --unavailable/--link)")
    chosen = select_candidate(plan, candidate)
    stages = chosen["layout"]["stages"]
    checks: list[str] = [f"plan {chosen['name']}: model and hardware fingerprints match"]
    warnings: list[str] = []
    for s in stages:
        for pid in s["parts"] + s["remote_parts"] + s["serves_parts"]:
            model.part(pid)
        if tuple(s["devices"]) != tuple(chosen["sets"][s["set"]]):
            raise ValidationError(f"plan stage {s['set']}: devices differ from the candidate's sets")
    front_stage, tail_stage = stages
    if not front_stage["has_embedding"] or not tail_stage["has_head"]:
        raise ValidationError("pipeline stages must be FRONT (embedding) then TAIL (head)")
    k = front_stage["layers"][1]
    if not 1 <= k < model.layers or tail_stage["layers"] != [k, model.layers]:
        raise ValidationError(f"pipeline cut {k} outside 1..{model.layers - 1}")
    plan_sha = canonical_sha256(plan)
    base = {"deployment": {"id": deployment_id, "epoch": int(epoch)},
            "plan": {"sha256": plan_sha, "candidate": chosen["name"], "mode": chosen["mode"],
                     "model": model.model_id, "model_fingerprint": model.fingerprint,
                     "hardware_fingerprint": hw.fingerprint}}

    def node_of(device: str) -> InvNode:
        d = hw.device(device)
        node = inv.nodes.get(d.node)
        if node is None:
            raise ValidationError(f"inventory has no node {d.node!r} (device {device})")
        return node

    def instance(binding: InvBinding, stage: dict, role: str) -> tuple[InvEngine, dict, list[str]]:
        if binding.set != stage["set"]:
            raise ValidationError(f"inventory.{binding.position}.set {binding.set!r} is not the plan's "
                                  f"{binding.position} set {stage['set']!r}")
        engine = inv.engines.get(binding.engine)
        if engine is None:
            raise ValidationError(f"inventory.{binding.position}.engine: unknown engine {binding.engine!r}")
        form = FORMS[engine.argv_form]
        if role != form["role"] or binding.position not in form["positions"]:
            raise ValidationError(f"inventory.{binding.position}: engine {engine.id} has argv form "
                                  f"{engine.argv_form}, not usable as {binding.position}")
        devices = list(stage["devices"])
        ranks = list(binding.ranks) or devices
        if sorted(ranks) != sorted(devices):
            raise ValidationError(f"inventory.{binding.position}.ranks {ranks} do not cover the set's devices {devices}")
        if len(devices) > 1:
            if not form["group"]:
                raise ValidationError(f"{binding.position}: argv form {engine.argv_form} runs one device, "
                                      f"set {stage['set']} has {len(devices)}")
            if not binding.group_mode or not binding.transport:
                raise ValidationError(f"inventory.{binding.position}: a device group needs group_mode and transport")
            if stage["set"] not in {g.id for g in hw.groups}:
                raise ValidationError(f"{binding.position}: set {stage['set']} is not a hardware group")
        elif form["group"] and not binding.transport:
            raise ValidationError(f"inventory.{binding.position}: transport required by {engine.argv_form}")
        required = _part_requires(model, stage["parts"] + stage.get("serves_parts", []))
        desc_record = _check_describe(describes.get(engine.id), engine, binding, role, required,
                                      model.model_id, len(devices), checks)
        return engine, desc_record, ranks

    def manifest(position: str, engine: InvEngine, binding: InvBinding, desc_record: dict, stage: dict,
                 rank: int, world: int, device: str, endpoints: dict, role: str, backend: str,
                 secret_files: dict, readiness: dict) -> dict:
        node = node_of(device)
        pid = f"{position}-r{rank}" if world > 1 else position
        fields = {"binary": engine.binary, "model_path": engine.model_path, "rank": rank,
                  "transport": binding.transport, "layers": stage["layers"], "backend": backend,
                  "max_positions": binding.max_positions, **endpoints, **secret_files}
        m = {
            "schema": MANIFEST_SCHEMA,
            **base,
            "process": {"id": pid, "position": position, "role": role, "rank": rank, "world": world},
            "node": {"id": node.id, "address": node.address, "launcher": node.launcher,
                     "ssh_target": node.ssh_target, "work_dir": f"{node.work_dir}/{deployment_id}"},
            "set": stage["set"],
            "devices": [device],
            "engine": {"id": engine.id, "binary": engine.binary, "sha256": engine.sha256, "backend": backend,
                       "argv_form": engine.argv_form, "group_mode": binding.group_mode if world > 1 else None,
                       "transport": binding.transport, "describe": desc_record},
            "model": {"id": model.model_id, "path": engine.model_path},
            "layers": list(stage["layers"]),
            "has_embedding": bool(stage.get("has_embedding")),
            "has_head": bool(stage.get("has_head")),
            "parts": list(stage["parts"]),
            "remote_parts": list(stage.get("remote_parts", [])),
            "serves_parts": list(stage.get("serves_parts", [])),
            "endpoints": endpoints,
            "secrets": {name: {"file": path, "bytes": 32, "encoding": "hex", "mode": "0600",
                               "generated": "at launch by TensorMeld"} for name, path in secret_files.items()},
            "max_positions": binding.max_positions,
            "env": engine.env,
            "launch_prefix": list(engine.launch_prefix),
            "argv": build_argv(engine.argv_form, fields),
            "readiness": readiness,
        }
        m["manifest_sha256"] = canonical_sha256(m)
        return m

    # TAIL: the AMD group runs [K, layers) + head, rank 0 listens for the FRONT.
    tail_engine, tail_desc, tail_ranks = instance(inv.tail, tail_stage, "pipeline-stage")
    rank0_node = node_of(tail_ranks[0])
    stage_endpoint = f"{rank0_node.address}:{inv.tail.stage_port}"
    tail = []
    for r, device in enumerate(tail_ranks):
        node = node_of(device)
        token = f"{node.work_dir}/{deployment_id}/tail-r{r}.stage-token" if len(tail_ranks) > 1 \
            else f"{node.work_dir}/{deployment_id}/tail.stage-token"
        tail.append(manifest("tail", tail_engine, inv.tail, tail_desc, tail_stage, r, len(tail_ranks), device,
                             {"stage_listen": stage_endpoint}, "pipeline-stage", tail_engine.backends[0],
                             {"token_file": token},
                             {"line": "READY" if r == 0 else "alive", "timeout_s": inv.timeouts["tail_ready_s"]}))
    # FRONT: one device, embedding + [0, K), HTTP, connects to the TAIL.
    front_engine, front_desc, front_ranks = instance(inv.front, front_stage, "pipeline-stage")
    if len(front_ranks) != 1:
        raise ValidationError("the FRONT stage runs on one device")
    planned_backend = hw.device(front_ranks[0]).backend
    backend = inv.front.backend or planned_backend
    if backend not in FRONT_BACKENDS or backend not in front_engine.backends:
        raise ValidationError(f"FRONT backend {backend!r}: expected one of the engine's {front_engine.backends} "
                              f"within {FRONT_BACKENDS}")
    if backend != planned_backend:
        warnings.append(f"FRONT runs backend {backend} on device {front_ranks[0]} planned as {planned_backend}: "
                        "a test substitute, not the planned numerics or speed")
    front_node = node_of(front_ranks[0])
    http_address = inv.front.http_address or "127.0.0.1"
    if not ipaddress.ip_address(http_address).is_loopback:
        raise ValidationError("FRONT http must bind loopback: the pipeline FRONT has no API-key flag; "
                              "expose it through the TensorMeld router")
    if front_node.launcher != "local":
        raise ValidationError("the FRONT binds loopback HTTP, so it runs on the controller's node (local launcher)")
    front = [manifest("front", front_engine, inv.front, front_desc, front_stage, 0, 1, front_ranks[0],
                      {"stage_connect": stage_endpoint, "http": f"{http_address}:{inv.front.http_port}"},
                      "pipeline-stage", backend,
                      {"token_file": f"{front_node.work_dir}/{deployment_id}/front.stage-token"},
                      {"line": "READY", "timeout_s": inv.timeouts["front_ready_s"], "http_health": "/health"})]
    # Ports must not collide on a node.
    used: dict[tuple[str, int], str] = {}

    def claim(node_id: str, port: int, what: str) -> None:
        if (node_id, port) in used:
            raise ValidationError(f"port {port} on node {node_id}: used by {used[(node_id, port)]} and {what}")
        used[(node_id, port)] = what

    claim(rank0_node.id, inv.tail.stage_port, "tail stage")
    claim(front_node.id, inv.front.http_port, "front http")
    if inv.router is not None and inv.router["port"]:
        claim(front_node.id, inv.router["port"], "router")
    # FALLBACK: the plan without the preemptible device, as a whole-model group.
    fallback: list[dict] = []
    fallback_start = None
    state_handoff = False
    fb = plan.get("fallback")
    if inv.fallback is not None:
        if not fb or not fb.get("best"):
            raise ValidationError("inventory has a fallback but the plan reports none")
        fb_best = fb["best"]
        without = set(fb.get("without") or [])
        if not without or not without <= set(chosen.get("uses_preemptible", []))                 or not without <= set(front_stage["devices"]):
            raise ValidationError(f"plan fallback (without {sorted(without)}) is not the retirement of this "
                                  "candidate's preemptible FRONT device")
        if fb_best.get("mode") not in ("single", "engine-group"):
            raise ValidationError(f"fallback {fb_best.get('name')!r} is {fb_best.get('mode')!r}: only a whole-model "
                                  "fallback deploys")
        (fb_set, fb_devices), = fb_best["sets"].items()
        fb_stage = {"set": fb_set, "devices": fb_devices, "layers": [0, model.layers], "has_embedding": True,
                    "has_head": True, "parts": sorted(p.id for p in model.parts if not p.optional),
                    "remote_parts": [], "serves_parts": []}
        if set(fb_devices) & set(front_stage["devices"]):
            raise ValidationError("the fallback uses a device of the FRONT")
        if inv.fallback.start == "standby":
            co = fb.get("co_resident") or {}
            if not co.get("fits"):
                raise ValidationError("fallback start=standby needs the plan's co_resident.fits (it does not fit)")
            warnings.append("standby fallback: the planner's co-residency counts one engine instance holding both "
                            "placements; two processes on the same devices need the engine to share weights")
        fb_engine, fb_desc, fb_ranks = instance(inv.fallback, fb_stage, "whole")
        fb_node0 = node_of(fb_ranks[0])
        fb_http = f"{inv.fallback.http_address or fb_node0.address}:{inv.fallback.http_port}"
        claim(fb_node0.id, inv.fallback.http_port, "fallback http")
        needs_key = not ipaddress.ip_address(fb_http.rsplit(":", 1)[0]).is_loopback
        for r, device in enumerate(fb_ranks):
            node = node_of(device)
            secrets = {"api_key_file": f"{node.work_dir}/{deployment_id}/fallback-r{r}.http-key"} if needs_key else {}
            fallback.append(manifest("fallback", fb_engine, inv.fallback, fb_desc, fb_stage, r, len(fb_ranks), device,
                                     {"http": fb_http}, "whole", fb_engine.backends[0], secrets,
                                     {"line": "READY" if r == 0 else "alive",
                                      "timeout_s": inv.timeouts["fallback_ready_s"], "http_health": "/health"}))
        fallback_start = inv.fallback.start
        exports = front_desc.get("state_export") is not False and tail_desc.get("state_export") is not False
        imports = fb_desc.get("state_import") is not False
        state_handoff = bool(exports and imports)
        if not state_handoff:
            warnings.append("an engine does not declare state export/import: retiring the FRONT device loses "
                            "the requests' state (the fallback starts empty)")
        checks.append(f"fallback {fb_best['name']}: {len(fb_ranks)} rank(s), start {fallback_start}")
    elif fb and fb.get("best"):
        warnings.append(f"plan has a fallback ({fb['best']['name']}) but the inventory binds none: "
                        "retiring the FRONT device stops the service")
    return Deployment(deployment_id, int(epoch), plan_sha, chosen["name"], tuple(front), tuple(tail),
                      tuple(fallback), fallback_start, state_handoff, inv.router, dict(inv.timeouts),
                      tuple(checks), tuple(warnings))


# --- manifest validation -----------------------------------------------------------

MANIFEST_KEYS = {"schema", "deployment", "plan", "process", "node", "set", "devices", "engine", "model", "layers",
                 "has_embedding", "has_head", "parts", "remote_parts", "serves_parts", "endpoints", "secrets",
                 "max_positions", "env", "launch_prefix", "argv", "readiness", "manifest_sha256"}


def validate_manifest(m: Any) -> dict:
    """Strict check of one process manifest: closed keys, its own hash, and the argv re-derived
    from its fields (so a manifest cannot carry an argv its fields do not describe)."""
    r = record(m, "manifest", MANIFEST_KEYS)
    if r["schema"] != MANIFEST_SCHEMA:
        raise ValidationError(f"manifest.schema: expected {MANIFEST_SCHEMA}")
    body = {k: v for k, v in r.items() if k != "manifest_sha256"}
    if canonical_sha256(body) != r["manifest_sha256"]:
        raise ValidationError("manifest_sha256 does not match the manifest")
    proc = record(r["process"], "manifest.process", {"id", "position", "role", "rank", "world"})
    engine = record(r["engine"], "manifest.engine",
                    {"id", "binary", "sha256", "backend", "argv_form", "group_mode", "transport", "describe"})
    form = FORMS.get(engine["argv_form"])
    if form is None or proc["position"] not in form["positions"] or proc["role"] != form["role"]:
        raise ValidationError("manifest: role/position do not match the argv form")
    _sha(engine["sha256"], "manifest.engine.sha256")
    layers = r["layers"]
    if not (isinstance(layers, list) and len(layers) == 2 and all(type(x) is int for x in layers)
            and 0 <= layers[0] < layers[1]):
        raise ValidationError("manifest.layers: expected [first, end)")
    check_env(r["env"], "manifest.env")
    endpoints = r["endpoints"]
    if not isinstance(endpoints, dict) or not endpoints:
        raise ValidationError("manifest.endpoints: expected an object")
    for key, value in endpoints.items():
        host, _, port = str(value).rpartition(":")
        _ip(host, f"manifest.endpoints.{key}")
        _port(int(port) if port.isdigit() else -1, f"manifest.endpoints.{key}")
    secret_files = {name: s["file"] for name, s in r["secrets"].items()}
    fields = {"binary": engine["binary"], "model_path": r["model"]["path"], "rank": proc["rank"],
              "transport": engine["transport"], "layers": layers, "backend": engine["backend"],
              "max_positions": r["max_positions"], **endpoints, **secret_files}
    if build_argv(engine["argv_form"], fields) != r["argv"]:
        raise ValidationError("manifest.argv does not match its fields")
    if engine["binary"] != r["argv"][0]:
        raise ValidationError("manifest.argv must start with the approved binary")
    return r


__all__ = [
    "DEPLOYMENT_SCHEMA",
    "INVENTORY_SCHEMA",
    "MANIFEST_SCHEMA",
    "Deployment",
    "Inventory",
    "build_argv",
    "build_deployment",
    "canonical_sha256",
    "load_inventory",
    "load_plan",
    "load_recorded_describe",
    "parse_inventory",
    "select_candidate",
    "validate_manifest",
]
