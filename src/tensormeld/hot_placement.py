"""Hot-expert placement file: ``tensormeld/hot-expert-placement-v1``.

A placement says, for every MoE layer, which routed experts are resident on
which device of a "dense + hot experts" deployment:

* one ``parts-host`` device runs every dense part (attention, mixers, shared
  experts, head) and the experts listed for it;
* ``expert-tier`` devices (a second GPU next to the parts-host) and
  ``expert-cache`` devices (host memory, copied to the parts-host on demand)
  hold further experts;
* exactly one ``expert-server`` device (a set of servers, each owning a
  contiguous id range) holds *every* expert and serves the ids no other device
  lists for that layer.

``layers[l][dev]`` lists ascending, disjoint expert ids per device; a device
missing from an entry holds nothing for that layer.  ``expected`` carries the
producer's own out-of-sample numbers per row pattern (``R1`` required;
``R<n>``/``W<n>`` verify windows, ``B<n>``/``N<n>``/``batch-<n>`` batches, or any
name with an explicit ``rows``).  Readers reject an unknown ``format`` and accept
additive keys within v1.

The reader checks the file on its own (ranges, order, disjointness, per-device
memory ``count * expert_bytes <= expert_gib * 2^30``) and against a model-parts
descriptor (layers, family, and experts/top_k/expert_bytes when the descriptor
declares its ``moe`` geometry).  It never makes a plan executable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .hetero_hardware import Hardware
from .hetero_model import ModelParts
from .schema import ValidationError, _no_duplicates, number, text

HOT_PLACEMENT_FORMAT = "tensormeld/hot-expert-placement-v1"
MAX_PLACEMENT_BYTES = 8 * 1024 * 1024
MAX_DEVICES = 16
MAX_SERVERS = 16
MAX_PATTERNS = 64
MAX_LAYERS = 512
MAX_EXPERTS = 1 << 16
GIB = 2**30
ROLES = frozenset({"parts-host", "expert-tier", "expert-cache", "expert-server"})
KINDS = frozenset({"cuda", "hip", "host"})
UNITS = frozenset({"step", "token"})
_ROW_PATTERN = re.compile(r"^[RW]([1-9][0-9]{0,2})$")
_BATCH_PATTERN = re.compile(r"^(?:B|N|batch-?)([1-9][0-9]{0,2})$")
_IDENTITY = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class HotServer:
    node: str
    first: int
    end: int  # exclusive


@dataclass(frozen=True)
class HotDevice:
    id: str
    kind: str
    role: str
    budget_gib: float
    reserve_gib: float
    expert_gib: float
    servers: tuple[HotServer, ...] = ()


@dataclass(frozen=True)
class Expectation:
    """The producer's numbers for one row pattern (per decode step unless ``unit`` is token)."""
    pattern: str
    rows: int | None
    batch: bool
    remote_layers_per_token: float
    cold_vectors_per_token: float
    tok_s: float
    unit: str = "step"
    servers_per_remote_layer: float | None = None
    ram_vectors_per_token: float | None = None
    tokens_per_window: float | None = None  # accepted tokens per step (verify windows); the file's tok_s uses it

    def per_step(self, value: float | None) -> float | None:
        """A rate of this pattern expressed per decode step of ``rows`` rows."""
        if value is None:
            return None
        return value * (self.rows or 1) if self.unit == "token" else value

    def public(self) -> dict:
        out = {"rows": self.rows, "batch": self.batch, "unit": self.unit,
               "remote_layers_per_token": self.remote_layers_per_token,
               "cold_vectors_per_token": self.cold_vectors_per_token, "tok_s": self.tok_s}
        if self.servers_per_remote_layer is not None:
            out["servers_per_remote_layer"] = self.servers_per_remote_layer
        if self.ram_vectors_per_token is not None:
            out["ram_vectors_per_token"] = self.ram_vectors_per_token
        if self.tokens_per_window is not None:
            out["tokens_per_window"] = self.tokens_per_window
        return out


@dataclass(frozen=True)
class HotPlacement:
    family: str
    model_file: str
    model_identity: str | None
    layers: int
    experts: int
    top_k: int
    expert_bytes: int
    canonical_split: int
    dense_device: str
    devices: tuple[HotDevice, ...]
    layer_experts: tuple[dict[str, tuple[int, ...]], ...]  # per layer: device id -> ascending ids
    expected: dict[str, Expectation]
    combine_identity: str
    fit: dict
    fingerprint: str
    ignored_keys: tuple[str, ...] = field(default=())

    def device(self, device_id: str) -> HotDevice:
        for d in self.devices:
            if d.id == device_id:
                return d
        raise KeyError(device_id)

    @property
    def server_device(self) -> HotDevice:
        return next(d for d in self.devices if d.role == "expert-server")

    @property
    def resident_devices(self) -> tuple[HotDevice, ...]:
        """Devices that list experts explicitly (everything but the expert server)."""
        return tuple(d for d in self.devices if d.role != "expert-server")

    def per_layer(self, device_id: str) -> tuple[int, ...]:
        return tuple(len(entry.get(device_id, ())) for entry in self.layer_experts)

    def expert_count(self, device_id: str) -> int:
        dev = self.device(device_id)
        if dev.role == "expert-server":
            return self.layers * self.experts
        return sum(self.per_layer(device_id))

    def resident_bytes(self, device_id: str) -> int:
        return self.expert_count(device_id) * self.expert_bytes

    def server_bytes(self, server: HotServer) -> int:
        return (server.end - server.first) * self.layers * self.expert_bytes

    def summary(self) -> dict:
        devices = {}
        for d in self.devices:
            e = {"kind": d.kind, "role": d.role}
            if d.role == "expert-server":
                e["servers"] = [{"node": s.node, "experts": [s.first, s.end],
                                 "gib": round(self.server_bytes(s) / GIB, 2)} for s in d.servers]
            else:
                counts = self.per_layer(d.id)
                e.update({"experts": sum(counts), "gib": round(self.resident_bytes(d.id) / GIB, 2),
                          "expert_gib": d.expert_gib, "per_layer_min": min(counts), "per_layer_max": max(counts)})
            devices[d.id] = e
        out = {
            "format": HOT_PLACEMENT_FORMAT,
            "fingerprint": self.fingerprint,
            "model": {"family": self.family, "file": self.model_file, "model_identity": self.model_identity,
                      "layers": self.layers, "experts": self.experts, "top_k": self.top_k,
                      "expert_bytes": self.expert_bytes, "canonical_split": self.canonical_split},
            "dense_device": self.dense_device,
            "combine_identity": self.combine_identity,
            "devices": devices,
            "expected": {k: v.public() for k, v in self.expected.items()},
        }
        if self.ignored_keys:
            out["ignored_keys"] = list(self.ignored_keys)
        return out


# --- parsing ---------------------------------------------------------------------

def _obj(value, where: str, required: set[str], ignored: list[str], known: set[str] = frozenset()) -> dict:
    """An object with ``required`` keys; additive keys outside ``required | known`` are recorded, not rejected."""
    if not isinstance(value, dict):
        raise ValidationError(f"{where}: expected object")
    missing = required - value.keys()
    if missing:
        raise ValidationError(f"{where}: missing={sorted(missing)}")
    ignored.extend(f"{where}.{k}" for k in sorted(value.keys() - required - set(known)))
    return value


def _int(value, where: str, minimum: int = 0) -> int:
    return int(number(value, where, minimum, True))


def _float(value, where: str) -> float:
    return float(number(value, where))


def _expectation(name: str, raw, layers: int, top_k: int, ignored: list[str]) -> Expectation:
    where = f"expected.{name}"
    r = _obj(raw, where, {"remote_layers_per_token", "cold_vectors_per_token", "tok_s"}, ignored,
             {"rows", "unit", "servers_per_remote_layer", "ram_vectors_per_token", "tokens_per_window", "notes"})
    m_rows, m_batch = _ROW_PATTERN.match(name), _BATCH_PATTERN.match(name)
    rows = int(m_rows.group(1)) if m_rows else int(m_batch.group(1)) if m_batch else None
    if "rows" in r:
        declared = _int(r["rows"], f"{where}.rows", 1)
        if rows is not None and declared != rows:
            raise ValidationError(f"{where}.rows: {declared} contradicts the pattern name")
        rows = declared
    if rows is not None and rows > 1024:
        raise ValidationError(f"{where}.rows: at most 1024")
    unit = r.get("unit", "step")
    if unit not in UNITS:
        raise ValidationError(f"{where}.unit: expected one of {sorted(UNITS)}")
    remote = _float(r["remote_layers_per_token"], f"{where}.remote_layers_per_token")
    cold = _float(r["cold_vectors_per_token"], f"{where}.cold_vectors_per_token")
    tok_s = _float(r["tok_s"], f"{where}.tok_s")
    step_rows = 1 if unit == "token" else (rows or 1)
    if remote > layers:
        raise ValidationError(f"{where}.remote_layers_per_token: above the {layers} layers")
    if cold < remote - 1e-9:
        raise ValidationError(f"{where}: cold_vectors_per_token below remote_layers_per_token "
                              "(a remote layer has at least one cold vector)")
    if rows is not None and cold > remote * top_k * step_rows + 1e-9:
        raise ValidationError(f"{where}.cold_vectors_per_token: above remote layers x top_k x rows")
    servers = r.get("servers_per_remote_layer")
    if servers is not None:
        servers = _float(servers, f"{where}.servers_per_remote_layer")
        if servers < 1 and remote > 0:
            raise ValidationError(f"{where}.servers_per_remote_layer: expected >= 1")
    ram = r.get("ram_vectors_per_token")
    if ram is not None:
        ram = _float(ram, f"{where}.ram_vectors_per_token")
    tpw = r.get("tokens_per_window")
    if tpw is not None:
        tpw = _float(tpw, f"{where}.tokens_per_window")
        if not 0 < tpw <= (rows or 1024):
            raise ValidationError(f"{where}.tokens_per_window: expected 0 < tokens <= rows")
    return Expectation(name, rows, bool(m_batch) and not m_rows, remote, cold, tok_s, unit, servers, ram, tpw)


def parse_hot_placement(data) -> HotPlacement:
    ignored: list[str] = []
    if not isinstance(data, dict):
        raise ValidationError("placement: expected object")
    if data.get("format") != HOT_PLACEMENT_FORMAT:
        raise ValidationError(f"format: unknown {data.get('format')!r}; this reader accepts {HOT_PLACEMENT_FORMAT}")
    r = _obj(data, "placement", {"format", "model", "dense", "devices", "layers", "combine", "expected"},
             ignored, {"fit", "notes"})
    m = _obj(r["model"], "model", {"family", "layers", "experts", "top_k", "expert_bytes"}, ignored,
             {"file", "model_identity", "canonical_split"})
    layers = _int(m["layers"], "model.layers", 1)
    experts = _int(m["experts"], "model.experts", 1)
    top_k = _int(m["top_k"], "model.top_k", 1)
    expert_bytes = _int(m["expert_bytes"], "model.expert_bytes", 1)
    if layers > MAX_LAYERS or experts > MAX_EXPERTS or top_k > experts:
        raise ValidationError("model: expected layers <= 512 and 1 <= top_k <= experts <= 65536")
    split = _int(m.get("canonical_split", experts // 2), "model.canonical_split", 1)
    if split >= experts:
        raise ValidationError("model.canonical_split: expected 1..experts-1")
    identity = m.get("model_identity")
    if identity is not None and (not isinstance(identity, str) or not _IDENTITY.match(identity)):
        raise ValidationError("model.model_identity: expected 16 lowercase hex characters or null")
    model_file = m.get("file", "")
    if not isinstance(model_file, str) or len(model_file) > 256:
        raise ValidationError("model.file: expected a string of at most 256 characters")

    raw_devices = r["devices"]
    if not isinstance(raw_devices, list) or not 2 <= len(raw_devices) <= MAX_DEVICES:
        raise ValidationError(f"devices: expected 2..{MAX_DEVICES} entries")
    devices = []
    declared: dict[str, int] = {}  # optional per-device expert totals, cross-checked against layers
    for i, raw in enumerate(raw_devices):
        where = f"devices[{i}]"
        d = _obj(raw, where, {"id", "kind", "role"}, ignored,
                 {"budget_gib", "reserve_gib", "expert_gib", "servers", "experts", "notes"})
        did = text(d["id"], f"{where}.id")
        if d["kind"] not in KINDS:
            raise ValidationError(f"{where}.kind: expected one of {sorted(KINDS)}")
        if d["role"] not in ROLES:
            raise ValidationError(f"{where}.role: expected one of {sorted(ROLES)}")
        budget = _float(d.get("budget_gib", 0.0), f"{where}.budget_gib")
        reserve = _float(d.get("reserve_gib", 0.0), f"{where}.reserve_gib")
        if reserve > budget and budget:
            raise ValidationError(f"{where}.reserve_gib: above budget_gib")
        expert_gib = _float(d.get("expert_gib", max(budget - reserve, 0.0)), f"{where}.expert_gib")
        if budget and expert_gib > budget - reserve + 1e-9:
            raise ValidationError(f"{where}.expert_gib: above budget_gib - reserve_gib")
        servers: list[HotServer] = []
        if d["role"] == "expert-server":
            raw_servers = d.get("servers")
            if not isinstance(raw_servers, list) or not 1 <= len(raw_servers) <= MAX_SERVERS:
                raise ValidationError(f"{where}.servers: expected 1..{MAX_SERVERS} servers")
            for j, s in enumerate(raw_servers):
                s = _obj(s, f"{where}.servers[{j}]", {"node", "experts"}, ignored, {"notes"})
                rng = s["experts"]
                if not isinstance(rng, list) or len(rng) != 2:
                    raise ValidationError(f"{where}.servers[{j}].experts: expected [first, end)")
                first = _int(rng[0], f"{where}.servers[{j}].experts[0]")
                end = _int(rng[1], f"{where}.servers[{j}].experts[1]", 1)
                servers.append(HotServer(text(s["node"], f"{where}.servers[{j}].node"), first, end))
            servers.sort(key=lambda s: s.first)
            cursor = 0
            for s in servers:
                if s.first != cursor or s.end <= s.first:
                    raise ValidationError(f"{where}.servers: id ranges must tile [0, {experts}) without gaps "
                                          "or overlaps (the expert server holds every expert)")
                cursor = s.end
            if cursor != experts:
                raise ValidationError(f"{where}.servers: id ranges must tile [0, {experts})")
            if len({s.node for s in servers}) != len(servers):
                raise ValidationError(f"{where}.servers: duplicate node")
        elif "servers" in d:
            raise ValidationError(f"{where}.servers: only an expert-server has servers")
        if "experts" in d and d["role"] != "expert-server":
            declared[did] = _int(d["experts"], f"{where}.experts")
        devices.append(HotDevice(did, d["kind"], d["role"], budget, reserve, expert_gib, tuple(servers)))
    ids = [d.id for d in devices]
    if len(set(ids)) != len(ids):
        raise ValidationError("devices: duplicate id")
    roles = [d.role for d in devices]
    if roles.count("parts-host") != 1 or roles.count("expert-server") != 1:
        raise ValidationError("devices: expected exactly one parts-host and one expert-server")
    by_id = {d.id: d for d in devices}

    dense = _obj(r["dense"], "dense", {"device", "layers"}, ignored)
    if dense["device"] not in by_id or by_id[dense["device"]].role != "parts-host":
        raise ValidationError("dense.device: must name the parts-host device")
    if dense["layers"] != "all":
        raise ValidationError('dense.layers: v1 readers accept only "all"')

    raw_layers = r["layers"]
    if not isinstance(raw_layers, list) or len(raw_layers) != layers:
        raise ValidationError(f"layers: expected one entry per layer 0..{layers - 1}")
    server_id = next(d.id for d in devices if d.role == "expert-server")
    layer_experts = []
    for i, raw in enumerate(raw_layers):
        where = f"layers[{i}]"
        if not isinstance(raw, dict) or "layer" not in raw:
            raise ValidationError(f"{where}: expected an object with 'layer'")
        if _int(raw["layer"], f"{where}.layer") != i:
            raise ValidationError(f"{where}.layer: expected {i} (entries in layer order)")
        entry: dict[str, tuple[int, ...]] = {}
        seen: dict[int, str] = {}
        for key, ids_raw in raw.items():
            if key == "layer":
                continue
            if key == server_id:
                raise ValidationError(f"{where}.{key}: the expert server is implicit (it serves every unlisted id)")
            if key not in by_id:
                raise ValidationError(f"{where}.{key}: unknown device")
            if not isinstance(ids_raw, list) or len(ids_raw) > experts:
                raise ValidationError(f"{where}.{key}: expected a list of at most {experts} expert ids")
            prev = -1
            for x in ids_raw:
                if type(x) is not int or not 0 <= x < experts:
                    raise ValidationError(f"{where}.{key}: expert id {x!r} out of range 0..{experts - 1}")
                if x <= prev:
                    raise ValidationError(f"{where}.{key}: ids must be strictly ascending")
                prev = x
                if x in seen:
                    raise ValidationError(f"{where}: expert {x} listed on both {seen[x]} and {key}")
                seen[x] = key
            entry[key] = tuple(ids_raw)
        layer_experts.append(entry)

    for d in devices:
        if d.role == "expert-server":
            continue
        count = sum(len(e.get(d.id, ())) for e in layer_experts)
        if d.id in declared and declared[d.id] != count:
            raise ValidationError(f"{d.id}: declares {declared[d.id]} experts but layers list {count}")
        need = count * expert_bytes
        if need > d.expert_gib * GIB:
            raise ValidationError(f"{d.id}: {count} experts x {expert_bytes} B = {need / GIB:.2f} GiB "
                                  f"> expert_gib {d.expert_gib:g}")

    combine = _obj(r["combine"], "combine", {"identity"}, ignored, {"remote_return"})
    raw_expected = r["expected"]
    if not isinstance(raw_expected, dict) or not 1 <= len(raw_expected) <= MAX_PATTERNS:
        raise ValidationError(f"expected: expected an object of 1..{MAX_PATTERNS} row patterns")
    if not isinstance(raw_expected.get("R1"), dict):
        raise ValidationError("expected.R1: required")
    expected = {}
    for name, raw in raw_expected.items():
        if not isinstance(name, str) or not 1 <= len(name) <= 64:
            raise ValidationError("expected: pattern names are 1..64 characters")
        if not isinstance(raw, dict):
            ignored.append(f"expected.{name}")  # additive metadata next to the patterns (not a pattern)
            continue
        expected[name] = _expectation(name, raw, layers, top_k, ignored)
    fit = r.get("fit", {})
    if not isinstance(fit, dict):
        raise ValidationError("fit: expected object")
    digest = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return HotPlacement(
        family=text(m["family"], "model.family"), model_file=model_file, model_identity=identity,
        layers=layers, experts=experts, top_k=top_k, expert_bytes=expert_bytes, canonical_split=split,
        dense_device=dense["device"], devices=tuple(devices), layer_experts=tuple(layer_experts),
        expected=expected, combine_identity=text(combine["identity"], "combine.identity"), fit=fit,
        fingerprint=digest, ignored_keys=tuple(ignored))


def load_hot_placement(path: Path) -> HotPlacement:
    raw = Path(path).read_bytes()
    if len(raw) > MAX_PLACEMENT_BYTES:
        raise ValidationError(f"{path}: above {MAX_PLACEMENT_BYTES} bytes")
    return parse_hot_placement(json.loads(raw, object_pairs_hook=_no_duplicates))


def check_against_model(placement: HotPlacement, model: ModelParts) -> None:
    """Reject a placement made for another model geometry."""
    if placement.family != model.model_id:
        raise ValidationError(f"placement model.family {placement.family!r} is not model {model.model_id!r}")
    if placement.layers != model.layers:
        raise ValidationError(f"placement has {placement.layers} layers; the model has {model.layers}")
    g = model.moe
    if g is None:
        return
    if placement.experts != g.experts:
        raise ValidationError(f"placement has {placement.experts} experts per layer; the model has {g.experts}")
    if placement.top_k != g.top_k:
        raise ValidationError(f"placement top_k {placement.top_k}; the model routes top-{g.top_k}")
    if g.expert_bytes and placement.expert_bytes != g.expert_bytes:
        raise ValidationError(f"placement expert_bytes {placement.expert_bytes}; the model declares {g.expert_bytes}")


# --- device mapping ------------------------------------------------------------------

def resolve_devices(placement: HotPlacement, hw: Hardware, overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    """Map placement devices (and ``<server-device>/<node>`` servers) to hardware device ids.

    Order: an explicit override; a hardware device with the same id; then by role: the
    parts-host is the only hardware device whose backend is its kind; an expert tier or
    cache is the only remaining device of its backend on the parts-host's node; a server
    is the only device of the server kind's backend on its node.  A device that holds no
    expert may stay unmapped (``None``); an ambiguous or missing mapping of a used device
    is an error.
    """
    overrides = dict(overrides or {})
    names = {d.id for d in placement.devices} | {f"{placement.server_device.id}/{s.node}"
                                                  for s in placement.server_device.servers}
    for key, value in overrides.items():
        if key not in names:
            raise ValidationError(f"--placement-device {key}: unknown placement device (known: {sorted(names)})")
        try:
            hw.device(value)
        except KeyError:
            raise ValidationError(f"--placement-device {key}={value}: unknown hardware device") from None
    hw_ids = {d.id for d in hw.devices}
    out: dict[str, str | None] = {}

    def unique(candidates, what):
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None
        raise ValidationError(f"{what}: ambiguous hardware devices {candidates}; use --placement-device")

    host = placement.device(placement.dense_device)
    hid = overrides.get(host.id) or (host.id if host.id in hw_ids else
                                    unique((d.id for d in hw.devices if d.backend == host.kind), host.id))
    if hid is None:
        raise ValidationError(f"{host.id}: no hardware device with backend {host.kind}; use --placement-device")
    out[host.id] = hid
    host_node = hw.device(hid).node
    taken = {hid}
    for d in placement.devices:
        if d.role in ("parts-host", "expert-server"):
            continue
        mapped = overrides.get(d.id) or (d.id if d.id in hw_ids else None)
        if mapped is None:
            try:
                mapped = unique((x.id for x in hw.devices if x.node == host_node and x.backend == d.kind
                                 and x.id not in taken), d.id)
            except ValidationError:
                if placement.expert_count(d.id):
                    raise
                mapped = None
        if mapped is None and placement.expert_count(d.id):
            raise ValidationError(f"{d.id}: holds {placement.expert_count(d.id)} experts but maps to no hardware "
                                  "device; use --placement-device")
        if mapped is not None:
            if mapped in taken:
                raise ValidationError(f"{d.id}: hardware device {mapped} is already used by another placement device")
            taken.add(mapped)
        out[d.id] = mapped
    server = placement.server_device
    for s in server.servers:
        key = f"{server.id}/{s.node}"
        mapped = overrides.get(key)
        if mapped is None:
            mapped = unique((x.id for x in hw.devices if x.node == s.node and x.backend == server.kind), key)
        if mapped is None:
            raise ValidationError(f"{key}: no {server.kind} device on node {s.node}; use --placement-device")
        if mapped in taken:
            raise ValidationError(f"{key}: hardware device {mapped} is already used by another placement device")
        taken.add(mapped)
        out[key] = mapped
    return out


@dataclass(frozen=True)
class HotInput:
    """What ``plan_hetero`` needs for the ``dense+hot`` candidate."""
    placement: HotPlacement
    device_overrides: tuple[tuple[str, str], ...] = ()
    host_copy_bytes_per_s: int = 25_000_000_000  # declared PCIe host->device rate for expert-cache misses
    source: str = ""
