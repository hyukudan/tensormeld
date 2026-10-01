"""Strict, bounded input contracts for the analytical C1 decode planner.

A profile applies only to the exact workload in its scenario. Memory budgets are
allocatable pool limits AFTER the user's OS/display/driver safety reservation.
No hardware compatibility is inferred from these manually supplied profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 2 * 1024 * 1024


class ValidationError(ValueError):
    """Untrusted or inconsistent scenario input."""


def record(value: Any, where: str, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{where}: expected object")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing or extra:
        raise ValidationError(f"{where}: missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValidationError(f"{where}: expected nonempty string of at most 256 characters")
    return value


def number(value: Any, where: str, minimum: float = 0, integer: bool = False) -> int | float:
    if type(value) not in (int, float) or (integer and type(value) is not int):
        raise ValidationError(f"{where}: expected {'integer' if integer else 'number'}")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or value > 2**63 - 1:
        raise ValidationError(f"{where}: expected finite value in [{minimum}, 2^63-1]")
    return value


def items(value: Any, where: str, maximum: int, minimum: int = 1) -> list:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValidationError(f"{where}: expected {minimum}..{maximum} items")
    return value


def unique(values: list[str], where: str) -> None:
    if len(set(values)) != len(values):
        raise ValidationError(f"{where}: duplicate identifiers")


@dataclass(frozen=True)
class Pool:
    id: str
    node: str
    budget_bytes: int


@dataclass(frozen=True)
class Device:
    id: str
    node: str
    pool: str
    backend: str
    runtime_bytes: int


@dataclass(frozen=True)
class Link:
    id: str
    source: str
    target: str
    payload_bytes_per_s: float
    fixed_latency_us: float
    physical_group: str

    def cost_ms(self, payload_bytes: int) -> float:
        return self.fixed_latency_us / 1000 + payload_bytes * 1000 / self.payload_bytes_per_s


@dataclass(frozen=True)
class Stage:
    id: str
    weights_bytes: int
    state_bytes: int
    workspace_bytes: int
    output_bytes: int
    decode_ms: dict[str, float]


@dataclass(frozen=True)
class Scenario:
    name: str
    provenance: str
    workload: dict[str, Any]
    pools: tuple[Pool, ...]
    devices: tuple[Device, ...]
    links: tuple[Link, ...]
    route_mode: str
    coordinators: tuple[str, ...]
    stages: tuple[Stage, ...]
    feedback_bytes: int
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> Scenario:
        root = record(data, "scenario", {
            "schema_version", "name", "provenance", "workload", "pools", "devices",
            "links", "route_mode", "coordinators", "stages", "feedback_bytes",
        })
        if type(root["schema_version"]) is not int or root["schema_version"] != 1:
            raise ValidationError("schema_version: only 1 is supported")
        name = text(root["name"], "name")
        provenance = root["provenance"]
        if provenance not in ("synthetic", "measured"):
            raise ValidationError("provenance: use synthetic or measured; guesses are synthetic")
        w = record(root["workload"], "workload", {
            "model_revision", "quantization", "context_tokens", "concurrency", "phase",
        })
        text(w["model_revision"], "workload.model_revision")
        text(w["quantization"], "workload.quantization")
        number(w["context_tokens"], "workload.context_tokens", 1, True)
        number(w["concurrency"], "workload.concurrency", 1, True)
        if w["concurrency"] != 1 or w["phase"] != "decode":
            raise ValidationError("M0 planner supports only phase=decode, concurrency=1")
        pools = []
        for i, item in enumerate(items(root["pools"], "pools", 8)):
            p = record(item, f"pools[{i}]", {"id", "node", "budget_bytes"})
            pools.append(Pool(text(p["id"], "pool.id"), text(p["node"], "pool.node"),
                              number(p["budget_bytes"], "pool.budget_bytes", 1, True)))
        unique([p.id for p in pools], "pools")
        pool_map = {p.id: p for p in pools}
        devices = []
        for i, item in enumerate(items(root["devices"], "devices", 3)):
            d = record(item, f"devices[{i}]", {"id", "node", "pool", "backend", "runtime_bytes"})
            device = Device(text(d["id"], "device.id"), text(d["node"], "device.node"),
                            text(d["pool"], "device.pool"), text(d["backend"], "device.backend"),
                            number(d["runtime_bytes"], "device.runtime_bytes", 0, True))
            if device.pool not in pool_map or pool_map[device.pool].node != device.node:
                raise ValidationError(f"device {device.id}: pool must exist on the same node")
            devices.append(device)
        unique([d.id for d in devices], "devices")
        ids = {d.id for d in devices}
        links = []
        for i, item in enumerate(items(root["links"], "links", 32, 0)):
            l = record(item, f"links[{i}]", {
                "id", "source", "target", "payload_bytes_per_s", "fixed_latency_us", "physical_group",
            })
            link = Link(text(l["id"], "link.id"), text(l["source"], "link.source"),
                        text(l["target"], "link.target"),
                        number(l["payload_bytes_per_s"], "link.payload_bytes_per_s", 1),
                        number(l["fixed_latency_us"], "link.fixed_latency_us", 0),
                        text(l["physical_group"], "link.physical_group"))
            if link.source not in ids or link.target not in ids or link.source == link.target:
                raise ValidationError(f"link {link.id}: expected distinct known source and target devices")
            links.append(link)
        unique([l.id for l in links], "links")
        route_mode = root["route_mode"]
        if route_mode not in ("direct", "via_coordinator"):
            raise ValidationError("route_mode: expected direct or via_coordinator")
        coordinators = tuple(text(c, "coordinator") for c in items(root["coordinators"], "coordinators", 3))
        unique(list(coordinators), "coordinators")
        if not set(coordinators) <= ids:
            raise ValidationError("coordinators: unknown device")
        stages = []
        for i, item in enumerate(items(root["stages"], "stages", 128)):
            s = record(item, f"stages[{i}]", {
                "id", "weights_bytes", "state_bytes", "workspace_bytes", "output_bytes", "decode_ms",
            })
            costs = s["decode_ms"]
            if not isinstance(costs, dict) or not costs or not set(costs) <= ids:
                raise ValidationError("stage.decode_ms: expected nonempty map of known device IDs")
            stages.append(Stage(
                text(s["id"], "stage.id"),
                number(s["weights_bytes"], "stage.weights_bytes", 0, True),
                number(s["state_bytes"], "stage.state_bytes", 0, True),
                number(s["workspace_bytes"], "stage.workspace_bytes", 0, True),
                number(s["output_bytes"], "stage.output_bytes", 1, True),
                {d: number(v, f"stage.decode_ms.{d}", 0.000001) for d, v in costs.items()},
            ))
        unique([s.id for s in stages], "stages")
        feedback = number(root["feedback_bytes"], "feedback_bytes", 1, True)
        encoded = json.dumps(root, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return cls(name, provenance, dict(w), tuple(pools), tuple(devices), tuple(links),
                   route_mode, coordinators, tuple(stages), feedback, hashlib.sha256(encoded).hexdigest())


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: str | Path) -> Scenario:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("scenario exceeds 2 MiB limit")
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    return Scenario.parse(data)
