"""Static native-adapter representability contract.

This module answers a narrow question: can a declared adapter represent a planner
candidate *without changing its placement semantics*? It does not launch a worker,
reserve memory, qualify a model, or make a plan executable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config_v2 import Config
from .planning_contract import PlanningInput
from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, items, number, record, text, unique

MAX_ADAPTER_DEVICES = 128
MAX_ADAPTER_NODES = 64
MAX_STRATEGIES = 16
MAX_ROUTE_MODES = 8


@dataclass(frozen=True)
class AdapterDevice:
    id: str
    node: str
    backend: str


@dataclass(frozen=True)
class AdapterCapabilities:
    adapter_id: str
    engine: str
    engine_revision: str
    exact_owner_binding: bool
    explicit_unit_ranges: bool
    mixed_backends: bool
    remote_compute: bool
    coordinator_outside_compute: bool
    max_compute_devices: int
    max_compute_nodes: int
    max_segments: int
    strategies: frozenset[str]
    route_modes: frozenset[str]
    coordinator_nodes: frozenset[str]
    devices: tuple[AdapterDevice, ...]
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> "AdapterCapabilities":
        r = record(data, "adapter capabilities", {
            "adapter_schema", "adapter_id", "engine", "engine_revision", "placement",
            "route_modes", "coordinator_nodes", "devices",
        })
        if r["adapter_schema"] != "tensormeld/adapter-capabilities-v1":
            raise ValidationError("adapter_schema: expected tensormeld/adapter-capabilities-v1")
        p = record(r["placement"], "placement", {
            "strategies", "exact_owner_binding", "explicit_unit_ranges", "mixed_backends",
            "remote_compute", "coordinator_outside_compute", "max_compute_devices",
            "max_compute_nodes", "max_segments",
        })
        bool_fields = (
            "exact_owner_binding", "explicit_unit_ranges", "mixed_backends",
            "remote_compute", "coordinator_outside_compute",
        )
        for field in bool_fields:
            if type(p[field]) is not bool:
                raise ValidationError(f"placement.{field}: expected boolean")
        strategies = frozenset(
            text(x, "placement.strategies[]")
            for x in items(p["strategies"], "placement.strategies", MAX_STRATEGIES, 1)
        )
        unique(list(strategies), "placement.strategies")
        route_modes = frozenset(
            text(x, "route_modes[]")
            for x in items(r["route_modes"], "route_modes", MAX_ROUTE_MODES, 1)
        )
        unique(list(route_modes), "route_modes")
        if not route_modes <= {"direct", "via_coordinator"}:
            raise ValidationError("route_modes: unsupported route mode")
        coordinators = tuple(
            text(x, "coordinator_nodes[]")
            for x in items(r["coordinator_nodes"], "coordinator_nodes", MAX_ADAPTER_NODES, 1)
        )
        unique(list(coordinators), "coordinator_nodes")
        devices = []
        for i, raw in enumerate(items(r["devices"], "devices", MAX_ADAPTER_DEVICES, 1)):
            d = record(raw, f"devices[{i}]", {"id", "node", "backend"})
            devices.append(
                AdapterDevice(
                    text(d["id"], "device.id"),
                    text(d["node"], "device.node"),
                    text(d["backend"], "device.backend"),
                )
            )
        unique([d.id for d in devices], "devices")
        digest = hashlib.sha256(
            json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return cls(
            text(r["adapter_id"], "adapter_id"),
            text(r["engine"], "engine"),
            text(r["engine_revision"], "engine_revision"),
            p["exact_owner_binding"],
            p["explicit_unit_ranges"],
            p["mixed_backends"],
            p["remote_compute"],
            p["coordinator_outside_compute"],
            int(number(p["max_compute_devices"], "max_compute_devices", 1, True)),
            int(number(p["max_compute_nodes"], "max_compute_nodes", 1, True)),
            int(number(p["max_segments"], "max_segments", 1, True)),
            strategies,
            route_modes,
            frozenset(coordinators),
            tuple(devices),
            digest,
        )


def load_adapter_capabilities(path: str | Path) -> AdapterCapabilities:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("adapter capabilities exceed 2 MiB")
    try:
        return AdapterCapabilities.parse(json.loads(raw, object_pairs_hook=_no_duplicates))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid adapter capabilities JSON: {exc}") from exc


def validate_candidate_representability(
    config: Config,
    planning: PlanningInput,
    candidate: dict[str, Any],
    adapter: AdapterCapabilities,
) -> dict[str, Any]:
    """Return whether an adapter can represent a whole-block candidate exactly.

    This deliberately does not call the adapter or trust its self-description as
    qualification evidence. A REPRESENTABLE result remains non-executable.
    """
    reasons: list[dict[str, str]] = []

    def reject(code: str, detail: str) -> None:
        reasons.append({"code": code, "detail": detail})

    required = {
        "plan_sha256", "coordinator_node", "compute_nodes", "compute_devices", "segments",
        "unit_ids", "owners", "routes", "executable", "qualified",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValidationError(f"candidate: missing required fields {missing}")
    if candidate["executable"] is not False or candidate["qualified"] is not False:
        raise ValidationError(
            "candidate must remain non-executable and non-qualified before adapter validation"
        )
    if candidate["unit_ids"] != [u.id for u in planning.units]:
        raise ValidationError("candidate.unit_ids do not match planning input")
    if len(candidate["owners"]) != len(planning.units):
        raise ValidationError("candidate.owners length does not match planning input")

    config_devices = {d.id: d for d in config.devices}
    adapter_devices = {d.id: d for d in adapter.devices}
    compute_devices = tuple(candidate["compute_devices"])
    compute_nodes = tuple(candidate["compute_nodes"])
    segments = tuple(candidate["segments"])

    if "whole_blocks" not in adapter.strategies:
        reject("UNSUPPORTED_STRATEGY", "adapter does not declare whole_blocks")
    if not adapter.exact_owner_binding:
        reject(
            "EXACT_OWNER_BINDING_UNSUPPORTED",
            "adapter cannot guarantee requested device ownership",
        )
    if not adapter.explicit_unit_ranges:
        reject(
            "EXPLICIT_UNIT_RANGES_UNSUPPORTED",
            "adapter cannot pin explicit contiguous unit ranges",
        )
    if planning.route_mode not in adapter.route_modes:
        reject("ROUTE_MODE_UNSUPPORTED", f"adapter does not declare {planning.route_mode}")
    if len(compute_devices) > adapter.max_compute_devices:
        reject(
            "ADAPTER_DEVICE_LIMIT",
            f"candidate uses {len(compute_devices)} devices; adapter limit is {adapter.max_compute_devices}",
        )
    if len(compute_nodes) > adapter.max_compute_nodes:
        reject(
            "ADAPTER_NODE_LIMIT",
            f"candidate uses {len(compute_nodes)} nodes; adapter limit is {adapter.max_compute_nodes}",
        )
    if len(segments) > adapter.max_segments:
        reject(
            "ADAPTER_SEGMENT_LIMIT",
            f"candidate uses {len(segments)} segments; adapter limit is {adapter.max_segments}",
        )
    if len(compute_nodes) > 1 and not adapter.remote_compute:
        reject("REMOTE_COMPUTE_UNSUPPORTED", "candidate spans multiple compute nodes")
    if candidate["coordinator_node"] not in adapter.coordinator_nodes:
        reject(
            "COORDINATOR_UNSUPPORTED",
            f"coordinator {candidate['coordinator_node']} is not exposed by adapter",
        )
    if (
        candidate["coordinator_node"] not in compute_nodes
        and not adapter.coordinator_outside_compute
    ):
        reject(
            "EXTERNAL_COORDINATOR_UNSUPPORTED",
            "coordinator is outside the compute-owner set",
        )

    backends = set()
    for dev_id in compute_devices:
        cfg = config_devices.get(dev_id)
        exposed = adapter_devices.get(dev_id)
        if cfg is None:
            raise ValidationError(
                f"candidate references unknown config device {dev_id}"
            )
        if exposed is None:
            reject("DEVICE_NOT_EXPOSED", f"adapter does not expose {dev_id}")
            continue
        if exposed.node != cfg.node:
            reject(
                "DEVICE_NODE_MISMATCH",
                f"{dev_id}: adapter node {exposed.node} != config node {cfg.node}",
            )
        if exposed.backend != cfg.backend:
            reject(
                "DEVICE_BACKEND_MISMATCH",
                f"{dev_id}: adapter backend {exposed.backend} != config backend {cfg.backend}",
            )
        backends.add(cfg.backend)
    if len(backends) > 1 and not adapter.mixed_backends:
        reject(
            "MIXED_BACKENDS_UNSUPPORTED",
            "candidate requires more than one backend family",
        )

    expected_segments = []
    for i, owner in enumerate(candidate["owners"]):
        if owner not in compute_devices:
            raise ValidationError(
                f"candidate owner {owner} is absent from compute_devices"
            )
        if not expected_segments or expected_segments[-1]["device"] != owner:
            expected_segments.append(
                {"device": owner, "first_unit": i, "last_unit_exclusive": i + 1}
            )
        else:
            expected_segments[-1]["last_unit_exclusive"] = i + 1
    if list(segments) != expected_segments:
        raise ValidationError(
            "candidate.segments are not the exact contiguous ranges implied by owners"
        )

    status = "REPRESENTABLE" if not reasons else "REJECTED"
    report_core = {
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "plan_sha256": candidate["plan_sha256"],
        "planning_input_sha256": planning.fingerprint,
        "config_sha256": config.fingerprint,
        "status": status,
        "exact": status == "REPRESENTABLE",
        "reasons": reasons,
        "qualified": False,
        "executable": False,
    }
    report_core["report_sha256"] = hashlib.sha256(
        json.dumps(report_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "result_schema": "tensormeld/adapter-representability-v1",
        **report_core,
    }
