"""Device and link descriptor for heterogeneous placement.

``tensormeld/hetero-hardware-v1`` lists nodes, compute devices (each with its
own memory, effective memory bandwidth, effective matrix throughput, latency
floors, availability and the part capabilities its engine implements), the
links between nodes, and device groups (sets of devices that one engine runs
as a sharded group over a named link).

Every device and link carries ``provenance``: ``measured`` (a probe or engine
receipt), ``estimate`` (arithmetic or a data sheet) or ``declared``.  The
planner reports which inputs were not measured; a descriptor never marks a plan
qualified.

Availability: ``available``; ``preemptible`` (usable, but the owner may take it
back: the planner also reports the plan without it); ``prefill-only`` (lent
for short prefill bursts: it may hold weights and run prefill, never decode,
so decode stays on the other devices); ``unavailable``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, number, record, text

HARDWARE_SCHEMA = "tensormeld/hetero-hardware-v1"
MAX_NODES = 64
MAX_DEVICES = 128
MAX_LINKS = 256
AVAILABILITY = frozenset({"available", "preemptible", "prefill-only", "unavailable"})
PROVENANCE = frozenset({"measured", "estimate", "declared"})


@dataclass(frozen=True)
class HwNode:
    id: str
    disk_free_bytes: int
    disk_bytes_per_s: int
    model_files: tuple[str, ...]


@dataclass(frozen=True)
class HwDevice:
    id: str
    node: str
    kind: str
    backend: str
    memory_bytes: int
    reserved_bytes: int
    mem_bytes_per_s: int
    prefill_ops_per_s: int
    decode_ops_per_s: int
    step_overhead_us: float
    crossing_overhead_us: float
    availability: str
    supports: tuple[str, ...]
    provenance: str
    prefill_rates: tuple[tuple[str, int], ...] = ()

    def prefill_rate(self, requires: tuple[str, ...]) -> int:
        """Effective prefill ops/s for a part: the first capability with a fitted rate, else the default."""
        rates = dict(self.prefill_rates)
        for cap in requires:
            if cap in rates:
                return rates[cap]
        return self.prefill_ops_per_s

    @property
    def usable_bytes(self) -> int:
        return max(self.memory_bytes - self.reserved_bytes, 0)

    def supports_part(self, requires: tuple[str, ...]) -> bool:
        return "*" in self.supports or all(r in self.supports for r in requires)


@dataclass(frozen=True)
class HwLink:
    id: str
    a: str
    b: str
    bytes_per_s: int
    latency_us: float
    shared_group: str
    provenance: str

    def cost_us(self, payload: int) -> float:
        return self.latency_us + payload * 1e6 / self.bytes_per_s


@dataclass(frozen=True)
class HwGroup:
    id: str
    devices: tuple[str, ...]
    link: str
    supports: tuple[str, ...] = ("*",)

    def supports_part(self, requires: tuple[str, ...]) -> bool:
        """Capabilities the engine can run *sharded across this group* (its group modes)."""
        return "*" in self.supports or all(r in self.supports for r in requires)


@dataclass(frozen=True)
class Hardware:
    nodes: tuple[HwNode, ...]
    devices: tuple[HwDevice, ...]
    links: tuple[HwLink, ...]
    groups: tuple[HwGroup, ...]
    fingerprint: str

    def device(self, device_id: str) -> HwDevice:
        for d in self.devices:
            if d.id == device_id:
                return d
        raise KeyError(device_id)

    def node(self, node_id: str) -> HwNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def link(self, link_id: str) -> HwLink:
        for link in self.links:
            if link.id == link_id:
                return link
        raise KeyError(link_id)

    def route(self, src_node: str, dst_node: str) -> tuple[HwLink, ...]:
        """Lowest-latency chain of links between two nodes (empty if same node)."""
        if src_node == dst_node:
            return ()
        best = {src_node: 0.0}
        prev: dict[str, tuple[str, HwLink]] = {}
        heap = [(0.0, src_node)]
        while heap:
            cost, here = heapq.heappop(heap)
            if here == dst_node:
                break
            if cost > best.get(here, float("inf")):
                continue
            for link in self.links:
                if here not in (link.a, link.b):
                    continue
                there = link.b if link.a == here else link.a
                c = cost + link.latency_us + 1.0  # +1 us per hop prefers fewer hops on ties
                if c < best.get(there, float("inf")):
                    best[there] = c
                    prev[there] = (here, link)
                    heapq.heappush(heap, (c, there))
        if dst_node not in prev:
            raise ValidationError(f"no link route from {src_node} to {dst_node}")
        chain = []
        node = dst_node
        while node != src_node:
            node, link = prev[node]
            chain.append(link)
        return tuple(reversed(chain))

    def with_overrides(self, availability: dict[str, str] | None = None,
                       links: dict[str, tuple[int, float]] | None = None) -> "Hardware":
        """A copy with device availability and/or link bandwidth/latency replaced (what-if)."""
        availability = availability or {}
        links = links or {}
        for key in availability:
            self.device(key)
        for key in links:
            self.link(key)
        devices = tuple(
            HwDevice(**{**d.__dict__, "availability": availability.get(d.id, d.availability)}) for d in self.devices)
        new_links = tuple(
            HwLink(**{**link.__dict__, "bytes_per_s": links[link.id][0], "latency_us": links[link.id][1],
                      "provenance": "estimate"}) if link.id in links else link
            for link in self.links)
        tag = json.dumps({"a": availability, "l": {k: list(v) for k, v in links.items()}}, sort_keys=True)
        fingerprint = hashlib.sha256((self.fingerprint + tag).encode()).hexdigest()
        return Hardware(self.nodes, devices, new_links, self.groups, fingerprint)


def _int(value, where, minimum=0) -> int:
    return int(number(value, where, minimum, True))


def parse_hardware(data) -> Hardware:
    r = record(data, "hardware", {"schema", "nodes", "devices", "links"}, {"groups", "notes"})
    if r["schema"] != HARDWARE_SCHEMA:
        raise ValidationError(f"schema: expected {HARDWARE_SCHEMA}")
    for key, limit in (("nodes", MAX_NODES), ("devices", MAX_DEVICES), ("links", MAX_LINKS)):
        if not isinstance(r[key], list) or len(r[key]) > limit:
            raise ValidationError(f"{key}: expected a list of at most {limit}")
    nodes = []
    for i, raw in enumerate(r["nodes"]):
        n = record(raw, f"nodes[{i}]", {"id"}, {"disk_free_bytes", "disk_bytes_per_s", "model_files", "notes"})
        files = n.get("model_files", [])
        if not isinstance(files, list) or len(files) > 64:
            raise ValidationError(f"nodes[{i}].model_files: expected a bounded list")
        nodes.append(HwNode(text(n["id"], f"nodes[{i}].id"), _int(n.get("disk_free_bytes", 0), "disk_free_bytes"),
                            _int(n.get("disk_bytes_per_s", 1_000_000_000), "disk_bytes_per_s", 1),
                            tuple(text(f, "model_files[]") for f in files)))
    node_ids = {n.id for n in nodes}
    if len(node_ids) != len(nodes):
        raise ValidationError("nodes: duplicate id")
    devices = []
    for i, raw in enumerate(r["devices"]):
        d = record(raw, f"devices[{i}]",
                   {"id", "node", "kind", "backend", "memory_bytes", "mem_bytes_per_s", "prefill_ops_per_s",
                    "availability", "supports", "provenance"},
                   {"reserved_bytes", "decode_ops_per_s", "step_overhead_us", "crossing_overhead_us",
                    "prefill_rates", "notes"})
        if d["node"] not in node_ids:
            raise ValidationError(f"devices[{i}].node: unknown node {d['node']!r}")
        if d["availability"] not in AVAILABILITY:
            raise ValidationError(f"devices[{i}].availability: expected one of {sorted(AVAILABILITY)}")
        if d["provenance"] not in PROVENANCE:
            raise ValidationError(f"devices[{i}].provenance: expected one of {sorted(PROVENANCE)}")
        supports = d["supports"]
        if not isinstance(supports, list) or not 1 <= len(supports) <= 256:
            raise ValidationError(f"devices[{i}].supports: expected 1..256 capability strings or ['*']")
        prefill = _int(d["prefill_ops_per_s"], f"devices[{i}].prefill_ops_per_s", 1)
        rates = d.get("prefill_rates", {})
        if not isinstance(rates, dict) or len(rates) > 64:
            raise ValidationError(f"devices[{i}].prefill_rates: expected an object of at most 64 rates")
        rates = tuple(sorted((text(k, "prefill_rates key"), _int(v, f"prefill_rates.{k}", 1)) for k, v in rates.items()))
        devices.append(HwDevice(
            id=text(d["id"], f"devices[{i}].id"), node=d["node"], kind=text(d["kind"], "kind"),
            backend=text(d["backend"], "backend"),
            memory_bytes=_int(d["memory_bytes"], "memory_bytes", 1),
            reserved_bytes=_int(d.get("reserved_bytes", 0), "reserved_bytes"),
            mem_bytes_per_s=_int(d["mem_bytes_per_s"], "mem_bytes_per_s", 1),
            prefill_ops_per_s=prefill,
            decode_ops_per_s=_int(d.get("decode_ops_per_s", prefill), "decode_ops_per_s", 1),
            step_overhead_us=float(number(d.get("step_overhead_us", 0), "step_overhead_us")),
            crossing_overhead_us=float(number(d.get("crossing_overhead_us", 0), "crossing_overhead_us")),
            availability=d["availability"],
            supports=tuple(sorted(text(s, "supports[]") for s in supports)),
            provenance=d["provenance"], prefill_rates=rates))
    device_ids = {d.id for d in devices}
    if len(device_ids) != len(devices) or device_ids & node_ids:
        raise ValidationError("devices: duplicate id (or same id as a node)")
    links = []
    for i, raw in enumerate(r["links"]):
        link = record(raw, f"links[{i}]", {"id", "a", "b", "bytes_per_s", "latency_us", "provenance"},
                      {"shared_group", "notes"})
        if link["a"] not in node_ids or link["b"] not in node_ids or link["a"] == link["b"]:
            raise ValidationError(f"links[{i}]: endpoints must be two distinct known nodes")
        if link["provenance"] not in PROVENANCE:
            raise ValidationError(f"links[{i}].provenance: expected one of {sorted(PROVENANCE)}")
        links.append(HwLink(text(link["id"], "link.id"), link["a"], link["b"],
                            _int(link["bytes_per_s"], "bytes_per_s", 1),
                            float(number(link["latency_us"], "latency_us")),
                            str(link.get("shared_group", link["id"])), link["provenance"]))
    if len({link.id for link in links}) != len(links):
        raise ValidationError("links: duplicate id")
    groups = []
    for i, raw in enumerate(r.get("groups", [])):
        g = record(raw, f"groups[{i}]", {"id", "devices", "link"}, {"supports", "notes"})
        gsup = g.get("supports", ["*"])
        if not isinstance(gsup, list) or not 1 <= len(gsup) <= 256:
            raise ValidationError(f"groups[{i}].supports: expected 1..256 capability strings or ['*']")
        members = g["devices"]
        if not isinstance(members, list) or not 2 <= len(members) <= 16 or len(set(members)) != len(members):
            raise ValidationError(f"groups[{i}].devices: expected 2..16 distinct devices")
        for m in members:
            if m not in device_ids:
                raise ValidationError(f"groups[{i}]: unknown device {m!r}")
        if g["link"] not in {link.id for link in links}:
            raise ValidationError(f"groups[{i}].link: unknown link")
        gid = text(g["id"], "group.id")
        if gid in device_ids or gid in node_ids:
            raise ValidationError(f"groups[{i}].id: clashes with a device or node id")
        groups.append(HwGroup(gid, tuple(members), g["link"], tuple(sorted(text(x, "supports[]") for x in gsup))))
    if len({g.id for g in groups}) != len(groups):
        raise ValidationError("groups: duplicate id")
    digest = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return Hardware(tuple(nodes), tuple(devices), tuple(links), tuple(groups), digest)


def load_hardware(path: Path) -> Hardware:
    raw = Path(path).read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError(f"{path}: above {MAX_INPUT_BYTES} bytes")
    return parse_hardware(json.loads(raw, object_pairs_hook=_no_duplicates))
