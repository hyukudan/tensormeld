"""Strict synthetic multi-pool model for the v2 planner, NOT an execution manifest."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .config_v2 import Config
from .schema import ValidationError, MAX_INPUT_BYTES, _no_duplicates, items, number, record, text, unique

MAX_UNITS = 256
MAX_LINKS = 2048


@dataclass(frozen=True)
class UnitPlacement:
    device: str
    decode_us: int
    resident: dict[str, int]
    workspace: dict[str, int]


@dataclass(frozen=True)
class Unit:
    id: str
    output_bytes: int
    placements: dict[str, UnitPlacement]


@dataclass(frozen=True)
class TransferLink:
    id: str
    source: str
    target: str
    bandwidth: int
    latency_us: int
    physical_group: str

    def cost_us(self, size: int) -> int:
        return self.latency_us + (size * 1_000_000 + self.bandwidth - 1) // self.bandwidth


@dataclass(frozen=True)
class PlanningInput:
    manifest_ref: str
    context_tokens: int
    max_output_tokens: int
    units: tuple[Unit, ...]
    worker_resident: dict[str, dict[str, int]]
    coordinator_resident: dict[str, dict[str, int]]
    links: tuple[TransferLink, ...]
    route_mode: str
    feedback_bytes: int
    fingerprint: str

    @classmethod
    def parse(cls, data: Any, config: Config) -> 'PlanningInput':
        r = record(data, 'planning input', {
            'planning_schema', 'provenance', 'manifest_ref', 'workload', 'units',
            'worker_resident', 'coordinator_resident', 'links', 'route_mode', 'feedback_bytes'})
        if r['planning_schema'] != 'tensormeld/planning-v1' or r['provenance'] != 'synthetic':
            raise ValidationError('planning_schema/provenance: expected tensormeld/planning-v1 and synthetic')
        w = record(r['workload'], 'workload', {'phase', 'context_tokens', 'max_output_tokens', 'concurrency'})
        if w['phase'] != 'decode' or type(w['concurrency']) is not int or w['concurrency'] != 1:
            raise ValidationError('plan-v2 supports only synthetic C1 decode workloads')
        context = int(number(w['context_tokens'], 'context_tokens', 1, True))
        output = int(number(w['max_output_tokens'], 'max_output_tokens', 1, True))
        if output > context:
            raise ValidationError('max_output_tokens exceeds context_tokens')
        devices = {d.id: d for d in config.devices}
        pools = {p.id: p for p in config.pools}
        nodes = {n.id: n for n in config.nodes}

        def demand(raw: Any, where: str, owner: str) -> dict[str, int]:
            if not isinstance(raw, dict) or len(raw) > len(pools):
                raise ValidationError(f'{where}: expected a bounded physical-pool byte map')
            result = {}
            for pool, amount in raw.items():
                if pool not in pools or pools[pool].node != owner:
                    raise ValidationError(f'{where}: pool {pool!r} is not local to owner {owner!r}')
                result[pool] = int(number(amount, where, 0, True))
            return result

        worker = record(r['worker_resident'], 'worker_resident', set(devices))
        worker = {d: demand(v, f'worker_resident.{d}', devices[d].node) for d, v in worker.items()}
        coordinators = record(r['coordinator_resident'], 'coordinator_resident', set(config.selection.coordinator.allowed_nodes))
        coordinators = {n: demand(v, f'coordinator_resident.{n}', n) for n, v in coordinators.items()}
        units = []
        for raw in items(r['units'], 'units', MAX_UNITS):
            u = record(raw, 'unit', {'id', 'output_bytes', 'placements'})
            placements = u['placements']
            if not isinstance(placements, dict) or not placements or len(placements) > len(devices):
                raise ValidationError('unit.placements: expected known device profiles')
            profiles = {}
            for dev, v in placements.items():
                if dev not in devices:
                    raise ValidationError(f'unit.placements: unknown device {dev}')
                p = record(v, f'placement.{dev}', {'decode_us', 'resident_bytes', 'workspace_bytes'})
                profiles[dev] = UnitPlacement(dev, int(number(p['decode_us'], 'decode_us', 0, True)),
                    demand(p['resident_bytes'], 'resident_bytes', devices[dev].node),
                    demand(p['workspace_bytes'], 'workspace_bytes', devices[dev].node))
            units.append(Unit(text(u['id'], 'unit.id'), int(number(u['output_bytes'], 'output_bytes', 0, True)), profiles))
        unique([u.id for u in units], 'units')
        endpoints = {'device:' + d for d in devices} | {'node:' + n for n in nodes}
        links = []
        for raw in items(r['links'], 'links', MAX_LINKS, 0):
            l = record(raw, 'link', {'id', 'source', 'target', 'payload_bytes_per_s', 'latency_us', 'physical_group'})
            text(l['source'], 'link.source'); text(l['target'], 'link.target')
            if l['source'] not in endpoints or l['target'] not in endpoints or l['source'] == l['target']:
                raise ValidationError('link endpoints must be distinct known device:/node: identities')
            links.append(TransferLink(text(l['id'], 'link.id'), l['source'], l['target'],
                int(number(l['payload_bytes_per_s'], 'payload_bytes_per_s', 1, True)),
                int(number(l['latency_us'], 'latency_us', 0, True)), text(l['physical_group'], 'physical_group')))
        unique([l.id for l in links], 'links')
        if r['route_mode'] not in ('direct', 'via_coordinator'):
            raise ValidationError('route_mode: expected direct or via_coordinator')
        digest = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        return cls(text(r['manifest_ref'], 'manifest_ref'), context, output, tuple(units), worker, coordinators,
                   tuple(links), r['route_mode'], int(number(r['feedback_bytes'], 'feedback_bytes', 0, True)), digest)


def load_planning_input(path: str | Path, config: Config) -> PlanningInput:
    with Path(path).open('rb') as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError('planning input exceeds 2 MiB')
    try:
        return PlanningInput.parse(json.loads(raw, object_pairs_hook=_no_duplicates), config)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f'invalid planning JSON: {exc}') from exc
