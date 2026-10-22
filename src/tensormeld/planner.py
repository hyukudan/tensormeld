"""Exhaustive bounded whole-stage placement, NOT a backend execution plan.

Each selected device owns one nonempty contiguous range. All subsets and orders
are considered; adding a helper is never required. Cost is an explicitly serial
C1 decode model. There is no invented overlap, multirail sum, or token throughput.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
from typing import Any

from .schema import Scenario, ValidationError


class NoRoute(ValueError):
    pass


def transfer(scenario: Scenario, source: str, target: str, size: int, coordinator: str) -> dict:
    if source == target:
        return {"source": source, "target": target, "bytes": size, "cost_ms": 0.0, "hops": []}
    path = [source, target]
    if scenario.route_mode == "via_coordinator" and coordinator not in (source, target):
        path = [source, coordinator, target]
    hops = []
    for a, b in zip(path, path[1:]):
        possible = [l for l in scenario.links if l.source == a and l.target == b]
        if not possible:
            raise NoRoute(f"no declared directed transfer profile: {a} -> {b}")
        chosen = min(possible, key=lambda l: (l.cost_ms(size), l.id))
        hops.append({"link": chosen.id, "source": a, "target": b,
                     "physical_group": chosen.physical_group, "cost_ms": chosen.cost_ms(size)})
    return {"source": source, "target": target, "bytes": size,
            "cost_ms": sum(h["cost_ms"] for h in hops), "hops": hops}


def _evaluate(s: Scenario, ranges: list[tuple[str, int, int]], coordinator: str) -> tuple[dict | None, str]:
    devices = {d.id: d for d in s.devices}
    pools = {p.id: p for p in s.pools}
    pool_used = {p.id: 0 for p in s.pools}
    stages = []
    compute = 0.0
    for device_id, start, end in ranges:
        device = devices[device_id]
        block = s.stages[start:end]
        if any(device_id not in stage.decode_ms for stage in block):
            return None, "missing_execution_profile"
        resident = sum(stage.weights_bytes + stage.state_bytes for stage in block)
        workspace = max(stage.workspace_bytes for stage in block)
        total = resident + workspace + device.runtime_bytes
        pool_used[device.pool] += total
        block_ms = sum(stage.decode_ms[device_id] for stage in block)
        compute += block_ms
        stages.append({"device": device_id, "node": device.node, "backend": device.backend,
                       "stage_start": start, "stage_end_exclusive": end,
                       "stage_ids": [stage.id for stage in block], "resident_bytes": resident,
                       "workspace_peak_bytes": workspace, "runtime_bytes": device.runtime_bytes,
                       "estimated_compute_ms": block_ms})
    if any(used > pools[pid].budget_bytes for pid, used in pool_used.items()):
        return None, "memory_budget_exceeded"
    boundaries = []
    try:
        for current, following in zip(ranges, ranges[1:]):
            boundaries.append(transfer(s, current[0], following[0],
                                       s.stages[current[2] - 1].output_bytes, coordinator))
        feedback = transfer(s, ranges[-1][0], ranges[0][0], s.feedback_bytes, coordinator)
    except NoRoute:
        return None, "missing_transfer_profile"
    comm = sum(b["cost_ms"] for b in boundaries) + feedback["cost_ms"]
    return {"coordinator": coordinator, "ranges": stages, "boundaries": boundaries,
            "feedback": feedback, "pool_usage_bytes": pool_used,
            "estimated_compute_ms": compute, "estimated_communication_ms": comm,
            "estimated_serial_decode_ms": compute + comm}, "feasible"


def plan(s: Scenario, enabled_devices: list[str] | None = None) -> dict[str, Any]:
    known = {d.id for d in s.devices}
    if enabled_devices is None:
        enabled_devices = sorted(known)
    if not enabled_devices or len(set(enabled_devices)) != len(enabled_devices) or not set(enabled_devices) <= known:
        raise ValidationError("enabled_devices must be a nonempty, unique subset of known devices")
    coordinators = sorted(set(s.coordinators) & set(enabled_devices))
    if not coordinators:
        raise ValidationError("no enabled coordinator")
    n = len(s.stages)
    best = None
    best_key = None
    counts: Counter = Counter()
    for count in range(1, min(len(enabled_devices), n) + 1):
        for order in permutations(sorted(enabled_devices), count):
            for cuts in combinations(range(1, n), count - 1):
                positions = (0, *cuts, n)
                ranges = [(device, positions[i], positions[i + 1]) for i, device in enumerate(order)]
                for coordinator in coordinators:
                    candidate, reason = _evaluate(s, ranges, coordinator)
                    counts[reason] += 1
                    if candidate is None:
                        continue
                    key = (candidate["estimated_serial_decode_ms"], count, order, cuts, coordinator)
                    if best_key is None or key < best_key:
                        best, best_key = candidate, key
    warnings = [
        "Analytical serial decode estimate only; not a measured inference result.",
        "Placement is advisory: no backend adapter has qualified or executed it.",
        "Prefill, overlap, concurrent requests, expert-level splits and model parsing are not implemented.",
        "Pool budgets must already exclude OS/display/driver safety reservations.",
    ]
    if s.provenance == "synthetic":
        warnings.insert(0, "SYNTHETIC INPUTS: not a GLM/DeepSeek benchmark or hardware prediction.")
    return {"schema_version": 1, "scenario": s.name, "scenario_sha256": s.fingerprint,
            "provenance": s.provenance, "workload": s.workload, "qualified": False,
            "status": "advisory_candidate" if best else "no_feasible_candidate",
            "route_mode": s.route_mode, "search_counts": dict(sorted(counts.items())),
            "best": best, "warnings": warnings}
