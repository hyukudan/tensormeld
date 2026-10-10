"""Bounded advisory placement over adapter-declared legal model units.

This planner searches only tensor-resident ownership. It does not model compute time,
persistent state migration, workspace, staging, communication or native execution.
Owner changes are legal only after an explicit adapter-declared cut_after boundary.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable

from .config_v2 import Config
from .legal_model_units import LegalModelUnitsProfile
from .schema import ValidationError
from .selection import resolve_candidates

RESULT_SCHEMA = "tensormeld/legal-unit-planning-result-v1"


class _BudgetExpired(Exception):
    pass


@dataclass
class _Budget:
    limit: int
    deadline: float
    clock: Callable[[], float]
    used: int = 0
    reason: str | None = None

    def consume(self) -> None:
        if self.used >= self.limit:
            self.reason = "work_limit"
            raise _BudgetExpired
        if self.clock() >= self.deadline:
            self.reason = "deadline"
            raise _BudgetExpired
        self.used += 1


@dataclass(frozen=True)
class _State:
    owners: tuple[str, ...]
    pool_usage: tuple[tuple[str, int], ...]

    def usage_map(self) -> dict[str, int]:
        return dict(self.pool_usage)


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def plan_legal_units(
    config: Config,
    legal_units: LegalModelUnitsProfile,
    profile_name: str | None = None,
    top_k: int = 5,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValidationError("top_k must be an integer in 1..20")
    if not legal_units.units:
        raise ValidationError("legal-unit planner requires non-empty legal units")

    start = clock()
    budget = _Budget(
        config.planning.candidate_limit,
        start + config.planning.deadline_ms / 1000,
        clock,
    )
    selection = resolve_candidates(config, profile_name)
    devices = {d["id"]: d for d in selection["eligible_compute_devices"]}
    if not devices:
        raise ValidationError("legal-unit planner has no eligible compute devices")
    device_ids = tuple(sorted(devices))
    budgets = selection["physical_pool_budgets"]
    required_devices = set(selection["required_devices"])
    required_nodes = set(selection["required_nodes"])
    required_local = set(selection["required_any_local_gpu"])
    rejections: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    complete_candidates = 0

    for unit in legal_units.units:
        if not set(unit.allowed_devices) <= set(devices):
            # Tensor evidence may mention configured devices filtered out by current policy.
            pass
        legal_now = set(unit.allowed_devices) & set(devices)
        if not legal_now:
            return {
                "result_schema": RESULT_SCHEMA,
                "status": "NO_CANDIDATE_IN_SEARCH_SPACE",
                "config_sha256": config.fingerprint,
                "legal_model_units_sha256": legal_units.fingerprint,
                "profile": selection["profile"],
                "objective": "resident_capacity_only",
                "qualified": False,
                "executable": False,
                "best": None,
                "candidates": [],
                "search": {
                    "complete": True,
                    "reason": None,
                    "work_units": 0,
                    "work_limit": config.planning.candidate_limit,
                    "deadline_ms": config.planning.deadline_ms,
                    "elapsed_ms": 0,
                    "complete_candidates_evaluated": 0,
                    "rejections": {"unit_has_no_policy_eligible_device": 1},
                },
                "warnings": [
                    "Legal-unit planning is advisory tensor-resident capacity analysis only.",
                    "No compute, state, workspace, staging, transfer or performance cost is modeled.",
                ],
            }

    def fits(usage: dict[str, int]) -> bool:
        for pool, amount in usage.items():
            limit = budgets.get(pool)
            if limit is None:
                rejections["unknown_pool_budget"] += 1
                return False
            if amount > limit:
                rejections["pool_budget_exceeded"] += 1
                return False
        return True

    def candidate_nodes(owners: tuple[str, ...]) -> set[str]:
        return {devices[d]["node"] for d in set(owners)}

    def final_requirements(owners: tuple[str, ...]) -> bool:
        used = set(owners)
        nodes = candidate_nodes(owners)
        if not required_devices <= used:
            rejections["required_device_missing"] += 1
            return False
        if not required_nodes <= nodes:
            rejections["required_node_missing"] += 1
            return False
        if required_local and not used & required_local:
            rejections["required_local_gpu_missing"] += 1
            return False
        if len(used) < selection["min_compute_devices"]:
            rejections["min_compute_devices_unmet"] += 1
            return False
        if len(nodes) < selection["min_compute_nodes"]:
            rejections["min_compute_nodes_unmet"] += 1
            return False
        return True

    def score(candidate: dict[str, Any]) -> tuple:
        return (
            candidate["compute_device_count"],
            candidate["compute_node_count"],
            candidate["max_pool_utilization_ppm"],
            candidate["plan_sha256"],
        )

    def finish(state: _State) -> None:
        nonlocal complete_candidates
        owners = state.owners
        if not final_requirements(owners):
            return
        complete_candidates += 1
        used = sorted(set(owners))
        nodes = sorted(candidate_nodes(owners))
        usage = state.usage_map()
        segments = []
        for i, owner in enumerate(owners):
            if not segments or segments[-1]["device"] != owner:
                segments.append({
                    "device": owner,
                    "first_unit": i,
                    "last_unit_exclusive": i + 1,
                })
            else:
                segments[-1]["last_unit_exclusive"] = i + 1

        utilization = []
        for pool, amount in usage.items():
            limit = budgets[pool]
            if limit is None or limit <= 0:
                continue
            utilization.append((amount * 1_000_000 + limit - 1) // limit)
        core = {
            "compute_nodes": nodes,
            "compute_devices": used,
            "compute_node_count": len(nodes),
            "compute_device_count": len(used),
            "owners": list(owners),
            "segments": segments,
            "unit_ids": [unit.id for unit in legal_units.units],
            "pool_tensor_resident_bytes": dict(sorted(usage.items())),
            "max_pool_utilization_ppm": max(utilization, default=0),
            "qualified": False,
            "executable": False,
        }
        identity = {
            "config_sha256": config.fingerprint,
            "legal_model_units_sha256": legal_units.fingerprint,
            "profile": selection["profile"],
            "plan": core,
        }
        core["plan_sha256"] = _sha256(identity)
        if all(x["plan_sha256"] != core["plan_sha256"] for x in candidates):
            candidates.append(core)
            candidates.sort(key=score)
            del candidates[top_k:]

    root = _State((), ())
    stack: list[tuple[int, _State, tuple[str, ...], int]] = []

    def choices_for(index: int, state: _State) -> tuple[str, ...]:
        unit = legal_units.units[index]
        allowed = set(unit.allowed_devices) & set(device_ids)
        if index > 0 and state.owners:
            previous_owner = state.owners[-1]
            if not legal_units.units[index - 1].cut_after:
                return (previous_owner,) if previous_owner in allowed else ()
        ordered = []
        if state.owners and state.owners[-1] in allowed:
            ordered.append(state.owners[-1])
        ordered.extend(d for d in device_ids if d in allowed and d not in ordered)
        return tuple(ordered)

    try:
        stack.append((0, root, choices_for(0, root), 0))
        while stack:
            index, state, choices, pos = stack.pop()
            if pos >= len(choices):
                continue
            stack.append((index, state, choices, pos + 1))
            dev = choices[pos]
            budget.consume()

            unit = legal_units.units[index]
            used = set(state.owners) | {dev}
            nodes = {devices[d]["node"] for d in used}
            if len(used) > selection["max_compute_devices"]:
                rejections["max_compute_devices_exceeded"] += 1
                continue
            if len(nodes) > selection["max_compute_nodes"]:
                rejections["max_compute_nodes_exceeded"] += 1
                continue

            remaining = len(legal_units.units) - index - 1
            if len(required_devices - used) > remaining:
                rejections["required_device_unreachable"] += 1
                continue
            if len(required_nodes - nodes) > remaining:
                rejections["required_node_unreachable"] += 1
                continue

            usage = state.usage_map()
            pool = devices[dev]["pool"]
            resident = unit.hard_resident_bytes + unit.reclaimable_file_backed_bytes
            usage[pool] = usage.get(pool, 0) + resident
            if not fits(usage):
                continue

            next_state = _State(
                state.owners + (dev,),
                tuple(sorted(usage.items())),
            )
            if index + 1 == len(legal_units.units):
                finish(next_state)
                continue
            next_choices = choices_for(index + 1, next_state)
            if not next_choices:
                rejections["cut_or_device_constraint_blocks_progress"] += 1
                continue
            stack.append((index + 1, next_state, next_choices, 0))
    except _BudgetExpired:
        pass

    incomplete = budget.reason is not None
    status = (
        "SEARCH_INCOMPLETE"
        if incomplete
        else "CANDIDATES_FOUND"
        if candidates
        else "NO_CANDIDATE_IN_SEARCH_SPACE"
    )
    return {
        "result_schema": RESULT_SCHEMA,
        "status": status,
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal_units.fingerprint,
        "profile": selection["profile"],
        "objective": "resident_capacity_only",
        "qualified": False,
        "executable": False,
        "best": candidates[0] if candidates else None,
        "candidates": candidates,
        "search": {
            "complete": not incomplete,
            "reason": budget.reason,
            "work_units": budget.used,
            "work_limit": budget.limit,
            "deadline_ms": config.planning.deadline_ms,
            "elapsed_ms": max(0, int((clock() - start) * 1000)),
            "complete_candidates_evaluated": complete_candidates,
            "retained_candidates": len(candidates),
            "rejections": dict(sorted(rejections.items())),
        },
        "warnings": [
            "Advisory tensor-resident capacity planner only; no native execution path is enabled.",
            "Owner changes are permitted only after explicit adapter-declared cut_after boundaries.",
            "Unit allowed_devices come from explicit legal-unit evidence and current installation policy.",
            "Memory accounts only tensor hard-resident plus reclaimable/file-backed bytes on the owning physical pool.",
            "Persistent state, workspace, staging, transfer and compute costs are not assigned per legal unit yet.",
            "Reclaimable/file-backed bytes remain counted in full for conservative resident capacity.",
            "Ranking minimizes device count, node count and maximum static pool utilization; it is not a latency or throughput ranking.",
            "Whole-block placement remains the validated executable baseline.",
        ],
    }
