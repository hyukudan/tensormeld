"""Bounded performance-aware advisory planning over legal model units.

The objective is an ordered unit-chain upper bound:
  sum(explicit unit compute_us)
+ sum(conservative directed boundary transfer upper bounds)

It is NOT token latency, prefill latency, throughput, or an executable plan. Sampling,
token feedback, overlap, queueing and phase-specific behavior are outside this contract.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable

from .config_v2 import Config
from .directional_path_evidence import (
    DirectionalPathEvidence,
    lookup_transfer_upper_bound,
)
from .legal_model_units import LegalModelUnitsProfile
from .legal_unit_costs import LegalUnitCostsProfile, UnitDeviceCost
from .schema import ValidationError
from .selection import resolve_candidates
from .tensor_movability import TensorMovabilityProfile

RESULT_SCHEMA = "tensormeld/legal-unit-performance-result-v1"


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
    tensor_resident: tuple[tuple[str, int], ...]
    persistent_state: tuple[tuple[str, int], ...]
    workspace_by_device_pool: tuple[tuple[str, str, int], ...]
    staging_by_device_pool: tuple[tuple[str, str, int], ...]
    compute_us: int
    transfer_us: int
    transfers: tuple[dict[str, Any], ...]

    def map(self, field: str) -> dict:
        if field == "tensor":
            return dict(self.tensor_resident)
        if field == "state":
            return dict(self.persistent_state)
        if field == "workspace":
            return {(d, p): n for d, p, n in self.workspace_by_device_pool}
        if field == "staging":
            return {(d, p): n for d, p, n in self.staging_by_device_pool}
        raise KeyError(field)


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


def _device_profile(
    costs: LegalUnitCostsProfile,
    index: int,
    device_id: str,
) -> UnitDeviceCost:
    unit = costs.units[index]
    for profile in unit.device_profiles:
        if profile.device == device_id:
            return profile
    raise ValidationError(
        f"legal-unit costs missing device profile for unit {unit.unit_id} / {device_id}"
    )


def plan_legal_unit_performance(
    config: Config,
    legal_units: LegalModelUnitsProfile,
    costs: LegalUnitCostsProfile,
    paths: DirectionalPathEvidence,
    movability: TensorMovabilityProfile,
    profile_name: str | None = None,
    top_k: int = 5,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValidationError("top_k must be an integer in 1..20")
    if costs.config_sha256 != config.fingerprint:
        raise ValidationError("legal-unit cost config identity mismatch")
    if paths.config_sha256 != config.fingerprint:
        raise ValidationError("directional path config identity mismatch")
    if costs.legal_model_units_sha256 != legal_units.fingerprint:
        raise ValidationError("legal-unit cost/legal-unit identity mismatch")
    if legal_units.tensor_movability_sha256 != movability.fingerprint:
        raise ValidationError("legal-unit/tensor-movability identity mismatch")
    if tuple(unit.unit_id for unit in costs.units) != tuple(
        unit.id for unit in legal_units.units
    ):
        raise ValidationError("legal-unit cost sequence differs from legal-unit sequence")
    if paths.provenance != costs.provenance:
        raise ValidationError("path and legal-unit cost provenance must match")
    if not set(paths.runtime_identity_sha256) <= set(movability.runtime_identity_sha256):
        raise ValidationError(
            "path runtime identities are not from the tensor-movability environment"
        )

    start = clock()
    budget = _Budget(
        config.planning.candidate_limit,
        start + config.planning.deadline_ms / 1000,
        clock,
    )
    selection = resolve_candidates(config, profile_name)
    devices = {d["id"]: d for d in selection["eligible_compute_devices"]}
    if not devices:
        raise ValidationError("performance planner has no eligible compute devices")
    device_ids = tuple(sorted(devices))
    pool_budgets = selection["physical_pool_budgets"]
    required_devices = set(selection["required_devices"])
    required_nodes = set(selection["required_nodes"])
    required_local = set(selection["required_any_local_gpu"])
    rejections: Counter[str] = Counter()
    retained: list[dict[str, Any]] = []
    complete_candidates = 0

    for index, unit in enumerate(legal_units.units):
        legal_now = set(unit.allowed_devices) & set(device_ids)
        cost_devices = {p.device for p in costs.units[index].device_profiles}
        if not legal_now:
            return _empty_result(
                config, legal_units, costs, paths, selection,
                reason="unit_has_no_policy_eligible_device",
            )
        if not legal_now <= cost_devices:
            raise ValidationError(
                f"cost evidence lacks current legal device profile for {unit.id}"
            )

    def physical_usage(state: _State) -> dict[str, int]:
        usage = state.map("tensor")
        for pool, amount in state.map("state").items():
            usage[pool] = usage.get(pool, 0) + amount
        workspace_by_pool: dict[str, int] = {}
        for (_, pool), amount in state.map("workspace").items():
            workspace_by_pool[pool] = workspace_by_pool.get(pool, 0) + amount
        staging_by_pool: dict[str, int] = {}
        for (_, pool), amount in state.map("staging").items():
            staging_by_pool[pool] = staging_by_pool.get(pool, 0) + amount
        for pool, amount in workspace_by_pool.items():
            usage[pool] = usage.get(pool, 0) + amount
        for pool, amount in staging_by_pool.items():
            usage[pool] = usage.get(pool, 0) + amount
        return usage

    def fits(state: _State) -> bool:
        for pool, amount in physical_usage(state).items():
            limit = pool_budgets.get(pool)
            if limit is None:
                rejections["unknown_pool_budget"] += 1
                return False
            if amount > limit:
                rejections["pool_budget_exceeded"] += 1
                return False
        return True

    def owner_nodes(owners: tuple[str, ...]) -> set[str]:
        return {devices[d]["node"] for d in set(owners)}

    def requirements_met(owners: tuple[str, ...]) -> bool:
        used = set(owners)
        nodes = owner_nodes(owners)
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

    def choices_for(index: int, state: _State) -> tuple[str, ...]:
        allowed = set(legal_units.units[index].allowed_devices) & set(device_ids)
        if index > 0 and state.owners:
            previous = state.owners[-1]
            if not legal_units.units[index - 1].cut_after:
                return (previous,) if previous in allowed else ()
        ordered: list[str] = []
        if state.owners and state.owners[-1] in allowed:
            ordered.append(state.owners[-1])
        ordered.extend(d for d in device_ids if d in allowed and d not in ordered)
        return tuple(ordered)

    def extend(state: _State, index: int, device_id: str) -> _State | None:
        budget.consume()
        unit = legal_units.units[index]
        cost = _device_profile(costs, index, device_id)

        used = set(state.owners) | {device_id}
        nodes = {devices[d]["node"] for d in used}
        if len(used) > selection["max_compute_devices"]:
            rejections["max_compute_devices_exceeded"] += 1
            return None
        if len(nodes) > selection["max_compute_nodes"]:
            rejections["max_compute_nodes_exceeded"] += 1
            return None

        remaining = len(legal_units.units) - index - 1
        if len(required_devices - used) > remaining:
            rejections["required_device_unreachable"] += 1
            return None
        if len(required_nodes - nodes) > remaining:
            rejections["required_node_unreachable"] += 1
            return None

        tensor = state.map("tensor")
        state_bytes = state.map("state")
        workspace = state.map("workspace")
        staging = state.map("staging")

        owner_pool = devices[device_id]["pool"]
        tensor_bytes = unit.hard_resident_bytes + unit.reclaimable_file_backed_bytes
        tensor[owner_pool] = tensor.get(owner_pool, 0) + tensor_bytes

        for pool, amount in cost.persistent_state:
            state_bytes[pool] = state_bytes.get(pool, 0) + amount
        for pool, amount in cost.workspace_peak:
            key = (device_id, pool)
            workspace[key] = max(workspace.get(key, 0), amount)
        for pool, amount in cost.staging_peak:
            key = (device_id, pool)
            staging[key] = max(staging.get(key, 0), amount)

        transfer_us = state.transfer_us
        transfers = state.transfers
        if index > 0 and state.owners[-1] != device_id:
            previous = state.owners[-1]
            payload = costs.units[index - 1].boundary_output_bytes
            transfer = lookup_transfer_upper_bound(
                paths,
                source_device=previous,
                target_device=device_id,
                payload_bytes=payload,
            )
            if transfer is None:
                rejections["missing_directional_path_bucket"] += 1
                return None
            transfer_us += transfer["upper_bound_us"]
            transfers += (transfer,)

        next_state = _State(
            state.owners + (device_id,),
            tuple(sorted(tensor.items())),
            tuple(sorted(state_bytes.items())),
            tuple(sorted((d, p, n) for (d, p), n in workspace.items())),
            tuple(sorted((d, p, n) for (d, p), n in staging.items())),
            state.compute_us + cost.compute_us,
            transfer_us,
            transfers,
        )
        return next_state if fits(next_state) else None

    def score(candidate: dict[str, Any]) -> tuple:
        return (
            candidate["ordered_unit_chain_upper_bound_us"],
            candidate["compute_device_count"],
            candidate["compute_node_count"],
            candidate["max_pool_utilization_ppm"],
            candidate["plan_sha256"],
        )

    def finish(state: _State) -> None:
        nonlocal complete_candidates
        if not requirements_met(state.owners):
            return
        complete_candidates += 1
        used = sorted(set(state.owners))
        nodes = sorted(owner_nodes(state.owners))
        usage = physical_usage(state)
        utilization = []
        for pool, amount in usage.items():
            limit = pool_budgets.get(pool)
            if limit is not None and limit > 0:
                utilization.append((amount * 1_000_000 + limit - 1) // limit)

        segments = []
        for i, owner in enumerate(state.owners):
            if not segments or segments[-1]["device"] != owner:
                segments.append({
                    "device": owner,
                    "first_unit": i,
                    "last_unit_exclusive": i + 1,
                })
            else:
                segments[-1]["last_unit_exclusive"] = i + 1

        core = {
            "compute_nodes": nodes,
            "compute_devices": used,
            "compute_node_count": len(nodes),
            "compute_device_count": len(used),
            "owners": list(state.owners),
            "segments": segments,
            "unit_ids": [unit.id for unit in legal_units.units],
            "compute_upper_bound_us": state.compute_us,
            "transfer_upper_bound_us": state.transfer_us,
            "ordered_unit_chain_upper_bound_us": state.compute_us + state.transfer_us,
            "transfers": list(state.transfers),
            "pool_preparation_upper_bound_bytes": dict(sorted(usage.items())),
            "max_pool_utilization_ppm": max(utilization, default=0),
            "qualified": False,
            "executable": False,
        }
        identity = {
            "config_sha256": config.fingerprint,
            "legal_model_units_sha256": legal_units.fingerprint,
            "legal_unit_costs_sha256": costs.fingerprint,
            "directional_path_evidence_sha256": paths.fingerprint,
            "profile": selection["profile"],
            "plan": core,
        }
        core["plan_sha256"] = _sha256(identity)
        if all(c["plan_sha256"] != core["plan_sha256"] for c in retained):
            retained.append(core)
            retained.sort(key=score)
            del retained[top_k:]

    root = _State((), (), (), (), (), 0, 0, ())
    stack: list[tuple[int, _State, tuple[str, ...], int]] = [
        (0, root, choices_for(0, root), 0)
    ]
    try:
        while stack:
            index, state, choices, pos = stack.pop()
            if pos >= len(choices):
                continue
            stack.append((index, state, choices, pos + 1))
            device_id = choices[pos]
            next_state = extend(state, index, device_id)
            if next_state is None:
                continue
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
        if retained
        else "NO_CANDIDATE_IN_SEARCH_SPACE"
    )
    return {
        "result_schema": RESULT_SCHEMA,
        "status": status,
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal_units.fingerprint,
        "legal_unit_costs_sha256": costs.fingerprint,
        "directional_path_evidence_sha256": paths.fingerprint,
        "profile": selection["profile"],
        "objective": "ordered_unit_chain_upper_bound",
        "qualified": False,
        "executable": False,
        "best": retained[0] if retained else None,
        "candidates": retained,
        "search": {
            "complete": not incomplete,
            "reason": budget.reason,
            "work_units": budget.used,
            "work_limit": budget.limit,
            "deadline_ms": config.planning.deadline_ms,
            "elapsed_ms": max(0, int((clock() - start) * 1000)),
            "complete_candidates_evaluated": complete_candidates,
            "retained_candidates": len(retained),
            "rejections": dict(sorted(rejections.items())),
        },
        "warnings": [
            "Objective is an ordered legal-unit chain upper bound, not token latency, TTFT or throughput.",
            "Compute costs come only from exact legal-unit cost evidence.",
            "Boundary transfer costs come only from conservative directional path buckets; uncovered payloads reject that candidate.",
            "Tensor resident and persistent state bytes are additive; workspace and staging use per-device/pool maxima and are conservatively summed across devices sharing a physical pool.",
            "Reclaimable/file-backed tensor bytes remain counted in full.",
            "Sampling, token feedback, prefill/decode phase distinctions, overlap, contention and queueing are not modeled.",
            "Results remain advisory, qualified=false and executable=false; whole-block remains the validated executable baseline.",
        ],
    }


def _empty_result(
    config: Config,
    legal_units: LegalModelUnitsProfile,
    costs: LegalUnitCostsProfile,
    paths: DirectionalPathEvidence,
    selection: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "result_schema": RESULT_SCHEMA,
        "status": "NO_CANDIDATE_IN_SEARCH_SPACE",
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal_units.fingerprint,
        "legal_unit_costs_sha256": costs.fingerprint,
        "directional_path_evidence_sha256": paths.fingerprint,
        "profile": selection["profile"],
        "objective": "ordered_unit_chain_upper_bound",
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
            "retained_candidates": 0,
            "rejections": {reason: 1},
        },
        "warnings": [
            "No policy-eligible legal ownership exists for at least one unit.",
            "This is advisory evidence evaluation only, not native execution.",
        ],
    }
