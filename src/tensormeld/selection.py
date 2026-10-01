"""Resolve user policy, without claiming availability or backend qualification."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config_v2 import Config, Device
from .schema import ValidationError


def pool_budgets(config: Config) -> dict[str, int | None]:
    """Intersect static owner caps and capacity-minus-reserve; NOT live free memory."""
    policies = {p.pool: p for p in config.resource_policies}
    result: dict[str, int | None] = {}
    for pool in config.pools:
        policy = policies[pool.id]
        ceilings = []
        if policy.allocation_cap_bytes is not None:
            ceilings.append(policy.allocation_cap_bytes)
        if pool.reported_capacity_bytes is not None:
            ceilings.append(max(0, pool.reported_capacity_bytes - policy.safety_headroom_bytes))
        result[pool.id] = min(ceilings) if ceilings else None
    return result


def resolve_candidates(config: Config, profile_name: str | None = None) -> dict[str, Any]:
    name = profile_name or config.installation.default_profile
    if name not in config.profile_map:
        raise ValidationError(f"unknown profile: {name}")
    profile = config.profile_map[name]
    s = config.selection
    enabled_nodes = {n.id for n in config.nodes if n.enabled and "compute" in n.allowed_roles}
    legal_nodes = enabled_nodes if s.allowed_nodes is None else enabled_nodes & set(s.allowed_nodes)
    legal_nodes -= set(s.excluded_nodes)
    required_nodes = set(s.required_nodes)
    if s.mode == "manual":
        legal_nodes &= set(s.selected_nodes)
        required_nodes |= set(s.selected_nodes)
    local = config.installation.entrypoint_node
    if profile.execution_mode == "local_only":
        legal_nodes &= {local}
    elif profile.execution_mode == "companion_only":
        legal_nodes.discard(local)
    candidates: list[Device] = []
    for device in config.devices:
        is_local_gpu = device.node == local and device.kind != "cpu"
        if (not device.enabled or device.node not in legal_nodes
                or device.kind not in s.allowed_device_kinds or device.id in s.excluded_devices
                or (s.local_gpu == "excluded" and is_local_gpu)
                or (profile.placement.cpu_offload == "disabled" and device.kind == "cpu")):
            continue
        candidates.append(device)
    required_devices = set(s.required_devices)
    candidate_ids = {d.id for d in candidates}
    if required_devices - candidate_ids:
        raise ValidationError(f"required compute devices are unavailable or disallowed: {sorted(required_devices - candidate_ids)}")
    required_nodes |= {d.node for d in candidates if d.id in required_devices}
    local_gpu_candidates = {d.id for d in candidates if d.node == local and d.kind != "cpu"}
    if s.local_gpu == "required":
        if not local_gpu_candidates:
            raise ValidationError("local_gpu=required has no eligible local GPU")
        required_nodes.add(local)
    candidate_nodes = {d.node for d in candidates}
    if required_nodes - candidate_nodes:
        raise ValidationError(f"required nodes have no eligible compute device: {sorted(required_nodes - candidate_nodes)}")
    min_devices = max(s.min_compute_devices, 2 if profile.execution_mode == "distributed" else 1)
    max_nodes = min(s.max_compute_nodes or len(candidate_nodes), len(candidate_nodes))
    max_devices = min(s.max_compute_devices or len(candidates), len(candidates))
    if len(candidate_nodes) < s.min_compute_nodes or max_nodes < s.min_compute_nodes:
        raise ValidationError("not enough eligible compute nodes to satisfy min_compute_nodes")
    if len(candidates) < min_devices or max_devices < min_devices:
        raise ValidationError("not enough eligible compute devices to satisfy execution mode/min_compute_devices")
    if s.min_compute_nodes > max_devices or len(required_nodes) > max_devices:
        raise ValidationError("required/minimum node count exceeds max_compute_devices")
    if len(required_nodes) > max_nodes or len(required_devices) > max_devices:
        raise ValidationError("required compute owners exceed configured maxima")
    if s.local_gpu == "required" and not (required_devices & local_gpu_candidates) and len(required_devices) == max_devices:
        raise ValidationError("local GPU requirement cannot fit alongside required devices")
    coordinators = sorted(n.id for n in config.nodes
                          if n.enabled and "coordinator" in n.allowed_roles
                          and n.id in s.coordinator.allowed_nodes)
    if not coordinators:
        raise ValidationError("no enabled allowed coordinator node")
    budgets = pool_budgets(config)
    resolved = [{"id": d.id, "node": d.node, "kind": d.kind, "backend": d.backend,
                 "pool": d.pool, "pool_budget_bytes": budgets[d.pool],
                 "required": d.id in required_devices}
                for d in sorted(candidates, key=lambda x: (x.node, x.id))]
    return {
        "config_schema": "tensormeld/v2", "config_sha256": config.fingerprint,
        "profile": name, "execution_mode": profile.execution_mode, "objective": profile.objective,
        "entrypoint_node": local, "eligible_compute_nodes": sorted(candidate_nodes),
        "eligible_compute_devices": resolved, "required_nodes": sorted(required_nodes),
        "required_devices": sorted(required_devices),
        "required_any_local_gpu": sorted(local_gpu_candidates) if s.local_gpu == "required" else [],
        "min_compute_nodes": s.min_compute_nodes, "min_compute_devices": min_devices,
        "max_compute_nodes": max_nodes, "max_compute_devices": max_devices,
        "coordinator_candidates": coordinators, "physical_pool_budgets": budgets,
        "planning_budget": asdict(config.planning), "qualified": False,
        "warnings": [
            "Policy eligibility is not live availability or model/backend compatibility.",
            "Pool budgets are intersected static ceilings, not guaranteed allocatable memory.",
            "No performance ranking or distributed inference is performed by select.",
        ],
    }
