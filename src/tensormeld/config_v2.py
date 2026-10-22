"""Strict installation configuration contracts for TensorMeld v2.

This module models computers, compute devices, physical memory pools and user policy
as separate concepts. It intentionally does not probe hardware or execute inference.
The contract is platform-neutral and is designed to survive heterogeneous Windows /
Linux deployments and multiple devices per node.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import MAX_INPUT_BYTES, ValidationError, items, number, record, text, unique, _no_duplicates

MAX_NODES = 64
MAX_DEVICES = 128
MAX_POOLS = 192
MAX_POLICIES = 192
MAX_PROFILES = 64

ROLE_VALUES = {"front_door", "control", "coordinator", "compute", "cache"}
DEVICE_KINDS = {"discrete_gpu", "integrated_gpu", "cpu"}
EXECUTION_MODES = {"auto", "local_only", "companion_only", "distributed"}
LOCAL_GPU_POLICIES = {"auto", "required", "excluded"}
OBJECTIVES = {"interactive_latency", "throughput", "capacity", "balanced"}
PLACEMENT_STRATEGIES = {"auto", "whole_blocks", "expert", "tensor", "phase"}


def _string_list(value: Any, where: str, maximum: int = 128, minimum: int = 0) -> tuple[str, ...]:
    result = tuple(text(v, f"{where}[]") for v in items(value, where, maximum, minimum))
    unique(list(result), where)
    return result


def _auto_or_int(value: Any, where: str, minimum: int = 1) -> int | None:
    if value == "auto":
        return None
    return int(number(value, where, minimum, True))


@dataclass(frozen=True)
class Installation:
    id: str
    entrypoint_node: str
    default_profile: str


@dataclass(frozen=True)
class Node:
    id: str
    enrollment_ref: str
    enabled: bool
    allowed_roles: frozenset[str]


@dataclass(frozen=True)
class Pool:
    id: str
    node: str
    kind: str
    reported_capacity_bytes: int | None


@dataclass(frozen=True)
class Device:
    id: str
    node: str
    kind: str
    backend: str
    pool: str
    enabled: bool


@dataclass(frozen=True)
class ResourcePolicy:
    id: str
    owner_node: str
    pool: str
    allocation_cap_bytes: int | None
    safety_headroom_bytes: int


@dataclass(frozen=True)
class CoordinatorPolicy:
    mode: str
    allowed_nodes: tuple[str, ...]


@dataclass(frozen=True)
class SelectionPolicy:
    mode: str
    allowed_nodes: tuple[str, ...] | None
    selected_nodes: tuple[str, ...]
    required_nodes: tuple[str, ...]
    excluded_nodes: tuple[str, ...]
    min_compute_nodes: int
    max_compute_nodes: int | None
    min_compute_devices: int
    max_compute_devices: int | None
    allowed_device_kinds: frozenset[str]
    required_devices: tuple[str, ...]
    excluded_devices: tuple[str, ...]
    coordinator: CoordinatorPolicy
    local_gpu: str


@dataclass(frozen=True)
class Workload:
    task: str
    context_tokens: int
    max_output_tokens: int
    max_active_requests: int


@dataclass(frozen=True)
class SemanticPolicy:
    automatic_model_change: bool
    automatic_weight_encoding_change: bool
    automatic_state_encoding_change: bool
    automatic_context_reduction: bool
    automatic_history_truncation: bool


@dataclass(frozen=True)
class PlacementPolicy:
    strategy: str
    cpu_offload: str
    require_qualified_execution: bool


@dataclass(frozen=True)
class LifecyclePolicy:
    queue_limit: int
    on_owner_loss: str
    on_resource_pressure: str


@dataclass(frozen=True)
class Profile:
    name: str
    execution_mode: str
    objective: str
    model_manifest_ref: str
    workload: Workload
    semantic_policy: SemanticPolicy
    placement: PlacementPolicy
    lifecycle: LifecyclePolicy


@dataclass(frozen=True)
class NetworkPolicy:
    transport: str
    interface_selection: str
    multirail: str
    max_parallel_probes: int


@dataclass(frozen=True)
class PrivacyPolicy:
    persist_prompts: bool
    external_telemetry: bool


@dataclass(frozen=True)
class PlanningPolicy:
    candidate_limit: int
    deadline_ms: int
    on_budget_exhaustion: str


@dataclass(frozen=True)
class Config:
    installation: Installation
    nodes: tuple[Node, ...]
    pools: tuple[Pool, ...]
    devices: tuple[Device, ...]
    selection: SelectionPolicy
    resource_policies: tuple[ResourcePolicy, ...]
    profiles: tuple[Profile, ...]
    network: NetworkPolicy
    privacy: PrivacyPolicy
    planning: PlanningPolicy
    fingerprint: str

    @property
    def profile_map(self) -> dict[str, Profile]:
        return {p.name: p for p in self.profiles}

    @classmethod
    def parse(cls, data: Any) -> "Config":
        root = record(data, "config", {
            "config_schema", "installation", "nodes", "resource_pools", "devices",
            "selection", "resource_policies", "profiles", "network_policy", "privacy",
            "planning_policy",
        })
        if root["config_schema"] != "tensormeld/v2":
            raise ValidationError("config_schema: expected tensormeld/v2")

        inst = record(root["installation"], "installation", {"id", "entrypoint_node", "default_profile"})
        installation = Installation(text(inst["id"], "installation.id"),
                                    text(inst["entrypoint_node"], "installation.entrypoint_node"),
                                    text(inst["default_profile"], "installation.default_profile"))

        nodes: list[Node] = []
        for i, raw in enumerate(items(root["nodes"], "nodes", MAX_NODES)):
            n = record(raw, f"nodes[{i}]", {"id", "enrollment_ref", "enabled", "allowed_roles"})
            if type(n["enabled"]) is not bool:
                raise ValidationError(f"nodes[{i}].enabled: expected boolean")
            roles = frozenset(_string_list(n["allowed_roles"], f"nodes[{i}].allowed_roles", 8, 1))
            if not roles <= ROLE_VALUES:
                raise ValidationError(f"nodes[{i}].allowed_roles: unsupported role")
            nodes.append(Node(text(n["id"], "node.id"), text(n["enrollment_ref"], "node.enrollment_ref"),
                              n["enabled"], roles))
        unique([n.id for n in nodes], "nodes")
        node_map = {n.id: n for n in nodes}
        if installation.entrypoint_node not in node_map:
            raise ValidationError("installation.entrypoint_node: unknown node")

        pools: list[Pool] = []
        for i, raw in enumerate(items(root["resource_pools"], "resource_pools", MAX_POOLS)):
            p = record(raw, f"resource_pools[{i}]", {"id", "node", "kind", "reported_capacity_bytes"})
            node = text(p["node"], "resource_pool.node")
            if node not in node_map:
                raise ValidationError(f"resource_pools[{i}].node: unknown node")
            cap = p["reported_capacity_bytes"]
            if cap is not None:
                cap = int(number(cap, f"resource_pools[{i}].reported_capacity_bytes", 1, True))
            pools.append(Pool(text(p["id"], "resource_pool.id"), node, text(p["kind"], "resource_pool.kind"), cap))
        unique([p.id for p in pools], "resource_pools")
        pool_map = {p.id: p for p in pools}

        devices: list[Device] = []
        for i, raw in enumerate(items(root["devices"], "devices", MAX_DEVICES)):
            d = record(raw, f"devices[{i}]", {"id", "node", "kind", "backend", "pool_ref", "enabled"})
            if type(d["enabled"]) is not bool:
                raise ValidationError(f"devices[{i}].enabled: expected boolean")
            node = text(d["node"], "device.node")
            pool = text(d["pool_ref"], "device.pool_ref")
            kind = text(d["kind"], "device.kind")
            if node not in node_map:
                raise ValidationError(f"devices[{i}].node: unknown node")
            if pool not in pool_map or pool_map[pool].node != node:
                raise ValidationError(f"devices[{i}].pool_ref: pool must exist on the same node")
            if kind not in DEVICE_KINDS:
                raise ValidationError(f"devices[{i}].kind: unsupported device kind")
            devices.append(Device(text(d["id"], "device.id"), node, kind,
                                  text(d["backend"], "device.backend"), pool, d["enabled"]))
        unique([d.id for d in devices], "devices")
        device_map = {d.id: d for d in devices}

        s = record(root["selection"], "selection", {
            "mode", "required_nodes", "excluded_nodes", "min_compute_nodes",
            "max_compute_nodes", "min_compute_devices", "max_compute_devices", "allowed_device_kinds",
            "required_devices", "excluded_devices", "coordinator", "participation",
        }, {"allowed_nodes", "selected_nodes"})
        mode = text(s["mode"], "selection.mode")
        if mode not in ("auto", "manual"):
            raise ValidationError("selection.mode: expected auto or manual")
        allowed_nodes = (_string_list(s["allowed_nodes"], "selection.allowed_nodes", MAX_NODES, 0)
                         if "allowed_nodes" in s else None)
        selected_nodes = _string_list(s.get("selected_nodes", []), "selection.selected_nodes", MAX_NODES, 0)
        if mode == "manual" and not selected_nodes:
            raise ValidationError("manual selection requires a nonempty selected_nodes exact compute set")
        if mode == "auto" and selected_nodes:
            raise ValidationError("selected_nodes is only valid for manual selection")
        required_nodes = _string_list(s["required_nodes"], "selection.required_nodes", MAX_NODES, 0)
        excluded_nodes = _string_list(s["excluded_nodes"], "selection.excluded_nodes", MAX_NODES, 0)
        all_node_ids = set(node_map)
        for where, values in (("allowed_nodes", allowed_nodes or ()), ("selected_nodes", selected_nodes), ("required_nodes", required_nodes),
                              ("excluded_nodes", excluded_nodes)):
            if not set(values) <= all_node_ids:
                raise ValidationError(f"selection.{where}: unknown node")
        if set(required_nodes) & set(excluded_nodes):
            raise ValidationError("selection: a node cannot be both required and excluded")
        if allowed_nodes is not None and not set(required_nodes) <= set(allowed_nodes):
            raise ValidationError("selection.required_nodes must be within allowed_nodes")

        allowed_kinds = frozenset(_string_list(s["allowed_device_kinds"], "selection.allowed_device_kinds", 8, 1))
        if not allowed_kinds <= DEVICE_KINDS:
            raise ValidationError("selection.allowed_device_kinds: unsupported device kind")
        required_devices = _string_list(s["required_devices"], "selection.required_devices", MAX_DEVICES, 0)
        excluded_devices = _string_list(s["excluded_devices"], "selection.excluded_devices", MAX_DEVICES, 0)
        if not set(required_devices + excluded_devices) <= set(device_map):
            raise ValidationError("selection required/excluded_devices: unknown device")
        if set(required_devices) & set(excluded_devices):
            raise ValidationError("selection: a device cannot be both required and excluded")

        coord = record(s["coordinator"], "selection.coordinator", {"mode", "allowed_nodes"})
        coord_mode = text(coord["mode"], "selection.coordinator.mode")
        if coord_mode not in ("auto", "fixed"):
            raise ValidationError("selection.coordinator.mode: expected auto or fixed")
        coord_nodes = _string_list(coord["allowed_nodes"], "selection.coordinator.allowed_nodes", MAX_NODES, 1)
        if not set(coord_nodes) <= all_node_ids:
            raise ValidationError("selection.coordinator.allowed_nodes: unknown node")
        if coord_mode == "fixed" and len(coord_nodes) != 1:
            raise ValidationError("fixed coordinator mode requires exactly one allowed node")
        part = record(s["participation"], "selection.participation", {"local_gpu"})
        local_gpu = text(part["local_gpu"], "selection.participation.local_gpu")
        if local_gpu not in LOCAL_GPU_POLICIES:
            raise ValidationError("selection.participation.local_gpu: invalid value")

        min_nodes = int(number(s["min_compute_nodes"], "selection.min_compute_nodes", 1, True))
        max_nodes = _auto_or_int(s["max_compute_nodes"], "selection.max_compute_nodes")
        min_devices = int(number(s["min_compute_devices"], "selection.min_compute_devices", 1, True))
        max_devices = _auto_or_int(s["max_compute_devices"], "selection.max_compute_devices")
        if max_nodes is not None and min_nodes > max_nodes:
            raise ValidationError("selection: min_compute_nodes exceeds max_compute_nodes")
        if max_devices is not None and min_devices > max_devices:
            raise ValidationError("selection: min_compute_devices exceeds max_compute_devices")
        selection = SelectionPolicy(mode, allowed_nodes, selected_nodes, required_nodes, excluded_nodes,
                                    min_nodes, max_nodes, min_devices, max_devices, allowed_kinds,
                                    required_devices, excluded_devices,
                                    CoordinatorPolicy(coord_mode, coord_nodes), local_gpu)

        policies: list[ResourcePolicy] = []
        for i, raw in enumerate(items(root["resource_policies"], "resource_policies", MAX_POLICIES)):
            p = record(raw, f"resource_policies[{i}]", {
                "id", "owner_node", "pool_ref", "allocation_cap_bytes", "safety_headroom_bytes",
            })
            owner = text(p["owner_node"], "resource_policy.owner_node")
            pool = text(p["pool_ref"], "resource_policy.pool_ref")
            if owner not in node_map or pool not in pool_map or pool_map[pool].node != owner:
                raise ValidationError(f"resource_policies[{i}]: pool and owner must be on the same known node")
            cap = p["allocation_cap_bytes"]
            if cap is not None:
                cap = int(number(cap, f"resource_policies[{i}].allocation_cap_bytes", 1, True))
            headroom = int(number(p["safety_headroom_bytes"], f"resource_policies[{i}].safety_headroom_bytes", 0, True))
            if pool_map[pool].reported_capacity_bytes is not None:
                physical = pool_map[pool].reported_capacity_bytes
                if cap is not None and cap > physical:
                    raise ValidationError(f"resource_policies[{i}].allocation_cap_bytes exceeds reported capacity")
                if headroom > physical:
                    raise ValidationError(f"resource_policies[{i}].safety_headroom_bytes exceeds reported capacity")
            policies.append(ResourcePolicy(text(p["id"], "resource_policy.id"), owner, pool, cap, headroom))
        unique([p.id for p in policies], "resource_policies")
        policy_by_pool = {p.pool: p for p in policies}
        if len(policy_by_pool) != len(policies):
            raise ValidationError("resource_policies: at most one policy per physical pool")
        if set(pool_map) - set(policy_by_pool):
            raise ValidationError("resource_policies: every physical pool requires a policy")

        raw_profiles = root["profiles"]
        if not isinstance(raw_profiles, dict) or not 1 <= len(raw_profiles) <= MAX_PROFILES:
            raise ValidationError(f"profiles: expected 1..{MAX_PROFILES} named profiles")
        profiles: list[Profile] = []
        for name, raw in raw_profiles.items():
            profile_name = text(name, "profile.name")
            p = record(raw, f"profiles.{profile_name}", {
                "execution_mode", "objective", "model_manifest_ref", "workload", "semantic_policy",
                "placement", "lifecycle",
            })
            execution_mode = text(p["execution_mode"], "profile.execution_mode")
            if execution_mode not in EXECUTION_MODES:
                raise ValidationError("profile.execution_mode: invalid value")
            objective = text(p["objective"], "profile.objective")
            if objective not in OBJECTIVES:
                raise ValidationError("profile.objective: invalid value")
            w = record(p["workload"], "profile.workload", {"task", "context_tokens", "max_output_tokens", "max_active_requests"})
            workload = Workload(text(w["task"], "profile.workload.task"),
                                int(number(w["context_tokens"], "profile.workload.context_tokens", 1, True)),
                                int(number(w["max_output_tokens"], "profile.workload.max_output_tokens", 1, True)),
                                int(number(w["max_active_requests"], "profile.workload.max_active_requests", 1, True)))
            if workload.max_output_tokens > workload.context_tokens:
                raise ValidationError("max_output_tokens cannot exceed context_tokens")
            sem = record(p["semantic_policy"], "profile.semantic_policy", {
                "automatic_model_change", "automatic_weight_encoding_change", "automatic_state_encoding_change",
                "automatic_context_reduction", "automatic_history_truncation",
            })
            for key, value in sem.items():
                if type(value) is not bool:
                    raise ValidationError(f"profile.semantic_policy.{key}: expected boolean")
            semantic = SemanticPolicy(**sem)
            pl = record(p["placement"], "profile.placement", {"strategy", "cpu_offload", "require_qualified_execution"})
            strategy = text(pl["strategy"], "profile.placement.strategy")
            if strategy not in PLACEMENT_STRATEGIES:
                raise ValidationError("profile.placement.strategy: unsupported strategy")
            if type(pl["require_qualified_execution"]) is not bool:
                raise ValidationError("profile.placement.require_qualified_execution: expected boolean")
            if pl["cpu_offload"] not in ("disabled", "allowed"):
                raise ValidationError("profile.placement.cpu_offload: expected disabled or allowed")
            placement = PlacementPolicy(strategy, text(pl["cpu_offload"], "profile.placement.cpu_offload"),
                                        pl["require_qualified_execution"])
            lc = record(p["lifecycle"], "profile.lifecycle", {"queue_limit", "on_owner_loss", "on_resource_pressure"})
            lifecycle = LifecyclePolicy(int(number(lc["queue_limit"], "profile.lifecycle.queue_limit", 0, True)),
                                        text(lc["on_owner_loss"], "profile.lifecycle.on_owner_loss"),
                                        text(lc["on_resource_pressure"], "profile.lifecycle.on_resource_pressure"))
            profiles.append(Profile(profile_name, execution_mode, objective,
                                    text(p["model_manifest_ref"], "profile.model_manifest_ref"),
                                    workload, semantic, placement, lifecycle))
        if installation.default_profile not in {p.name for p in profiles}:
            raise ValidationError("installation.default_profile: unknown profile")

        nw = record(root["network_policy"], "network_policy", {
            "transport", "interface_selection", "multirail", "max_parallel_probes",
        })
        network = NetworkPolicy(text(nw["transport"], "network_policy.transport"),
                                text(nw["interface_selection"], "network_policy.interface_selection"),
                                text(nw["multirail"], "network_policy.multirail"),
                                int(number(nw["max_parallel_probes"], "network_policy.max_parallel_probes", 1, True)))
        privacy_raw = record(root["privacy"], "privacy", {"persist_prompts", "external_telemetry"})
        if any(type(v) is not bool for v in privacy_raw.values()):
            raise ValidationError("privacy: expected booleans")
        privacy = PrivacyPolicy(**privacy_raw)
        pp = record(root["planning_policy"], "planning_policy", {"search_budget", "on_budget_exhaustion"})
        sb = record(pp["search_budget"], "planning_policy.search_budget", {"candidate_limit", "deadline_ms"})
        planning = PlanningPolicy(int(number(sb["candidate_limit"], "planning_policy.search_budget.candidate_limit", 1, True)),
                                  int(number(sb["deadline_ms"], "planning_policy.search_budget.deadline_ms", 1, True)),
                                  text(pp["on_budget_exhaustion"], "planning_policy.on_budget_exhaustion"))

        if planning.candidate_limit > 100_000 or planning.deadline_ms > 60_000:
            raise ValidationError("planning budget exceeds this build: 100000 work units / 60000 ms")
        if planning.on_budget_exhaustion != "return_search_incomplete":
            raise ValidationError("on_budget_exhaustion: expected return_search_incomplete")

        encoded = json.dumps(root, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return cls(installation, tuple(nodes), tuple(pools), tuple(devices), selection, tuple(policies),
                   tuple(profiles), network, privacy, planning, hashlib.sha256(encoded).hexdigest())


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("config exceeds 2 MiB limit")
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    return Config.parse(data)
