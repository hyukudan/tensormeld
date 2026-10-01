"""Bounded synthetic whole-block search with multiple physical memory demands.

Never emits an executable plan. A completed search is complete only within this
finite, contiguous-block, sequential-C1 model, not across arbitrary inference engines.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Callable

from .config_v2 import Config
from .planning_contract import PlanningInput
from .schema import ValidationError
from .selection import resolve_candidates


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
            self.reason = 'work_limit'
            raise _BudgetExpired
        if self.clock() >= self.deadline:
            self.reason = 'deadline'
            raise _BudgetExpired
        self.used += 1


@dataclass
class _State:
    owners: tuple[str, ...] = ()
    resident: dict[str, int] = field(default_factory=dict)
    workspace: dict[tuple[str, str], int] = field(default_factory=dict)
    compute_us: int = 0
    transfer_us: int = 0
    routes: tuple[dict, ...] = ()


def plan_v2(config: Config, model: PlanningInput, profile_name: str | None = None,
            top_k: int = 3, *, clock: Callable[[], float] = time.monotonic) -> dict:
    if type(top_k) is not int or not 1 <= top_k <= 20:
        raise ValidationError('top_k must be an integer in 1..20')
    start = clock()
    budget = _Budget(config.planning.candidate_limit, start + config.planning.deadline_ms / 1000, clock)
    selection = resolve_candidates(config, profile_name)
    profile = config.profile_map[selection['profile']]
    if profile.placement.strategy not in ('auto', 'whole_blocks'):
        raise ValidationError('plan-v2 implements only whole_blocks; requested strategy was not changed')
    if profile.objective not in ('interactive_latency', 'capacity'):
        raise ValidationError('plan-v2 cannot score throughput or balanced objectives; request not changed')
    if (profile.model_manifest_ref != model.manifest_ref
            or profile.workload.context_tokens != model.context_tokens
            or profile.workload.max_output_tokens != model.max_output_tokens
            or profile.workload.max_active_requests != 1
            or profile.workload.task != 'text_generation'):
        raise ValidationError('planning input must match the exact profile manifest and C1 workload')
    devices = {d['id']: d for d in selection['eligible_compute_devices']}
    device_ids = sorted(devices)
    budgets = selection['physical_pool_budgets']
    required = set(selection['required_devices'])
    required_nodes = set(selection['required_nodes'])
    local_group = set(selection['required_any_local_gpu'])
    links = {}
    for link in model.links:
        links.setdefault((link.source, link.target), []).append(link)
    rejections: Counter = Counter()
    solutions: list[dict] = []
    completed = 0

    def route(source: str, target: str, payload: int, coord: str, role: str) -> dict | None:
        if source == target:
            return {'role': role, 'source': source, 'target': target, 'payload_bytes': payload,
                    'link_ids': [], 'cost_us': 0}
        a, b = 'device:' + source, 'device:' + target
        hops = [(a, b)] if model.route_mode == 'direct' else [(a, 'node:' + coord), ('node:' + coord, b)]
        chosen = []
        for x, y in hops:
            options = links.get((x, y), [])
            if not options:
                return None
            chosen.append(min(options, key=lambda edge: (edge.cost_us(payload), edge.id)))
        return {'role': role, 'source': source, 'target': target, 'payload_bytes': payload,
                'link_ids': [l.id for l in chosen], 'cost_us': sum(l.cost_us(payload) for l in chosen)}

    def totals(state: _State) -> dict[str, int]:
        amounts = state.resident.copy()
        for (_, pool), amount in state.workspace.items():
            amounts[pool] = amounts.get(pool, 0) + amount
        return amounts

    def fits(state: _State) -> bool:
        for pool, amount in totals(state).items():
            if amount and budgets[pool] is None:
                rejections['unknown_pool_budget'] += 1
                return False
            if amount > (budgets[pool] or 0):
                rejections['pool_budget_exceeded'] += 1
                return False
        return True

    def extend(state: _State, dev: str, coord: str) -> _State | None:
        budget.consume()
        i = len(state.owners)
        placement = model.units[i].placements.get(dev)
        if placement is None:
            rejections['missing_device_profile'] += 1
            return None
        used = set(state.owners)
        if dev in used and state.owners[-1] != dev:
            rejections['noncontiguous_device_block'] += 1
            return None
        new_used = used | {dev}
        node_ids = {devices[d]['node'] for d in new_used}
        if len(new_used) > selection['max_compute_devices'] or len(node_ids) > selection['max_compute_nodes']:
            rejections['owner_maximum'] += 1
            return None
        remaining = len(model.units) - i - 1
        if (len(required - new_used) > remaining or len(required_nodes - node_ids) > remaining
                or len(new_used) + remaining < selection['min_compute_devices']
                or len(node_ids) + remaining < selection['min_compute_nodes']):
            rejections['unreachable_required_owners'] += 1
            return None
        resident = state.resident.copy()
        for pool, amount in placement.resident.items():
            resident[pool] = resident.get(pool, 0) + amount
        if dev not in used:
            for pool, amount in model.worker_resident[dev].items():
                resident[pool] = resident.get(pool, 0) + amount
        workspace = state.workspace.copy()
        for pool, amount in placement.workspace.items():
            key = (dev, pool)
            workspace[key] = max(workspace.get(key, 0), amount)
        routes = state.routes
        transfer_us = state.transfer_us
        if i and state.owners[-1] != dev:
            transfer = route(state.owners[-1], dev, model.units[i-1].output_bytes, coord, 'boundary')
            if transfer is None:
                rejections['missing_forward_route'] += 1
                return None
            routes += (transfer,)
            transfer_us += transfer['cost_us']
        next_state = _State(state.owners + (dev,), resident, workspace,
                            state.compute_us + placement.decode_us, transfer_us, routes)
        return next_state if fits(next_state) else None

    def score(sol: dict) -> tuple:
        if profile.objective == 'capacity':
            return (sol['compute_node_count'], sol['compute_device_count'], sol['estimated_decode_us'], sol['plan_sha256'])
        return (sol['estimated_decode_us'], sol['compute_node_count'], sol['compute_device_count'], sol['plan_sha256'])

    def finish(state: _State, coord: str) -> None:
        nonlocal completed
        used = set(state.owners)
        nodes = {devices[d]['node'] for d in used}
        if (not required <= used or not required_nodes <= nodes or (local_group and not used & local_group)
                or len(used) < selection['min_compute_devices'] or len(nodes) < selection['min_compute_nodes']):
            rejections['final_owner_requirement'] += 1
            return
        feedback = route(state.owners[-1], state.owners[0], model.feedback_bytes, coord, 'token_feedback')
        if feedback is None:
            rejections['missing_feedback_route'] += 1
            return
        completed += 1
        pool_usage = totals(state)
        segments = []
        for i, owner in enumerate(state.owners):
            if not segments or segments[-1]['device'] != owner:
                segments.append({'device': owner, 'first_unit': i, 'last_unit_exclusive': i + 1})
            else:
                segments[-1]['last_unit_exclusive'] = i + 1
        sol = {
            'coordinator_node': coord, 'compute_nodes': sorted(nodes), 'compute_devices': sorted(used),
            'compute_node_count': len(nodes), 'compute_device_count': len(used), 'segments': segments,
            'unit_ids': [u.id for u in model.units], 'owners': list(state.owners),
            'compute_us': state.compute_us, 'transfer_us': state.transfer_us + feedback['cost_us'],
            'estimated_decode_us': state.compute_us + state.transfer_us + feedback['cost_us'],
            'pool_usage_bytes': pool_usage,
            'routes': list(state.routes) + ([feedback] if feedback['link_ids'] else []),
            'executable': False, 'qualified': False,
        }
        identity = {'config': config.fingerprint, 'planning': model.fingerprint, 'profile': profile.name, 'plan': sol}
        sol['plan_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if all(p['plan_sha256'] != sol['plan_sha256'] for p in solutions):
            solutions.append(sol)
            solutions.sort(key=score)
            del solutions[top_k:]

    def initial(coord: str) -> _State | None:
        budget.consume()
        s = _State(resident=model.coordinator_resident[coord].copy())
        return s if fits(s) else None

    try:
        for coord in selection['coordinator_candidates']:
            for dev in device_ids:
                state = initial(coord)
                if state is None:
                    continue
                for _ in model.units:
                    state = extend(state, dev, coord)
                    if state is None:
                        break
                if state is not None:
                    finish(state, coord)
        for coord in selection['coordinator_candidates']:
            root = initial(coord)
            if root is None:
                continue
            stack = [(root, iter(device_ids))]
            while stack:
                state, choices = stack[-1]
                dev = next(choices, None)
                if dev is None:
                    stack.pop()
                    continue
                next_state = extend(state, dev, coord)
                if next_state is None:
                    continue
                if len(next_state.owners) == len(model.units):
                    finish(next_state, coord)
                else:
                    order = [dev] + [d for d in device_ids if d != dev]
                    stack.append((next_state, iter(order)))
    except _BudgetExpired:
        pass
    incomplete = budget.reason is not None
    status = 'SEARCH_INCOMPLETE' if incomplete else ('CANDIDATES_FOUND' if solutions else 'NO_CANDIDATE_IN_SEARCH_SPACE')
    return {
        'result_schema': 'tensormeld/planning-result-v1', 'status': status,
        'config_sha256': config.fingerprint, 'planning_input_sha256': model.fingerprint,
        'profile': profile.name, 'objective': profile.objective, 'provenance': 'synthetic',
        'qualified': False, 'executable': False, 'best': solutions[0] if solutions else None,
        'candidates': solutions,
        'search': {'complete': not incomplete, 'reason': budget.reason, 'work_units': budget.used,
                   'work_limit': budget.limit, 'deadline_ms': config.planning.deadline_ms,
                   'elapsed_ms': max(0, int((clock() - start) * 1000)),
                   'complete_candidates_evaluated': completed, 'retained_candidates': len(solutions),
                   'max_stack_depth': len(model.units), 'rejections': dict(sorted(rejections.items()))},
        'warnings': [
            'Synthetic sequential C1 decode estimates; not measurements or inference execution.',
            'Only contiguous whole-unit ownership is searched; each device owns at most one segment.',
            'Coordinator, worker and per-worker peak workspace all consume physical-pool budgets.',
            'Weights and state are supplied together as resident_bytes, not inferred from GGUF size.',
            'Missing input profiles/routes are not evidence of physical impossibility.',
            'Single-owner seeds and depth-first order do not guarantee an optimum on budget exhaustion.',
            'No multirail bandwidth addition, pipeline overlap, prefill or request-throughput prediction.',
            'Backend approval and live memory reservations remain required before any execution.',
        ],
    }
