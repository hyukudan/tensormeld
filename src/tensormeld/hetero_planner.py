"""Placement planner for a model split across unlike devices.

Input: a ``tensormeld/model-parts-v1`` descriptor, a
``tensormeld/hetero-hardware-v1`` descriptor and a request shape (prompt
tokens, context, output tokens, rows per decode step).  Output: ranked
candidate placements with an explicit cost breakdown and the chosen mode:

``single``          whole model on one device
``engine-group``    whole model on one declared device group (sharded parts split)
``replicas``        whole model on each of several disjoint device sets
``hetero-pipeline`` layers ``0..c`` on one set, ``c+1..`` on another
``hetero-parts``    parts placed by need: hot-per-byte parts on the fast set,
                    capacity-heavy parts on the large set, per-layer round trips
``prefill-decode``  prefill runs on one placement, decode on another, and the
                    request state moves between them once
``dense+hot``       (only with a hot-expert placement file) every dense part on the
                    parts-host, routed experts split per layer and per expert id as the
                    file says; ids no local device holds are served remotely

The cost model is deliberately simple and fully reported, so a measured
receipt can replace any term:

* decode: per layer, anchors on the home set (bytes / bandwidth or ops /
  throughput, whichever is larger, plus a per-layer step floor), branches in
  parallel (a remote branch pays a round trip: route latency + payload / rate +
  a crossing overhead on both ends), a home change pays the stream bytes,
  sharded parts on a group pay an exchange over the group link;
* prefill: the same walk with prompt rows per part (``all``/``last``/bounded),
  weights re-read once per chunk, no overlap between layers (an upper bound);
* prefill-decode: the state of every part whose device set changes moves once.

Nothing here is a measurement and nothing becomes executable: results carry
``qualified: false, executable: false`` and list the unmeasured inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .hetero_hardware import Hardware, HwDevice
from .hetero_model import ModelParts, Part
from .hot_placement import Expectation, HotInput, check_against_model, resolve_devices
from .prefill_route import MAX_TOKENS as ROUTE_MAX_TOKENS, TtftCurves, crossover, route_section
from .schema import ValidationError

HETERO_PLAN_SCHEMA = "tensormeld/hetero-plan-v1"
OBJECTIVES = ("latency", "decode", "prefill", "throughput")
MAX_CANDIDATES_REPORTED = 20
MAX_PD_SIDE = 8


@dataclass(frozen=True)
class Request:
    prompt_tokens: int = 8192
    context_tokens: int = 8192
    output_tokens: int = 256
    decode_rows: int = 1
    slots: int = 1
    prefill_chunk_rows: int = 4096
    with_draft: bool = False
    objective: str = "latency"
    prefill_local_max_tokens: int | None = None  # configured length-routing threshold (None: measured or model)

    def validate(self) -> "Request":
        if self.objective not in OBJECTIVES:
            raise ValidationError(f"objective: expected one of {OBJECTIVES}")
        for name in ("prompt_tokens", "context_tokens", "decode_rows", "slots", "prefill_chunk_rows"):
            if not 1 <= getattr(self, name) <= 1 << 24:
                raise ValidationError(f"{name}: out of range")
        if not 0 <= self.output_tokens <= 1 << 24:
            raise ValidationError("output_tokens: out of range")
        if self.context_tokens < self.prompt_tokens:
            raise ValidationError("context_tokens: must be >= prompt_tokens")
        if self.prefill_local_max_tokens is not None and not 0 <= self.prefill_local_max_tokens <= 1 << 24:
            raise ValidationError("prefill_local_max_tokens: out of range")
        return self


class _Ctx:
    def __init__(self, model: ModelParts, hw: Hardware, req: Request):
        self.model = model
        self.hw = hw
        self.req = req
        self.parts = tuple(p for p in model.parts if req.with_draft or not p.optional)
        available = [d.id for d in hw.devices if d.availability != "unavailable"]
        self.sets: dict[str, tuple[str, ...]] = {d: (d,) for d in available}
        for g in hw.groups:
            if all(m in available for m in g.devices):
                self.sets[g.id] = g.devices
        self.group_link = {g.id: hw.link(g.link) for g in hw.groups}
        self.group = {g.id: g for g in hw.groups}

    # --- helpers -----------------------------------------------------------
    def devs(self, set_id: str) -> tuple[HwDevice, ...]:
        return tuple(self.hw.device(d) for d in self.sets[set_id])

    def entry_node(self, set_id: str) -> str:
        return self.hw.device(self.sets[set_id][0]).node

    def route_us(self, src: str, dst: str, payload: int, tally: dict) -> float:
        """Transfer ``payload`` bytes from set ``src`` to set ``dst`` (entry devices)."""
        if src == dst or payload <= 0:
            return 0.0
        links = self.hw.route(self.entry_node(src), self.entry_node(dst))
        t = 0.0
        for link in links:
            t += link.cost_us(payload)
            tally[link.id] = tally.get(link.id, 0) + payload
        return t

    def group_us(self, set_id: str, payload: int, tally: dict) -> float:
        if len(self.sets[set_id]) < 2 or payload <= 0:
            return 0.0
        link = self.group_link[set_id]
        tally[link.id] = tally.get(link.id, 0) + payload
        return link.cost_us(payload)

    def shards(self, p: Part, set_id: str) -> int:
        return len(self.sets[set_id]) if p.shardable else 1

    def supports(self, p: Part, set_id: str) -> bool:
        if set_id in self.group and not self.group[set_id].supports_part(p.requires):
            return False
        return all(d.supports_part(p.requires) for d in self.devs(set_id))

    def storage_problem(self, p: Part, set_id: str) -> str | None:
        """Disk-backed tables must already be on the node (TensorMeld never copies weights implicitly)."""
        if not p.storage_bytes:
            return None
        for d in self.devs(set_id):
            node = self.hw.node(d.node)
            if self.model.model_id not in node.model_files:
                return (f"{d.id}: {p.id} reads a {p.storage_bytes / 2**30:.0f} GiB table that is not on node "
                        f"{node.id} (copy it first)")
        return None


def _part_decode_us(ctx: _Ctx, p: Part, set_id: str, rows: int, tally: dict) -> float:
    n = ctx.shards(p, set_id)
    share = p.shard_share(n)
    t = 0.0
    for d in ctx.devs(set_id):
        t = max(t, max(p.decode_bytes(rows) * share / d.mem_bytes_per_s,
                       p.decode_ops_per_row * rows * share / d.decode_ops_per_s) * 1e6)
    if n > 1 and p.role == "anchor" and p.out_bytes_per_row:
        t += ctx.group_us(set_id, rows * p.out_bytes_per_row, tally)
    return t


def _part_prefill_us(ctx: _Ctx, p: Part, set_id: str, rows: int, tally: dict) -> float:
    if rows <= 0:
        return 0.0
    n = ctx.shards(p, set_id)
    share = p.shard_share(n)
    chunks = math.ceil(rows / ctx.req.prefill_chunk_rows)
    chunk_rows = min(rows, ctx.req.prefill_chunk_rows)
    weight_read = min(p.weight_bytes, p.decode_bytes(chunk_rows)) * share * chunks
    t = 0.0
    for d in ctx.devs(set_id):
        t = max(t, max(weight_read / d.mem_bytes_per_s,
                       p.prefill_ops_per_row * rows * share / d.prefill_rate(p.requires)) * 1e6)
    if n > 1 and p.role == "anchor" and p.out_bytes_per_row:
        t += ctx.group_us(set_id, rows * p.out_bytes_per_row, tally)
        t += (chunks - 1) * ctx.group_link[set_id].latency_us
    return t


def _step_floor(ctx: _Ctx, set_id: str) -> float:
    return max(d.step_overhead_us for d in ctx.devs(set_id))


def _cross_overhead(ctx: _Ctx, a: str, b: str) -> float:
    return max(d.crossing_overhead_us for d in ctx.devs(a)) + max(d.crossing_overhead_us for d in ctx.devs(b))


def _homes(ctx: _Ctx, assign: dict[str, str]) -> list[str]:
    homes = []
    for layer in range(ctx.model.layers):
        sets = {assign[p.id] for p in ctx.model.anchors(layer)}
        if len(sets) != 1:
            raise ValidationError(f"layer {layer}: anchors on more than one device set")
        homes.append(sets.pop())
    return homes


def evaluate_decode(ctx: _Ctx, assign: dict[str, str], rows: int | None = None) -> dict:
    rows = rows or ctx.req.decode_rows
    model = ctx.model
    homes = _homes(ctx, assign)
    tally: dict[str, int] = {}
    hidden: dict[str, int] = {}
    busy: dict[str, float] = {}
    wire_us = 0.0
    crossings = 0
    total = 0.0
    parts_by_layer: dict[int, list[Part]] = {}
    for p in ctx.parts:
        parts_by_layer.setdefault(p.layer, []).append(p)

    def add_busy(set_id, us):
        for d in ctx.sets[set_id]:
            busy[d] = busy.get(d, 0.0) + us

    # Lookups: critical only when the first layer waits for them.
    for p in (q for q in ctx.parts if q.role == "lookup"):
        consumer = homes[min(max(p.layer, 0), model.layers - 1)]
        where = assign[p.id]
        t = _part_decode_us(ctx, p, where, rows, {})
        add_busy(where, t)
        if where != consumer:
            local: dict[str, int] = {}
            move = ctx.route_us(where, consumer, rows * p.out_bytes_per_row, local)
            for k, v in local.items():
                tally[k] = tally.get(k, 0) + v
            if p.layer <= 0:
                cost = move + _cross_overhead(ctx, where, consumer)
                total += t + cost
                wire_us += cost
                crossings += 1
            else:
                for k, v in local.items():
                    hidden[k] = hidden.get(k, 0) + v
    prev = homes[0]
    for layer in range(model.layers):
        home = homes[layer]
        if home != prev:
            payload = rows * (model.stream_bytes_per_row + model.side_bytes_after(layer - 1))
            cost = ctx.route_us(prev, home, payload, tally)
            cost += ctx.group_us(home, payload, tally) + _cross_overhead(ctx, prev, home)
            total += cost
            wire_us += cost
            crossings += 1
        anchors = [p for p in parts_by_layer.get(layer, []) if p.role == "anchor"]
        a = sum(_part_decode_us(ctx, p, home, rows, tally) for p in anchors) + _step_floor(ctx, home)
        add_busy(home, a)
        branches = [p for p in parts_by_layer.get(layer, []) if p.role == "branch"]
        local_t = 0.0
        remote: dict[str, list[Part]] = {}
        for p in branches:
            where = assign[p.id]
            if where == home:
                t = _part_decode_us(ctx, p, home, rows, tally)
                if ctx.shards(p, home) > 1:
                    t += ctx.group_us(home, rows * p.out_bytes_per_row, tally)
                local_t += t
            else:
                remote.setdefault(where, []).append(p)
        add_busy(home, local_t)
        ffn = local_t
        for where, ps in remote.items():
            inb = rows * max(p.in_bytes_per_row for p in ps)
            outb = rows * max(p.out_bytes_per_row for p in ps)
            send = ctx.route_us(home, where, inb, tally) + ctx.group_us(where, inb, tally)
            comp = sum(_part_decode_us(ctx, p, where, rows, tally) for p in ps) + _step_floor(ctx, where)
            shards = max(ctx.shards(p, where) for p in ps)
            per_shard = any(p.combine == "per-shard" for p in ps)
            back = ctx.group_us(where, outb if shards > 1 else 0, tally)
            back += ctx.route_us(where, home, outb * (shards if per_shard else 1), tally)
            over = _cross_overhead(ctx, home, where)
            add_busy(where, comp)
            wire_us += send + back + over
            crossings += 1
            ffn = max(ffn, send + comp + back + over)
        total += a + ffn
        prev = home
    for p in (q for q in ctx.parts if q.role == "head"):
        where = assign[p.id]
        if where != prev:
            cost = ctx.route_us(prev, where, rows * model.stream_bytes_per_row, tally) + _cross_overhead(ctx, prev, where)
            total += cost
            wire_us += cost
            crossings += 1
        t = _part_decode_us(ctx, p, where, rows, tally)
        add_busy(where, t)
        total += t
        first = homes[0]
        emb = [assign[q.id] for q in ctx.parts if q.role == "lookup" and q.layer < 0]
        back_to = emb[0] if emb else first
        if back_to != where:
            cost = ctx.route_us(where, back_to, rows * model.token_bytes, tally) + _cross_overhead(ctx, where, back_to)
            total += cost
            wire_us += cost
            crossings += 1
        prev = where
    return {
        "rows": rows,
        "step_ms": round(total / 1e3, 3),
        "tokens_per_s": round(rows * 1e6 / total, 2) if total else None,
        "wire_ms": round(wire_us / 1e3, 3),
        "wire_share": round(wire_us / total, 3) if total else 0.0,
        "crossings_per_step": crossings,
        "link_bytes_per_step": dict(sorted(tally.items())),
        "link_bytes_prefetchable": dict(sorted(hidden.items())),
        "device_busy_ms": {k: round(v / 1e3, 3) for k, v in sorted(busy.items())},
    }


def evaluate_prefill(ctx: _Ctx, assign: dict[str, str], prompt: int | None = None) -> dict:
    prompt = prompt or ctx.req.prompt_tokens
    model = ctx.model
    homes = _homes(ctx, assign)
    tally: dict[str, int] = {}
    busy: dict[str, float] = {}
    wire_us = 0.0
    total = 0.0
    chunks_of = lambda rows: max(1, math.ceil(rows / ctx.req.prefill_chunk_rows))  # noqa: E731

    def add_busy(set_id, us):
        for d in ctx.sets[set_id]:
            busy[d] = busy.get(d, 0.0) + us

    for p in (q for q in ctx.parts if q.role == "lookup"):
        consumer = homes[min(max(p.layer, 0), model.layers - 1)]
        where = assign[p.id]
        rows = p.rows_for_prompt(prompt)
        t = _part_prefill_us(ctx, p, where, rows, tally)
        if where != consumer:
            t += ctx.route_us(where, consumer, rows * p.out_bytes_per_row, tally)
        add_busy(where, t)
        total += t
    prev = homes[0]
    for layer in range(model.layers):
        home = homes[layer]
        layer_parts = [p for p in ctx.parts if p.layer == layer]
        anchors = [p for p in layer_parts if p.role == "anchor"]
        rows_in = max(p.rows_for_prompt(prompt) for p in anchors)
        if home != prev:
            payload = rows_in * model.stream_bytes_per_row + model.side_bytes_after(layer - 1) * prompt
            cost = ctx.route_us(prev, home, payload, tally) + ctx.group_us(home, payload, tally)
            cost += _cross_overhead(ctx, prev, home)
            total += cost
            wire_us += cost
        chunks = chunks_of(rows_in)
        a = sum(_part_prefill_us(ctx, p, home, p.rows_for_prompt(prompt), tally) for p in anchors)
        a += _step_floor(ctx, home) * chunks
        add_busy(home, a)
        total += a
        for p in (q for q in layer_parts if q.role == "branch"):
            where = assign[p.id]
            rows = p.rows_for_prompt(prompt)
            t = _part_prefill_us(ctx, p, where, rows, tally)
            if ctx.shards(p, where) > 1:
                t += ctx.group_us(where, rows * p.out_bytes_per_row, tally)
            if where != home:
                shards = ctx.shards(p, where)
                w = ctx.route_us(home, where, rows * p.in_bytes_per_row, tally)
                w += ctx.group_us(where, rows * p.in_bytes_per_row, tally)
                w += ctx.route_us(where, home, rows * p.out_bytes_per_row * (shards if p.combine == "per-shard" else 1), tally)
                links = ctx.hw.route(ctx.entry_node(home), ctx.entry_node(where))
                w += (chunks_of(rows) - 1) * 2 * sum(link.latency_us for link in links)
                w += chunks_of(rows) * _cross_overhead(ctx, home, where)
                wire_us += w
                t += w
            add_busy(where, t)
            total += t
        prev = home
    for p in (q for q in ctx.parts if q.role == "head"):
        where = assign[p.id]
        if where != prev:
            cost = ctx.route_us(prev, where, model.stream_bytes_per_row, tally) + _cross_overhead(ctx, prev, where)
            total += cost
            wire_us += cost
        total += _part_prefill_us(ctx, p, where, 1, tally)
    return {
        "prompt_tokens": prompt,
        "seconds": round(total / 1e6, 3),
        "tokens_per_s": round(prompt * 1e6 / total, 1) if total else None,
        "wire_s": round(wire_us / 1e6, 3),
        "link_bytes": dict(sorted(tally.items())),
        "device_busy_s": {k: round(v / 1e6, 3) for k, v in sorted(busy.items())},
    }


def memory_by_device(ctx: _Ctx, *assigns: dict[str, str]) -> dict[str, int]:
    """Resident bytes per device for the union of one or more placements."""
    placed: dict[tuple[str, str], int] = {}
    for assign in assigns:
        for p in ctx.parts:
            set_id = assign[p.id]
            share = p.shard_share(ctx.shards(p, set_id))
            for d in ctx.sets[set_id]:
                state = p.state_bytes(ctx.req.context_tokens) * ctx.req.slots
                b = int(p.weight_bytes * share) + state
                placed[(p.id, d)] = max(placed.get((p.id, d), 0), b)
    usage: dict[str, int] = {}
    for (_, d), b in placed.items():
        usage[d] = usage.get(d, 0) + b
    return usage


def _fits(ctx: _Ctx, usage: dict[str, int]) -> list[str]:
    problems = []
    for d, b in sorted(usage.items()):
        dev = ctx.hw.device(d)
        if b > dev.usable_bytes:
            problems.append(f"{d}: needs {b / 2**30:.1f} GiB > usable {dev.usable_bytes / 2**30:.1f} GiB")
    return problems


def _unsupported(ctx: _Ctx, assign: dict[str, str]) -> list[str]:
    out = {f"{assign[p.id]} lacks {','.join(p.requires)} ({p.kind})" for p in ctx.parts
           if not ctx.supports(p, assign[p.id])}
    out |= {x for x in (ctx.storage_problem(p, assign[p.id]) for p in ctx.parts) if x}
    return sorted(out)


# --- candidate generation ----------------------------------------------------

def _whole(ctx: _Ctx, set_id: str) -> dict[str, str]:
    return {p.id: set_id for p in ctx.parts}


def _whole_offload(ctx: _Ctx, set_id: str) -> dict[str, str] | None:
    """Whole model on ``set_id`` except lookups, which go to another device on the same node
    (host memory next to a discrete GPU) when that device supports them and has room."""
    nodes = {d.node for d in ctx.devs(set_id)}
    hosts = [s for s, devs in ctx.sets.items()
             if len(devs) == 1 and s != set_id and ctx.hw.device(devs[0]).node in nodes]
    if not hosts:
        return None
    assign = _whole(ctx, set_id)
    moved = False
    for host in hosts:
        room = ctx.hw.device(host).usable_bytes
        for p in ctx.parts:
            if (p.role == "lookup" and assign[p.id] == set_id and ctx.supports(p, host)
                    and not ctx.storage_problem(p, host)):
                need = p.weight_bytes + p.state_bytes(ctx.req.context_tokens) * ctx.req.slots
                if need <= room:
                    assign[p.id] = host
                    room -= need
                    moved = True
    return assign if moved else None


def _pipeline(ctx: _Ctx, first: str, second: str, cut: int) -> dict[str, str]:
    return _keep_tables(ctx, {p.id: (first if p.layer <= cut else second) for p in ctx.parts}, (second, first))


def _keep_tables(ctx: _Ctx, assign: dict[str, str], alternatives: tuple[str, ...]) -> dict[str, str]:
    """A disk-backed lookup whose table is not on its node moves to a set that has the table
    (it then runs as a remote, prefetchable lookup); weights are never copied implicitly."""
    for p in ctx.parts:
        if p.role == "lookup" and ctx.storage_problem(p, assign[p.id]):
            for alt in alternatives:
                if ctx.supports(p, alt) and not ctx.storage_problem(p, alt):
                    assign[p.id] = alt
                    break
    return assign


def _units(ctx: _Ctx) -> list[tuple[str, int, float, tuple[Part, ...]]]:
    """Placement units: all anchors of a layer together, every other part alone."""
    units = []
    for layer in range(ctx.model.layers):
        ps = tuple(p for p in ctx.parts if p.layer == layer and p.role == "anchor")
        units.append(("anchors", layer, _density(ps), ps))
    for p in ctx.parts:
        if p.role != "anchor":
            units.append((p.role, p.layer, _density((p,)), (p,)))
    return units


def _density(ps: tuple[Part, ...]) -> float:
    w = sum(p.weight_bytes for p in ps)
    return sum(p.decode_read_bytes for p in ps) / w if w else 0.0


def _unit_bytes(ctx: _Ctx, ps: tuple[Part, ...], set_id: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in ps:
        share = p.shard_share(ctx.shards(p, set_id))
        for d in ctx.sets[set_id]:
            out[d] = out.get(d, 0) + int(p.weight_bytes * share) + p.state_bytes(ctx.req.context_tokens) * ctx.req.slots
    return out


def _split_variants(ctx: _Ctx, fast: str, big: str) -> list[tuple[str, dict[str, str]]]:
    """Hot-per-byte units on ``fast``; branch units then fill ``fast`` in layer order.

    Yields one candidate per number ``k`` of leading layers whose branches also
    live on ``fast`` (every ``k`` that fits), plus the pure density-greedy fill.
    """
    units = _units(ctx)
    cap = {d.id: d.usable_bytes for d in ctx.devs(fast)}

    def fits_add(usage, add):
        return all(usage.get(d, 0) + b <= cap[d] for d, b in add.items())

    def place(order):
        usage: dict[str, int] = {}
        assign = {p.id: big for p in ctx.parts}
        for _, _, _, ps in order:
            if not all(ctx.supports(p, fast) and not ctx.storage_problem(p, fast) for p in ps):
                continue
            add = _unit_bytes(ctx, ps, fast)
            if fits_add(usage, add):
                for d, b in add.items():
                    usage[d] = usage.get(d, 0) + b
                for p in ps:
                    assign[p.id] = fast
        return assign, usage

    out = []
    greedy = sorted(units, key=lambda u: (-u[2], u[1]))
    out.append(("density-greedy", place(greedy)[0]))
    hot = [u for u in units if u[0] in ("anchors", "head") or (u[0] == "branch" and u[2] >= 0.5)]
    branches = sorted((u for u in units if u[0] == "branch" and u[2] < 0.5), key=lambda u: u[1])
    rest = [u for u in units if u not in hot and u not in branches]
    base_assign, base_usage = place(hot)
    if any(base_assign[p.id] != fast for u in hot for p in u[3]):
        return out  # the hot set itself does not fit: only the greedy fill applies
    branch_layers = sorted({u[1] for u in branches})
    for k in range(len(branch_layers) + 1):
        chosen = [u for u in branches if u[1] < (branch_layers[k - 1] + 1 if k else -1)]
        assign, usage = dict(base_assign), dict(base_usage)
        ok = True
        for _, _, _, ps in chosen:
            add = _unit_bytes(ctx, ps, fast)
            if not all(ctx.supports(p, fast) for p in ps) or not fits_add(usage, add):
                ok = False
                break
            for d, b in add.items():
                usage[d] = usage.get(d, 0) + b
            for p in ps:
                assign[p.id] = fast
        if not ok:
            break
        # Lookups go next to their consumer when they still fit, else stay on ``big``.
        for _, _, _, ps in rest:
            for p in ps:
                if p.role != "lookup":
                    continue
                anchors = ctx.model.anchors(min(max(p.layer, 0), ctx.model.layers - 1))
                consumer = assign[anchors[0].id]
                if consumer == fast and ctx.supports(p, fast) and not ctx.storage_problem(p, fast):
                    add = _unit_bytes(ctx, (p,), fast)
                    if fits_add(usage, add):
                        for d, b in add.items():
                            usage[d] = usage.get(d, 0) + b
                        assign[p.id] = fast
        out.append((f"hot+branches[0..{branch_layers[k - 1] if k else -1}]", assign))
    return out


def _mode_of(ctx: _Ctx, assign: dict[str, str]) -> str:
    used = {assign[p.id] for p in ctx.parts if p.role != "lookup"}
    if len(used) == 1:
        return "engine-group" if len(ctx.sets[used.pop()]) > 1 else "single"
    homes = _homes(ctx, assign)
    contiguous = all(assign[p.id] == homes[min(max(p.layer, 0), ctx.model.layers - 1)]
                     for p in ctx.parts if p.role in ("anchor", "branch"))
    return "hetero-pipeline" if contiguous else "hetero-parts"


def _describe(ctx: _Ctx, assign: dict[str, str]) -> dict:
    by_set: dict[str, dict] = {}
    for p in ctx.parts:
        e = by_set.setdefault(assign[p.id], {"parts": 0, "weight_bytes": 0, "kinds": {}})
        e["parts"] += 1
        e["weight_bytes"] += p.weight_bytes
        layers = e["kinds"].setdefault(p.kind, [])
        layers.append(p.layer)
    for e in by_set.values():
        e["kinds"] = {k: _ranges(v) for k, v in sorted(e["kinds"].items())}
    return dict(sorted(by_set.items()))


def _layout(ctx: _Ctx, assign: dict[str, str]) -> dict:
    """Stages of a layer pipeline, in stream order, with the exact parts each set runs.

    A stage owns the layers ``[first, end)`` whose anchors live on its set (the first stage
    also owns layer -1, the last the head). ``parts`` are the stage's own parts on its set;
    ``remote_parts`` are parts of its layers placed on another set (lookups kept where their
    table is); ``serves_parts`` are parts of other stages' layers that this set runs for them.
    """
    homes = _homes(ctx, assign)
    bounds = []
    start = 0
    for layer in range(1, len(homes) + 1):
        if layer == len(homes) or homes[layer] != homes[start]:
            bounds.append((homes[start], start, layer))
            start = layer
    stages = []
    last = len(bounds) - 1
    for i, (set_id, first, end) in enumerate(bounds):
        def owns(p: Part) -> bool:
            if p.layer < 0:
                return i == 0
            if p.layer >= ctx.model.layers:
                return i == last
            return first <= p.layer < end
        stages.append({
            "set": set_id,
            "devices": list(ctx.sets[set_id]),
            "layers": [first, end],
            "has_embedding": i == 0,
            "has_head": i == last,
            "parts": sorted(p.id for p in ctx.parts if owns(p) and assign[p.id] == set_id),
            "remote_parts": sorted(p.id for p in ctx.parts if owns(p) and assign[p.id] != set_id),
            "serves_parts": sorted(p.id for p in ctx.parts if not owns(p) and assign[p.id] == set_id),
        })
    return {"stages": stages, "cuts_after_layer": [s["layers"][1] - 1 for s in stages[:-1]]}


def _ranges(values: list[int]) -> str:
    values = sorted(set(values))
    out = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = v
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def _candidate(ctx: _Ctx, name: str, decode_assign: dict[str, str], prefill_assign: dict[str, str] | None = None) -> dict:
    prefill_assign = prefill_assign or decode_assign
    problems = _unsupported(ctx, decode_assign) + (
        _unsupported(ctx, prefill_assign) if prefill_assign is not decode_assign else [])
    lent = sorted({d for s in set(decode_assign.values()) for d in ctx.sets[s]
                   if ctx.hw.device(d).availability == "prefill-only"})
    if lent:
        problems.append(f"decode placement uses prefill-only device(s) {','.join(lent)}")
    usage = memory_by_device(ctx, decode_assign, prefill_assign)
    problems += _fits(ctx, usage)
    decode = evaluate_decode(ctx, decode_assign)
    prefill = evaluate_prefill(ctx, prefill_assign)
    handoff = _handoff(ctx, prefill_assign, decode_assign)
    ttft = prefill["seconds"] + handoff["seconds"]
    mode = _mode_of(ctx, decode_assign)
    if prefill_assign is not decode_assign:
        mode = f"prefill-decode({_mode_of(ctx, prefill_assign)} -> {mode})"
    request_s = ttft + ctx.req.output_tokens * decode["step_ms"] / 1e3 / ctx.req.decode_rows
    devices = sorted(usage)
    return {
        "name": name,
        "mode": mode,
        "feasible": not problems,
        "problems": problems,
        "devices": devices,
        "uses_preemptible": sorted(d for d in devices
                                   if ctx.hw.device(d).availability in ("preemptible", "prefill-only")),
        "decode": decode,
        "prefill": prefill,
        "handoff": handoff,
        "ttft_s": round(ttft, 3),
        "request_s": round(request_s, 3),
        "memory_gib": {d: round(b / 2**30, 2) for d, b in sorted(usage.items())},
        "placement": _describe(ctx, decode_assign),
        "prefill_placement": None if prefill_assign is decode_assign else _describe(ctx, prefill_assign),
        "sets": {s: list(ctx.sets[s]) for s in sorted(set(decode_assign.values()) | set(prefill_assign.values()))},
        "layout": _layout(ctx, decode_assign) if mode == "hetero-pipeline" else None,
        "_assign": decode_assign,
        "_prefill_assign": prefill_assign,
    }


def _handoff(ctx: _Ctx, src: dict[str, str], dst: dict[str, str]) -> dict:
    tally: dict[str, int] = {}
    us = 0.0
    moved = 0
    for p in ctx.parts:
        if src[p.id] == dst[p.id]:
            continue
        b = p.state_bytes(ctx.req.prompt_tokens)
        if not b:
            continue
        moved += b
        us += ctx.route_us(src[p.id], dst[p.id], b, tally) + ctx.group_us(dst[p.id], b, tally)
    return {"state_bytes": moved, "seconds": round(us / 1e6, 3), "link_bytes": dict(sorted(tally.items()))}


def _score(ctx: _Ctx, c: dict) -> float:
    o = ctx.req.objective
    if o == "decode":
        return c["decode"]["step_ms"] / c["decode"]["rows"]
    if o == "prefill":
        return c["ttft_s"]
    if o == "throughput":
        return -c.get("aggregate_tokens_per_s", c["decode"]["tokens_per_s"] or 0)
    return c["request_s"]


def plan_hetero(model: ModelParts, hw: Hardware, req: Request, top_k: int = 5,
                hot: HotInput | None = None, ttft_curves: TtftCurves | None = None) -> dict:
    req = req.validate()
    if not 1 <= top_k <= MAX_CANDIDATES_REPORTED:
        raise ValidationError(f"top_k: expected 1..{MAX_CANDIDATES_REPORTED}")
    ctx = _Ctx(model, hw, req)
    if not ctx.sets:
        raise ValidationError("no available device")
    cands: list[dict] = []
    for s in ctx.sets:
        cands.append(_candidate(ctx, f"whole:{s}", _whole(ctx, s)))
        off = _whole_offload(ctx, s)
        if off:
            cands.append(_candidate(ctx, f"whole:{s}+host-lookups", off))
    names = list(ctx.sets)
    disjoint = [(a, b) for a in names for b in names
                if a != b and not set(ctx.sets[a]) & set(ctx.sets[b])]
    pool: list[dict] = []  # extra feasible placements offered to prefill-decode pairing only
    for a, b in disjoint:
        cuts = []
        for cut in range(model.layers - 1):
            c = _candidate(ctx, f"pipeline:{a}[0..{cut}]>{b}", _pipeline(ctx, a, b, cut))
            if c["feasible"] or c["problems"] and all("prefill-only" in x for x in c["problems"]):
                cuts.append(c)
        cuts.sort(key=lambda c: _score(ctx, c))
        cands.extend(cuts[:1])
        pool.extend(cuts[1:4])
        lent_ok = [c for c in cuts if not c["feasible"]]
        pool.extend(sorted(lent_ok, key=lambda c: c["prefill"]["seconds"])[:4])
        fast_bw = min(d.mem_bytes_per_s for d in ctx.devs(a))
        if fast_bw > max(d.mem_bytes_per_s for d in ctx.devs(b)):
            for vname, assign in _split_variants(ctx, a, b):
                cands.append(_candidate(ctx, f"parts:{a}+{b}:{vname}", assign))
    hot_report = None
    if hot is not None:
        h = _Hot(ctx, hot)
        c = _hot_candidate(ctx, h)
        cands.append(c)
        hot_report = _hot_report(ctx, h, c)
    feasible = [c for c in cands if c["feasible"]]
    # Replicas: several disjoint sets that each hold the whole model.
    wholes = [c for c in feasible if c["mode"] in ("single", "engine-group")]
    replica_sets = _disjoint_cover(ctx, wholes)
    if len(replica_sets) > 1:
        agg = sum(c["decode"]["tokens_per_s"] or 0 for c in replica_sets)
        best_single = min(replica_sets, key=lambda c: c["request_s"])
        rep = dict(best_single)
        rep.update({"name": "replicas:" + "+".join(c["name"].split(":", 1)[1] for c in replica_sets),
                    "mode": "replicas", "aggregate_tokens_per_s": round(agg, 2),
                    "devices": sorted({d for c in replica_sets for d in c["devices"]}),
                    "replicas": [c["name"] for c in replica_sets]})
        rep["uses_preemptible"] = sorted(d for d in rep["devices"] if hw.device(d).availability == "preemptible")
        feasible.append(rep)
        cands.append(rep)
    # Prefill-decode: best prefill placements x best decode placements.
    singles = [c for c in feasible if c["mode"] != "replicas"]
    prefill_side = singles + pool + [c for c in cands if not c["feasible"] and c["problems"]
                                     and all("prefill-only" in x for x in c["problems"])]
    by_prefill = sorted(prefill_side, key=lambda c: c["prefill"]["seconds"])[:MAX_PD_SIDE]
    by_decode = sorted(singles, key=lambda c: c["decode"]["step_ms"])[:MAX_PD_SIDE]
    for p in by_prefill:
        if p.get("_hot"):
            continue  # a dense+hot prefill is its base placement's prefill
        for d in by_decode:
            if p is d or p["_assign"] == d["_assign"]:
                continue
            if d.get("_hot"):
                c = _hot_candidate(ctx, d["_hot"], p["_assign"], f"pd:{p['name']}=>{d['name']}")
            else:
                c = _candidate(ctx, f"pd:{p['name']}=>{d['name']}", d["_assign"], p["_assign"])
            cands.append(c)
            if c["feasible"]:
                feasible.append(c)
    ranked = sorted(feasible, key=lambda c: _score(ctx, c))
    best = ranked[0] if ranked else None
    fallback = None
    if best and best["uses_preemptible"]:
        off = {d: "unavailable" for d in best["uses_preemptible"]}
        try:
            sub = plan_hetero(model, hw.with_overrides(availability=off), req, top_k=1)
            fallback = {"without": sorted(off), "best": sub["best"]}
            fb = sub.get("_best_assigns")
            if fb:
                union = memory_by_device(ctx, best["_assign"], best["_prefill_assign"], *fb)
                for d, b in best.get("_extra_bytes", {}).items():
                    union[d] = union.get(d, 0) + b
                problems = _fits(ctx, union)
                fallback["co_resident"] = {
                    "fits": not problems, "problems": problems,
                    "memory_gib": {d: round(b / 2**30, 2) for d, b in sorted(union.items())},
                    "meaning": "fallback weights stay loaded next to the hetero plan: losing the preemptible "
                               "device costs a drain and a state move, not a reload",
                }
        except ValidationError as exc:
            fallback = {"without": sorted(off), "best": None, "reason": str(exc)}
    prefill_route = _prefill_route(model, hw, req, best, ranked, ttft_curves)
    modes: dict[str, dict] = {}
    for c in ranked:
        key = c["mode"].split("(")[0]
        if key not in modes:
            modes[key] = _public(c)
    unmeasured = sorted({f"device:{d.id}" for d in hw.devices if d.provenance != "measured"} |
                        {f"link:{link.id}" for link in hw.links if link.provenance != "measured"})
    return {
        "schema": HETERO_PLAN_SCHEMA,
        "model": model.model_id,
        "model_fingerprint": model.fingerprint,
        "hardware_fingerprint": hw.fingerprint,
        "request": req.__dict__,
        "status": "OK" if best else "NO_FEASIBLE_PLACEMENT",
        "best": _public(best) if best else None,
        "best_by_mode": modes,
        "candidates": [_public(c) for c in ranked[:top_k]],
        "infeasible": [{"name": c["name"], "problems": c["problems"][:4]} for c in cands if not c["feasible"]][:top_k],
        "fallback": fallback,
        "dense_hot": hot_report,
        "prefill_route": prefill_route,
        "evaluated": len(cands),
        "unmeasured_inputs": unmeasured,
        "qualified": False,
        "executable": False,
        "_best_assigns": (best["_assign"], best["_prefill_assign"]) if best else None,
    }


def _route_pair(best: dict | None, ranked: list[dict]) -> dict | None:
    """The prefill-decode candidate that shares the best placement's decode side (fastest prefill first)."""
    if best is None:
        return None
    if best["_prefill_assign"] is not best["_assign"]:
        return best
    pds = [c for c in ranked if c["_prefill_assign"] is not c["_assign"] and c["_assign"] == best["_assign"]]
    return min(pds, key=lambda c: c["prefill"]["seconds"] + c["handoff"]["seconds"]) if pds else None


def _prefill_route(model: ModelParts, hw: Hardware, req: Request, best: dict | None, ranked: list[dict],
                   curves: TtftCurves | None) -> dict | None:
    """Length routing between prefilling on the decode placement and on the offload placement of the
    best prefill-decode pair; see ``prefill_route``.  None when the plan has no such pair."""
    pd = _route_pair(best, ranked)
    if pd is None:
        return None
    local, offload = pd["_assign"], pd["_prefill_assign"]
    hi = max(req.context_tokens, 1 << 20)
    hi = min(hi, ROUTE_MAX_TOKENS)

    def ctx_at(n: int) -> _Ctx:
        return _Ctx(model, hw, replace(req, prompt_tokens=n, context_tokens=max(n, req.context_tokens)))

    def t_local(n: int) -> float:
        return evaluate_prefill(ctx_at(n), local)["seconds"]

    def t_offload(n: int) -> float:
        c = ctx_at(n)
        return evaluate_prefill(c, offload)["seconds"] + _handoff(c, offload, local)["seconds"]

    model_x = crossover(t_local, t_offload, 1, hi)
    local_name = pd["name"].split("=>", 1)[1] if pd["name"].startswith("pd:") else "decode placement"
    offload_name = pd["name"].split("=>", 1)[0][3:] if pd["name"].startswith("pd:") else "prefill placement"
    return route_section(local_name, offload_name, model_x, curves, req.prefill_local_max_tokens)


def _disjoint_cover(ctx: _Ctx, wholes: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    used: set[str] = set()
    for c in sorted(wholes, key=lambda c: (len(c["devices"]), -(c["decode"]["tokens_per_s"] or 0))):
        devs = set(c["devices"])
        if devs & used:
            continue
        chosen.append(c)
        used |= devs
    return chosen


def _public(c: dict | None) -> dict | None:
    if c is None:
        return None
    if c.get("fallback_of"):
        return c
    return {k: v for k, v in c.items() if not k.startswith("_")}


# --- dense + hot experts -------------------------------------------------------------
#
# Decode step = the planner's own walk with every routed expert local on the parts-host
# (anchors, shared and routed branches, head, lookups as placed), plus, per remote layer,
# the part of the remote branch that the local branch does not cover:
#     max(0, send + expert-server compute + return + crossing - local branch)
# The walk already takes max(local, remote) per layer, so the overlap of shared/local
# experts with the round trip is counted once, here, and never again.  Remote layers and
# cold vectors per step come from the placement file's ``expected`` block.

ROUTED_KIND = "routed-experts"


class _Hot:
    def __init__(self, ctx: _Ctx, hot: HotInput):
        p = hot.placement
        check_against_model(p, ctx.model)
        self.input = hot
        self.placement = p
        self.map = resolve_devices(p, ctx.hw, dict(hot.device_overrides))
        self.host = self.map[p.dense_device]
        server = p.server_device
        self.servers = tuple((s, self.map[f"{server.id}/{s.node}"]) for s in server.servers)
        self.problems: list[str] = []
        self.routed = tuple(q for q in ctx.parts if q.kind == ROUTED_KIND)
        if (sorted(q.layer for q in self.routed) != list(range(ctx.model.layers))
                or any(q.role != "branch" for q in self.routed)):
            raise ValidationError(f"dense+hot: the model needs exactly one {ROUTED_KIND!r} branch part per layer")
        devs = {d for _, d in self.servers}
        for name, dev in sorted(self.map.items()):
            if dev is not None and ctx.hw.device(dev).availability == "unavailable":
                self.problems.append(f"{name} maps to unavailable device {dev}")
        self.server_set = next((sid for sid, members in ctx.sets.items()
                                if set(members) == devs and (len(members) == 1 or sid in ctx.group)), None)
        if self.server_set is None and not self.problems:
            self.problems.append(f"expert servers {sorted(devs)} are not one device or one declared device group")
        for d in p.devices:
            dev = self.map.get(d.id)
            if dev is not None and d.role != "expert-server" and ctx.hw.device(dev).backend != d.kind:
                self.problems.append(f"{d.id} is {d.kind} but maps to {dev} ({ctx.hw.device(dev).backend})")
        self.base: dict[str, str] | None = None
        self.local: dict[str, str] | None = None
        if not self.problems:
            routed = {q.id for q in self.routed}
            base = {q.id: (self.server_set if q.id in routed else self.host) for q in ctx.parts}
            self.base = _keep_tables(ctx, base, (self.server_set, *sorted(devs)))
            self.local = {k: (self.host if k in routed else v) for k, v in self.base.items()}

    def extra_bytes(self) -> dict[str, int]:
        """Hot copies on top of the base placement (dense on the parts-host, all experts on the servers)."""
        out: dict[str, int] = {}
        for d in self.placement.resident_devices:
            dev = self.map.get(d.id)
            if dev is not None and self.placement.expert_count(d.id):
                out[dev] = out.get(dev, 0) + self.placement.resident_bytes(d.id)
        return out


def _hot_memory(ctx: _Ctx, h: _Hot, prefill_assign: dict[str, str]) -> dict[str, int]:
    """Resident bytes: parts as placed, but routed experts charged from the placement file."""
    routed = {q.id for q in h.routed}
    servers = {d for _, d in h.servers}
    placed: dict[tuple[str, str], int] = {}
    for assign, is_base in ((h.base, True), (prefill_assign, False)):
        for p in ctx.parts:
            set_id = assign[p.id]
            if p.id in routed and (is_base or set(ctx.sets[set_id]) & servers):
                continue
            share = p.shard_share(ctx.shards(p, set_id))
            for d in ctx.sets[set_id]:
                b = int(p.weight_bytes * share) + p.state_bytes(ctx.req.context_tokens) * ctx.req.slots
                placed[(p.id, d)] = max(placed.get((p.id, d), 0), b)
    # Hot copies of layer L on a device are a subset of that layer's routed tensor when a
    # prefill placement already keeps the whole tensor there.
    by_layer = {q.layer: q.id for q in h.routed}
    for d in h.placement.resident_devices:
        dev = h.map.get(d.id)
        if dev is None:
            continue
        for layer, count in enumerate(h.placement.per_layer(d.id)):
            if count:
                key = (by_layer[layer], dev)
                placed[key] = max(placed.get(key, 0), count * h.placement.expert_bytes)
    usage: dict[str, int] = {}
    for (_, d), b in placed.items():
        usage[d] = usage.get(d, 0) + b
    for s, d in h.servers:
        usage[d] = usage.get(d, 0) + h.placement.server_bytes(s)
    return usage


def _hot_patterns(h: _Hot, rows: int) -> tuple[Expectation, list[Expectation], list[str]]:
    """The pattern that prices the request's rows, and every pattern worth evaluating."""
    exp = h.placement.expected
    notes = []
    evaluate = [exp["R1"]]
    if rows > 1:
        evaluate += [e for e in exp.values() if e.rows and e.pattern != "R1"]
    match = sorted((e for e in exp.values() if e.rows == rows), key=lambda e: (e.batch, e.pattern))
    main = match[0] if match else exp["R1"]
    if not match and rows > 1:
        notes.append(f"the placement has no expected entry for {rows} rows; R1 remote rates are used "
                     "(optimistic: more rows touch more experts)")
    return main, evaluate, notes


def _hot_decode(ctx: _Ctx, h: _Hot, e: Expectation, rows: int) -> dict:
    p = h.placement
    walk = evaluate_decode(ctx, h.local, rows)
    host = ctx.hw.device(h.host)
    routed = h.routed[0]
    branches = [q for q in ctx.parts if q.role == "branch"]
    local_branch = sum(_part_decode_us(ctx, q, h.host, rows, {}) for q in branches) / ctx.model.layers
    remote_layers = e.per_step(e.remote_layers_per_token)
    cold = e.per_step(e.cold_vectors_per_token)
    n = len(h.servers)
    v = cold / remote_layers if remote_layers else 0.0
    if not v:
        s = 0.0
    elif e.servers_per_remote_layer is not None:
        s = min(float(n), e.servers_per_remote_layer)
    else:  # distinct servers hit by v cold vectors spread evenly over n equal id ranges
        s = min(float(n), max(1.0, n * (1 - (1 - 1 / n) ** v)))
    w = s / n
    per_server = v / s if s else 0.0
    in_b = rows * routed.in_bytes_per_row
    out_b = per_server * routed.out_bytes_per_row
    send_g: dict[str, list[float]] = {}
    back_g: dict[str, list[float]] = {}
    tally: dict[str, float] = {}
    busy: dict[str, float] = {}
    lat = comp = over = 0.0
    for _, dev_id in h.servers:
        dev = ctx.hw.device(dev_id)
        links = ctx.hw.route(host.node, dev.node)
        lat = max(lat, sum(link.latency_us for link in links))
        for link in links:
            for g, b in ((send_g, w * in_b), (back_g, w * out_b)):
                entry = g.setdefault(link.shared_group, [0.0, float(link.bytes_per_s)])
                entry[0] += b
                entry[1] = min(entry[1], link.bytes_per_s)
            tally[link.id] = tally.get(link.id, 0.0) + remote_layers * w * (in_b + out_b)
        t = max(per_server * p.expert_bytes / dev.mem_bytes_per_s,
                per_server * routed.decode_ops_per_row / p.top_k / dev.decode_ops_per_s) * 1e6
        t += dev.step_overhead_us
        comp = max(comp, t)
        over = max(over, host.crossing_overhead_us + dev.crossing_overhead_us)
        busy[dev_id] = remote_layers * w * t
    if v:
        send = lat + max(b * 1e6 / bw for b, bw in send_g.values())
        back = lat + max(b * 1e6 / bw for b, bw in back_g.values())
        remote = send + comp + back + over
    else:
        send = back = comp = over = remote = 0.0
    exposed = max(0.0, remote - local_branch)
    cache = [d.id for d in p.resident_devices if d.role == "expert-cache" and p.expert_count(d.id)]
    ram = e.per_step(e.ram_vectors_per_token) or 0.0
    ram_us = ram * p.expert_bytes * 1e6 / h.input.host_copy_bytes_per_s if cache else 0.0
    notes = []
    if cache and e.ram_vectors_per_token is None:
        notes.append(f"{','.join(cache)} holds experts but the file gives no ram_vectors_per_token: "
                     "host->device copies are not priced")
    tiers = [d.id for d in p.resident_devices if d.role == "expert-tier" and p.expert_count(d.id)]
    if tiers:
        notes.append(f"{','.join(tiers)} experts are priced as parts-host local (peer GPU on the same node)")
    total = walk["step_ms"] * 1e3 + remote_layers * exposed + ram_us
    links = dict(walk["link_bytes_per_step"])
    for k, b in tally.items():
        links[k] = links.get(k, 0) + int(round(b))
    device_busy = dict(walk["device_busy_ms"])
    for k, b in busy.items():
        device_busy[k] = round(device_busy.get(k, 0.0) + b / 1e3, 3)
    wire = walk["wire_ms"] * 1e3 + remote_layers * (send + back + over) + ram_us
    return {
        "rows": rows,
        "step_ms": round(total / 1e3, 3),
        "tokens_per_s": round(rows * 1e6 / total, 2) if total else None,
        "wire_ms": round(wire / 1e3, 3),
        "wire_share": round(wire / total, 3) if total else 0.0,
        "crossings_per_step": round(walk["crossings_per_step"] + remote_layers * s, 2),
        "link_bytes_per_step": dict(sorted(links.items())),
        "link_bytes_prefetchable": walk["link_bytes_prefetchable"],
        "device_busy_ms": dict(sorted(device_busy.items())),
        "hot": {
            "pattern": e.pattern,
            "all_local_step_ms": walk["step_ms"],
            "remote_layers_per_step": round(remote_layers, 4),
            "cold_vectors_per_step": round(cold, 4),
            "vectors_per_remote_layer": round(v, 3),
            "servers_per_remote_layer": round(s, 3),
            "local_branch_us_per_layer": round(local_branch, 2),
            "remote_branch_us": {"send": round(send, 2), "compute": round(comp, 2), "return": round(back, 2),
                                 "crossing": round(over, 2), "total": round(remote, 2)},
            "exposed_us_per_remote_layer": round(exposed, 2),
            "remote_penalty_ms": round(remote_layers * exposed / 1e3, 3),
            "host_copy_ms": round(ram_us / 1e3, 3),
            "notes": notes,
        },
    }


def _hot_candidate(ctx: _Ctx, h: _Hot, prefill_assign: dict[str, str] | None = None,
                   name: str = "dense+hot") -> dict:
    if h.problems:
        return {"name": name, "mode": "dense+hot", "feasible": False, "problems": list(h.problems)}
    base = h.base
    paired = prefill_assign is not None
    prefill_assign = prefill_assign or base
    problems = _unsupported(ctx, base) + (_unsupported(ctx, prefill_assign) if paired else [])
    lent = sorted({d for s in set(base.values()) for d in ctx.sets[s]
                   if ctx.hw.device(d).availability == "prefill-only"})
    if lent:
        problems.append(f"decode placement uses prefill-only device(s) {','.join(lent)}")
    usage = _hot_memory(ctx, h, prefill_assign)
    problems += _fits(ctx, usage)
    main, _, pattern_notes = _hot_patterns(h, ctx.req.decode_rows)
    decode = _hot_decode(ctx, h, main, ctx.req.decode_rows)
    decode["hot"]["notes"] = pattern_notes + decode["hot"]["notes"]
    prefill = evaluate_prefill(ctx, prefill_assign)
    handoff = _handoff(ctx, prefill_assign, base)
    ttft = prefill["seconds"] + handoff["seconds"]
    mode = f"prefill-decode({_mode_of(ctx, prefill_assign)} -> dense+hot)" if paired else "dense+hot"
    request_s = ttft + ctx.req.output_tokens * decode["step_ms"] / 1e3 / ctx.req.decode_rows
    devices = sorted(usage)
    sets = sorted(set(base.values()) | set(prefill_assign.values()))
    return {
        "name": name,
        "mode": mode,
        "feasible": not problems,
        "problems": problems,
        "devices": devices,
        "uses_preemptible": sorted(d for d in devices
                                   if ctx.hw.device(d).availability in ("preemptible", "prefill-only")),
        "decode": decode,
        "prefill": prefill,
        "handoff": handoff,
        "ttft_s": round(ttft, 3),
        "request_s": round(request_s, 3),
        "memory_gib": {d: round(b / 2**30, 2) for d, b in sorted(usage.items())},
        "placement": _describe(ctx, base),
        "prefill_placement": _describe(ctx, prefill_assign) if paired else None,
        "hot_experts": {d.id: {"device": h.map.get(d.id), "experts": h.placement.expert_count(d.id),
                               "gib": round(h.placement.resident_bytes(d.id) / 2**30, 2)}
                        for d in h.placement.resident_devices},
        "sets": {s: list(ctx.sets[s]) for s in sets},
        "layout": None,
        "_assign": base,
        "_prefill_assign": prefill_assign,
        "_extra_bytes": h.extra_bytes(),
        "_hot": h,
    }


def _hot_report(ctx: _Ctx, h: _Hot, c: dict) -> dict:
    """The dense+hot candidate next to the placement file's own expected numbers, per row pattern."""
    p = h.placement
    notes = []
    if not h.problems:
        dense = _hot_memory(ctx, h, h.base).get(h.host, 0) - h.extra_bytes().get(h.host, 0)
        reserve = p.device(p.dense_device).reserve_gib
        if reserve and dense > reserve * 2**30:
            notes.append(f"dense parts + state on {h.host} take {dense / 2**30:.2f} GiB, above the placement's "
                         f"reserve_gib {reserve:g}")
    _, evaluate, _ = _hot_patterns(h, ctx.req.decode_rows)
    names = {e.pattern for e in evaluate}
    patterns = {}
    for name, e in p.expected.items():
        planner = None
        if name in names and not h.problems:
            d = _hot_decode(ctx, h, e, e.rows or 1)
            planner = {"rows": d["rows"], "step_ms": d["step_ms"], "tokens_per_s": d["tokens_per_s"],
                       "all_local_step_ms": d["hot"]["all_local_step_ms"],
                       "remote_penalty_ms": d["hot"]["remote_penalty_ms"],
                       "exposed_us_per_remote_layer": d["hot"]["exposed_us_per_remote_layer"],
                       "host_copy_ms": d["hot"]["host_copy_ms"]}
            if e.tokens_per_window is not None:
                # Same acceptance as the file: comparable to its tok_s.
                planner["tok_s_at_file_tokens_per_window"] = round(e.tokens_per_window * 1e3 / d["step_ms"], 2)
        patterns[name] = {"file": e.public(), "planner": planner}
    return {
        "source": h.input.source,
        "placement": p.summary(),
        "device_map": dict(sorted(h.map.items())),
        "candidate": c["name"],
        "feasible": c["feasible"],
        "problems": c["problems"],
        "patterns": patterns,
        "assumptions": [
            "routed experts are priced by the local walk on the parts-host; a remote layer adds only the part "
            "of send + server compute + return + crossing that the local branch does not cover",
            "remote input goes once to each involved server; each cold (row, expert) vector returns raw",
            f"expert-cache misses copy at {h.input.host_copy_bytes_per_s / 1e9:g} GB/s (declared), not overlapped",
            "prefill uses the base placement (every routed expert on the expert servers) unless paired",
        ],
        "notes": notes,
    }
