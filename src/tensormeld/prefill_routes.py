"""Prefill route table (`prefill.policy = "routes"`): an ordered, versioned decision table chosen per request.

The orchestrator compiles it (schema `tensormeld/prefill-route-table-v1`) from a recipe's prefill routes; this module only
evaluates it. Per generation request, with the prompt length and a fresh availability state:

1. walk the routes in order, skipping disabled ones; a route matches when its `when` holds on the prompt tokens the
   prefill computes (the prompt minus the chosen replica's cached prefix, inclusive bounds);
2. a matched route is admissible when the request's context (whole prompt + generation budget) is within the route's
   `max_context_tokens`, its handoff route's `when` holds too, every member device of its sets is available (state
   file AND the router's own health: a `service` set needs a prefill service up, a `decode` set the chosen replica
   up), every link it needs is up, and on every host it uses `free_ram_gib - charge >= floor_gib` (and, with a
   commit floor, `free_commit_gib - commit charge >= commit_floor_gib`); a route whose process is already running is
   charged nothing but its hosts still keep the floors;
3. a matched route that is not admissible follows its `fallback` chain: a target is used only if its own `when`
   holds and it is admissible; a visited route ends the chain and the walk resumes after the original route;
4. nothing admissible: the request is refused (503) with the trace.

The state file is written by the orchestrator (schema `tensormeld/prefill-route-state-v1`, `at` = epoch seconds). Missing,
unreadable or older than `max_age_s` state means every device unavailable, every link down and every host's RAM
unknown: selection fails closed. A route whose hand-off failed for this request is inadmissible for the rest of its
walk (the engines' imports are parked by token prefix, so a late import cannot change any request's tokens).

Nothing here changes tokens: it only chooses where a prompt is prefilled.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
import threading
import time
import uuid
from typing import Any

from .schema import ValidationError, record, text

ROUTE_TABLE_SCHEMA = "tensormeld/prefill-route-table-v1"
ROUTE_STATE_SCHEMA = "tensormeld/prefill-route-state-v1"
PLACES = ("local", "offload")
ROLES = ("decode", "service", "idle")
RUNNING = ("decode", "service")
MAX_ROUTES = 64


def _id(value: Any, where: str) -> str:
    value = text(value, where)
    if len(value) > 64 or any(not (c.isalnum() or c in "-_.:") for c in value):
        raise ValidationError(f"{where}: expected an id")
    return value


def _nonneg(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValidationError(f"{where}: expected a finite number >= 0")
    return float(value)


def _tokens(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise ValidationError(f"{where}: expected an integer >= 0")
    return value


@dataclass(frozen=True)
class Route:
    id: str
    place: str
    tokens_min: int
    tokens_max: int | None
    handoff_min: int | None      # the handoff route's prompt bounds (None: no handoff route)
    handoff_max: int | None
    max_context_tokens: int
    fallback: str | None
    sets: tuple[str, ...]
    links: tuple[str, ...]
    hosts: tuple[str, ...]       # every host its sets (and its handoff route's) use: each keeps the floors
    disk_members: tuple[str, ...]  # member devices whose plan reads from disk: each needs `disk_available`
    charges: tuple[tuple[str, float], ...]
    commit_charges: tuple[tuple[str, float], ...]
    running_while: str
    enabled: bool = True

    def matches(self, tokens: int) -> bool:
        return tokens >= self.tokens_min and (self.tokens_max is None or tokens <= self.tokens_max)

    def handoff_matches(self, tokens: int) -> bool:
        return self.handoff_min is None or (
            tokens >= self.handoff_min and (self.handoff_max is None or tokens <= self.handoff_max))


@dataclass(frozen=True)
class RouteTable:
    recipe: str
    effective_sha256: str
    floor_gib: float
    binding: dict
    roles: dict
    routes: tuple[Route, ...]
    state_file: str
    max_age_s: float
    default_generation_tokens: int = 1024
    commit_floor_gib: float = 0.0

    @staticmethod
    def parse(raw: Any, where: str = "router.prefill.routes") -> "RouteTable":
        raw = record(raw, where, {"schema", "recipe", "recipe_sha256", "effective_sha256", "floor_gib", "commit_floor_gib",
                                  "binding", "roles", "routes", "state", "offload_max_wait_s",
                                  "default_generation_tokens"})
        if raw["schema"] != ROUTE_TABLE_SCHEMA:
            raise ValidationError(f"{where}.schema: expected {ROUTE_TABLE_SCHEMA}")
        binding = record(raw["binding"], f"{where}.binding", set(), set(raw["binding"]) if isinstance(raw["binding"], dict) else set())
        sets = {}
        for sid, members in binding.items():
            _id(sid, f"{where}.binding")
            if not isinstance(members, list):
                raise ValidationError(f"{where}.binding.{sid}: expected an array")
            sets[sid] = tuple((_id(m.get("id") if isinstance(m, dict) else None, f"{where}.binding.{sid}.id"),
                               _id(m.get("host"), f"{where}.binding.{sid}.host")) for m in members)
        roles = raw["roles"]
        if not isinstance(roles, dict) or set(roles) != set(sets) or any(r not in ROLES for r in roles.values()):
            raise ValidationError(f"{where}.roles: one of {', '.join(ROLES)} for every bound set")
        links_known: set[str] = set()
        routes = []
        if not isinstance(raw["routes"], list) or not 1 <= len(raw["routes"]) <= MAX_ROUTES:
            raise ValidationError(f"{where}.routes: expected 1..{MAX_ROUTES} routes")
        for i, item in enumerate(raw["routes"]):
            rw = f"{where}.routes[{i}]"
            item = record(item, rw, {"id", "place", "enabled", "when", "handoff_when", "max_context_tokens", "fallback",
                                     "sets", "links", "hosts", "disk_members", "charges", "commit_charges",
                                     "running_while"})
            if type(item["enabled"]) is not bool:
                raise ValidationError(f"{rw}.enabled: expected a boolean")

            def bounds(value, at):
                value = record(value, at, {"prompt_tokens_min", "prompt_tokens_max"})
                low = _tokens(value["prompt_tokens_min"], f"{at}.prompt_tokens_min")
                high = None if value["prompt_tokens_max"] is None else _tokens(value["prompt_tokens_max"],
                                                                             f"{at}.prompt_tokens_max")
                if high is not None and high < low:
                    raise ValidationError(f"{at}: prompt_tokens_max below prompt_tokens_min")
                return low, high
            tmin, tmax = bounds(item["when"], f"{rw}.when")
            hmin, hmax = (None, None) if item["handoff_when"] is None else bounds(item["handoff_when"], f"{rw}.handoff_when")
            max_context = item["max_context_tokens"]
            if type(max_context) is not int or not 1 <= max_context <= 1 << 40:
                raise ValidationError(f"{rw}.max_context_tokens: expected an integer 1..2^40")
            if item["place"] not in PLACES:
                raise ValidationError(f"{rw}.place: expected one of {', '.join(PLACES)}")
            if item["running_while"] not in RUNNING:
                raise ValidationError(f"{rw}.running_while: expected one of {', '.join(RUNNING)}")
            def ids(value, at):
                if not isinstance(value, list) or len(value) > 4096:
                    raise ValidationError(f"{at}: expected an array of ids")
                return [_id(v, at) for v in value]
            route_sets = ids(item["sets"], f"{rw}.sets")
            if not route_sets or any(s not in sets for s in route_sets):
                raise ValidationError(f"{rw}.sets: expected bound set ids")
            links = tuple(ids(item["links"], f"{rw}.links"))
            links_known.update(links)
            bound_hosts = {h for members in sets.values() for _, h in members}
            route_hosts = ids(item["hosts"], f"{rw}.hosts")
            if any(h not in bound_hosts for h in route_hosts):
                raise ValidationError(f"{rw}.hosts: expected bound hosts")
            bound_ids = {m for members in sets.values() for m, _ in members}
            disk_members = ids(item["disk_members"], f"{rw}.disk_members")
            if any(m not in bound_ids for m in disk_members):
                raise ValidationError(f"{rw}.disk_members: expected bound member ids")

            def per_host(value, at):
                value = record(value, at, set(), set(value) if isinstance(value, dict) else set())
                out = tuple((_id(h, at), _nonneg(g, f"{at}.{h}")) for h, g in value.items())
                if any(h not in route_hosts for h, _ in out):
                    raise ValidationError(f"{at}: a charge on a host the route does not use")
                return out
            charges = per_host(item["charges"], f"{rw}.charges")
            commit_charges = per_host(item["commit_charges"], f"{rw}.commit_charges")
            routes.append(Route(
                id=_id(item["id"], f"{rw}.id"), place=item["place"], tokens_min=tmin, tokens_max=tmax,
                handoff_min=hmin, handoff_max=hmax, max_context_tokens=max_context,
                fallback=None if item["fallback"] is None else _id(item["fallback"], f"{rw}.fallback"),
                sets=tuple(route_sets), links=links, hosts=tuple(route_hosts),
                disk_members=tuple(disk_members), charges=charges,
                commit_charges=commit_charges, running_while=item["running_while"], enabled=item["enabled"]))
        ids = [r.id for r in routes]
        if len(set(ids)) != len(ids):
            raise ValidationError(f"{where}.routes: duplicate route id")
        by_id = {r.id: r for r in routes}
        for r in routes:
            if r.fallback is not None and (r.fallback not in by_id or r.fallback == r.id):
                raise ValidationError(f"{where}.routes.{r.id}.fallback: expected another route id")
            seen, step = {r.id}, r.fallback
            while step is not None:
                if step in seen:
                    raise ValidationError(f"{where}.routes.{r.id}.fallback: cycle")
                seen.add(step)
                step = by_id[step].fallback
            if r.place == "offload" and not any(roles[s] in ("service", "idle") for s in r.sets):
                raise ValidationError(f"{where}.routes.{r.id}: an offload route needs a service set")
        last = routes[-1]
        if last.tokens_min != 0 or last.tokens_max is not None:
            raise ValidationError(f"{where}.routes: the last route must be a catch-all")
        commit_floor = _nonneg(raw["commit_floor_gib"], f"{where}.commit_floor_gib")
        state = record(raw["state"], f"{where}.state", {"file", "max_age_s"})
        max_age = _nonneg(state["max_age_s"], f"{where}.state.max_age_s")
        if not 0 < max_age <= 600:
            raise ValidationError(f"{where}.state.max_age_s: expected 0 < s <= 600")
        _nonneg(raw["offload_max_wait_s"], f"{where}.offload_max_wait_s")
        generation = raw["default_generation_tokens"]
        if type(generation) is not int or not 1 <= generation <= 1 << 24:
            raise ValidationError(f"{where}.default_generation_tokens: expected an integer 1..2^24")
        return RouteTable(recipe=text(raw["recipe"], f"{where}.recipe"),
                          effective_sha256=text(raw["effective_sha256"], f"{where}.effective_sha256"),
                          floor_gib=_nonneg(raw["floor_gib"], f"{where}.floor_gib"), binding=sets, roles=dict(roles),
                          routes=tuple(routes), state_file=text(state["file"], f"{where}.state.file"),
                          max_age_s=max_age, default_generation_tokens=generation, commit_floor_gib=commit_floor)


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name}")


def _check_state(data: Any) -> dict:
    """The state's shape, or ValueError: objects keyed by id, booleans where availability is read, finite numbers."""
    if not isinstance(data, dict) or data.get("schema") != ROUTE_STATE_SCHEMA:
        raise ValueError("unexpected route state schema")
    if type(data.get("at")) not in (int, float) or not math.isfinite(data["at"]):
        raise ValueError("route state has no time")
    for key in ("devices", "links", "hosts"):
        section = data.get(key, {})
        if not isinstance(section, dict) or any(not isinstance(v, dict) for v in section.values()):
            raise ValueError(f"route state {key}: expected an object of objects")
    for info in data.get("devices", {}).values():
        for flag in ("available", "disk_available"):
            if flag in info and type(info[flag]) is not bool:
                raise ValueError(f"route state devices: {flag} must be a boolean")
    for info in data.get("links", {}).values():
        if "up" in info and type(info["up"]) is not bool:
            raise ValueError("route state links: up must be a boolean")
    for info in data.get("hosts", {}).values():
        for key in ("free_ram_gib", "free_commit_gib"):
            value = info.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"route state hosts: {key} must be a finite number")
    return data


class StateFile:
    """The orchestrator's availability file, re-read when it changes; stale or unreadable = nothing available."""

    def __init__(self, path: str, max_age_s: float, clock=time.time):
        self.path, self.max_age_s, self.clock = path, max_age_s, clock
        self._lock = threading.Lock()
        self._stamp: tuple | None = None
        self._data: dict | None = None
        self._why = "no state read yet"

    def read(self) -> tuple[dict | None, str]:
        """(state or None, why None)."""
        with self._lock:
            try:
                st = os.stat(self.path)
                stamp = (st.st_mtime_ns, st.st_size)
                if stamp != self._stamp:
                    with open(self.path, "rb") as stream:
                        data = _check_state(json.loads(stream.read(1 << 20), parse_constant=_reject_constant))
                    self._stamp, self._data = stamp, data
            except (OSError, ValueError, RecursionError, OverflowError) as exc:
                self._stamp, self._data, self._why = None, None, f"route state unreadable ({type(exc).__name__})"
                return None, self._why
            age = self.clock() - self._data["at"]
            if age > self.max_age_s or age < -5:
                return None, f"route state stale ({age:.1f} s old, max {self.max_age_s:g} s)"
            return self._data, ""


@dataclass(frozen=True)
class Health:
    """The router's own view for one request: a prefill service up, and the replica chosen for decode up."""
    service_up: bool
    replica_up: bool


class NoRoute(Exception):
    def __init__(self, trace: list[dict]):
        super().__init__("; ".join(f"{t['route']}: {t['result']}" for t in trace) or "no route")
        self.trace = trace


def admissible(table: RouteTable, route: Route, tokens: int, context: int, state: dict | None, why_state: str,
               health: Health, failed: frozenset) -> str | None:
    """None when the route can be admitted now, else why not."""
    if not route.enabled:
        return "disabled"
    if route.id in failed:
        return "hand-off failed for this request"
    if not route.handoff_matches(tokens):
        return "handoff route: when does not hold"
    if context > route.max_context_tokens:
        return f"context {context} > max_context_tokens {route.max_context_tokens}"
    if state is None:
        return why_state
    devices, links, hosts = state.get("devices") or {}, state.get("links") or {}, state.get("hosts") or {}
    for sid in route.sets:
        role = table.roles[sid]
        if role == "idle":
            return f'set "{sid}" is not run by this deployment'
        if role == "service" and not health.service_up:
            return f'set "{sid}": no prefill service up'
        if role == "decode" and not health.replica_up:
            return f'set "{sid}": decode replica down'
        for member, _host in table.binding[sid]:
            info = devices.get(member)
            if not isinstance(info, dict) or info.get("available") is not True:
                why = info.get("why") if isinstance(info, dict) else "missing from the state"
                return f'set "{sid}" member {member} unavailable ({why})'
    for link in route.links:
        info = links.get(link)
        if not isinstance(info, dict) or info.get("up") is not True:
            why = info.get("why") if isinstance(info, dict) else "missing from the state"
            return f'link "{link}" down ({why})'
    for member in route.disk_members:
        info = devices.get(member)
        if not isinstance(info, dict) or info.get("disk_available") is not True:
            return f"member {member}: disk unavailable"
    running = (route.running_while == "decode" and health.replica_up) or (
        route.running_while == "service" and health.service_up)
    charges = {} if running else dict(route.charges)
    commits = {} if running else dict(route.commit_charges)
    for host in route.hosts:
        info = hosts.get(host)
        free = info.get("free_ram_gib") if isinstance(info, dict) else None
        if type(free) not in (int, float):
            return f"host {host}: free RAM unknown"
        charge = charges.get(host, 0.0)
        if free - charge < table.floor_gib:
            return f"host {host}: free RAM {free:.1f} GiB - charge {charge:.1f} GiB < floor {table.floor_gib:g} GiB"
        if table.commit_floor_gib > 0:
            commit = info.get("free_commit_gib")
            if type(commit) not in (int, float):
                return f"host {host}: free commit unknown"
            if commit - commits.get(host, 0.0) < table.commit_floor_gib:
                return (f"host {host}: free commit {commit:.1f} GiB - charge {commits.get(host, 0.0):.1f} GiB < commit "
                        f"floor {table.commit_floor_gib:g} GiB")
    return None


def select(table: RouteTable, tokens: int, context: int, state: dict | None, why_state: str, health: Health,
           failed: frozenset = frozenset()) -> tuple[Route, list[dict]]:
    """The admitted route and the trace; raises NoRoute with the trace when nothing can be admitted. ``tokens`` = the
    prompt tokens the prefill computes (uncached); ``context`` = the whole prompt + the request's generation budget."""
    by_id = {r.id: r for r in table.routes}
    visited: set[str] = set()
    trace: list[dict] = []
    for route in table.routes:
        if route.id in visited:
            continue
        visited.add(route.id)
        if not route.enabled:
            trace.append({"route": route.id, "result": "disabled"})
            continue
        if not route.matches(tokens):
            trace.append({"route": route.id, "result": "no match"})
            continue
        step: Route | None = route
        while step is not None:
            why = admissible(table, step, tokens, context, state, why_state, health, failed)
            if why is None:
                trace.append({"route": step.id, "result": "admitted"})
                return step, trace
            trace.append({"route": step.id, "result": why})
            target = by_id.get(step.fallback) if step.fallback else None
            step = None
            if target is not None and target.id not in visited:
                visited.add(target.id)
                if not target.enabled:
                    trace.append({"route": target.id, "result": "disabled (fallback)"})
                elif target.matches(tokens):
                    step = target
                else:
                    trace.append({"route": target.id, "result": "no match (fallback)"})
    raise NoRoute(trace)


RECENT_REQUESTS = 256


class RouteStats:
    """Requests per admitted route, refusals, failed hand-offs and the last refusal's trace (router status), plus a
    per-request record of the last RECENT_REQUESTS recipe-routed requests. `instance` changes when the router starts
    (counters and records reset with it)."""

    def __init__(self, table: RouteTable):
        self.table = table
        self._lock = threading.Lock()
        self.counts = {r.id: 0 for r in table.routes}
        self.refused = 0
        self.handoff_failed = 0
        self.not_admitted_after_wait = 0
        self.last_refusal: list[dict] | None = None
        self.instance = uuid.uuid4().hex
        self.started_at = time.time()
        self.recent: deque[dict] = deque(maxlen=RECENT_REQUESTS)

    def admitted(self, route_id: str) -> None:
        with self._lock:
            self.counts[route_id] += 1

    def refuse(self, trace: list[dict]) -> None:
        with self._lock:
            self.refused += 1
            self.last_refusal = trace

    def failed(self, sent: bool = True) -> None:
        """A hand-off that failed (sent) or an offload that never started (service wait, recheck, service gone)."""
        with self._lock:
            if sent:
                self.handoff_failed += 1
            else:
                self.not_admitted_after_wait += 1

    def begin(self, **fields) -> dict:
        """A new per-request record: {id, at, prompt_tokens, uncached_tokens, context_tokens, replica, attempts,
        route, place, outcome, status, ttfb_ms, total_ms}; `id` is returned to the client (X-TensorMeld-Route-Id)."""
        record = {"id": uuid.uuid4().hex[:16], "at": time.time(), "prompt_tokens": None, "uncached_tokens": None,
                  "context_tokens": None, "replica": None, "attempts": [], "route": None, "place": None,
                  "outcome": None, "status": None, "ttfb_ms": None, "total_ms": None, **fields}
        with self._lock:
            self.recent.append(record)
        return record

    def note(self, record: dict, attempt: dict | None = None, **changes) -> None:
        """Update a record: an attempt (replica, route, place, outcome: admitted | handoff_failed |
        not_admitted_after_wait | connect_failed | refused, why) and/or final fields."""
        with self._lock:
            if attempt is not None:
                record["attempts"].append(attempt)
            record.update(changes)

    def snapshot(self) -> dict:
        with self._lock:
            return {"recipe": self.table.recipe, "effective_sha256": self.table.effective_sha256,
                    "admitted": dict(self.counts), "refused": self.refused, "handoff_failed": self.handoff_failed,
                    "not_admitted_after_wait": self.not_admitted_after_wait,
                    "last_refusal": self.last_refusal, "instance": self.instance, "started_at": self.started_at,
                    "recent": [dict(r, attempts=[dict(a) for a in r["attempts"]]) for r in self.recent]}
