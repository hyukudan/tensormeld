"""Least-loaded HTTP router over whole-model engine replicas.

A replica is one OpenAI-compatible engine server process that holds a complete copy of
the model on one node. Replicas share no state: every request runs entirely on the
replica that receives it, so routing cannot change its tokens. The router therefore
needs no tensor transport and no execution qualification of its own; what it adds is
admission, health and placement of whole requests.

The contract is closed and bounded:
- replicas are numeric loopback/private endpoints from a validated config, never from
  a request;
- each replica has an explicit in-flight limit; when every healthy replica is full,
  requests wait in one bounded router queue with a deadline;
- the least-loaded healthy replica wins (in-flight / limit, then least recently
  assigned, then config order);
- a replica becomes routable only after its own health endpoint answers 200;
  failures to connect take it out of rotation until health recovers;
- a request is retried on another replica only when its connection failed before
  anything was sent; once sent, an upstream failure is reported, never replayed;
- responses are passed through as they arrive (server-sent event streams included);
  a client that disconnects closes its upstream connection, which lets the engine
  cancel the request;
- client credentials are never forwarded; each replica gets its own configured key;
- optional session affinity (`affinity.mode = "prefix"`): a request goes back to the replica
  that served the longest matching prefix of its conversation (or its explicit session id),
  so that replica's prefix state cache can skip that part of the prompt. Affinity only
  chooses among replicas with a free slot and yields to load: it never waits for a busy
  replica and never routes to one more than `max_extra_inflight` requests above the least
  loaded free replica. It cannot change tokens (each request still runs whole on one
  replica); only where it runs;
- optional prefill placement (`prefill`): per generation request the router decides where
  its prompt is prefilled and on which replica it decodes. "nodes" keeps the prefill on the
  decode replica. "auto" is a cost model (route_cost.py): with the prompt's token ids (from
  a replica's `/tokenize`), the cached prefix each replica holds (a mirror of the engine's
  checkpoint blocks, from the token ids actually served there), each replica's predicted
  slot wait and active requests, and each service's pending work in seconds, it takes the
  candidate with the lowest predicted cost (the request's own time plus the pause it imposes
  on the replica's active decodes); an offload must beat the best local candidate by a noise
  margin. With no service, "auto" still picks the replica by that cost (cache- and
  work-aware routing). "rtx" always offloads when a service is up. An offload reserves the
  service, waits for it (a busy service is a cost, not a refusal), sends the token ids with
  the decode replica's import route as handoff (`POST /v1/prefill {tokens, handoff,
  session}`), and only then takes the decode slot on that replica and forwards the original
  request. Any offload failure falls back to a local prefill: placement never fails a
  request.

This module launches no engine process. Process supervision and memory admission are
separate contracts. A supervisor (the pipeline deployment controller) may take a replica
out of admission (`set_admitting`), hold all new requests in the bounded queue while it
moves a deployment (`hold`), wait for a replica's in-flight requests to finish
(`wait_idle`), and register admin routes and extra status on the router.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import threading
import time
from typing import Any

from .route_cost import (
    SEED_SERVICE_PROFILES,
    CostPolicy,
    Curve,
    LocalCandidate,
    NodeProfile,
    ServiceCandidate,
    ServiceProfile,
    choose,
    local_cost,
)
from .prefill_routes import Health, NoRoute, RouteStats, RouteTable, StateFile, admissible as route_admissible
from .prefill_routes import select as select_route
from .prefill_offload import (
    OffloadStats,
    PrefillOffloadConfig,
    decide as decide_offload,
    parse_prefill_offload,
    prefill as rtx_prefill,
    replica_can_import,
    reserve_tokens,
    service_health,
)
from .schema import ValidationError, items, number, record, text, unique
from .serving_policy import JointReservation

ROUTER_CONFIG_SCHEMA = "tensormeld/replica-router-v1"
ROUTER_STATUS_SCHEMA = "tensormeld/replica-router-status-v1"
MAX_CONFIG_BYTES = 256 * 1024
MAX_REPLICAS = 64
MAX_KEY_BYTES = 4096
PIPE_CHUNK_BYTES = 64 * 1024

# Generation routes hold an in-flight slot; utility routes are forwarded without one.
GENERATION_ROUTES = frozenset({"/v1/chat/completions", "/v1/completions", "/v1/messages"})
UTILITY_POST_ROUTES = frozenset({"/tokenize", "/detokenize"})
PREFILL_ROUTES = frozenset({"/v1/chat/completions", "/v1/completions"})
FORWARDED_REQUEST_HEADERS = ("content-type", "accept", "anthropic-version", "anthropic-beta")
FORWARDED_RESPONSE_HEADERS = ("content-type", "cache-control")


def _endpoint_ip(address: Any, where: str, *, allow_private: bool) -> str:
    address = text(address, where)
    try:
        ip = ipaddress.ip_address(address)
    except ValueError as exc:
        raise ValidationError(f"{where}: must be a numeric IP literal") from exc
    if ip.is_unspecified or ip.is_multicast or ip.is_link_local:
        raise ValidationError(f"{where}: address class is not allowed")
    if not (ip.is_loopback or (allow_private and ip.is_private)):
        raise ValidationError(f"{where}: must be loopback{' or private' if allow_private else ''}")
    return str(ip)


def _port(value: Any, where: str, *, allow_zero: bool = False) -> int:
    port = int(number(value, where, 0 if allow_zero else 1, True))
    if port > 65535:
        raise ValidationError(f"{where}: exceeds 65535")
    return port


def _seconds(value: Any, where: str, low: float, high: float) -> float:
    seconds = float(number(value, where))
    if not low <= seconds <= high:
        raise ValidationError(f"{where}: expected {low}..{high} seconds")
    return seconds


def _bounded_int(value: Any, where: str, low: int, high: int) -> int:
    result = int(number(value, where, low, True))
    if result > high:
        raise ValidationError(f"{where}: expected {low}..{high}")
    return result


def _read_key(path: Any, where: str) -> str:
    path = Path(text(path, where))
    with path.open("rb") as f:
        raw = f.read(MAX_KEY_BYTES + 1)
    if len(raw) > MAX_KEY_BYTES:
        raise ValidationError(f"{where}: key file exceeds {MAX_KEY_BYTES} bytes")
    try:
        key = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise ValidationError(f"{where}: key must be ASCII") from exc
    if not key or any(c.isspace() for c in key):
        raise ValidationError(f"{where}: key must be one nonempty token")
    return key


@dataclass(frozen=True)
class ReplicaSpec:
    id: str
    node: str
    address: str
    port: int
    max_inflight: int
    api_key: str | None


AFFINITY_MODES = ("off", "prefix")
SESSION_HEADERS = ("x-session-id", "x-tensormeld-session")
COMPLETION_TEXT_BLOCK = 512    # characters per prefix block of a text prompt
COMPLETION_TOKEN_BLOCK = 256   # token ids per prefix block of a token prompt
MAX_PREFIX_KEYS = 256


@dataclass(frozen=True)
class AffinityConfig:
    mode: str = "off"
    max_entries: int = 65536
    max_extra_inflight: int = 1


PREFILL_POLICIES = ("nodes", "rtx", "auto", "routes")


@dataclass(frozen=True)
class PrefillConfig:
    policy: str = "nodes"
    services: tuple[ReplicaSpec, ...] = ()
    service_profiles: tuple[ServiceProfile | None, ...] = ()
    node_profile: NodeProfile | None = None
    decode_s_per_token: Curve | None = None   # per-request decode seconds/token as a function of active requests
    default_output_tokens: int = 256
    cache_block_tokens: int = 2048
    cache_entries: int = 65536
    cost: CostPolicy = CostPolicy()
    handoff_path: str = "/admin/import-state"
    tokenize_path: str = "/tokenize"
    prefill_path: str = "/v1/prefill"
    wait_s: float = 600.0
    routes: RouteTable | None = None   # policy "routes": the orchestrator's decision table (prefill_routes.py)


@dataclass(frozen=True)
class RouterConfig:
    listen_address: str
    listen_port: int
    api_key: str | None
    replicas: tuple[ReplicaSpec, ...]
    health_interval_s: float
    health_timeout_s: float
    failures_to_down: int
    connect_timeout_s: float
    read_timeout_s: float
    queue_timeout_s: float
    max_waiting: int
    max_connections: int
    max_request_bytes: int
    affinity: AffinityConfig = AffinityConfig()
    prefill: PrefillConfig = PrefillConfig()
    balance_active_decodes: bool = False
    prefill_offload: PrefillOffloadConfig | None = None  # opt-in GPU prefill offload; None = off

    @staticmethod
    def parse(data: Any, base_dir: str | Path | None = None) -> "RouterConfig":
        """`base_dir` resolves a relative `prefill_offload.route` receipt path (the config's directory)."""
        data = record(
            data,
            "router",
            {"schema", "listen", "replicas"},
            {"api_key_file", "health", "limits", "affinity", "prefill", "balance_active_decodes", "prefill_offload"},
        )
        if data["schema"] != ROUTER_CONFIG_SCHEMA:
            raise ValidationError(f"router.schema: expected {ROUTER_CONFIG_SCHEMA}")
        balance = data.get("balance_active_decodes", False)
        if not isinstance(balance, bool):
            raise ValidationError("router.balance_active_decodes: expected boolean")
        listen = record(data["listen"], "router.listen", {"address", "port"})
        api_key = (
            _read_key(data["api_key_file"], "router.api_key_file")
            if "api_key_file" in data
            else None
        )
        # A non-loopback front end is a LAN service: it must authenticate its clients.
        listen_address = _endpoint_ip(listen["address"], "router.listen.address", allow_private=True)
        if not ipaddress.ip_address(listen_address).is_loopback and api_key is None:
            raise ValidationError("router.listen: a private (non-loopback) listener requires api_key_file")
        replicas = []

        def endpoint_spec(raw: Any, where: str) -> ReplicaSpec:
            raw = record(raw, where, {"id", "node", "address", "port", "max_inflight"}, {"api_key_file"})
            return ReplicaSpec(
                id=text(raw["id"], f"{where}.id"),
                node=text(raw["node"], f"{where}.node"),
                address=_endpoint_ip(raw["address"], f"{where}.address", allow_private=True),
                port=_port(raw["port"], f"{where}.port"),
                max_inflight=_bounded_int(raw["max_inflight"], f"{where}.max_inflight", 1, 64),
                api_key=_read_key(raw["api_key_file"], f"{where}.api_key_file") if "api_key_file" in raw else None,
            )

        import_urls: dict[str, Any] = {}
        for i, raw in enumerate(items(data["replicas"], "router.replicas", MAX_REPLICAS)):
            if isinstance(raw, dict) and "import_url" in raw:
                if "prefill_offload" not in data:
                    raise ValidationError(f"router.replicas[{i}].import_url: needs prefill_offload")
                raw = dict(raw)
                url = raw.pop("import_url")
                spec = endpoint_spec(raw, f"router.replicas[{i}]")
                import_urls[spec.id] = url
            else:
                spec = endpoint_spec(raw, f"router.replicas[{i}]")
            replicas.append(spec)
        unique([r.id for r in replicas], "router.replicas.id")
        unique([f"{r.address}:{r.port}" for r in replicas], "router.replicas endpoint")
        health = record(
            data.get("health", {}),
            "router.health",
            set(),
            {"interval_s", "timeout_s", "failures_to_down"},
        )
        limits = record(
            data.get("limits", {}),
            "router.limits",
            set(),
            {
                "connect_timeout_s",
                "read_timeout_s",
                "queue_timeout_s",
                "max_waiting",
                "max_connections",
                "max_request_bytes",
            },
        )
        affinity = record(
            data.get("affinity", {}),
            "router.affinity",
            set(),
            {"mode", "max_entries", "max_extra_inflight"},
        )
        mode = text(affinity.get("mode", "off"), "router.affinity.mode")
        if mode not in AFFINITY_MODES:
            raise ValidationError(f"router.affinity.mode: expected one of {', '.join(AFFINITY_MODES)}")
        prefill = record(
            data.get("prefill", {}),
            "router.prefill",
            set(),
            {
                "policy", "services", "node_profile", "decode_s_per_token", "default_output_tokens", "cache",
                "margin_s", "margin_frac", "min_tokens_idle", "handoff_path", "tokenize_path", "prefill_path", "wait_s",
                "routes",
            },
        )
        policy = text(prefill.get("policy", "nodes"), "router.prefill.policy")
        if policy not in PREFILL_POLICIES:
            raise ValidationError(f"router.prefill.policy: expected one of {', '.join(PREFILL_POLICIES)}")
        services_list: list[ReplicaSpec] = []
        service_profiles: list[ServiceProfile | None] = []
        raw_services = items(prefill["services"], "router.prefill.services", MAX_REPLICAS) if "services" in prefill else []
        for i, raw in enumerate(raw_services):
            where = f"router.prefill.services[{i}]"
            if not isinstance(raw, dict):
                raise ValidationError(f"{where}: expected an object")
            body = {k: v for k, v in raw.items() if k != "profile"}
            services_list.append(endpoint_spec(body, where))
            prof = raw.get("profile")
            if prof is None:
                service_profiles.append(None)
            elif isinstance(prof, str) and prof.startswith("seed:"):
                match = [sp for sp in SEED_SERVICE_PROFILES if f"seed:{sp.model}:{sp.gpu.vram_gib:g}" == prof]
                if not match:
                    seeds = ", ".join(f"seed:{sp.model}:{sp.gpu.vram_gib:g}" for sp in SEED_SERVICE_PROFILES)
                    raise ValidationError(f"{where}.profile: unknown seed (known: {seeds})")
                service_profiles.append(match[0])
            else:
                service_profiles.append(ServiceProfile.parse(prof, f"{where}.profile"))
        services = tuple(services_list)
        unique([sv.id for sv in services] + [r.id for r in replicas], "router ids (replicas and prefill services)")
        if policy == "rtx" and not services:
            raise ValidationError("router.prefill: policy rtx needs at least one service")
        node_profile = (
            NodeProfile.parse(prefill["node_profile"], "router.prefill.node_profile") if "node_profile" in prefill else None
        )
        if policy == "auto" and (node_profile is None or any(sp is None for sp in service_profiles)):
            raise ValidationError("router.prefill: policy auto needs node_profile and a profile for every service")
        routes = None
        if policy == "routes":
            if "routes" not in prefill:
                raise ValidationError("router.prefill: policy routes needs a routes table")
            routes = RouteTable.parse(prefill["routes"])
            if any(r.place == "offload" and r.enabled for r in routes.routes) and not services:
                raise ValidationError("router.prefill.routes: an enabled offload route needs a prefill service")
            if len(services) > 1:
                raise ValidationError("router.prefill.routes: policy routes takes at most one prefill service (the one "
                                      "bound to the table's service set)")
        elif "routes" in prefill:
            raise ValidationError("router.prefill.routes: only with policy routes")
        decode_curve = (
            Curve.parse(prefill["decode_s_per_token"], "router.prefill.decode_s_per_token")
            if "decode_s_per_token" in prefill
            else Curve(((1, 1 / 46.0), (8, 1 / 6.0)))
        )
        cache = record(prefill.get("cache", {}), "router.prefill.cache", set(), {"block_tokens", "max_entries"})

        def route_path(key: str, default: str) -> str:
            value = text(prefill.get(key, default), f"router.prefill.{key}")
            if not value.startswith("/") or any(c in value for c in "?#\r\n ") or len(value) > 128:
                raise ValidationError(f"router.prefill.{key}: expected an absolute path")
            return value

        default_output_tokens = _bounded_int(
            prefill.get("default_output_tokens", 256), "router.prefill.default_output_tokens", 1, 1 << 20
        )
        prefill_offload = None
        if "prefill_offload" in data:
            if policy != "nodes":
                raise ValidationError("router.prefill_offload: needs prefill.policy nodes (one offload rule at a time)")
            prefill_offload = parse_prefill_offload(
                data["prefill_offload"], tuple(replicas), import_urls,
                base_dir=Path(base_dir) if base_dir is not None else None,
                read_key=_read_key, default_reserve=default_output_tokens,
            )
        return RouterConfig(
            prefill_offload=prefill_offload,
            balance_active_decodes=balance,
            prefill=PrefillConfig(
                policy=policy,
                services=services,
                service_profiles=tuple(service_profiles),
                node_profile=node_profile,
                decode_s_per_token=decode_curve,
                default_output_tokens=default_output_tokens,
                cache_block_tokens=_bounded_int(cache.get("block_tokens", 1024 if balance else 2048), "router.prefill.cache.block_tokens", 16, 1 << 20),
                cache_entries=_bounded_int(cache.get("max_entries", 65536), "router.prefill.cache.max_entries", 1, 1 << 22),
                cost=CostPolicy(
                    margin_s=_seconds(prefill.get("margin_s", 0.2), "router.prefill.margin_s", 0, 600),
                    margin_frac=_seconds(prefill.get("margin_frac", 0.10), "router.prefill.margin_frac", 0, 10),
                    min_tokens_idle=_bounded_int(prefill.get("min_tokens_idle", 2048), "router.prefill.min_tokens_idle", 0, 1 << 24),
                ),
                handoff_path=route_path("handoff_path", "/admin/import-state"),
                tokenize_path=route_path("tokenize_path", "/tokenize"),
                prefill_path=route_path("prefill_path", "/v1/prefill"),
                wait_s=_seconds(prefill.get("wait_s", 600.0), "router.prefill.wait_s", 0, 3600),
                routes=routes,
            ),
            affinity=AffinityConfig(
                mode=mode,
                max_entries=_bounded_int(affinity.get("max_entries", 65536), "router.affinity.max_entries", 1, 1 << 22),
                max_extra_inflight=_bounded_int(
                    affinity.get("max_extra_inflight", 1), "router.affinity.max_extra_inflight", 0, 64
                ),
            ),
            listen_address=listen_address,
            listen_port=_port(listen["port"], "router.listen.port", allow_zero=True),
            api_key=api_key,
            replicas=tuple(replicas),
            health_interval_s=_seconds(health.get("interval_s", 2.0), "router.health.interval_s", 0.05, 600),
            health_timeout_s=_seconds(health.get("timeout_s", 2.0), "router.health.timeout_s", 0.05, 60),
            failures_to_down=_bounded_int(health.get("failures_to_down", 2), "router.health.failures_to_down", 1, 100),
            connect_timeout_s=_seconds(limits.get("connect_timeout_s", 5.0), "router.limits.connect_timeout_s", 0.05, 120),
            read_timeout_s=_seconds(limits.get("read_timeout_s", 900.0), "router.limits.read_timeout_s", 1, 86400),
            queue_timeout_s=_seconds(limits.get("queue_timeout_s", 600.0), "router.limits.queue_timeout_s", 0, 86400),
            max_waiting=_bounded_int(limits.get("max_waiting", 64), "router.limits.max_waiting", 0, 4096),
            max_connections=_bounded_int(limits.get("max_connections", 128), "router.limits.max_connections", 1, 4096),
            max_request_bytes=_bounded_int(
                limits.get("max_request_bytes", 8 * 1024 * 1024),
                "router.limits.max_request_bytes",
                1,
                256 * 1024 * 1024,
            ),
        )


def load_router_config(path: str | Path) -> RouterConfig:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValidationError("router config exceeds 256 KiB limit")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        data = json.loads(raw, object_pairs_hook=no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid JSON: {exc}") from exc
    return RouterConfig.parse(data, base_dir=Path(path).parent)


def _digest(previous: bytes, value: Any) -> bytes:
    item = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(previous + len(item).to_bytes(8, "little") + item).digest()


def affinity_keys(path: str, body: bytes | None, session: str | None) -> list[bytes]:
    """Affinity keys of a generation request, most specific first.

    An explicit session id (header, OpenAI `user`, Anthropic `metadata.user_id`) comes
    first. Then the prefix chain of the conversation: for chat and messages routes, one
    digest per message over everything before it (system prompt, tools and model
    included), longest first; for text or token prompts, one digest per fixed block. A
    later turn of the same conversation repeats the earlier messages, so it shares the
    digests of the turns before it. Bodies that are not JSON objects give no prefix keys."""
    keys: list[bytes] = []
    data: Any = None
    if body:
        try:
            data = json.loads(body)
        except (UnicodeError, ValueError, RecursionError):
            data = None
    if not isinstance(data, dict):
        data = {}
    if session is None:
        user = data.get("user")
        meta = data.get("metadata")
        if isinstance(user, str) and user:
            session = user
        elif isinstance(meta, dict) and isinstance(meta.get("user_id"), str) and meta["user_id"]:
            session = meta["user_id"]
    if session:
        keys.append(_digest(b"session", session[:256]))
    chain: list[bytes] = []
    head = _digest(path.encode(), {k: data.get(k) for k in ("model", "system", "tools", "tool_choice")})
    messages = data.get("messages")
    prompt = data.get("prompt")
    if isinstance(messages, list):
        h = head
        for m in messages[:MAX_PREFIX_KEYS]:
            h = _digest(h, m)
            chain.append(h)
    elif isinstance(prompt, str):
        h = head
        for i in range(0, min(len(prompt), COMPLETION_TEXT_BLOCK * MAX_PREFIX_KEYS), COMPLETION_TEXT_BLOCK):
            h = _digest(h, prompt[i : i + COMPLETION_TEXT_BLOCK])
            chain.append(h)
    elif isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
        h = head
        for i in range(0, min(len(prompt), COMPLETION_TOKEN_BLOCK * MAX_PREFIX_KEYS), COMPLETION_TOKEN_BLOCK):
            h = _digest(h, prompt[i : i + COMPLETION_TOKEN_BLOCK])
            chain.append(h)
    keys.extend(reversed(chain))
    return keys


class AffinityTable:
    """Bounded LRU map from affinity keys to the replica that last served them."""

    def __init__(self, max_entries: int):
        self._max = max_entries
        self._map: OrderedDict[bytes, str] = OrderedDict()
        self._lock = threading.Lock()

    def lookup(self, keys: list[bytes]) -> list[str]:
        """Replica ids of the matching keys, most specific first, without repeats."""
        found: list[str] = []
        with self._lock:
            for k in keys:
                rid = self._map.get(k)
                if rid is not None:
                    self._map.move_to_end(k)
                    if rid not in found:
                        found.append(rid)
        return found

    def record(self, keys: list[bytes], replica_id: str) -> None:
        with self._lock:
            for k in keys:
                self._map[k] = replica_id
                self._map.move_to_end(k)
            while len(self._map) > self._max:
                self._map.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)


class CacheMirror:
    """The router's view of one replica's prefix state cache: the engine checkpoints a prompt
    at multiples of `block` tokens, keyed by the exact token ids before the boundary. The mirror
    records the chain digests of the boundaries of every prompt the replica served, so a new
    prompt's restorable prefix C is its longest recorded boundary. Bounded LRU: entries the
    engine may have evicted are forgotten here too, approximately."""

    def __init__(self, block: int, max_entries: int):
        self.block = block
        self._max = max_entries
        self._map: OrderedDict[bytes, int] = OrderedDict()
        self._lock = threading.Lock()

    def _chain(self, tokens: list[int]) -> list[tuple[int, bytes]]:
        out = []
        h = b"cache"
        for end in range(self.block, len(tokens) + 1, self.block):
            h = hashlib.sha256(h + json.dumps(tokens[end - self.block : end], separators=(",", ":")).encode()).digest()
            out.append((end, h))
        return out

    def lookup(self, tokens: list[int]) -> int:
        best = 0
        with self._lock:
            for end, h in self._chain(tokens):
                if h not in self._map:
                    break
                self._map.move_to_end(h)
                best = end
        return best

    def record(self, tokens: list[int]) -> None:
        with self._lock:
            for end, h in self._chain(tokens):
                self._map[h] = end
                self._map.move_to_end(h)
            while len(self._map) > self._max:
                self._map.popitem(last=False)


class NoReplicaAvailable(Exception):
    """No healthy replica, or the bounded wait for a free slot ended."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _ReplicaState:
    def __init__(self, spec: ReplicaSpec, index: int):
        self.spec = spec
        self.index = index
        self.up = False
        self.consecutive_failures = 0
        self.inflight = 0
        self.assigned = 0
        self.last_assigned = -1
        self.completed = 0
        self.failed = 0
        self.last_error = ""
        self.last_health_ms: float | None = None
        self.admitting = True

    @property
    def routable(self) -> bool:
        return self.up and self.admitting


class ReplicaPool:
    """Health and in-flight accounting of the replicas; all state behind one condition."""

    def __init__(self, config: RouterConfig):
        self.config = config
        self._cond = threading.Condition()
        self._replicas = [_ReplicaState(spec, i) for i, spec in enumerate(config.replicas)]
        self._waiting = 0
        self._sequence = 0
        self._held = False
        self.rejected = 0
        self._estimates: dict[str, dict[int, tuple[float, float]]] = {r.spec.id: {} for r in self._replicas}
        self._estimate_seq = 0
        self.affinity_hits = 0      # sent to the preferred replica
        self.affinity_yields = 0    # a preferred replica existed but was full, down or too loaded
        self.affinity_none = 0      # no preferred replica (new conversation)

    # Selection --------------------------------------------------------------------------

    def _least_loaded(self, exclude: frozenset[str]) -> _ReplicaState | None:
        if self._held:
            return None
        free = [
            r
            for r in self._replicas
            if r.routable and r.spec.id not in exclude and r.inflight < r.spec.max_inflight
        ]
        if not free:
            return None
        return min(free, key=self._load_key)

    def _load_key(self, r: _ReplicaState) -> tuple[float, int, int]:
        # Least loaded first; on ties the least recently assigned, then config order.
        load = r.inflight if self.config.balance_active_decodes else r.inflight / r.spec.max_inflight
        return (load, r.last_assigned, r.index)

    def _preferred(self, prefer: list[str], exclude: frozenset[str], least: _ReplicaState) -> _ReplicaState | None:
        """The first preferred replica that has a free slot and is at most max_extra_inflight
        requests above the least-loaded free replica."""
        limit = self.config.affinity.max_extra_inflight
        if self.config.balance_active_decodes:
            # Message/session affinity is not evidence of a restorable token block.
            return None
        for rid in prefer:
            for r in self._replicas:
                if (
                    r.spec.id == rid
                    and r.routable
                    and rid not in exclude
                    and r.inflight < r.spec.max_inflight
                    and r.inflight - least.inflight <= limit
                ):
                    return r
        return None

    def _select(self, exclude: frozenset[str], prefer: list[str] | None) -> _ReplicaState | None:
        least = self._least_loaded(exclude)
        if least is None or prefer is None:
            return least
        if not prefer:
            self.affinity_none += 1
            return least
        chosen = self._preferred(prefer, exclude, least)
        if chosen is None:
            self.affinity_yields += 1
            return least
        self.affinity_hits += 1
        return chosen

    def _required(self, rid: str, exclude: frozenset[str]) -> _ReplicaState | None:
        if self._held or rid in exclude:
            return None
        r = self._by_id(rid)
        return r if r.routable and r.inflight < r.spec.max_inflight else None

    def acquire(self, exclude: frozenset[str] = frozenset(), prefer: list[str] | None = None,
                require: str | None = None, scores: dict[str, float] | None = None) -> ReplicaSpec:
        """Take one in-flight slot on a healthy replica, waiting if all are full: the first
        acceptable preferred replica (affinity), else the least loaded. `prefer=None` means
        no affinity; an empty list counts as a request without a preferred replica.
        `require` waits for a slot on that replica only (a cost decision or an imported
        state already chose it); it fails when that replica is down."""
        deadline = time.monotonic() + self.config.queue_timeout_s

        def pick() -> _ReplicaState | None:
            if scores is not None and self.config.balance_active_decodes and require is None:
                least = self._least_loaded(exclude)
                if least is None:
                    return None
                free = [r for r in self._replicas if r.routable and r.spec.id not in exclude
                        and r.inflight < r.spec.max_inflight and r.inflight == least.inflight]
                return min(free, key=lambda r: (scores.get(r.spec.id, float("inf")), self._load_key(r)))
            return self._required(require, exclude) if require is not None else self._select(exclude, prefer)

        def routable_left() -> bool:
            if require is not None:
                return require not in exclude and self._by_id(require).routable
            return self._any_routable(exclude)

        with self._cond:
            chosen = pick()
            if chosen is None:
                if not self._held and not routable_left():
                    self.rejected += 1
                    raise NoReplicaAvailable("no healthy replica")
                if self._waiting >= self.config.max_waiting:
                    self.rejected += 1
                    raise NoReplicaAvailable("router queue is full")
                self._waiting += 1
                try:
                    while chosen is None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self.rejected += 1
                            raise NoReplicaAvailable("timed out waiting for a free replica")
                        if not self._held and not routable_left():
                            self.rejected += 1
                            raise NoReplicaAvailable("no healthy replica")
                        self._cond.wait(remaining)
                        chosen = pick()
                finally:
                    self._waiting -= 1
            chosen.inflight += 1
            chosen.assigned += 1
            chosen.last_assigned = self._sequence
            self._sequence += 1
            return chosen.spec

    # Work estimates (cost policy) ---------------------------------------------------------

    def add_estimate(self, replica_id: str, seconds: float) -> int:
        """Record the predicted duration of a request admitted on the replica."""
        with self._cond:
            self._estimate_seq += 1
            self._estimates[replica_id][self._estimate_seq] = (time.monotonic(), max(0.0, seconds))
            return self._estimate_seq

    def drop_estimate(self, replica_id: str, token: int) -> None:
        with self._cond:
            self._estimates[replica_id].pop(token, None)

    def load_view(self, exclude: frozenset[str] = frozenset()) -> list[tuple[str, float, int]]:
        """(replica id, predicted wait for a free slot in seconds, active requests) of every
        routable replica: the wait is 0 with a free slot, else the earliest predicted finish."""
        now = time.monotonic()
        out = []
        with self._cond:
            if self._held:
                return out
            for r in self._replicas:
                if not r.routable or r.spec.id in exclude:
                    continue
                if r.inflight < r.spec.max_inflight:
                    wait = 0.0
                else:
                    ends = [max(0.0, t0 + est - now) for t0, est in self._estimates[r.spec.id].values()]
                    wait = min(ends) if ends else 1.0
                out.append((r.spec.id, wait, r.inflight))
        return out

    def _any_routable(self, exclude: frozenset[str]) -> bool:
        return any(r.routable and r.spec.id not in exclude for r in self._replicas)

    def pick_any(self) -> ReplicaSpec:
        """Least-loaded healthy replica for a utility call that holds no slot."""
        with self._cond:
            up = [] if self._held else [r for r in self._replicas if r.routable]
            if not up:
                raise NoReplicaAvailable("no healthy replica")
            return min(up, key=self._load_key).spec

    def release(self, replica_id: str, *, ok: bool) -> None:
        with self._cond:
            r = self._by_id(replica_id)
            r.inflight -= 1
            if ok:
                r.completed += 1
            else:
                r.failed += 1
            self._cond.notify_all()

    # Supervisor controls ------------------------------------------------------------------

    def set_admitting(self, replica_id: str, admitting: bool) -> None:
        """Administrative drain: a non-admitting replica gets no new request (health is kept)."""
        with self._cond:
            self._by_id(replica_id).admitting = bool(admitting)
            self._cond.notify_all()

    def hold(self, held: bool) -> None:
        """While held, new requests wait in the bounded queue (up to its deadline) instead of
        being routed or rejected for lack of a healthy replica."""
        with self._cond:
            self._held = bool(held)
            self._cond.notify_all()

    def inflight(self, replica_id: str) -> int:
        with self._cond:
            return self._by_id(replica_id).inflight

    def wait_idle(self, replica_id: str, timeout_s: float) -> bool:
        """Wait until the replica has no in-flight request; False on timeout."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            r = self._by_id(replica_id)
            while r.inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    # Health -----------------------------------------------------------------------------

    def _by_id(self, replica_id: str) -> _ReplicaState:
        for r in self._replicas:
            if r.spec.id == replica_id:
                return r
        raise KeyError(replica_id)

    def report_health(self, replica_id: str, ok: bool, *, elapsed_ms: float, error: str = "") -> None:
        with self._cond:
            r = self._by_id(replica_id)
            r.last_health_ms = round(elapsed_ms, 3)
            if ok:
                r.consecutive_failures = 0
                r.last_error = ""
                if not r.up:
                    r.up = True
                    self._cond.notify_all()
                return
            r.consecutive_failures += 1
            r.last_error = error[:200]
            if r.consecutive_failures >= self.config.failures_to_down and r.up:
                r.up = False
                self._cond.notify_all()

    def mark_down(self, replica_id: str, error: str) -> None:
        """Passive failure: the replica refused or dropped a connection."""
        with self._cond:
            r = self._by_id(replica_id)
            r.up = False
            r.consecutive_failures = max(r.consecutive_failures, self.config.failures_to_down)
            r.last_error = error[:200]
            self._cond.notify_all()

    def specs(self) -> tuple[ReplicaSpec, ...]:
        return self.config.replicas

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {
                "schema": ROUTER_STATUS_SCHEMA,
                "status": "ok" if any(r.routable for r in self._replicas) else "unavailable",
                "held": self._held,
                "waiting": self._waiting,
                "rejected": self.rejected,
                "affinity": {
                    "mode": self.config.affinity.mode,
                    "hits": self.affinity_hits,
                    "yields": self.affinity_yields,
                    "none": self.affinity_none,
                },
                "replicas": [
                    {
                        "id": r.spec.id,
                        "node": r.spec.node,
                        "endpoint": f"{r.spec.address}:{r.spec.port}",
                        "state": "up" if r.up else "down",
                        "admitting": r.admitting,
                        "inflight": r.inflight,
                        "max_inflight": r.spec.max_inflight,
                        "assigned": r.assigned,
                        "completed": r.completed,
                        "failed": r.failed,
                        "last_health_ms": r.last_health_ms,
                        "last_error": r.last_error,
                    }
                    for r in self._replicas
                ],
            }


class ServicePool:
    """Health, reservations and slots of the prefill services. A service is serial up to its
    max_inflight: requests reserve it with a predicted hold time, wait in order for a slot and
    hold it for prefill + handoff. Its pending work in seconds (reserved, queued and running)
    is what the cost model sees: a busy service is a cost, not a refusal."""

    def __init__(self, specs: tuple[ReplicaSpec, ...], failures_to_down: int, wait_s: float):
        self._specs = specs
        self._failures_to_down = failures_to_down
        self._wait_s = wait_s
        self._cond = threading.Condition()
        self._state = {sv.id: _ReplicaState(sv, i) for i, sv in enumerate(specs)}
        # per service: token -> [predicted hold s, start time or None]; FIFO by token
        self._reservations: dict[str, OrderedDict[int, list]] = {sv.id: OrderedDict() for sv in specs}
        self._seq = 0
        self.counts = {"local": 0, "offload": 0, "fallback": 0, "cache": 0}
        self.offload_ms = 0.0

    def specs(self) -> tuple[ReplicaSpec, ...]:
        return self._specs

    def up_ids(self) -> list[str]:
        with self._cond:
            return [sid for sid, r in self._state.items() if r.up]

    def pending_s(self, service_id: str) -> float:
        now = time.monotonic()
        with self._cond:
            r = self._state[service_id]
            total = 0.0
            for hold, start in self._reservations[service_id].values():
                total += hold if start is None else max(0.0, start + hold - now)
            return total / r.spec.max_inflight

    def reserve(self, service_id: str, hold_s: float) -> int:
        with self._cond:
            self._seq += 1
            self._reservations[service_id][self._seq] = [max(0.0, hold_s), None]
            return self._seq

    def start(self, service_id: str, token: int) -> bool:
        """Wait (FIFO, at most wait_s) for a slot; False if the service went down or the wait ended."""
        deadline = time.monotonic() + self._wait_s
        with self._cond:
            r = self._state[service_id]
            queue = self._reservations[service_id]
            while True:
                if not r.up:
                    return False
                waiting = [t for t, v in queue.items() if v[1] is None]
                if r.inflight < r.spec.max_inflight and waiting and waiting[0] == token:
                    r.inflight += 1
                    r.assigned += 1
                    queue[token][1] = time.monotonic()
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)

    def finish(self, service_id: str, token: int, *, started: bool, ok: bool) -> None:
        with self._cond:
            r = self._state[service_id]
            self._reservations[service_id].pop(token, None)
            if started:
                r.inflight -= 1
                if ok:
                    r.completed += 1
                else:
                    r.failed += 1
            self._cond.notify_all()

    def count(self, place: str, offload_ms: float = 0.0) -> None:
        with self._cond:
            self.counts[place] += 1
            self.offload_ms += offload_ms

    def report_health(self, service_id: str, ok: bool, *, elapsed_ms: float, error: str = "") -> None:
        with self._cond:
            r = self._state[service_id]
            r.last_health_ms = round(elapsed_ms, 3)
            if ok:
                r.consecutive_failures = 0
                r.last_error = ""
                r.up = True
            else:
                r.consecutive_failures += 1
                r.last_error = error[:200]
                if r.consecutive_failures >= self._failures_to_down:
                    r.up = False
            self._cond.notify_all()

    def mark_down(self, service_id: str, error: str) -> None:
        with self._cond:
            r = self._state[service_id]
            r.up = False
            r.consecutive_failures = max(r.consecutive_failures, self._failures_to_down)
            r.last_error = error[:200]
            self._cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {
                "placements": dict(self.counts),
                "offload_ms_total": round(self.offload_ms, 1),
                "pending_s": {sid: round(sum(v[0] for v in q.values()), 3) for sid, q in self._reservations.items()},
                "services": [
                    {
                        "id": r.spec.id,
                        "node": r.spec.node,
                        "endpoint": f"{r.spec.address}:{r.spec.port}",
                        "state": "up" if r.up else "down",
                        "inflight": r.inflight,
                        "max_inflight": r.spec.max_inflight,
                        "assigned": r.assigned,
                        "completed": r.completed,
                        "failed": r.failed,
                        "last_error": r.last_error,
                    }
                    for r in self._state.values()
                ],
            }


def probe_replica_health(spec: ReplicaSpec, timeout_s: float) -> tuple[bool, str]:
    """GET /health on one replica: healthy only on HTTP 200."""
    conn = http.client.HTTPConnection(spec.address, spec.port, timeout=timeout_s)
    try:
        conn.request("GET", "/health", headers={"Connection": "close"})
        resp = conn.getresponse()
        resp.read(4096)
        if resp.status == 200:
            return True, ""
        return False, f"health HTTP {resp.status}"
    except (OSError, http.client.HTTPException) as exc:
        return False, f"health {type(exc).__name__}: {exc}"
    finally:
        conn.close()


class HealthMonitor:
    def __init__(self, pool: ReplicaPool, services: "ServicePool | None" = None):
        self.pool = pool
        self.services = services
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="tensormeld-router-health", daemon=True)

    def check_once(self) -> None:
        targets: list[tuple[Any, ReplicaSpec]] = [(self.pool, spec) for spec in self.pool.specs()]
        if self.services is not None:
            targets += [(self.services, spec) for spec in self.services.specs()]
        for owner, spec in targets:
            t0 = time.monotonic()
            ok, error = probe_replica_health(spec, self.pool.config.health_timeout_s)
            owner.report_health(spec.id, ok, elapsed_ms=(time.monotonic() - t0) * 1000, error=error)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check_once()
            self._stop.wait(self.pool.config.health_interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        count = len(self.pool.specs()) + (len(self.services.specs()) if self.services is not None else 0)
        self._thread.join(timeout=self.pool.config.health_timeout_s * count + 5)


def _ms(since: float) -> float:
    return round((time.monotonic() - since) * 1000, 1)


class _ConnectFailed(Exception):
    """The upstream connection failed before any request byte was sent."""


class _ClientGone(Exception):
    pass


def _metrics_text(snapshot: dict[str, Any]) -> str:
    lines = [
        "# TYPE tensormeld_router_waiting gauge",
        f"tensormeld_router_waiting {snapshot['waiting']}",
        "# TYPE tensormeld_router_rejected_total counter",
        f"tensormeld_router_rejected_total {snapshot['rejected']}",
    ]
    for name in ("hits", "yields", "none"):
        lines.append(f"# TYPE tensormeld_router_affinity_{name}_total counter")
        lines.append(f"tensormeld_router_affinity_{name}_total {snapshot['affinity'][name]}")
    series = (
        ("up", "gauge", lambda r: 1 if r["state"] == "up" else 0),
        ("inflight", "gauge", lambda r: r["inflight"]),
        ("assigned_total", "counter", lambda r: r["assigned"]),
        ("completed_total", "counter", lambda r: r["completed"]),
        ("failed_total", "counter", lambda r: r["failed"]),
    )
    for name, kind, value in series:
        lines.append(f"# TYPE tensormeld_router_replica_{name} {kind}")
        for r in snapshot["replicas"]:
            lines.append(
                f'tensormeld_router_replica_{name}{{replica="{r["id"]}",node="{r["node"]}"}} {value(r)}'
            )
    return "\n".join(lines) + "\n"


def _offload_metrics_text(stats: OffloadStats) -> str:
    counts, reasons, secs = stats.totals()
    lines = ["# TYPE tensormeld_router_prefill_offload_total counter"]
    lines += [f'tensormeld_router_prefill_offload_total{{route="{k}"}} {v}' for k, v in counts.items()]
    lines.append("# TYPE tensormeld_router_prefill_offload_reason_total counter")
    lines += [f'tensormeld_router_prefill_offload_reason_total{{reason="{k}"}} {v}' for k, v in sorted(reasons.items())]
    lines.append("# TYPE tensormeld_router_prefill_offload_rtx_seconds_total counter")
    lines.append(f"tensormeld_router_prefill_offload_rtx_seconds_total {secs:.3f}")
    return "\n".join(lines) + "\n"


class _RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "tensormeld-router"
    server: "ReplicaRouterServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass

    # Plain responses --------------------------------------------------------------------

    def _send_json(self, status: int, value: Any) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        record = getattr(self, "_route_record", None)
        if record is not None:
            self.send_header("X-TensorMeld-Route-Id", record["id"])
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str, kind: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": kind}})

    def _authorized(self) -> bool:
        key = self.server.config.api_key
        if key is None:
            return True
        bearer = self.headers.get("Authorization", "")
        return bearer == f"Bearer {key}" or self.headers.get("x-api-key", "") == key

    # Routes -----------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        path = self.path.split("?", 1)[0]
        snapshot = self.server.pool.snapshot()
        if path in ("/health", "/v1/health"):
            up = sum(r["state"] == "up" and r["admitting"] for r in snapshot["replicas"])
            self._send_json(
                200 if up else 503,
                {"status": "ok" if up else "unavailable", "replicas_up": up, "replicas": len(snapshot["replicas"])},
            )
        elif path == "/api/status":
            if self.server.services is not None:
                snapshot = {**snapshot, "prefill": {"policy": self.server.config.prefill.policy,
                                                    **self.server.services.snapshot()}}
            if self.server.route_stats is not None:
                snapshot = {**snapshot, "prefill_routes": self.server.route_stats.snapshot()}
            if self.server.offload_stats is not None:
                snapshot = {**snapshot, "prefill_offload": self.server.offload_stats.snapshot(
                    self.server.config.prefill_offload)}
            extra = self.server.status_extra
            self._send_json(200, {**snapshot, **extra()} if extra is not None else snapshot)
        elif path == "/metrics":
            body = _metrics_text(snapshot)
            if self.server.offload_stats is not None:
                body += _offload_metrics_text(self.server.offload_stats)
            body = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/v1/models":
            if not self._authorized():
                self._send_error_json(401, "missing or invalid API key", "authentication_error")
                return
            self._forward(path, None, hold_slot=False)
        else:
            self._send_error_json(404, "unknown route", "not_found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        path = self.path.split("?", 1)[0]
        if path in self.server.admin_routes:
            self._admin(path)
            return
        if path not in GENERATION_ROUTES and path not in UTILITY_POST_ROUTES:
            self._send_error_json(404, "unknown route", "not_found")
            return
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self.close_connection = True
            self._send_error_json(411, "request body needs Content-Length", "invalid_request_error")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0:
            self.close_connection = True
            self._send_error_json(411, "request body needs Content-Length", "invalid_request_error")
            return
        if length > self.server.config.max_request_bytes:
            self.close_connection = True
            self._send_error_json(413, "request body too large", "invalid_request_error")
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self.close_connection = True
            return
        if not self._authorized():
            self._send_error_json(401, "missing or invalid API key", "authentication_error")
            return
        self._forward(path, body, hold_slot=path in GENERATION_ROUTES)

    def _admin(self, path: str) -> None:
        """Supervisor route: bodyless POST, authenticated like generation routes. A router
        without an API key only listens on loopback (enforced by the config)."""
        self.close_connection = True
        if self.headers.get("Content-Length", "0") not in ("", "0"):
            self._send_error_json(400, "admin routes take no body", "invalid_request_error")
            return
        if not self._authorized():
            self._send_error_json(401, "missing or invalid API key", "authentication_error")
            return
        try:
            status, value = self.server.admin_routes[path]()
        except Exception as exc:  # the supervisor reports its own failure
            status, value = 500, {"error": {"message": str(exc)[:400], "type": "api_error"}}
        self._send_json(status, value)

    # Forwarding -------------------------------------------------------------------------

    def _forward(self, path: str, body: bytes | None, *, hold_slot: bool) -> None:
        pool = self.server.pool
        self._prefill_note: str | None = None  # the handler serves every request of a keep-alive connection
        self._cost_note: str | None = None
        tried: set[str] = set()
        keys: list[bytes] = []
        prefer: list[str] | None = None
        affinity = self.server.affinity
        if hold_slot and affinity is not None:
            session = next((self.headers.get(h) for h in SESSION_HEADERS if self.headers.get(h)), None)
            keys = affinity_keys(path, body, session)
            prefer = affinity.lookup(keys)
        if hold_slot and path in PREFILL_ROUTES and self.server.config.prefill.policy == "routes":
            self._forward_routes(path, body, keys)
            return
        if (hold_slot and path in PREFILL_ROUTES and self.server.config.prefill.policy in ("auto", "rtx")
                and self._forward_placed(path, body)):
            return
        for _ in range(len(pool.specs())):
            try:
                spec = pool.acquire(frozenset(tried), prefer) if hold_slot else pool.pick_any()
            except NoReplicaAvailable as exc:
                self._send_error_json(503, exc.reason, "overloaded_error")
                return
            ok = False
            if keys and affinity is not None:
                affinity.record(keys, spec.id)
            try:
                if hold_slot and self.server.services is not None and path in PREFILL_ROUTES:
                    self._prefill_note = self._prefill_note or "local"
                if hold_slot and path in PREFILL_ROUTES and self.server.config.prefill_offload is not None:
                    self._prefill_offload(spec, path, body)
                ok = self._exchange(spec, path, body)
                return
            except _ConnectFailed as exc:
                pool.mark_down(spec.id, str(exc))
                tried.add(spec.id)
            finally:
                if hold_slot:
                    pool.release(spec.id, ok=ok)
        self._send_error_json(503, "no replica accepted the connection", "overloaded_error")

    def _prefill_offload(self, spec: ReplicaSpec, path: str, body: bytes | None) -> None:
        """`prefill_offload` rule on the replica already held for this request: maybe have the
        GPU prefill service prefill the prompt and hand its state to that replica. Never fails
        the request: every error leaves the replica to prefill itself."""
        server = self.server
        config = server.config.prefill_offload
        assert config is not None and server.offload_stats is not None
        t0 = time.monotonic()
        rec: dict[str, Any] = {"replica": spec.id, "path": path}
        try:
            data = json.loads(body or b"")
        except (UnicodeError, ValueError, RecursionError):
            data = None
        tokens = None
        if isinstance(data, dict):
            prompt = data.get("prompt") if path == "/v1/completions" else None
            if isinstance(prompt, list) and prompt and all(type(t) is int and t >= 0 for t in prompt):
                tokens = prompt
            elif path == "/v1/completions":
                if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
                    prompt = prompt[0]
                if isinstance(prompt, str) and prompt:
                    tokens = self._tokenize(spec, path, {"prompt": prompt})
            else:
                tokens = self._tokenize(spec, path, data)
        if tokens is None:
            rec.update(route="local", reason="no_tokens", tokens=None)
        else:
            rec["tokens"] = len(tokens)
            # Threshold first (no network); ask the service only when the prompt is long enough.
            decision = decide_offload(config.route, len(tokens), config.min_tokens, {"status": "ok"})
            if decision["route"] == "offload":
                decision = decide_offload(config.route, len(tokens), config.min_tokens, service_health(config))
            rec.update({k: v for k, v in decision.items() if k != "tokens"})
            if decision["route"] == "offload":
                if not replica_can_import(config, spec.id, spec.api_key):
                    rec.update(route="local", reason="replica_no_import")
                else:
                    result = rtx_prefill(config, spec.id, tokens, reserve_tokens(data, config.default_reserve_tokens))
                    rec.update(handoff=result["handoff"], session=result["session"],
                               rtx_wall_s=result.get("wall_s"), rtx_seconds=result.get("seconds"))
                    if result.get("local_rows"):
                        rec["local_rows"] = result["local_rows"]
                    if not result["ok"]:
                        rec["fallback"] = result["reason"]
        rec["router_s"] = round(time.monotonic() - t0, 3)
        self._prefill_note = (f"fallback:{rec['fallback']}" if rec.get("fallback")
                              else "offload:rtx" if rec["route"] == "offload" else f"local:{rec['reason']}")
        server.offload_stats.record(rec)
        if server.offload_log is not None:
            try:
                server.offload_log(rec)
            except Exception:  # a log sink never fails a request
                pass

    def _forward_placed(self, path: str, body: bytes | None) -> bool:
        """Cost-placed generation request (policy auto or rtx). Returns False when it cannot
        place (no token ids): the caller then routes it the ordinary way."""
        server = self.server
        pool = server.pool
        config = server.config.prefill
        services = server.services
        try:
            data = json.loads(body or b"")
        except (UnicodeError, ValueError, RecursionError):
            return False
        if not isinstance(data, dict):
            return False
        tokens = data.get("prompt") if path == "/v1/completions" else None
        if not (isinstance(tokens, list) and tokens and all(isinstance(t, int) and t >= 0 for t in tokens)):
            try:
                tokens = self._tokenize(pool.pick_any(), path, data)
            except NoReplicaAvailable:
                return False
            if tokens is None:
                return False
        out_tokens = data.get("max_tokens") or data.get("max_completion_tokens") or config.default_output_tokens
        out_tokens = min(int(out_tokens), config.default_output_tokens) if isinstance(out_tokens, int) else config.default_output_tokens
        if server.config.balance_active_decodes:
            return self._forward_balanced(path, body, tokens, out_tokens)
        tried: set[str] = set()
        for _ in range(len(pool.specs())):
            view = pool.load_view(frozenset(tried))
            if not view:
                self._send_error_json(503, "no healthy replica", "overloaded_error")
                return True
            locals_ = [
                LocalCandidate(rid, wait, active, server.mirrors[rid].lookup(tokens), config.node_profile)
                for rid, wait, active in view
            ] if config.node_profile is not None else []
            service_cands = []
            if services is not None:
                for sid in services.up_ids():
                    prof = server.service_profile(sid)
                    if prof is not None:
                        service_cands.append(ServiceCandidate(sid, services.pending_s(sid), prof))
            if config.policy == "rtx":
                choice_place = "offload" if services is not None and services.up_ids() else "local"
                rid = min(view, key=lambda v: (v[1], v[2]))[0]
                sid = min(services.up_ids(), key=services.pending_s) if choice_place == "offload" else None
                cost_note = ""
            else:
                choice = choose(len(tokens), locals_, service_cands, config.cost)
                choice_place, rid, sid = choice.place, choice.replica, choice.service
                cost_note = f"local={choice.best_local_s:.3f}" + (
                    f";offload={choice.best_offload_s:.3f}" if choice.best_offload_s is not None else "")
            self._cost_note = cost_note
            cached = server.mirrors[rid].lookup(tokens)
            note = "cache" if cached > 0 else "local"
            if choice_place == "offload" and sid is not None:
                if self._offload_reserved(sid, server.replica_spec(rid), tokens):
                    note = f"offload:{sid}"
                else:
                    note = f"fallback:{self._offload_why}"
            if note.startswith("offload") and services is not None:
                services.count("offload")
            elif services is not None:
                services.count("fallback" if note.startswith("fallback") else ("cache" if note == "cache" else "local"))
            self._prefill_note = note
            # Take the decode slot on the chosen replica only now (an offload did not hold it).
            try:
                spec = pool.acquire(frozenset(tried), require=rid)
            except NoReplicaAvailable:
                tried.add(rid)
                continue
            local_rows = 0 if note.startswith("offload") else len(tokens) - min(cached, len(tokens))
            per_token = config.decode_s_per_token(max(1, pool.inflight(rid))) if config.decode_s_per_token else 0.0
            est = (config.node_profile.prefill(local_rows) if config.node_profile else 0.0) + out_tokens * per_token
            token = pool.add_estimate(rid, est)
            ok = False
            try:
                ok = self._exchange(spec, path, body)
                if ok and not note.startswith("offload"):
                    # Only prompts the replica prefilled itself are known to be in its checkpoint cache.
                    server.mirrors[rid].record(tokens)
                return True
            except _ConnectFailed as exc:
                pool.mark_down(spec.id, str(exc))
                tried.add(spec.id)
            finally:
                pool.drop_estimate(rid, token)
                pool.release(spec.id, ok=ok)
        self._send_error_json(503, "no replica accepted the connection", "overloaded_error")
        return True

    def _route_tokens(self, path: str, body: bytes | None) -> tuple[list[int] | None, str, int]:
        """The prompt's token ids for route selection (prompt length and context bound): one prompt per request,
        never guessed. On failure: (None, why, HTTP status): 400 for the request, 503 when no replica tokenized it."""
        try:
            data = json.loads(body or b"")
        except (UnicodeError, ValueError, RecursionError):
            return None, "the request body is not JSON", 400
        if not isinstance(data, dict):
            return None, "the request body is not a JSON object", 400
        prompt = data.get("prompt") if path == "/v1/completions" else None
        if isinstance(prompt, list) and prompt and all(type(t) is int and t >= 0 for t in prompt):
            return prompt, "", 0
        if path == "/v1/completions" and not isinstance(prompt, str):
            return None, "one prompt per request (a string or one list of token ids)", 400
        try:
            tokens = self._tokenize(self.server.pool.pick_any(), path, data)
        except NoReplicaAvailable as exc:
            return None, f"no replica to tokenize ({exc.reason})", 503
        return (tokens, "", 0) if tokens else (None, "tokenization failed", 503)

    def _forward_routes(self, path: str, body: bytes | None, keys: list[bytes] | None = None) -> None:
        try:
            self._forward_routes_recorded(path, body, keys)
        finally:
            self._route_record = None

    def _forward_routes_recorded(self, path: str, body: bytes | None, keys: list[bytes] | None = None) -> None:
        """Policy routes: the orchestrator's decision table picks local prefill or an offload per request
        (prefill_routes.py). The decode slot is taken first; the route is chosen after that wait with a fresh state and
        checked again after the prefill service's wait, right before the hand-off. A failed hand-off resumes the walk at
        the route's fallback; nothing admissible is a 503 with the trace. Prefill placement never changes tokens."""
        server = self.server
        pool, services, table, stats = server.pool, server.services, server.config.prefill.routes, server.route_stats
        tokens, why, status = self._route_tokens(path, body)
        started = time.monotonic()
        record = stats.begin(prompt_tokens=None if tokens is None else len(tokens))
        self._route_record = record
        if tokens is None:
            stats.refuse([{"route": "-", "result": why}])
            stats.note(record, outcome="refused", status=status, why=why, total_ms=_ms(started))
            self._send_error_json(status, f"recipe routing needs the prompt length: {why}",
                                  "invalid_request_error" if status == 400 else "overloaded_error")
            return
        data = json.loads(body)
        budget = data.get("max_tokens") or data.get("max_completion_tokens")
        context = len(tokens) + (budget if type(budget) is int and budget > 0 else table.default_generation_tokens)
        stats.note(record, context_tokens=context)
        tried: set[str] = set()
        for _ in range(len(pool.specs())):
            view = pool.load_view(frozenset(tried))
            if not view:
                stats.note(record, outcome="no_replica", status=503, total_ms=_ms(started))
                self._send_error_json(503, "no healthy replica", "overloaded_error")
                return
            rid = min(view, key=lambda v: (v[1], v[2]))[0]
            try:
                spec = pool.acquire(frozenset(tried), require=rid)      # the decode slot first (it may wait)
            except NoReplicaAvailable:
                tried.add(rid)
                continue
            ok = False
            if keys and server.affinity is not None:
                server.affinity.record(keys, spec.id)
            try:
                # Thresholds are on the tokens the prefill computes: the prompt minus this replica's cached prefix.
                uncached = len(tokens) - min(server.mirrors[rid].lookup(tokens), len(tokens))
                stats.note(record, uncached_tokens=uncached, replica=rid)
                failed: set[str] = set()
                while True:
                    state, why_state = server.route_state.read()
                    health = Health(service_up=bool(services is not None and services.up_ids()), replica_up=True)
                    try:
                        route, _ = select_route(table, uncached, context, state, why_state, health, frozenset(failed))
                    except NoRoute as exc:
                        stats.refuse(exc.trace)
                        stats.note(record, {"replica": rid, "route": None, "place": None, "outcome": "refused",
                                            "why": str(exc)}, outcome="refused", status=503, total_ms=_ms(started))
                        self._send_error_json(503, f"no admissible prefill route: {exc}", "overloaded_error")
                        return
                    if route.place == "local":
                        note = f"route:{route.id};local"
                        break
                    up = services.up_ids()
                    if not up:
                        # The service went down after selection: the walk resumes at the route's fallback.
                        stats.note(record, {"replica": rid, "route": route.id, "place": "offload",
                                            "outcome": "not_admitted_after_wait", "why": "service gone after selection"})
                        stats.failed(sent=False)
                        services.count("fallback")
                        failed.add(route.id)
                        continue
                    sid = min(up, key=services.pending_s)

                    def recheck(route=route):
                        fresh, why_fresh = server.route_state.read()
                        return route_admissible(table, route, uncached, context, fresh, why_fresh,
                                                Health(service_up=True, replica_up=True), frozenset(failed))
                    if self._offload_checked(sid, spec, tokens, recheck):
                        note = f"route:{route.id};offload:{sid}"
                        services.count("offload")
                        break
                    # The engines park imports by token prefix: a late import cannot change any request's tokens.
                    why_offload = getattr(self, "_offload_why", "") or ""
                    sent = not why_offload.startswith(("not admitted", "wait", "accelerator"))
                    stats.note(record, {"replica": rid, "route": route.id, "place": "offload",
                                        "outcome": "handoff_failed" if sent else "not_admitted_after_wait",
                                        "why": why_offload})
                    stats.failed(sent=sent)
                    services.count("fallback")
                    failed.add(route.id)
                stats.admitted(route.id)
                stats.note(record, {"replica": rid, "route": route.id, "place": route.place, "outcome": "admitted",
                                    "why": None}, route=route.id, place=route.place)
                self._prefill_note = note
                self._relay_status = self._relay_outcome = self._relay_first_byte = None
                ok = self._exchange(spec, path, body)
                status = self._relay_status
                outcome = ("served" if ok else "client_gone" if self._relay_outcome == "client_gone"
                           else "upstream_error")
                stats.note(record, outcome=outcome, status=status, total_ms=_ms(started),
                           ttfb_ms=None if self._relay_first_byte is None else round(
                               (self._relay_first_byte - started) * 1000, 1))
                if ok and route.place == "local":
                    server.mirrors[rid].record(tokens)
                return
            except _ConnectFailed as exc:
                stats.note(record, {"replica": spec.id, "route": record["route"], "place": record["place"],
                                    "outcome": "connect_failed", "why": str(exc)})
                pool.mark_down(spec.id, str(exc))
                tried.add(spec.id)
            finally:
                pool.release(spec.id, ok=ok)
        stats.note(record, outcome="no_replica", status=503, total_ms=_ms(started))
        self._send_error_json(503, "no replica accepted the connection", "overloaded_error")

    def _offload_checked(self, sid: str, replica: ReplicaSpec, tokens: list[int], recheck) -> bool:
        """Reserve the service (bounded wait), check the route is still admissible after that wait, then prefill +
        hand off; True when the state reached the replica. `recheck()` returns None or why not."""
        resource = self.server.joint_reservation
        if resource is not None:
            with resource.use("prefill") as allowed:
                if not allowed:
                    self._offload_why = "accelerator_reserved"
                    return False
                return self._offload_service_checked(sid, replica, tokens, recheck)
        return self._offload_service_checked(sid, replica, tokens, recheck)

    def _offload_service_checked(self, sid: str, replica: ReplicaSpec, tokens: list[int], recheck) -> bool:
        services = self.server.services
        prof = self.server.service_profile(sid)
        hold = (prof.service_s + prof.prefill(len(tokens)) + prof.handoff(len(tokens))) if prof else 1.0
        token = services.reserve(sid, hold)
        started = ok = False
        t0 = time.monotonic()
        try:
            started = services.start(sid, token)
            if not started:
                self._offload_why = "wait"
                return False
            why = recheck()
            if why is not None:
                self._offload_why = "not admitted after the wait: " + why
                ok = True   # nothing was sent: not a service failure
                return False
            ok, self._offload_why = self._offload(self.server.service_spec(sid), replica, tokens)
            return ok
        finally:
            services.finish(sid, token, started=started, ok=ok)
            if ok and self._offload_why == "":
                services.offload_ms += (time.monotonic() - t0) * 1000

    def _forward_balanced(self, path: str, body: bytes | None, tokens: list[int], out_tokens: int) -> bool:
        """Reserve the decode slot atomically before prefill, including service waits.

        Outstanding generations are the conservative active-decode signal. Taking the
        slot first prevents simultaneous cold arrivals from all choosing one replica.
        A session hint alone never earns affinity; a restorable block must lower J.
        """
        server, tried = self.server, set()
        pool, config, services = server.pool, server.config.prefill, server.services
        for _ in pool.specs():
            view = pool.load_view(frozenset(tried))
            locals_ = [LocalCandidate(rid, wait, active, server.mirrors[rid].lookup(tokens), config.node_profile)
                       for rid, wait, active in view] if config.node_profile else []
            scores = {c.replica: local_cost(len(tokens), c) for c in locals_}
            try:
                spec = pool.acquire(frozenset(tried), scores=scores)
            except NoReplicaAvailable as exc:
                self._send_error_json(503, exc.reason, "overloaded_error")
                return True
            ok, estimate = False, None
            try:
                cached = server.mirrors[spec.id].lookup(tokens)
                active = max(0, pool.inflight(spec.id) - 1)
                service_cands = [ServiceCandidate(sid, services.pending_s(sid), server.service_profile(sid))
                                 for sid in services.up_ids() if server.service_profile(sid) is not None] if services else []
                sid = None
                self._cost_note = "balanced active decodes"
                if config.policy == "rtx" and service_cands:
                    sid = min(service_cands, key=lambda s: s.pending_s).service
                elif config.node_profile:
                    c = LocalCandidate(spec.id, 0.0, active, cached, config.node_profile)
                    choice = choose(len(tokens), [c], service_cands, config.cost)
                    sid = choice.service
                    self._cost_note += f";J={choice.cost_s:.3f}"
                note = "cache" if cached else "local"
                if sid is not None:
                    note = (f"offload:{sid}" if self._offload_reserved(sid, spec, tokens)
                            else f"fallback:{self._offload_why}")
                self._prefill_note = note
                if services:
                    services.count("offload" if note.startswith("offload") else
                                   "fallback" if note.startswith("fallback") else note)
                rows = 0 if note.startswith("offload") else len(tokens) - min(cached, len(tokens))
                per_token = config.decode_s_per_token(max(1, pool.inflight(spec.id))) if config.decode_s_per_token else 0.0
                estimate = pool.add_estimate(spec.id, (config.node_profile.prefill(rows) if config.node_profile else 0.0)
                                             + out_tokens * per_token)
                ok = self._exchange(spec, path, body)
                if ok and not note.startswith("offload"):
                    server.mirrors[spec.id].record(tokens)
                return True
            except _ConnectFailed as exc:
                pool.mark_down(spec.id, str(exc))
                tried.add(spec.id)
            finally:
                if estimate is not None:
                    pool.drop_estimate(spec.id, estimate)
                pool.release(spec.id, ok=ok)
        self._send_error_json(503, "no replica accepted the connection", "overloaded_error")
        return True

    def _offload_reserved(self, sid: str, replica: ReplicaSpec, tokens: list[int]) -> bool:
        """Reserve the service, wait for it, prefill + hand off; True when the state reached the replica."""
        resource = self.server.joint_reservation
        if resource is not None:
            with resource.use("prefill") as allowed:
                if not allowed:
                    self._offload_why = "accelerator_reserved"
                    return False
                return self._offload_service(sid, replica, tokens)
        return self._offload_service(sid, replica, tokens)

    def _offload_service(self, sid: str, replica: ReplicaSpec, tokens: list[int]) -> bool:
        services = self.server.services
        assert services is not None
        prof = self.server.service_profile(sid)
        hold = (prof.service_s + prof.prefill(len(tokens)) + prof.handoff(len(tokens))) if prof else 1.0
        token = services.reserve(sid, hold)
        started = ok = False
        t0 = time.monotonic()
        try:
            started = services.start(sid, token)
            if not started:
                self._offload_why = "wait"
                return False
            ok, self._offload_why = self._offload(self.server.service_spec(sid), replica, tokens)
            return ok
        finally:
            services.finish(sid, token, started=started, ok=ok)
            if ok:
                services.offload_ms += (time.monotonic() - t0) * 1000

    def _tokenize(self, spec: ReplicaSpec, path: str, data: dict) -> list[int] | None:
        if path == "/v1/completions":
            prompt = data.get("prompt")
            if not isinstance(prompt, str):
                return None
            request = {"prompt": prompt}
        else:
            if not isinstance(data.get("messages"), list):
                return None
            request = data  # the replica applies its own chat template, as for the generation itself
        config = self.server.config
        conn = http.client.HTTPConnection(spec.address, spec.port, timeout=config.connect_timeout_s)
        try:
            payload = json.dumps(request).encode("utf-8")
            conn.request("POST", config.prefill.tokenize_path, body=payload,
                         headers={**self._upstream_headers(spec, payload), "Content-Type": "application/json"})
            resp = conn.getresponse()
            reply = json.loads(resp.read(64 * 1024 * 1024))
            tokens = reply.get("tokens") if resp.status == 200 and isinstance(reply, dict) else None
            if isinstance(tokens, list) and tokens and all(isinstance(t, int) and t >= 0 for t in tokens):
                return tokens
            return None
        except (OSError, http.client.HTTPException, ValueError):
            return None
        finally:
            conn.close()

    def _offload(self, service: ReplicaSpec, spec: ReplicaSpec, tokens: list[int]) -> tuple[bool, str]:
        config = self.server.config
        body = json.dumps({
            "tokens": tokens,
            "handoff": f"http://{spec.address}:{spec.port}{config.prefill.handoff_path}",
            "session": f"{spec.id}-{time.monotonic_ns()}",
        }).encode("utf-8")
        conn = http.client.HTTPConnection(service.address, service.port, timeout=config.connect_timeout_s)
        try:
            try:
                conn.connect()
            except OSError as exc:
                self.server.services.mark_down(service.id, f"connect {type(exc).__name__}: {exc}")
                return False, "connect"
            conn.sock.settimeout(config.read_timeout_s)
            headers = {"Connection": "close", "Content-Type": "application/json", "Content-Length": str(len(body))}
            if service.api_key is not None:
                headers["Authorization"] = f"Bearer {service.api_key}"
            conn.request("POST", config.prefill.prefill_path, body=body, headers=headers)
            resp = conn.getresponse()
            resp.read(1024 * 1024)
            return (True, "") if resp.status == 200 else (False, f"http{resp.status}")
        except (OSError, http.client.HTTPException):
            return False, "io"
        finally:
            conn.close()

    def _upstream_headers(self, spec: ReplicaSpec, body: bytes | None) -> dict[str, str]:
        headers = {"Connection": "close"}
        for name in FORWARDED_REQUEST_HEADERS:
            value = self.headers.get(name)
            if value is not None:
                headers[name] = value
        if body is not None:
            headers["Content-Length"] = str(len(body))
        if spec.api_key is not None:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        record = getattr(self, "_route_record", None)
        if record is not None:
            headers["X-Request-Id"] = record["id"]     # lets an engine that logs it join its statistics
        return headers

    def _exchange(self, spec: ReplicaSpec, path: str, body: bytes | None) -> bool:
        """Relay one request; True when the whole upstream response reached the client."""
        config = self.server.config
        conn = http.client.HTTPConnection(spec.address, spec.port, timeout=config.connect_timeout_s)
        try:
            try:
                conn.connect()
            except OSError as exc:
                raise _ConnectFailed(f"connect {type(exc).__name__}: {exc}") from exc
            conn.sock.settimeout(config.read_timeout_s)
            try:
                conn.request("POST" if body is not None else "GET", path, body=body,
                             headers=self._upstream_headers(spec, body))
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                # Sent (or partly sent): never replayed on another replica.
                self._relay_status, self._relay_outcome = 502, "upstream_error"
                self._send_error_json(502, f"replica {spec.id} failed: {type(exc).__name__}", "api_error")
                self.close_connection = True
                return False
            return self._relay(spec, resp)
        except _ClientGone:
            self._relay_outcome = "client_gone"
            self.close_connection = True
            return False
        finally:
            # Closing the upstream socket is how a client disconnect reaches the engine.
            conn.close()

    def _write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
            self.wfile.flush()
        except OSError as exc:
            raise _ClientGone() from exc

    def _relay(self, spec: ReplicaSpec, resp: http.client.HTTPResponse) -> bool:
        length = resp.getheader("Content-Length")
        self._relay_status, self._relay_first_byte = resp.status, time.monotonic()
        try:
            self.send_response(resp.status)
            for name in FORWARDED_RESPONSE_HEADERS:
                value = resp.getheader(name)
                if value is not None:
                    self.send_header(name, value)
            self.send_header("X-TensorMeld-Replica", spec.id)
            note = getattr(self, "_prefill_note", None)
            if note is not None:
                self.send_header("X-TensorMeld-Prefill", note)
            cost = getattr(self, "_cost_note", None)
            if cost:
                self.send_header("X-TensorMeld-Route-Cost", cost)
            record = getattr(self, "_route_record", None)
            if record is not None:
                self.send_header("X-TensorMeld-Route-Id", record["id"])
            if length is not None and length.isdigit():
                self.send_header("Content-Length", length)
            else:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        except OSError as exc:
            raise _ClientGone() from exc
        chunked = length is None or not length.isdigit()
        remaining = None if chunked else int(length)
        while True:
            try:
                data = resp.read1(PIPE_CHUNK_BYTES)
            except (OSError, http.client.HTTPException):
                # Upstream broke mid-response: end the client response visibly incomplete.
                self._relay_outcome = "upstream_error"
                self.close_connection = True
                return False
            if not data:
                break
            if remaining is not None:
                remaining -= len(data)
            self._write(b"%x\r\n%s\r\n" % (len(data), data) if chunked else data)
        if remaining:
            # The upstream closed before its declared Content-Length: the client got an incomplete response.
            self._relay_outcome = "upstream_error"
            self.close_connection = True
            return False
        if chunked:
            self._write(b"0\r\n\r\n")
        return 200 <= resp.status < 300


class ReplicaRouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: RouterConfig, pool: ReplicaPool, joint_reservation: JointReservation | None = None):
        self.joint_reservation = joint_reservation
        self.config = config
        self.pool = pool
        self.status_extra: Any = None
        self.admin_routes: dict[str, Any] = {}
        self.offload_stats: OffloadStats | None = OffloadStats() if config.prefill_offload is not None else None
        self.offload_log: Any = None  # callable(record) for per-request prefill_offload records
        self.affinity: AffinityTable | None = (
            AffinityTable(config.affinity.max_entries) if config.affinity.mode == "prefix" else None
        )
        table = config.prefill.routes
        self.route_state: StateFile | None = StateFile(table.state_file, table.max_age_s) if table else None
        self.route_stats: RouteStats | None = RouteStats(table) if table else None
        self.services: ServicePool | None = (
            ServicePool(config.prefill.services, config.failures_to_down, config.prefill.wait_s)
            if config.prefill.services or config.prefill.policy != "nodes"
            else None
        )
        self.mirrors = {
            r.id: CacheMirror(config.prefill.cache_block_tokens, config.prefill.cache_entries) for r in config.replicas
        }
        self._service_profiles = dict(zip((sv.id for sv in config.prefill.services), config.prefill.service_profiles))
        self._service_specs = {sv.id: sv for sv in config.prefill.services}
        self._replica_specs = {r.id: r for r in config.replicas}
        self._connections = threading.BoundedSemaphore(config.max_connections)
        super().__init__((config.listen_address, config.listen_port), _RouterHandler)

    def service_profile(self, sid: str) -> ServiceProfile | None:
        return self._service_profiles.get(sid)

    def service_spec(self, sid: str) -> ReplicaSpec:
        return self._service_specs[sid]

    def replica_spec(self, rid: str) -> ReplicaSpec:
        return self._replica_specs[rid]

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._connections.acquire(blocking=False):
            try:
                body = b'{"error":{"message":"too many connections","type":"overloaded_error"}}'
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                    b"Connection: close\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


class ReplicaRouter:
    """Router front end plus its health monitor; `start()` returns once listening."""

    def __init__(self, config: RouterConfig, *, joint_reservation: JointReservation | None = None):
        self.config = config
        self.pool = ReplicaPool(config)
        self.server = ReplicaRouterServer(config, self.pool, joint_reservation)
        self.monitor = HealthMonitor(self.pool, self.server.services)
        self._thread = threading.Thread(target=self.server.serve_forever, name="tensormeld-router", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self.monitor.check_once()
        self.monitor.start()
        self._thread.start()

    def wait_for_replicas(self, count: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if sum(r["state"] == "up" for r in self.pool.snapshot()["replicas"]) >= count:
                return True
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.monitor.stop()
        self._thread.join(timeout=5)


def serve_replicas(config: RouterConfig, stop: threading.Event, log: Any = None) -> dict[str, Any]:
    """Run the router until `stop` is set; return the final status snapshot."""
    router = ReplicaRouter(config)
    if log is not None and config.prefill_offload is not None:
        router.server.offload_log = lambda rec: print(
            "tensormeld router: prefill_offload " + json.dumps(rec, separators=(",", ":")), file=log, flush=True)
    router.start()
    host, port = router.address
    if log is not None:
        print(f"tensormeld router: listening on {host}:{port}", file=log, flush=True)
    try:
        stop.wait()
    finally:
        router.stop()
    return router.pool.snapshot()


__all__ = [
    "CacheMirror",
    "PrefillConfig",
    "ServicePool",
    "AffinityConfig",
    "AffinityTable",
    "affinity_keys",
    "ROUTER_CONFIG_SCHEMA",
    "ROUTER_STATUS_SCHEMA",
    "NoReplicaAvailable",
    "ReplicaPool",
    "ReplicaRouter",
    "ReplicaSpec",
    "RouterConfig",
    "load_router_config",
    "probe_replica_health",
    "serve_replicas",
]
