"""Router-side prefill offload to a GPU prefill service, decode on the replica the router holds.

Opt-in (`prefill_offload` in the replica-router config); absent means off and the router
forwards exactly as without this module. Per generation request (`/v1/completions`,
`/v1/chat/completions`) the router first acquires and holds a replica as usual, then:

1. gets the prompt's token ids: token-id prompts as given; text prompts and chat bodies from
   the held replica's `/tokenize` (the replica applies the same chat template it applies to
   the generation itself, so the ids are exact);
2. decides the route (`decide`): local below `min_tokens`, or when the effective prompt is at
   or below the threshold (`local_max_tokens` override, else the route section's value, else
   the measured crossover of a `tensormeld/ttft-curve-v1` receipt); otherwise asks the
   service's `/health` and offloads only if it is ok, the prompt fits its `max_positions`,
   and, when it is busy or has a queue, `offload(n) + q_rtx < local(n)` with
   `q_rtx = (busy + queued) * offload(n)` (every job ahead counted as one prefill of the same
   length; `q_local = 0` because the held replica runs only this request);
3. on offload, checks the replica's import capability (`GET <import_path>/layout`), then
   `POST /v1/prefill {tokens, handoff: <replica import URL>, reserve_tokens, session}` and
   waits; whatever the result, the original request is then forwarded unchanged to the same
   replica (after a successful import it restores the parked state; otherwise it prefills
   itself).

Optional, both off when absent: `handoff_codec` ("bp-v1": lossless encoded chunks when the
replica decodes them) and `cooperative_local_rows` ([[min_tokens, q], ...]: for a prompt of at
least min_tokens the service also POSTs the replica's own completion of prompt[0, q] while it
computes and sends only the rows >= q; the replica, started with its prefix state cache, restores
its own checkpoint of [0, q). q must be a multiple of the replica's prefix-cache block; the
largest matching row wins).

Nothing here can change tokens: the replica checks the parked state against the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import http.client
import ipaddress
import json
import math
from pathlib import Path
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

from .prefill_route import (
    PREFILL_ROUTE_SCHEMA,
    TTFT_CURVE_SCHEMA,
    PrefillRoute,
    TtftCurves,
    interpolate,
    parse_ttft_curves,
    route_section,
)
from .schema import ValidationError, number, record, text

PREFILL_OFFLOAD_KEYS = {
    "endpoint", "api_key_file", "import_path", "timeout_s", "route", "local_max_tokens", "min_tokens",
    "default_reserve_tokens", "health_timeout_s", "prefill_path", "health_path", "handoff_codec",
    "cooperative_local_rows",
}
HANDOFF_CODECS = ("identity", "bp-v1")
MAX_RECEIPT_BYTES = 256 * 1024
RECENT_RECORDS = 64


@dataclass(frozen=True)
class Endpoint:
    address: str
    port: int
    path: str = ""

    def url(self, path: str | None = None) -> str:
        return f"http://{self.address}:{self.port}{self.path if path is None else path}"


@dataclass(frozen=True)
class PrefillOffloadConfig:
    service: Endpoint
    api_key: str | None
    route: PrefillRoute
    import_urls: dict  # replica id -> Endpoint (with the import path)
    timeout_s: float = 120.0
    health_timeout_s: float = 1.0
    min_tokens: int = 1024
    default_reserve_tokens: int = 256
    prefill_path: str = "/v1/prefill"
    health_path: str = "/health"
    handoff_codec: str | None = None
    cooperative_local_rows: tuple = ()  # ((min_tokens, q), ...) ascending by min_tokens


def local_rows_for(config: PrefillOffloadConfig, tokens: int) -> int:
    """The cooperative q for a prompt of `tokens` ids (0: the service sends every row)."""
    q = 0
    for min_tokens, rows in config.cooperative_local_rows:
        if tokens >= min_tokens and rows < tokens - 1:
            q = rows
    return q


def _ip(value: str, where: str) -> str:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValidationError(f"{where}: must be a numeric IP literal") from exc
    if ip.is_unspecified or ip.is_multicast or ip.is_link_local or not (ip.is_loopback or ip.is_private):
        raise ValidationError(f"{where}: must be a loopback or private address")
    return str(ip)


def parse_http_url(value: Any, where: str, *, need_path: bool) -> Endpoint:
    """`http://<numeric loopback/private IP>:<port>[/path]`; no query, fragment or credentials."""
    raw = text(value, where)
    if len(raw) > 256 or any(c in raw for c in "\r\n \\"):
        raise ValidationError(f"{where}: invalid URL")
    parts = urlsplit(raw)
    if parts.scheme != "http" or parts.query or parts.fragment or parts.username or parts.password:
        raise ValidationError(f"{where}: expected http://ADDR:PORT{'/path' if need_path else ''}")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValidationError(f"{where}: invalid port") from exc
    if port is None or not 1 <= port <= 65535 or parts.hostname is None:
        raise ValidationError(f"{where}: explicit address and port required")
    path = parts.path
    if need_path and not path.startswith("/"):
        raise ValidationError(f"{where}: expected a path")
    if not need_path and path not in ("", "/"):
        raise ValidationError(f"{where}: no path expected")
    return Endpoint(_ip(parts.hostname, where), port, path if need_path else "")


def _path(value: Any, where: str) -> str:
    value = text(value, where)
    if not value.startswith("/") or any(c in value for c in "?#\r\n ") or len(value) > 128:
        raise ValidationError(f"{where}: expected an absolute path")
    return value.rstrip("/") or "/"


def _int(value: Any, where: str, low: int, high: int) -> int:
    result = int(number(value, where, low, True))
    if result > high:
        raise ValidationError(f"{where}: expected {low}..{high}")
    return result


def _load_route(raw: Any, base_dir: Path | None, configured: int | None) -> PrefillRoute:
    if isinstance(raw, str):
        path = Path(raw)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        with path.open("rb") as f:
            data = f.read(MAX_RECEIPT_BYTES + 1)
        if len(data) > MAX_RECEIPT_BYTES:
            raise ValidationError("router.prefill_offload.route: receipt exceeds 256 KiB")
        try:
            raw = json.loads(data)
        except (UnicodeError, ValueError) as exc:
            raise ValidationError(f"router.prefill_offload.route: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValidationError("router.prefill_offload.route: expected a prefill-route section or a receipt path")
    if raw.get("schema") == TTFT_CURVE_SCHEMA:
        curves = parse_ttft_curves(raw)
        section = route_section(curves.local_name, curves.offload_name, None, curves, configured)
        return PrefillRoute.from_section(section)
    if raw.get("schema") == PREFILL_ROUTE_SCHEMA:
        route = PrefillRoute.from_section(raw)
        if configured is not None:
            route = PrefillRoute(configured, "configured", route.local, route.offload, route.curves).validate()
        return route
    raise ValidationError(f"router.prefill_offload.route: schema must be {PREFILL_ROUTE_SCHEMA} or {TTFT_CURVE_SCHEMA}")


def parse_prefill_offload(raw: Any, replicas: tuple, replica_import_urls: dict, *, base_dir: Path | None,
                          read_key: Any, default_reserve: int) -> PrefillOffloadConfig:
    where = "router.prefill_offload"
    raw = record(raw, where, {"endpoint", "route"}, PREFILL_OFFLOAD_KEYS - {"endpoint", "route"})
    service = parse_http_url(raw["endpoint"], f"{where}.endpoint", need_path=False)
    configured = (_int(raw["local_max_tokens"], f"{where}.local_max_tokens", 0, 1 << 24)
                  if "local_max_tokens" in raw else None)
    route = _load_route(raw["route"], base_dir, configured)
    import_path = _path(raw.get("import_path", "/admin/rtx-import"), f"{where}.import_path")
    urls = {}
    for r in replicas:
        if r.id in replica_import_urls:
            urls[r.id] = parse_http_url(replica_import_urls[r.id], f"router.replicas[{r.id}].import_url", need_path=True)
        else:
            urls[r.id] = Endpoint(r.address, r.port, import_path)
    timeout = float(number(raw.get("timeout_s", 120.0), f"{where}.timeout_s"))
    health_timeout = float(number(raw.get("health_timeout_s", 1.0), f"{where}.health_timeout_s"))
    if not 0.05 <= timeout <= 3600 or not 0.05 <= health_timeout <= 60:
        raise ValidationError(f"{where}: timeout_s 0.05..3600, health_timeout_s 0.05..60")
    codec = raw.get("handoff_codec")
    if codec is not None and codec not in HANDOFF_CODECS:
        raise ValidationError(f"{where}.handoff_codec: one of {', '.join(HANDOFF_CODECS)}")
    coop = []
    rows_raw = raw.get("cooperative_local_rows", [])
    if not isinstance(rows_raw, list):
        raise ValidationError(f"{where}.cooperative_local_rows: expected a list of [min_tokens, local_rows]")
    for i, pair in enumerate(rows_raw):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValidationError(f"{where}.cooperative_local_rows[{i}]: expected [min_tokens, local_rows]")
        lo = _int(pair[0], f"{where}.cooperative_local_rows[{i}][0]", 2, 1 << 24)
        q = _int(pair[1], f"{where}.cooperative_local_rows[{i}][1]", 1, 1 << 24)
        if q >= lo - 1:
            raise ValidationError(f"{where}.cooperative_local_rows[{i}]: local_rows must be below min_tokens - 1")
        coop.append((lo, q))
    coop.sort()
    return PrefillOffloadConfig(
        service=service,
        api_key=read_key(raw["api_key_file"], f"{where}.api_key_file") if "api_key_file" in raw else None,
        route=route,
        import_urls=urls,
        timeout_s=timeout,
        health_timeout_s=health_timeout,
        min_tokens=_int(raw.get("min_tokens", 1024), f"{where}.min_tokens", 0, 1 << 24),
        default_reserve_tokens=_int(raw.get("default_reserve_tokens", default_reserve),
                                    f"{where}.default_reserve_tokens", 0, 1 << 20),
        prefill_path=_path(raw.get("prefill_path", "/v1/prefill"), f"{where}.prefill_path"),
        health_path=_path(raw.get("health_path", "/health"), f"{where}.health_path"),
        handoff_codec=codec,
        cooperative_local_rows=tuple(coop),
    )


# Decision ---------------------------------------------------------------------------------

def decide(route: PrefillRoute, tokens: int, min_tokens: int, health: dict | None) -> dict:
    """Route one prompt of `tokens` ids. `health` is the service's `/health` reply (None when
    it was not asked or did not answer). Returns {route, reason, ...} with route local|offload."""
    out: dict = {"tokens": tokens, "local_max_tokens": route.local_max_tokens, "source": route.source}
    if tokens < min_tokens:
        return {**out, "route": "local", "reason": "below_min_tokens"}
    if tokens <= route.local_max_tokens:
        return {**out, "route": "local", "reason": "below_threshold"}
    if health is None:
        return {**out, "route": "local", "reason": "service_unreachable"}
    if health.get("status") != "ok":
        return {**out, "route": "local", "reason": "service_failed"}
    max_pos = health.get("max_positions")
    if isinstance(max_pos, int) and not isinstance(max_pos, bool) and max_pos > 0 and tokens > max_pos:
        return {**out, "route": "local", "reason": "service_max_positions"}
    busy = health.get("busy") is True
    queued = health.get("queued") if isinstance(health.get("queued"), int) and health.get("queued") > 0 else 0
    ahead = int(busy) + queued
    if not ahead:
        return {**out, "route": "offload", "reason": "threshold"}
    curves: TtftCurves | None = route.curves
    if curves is None:
        # No curves to price the queue: a busy service never wins.
        return {**out, "route": "local", "reason": "service_busy", "ahead": ahead}
    local_s = interpolate(curves.local, tokens)        # + q_local = 0: the held replica runs only this request
    offload_s = interpolate(curves.offload, tokens)
    q_rtx = ahead * offload_s                          # every job ahead priced as one prefill of this length
    out.update(local_s=round(local_s, 3), offload_s=round(offload_s, 3), q_rtx_s=round(q_rtx, 3), ahead=ahead)
    if offload_s + q_rtx < local_s:
        return {**out, "route": "offload", "reason": "queue_priced"}
    return {**out, "route": "local", "reason": "service_queue"}


# HTTP ---------------------------------------------------------------------------------------

def _request(ep: Endpoint, method: str, path: str, body: bytes | None, timeout: float,
             key: str | None, limit: int = 1024 * 1024) -> tuple[int, Any]:
    conn = http.client.HTTPConnection(ep.address, ep.port, timeout=timeout)
    try:
        headers = {"Connection": "close"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read(limit)
        try:
            value = json.loads(raw) if raw else None
        except (UnicodeError, ValueError):
            value = None
        return resp.status, value
    finally:
        conn.close()


def service_health(config: PrefillOffloadConfig) -> dict | None:
    try:
        status, value = _request(config.service, "GET", config.health_path, None, config.health_timeout_s,
                                 config.api_key, 64 * 1024)
    except (OSError, http.client.HTTPException):
        return None
    if not isinstance(value, dict):
        return None if status != 200 else {"status": "failed"}
    if status != 200 and value.get("status") == "ok":
        value = {**value, "status": "failed"}
    return value


def replica_can_import(config: PrefillOffloadConfig, rid: str, replica_key: str | None) -> bool:
    ep: Endpoint = config.import_urls[rid]
    try:
        status, _ = _request(ep, "GET", ep.path.rstrip("/") + "/layout", None, config.health_timeout_s,
                             replica_key, 64 * 1024)
    except (OSError, http.client.HTTPException):
        return False
    return status == 200


def reserve_tokens(data: dict, default: int) -> int:
    for key in ("max_completion_tokens", "max_tokens"):
        v = data.get(key)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return default


def prefill(config: PrefillOffloadConfig, rid: str, tokens: list[int], reserve: int) -> dict:
    """POST the prefill; returns {ok, reason, status, seconds, position, session}."""
    ep: Endpoint = config.import_urls[rid]
    session = f"{rid}-{uuid.uuid4().hex[:16]}"
    req: dict = {"tokens": tokens, "handoff": ep.url(), "reserve_tokens": reserve, "session": session}
    if config.handoff_codec is not None:
        req["handoff_codec"] = config.handoff_codec
    q = local_rows_for(config, len(tokens))
    if q:
        req["local_rows"] = q
        req["node_completions"] = ep.url("/v1/completions")
    body = json.dumps(req, separators=(",", ":")).encode()
    t0 = time.monotonic()
    out: dict = {"session": session, "handoff": ep.url()}
    if q:
        out["local_rows"] = q
    try:
        status, value = _request(config.service, "POST", config.prefill_path, body, config.timeout_s, config.api_key)
    except TimeoutError:
        return {**out, "ok": False, "reason": "timeout", "wall_s": round(time.monotonic() - t0, 3)}
    except (OSError, http.client.HTTPException) as exc:
        reason = "timeout" if "timed out" in str(exc) else "io"
        return {**out, "ok": False, "reason": reason, "wall_s": round(time.monotonic() - t0, 3)}
    out.update(status=status, wall_s=round(time.monotonic() - t0, 3))
    if status != 200 or not isinstance(value, dict):
        return {**out, "ok": False, "reason": f"http{status}"}
    secs = value.get("seconds")
    if isinstance(secs, dict):
        out["seconds"] = {k: v for k, v in secs.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    position = value.get("position")
    out["position"] = position
    if position != len(tokens) - 1:
        return {**out, "ok": False, "reason": "position_mismatch"}
    return {**out, "ok": True, "reason": ""}


class OffloadStats:
    """Per-request records (bounded) and counters for /api/status and /metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts = {"local": 0, "offload": 0, "fallback": 0}
        self.reasons: dict[str, int] = {}
        self.rtx_seconds = 0.0
        self.recent: deque = deque(maxlen=RECENT_RECORDS)

    def record(self, rec: dict) -> None:
        with self._lock:
            kind = "fallback" if rec.get("fallback") else rec["route"]
            self.counts[kind] += 1
            reason = rec.get("fallback") or rec.get("reason") or ""
            if reason:
                self.reasons[reason] = self.reasons.get(reason, 0) + 1
            self.rtx_seconds += float(rec.get("rtx_wall_s") or 0.0)
            self.recent.append(rec)

    def totals(self) -> tuple[dict, dict, float]:
        with self._lock:
            return dict(self.counts), dict(self.reasons), self.rtx_seconds

    def snapshot(self, config: PrefillOffloadConfig) -> dict:
        with self._lock:
            return {
                "endpoint": config.service.url(),
                "local_max_tokens": config.route.local_max_tokens,
                "source": config.route.source,
                "min_tokens": config.min_tokens,
                "counts": dict(self.counts),
                "reasons": dict(self.reasons),
                "rtx_seconds_total": round(self.rtx_seconds, 3),
                "recent": list(self.recent),
            }


__all__ = [
    "Endpoint",
    "OffloadStats",
    "PrefillOffloadConfig",
    "decide",
    "parse_http_url",
    "parse_prefill_offload",
    "prefill",
    "replica_can_import",
    "reserve_tokens",
    "service_health",
]
