"""Length routing of the prefill between the decode placement and an offload placement.

A ``prefill-decode`` plan pays a fixed price to prefill elsewhere (the offload
device's setup and weight streaming, the state export, the wire, the import on
the decode set).  Short prompts are faster on the decode set itself; long
prompts are faster on the offload set.  The plan therefore carries one number,
``local_max_tokens``: a request whose *effective* prompt length is at or below
it prefills where it decodes ("local"); a longer one goes to the offload
placement and its state moves once ("offload").

Effective length = prompt tokens - tokens served by a prefix-cache hit on the
decode set.  A hit leaves only the uncached suffix to prefill locally, while the
offload placement would have to redo (or first receive) the cached prefix, so a
hit pulls the request back to the local route.  Without a hit the rule is the
threshold.  With a hit and measured curves in the route, the two TTFTs are
compared directly: local(suffix) + cached_s_per_token * cached against
offload(whole prompt); without curves the threshold applies to the suffix.

Where the threshold comes from, in order of precedence:

``configured``  given by the operator (``--prefill-local-max-tokens``);
``measured``    the crossover of two measured end-to-end TTFT curves
                (``tensormeld/ttft-curve-v1`` receipt, ``--ttft-receipt``);
``model``       the crossover of the planner's own cost model (unmeasured).

Nothing here starts or moves anything; the route is data for the front/serve
process that admits requests.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .schema import ValidationError

PREFILL_ROUTE_SCHEMA = "tensormeld/prefill-route-v1"
TTFT_CURVE_SCHEMA = "tensormeld/ttft-curve-v1"
ROUTES = ("local", "offload")
SOURCES = ("configured", "measured", "model")
MAX_TOKENS = 1 << 24
MAX_POINTS = 64


@dataclass(frozen=True)
class CurvePoint:
    tokens: int
    ttft_s: float
    provenance: str  # measured | estimate


def _points(raw: Any, where: str) -> tuple[CurvePoint, ...]:
    if not isinstance(raw, list) or not 2 <= len(raw) <= MAX_POINTS:
        raise ValidationError(f"{where}: expected 2..{MAX_POINTS} points")
    out = []
    for i, p in enumerate(raw):
        if not isinstance(p, dict) or set(p) - {"tokens", "ttft_s", "provenance", "source", "parts"}:
            raise ValidationError(f"{where}[{i}]: expected tokens, ttft_s, provenance[, source, parts]")
        n, t = p.get("tokens"), p.get("ttft_s")
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_TOKENS:
            raise ValidationError(f"{where}[{i}].tokens: out of range")
        if not isinstance(t, (int, float)) or isinstance(t, bool) or not 0 < t < 1e6 or not math.isfinite(t):
            raise ValidationError(f"{where}[{i}].ttft_s: out of range")
        prov = p.get("provenance", "measured")
        if prov not in ("measured", "estimate"):
            raise ValidationError(f"{where}[{i}].provenance: measured|estimate")
        out.append(CurvePoint(n, float(t), prov))
    out.sort(key=lambda p: p.tokens)
    if len({p.tokens for p in out}) != len(out):
        raise ValidationError(f"{where}: duplicate token counts")
    return tuple(out)


@dataclass(frozen=True)
class TtftCurves:
    local: tuple[CurvePoint, ...]
    offload: tuple[CurvePoint, ...]
    local_name: str
    offload_name: str
    cached_s_per_token: float = 0.0  # local route: extra seconds per prefix-cache-hit token of context

    @property
    def provenance(self) -> str:
        pts = self.local + self.offload
        return "measured" if all(p.provenance == "measured" for p in pts) else "partly-estimated"


def parse_ttft_curves(raw: Any) -> TtftCurves:
    """``{"schema": "tensormeld/ttft-curve-v1", "routes": {"local": {"name", "points"}, "offload": {...}}}``"""
    if not isinstance(raw, dict) or raw.get("schema") != TTFT_CURVE_SCHEMA:
        raise ValidationError(f"ttft receipt: schema must be {TTFT_CURVE_SCHEMA}")
    routes = raw.get("routes")
    if not isinstance(routes, dict) or set(routes) != set(ROUTES):
        raise ValidationError("ttft receipt: routes must have exactly local and offload")
    names = {}
    pts = {}
    for r in ROUTES:
        v = routes[r]
        if not isinstance(v, dict) or not isinstance(v.get("name", ""), str):
            raise ValidationError(f"ttft receipt: routes.{r} must be an object")
        if set(v) - {"name", "points", "cached_s_per_token", "notes"}:
            raise ValidationError(f"ttft receipt: routes.{r}: unknown member")
        names[r] = v.get("name", r)
        pts[r] = _points(v.get("points"), f"routes.{r}.points")
    k = routes["local"].get("cached_s_per_token", 0.0)
    if not isinstance(k, (int, float)) or isinstance(k, bool) or not 0 <= k < 1 or not math.isfinite(k):
        raise ValidationError("ttft receipt: routes.local.cached_s_per_token out of range")
    return TtftCurves(pts["local"], pts["offload"], names["local"], names["offload"], float(k))


def load_ttft_curves(path: str | Path) -> TtftCurves:
    return parse_ttft_curves(json.loads(Path(path).read_text(encoding="utf-8")))


def interpolate(points: tuple[CurvePoint, ...], tokens: int) -> float:
    """Piecewise linear through the points; the end segments extend linearly (never below 0)."""
    if tokens <= points[0].tokens:
        a, b = points[0], points[1]
    elif tokens >= points[-1].tokens:
        a, b = points[-2], points[-1]
    else:
        i = next(k for k in range(1, len(points)) if points[k].tokens >= tokens)
        a, b = points[i - 1], points[i]
    slope = (b.ttft_s - a.ttft_s) / (b.tokens - a.tokens)
    return max(0.0, a.ttft_s + (tokens - a.tokens) * slope)


def crossover(local: Callable[[int], float], offload: Callable[[int], float],
              lo: int = 1, hi: int = MAX_TOKENS) -> int | None:
    """Largest token count at which the local route is still no slower (the ``local_max_tokens``).

    Returns ``hi`` when local never loses in range, ``0`` when offload already wins at ``lo``.
    Assumes one crossing (local grows faster than offload), which is what a fixed offload price means.
    """
    if offload(lo) < local(lo):
        return 0
    if offload(hi) >= local(hi):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if offload(mid) < local(mid):
            hi = mid
        else:
            lo = mid
    return lo


def measured_crossover(curves: TtftCurves) -> int | None:
    return crossover(lambda n: interpolate(curves.local, n), lambda n: interpolate(curves.offload, n))


@dataclass(frozen=True)
class PrefillRoute:
    local_max_tokens: int  # effective length <= this prefills locally; 0 = always offload
    source: str
    local: str = "local"
    offload: str = "offload"
    curves: TtftCurves | None = None  # measured TTFT curves: used to route prefix-cache hits

    def validate(self) -> "PrefillRoute":
        if not isinstance(self.local_max_tokens, int) or not 0 <= self.local_max_tokens <= MAX_TOKENS:
            raise ValidationError("local_max_tokens: out of range")
        if self.source not in SOURCES:
            raise ValidationError(f"source: one of {SOURCES}")
        return self

    @staticmethod
    def from_section(section: dict) -> "PrefillRoute":
        """From a plan's ``prefill_route`` object (as emitted by ``route_section``)."""
        if not isinstance(section, dict) or section.get("schema") != PREFILL_ROUTE_SCHEMA:
            raise ValidationError(f"prefill_route: schema must be {PREFILL_ROUTE_SCHEMA}")
        c = section.get("curves")
        curves = None
        if c is not None:
            if not isinstance(c, dict):
                raise ValidationError("prefill_route.curves: expected an object")
            pts = lambda key: [{"tokens": n, "ttft_s": t, "provenance": "measured"}  # noqa: E731
                               for n, t in (c.get(key) or [])]
            curves = parse_ttft_curves({"schema": TTFT_CURVE_SCHEMA, "routes": {
                "local": {"name": section.get("local", "local"), "points": pts("local"),
                          "cached_s_per_token": c.get("cached_s_per_token", 0.0)},
                "offload": {"name": section.get("offload", "offload"), "points": pts("offload")}}})
        return PrefillRoute(section.get("local_max_tokens"), section.get("source"), section.get("local", "local"),
                            section.get("offload", "offload"), curves).validate()

    def decide(self, prompt_tokens: int, cached_tokens: int = 0, offload_up: bool = True) -> dict:
        """Route one request.  ``cached_tokens``: prefix-cache hit on the decode set (0 = miss)."""
        if not 0 <= cached_tokens <= prompt_tokens:
            raise ValidationError("cached_tokens: must be within 0..prompt_tokens")
        effective = prompt_tokens - cached_tokens
        out: dict = {"effective_tokens": effective, "cached_tokens": cached_tokens,
                     "local_max_tokens": self.local_max_tokens}
        if cached_tokens and self.curves is not None:
            local_s = interpolate(self.curves.local, max(effective, 1)) + self.curves.cached_s_per_token * cached_tokens
            offload_s = interpolate(self.curves.offload, prompt_tokens)
            offload = offload_s < local_s
            out.update(rule="curves", local_s=round(local_s, 3), offload_s=round(offload_s, 3))
        else:
            offload = effective > self.local_max_tokens
            out["rule"] = "threshold"
        route = "offload" if offload_up and offload else "local"
        out.update(route=route, placement=self.offload if route == "offload" else self.local)
        return out


def route_section(local_name: str, offload_name: str, model_crossover: int | None,
                  curves: TtftCurves | None = None, configured: int | None = None) -> dict:
    """The plan's ``prefill_route`` object (precedence configured > measured > model)."""
    measured = measured_crossover(curves) if curves else None
    if configured is not None:
        value, source = configured, "configured"
    elif measured is not None:
        value, source = measured, "measured"
    else:
        value, source = model_crossover, "model"
    route = PrefillRoute(int(value if value is not None else MAX_TOKENS), source, local_name, offload_name,
                         curves).validate()
    return {
        "schema": PREFILL_ROUTE_SCHEMA,
        "local": local_name,
        "offload": offload_name,
        "local_max_tokens": route.local_max_tokens,
        "source": source,
        "model_crossover_tokens": model_crossover,
        "measured_crossover_tokens": measured,
        "measured_provenance": curves.provenance if curves else None,
        "receipt_routes": {"local": curves.local_name, "offload": curves.offload_name} if curves else None,
        "effective_tokens": "prompt_tokens - prefix_cache_hit_tokens (hit on the decode placement)",
        "rule": "offload when the offload placement is up and: no hit -> effective_tokens > local_max_tokens; "
                "hit + curves -> offload(prompt) < local(effective) + cached_s_per_token * cached; "
                "hit without curves -> effective_tokens > local_max_tokens",
        "curves": {"local": [[p.tokens, p.ttft_s] for p in curves.local],
                   "offload": [[p.tokens, p.ttft_s] for p in curves.offload],
                   "cached_s_per_token": curves.cached_s_per_token} if curves else None,
    }
