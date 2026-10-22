"""Cost model for placing a request's prefill: on its decode replica (resuming from a cached
prefix when one exists) or on a prefill service (a discrete GPU) that hands the request state
to the replica.

Notation (per request): P prompt tokens after the chat template, C tokens restorable from a
cache snapshot on the replica, U = P - C rows to prefill there.

Candidates and their predicted cost J (the request's own wait + prefill, plus the delay it
causes to requests already decoding on that replica):

- ``local`` on replica r:
  J = W_r + R + F_r(U) + N_r * F_r(U)
  W_r is the predicted wait in seconds for a free decode slot on r (0 with a free slot), R the snapshot restore, F_r the replica's
  measured prefill curve, N_r its active requests: the engine pauses their decodes while it
  prefills, so each of them is delayed by about F_r(U).
- ``offload`` to service s, decoding on replica r:
  J = max(Q_s + S_s + F_s(P) + H_s(P), W_r) + N_r * I
  Q_s is the service's pending work in seconds, S_s its per-request overhead, F_s its prefill
  curve over the WHOLE prompt (a service has no cache), H_s the state handoff, I the import
  pause seen by the replica's active decodes.

The output length cancels between the two (both end with the same decode on r). An offload
is chosen only when it beats the best local candidate by more than
max(margin_s, margin_frac * J_local): the margin only absorbs noise. Without active decodes
on the replica, prompts below `min_tokens_idle` stay local (the service's fixed costs
dominate there); with active decodes the offload is always a candidate and J decides.

Curves are measured (seconds as a function of tokens, piecewise linear, extrapolated from the
last segment). Service profiles are keyed by their GPU (name, VRAM, pinned host memory, PCIe)
and by the model they serve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .schema import ValidationError

MAX_POINTS = 64


@dataclass(frozen=True)
class Curve:
    """Seconds as a function of tokens: piecewise linear through measured points."""

    points: tuple[tuple[int, float], ...]

    @staticmethod
    def parse(raw: Any, where: str) -> "Curve":
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_POINTS:
            raise ValidationError(f"{where}: expected 1..{MAX_POINTS} [tokens, seconds] points")
        pts = []
        for i, p in enumerate(raw):
            if not (isinstance(p, list) and len(p) == 2):
                raise ValidationError(f"{where}[{i}]: expected [tokens, seconds]")
            n, s = p
            if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 1 << 24:
                raise ValidationError(f"{where}[{i}].tokens: out of range")
            if not isinstance(s, (int, float)) or isinstance(s, bool) or not 0 <= s < 1e6 or not math.isfinite(s):
                raise ValidationError(f"{where}[{i}].seconds: out of range")
            pts.append((n, float(s)))
        pts.sort()
        if len({n for n, _ in pts}) != len(pts):
            raise ValidationError(f"{where}: duplicate token counts")
        return Curve(tuple(pts))

    def __call__(self, tokens: int) -> float:
        pts = self.points
        if tokens <= 0:
            return 0.0
        if len(pts) == 1:
            n, s = pts[0]
            return s * tokens / n if n else s
        if tokens <= pts[0][0]:
            (n0, s0), (n1, s1) = pts[0], pts[1]
        elif tokens >= pts[-1][0]:
            (n0, s0), (n1, s1) = pts[-2], pts[-1]
        else:
            for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
                if n0 <= tokens <= n1:
                    break
        value = s0 + (s1 - s0) * (tokens - n0) / (n1 - n0)
        return max(0.0, value)


@dataclass(frozen=True)
class NodeProfile:
    """A decode replica's measured prefill curve and restore cost."""

    prefill: Curve
    restore_s: float = 0.01

    @staticmethod
    def parse(raw: Any, where: str) -> "NodeProfile":
        if not isinstance(raw, dict) or set(raw) - {"prefill_curve", "restore_s"} or "prefill_curve" not in raw:
            raise ValidationError(f"{where}: expected prefill_curve[, restore_s]")
        restore = raw.get("restore_s", 0.01)
        if not isinstance(restore, (int, float)) or not 0 <= restore < 60:
            raise ValidationError(f"{where}.restore_s: out of range")
        return NodeProfile(Curve.parse(raw["prefill_curve"], f"{where}.prefill_curve"), float(restore))


@dataclass(frozen=True)
class GpuKey:
    name: str
    vram_gib: float
    pinned_gib: float
    pcie_gbps: float


@dataclass(frozen=True)
class ServiceProfile:
    """A prefill service's measured curves, keyed by its GPU and model."""

    gpu: GpuKey
    model: str
    prefill: Curve
    handoff: Curve
    service_s: float = 0.06
    import_pause_s: float = 0.05
    provenance: str = "measured"

    @staticmethod
    def parse(raw: Any, where: str) -> "ServiceProfile":
        allowed = {"gpu", "model", "prefill_curve", "handoff_curve", "service_s", "import_pause_s", "provenance"}
        if not isinstance(raw, dict) or set(raw) - allowed or not {"gpu", "model", "prefill_curve", "handoff_curve"} <= set(raw):
            raise ValidationError(f"{where}: expected gpu, model, prefill_curve, handoff_curve[, service_s, import_pause_s, provenance]")
        g = raw["gpu"]
        if not isinstance(g, dict) or set(g) != {"name", "vram_gib", "pinned_gib", "pcie_gbps"}:
            raise ValidationError(f"{where}.gpu: expected name, vram_gib, pinned_gib, pcie_gbps")
        for k in ("vram_gib", "pinned_gib", "pcie_gbps"):
            if not isinstance(g[k], (int, float)) or not 0 <= g[k] < 1e5:
                raise ValidationError(f"{where}.gpu.{k}: out of range")
        if not isinstance(g["name"], str) or not g["name"] or not isinstance(raw["model"], str) or not raw["model"]:
            raise ValidationError(f"{where}: gpu.name and model are non-empty strings")
        prov = raw.get("provenance", "measured")
        if prov not in ("measured", "estimate", "seed"):
            raise ValidationError(f"{where}.provenance: measured|estimate|seed")
        num = {}
        for k, d in (("service_s", 0.06), ("import_pause_s", 0.05)):
            v = raw.get(k, d)
            if not isinstance(v, (int, float)) or not 0 <= v < 60:
                raise ValidationError(f"{where}.{k}: out of range")
            num[k] = float(v)
        return ServiceProfile(
            GpuKey(g["name"], float(g["vram_gib"]), float(g["pinned_gib"]), float(g["pcie_gbps"])),
            raw["model"],
            Curve.parse(raw["prefill_curve"], f"{where}.prefill_curve"),
            Curve.parse(raw["handoff_curve"], f"{where}.handoff_curve"),
            num["service_s"],
            num["import_pause_s"],
            prov,
        )


# Seed profiles: the DeepSeek V4.1 resident prefill service on an RTX PRO 6000 (96 GB) under VRAM caps, fast
# policy, 2K / 8K prompts (hetero/w2-report.md, 2026-10-21: 559/2052, 587/2190, 666/2499, 901/3025 tok/s), with
# the measured handoff (0.15 s at 2K, 0.27 s at 8K) and service overhead (0.05-0.07 s). Budget-cap runs, not
# distinct physical GPUs; a service without a matching profile should be calibrated at startup.
def _seed(cap: float, tps2k: float, tps8k: float) -> ServiceProfile:
    return ServiceProfile(
        GpuKey("RTX PRO 6000 Blackwell", cap, 0.0, 40.0),
        "deepseek41",
        Curve(((2048, 2048 / tps2k), (8192, 8192 / tps8k))),
        Curve(((2048, 0.15), (8192, 0.27))),
        0.06,
        0.05,
        "seed",
    )


SEED_SERVICE_PROFILES: tuple[ServiceProfile, ...] = (
    _seed(24.0, 559.0, 2052.0),
    _seed(32.0, 587.0, 2190.0),
    _seed(48.0, 666.0, 2499.0),
    _seed(74.0, 901.0, 3025.0),
)


@dataclass(frozen=True)
class CostPolicy:
    margin_s: float = 0.2
    margin_frac: float = 0.10
    min_tokens_idle: int = 2048


@dataclass(frozen=True)
class LocalCandidate:
    replica: str
    pending_s: float      # W_r: predicted wait for a free decode slot
    active: int           # N_r
    cached_tokens: int    # C on this replica
    profile: NodeProfile


@dataclass(frozen=True)
class ServiceCandidate:
    service: str
    pending_s: float      # Q_s
    profile: ServiceProfile


@dataclass(frozen=True)
class Choice:
    place: str            # local | offload
    replica: str
    service: str | None
    cost_s: float         # J of the choice
    best_local_s: float
    best_offload_s: float | None
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


def local_cost(prompt_tokens: int, c: LocalCandidate) -> float:
    rows = max(0, prompt_tokens - min(c.cached_tokens, prompt_tokens))
    prefill = c.profile.prefill(rows)
    restore = c.profile.restore_s if c.cached_tokens > 0 else 0.0
    return c.pending_s + restore + prefill + c.active * prefill


def offload_cost(prompt_tokens: int, s: ServiceCandidate, decode_on: LocalCandidate) -> float:
    # The state is usable once both the service path is done and the replica has a slot for it.
    p = s.profile
    ready = s.pending_s + p.service_s + p.prefill(prompt_tokens) + p.handoff(prompt_tokens)
    return max(ready, decode_on.pending_s) + decode_on.active * p.import_pause_s


def choose(prompt_tokens: int, locals_: list[LocalCandidate], services: list[ServiceCandidate],
           policy: CostPolicy) -> Choice:
    """The lowest-J placement. Offloads must win by more than the noise margin."""
    if not locals_:
        raise ValueError("no replica candidate")
    scored = sorted(((local_cost(prompt_tokens, c), c.replica, c) for c in locals_), key=lambda t: (t[0], t[1]))
    best_local_cost, _, best_local = scored[0]
    detail: dict[str, Any] = {"local": {c.replica: round(j, 4) for j, _, c in scored}}
    best_off: tuple[float, ServiceCandidate, LocalCandidate] | None = None
    for s in services:
        for c in locals_:
            if c.active == 0 and prompt_tokens < policy.min_tokens_idle:
                continue  # an idle replica keeps short prompts: the service's fixed costs dominate there
            j = offload_cost(prompt_tokens, s, c)
            if best_off is None or j < best_off[0]:
                best_off = (j, s, c)
    if best_off is None:
        return Choice("local", best_local.replica, None, best_local_cost, best_local_cost, None,
                      "no eligible service", detail)
    off_cost, s, c = best_off
    detail["offload"] = {"service": s.service, "replica": c.replica, "cost_s": round(off_cost, 4)}
    margin = max(policy.margin_s, policy.margin_frac * best_local_cost)
    if best_local_cost - off_cost > margin:
        return Choice("offload", c.replica, s.service, off_cost, best_local_cost, off_cost,
                      f"saves {best_local_cost - off_cost:.3f} s > margin {margin:.3f} s", detail)
    return Choice("local", best_local.replica, None, best_local_cost, best_local_cost, off_cost,
                  f"saving {best_local_cost - off_cost:.3f} s within margin {margin:.3f} s", detail)


__all__ = [
    "Choice", "CostPolicy", "Curve", "GpuKey", "LocalCandidate", "NodeProfile", "SEED_SERVICE_PROFILES",
    "ServiceCandidate", "ServiceProfile", "choose", "local_cost", "offload_cost",
]
