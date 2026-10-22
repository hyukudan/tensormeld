"""Opt-in serving policy and process-local exclusive accelerator reservations.

These contracts select modes; they do not launch workers or migrate model state.
Recorded throughput inputs are predictions, never execution qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
import threading
from typing import Any

from .route_cost import CostPolicy, LocalCandidate, ServiceCandidate, local_cost, offload_cost
from .schema import ValidationError

MODES = frozenset({"pair_ar", "pair_dspark", "split_dspark", "split_adaptive_dspark",
                   "pair_dspark_batch", "pair_batch"})
DOMAINS = frozenset({"code", "en", "es"})
SAFE_POINTS = frozenset({"between_requests", "window_boundary"})


def _finite(value: Any, name: str, low: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < low:
        raise ValidationError(f"{name}: expected finite number >= {low}")
    return float(value)


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValidationError(f"{name}: expected integer {low}..{high}")
    return value


@dataclass(frozen=True)
class Rate:
    mode: str
    load: int
    domain: str
    rows: int
    per_user_tps: float
    provenance: str
    source: str


class RecordedProfiles:
    """Exact table lookup only: no interpolation across load, domain or row limits."""

    def __init__(self, rates: tuple[Rate, ...]):
        if not 1 <= len(rates) <= 512:
            raise ValidationError("rates: expected 1..512 records")
        table = {}
        for r in rates:
            if (not isinstance(r.mode, str) or not isinstance(r.domain, str)
                    or r.mode not in MODES or r.domain not in DOMAINS or type(r.rows) is not int or r.rows not in (8, 16)):
                raise ValidationError("rate: unsupported mode, domain or rows")
            _integer(r.load, "rate.load", 1, 16)
            _finite(r.per_user_tps, "rate.per_user_tps", 1e-9)
            if (r.provenance not in ("measured", "estimate", "replay", "synthetic")
                    or not isinstance(r.source, str) or not 1 <= len(r.source) <= 2048):
                raise ValidationError("rate: provenance and source required")
            key = (r.mode, r.load, r.domain, r.rows)
            if key in table:
                raise ValidationError("rate: duplicate identity")
            table[key] = r
        self._table = table

    @classmethod
    def parse(cls, raw: Any) -> "RecordedProfiles":
        if not isinstance(raw, dict) or set(raw) != {"schema", "rates"} or raw["schema"] != "tensormeld/serving-rates-v1":
            raise ValidationError("profiles: expected serving-rates-v1 schema and rates")
        if not isinstance(raw["rates"], list) or not 1 <= len(raw["rates"]) <= 512:
            raise ValidationError("profiles.rates: expected 1..512 records")
        fields = {"mode", "load", "domain", "rows", "per_user_tps", "provenance", "source"}
        if any(not isinstance(r, dict) or set(r) != fields for r in raw["rates"]):
            raise ValidationError("rate: unexpected fields")
        return cls(tuple(Rate(**r) for r in raw["rates"]))

    def lookup(self, mode: str, load: int, domain: str, rows: int) -> Rate | None:
        return self._table.get((mode, load, domain, rows))


@dataclass(frozen=True)
class ModePolicy:
    enabled: bool = False
    split_enabled: bool = False
    batch_dspark_enabled: bool = False
    rows16_enabled: bool = False
    selective_gain: float = 0.10
    stable_observations: int = 2
    min_dwell_s: float = 1.0

    def __post_init__(self):
        for name in ("enabled", "split_enabled", "batch_dspark_enabled", "rows16_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"{name}: expected boolean")
        _finite(self.selective_gain, "selective_gain")
        _finite(self.min_dwell_s, "min_dwell_s")
        _integer(self.stable_observations, "stable_observations", 1, 1000)


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    rows: int
    accelerator_role: str       # front | prefill | none; not a granted lease
    rate: Rate | None
    reason: str
    executable: bool = False


def recommend(load: int, domain: str, profiles: RecordedProfiles, policy: ModePolicy,
              joint: JointChoice | None = None) -> ModeDecision:
    """Per-load policy; feature flags declare availability, not measured qualification."""
    _integer(load, "load", 1, 16)
    if not isinstance(domain, str) or domain not in DOMAINS:
        raise ValidationError("domain: expected code, en or es")
    rows, mode, role = 8, "pair_ar", "none"
    reason = "policy disabled"
    if policy.enabled:
        role = "prefill" if load >= 2 else "none"
        reason = "per-load policy"
        rows = 16 if policy.rows16_enabled and load >= 2 else 8
        if load == 1:
            mode = "split_dspark" if domain == "code" else "split_adaptive_dspark"
            role = "front"
            if not policy.split_enabled:
                mode, role, reason = "pair_dspark", "none", "split unavailable; pair fallback"
            elif joint is not None and joint.owner == "prefill":
                mode, role, reason = "pair_dspark", "prefill", "J favors prefill over split decode gain"
        elif load <= 4:
            if policy.batch_dspark_enabled:
                mode = "pair_dspark_batch"
                if load >= 3 and domain != "code":
                    ar = profiles.lookup("pair_ar", load, domain, rows)
                    ds = profiles.lookup(mode, load, domain, rows)
                    if ar is None or ds is None or ds.per_user_tps <= ar.per_user_tps * (1 + policy.selective_gain):
                        mode, reason = "pair_ar", "selective speculation does not pay or lacks a profile"
            else:
                reason = "batch speculation unavailable; AR fallback"
        else:
            mode = "pair_batch"
    return ModeDecision(mode, rows, role, profiles.lookup(mode, load, domain, rows), reason)


class ModeController:
    """Serialize mode observations with safe-point debounce and minimum dwell time.

    A window boundary must mean all affected workers have committed or rolled back
    their windows. The caller owns that proof. An observation never changes workers.
    """

    def __init__(self, profiles: RecordedProfiles, policy: ModePolicy):
        self.profiles, self.policy = profiles, policy
        self.current: ModeDecision | None = None
        self.target: ModeDecision | None = None
        self._pending: tuple | None = None
        self._count, self._last_switch, self._last_time = 0, None, None
        self._lock = threading.Lock()

    @property
    def transition_pending(self) -> bool:
        with self._lock:
            if self.target is None:
                return False
            return self.current is None or (self.current.mode, self.current.rows, self.current.accelerator_role) != (
                self.target.mode, self.target.rows, self.target.accelerator_role)

    def observe(self, load: int, domain: str, *, safe_point: str | None, now_s: float,
                joint: JointChoice | None = None) -> ModeDecision | None:
        desired = recommend(load, domain, self.profiles, self.policy, joint)
        _finite(now_s, "now_s")
        if safe_point is not None and safe_point not in SAFE_POINTS:
            raise ValidationError("safe_point: unknown boundary")
        with self._lock:
            if self._last_time is not None and now_s < self._last_time:
                raise ValidationError("now_s: must be monotonic")
            self._last_time = now_s
            self.target = desired
            key = (desired.mode, desired.rows, desired.accelerator_role)
            current_key = ((self.current.mode, self.current.rows, self.current.accelerator_role)
                           if self.current else None)
            if key == current_key:
                self._pending, self._count = None, 0
                self.current = desired
            elif safe_point is None:
                # Unsafe observations neither earn debounce credit nor preserve a
                # candidate that the observed load has already contradicted.
                if key != self._pending:
                    self._pending, self._count = None, 0
            else:
                self._count = self._count + 1 if key == self._pending else 1
                self._pending = key
                dwell = self._last_switch is None or now_s - self._last_switch >= self.policy.min_dwell_s
                if self.current is None or (self._count >= self.policy.stable_observations and dwell):
                    self.current, self._last_switch = desired, now_s
                    self._pending, self._count = None, 0
            return self.current


@dataclass(frozen=True)
class JointChoice:
    owner: str
    front_j_s: float | None
    prefill_j_s: float
    reason: str


def choose_joint(prompt_tokens: int, local: LocalCandidate, service: ServiceCandidate, *,
                 load: int, remaining_tokens: int, pair_tps: float, split_tps: float,
                 switch_s: float = 0.0, policy: CostPolicy = CostPolicy()) -> JointChoice:
    """Compare prefill TTFT + affected decoders' remaining completion seconds.

    Keeping FRONT means local prefill and split decode. Giving the resource to
    prefill means service prefill and pair decode for the decision horizon. The
    lost split gain is N * remaining_tokens * (1/pair_tps - 1/split_tps).
    Queue, restore, handoff and import pauses come from the existing J model.
    The caller supplies a bounded remaining-output horizon and switch/load cost.
    """
    _integer(load, "load", 1, 16)
    _integer(prompt_tokens, "prompt_tokens", 0, 1 << 24)
    _integer(remaining_tokens, "remaining_tokens", 0, 1 << 20)
    _finite(pair_tps, "pair_tps", 1e-9)
    _finite(split_tps, "split_tps", 1e-9)
    _finite(switch_s, "switch_s")
    off = offload_cost(prompt_tokens, service, local) + load * remaining_tokens / pair_tps + switch_s
    if load >= 2:
        return JointChoice("prefill", None, off, "concurrent load reserves accelerator for prefill")
    front = local_cost(prompt_tokens, local) + load * remaining_tokens / split_tps
    margin = max(policy.margin_s, policy.margin_frac * front)
    owner = "prefill" if front - off > margin else "front"
    return JointChoice(owner, front, off, "J including lost split decode gain and switching cost")


@dataclass(frozen=True)
class Reservation:
    generation: int
    owner: str
    bytes: int


class JointReservation:
    """One atomic owner of an entire shared accelerator allocation domain.

    Capacity is an explicit owner budget in bytes. No telemetry subtraction, TTL
    expiry or cancellation implies release. Even same-owner concurrent acquires
    fail; stale releases cannot free a newer generation. No distributed lease.
    """

    def __init__(self, capacity_bytes: int, *, enabled: bool = False):
        _integer(capacity_bytes, "capacity_bytes", 1, 1 << 60)
        if type(enabled) is not bool:
            raise ValidationError("enabled: expected boolean")
        self.capacity_bytes, self.enabled = capacity_bytes, enabled
        self._current: Reservation | None = None
        self._generation = 0
        self._users = 0
        self._lock = threading.Lock()

    @property
    def current(self) -> Reservation | None:
        with self._lock:
            return self._current

    def acquire(self, owner: str, bytes_required: int, *, safe_point: str | None) -> Reservation | None:
        if owner not in ("front", "prefill"):
            raise ValidationError("owner: expected front or prefill")
        _integer(bytes_required, "bytes_required", 1, 1 << 60)
        if safe_point not in SAFE_POINTS and safe_point is not None:
            raise ValidationError("safe_point: unknown boundary")
        with self._lock:
            if not self.enabled or safe_point is None or self._current or bytes_required > self.capacity_bytes:
                return None
            self._generation += 1
            self._current = Reservation(self._generation, owner, bytes_required)
            return self._current

    def release(self, lease: Reservation, *, buffers_released: bool, safe_point: str | None) -> bool:
        with self._lock:
            if (buffers_released is not True or safe_point not in SAFE_POINTS
                    or lease is not self._current or self._current is None or self._users):
                return False
            self._current = None
            return True

    @contextmanager
    def use(self, owner: str):
        """Pin a resident owner's lease during a task; release cannot race it.

        The supervisor must acquire before loading, and acknowledge buffer release
        after unloading. Finishing a task does not release resident model memory.
        """
        with self._lock:
            allowed = self.enabled and self._current is not None and self._current.owner == owner
            if allowed:
                self._users += 1
        try:
            yield allowed
        finally:
            if allowed:
                with self._lock:
                    self._users -= 1
