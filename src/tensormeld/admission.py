"""Local atomic physical-pool reservation and launch admission.

This module provides an in-process control-plane lease primitive. It is intentionally
not a distributed lock and does not make a workload executable. It consumes exact
runtime-manifest preparation peaks and intersects them with fresh runtime observations
and owner policy.

Observations can explicitly name active lease IDs whose allocations are already
reflected in reported available bytes. Active leases not named as reflected are
subtracted conservatively. This avoids mandatory double subtraction once a worker
allocation is visible in telemetry while preserving safety for pending reservations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import Lock
from typing import Any

from .config_v2 import Config
from .runtime_model_manifest import RuntimeModelManifest
from .schema import ValidationError, record, text, items
from .selection import resolve_runtime_candidates

ADMISSION_SNAPSHOT_SCHEMA = "tensormeld/admission-snapshot-v1"
LEASE_SCHEMA = "tensormeld/local-pool-lease-v1"
MAX_REFLECTED_LEASES = 1024


@dataclass(frozen=True)
class LocalPoolLease:
    lease_id: str
    config_sha256: str
    runtime_manifest_sha256: str
    reservation_observation_id: str
    pool_bytes: tuple[tuple[str, int], ...]
    state: str
    lease_sha256: str


def _snapshot(
    config: Config,
    raw: dict[str, Any],
    *,
    profile_name: str,
) -> tuple[str, dict[str, Any], frozenset[str]]:
    r = record(
        raw,
        "admission snapshot",
        {
            "admission_snapshot_schema",
            "observation_id",
            "runtime_observation",
            "reflected_lease_ids",
        },
    )
    if r["admission_snapshot_schema"] != ADMISSION_SNAPSHOT_SCHEMA:
        raise ValidationError(
            f"admission_snapshot_schema: expected {ADMISSION_SNAPSHOT_SCHEMA}"
        )
    observation_id = text(r["observation_id"], "observation_id")
    reflected = frozenset(
        text(x, "reflected_lease_ids[]")
        for x in items(
            r["reflected_lease_ids"],
            "reflected_lease_ids",
            MAX_REFLECTED_LEASES,
            0,
        )
    )
    runtime = resolve_runtime_candidates(
        config,
        r["runtime_observation"],
        profile_name,
    )
    return observation_id, runtime, reflected


class LocalAdmissionController:
    """Serialize local reservations over physical pools within one process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._leases: dict[str, LocalPoolLease] = {}

    def active_leases(self) -> tuple[LocalPoolLease, ...]:
        with self._lock:
            return tuple(
                self._leases[k]
                for k in sorted(self._leases)
                if self._leases[k].state in {"reserved", "launched"}
            )

    def _committed_not_reflected(
        self,
        reflected: frozenset[str],
        *,
        exclude: str | None = None,
    ) -> dict[str, int]:
        totals: dict[str, int] = {}
        for lease_id, lease in self._leases.items():
            if lease_id == exclude:
                continue
            if lease.state not in {"reserved", "launched"}:
                continue
            if lease_id in reflected:
                continue
            for pool, amount in lease.pool_bytes:
                totals[pool] = totals.get(pool, 0) + amount
        return totals

    def _validate_manifest_runtime(
        self,
        config: Config,
        manifest: RuntimeModelManifest,
        runtime: dict[str, Any],
        owned_node_id: str | None = None,
    ) -> dict[str, int]:
        if manifest.config_sha256 != config.fingerprint:
            raise ValidationError("runtime manifest config identity mismatch")
        if runtime["status"] != "RUNTIME_CANDIDATES_READY":
            raise ValidationError("runtime requirements are not currently ready")
        if manifest.operator_coverage_complete is not True:
            raise ValidationError("runtime manifest operator coverage is incomplete")

        eligible_ids = {
            item["id"] for item in runtime["runtime_eligible_compute_devices"]
        }
        manifest_ids = {
            device.id for device in manifest.devices
            if owned_node_id is None or device.node == owned_node_id
        }
        if owned_node_id is not None and not manifest_ids:
            raise ValidationError("runtime manifest has no devices owned by this node")
        missing = sorted(manifest_ids - eligible_ids)
        if missing:
            raise ValidationError(
                f"runtime manifest devices are not currently ready: {missing}"
            )

        budgets = runtime["runtime_pool_budgets"]
        demand: dict[str, int] = {}
        for pool in manifest.pools:
            if owned_node_id is not None and pool.node != owned_node_id:
                continue
            if pool.pool not in budgets:
                raise ValidationError(
                    f"runtime manifest pool {pool.pool} has no live observation"
                )
            demand[pool.pool] = pool.preparation_peak_bytes
        if not demand:
            raise ValidationError("runtime manifest has no physical pools owned by this node")
        return demand

    def reserve(
        self,
        *,
        lease_id: str,
        config: Config,
        manifest: RuntimeModelManifest,
        snapshot: dict[str, Any],
        owned_node_id: str | None = None,
    ) -> dict[str, Any]:
        lease_id = text(lease_id, "lease_id")
        observation_id, runtime, reflected = _snapshot(
            config,
            snapshot,
            profile_name=manifest.profile,
        )
        demand = self._validate_manifest_runtime(
            config, manifest, runtime, owned_node_id
        )

        with self._lock:
            if lease_id in self._leases:
                raise ValidationError(f"lease_id {lease_id!r} already exists")
            unknown_reflected = sorted(reflected - set(self._leases))
            if unknown_reflected:
                raise ValidationError(
                    f"snapshot reflects unknown lease IDs: {unknown_reflected}"
                )
            committed = self._committed_not_reflected(reflected)
            shortfalls = []
            for pool, required in sorted(demand.items()):
                budget = runtime["runtime_pool_budgets"][pool]
                effective = max(0, budget - committed.get(pool, 0))
                if required > effective:
                    shortfalls.append({
                        "pool_ref": pool,
                        "required_bytes": required,
                        "effective_available_bytes": effective,
                    })
            if shortfalls:
                return {
                    "result_schema": "tensormeld/local-admission-result-v1",
                    "status": "REJECTED",
                    "lease_id": lease_id,
                    "node_id": owned_node_id,
                    "runtime_manifest_sha256": manifest.fingerprint,
                    "observation_id": observation_id,
                    "shortfalls": shortfalls,
                    "reservation_created": False,
                    "launch_authorized": False,
                    "qualified": False,
                    "executable": False,
                }

            core = {
                "lease_schema": LEASE_SCHEMA,
                "lease_id": lease_id,
                "config_sha256": config.fingerprint,
                "runtime_manifest_sha256": manifest.fingerprint,
                "reservation_observation_id": observation_id,
                "pool_bytes": [[pool, amount] for pool, amount in sorted(demand.items())],
                "state": "reserved",
            }
            digest = hashlib.sha256(
                json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            lease = LocalPoolLease(
                lease_id,
                config.fingerprint,
                manifest.fingerprint,
                observation_id,
                tuple(sorted(demand.items())),
                "reserved",
                digest,
            )
            self._leases[lease_id] = lease
            return {
                "result_schema": "tensormeld/local-admission-result-v1",
                "status": "RESERVED",
                "lease_id": lease_id,
                "node_id": owned_node_id,
                "lease_sha256": digest,
                "runtime_manifest_sha256": manifest.fingerprint,
                "observation_id": observation_id,
                "pool_bytes": dict(lease.pool_bytes),
                "reservation_created": True,
                "launch_authorized": False,
                "qualified": False,
                "executable": False,
            }

    def launch_recheck(
        self,
        *,
        lease_id: str,
        config: Config,
        manifest: RuntimeModelManifest,
        snapshot: dict[str, Any],
        owned_node_id: str | None = None,
    ) -> dict[str, Any]:
        lease_id = text(lease_id, "lease_id")
        observation_id, runtime, reflected = _snapshot(
            config,
            snapshot,
            profile_name=manifest.profile,
        )
        demand = self._validate_manifest_runtime(
            config, manifest, runtime, owned_node_id
        )

        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease.state != "reserved":
                raise ValidationError("launch recheck requires an active reserved lease")
            if lease.config_sha256 != config.fingerprint:
                raise ValidationError("lease config identity mismatch")
            if lease.runtime_manifest_sha256 != manifest.fingerprint:
                raise ValidationError("lease runtime manifest identity mismatch")
            if observation_id == lease.reservation_observation_id:
                raise ValidationError(
                    "launch recheck requires a newly identified runtime observation"
                )
            if dict(lease.pool_bytes) != demand:
                raise ValidationError("lease memory demand no longer matches manifest")
            unknown_reflected = sorted(reflected - set(self._leases))
            if unknown_reflected:
                raise ValidationError(
                    f"snapshot reflects unknown lease IDs: {unknown_reflected}"
                )

            committed = self._committed_not_reflected(
                reflected,
                exclude=lease_id,
            )
            shortfalls = []
            for pool, required in lease.pool_bytes:
                budget = runtime["runtime_pool_budgets"][pool]
                own_reflected = lease_id in reflected
                own_charge = 0 if own_reflected else required
                effective = max(0, budget - committed.get(pool, 0))
                if own_charge > effective:
                    shortfalls.append({
                        "pool_ref": pool,
                        "required_bytes": own_charge,
                        "effective_available_bytes": effective,
                    })
            if shortfalls:
                return {
                    "result_schema": "tensormeld/local-launch-recheck-v1",
                    "status": "REJECTED",
                    "lease_id": lease_id,
                    "node_id": owned_node_id,
                    "config_sha256": config.fingerprint,
                    "runtime_manifest_sha256": manifest.fingerprint,
                    "observation_id": observation_id,
                    "shortfalls": shortfalls,
                    "reservation_created": True,
                    "launch_authorized": False,
                    "qualified": False,
                    "executable": False,
                }

            launched = LocalPoolLease(
                lease.lease_id,
                lease.config_sha256,
                lease.runtime_manifest_sha256,
                lease.reservation_observation_id,
                lease.pool_bytes,
                "launched",
                lease.lease_sha256,
            )
            self._leases[lease_id] = launched
            return {
                "result_schema": "tensormeld/local-launch-recheck-v1",
                "status": "LAUNCH_ADMITTED",
                "lease_id": lease_id,
                "node_id": owned_node_id,
                "config_sha256": config.fingerprint,
                "runtime_manifest_sha256": manifest.fingerprint,
                "lease_sha256": lease.lease_sha256,
                "observation_id": observation_id,
                "reservation_created": True,
                "launch_authorized": True,
                "qualified": False,
                "executable": False,
                "warnings": [
                    "Launch admission covers local physical-pool capacity only.",
                    "Model correctness and backend qualification remain separate gates.",
                    "This in-process lease is not a distributed lock or remote reservation.",
                ],
            }

    def release(self, lease_id: str) -> dict[str, Any]:
        lease_id = text(lease_id, "lease_id")
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise ValidationError(f"unknown lease_id {lease_id!r}")
            if lease.state == "released":
                return {
                    "result_schema": "tensormeld/local-release-v1",
                    "status": "ALREADY_RELEASED",
                    "lease_id": lease_id,
                    "released": True,
                }
            self._leases[lease_id] = LocalPoolLease(
                lease.lease_id,
                lease.config_sha256,
                lease.runtime_manifest_sha256,
                lease.reservation_observation_id,
                lease.pool_bytes,
                "released",
                lease.lease_sha256,
            )
            return {
                "result_schema": "tensormeld/local-release-v1",
                "status": "RELEASED",
                "lease_id": lease_id,
                "released": True,
            }
