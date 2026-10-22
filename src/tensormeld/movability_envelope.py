"""Auditable tensor eligibility envelopes derived from exact movability evidence.

Eligibility is not ownership and is not a runtime-memory measurement. The same movable
tensor may be eligible for multiple devices/pools, so per-device totals overlap. Pool
unions count each tensor once per physical pool to avoid multiplying shared-memory aliases.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .config_v2 import Config
from .predictive_memory import PredictiveMemoryProfile
from .runtime_model_manifest import RuntimeModelManifest
from .schema import ValidationError
from .tensor_movability import TensorMovabilityProfile


@dataclass(frozen=True)
class DeviceEligibilityEnvelope:
    device: str
    node: str
    pool: str
    hard_resident_eligible_bytes: int
    reclaimable_file_backed_eligible_bytes: int
    pinned_bytes: int
    tensor_count: int


@dataclass(frozen=True)
class PoolEligibilityEnvelope:
    pool: str
    node: str
    hard_resident_union_bytes: int
    reclaimable_file_backed_union_bytes: int
    pinned_bytes: int
    tensor_count: int


@dataclass(frozen=True)
class MovabilityEnvelope:
    tensor_movability_sha256: str
    predictive_memory_sha256: str
    runtime_manifest_sha256: str
    devices: tuple[DeviceEligibilityEnvelope, ...]
    pools: tuple[PoolEligibilityEnvelope, ...]
    fingerprint: str


def derive_movability_envelope(
    *,
    config: Config,
    runtime_manifest: RuntimeModelManifest,
    predictive_memory: PredictiveMemoryProfile,
    movability: TensorMovabilityProfile,
) -> MovabilityEnvelope:
    if predictive_memory.runtime_manifest_sha256 != runtime_manifest.fingerprint:
        raise ValidationError("predictive memory/runtime manifest identity mismatch")
    if movability.model_manifest_sha256 != runtime_manifest.model_manifest_sha256:
        raise ValidationError("movability/runtime model identity mismatch")
    if (
        movability.adapter_capabilities_sha256
        != runtime_manifest.adapter_capabilities_sha256
    ):
        raise ValidationError("movability/runtime adapter capability identity mismatch")

    config_devices = {device.id: device for device in config.devices}
    config_pools = {pool.id: pool for pool in config.pools}
    predictive_pools = {pool.pool: pool for pool in predictive_memory.pools}
    runtime_pools = {pool.pool: pool for pool in runtime_manifest.pools}

    if set(predictive_pools) != set(runtime_pools):
        raise ValidationError("predictive memory must cover exact runtime physical pools")

    device_rows: list[DeviceEligibilityEnvelope] = []
    for device_id in sorted(config_devices):
        device = config_devices[device_id]
        allowed = [
            tensor for tensor in movability.tensors
            if device_id in tensor.allowed_devices
        ]
        if not allowed:
            continue
        pinned = [
            tensor for tensor in allowed
            if tensor.movement_class == "pinned"
            and tensor.allowed_devices == (device_id,)
        ]
        device_rows.append(DeviceEligibilityEnvelope(
            device_id,
            device.node,
            device.pool,
            sum(
                tensor.n_bytes for tensor in allowed
                if tensor.storage_class == "hard_resident"
            ),
            sum(
                tensor.n_bytes for tensor in allowed
                if tensor.storage_class == "reclaimable_file_backed"
            ),
            sum(tensor.n_bytes for tensor in pinned),
            len(allowed),
        ))

    pool_rows: list[PoolEligibilityEnvelope] = []
    for pool_id in sorted(runtime_pools):
        pool = config_pools.get(pool_id)
        if pool is None:
            raise ValidationError(f"runtime pool {pool_id!r} missing from config")
        pool_devices = {
            device.id for device in config.devices if device.pool == pool_id
        }
        eligible = [
            tensor for tensor in movability.tensors
            if pool_devices & set(tensor.allowed_devices)
        ]
        pinned = [
            tensor for tensor in movability.tensors
            if tensor.movement_class == "pinned"
            and tensor.allowed_devices[0] in pool_devices
        ]
        pool_rows.append(PoolEligibilityEnvelope(
            pool_id,
            pool.node,
            sum(
                tensor.n_bytes for tensor in eligible
                if tensor.storage_class == "hard_resident"
            ),
            sum(
                tensor.n_bytes for tensor in eligible
                if tensor.storage_class == "reclaimable_file_backed"
            ),
            sum(tensor.n_bytes for tensor in pinned),
            len(eligible),
        ))

    canonical = {
        "envelope_schema": "tensormeld/movability-envelope-v1",
        "tensor_movability_sha256": movability.fingerprint,
        "predictive_memory_sha256": predictive_memory.fingerprint,
        "runtime_manifest_sha256": runtime_manifest.fingerprint,
        "devices": [
            {
                "device": row.device,
                "node": row.node,
                "pool": row.pool,
                "hard_resident_eligible_bytes": row.hard_resident_eligible_bytes,
                "reclaimable_file_backed_eligible_bytes":
                    row.reclaimable_file_backed_eligible_bytes,
                "pinned_bytes": row.pinned_bytes,
                "tensor_count": row.tensor_count,
            }
            for row in device_rows
        ],
        "pools": [
            {
                "pool": row.pool,
                "node": row.node,
                "hard_resident_union_bytes": row.hard_resident_union_bytes,
                "reclaimable_file_backed_union_bytes":
                    row.reclaimable_file_backed_union_bytes,
                "pinned_bytes": row.pinned_bytes,
                "tensor_count": row.tensor_count,
            }
            for row in pool_rows
        ],
        "qualified": False,
        "executable": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return MovabilityEnvelope(
        movability.fingerprint,
        predictive_memory.fingerprint,
        runtime_manifest.fingerprint,
        tuple(device_rows),
        tuple(pool_rows),
        fingerprint,
    )


def movability_envelope_summary(envelope: MovabilityEnvelope) -> dict[str, Any]:
    return {
        "result_schema": "tensormeld/movability-envelope-validation-v1",
        "movability_envelope_sha256": envelope.fingerprint,
        "tensor_movability_sha256": envelope.tensor_movability_sha256,
        "predictive_memory_sha256": envelope.predictive_memory_sha256,
        "runtime_manifest_sha256": envelope.runtime_manifest_sha256,
        "devices": [row.__dict__ for row in envelope.devices],
        "pools": [row.__dict__ for row in envelope.pools],
        "qualified": False,
        "executable": False,
        "warnings": [
            "Eligibility is not current ownership and is not a runtime-memory measurement.",
            "Per-device eligible-byte totals may overlap because one movable tensor can be legal on multiple devices.",
            "Per-pool unions count each tensor once per physical pool to avoid double counting multiple logical devices sharing a pool.",
            "No planner candidate is created or modified by this envelope.",
        ],
    }
