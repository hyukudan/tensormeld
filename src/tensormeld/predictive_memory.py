"""Predictive physical-pool memory classes refining RuntimeModelManifest v1.

This contract does not change admission semantics. It partitions the already-declared
runtime-manifest memory into classes so future planners can distinguish hard committed
memory from reclaimable file-backed pressure without treating reclaimable bytes as free.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .config_v2 import Config
from .runtime_model_manifest import RuntimeModelManifest
from .schema import ValidationError, items, number, record, text, unique

PREDICTIVE_MEMORY_SCHEMA = "tensormeld/predictive-memory-v1"
MAX_POOLS = 192
PROVENANCE_VALUES = {"fixture", "native-adapter"}


@dataclass(frozen=True)
class PredictivePoolMemory:
    pool: str
    node: str
    hard_resident_bytes: int
    reclaimable_file_backed_bytes: int
    persistent_state_bytes: int
    workspace_peak_bytes: int
    staging_peak_bytes: int

    @property
    def hard_peak_bytes(self) -> int:
        return (
            self.hard_resident_bytes
            + self.persistent_state_bytes
            + self.workspace_peak_bytes
            + self.staging_peak_bytes
        )

    @property
    def worst_case_physical_bytes(self) -> int:
        return self.hard_peak_bytes + self.reclaimable_file_backed_bytes


@dataclass(frozen=True)
class PredictiveMemoryProfile:
    provenance: str
    runtime_manifest_sha256: str
    pools: tuple[PredictivePoolMemory, ...]
    fingerprint: str


def parse_predictive_memory_profile(
    data: Any,
    *,
    config: Config,
    runtime_manifest: RuntimeModelManifest,
) -> PredictiveMemoryProfile:
    root = record(
        data,
        "predictive memory profile",
        {
            "predictive_memory_schema",
            "provenance",
            "runtime_manifest_sha256",
            "physical_pool_memory",
            "qualified",
            "executable",
        },
    )
    if root["predictive_memory_schema"] != PREDICTIVE_MEMORY_SCHEMA:
        raise ValidationError(
            f"predictive_memory_schema: expected {PREDICTIVE_MEMORY_SCHEMA}"
        )
    provenance = text(root["provenance"], "provenance")
    if provenance not in PROVENANCE_VALUES:
        raise ValidationError("provenance: expected fixture or native-adapter")
    if root["runtime_manifest_sha256"] != runtime_manifest.fingerprint:
        raise ValidationError("predictive memory runtime manifest identity mismatch")
    if root["qualified"] is not False or root["executable"] is not False:
        raise ValidationError("predictive memory profile cannot self-promote")

    manifest_pools = {pool.pool: pool for pool in runtime_manifest.pools}
    config_pools = {pool.id: pool for pool in config.pools}
    parsed: list[PredictivePoolMemory] = []

    for i, raw in enumerate(
        items(
            root["physical_pool_memory"],
            "physical_pool_memory",
            MAX_POOLS,
            1,
        )
    ):
        p = record(
            raw,
            f"physical_pool_memory[{i}]",
            {
                "pool_ref",
                "node",
                "hard_resident_bytes",
                "reclaimable_file_backed_bytes",
                "persistent_state_bytes",
                "workspace_peak_bytes",
                "staging_peak_bytes",
            },
        )
        pool_id = text(p["pool_ref"], f"physical_pool_memory[{i}].pool_ref")
        manifest_pool = manifest_pools.get(pool_id)
        config_pool = config_pools.get(pool_id)
        if manifest_pool is None or config_pool is None:
            raise ValidationError(
                f"physical_pool_memory[{i}]: unknown runtime-manifest pool {pool_id}"
            )
        node = text(p["node"], f"physical_pool_memory[{i}].node")
        if node != manifest_pool.node or node != config_pool.node:
            raise ValidationError(
                f"physical_pool_memory[{i}]: pool/node identity mismatch"
            )

        hard = int(number(
            p["hard_resident_bytes"],
            f"physical_pool_memory[{i}].hard_resident_bytes",
            0,
            True,
        ))
        reclaimable = int(number(
            p["reclaimable_file_backed_bytes"],
            f"physical_pool_memory[{i}].reclaimable_file_backed_bytes",
            0,
            True,
        ))
        state = int(number(
            p["persistent_state_bytes"],
            f"physical_pool_memory[{i}].persistent_state_bytes",
            0,
            True,
        ))
        workspace = int(number(
            p["workspace_peak_bytes"],
            f"physical_pool_memory[{i}].workspace_peak_bytes",
            0,
            True,
        ))
        staging = int(number(
            p["staging_peak_bytes"],
            f"physical_pool_memory[{i}].staging_peak_bytes",
            0,
            True,
        ))

        if hard + reclaimable != manifest_pool.resident_bytes:
            raise ValidationError(
                f"physical_pool_memory[{i}]: hard + reclaimable must exactly "
                "partition runtime resident_bytes"
            )
        if state != manifest_pool.state_bytes:
            raise ValidationError(
                f"physical_pool_memory[{i}]: persistent state must equal runtime state_bytes"
            )
        if workspace != manifest_pool.workspace_peak_bytes:
            raise ValidationError(
                f"physical_pool_memory[{i}]: workspace must equal runtime workspace_peak_bytes"
            )
        expected_staging = (
            manifest_pool.preparation_peak_bytes - manifest_pool.steady_peak_bytes
        )
        if staging != expected_staging:
            raise ValidationError(
                f"physical_pool_memory[{i}]: staging must exactly explain runtime "
                "preparation peak above steady peak"
            )

        classified = PredictivePoolMemory(
            pool_id,
            node,
            hard,
            reclaimable,
            state,
            workspace,
            staging,
        )
        if classified.worst_case_physical_bytes != manifest_pool.preparation_peak_bytes:
            raise ValidationError(
                f"physical_pool_memory[{i}]: classification does not reconcile "
                "runtime preparation peak"
            )
        if (
            config_pool.reported_capacity_bytes is not None
            and classified.worst_case_physical_bytes
            > config_pool.reported_capacity_bytes
        ):
            raise ValidationError(
                f"physical_pool_memory[{i}]: worst-case physical bytes exceed capacity"
            )
        parsed.append(classified)

    unique([p.pool for p in parsed], "physical_pool_memory.pool_ref")
    if set(p.pool for p in parsed) != set(manifest_pools):
        raise ValidationError(
            "predictive memory profile must classify every runtime-manifest physical pool"
        )

    canonical = {
        "predictive_memory_schema": PREDICTIVE_MEMORY_SCHEMA,
        "provenance": provenance,
        "runtime_manifest_sha256": runtime_manifest.fingerprint,
        "physical_pool_memory": [
            {
                "pool_ref": p.pool,
                "node": p.node,
                "hard_resident_bytes": p.hard_resident_bytes,
                "reclaimable_file_backed_bytes": p.reclaimable_file_backed_bytes,
                "persistent_state_bytes": p.persistent_state_bytes,
                "workspace_peak_bytes": p.workspace_peak_bytes,
                "staging_peak_bytes": p.staging_peak_bytes,
                "hard_peak_bytes": p.hard_peak_bytes,
                "worst_case_physical_bytes": p.worst_case_physical_bytes,
            }
            for p in sorted(parsed, key=lambda item: item.pool)
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
    return PredictiveMemoryProfile(
        provenance,
        runtime_manifest.fingerprint,
        tuple(sorted(parsed, key=lambda item: item.pool)),
        fingerprint,
    )


def predictive_memory_summary(
    profile: PredictiveMemoryProfile,
) -> dict[str, Any]:
    return {
        "result_schema": "tensormeld/predictive-memory-validation-v1",
        "predictive_memory_sha256": profile.fingerprint,
        "runtime_manifest_sha256": profile.runtime_manifest_sha256,
        "provenance": profile.provenance,
        "physical_pool_memory": [
            {
                "pool_ref": p.pool,
                "node": p.node,
                "hard_resident_bytes": p.hard_resident_bytes,
                "reclaimable_file_backed_bytes": p.reclaimable_file_backed_bytes,
                "persistent_state_bytes": p.persistent_state_bytes,
                "workspace_peak_bytes": p.workspace_peak_bytes,
                "staging_peak_bytes": p.staging_peak_bytes,
                "hard_peak_bytes": p.hard_peak_bytes,
                "worst_case_physical_bytes": p.worst_case_physical_bytes,
            }
            for p in profile.pools
        ],
        "qualified": False,
        "executable": False,
        "warnings": [
            "This profile refines an existing runtime manifest and does not change current admission peaks.",
            "Reclaimable/file-backed bytes are not free: worst-case physical bytes still include them.",
            "Hard peak is a future planning lower bound, not permission to overcommit page cache.",
            "Fixture provenance is contract evidence only, not a native memory measurement.",
        ],
    }
