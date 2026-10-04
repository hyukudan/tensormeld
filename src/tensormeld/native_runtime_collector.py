"""Handoff-bound native runtime model manifest collector.

The collector accepts explicit operator/memory measurements and turns them into the
existing RuntimeModelManifest only after proving that they belong to the exact
TargetHostQualificationHandoff.

It performs no measurement, subprocess launch, reservation or admission action.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .adapter_contract import AdapterCapabilities
from .config_v2 import Config
from .model_manifest import ModelManifest, _sha256
from .runtime_model_manifest import (
    RuntimeModelManifest,
    parse_runtime_model_manifest,
    runtime_manifest_summary,
)
from .schema import ValidationError, items, number, record, text, unique
from .target_host_qualification import (
    TargetHostQualificationHandoff,
    validate_target_host_handoff,
)

MEASUREMENT_SCHEMA = "tensormeld/native-runtime-measurement-v1"
OPERATOR_REQUIREMENTS_SCHEMA = "tensormeld/runtime-operator-requirements-v1"
COLLECTOR_SCHEMA = "tensormeld/native-runtime-manifest-collector-v1"
MAX_MEASURED_DEVICES = 128
MAX_MEASURED_POOLS = 192
MAX_MEASURED_OPERATORS = 512


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _operators(value: Any, where: str) -> tuple[str, ...]:
    result = tuple(
        text(item, f"{where}[]")
        for item in items(value, where, MAX_MEASURED_OPERATORS, 1)
    )
    unique(list(result), where)
    return tuple(sorted(result))


@dataclass(frozen=True)
class RuntimeOperatorRequirements:
    source: str
    handoff_sha256: str
    model_manifest_sha256: str
    candidate_plan_sha256: str
    placement_sha256: str
    required_operators: tuple[str, ...]
    fingerprint: str

    @classmethod
    def parse(
        cls,
        data: Any,
        *,
        handoff: TargetHostQualificationHandoff,
    ) -> "RuntimeOperatorRequirements":
        r = record(data, "runtime operator requirements", {
            "operator_requirements_schema",
            "requirements_source",
            "handoff_sha256",
            "model_manifest_sha256",
            "candidate_plan_sha256",
            "placement_sha256",
            "required_operators",
            "qualified",
            "executable",
        })
        if r["operator_requirements_schema"] != OPERATOR_REQUIREMENTS_SCHEMA:
            raise ValidationError(
                f"operator_requirements_schema: expected {OPERATOR_REQUIREMENTS_SCHEMA}"
            )
        source = text(r["requirements_source"], "requirements_source")
        if source not in {"native-adapter", "fixture"}:
            raise ValidationError(
                "requirements_source: expected native-adapter or fixture"
            )
        if r["qualified"] is not False or r["executable"] is not False:
            raise ValidationError(
                "operator requirements cannot self-promote qualification/execution"
            )
        hr = validate_target_host_handoff(handoff)
        checks = {
            "handoff_sha256": handoff.fingerprint,
            "model_manifest_sha256": hr["model_manifest_sha256"],
            "candidate_plan_sha256": hr["candidate_plan_sha256"],
            "placement_sha256": hr["placement_sha256"],
        }
        for field, expected in checks.items():
            if r[field] != expected:
                raise ValidationError(
                    f"operator requirements {field} mismatch"
                )
        required = _operators(
            r["required_operators"], "required_operators"
        )
        canonical = {
            "operator_requirements_schema": OPERATOR_REQUIREMENTS_SCHEMA,
            "requirements_source": source,
            "handoff_sha256": handoff.fingerprint,
            "model_manifest_sha256": hr["model_manifest_sha256"],
            "candidate_plan_sha256": hr["candidate_plan_sha256"],
            "placement_sha256": hr["placement_sha256"],
            "required_operators": list(required),
            "qualified": False,
            "executable": False,
        }
        return cls(
            source,
            handoff.fingerprint,
            hr["model_manifest_sha256"],
            hr["candidate_plan_sha256"],
            hr["placement_sha256"],
            required,
            _canonical_sha256(canonical),
        )


@dataclass(frozen=True)
class NativeRuntimeMeasurement:
    source: str
    handoff_sha256: str
    config_sha256: str
    model_manifest_sha256: str
    adapter_capabilities_sha256: str
    engine_revision: str
    worker_artifact_sha256: str
    workload: dict[str, Any]
    devices: tuple[dict[str, Any], ...]
    physical_pool_memory: tuple[dict[str, Any], ...]
    fingerprint: str

    @classmethod
    def parse(
        cls,
        data: Any,
        *,
        handoff: TargetHostQualificationHandoff,
    ) -> "NativeRuntimeMeasurement":
        r = record(data, "native runtime measurement", {
            "measurement_schema",
            "measurement_source",
            "handoff_sha256",
            "config_sha256",
            "model_manifest_sha256",
            "adapter_capabilities_sha256",
            "engine_revision",
            "worker_artifact_sha256",
            "workload",
            "devices",
            "physical_pool_memory",
            "reservation_created",
            "qualified",
            "executable",
        })
        if r["measurement_schema"] != MEASUREMENT_SCHEMA:
            raise ValidationError(
                f"measurement_schema: expected {MEASUREMENT_SCHEMA}"
            )
        source = text(r["measurement_source"], "measurement_source")
        if source not in {"native-adapter", "fixture"}:
            raise ValidationError(
                "measurement_source: expected native-adapter or fixture"
            )
        if r["reservation_created"] is not False:
            raise ValidationError("runtime measurement cannot create a reservation")
        if r["qualified"] is not False or r["executable"] is not False:
            raise ValidationError(
                "runtime measurement cannot self-promote qualification/execution"
            )

        hr = validate_target_host_handoff(handoff)
        if r["handoff_sha256"] != handoff.fingerprint:
            raise ValidationError("measurement handoff fingerprint mismatch")
        for field in (
            "config_sha256",
            "model_manifest_sha256",
            "adapter_capabilities_sha256",
            "engine_revision",
            "worker_artifact_sha256",
        ):
            expected_field = (
                "probe_artifact_sha256"
                if field == "worker_artifact_sha256"
                else field
            )
            expected = hr.get(expected_field)
            if r[field] != expected:
                raise ValidationError(
                    f"measurement {field} does not match qualification handoff"
                )

        req = hr.get("runtime_manifest_requirements")
        if not isinstance(req, dict):
            raise ValidationError(
                "qualification handoff has no runtime manifest requirements"
            )
        workload_raw = record(
            r["workload"],
            "workload",
            {"task", "context_tokens", "max_output_tokens", "concurrency"},
        )
        workload = {
            "task": text(workload_raw["task"], "workload.task"),
            "context_tokens": int(number(
                workload_raw["context_tokens"],
                "workload.context_tokens",
                1,
                True,
            )),
            "max_output_tokens": int(number(
                workload_raw["max_output_tokens"],
                "workload.max_output_tokens",
                1,
                True,
            )),
            "concurrency": int(number(
                workload_raw["concurrency"],
                "workload.concurrency",
                1,
                True,
            )),
        }
        if workload != req.get("required_workload"):
            raise ValidationError(
                "runtime measurement workload does not match handoff target workload"
            )

        expected_devices = tuple(req.get("required_devices", ()))
        expected_identity_sha = tuple(hr.get("runtime_identity_sha256", ()))
        if len(expected_devices) != len(expected_identity_sha):
            raise ValidationError(
                "qualification handoff runtime identity/device cardinality mismatch"
            )
        expected_identity_by_device = dict(
            zip(expected_devices, expected_identity_sha)
        )

        devices: list[dict[str, Any]] = []
        for i, raw in enumerate(items(
            r["devices"],
            "devices",
            MAX_MEASURED_DEVICES,
            1,
        )):
            d = record(raw, f"devices[{i}]", {
                "id",
                "node",
                "backend",
                "runtime_identity_sha256",
                "operators",
            })
            device_id = text(d["id"], f"devices[{i}].id")
            runtime_sha = _sha256(
                d["runtime_identity_sha256"],
                f"devices[{i}].runtime_identity_sha256",
            )
            if expected_identity_by_device.get(device_id) != runtime_sha:
                raise ValidationError(
                    f"devices[{i}]: runtime identity does not match handoff"
                )
            devices.append({
                "id": device_id,
                "node": text(d["node"], f"devices[{i}].node"),
                "backend": text(d["backend"], f"devices[{i}].backend"),
                "runtime_identity_sha256": runtime_sha,
                "operators": list(_operators(
                    d["operators"], f"devices[{i}].operators"
                )),
            })
        unique([d["id"] for d in devices], "devices.id")
        if tuple(sorted(d["id"] for d in devices)) != tuple(expected_devices):
            raise ValidationError(
                "runtime measurement devices must exactly match handoff devices"
            )

        pools: list[dict[str, Any]] = []
        for i, raw in enumerate(items(
            r["physical_pool_memory"],
            "physical_pool_memory",
            MAX_MEASURED_POOLS,
            1,
        )):
            p = record(raw, f"physical_pool_memory[{i}]", {
                "pool_ref",
                "node",
                "resident_bytes",
                "state_bytes",
                "workspace_peak_bytes",
                "preparation_peak_bytes",
            })
            pools.append({
                "pool_ref": text(
                    p["pool_ref"],
                    f"physical_pool_memory[{i}].pool_ref",
                ),
                "node": text(p["node"], f"physical_pool_memory[{i}].node"),
                "resident_bytes": int(number(
                    p["resident_bytes"],
                    f"physical_pool_memory[{i}].resident_bytes",
                    0,
                    True,
                )),
                "state_bytes": int(number(
                    p["state_bytes"],
                    f"physical_pool_memory[{i}].state_bytes",
                    0,
                    True,
                )),
                "workspace_peak_bytes": int(number(
                    p["workspace_peak_bytes"],
                    f"physical_pool_memory[{i}].workspace_peak_bytes",
                    0,
                    True,
                )),
                "preparation_peak_bytes": int(number(
                    p["preparation_peak_bytes"],
                    f"physical_pool_memory[{i}].preparation_peak_bytes",
                    0,
                    True,
                )),
            })
        unique(
            [p["pool_ref"] for p in pools],
            "physical_pool_memory.pool_ref",
        )

        canonical = {
            "measurement_schema": MEASUREMENT_SCHEMA,
            "measurement_source": source,
            "handoff_sha256": handoff.fingerprint,
            "config_sha256": r["config_sha256"],
            "model_manifest_sha256": r["model_manifest_sha256"],
            "adapter_capabilities_sha256": r["adapter_capabilities_sha256"],
            "engine_revision": r["engine_revision"],
            "worker_artifact_sha256": r["worker_artifact_sha256"],
            "workload": workload,
            "devices": sorted(devices, key=lambda d: d["id"]),
            "physical_pool_memory": sorted(
                pools, key=lambda p: p["pool_ref"]
            ),
            "reservation_created": False,
            "qualified": False,
            "executable": False,
        }
        return cls(
            source,
            handoff.fingerprint,
            r["config_sha256"],
            r["model_manifest_sha256"],
            r["adapter_capabilities_sha256"],
            r["engine_revision"],
            r["worker_artifact_sha256"],
            workload,
            tuple(canonical["devices"]),
            tuple(canonical["physical_pool_memory"]),
            _canonical_sha256(canonical),
        )


@dataclass(frozen=True)
class NativeRuntimeManifestCollection:
    manifest: RuntimeModelManifest
    record: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return self.record["collector_sha256"]


def collect_native_runtime_manifest(
    *,
    config: Config,
    model: ModelManifest,
    adapter: AdapterCapabilities,
    handoff: TargetHostQualificationHandoff,
    measurement: NativeRuntimeMeasurement,
    operator_requirements: RuntimeOperatorRequirements,
) -> NativeRuntimeManifestCollection:
    hr = validate_target_host_handoff(handoff)
    req = hr.get("runtime_manifest_requirements")
    if not isinstance(req, dict):
        raise ValidationError("handoff has no runtime manifest requirements")
    if measurement.handoff_sha256 != handoff.fingerprint:
        raise ValidationError("measurement belongs to another handoff")
    if operator_requirements.handoff_sha256 != handoff.fingerprint:
        raise ValidationError("operator requirements belong to another handoff")
    if operator_requirements.model_manifest_sha256 != model.manifest_sha256:
        raise ValidationError("operator requirements model identity mismatch")
    if operator_requirements.candidate_plan_sha256 != hr.get("candidate_plan_sha256"):
        raise ValidationError("operator requirements plan identity mismatch")
    if operator_requirements.placement_sha256 != hr.get("placement_sha256"):
        raise ValidationError("operator requirements placement identity mismatch")
    if measurement.source not in {"native-adapter", "fixture"}:
        raise ValidationError("unsupported runtime measurement source")

    if config.fingerprint != hr.get("config_sha256"):
        raise ValidationError("collector config identity mismatch")
    if model.manifest_sha256 != hr.get("model_manifest_sha256"):
        raise ValidationError("collector model identity mismatch")
    if adapter.adapter_id != hr.get("adapter_id"):
        raise ValidationError("collector adapter identity mismatch")
    if adapter.fingerprint != hr.get("adapter_capabilities_sha256"):
        raise ValidationError("collector adapter fingerprint mismatch")
    if adapter.engine_revision != hr.get("engine_revision"):
        raise ValidationError("collector engine revision mismatch")
    if measurement.worker_artifact_sha256 != hr.get("probe_artifact_sha256"):
        raise ValidationError("collector worker artifact mismatch")

    cfg_devices = {d.id: d for d in config.devices}
    expected_devices = tuple(req.get("required_devices", ()))
    expected_nodes = set(req.get("required_nodes", ()))
    for d in measurement.devices:
        cfg = cfg_devices.get(d["id"])
        if cfg is None:
            raise ValidationError(f"measurement references unknown device {d['id']}")
        if d["node"] != cfg.node or d["backend"] != cfg.backend:
            raise ValidationError(
                f"measurement device identity mismatch for {d['id']}"
            )
        if cfg.node not in expected_nodes:
            raise ValidationError(
                f"measurement device {d['id']} belongs to an unused node"
            )

    pools_by_id = {p.id: p for p in config.pools}
    measured_pool_ids = {p["pool_ref"] for p in measurement.physical_pool_memory}
    required_device_pools = {cfg_devices[d].pool for d in expected_devices}
    if not required_device_pools <= measured_pool_ids:
        missing = sorted(required_device_pools - measured_pool_ids)
        raise ValidationError(
            f"runtime measurement omits compute-device physical pools: {missing}"
        )
    for p in measurement.physical_pool_memory:
        cfg_pool = pools_by_id.get(p["pool_ref"])
        if cfg_pool is None:
            raise ValidationError(
                f"measurement references unknown pool {p['pool_ref']}"
            )
        if p["node"] != cfg_pool.node:
            raise ValidationError(
                f"measurement pool/node mismatch for {p['pool_ref']}"
            )
        if cfg_pool.node not in expected_nodes:
            raise ValidationError(
                f"measurement pool {p['pool_ref']} belongs to an unused node"
            )

    raw_manifest = {
        "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
        "provenance": measurement.source,
        "config_sha256": config.fingerprint,
        "profile": req["profile"],
        "model_manifest_sha256": model.manifest_sha256,
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": measurement.worker_artifact_sha256,
        "workload": dict(measurement.workload),
        "required_operators": list(operator_requirements.required_operators),
        "devices": [
            {
                "id": d["id"],
                "node": d["node"],
                "backend": d["backend"],
                "operators": list(d["operators"]),
            }
            for d in measurement.devices
        ],
        "physical_pool_memory": [
            dict(p) for p in measurement.physical_pool_memory
        ],
        "reservation_created": False,
        "qualified": False,
        "executable": False,
    }
    manifest = parse_runtime_model_manifest(
        raw_manifest,
        config=config,
        model=model,
        adapter=adapter,
        profile_name=req["profile"],
    )

    summary = runtime_manifest_summary(manifest)
    core = {
        "collector_schema": COLLECTOR_SCHEMA,
        "handoff_sha256": handoff.fingerprint,
        "measurement_sha256": measurement.fingerprint,
        "operator_requirements_sha256": operator_requirements.fingerprint,
        "runtime_manifest_sha256": manifest.fingerprint,
        "measurement_source": measurement.source,
        "operator_coverage_complete": manifest.operator_coverage_complete,
        "e3_workload_matches_profile": req.get("e3_workload_matches_profile") is True,
        "admission_ready_inputs": (
            measurement.source == "native-adapter"
            and operator_requirements.source == "native-adapter"
            and manifest.operator_coverage_complete
            and req.get("e3_workload_matches_profile") is True
        ),
        "runtime_manifest_summary": summary,
        "reservation_created": False,
        "launch_authorized": False,
        "executable": False,
        "warnings": [
            "Collector validates supplied measurements; it does not perform measurements.",
            "Fixture measurement provenance never qualifies as native admission-ready input.",
            "Admission still requires a fresh runtime observation and an atomic lease.",
        ],
    }
    core["collector_sha256"] = _canonical_sha256(core)
    return NativeRuntimeManifestCollection(manifest, core)
