"""Target-host admission orchestration for one exact qualified workload.

This module performs no inference. It consumes:
- a target-host qualification handoff with exact E3 v2 evidence/readiness identities;
- an admission-ready native runtime-manifest collection;
- two distinct runtime snapshots.

It atomically reserves exact physical-pool demand, requires a fresh launch observation,
performs launch recheck, then constructs the existing AcceptedExecutionBundle. Any failure
after reservation releases the lease deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .adapter_contract import AdapterCapabilities
from .admission import LocalAdmissionController
from .config_v2 import Config
from .model_manifest import ModelManifest
from .native_runtime_collector import NativeRuntimeManifestCollection
from .planning_contract import PlanningInput
from .schema import ValidationError, text
from .target_host_qualification import (
    TargetHostQualificationHandoff,
    validate_target_host_handoff,
)
from .whole_block_execution import AcceptedExecutionBundle, accept_execution_bundle

ORCHESTRATOR_SCHEMA = "tensormeld/target-host-admission-v1"


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


@dataclass(frozen=True)
class TargetHostAdmissionResult:
    bundle: AcceptedExecutionBundle
    record: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return self.record["orchestrator_sha256"]


def _snapshot_id(snapshot: dict[str, Any], where: str) -> str:
    if not isinstance(snapshot, dict):
        raise ValidationError(f"{where}: expected object")
    if snapshot.get("admission_snapshot_schema") != "tensormeld/admission-snapshot-v1":
        raise ValidationError(
            f"{where}: expected tensormeld/admission-snapshot-v1"
        )
    return text(snapshot.get("observation_id"), f"{where}.observation_id")


def orchestrate_target_host_admission(
    *,
    controller: LocalAdmissionController,
    lease_id: str,
    config: Config,
    planning: PlanningInput,
    candidate: dict[str, Any],
    adapter: AdapterCapabilities,
    model: ModelManifest,
    handoff: TargetHostQualificationHandoff,
    collection: NativeRuntimeManifestCollection,
    reservation_snapshot: dict[str, Any],
    launch_snapshot: dict[str, Any],
) -> TargetHostAdmissionResult:
    lease_id = text(lease_id, "lease_id")
    hr = validate_target_host_handoff(handoff)
    manifest = collection.manifest
    cr = collection.record

    if cr.get("admission_ready_inputs") is not True:
        raise ValidationError(
            "runtime manifest collection is not admission-ready native input"
        )
    if manifest.provenance != "native-adapter":
        raise ValidationError(
            "target-host admission requires native-adapter runtime manifest"
        )
    if cr.get("runtime_manifest_sha256") != manifest.fingerprint:
        raise ValidationError("runtime collection manifest fingerprint mismatch")
    if cr.get("handoff_sha256") != handoff.fingerprint:
        raise ValidationError("runtime collection belongs to another handoff")
    if config.fingerprint != hr.get("config_sha256"):
        raise ValidationError("admission config identity mismatch")
    if planning.fingerprint != hr.get("planning_input_sha256"):
        raise ValidationError("admission planning identity mismatch")
    if candidate.get("plan_sha256") != hr.get("candidate_plan_sha256"):
        raise ValidationError("admission candidate plan identity mismatch")
    if model.manifest_sha256 != hr.get("model_manifest_sha256"):
        raise ValidationError("admission model identity mismatch")
    if adapter.adapter_id != hr.get("adapter_id"):
        raise ValidationError("admission adapter identity mismatch")
    if adapter.fingerprint != hr.get("adapter_capabilities_sha256"):
        raise ValidationError("admission adapter fingerprint mismatch")
    if adapter.engine_revision != hr.get("engine_revision"):
        raise ValidationError("admission engine revision mismatch")
    if manifest.worker_artifact_sha256 != hr.get("probe_artifact_sha256"):
        raise ValidationError("admission worker artifact mismatch")
    if manifest.model_manifest_sha256 != model.manifest_sha256:
        raise ValidationError("admission runtime manifest model mismatch")
    if manifest.config_sha256 != config.fingerprint:
        raise ValidationError("admission runtime manifest config mismatch")
    if manifest.operator_coverage_complete is not True:
        raise ValidationError("admission runtime manifest operator coverage incomplete")

    reservation_observation_id = _snapshot_id(
        reservation_snapshot, "reservation_snapshot"
    )
    launch_observation_id = _snapshot_id(
        launch_snapshot, "launch_snapshot"
    )
    if launch_observation_id == reservation_observation_id:
        raise ValidationError(
            "launch recheck requires a distinct runtime observation"
        )

    readiness = hr.get("backend_readiness")
    if not isinstance(readiness, list):
        raise ValidationError("qualification handoff has no backend readiness list")
    required_devices = tuple(candidate.get("compute_devices", ()))
    ready_by_device: dict[str, dict[str, Any]] = {}
    for item in readiness:
        if not isinstance(item, dict):
            raise ValidationError("handoff backend readiness item must be an object")
        device_id = text(item.get("device_id"), "backend_readiness.device_id")
        if device_id in ready_by_device:
            raise ValidationError("duplicate backend readiness device")
        if item.get("runtime_ready") is not True:
            raise ValidationError(
                f"handoff backend readiness is not ready for {device_id}"
            )
        ready_by_device[device_id] = {
            "evidence_schema": "tensormeld/backend-readiness-evidence-v2",
            "evidence_sha256": item.get("evidence_sha256"),
            "config_sha256": config.fingerprint,
            "tensormeld_device_id": device_id,
            "identity_applicable": True,
            "backend_readiness_recorded": True,
            "runtime_identity_sha256": item.get("runtime_identity_sha256"),
            "requires_live_runtime_recheck": False,
            "runtime_ready": True,
            "reservation_created": False,
            "qualified": False,
            "executable": False,
        }
    if tuple(sorted(ready_by_device)) != required_devices:
        raise ValidationError(
            "handoff backend readiness does not cover exact candidate devices"
        )

    reserved = False
    try:
        reservation = controller.reserve(
            lease_id=lease_id,
            config=config,
            manifest=manifest,
            snapshot=reservation_snapshot,
        )
        if reservation.get("status") != "RESERVED":
            raise ValidationError(
                f"target-host reservation rejected: {reservation.get('shortfalls')}"
            )
        reserved = True

        launch = controller.launch_recheck(
            lease_id=lease_id,
            config=config,
            manifest=manifest,
            snapshot=launch_snapshot,
        )
        if launch.get("status") != "LAUNCH_ADMITTED":
            controller.release(lease_id)
            reserved = False
            raise ValidationError(
                f"target-host launch recheck rejected: {launch.get('shortfalls')}"
            )

        bundle = accept_execution_bundle(
            config=config,
            planning=planning,
            candidate=candidate,
            adapter=adapter,
            model=model,
            qualification_evidence=handoff.qualification_evidence,
            runtime_manifest=manifest,
            backend_readiness_results=[
                ready_by_device[device] for device in required_devices
            ],
            launch_admissions=[launch],
        )

        core = {
            "orchestrator_schema": ORCHESTRATOR_SCHEMA,
            "handoff_sha256": handoff.fingerprint,
            "collector_sha256": collection.fingerprint,
            "runtime_manifest_sha256": manifest.fingerprint,
            "reservation_observation_id": reservation_observation_id,
            "launch_observation_id": launch_observation_id,
            "lease_id": lease_id,
            "lease_sha256": launch["lease_sha256"],
            "qualification_evidence_sha256": (
                handoff.qualification_evidence.evidence_sha256
            ),
            "accepted_execution_bundle_sha256": bundle.bundle_sha256,
            "reservation_created": True,
            "launch_authorized": True,
            "execution_authorized": True,
            "inference_started": False,
            "real_model_inference": False,
            "warnings": [
                "Admission authorization does not start inference.",
                "AcceptedExecutionBundle is bound to the admitted lease and exact evidence tuple.",
                "Caller remains responsible for deterministic release after execution/cancel/failure.",
            ],
        }
        core["orchestrator_sha256"] = _canonical_sha256(core)
        return TargetHostAdmissionResult(bundle, core)
    except BaseException:
        if reserved:
            try:
                controller.release(lease_id)
            except BaseException:
                pass
        raise
