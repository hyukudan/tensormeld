"""First bounded whole-block executable adapter reference path.

This module is intentionally a control/lifecycle implementation, not native model
inference. It combines existing gates into one immutable accepted execution bundle and
runs only through an injected whole-block backend. Portable tests use a deterministic
fixture backend.

An accepted bundle requires:
- exact planner candidate integrity,
- exact adapter representability,
- exact E3+ model qualification applicability,
- exact runtime model/operator/memory manifest identity,
- one current backend-ready proof per compute device,
- one launch-admitted host lease per compute node.

Only the accepted bundle is executable. Source planner/manifest/evidence artifacts remain
unchanged and keep their original non-executable semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import Lock
from typing import Any, Protocol

from .adapter_contract import AdapterCapabilities, validate_candidate_representability
from .config_v2 import Config
from .model_manifest import ModelManifest
from .planning_contract import PlanningInput
from .qualification import QualificationEvidence, evidence_applies
from .runtime_model_manifest import RuntimeModelManifest
from .schema import ValidationError, text

EXECUTION_BUNDLE_SCHEMA = "tensormeld/accepted-execution-bundle-v1"
MAX_INPUT_BYTES = 1024 * 1024


def _sha256_hex(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{where}: expected SHA-256 hex")
    normalized = value.lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValidationError(f"{where}: expected SHA-256 hex")
    return normalized


def _canonical_sha256(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("execution bundle contains non-canonical data") from exc
    return hashlib.sha256(raw).hexdigest()


def _candidate_hash(
    config: Config,
    planning: PlanningInput,
    profile_name: str,
    candidate: dict[str, Any],
) -> str:
    core = dict(candidate)
    supplied = core.pop("plan_sha256", None)
    identity = {
        "config": config.fingerprint,
        "planning": planning.fingerprint,
        "profile": profile_name,
        "plan": core,
    }
    expected = _canonical_sha256(identity)
    if supplied != expected:
        raise ValidationError("candidate plan_sha256 does not match candidate contents")
    return expected


@dataclass(frozen=True)
class AcceptedExecutionBundle:
    config_sha256: str
    profile: str
    planning_input_sha256: str
    plan_sha256: str
    representability_sha256: str
    adapter_id: str
    adapter_capabilities_sha256: str
    model_manifest_sha256: str
    qualification_evidence_sha256: str
    runtime_manifest_sha256: str
    worker_artifact_sha256: str
    backend_readiness: tuple[tuple[str, str, str], ...]
    launch_leases: tuple[tuple[str, str, str], ...]
    segments: tuple[tuple[str, int, int], ...]
    unit_ids: tuple[str, ...]
    compute_devices: tuple[str, ...]
    compute_nodes: tuple[str, ...]
    bundle_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "execution_bundle_schema": EXECUTION_BUNDLE_SCHEMA,
            "config_sha256": self.config_sha256,
            "profile": self.profile,
            "planning_input_sha256": self.planning_input_sha256,
            "plan_sha256": self.plan_sha256,
            "representability_sha256": self.representability_sha256,
            "adapter_id": self.adapter_id,
            "adapter_capabilities_sha256": self.adapter_capabilities_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "qualification_evidence_sha256": self.qualification_evidence_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "worker_artifact_sha256": self.worker_artifact_sha256,
            "backend_readiness": [
                {
                    "device_id": device_id,
                    "evidence_sha256": evidence_sha,
                    "runtime_identity_sha256": identity_sha,
                }
                for device_id, evidence_sha, identity_sha in self.backend_readiness
            ],
            "launch_leases": [
                {
                    "node_id": node_id,
                    "lease_id": lease_id,
                    "lease_sha256": lease_sha,
                }
                for node_id, lease_id, lease_sha in self.launch_leases
            ],
            "segments": [
                {
                    "device": device,
                    "first_unit": first,
                    "last_unit_exclusive": last,
                }
                for device, first, last in self.segments
            ],
            "unit_ids": list(self.unit_ids),
            "compute_devices": list(self.compute_devices),
            "compute_nodes": list(self.compute_nodes),
            "qualified": True,
            "execution_authorized": True,
            "real_model_inference": False,
            "bundle_sha256": self.bundle_sha256,
        }


def accept_execution_bundle(
    *,
    config: Config,
    planning: PlanningInput,
    candidate: dict[str, Any],
    adapter: AdapterCapabilities,
    model: ModelManifest,
    qualification_evidence: QualificationEvidence,
    runtime_manifest: RuntimeModelManifest,
    backend_readiness_results: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    launch_admissions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> AcceptedExecutionBundle:
    profile = config.profile_map.get(runtime_manifest.profile)
    if profile is None:
        raise ValidationError("runtime manifest profile is absent from config")

    plan_sha = _candidate_hash(
        config, planning, runtime_manifest.profile, candidate
    )
    representability = validate_candidate_representability(
        config, planning, candidate, adapter
    )
    if (
        representability["status"] != "REPRESENTABLE"
        or representability["exact"] is not True
    ):
        raise ValidationError("planner candidate is not exactly representable")

    candidate_devices = tuple(candidate["compute_devices"])
    candidate_nodes = tuple(candidate["compute_nodes"])
    manifest_devices = tuple(sorted(d.id for d in runtime_manifest.devices))
    if candidate_devices != manifest_devices:
        raise ValidationError(
            "runtime manifest devices must exactly match candidate compute devices"
        )
    if tuple(sorted({d.node for d in runtime_manifest.devices})) != candidate_nodes:
        raise ValidationError(
            "runtime manifest nodes must exactly match candidate compute nodes"
        )
    if runtime_manifest.operator_coverage_complete is not True:
        raise ValidationError("runtime manifest operator coverage is incomplete")
    if runtime_manifest.config_sha256 != config.fingerprint:
        raise ValidationError("runtime manifest config identity mismatch")
    if runtime_manifest.model_manifest_sha256 != model.manifest_sha256:
        raise ValidationError("runtime manifest model identity mismatch")
    if runtime_manifest.adapter_id != adapter.adapter_id:
        raise ValidationError("runtime manifest adapter identity mismatch")
    if runtime_manifest.adapter_capabilities_sha256 != adapter.fingerprint:
        raise ValidationError("runtime manifest adapter capabilities mismatch")
    if runtime_manifest.worker_artifact_sha256 != qualification_evidence.worker_artifact_sha256:
        raise ValidationError(
            "E3 worker artifact does not match runtime manifest worker artifact"
        )

    qualification = evidence_applies(
        qualification_evidence,
        adapter=adapter,
        model=model,
        config_sha256=config.fingerprint,
        device_ids=list(candidate_devices),
        context_tokens=runtime_manifest.workload["context_tokens"],
        max_output_tokens=runtime_manifest.workload["max_output_tokens"],
        concurrency=runtime_manifest.workload["concurrency"],
        minimum_level="E3",
    )
    if not qualification["applies"] or qualification["qualified"] is not True:
        raise ValidationError(
            f"E3 model qualification does not apply: {qualification['reasons']}"
        )

    readiness_by_device: dict[str, tuple[str, str]] = {}
    for result in backend_readiness_results:
        if not isinstance(result, dict):
            raise ValidationError("backend readiness result must be an object")
        device_id = text(result.get("tensormeld_device_id"), "tensormeld_device_id")
        if device_id in readiness_by_device:
            raise ValidationError("duplicate backend readiness result for device")
        if (
            result.get("identity_applicable") is not True
            or result.get("backend_readiness_recorded") is not True
            or result.get("runtime_ready") is not True
            or result.get("requires_live_runtime_recheck") is not False
            or result.get("reservation_created") is not False
            or result.get("qualified") is not False
            or result.get("executable") is not False
        ):
            raise ValidationError(
                f"backend readiness gate is not satisfied for {device_id}"
            )
        if result.get("config_sha256") != config.fingerprint:
            raise ValidationError(
                f"backend readiness config mismatch for {device_id}"
            )
        evidence_sha = _sha256_hex(
            result.get("evidence_sha256"), "evidence_sha256"
        )
        identity_sha = _sha256_hex(
            result.get("runtime_identity_sha256"), "runtime_identity_sha256"
        )
        readiness_by_device[device_id] = (evidence_sha, identity_sha)
    if tuple(sorted(readiness_by_device)) != candidate_devices:
        raise ValidationError(
            "backend readiness must cover exactly the candidate compute devices"
        )

    lease_by_node: dict[str, tuple[str, str]] = {}
    for admission in launch_admissions:
        if not isinstance(admission, dict):
            raise ValidationError("launch admission must be an object")
        node_id = text(admission.get("node_id"), "node_id")
        if node_id in lease_by_node:
            raise ValidationError("duplicate launch admission for node")
        if (
            admission.get("result_schema") != "tensormeld/local-launch-recheck-v1"
            or admission.get("status") != "LAUNCH_ADMITTED"
            or admission.get("reservation_created") is not True
            or admission.get("launch_authorized") is not True
            or admission.get("qualified") is not False
            or admission.get("executable") is not False
        ):
            raise ValidationError(f"launch admission gate is not satisfied for {node_id}")
        if admission.get("config_sha256") != config.fingerprint:
            raise ValidationError(f"launch admission config mismatch for {node_id}")
        if admission.get("runtime_manifest_sha256") != runtime_manifest.fingerprint:
            raise ValidationError(
                f"launch admission runtime manifest mismatch for {node_id}"
            )
        lease_id = text(admission.get("lease_id"), "lease_id")
        lease_sha = _sha256_hex(admission.get("lease_sha256"), "lease_sha256")
        lease_by_node[node_id] = (lease_id, lease_sha)
    if tuple(sorted(lease_by_node)) != candidate_nodes:
        raise ValidationError(
            "launch admissions must cover exactly the candidate compute nodes"
        )

    segments = tuple(
        (
            text(segment["device"], "segment.device"),
            int(segment["first_unit"]),
            int(segment["last_unit_exclusive"]),
        )
        for segment in candidate["segments"]
    )
    unit_ids = tuple(candidate["unit_ids"])
    core = {
        "execution_bundle_schema": EXECUTION_BUNDLE_SCHEMA,
        "config_sha256": config.fingerprint,
        "profile": runtime_manifest.profile,
        "planning_input_sha256": planning.fingerprint,
        "plan_sha256": plan_sha,
        "representability_sha256": representability["report_sha256"],
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "model_manifest_sha256": model.manifest_sha256,
        "qualification_evidence_sha256": qualification_evidence.evidence_sha256,
        "runtime_manifest_sha256": runtime_manifest.fingerprint,
        "worker_artifact_sha256": runtime_manifest.worker_artifact_sha256,
        "backend_readiness": [
            [device, *readiness_by_device[device]]
            for device in sorted(readiness_by_device)
        ],
        "launch_leases": [
            [node, *lease_by_node[node]] for node in sorted(lease_by_node)
        ],
        "segments": [list(segment) for segment in segments],
        "unit_ids": list(unit_ids),
        "compute_devices": list(candidate_devices),
        "compute_nodes": list(candidate_nodes),
        "qualified": True,
        "execution_authorized": True,
        "real_model_inference": False,
    }
    bundle_sha = _canonical_sha256(core)
    return AcceptedExecutionBundle(
        config.fingerprint,
        runtime_manifest.profile,
        planning.fingerprint,
        plan_sha,
        representability["report_sha256"],
        adapter.adapter_id,
        adapter.fingerprint,
        model.manifest_sha256,
        qualification_evidence.evidence_sha256,
        runtime_manifest.fingerprint,
        runtime_manifest.worker_artifact_sha256,
        tuple(
            (device, *readiness_by_device[device])
            for device in sorted(readiness_by_device)
        ),
        tuple(
            (node, *lease_by_node[node])
            for node in sorted(lease_by_node)
        ),
        segments,
        unit_ids,
        candidate_devices,
        candidate_nodes,
        bundle_sha,
    )


class WholeBlockBackend(Protocol):
    def execute_segment(
        self,
        *,
        device_id: str,
        unit_ids: tuple[str, ...],
        payload: bytes,
    ) -> bytes: ...


class ReferenceWholeBlockSession:
    def __init__(
        self,
        bundle: AcceptedExecutionBundle,
        backend: WholeBlockBackend,
    ) -> None:
        self.bundle = bundle
        self.backend = backend
        self._lock = Lock()
        self.state = "prepared"
        self.cancel_requested = False

    def cancel(self) -> None:
        with self._lock:
            if self.state == "released":
                raise ValidationError("released session cannot be cancelled")
            if self.state in {"completed", "cancelled"}:
                return
            self.cancel_requested = True
            if self.state == "prepared":
                self.state = "cancelled"

    def run(self, payload: bytes) -> dict[str, Any]:
        if not isinstance(payload, bytes) or len(payload) > MAX_INPUT_BYTES:
            raise ValidationError("reference execution payload must be bytes up to 1 MiB")
        with self._lock:
            if self.state != "prepared":
                raise ValidationError(
                    f"reference session cannot run from state {self.state}"
                )
            if self.cancel_requested:
                self.state = "cancelled"
                raise ValidationError("reference execution was cancelled")
            self.state = "running"

        current = payload
        executed: list[dict[str, Any]] = []
        try:
            for device, first, last in self.bundle.segments:
                with self._lock:
                    if self.cancel_requested:
                        self.state = "cancelled"
                        return {
                            "result_schema": "tensormeld/reference-whole-block-run-v1",
                            "status": "CANCELLED",
                            "bundle_sha256": self.bundle.bundle_sha256,
                            "segments_executed": executed,
                            "output": None,
                            "real_model_inference": False,
                        }
                units = self.bundle.unit_ids[first:last]
                current = self.backend.execute_segment(
                    device_id=device,
                    unit_ids=units,
                    payload=current,
                )
                if not isinstance(current, bytes) or len(current) > MAX_INPUT_BYTES:
                    raise ValidationError(
                        "reference backend returned invalid/oversized bytes"
                    )
                executed.append({
                    "device_id": device,
                    "unit_ids": list(units),
                })
        except BaseException:
            with self._lock:
                self.state = "failed"
            raise

        with self._lock:
            self.state = "completed"
        return {
            "result_schema": "tensormeld/reference-whole-block-run-v1",
            "status": "COMPLETED",
            "bundle_sha256": self.bundle.bundle_sha256,
            "segments_executed": executed,
            "output": current,
            "real_model_inference": False,
        }

    def release(self) -> dict[str, Any]:
        with self._lock:
            if self.state == "running":
                raise ValidationError("running session must be cancelled/completed before release")
            previous = self.state
            self.state = "released"
        return {
            "result_schema": "tensormeld/reference-whole-block-release-v1",
            "status": "RELEASED" if previous != "released" else "ALREADY_RELEASED",
            "bundle_sha256": self.bundle.bundle_sha256,
            "released": True,
        }
