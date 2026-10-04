"""Fail-closed target-host qualification chain assembler.

This module performs no subprocess launch and no hardware discovery. It only validates
already-produced artifacts from the existing TensorMeld gates and proves that they all
belong to one exact config/model/plan/runtime tuple.

A successful handoff means:
- probe/binding identity is current;
- retained native E2 applies to every compute device and current runtime identity;
- pre-E3 placement is exactly reproducible from the current candidate/GGUF/binding;
- the retained native trial passes the approved E3 correctness reference;
- emitted E3 v2 applies to the same plan/runtime identities.

It does NOT create a runtime model manifest, reserve memory, authorize launch or make an
execution bundle. Native operator/memory measurement and admission remain mandatory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .adapter_contract import AdapterCapabilities
from .backend_evidence import validate_llamacpp_backend_evidence
from .config_v2 import Config
from .device_binding import LlamaCppBinding, bind_llamacpp_probe
from .llamacpp_e3 import LlamaCppE3Reference, evaluate_llamacpp_native_e3
from .llamacpp_native_trial import LlamaCppNativeTrialSpec
from .llamacpp_placement import (
    LlamaCppPlacementBinding,
    LlamaCppQualificationPlacement,
    translate_candidate_for_llamacpp_qualification,
)
from .model_manifest import ModelManifest
from .planning_contract import PlanningInput
from .qualification import QualificationEvidence, evidence_applies
from .runtime_identity import RuntimeIdentity
from .schema import ValidationError

HANDOFF_SCHEMA = "tensormeld/target-host-qualification-handoff-v1"


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
class TargetHostQualificationHandoff:
    record: dict[str, Any]
    qualification_evidence: QualificationEvidence

    @property
    def fingerprint(self) -> str:
        return self.record["handoff_sha256"]


def validate_target_host_handoff(
    handoff: TargetHostQualificationHandoff,
) -> dict[str, Any]:
    if not isinstance(handoff, TargetHostQualificationHandoff):
        raise ValidationError("expected TargetHostQualificationHandoff")
    record = handoff.record
    if not isinstance(record, dict):
        raise ValidationError("qualification handoff record must be an object")
    if record.get("handoff_schema") != HANDOFF_SCHEMA:
        raise ValidationError(f"handoff_schema: expected {HANDOFF_SCHEMA}")
    supplied = record.get("handoff_sha256")
    if not isinstance(supplied, str):
        raise ValidationError("qualification handoff has no fingerprint")
    core = dict(record)
    core.pop("handoff_sha256", None)
    if _canonical_sha256(core) != supplied:
        raise ValidationError("qualification handoff fingerprint mismatch")
    if record.get("runtime_manifest_required") is not True:
        raise ValidationError("qualification handoff must require a runtime manifest")
    if (
        record.get("reservation_created") is not False
        or record.get("launch_authorized") is not False
        or record.get("executable") is not False
    ):
        raise ValidationError("qualification handoff cannot self-promote execution")
    return record


def assemble_target_host_qualification(
    *,
    config: Config,
    planning: PlanningInput,
    candidate: dict[str, Any],
    adapter: AdapterCapabilities,
    model: ModelManifest,
    gguf_index: dict[str, Any],
    probe: dict[str, Any],
    native_binding: LlamaCppBinding,
    bound_result: dict[str, Any],
    placement_binding: LlamaCppPlacementBinding,
    placement: LlamaCppQualificationPlacement,
    trial_spec: LlamaCppNativeTrialSpec,
    trial_result: dict[str, Any],
    e3_reference: LlamaCppE3Reference,
    runtime_identities: list[RuntimeIdentity] | tuple[RuntimeIdentity, ...],
    backend_evidence_by_device: dict[str, dict[str, Any]],
    backend_test_artifact_sha256_by_device: dict[str, str],
    observed_at: str | None = None,
) -> TargetHostQualificationHandoff:
    if planning.manifest_ref != model.manifest_sha256:
        raise ValidationError("planning/model identity mismatch in qualification chain")
    if candidate.get("plan_sha256") != placement.candidate_plan_sha256:
        raise ValidationError("candidate/placement plan identity mismatch")
    if tuple(candidate.get("compute_devices", ())) != tuple(sorted(candidate.get("compute_devices", ()))):
        raise ValidationError("candidate compute_devices must be canonical/sorted")

    recomputed_bound = bind_llamacpp_probe(config, probe, native_binding)
    for key in (
        "result_schema",
        "config_sha256",
        "probe_artifact_sha256",
        "binding_sha256",
        "node_id",
        "resolved_mappings",
    ):
        if bound_result.get(key) != recomputed_bound.get(key):
            raise ValidationError(f"supplied bound result differs from current probe/binding at {key}")

    if trial_spec.llama_cli_sha256 != probe.get("artifact_sha256"):
        raise ValidationError("native trial llama-cli differs from probed llama.cpp artifact")

    recomputed_placement = translate_candidate_for_llamacpp_qualification(
        config,
        planning,
        candidate,
        adapter=adapter,
        binding=placement_binding,
        bound_result=bound_result,
        model=model,
        gguf_index=gguf_index,
        profile_name=config.installation.default_profile,
    )
    if recomputed_placement.fingerprint != placement.fingerprint:
        raise ValidationError("supplied qualification placement is stale or mismatched")
    if trial_spec.placement_sha256 != placement.fingerprint:
        raise ValidationError("trial spec placement identity mismatch")
    if trial_spec.config_sha256 != config.fingerprint:
        raise ValidationError("trial spec config identity mismatch")
    if trial_spec.planning_input_sha256 != planning.fingerprint:
        raise ValidationError("trial spec planning identity mismatch")
    if trial_spec.candidate_plan_sha256 != candidate.get("plan_sha256"):
        raise ValidationError("trial spec candidate plan identity mismatch")
    if trial_spec.model_manifest_sha256 != model.manifest_sha256:
        raise ValidationError("trial spec model identity mismatch")

    identities = tuple(runtime_identities)
    by_device = {identity.tensormeld_device_id: identity for identity in identities}
    if len(by_device) != len(identities):
        raise ValidationError("duplicate runtime identity for qualification device")
    compute_devices = tuple(candidate.get("compute_devices", ()))
    if set(by_device) != set(compute_devices):
        raise ValidationError("runtime identities must cover exactly candidate compute devices")
    if set(backend_evidence_by_device) != set(compute_devices):
        raise ValidationError("retained E2 evidence must cover exactly candidate compute devices")
    if set(backend_test_artifact_sha256_by_device) != set(compute_devices):
        raise ValidationError("backend self-test artifacts must cover exactly candidate compute devices")

    readiness: list[dict[str, Any]] = []
    combined_observation = deepcopy(bound_result.get("runtime_observation"))
    if not isinstance(combined_observation, dict):
        raise ValidationError("bound result has no runtime observation")
    devices = combined_observation.get("devices")
    if not isinstance(devices, dict):
        raise ValidationError("bound runtime observation has no device map")

    for device_id in compute_devices:
        result = validate_llamacpp_backend_evidence(
            backend_evidence_by_device[device_id],
            config=config,
            bound_result=bound_result,
            expected_test_artifact_sha256=backend_test_artifact_sha256_by_device[device_id],
            tensormeld_device_id=device_id,
            current_runtime_identity=by_device[device_id],
        )
        if result.get("runtime_ready") is not True:
            raise ValidationError(f"backend E2 did not promote {device_id} to ready")
        readiness.append(result)
        devices[device_id]["state"] = "ready"

    combined_observation["qualified"] = False
    combined_observation["executable"] = False

    evaluation, evidence = evaluate_llamacpp_native_e3(
        adapter=adapter,
        model=model,
        placement=placement,
        spec=trial_spec,
        trial_result=trial_result,
        reference=e3_reference,
        runtime_identities=identities,
        observed_at=observed_at,
    )

    exact = evidence_applies(
        evidence,
        adapter=adapter,
        model=model,
        config_sha256=config.fingerprint,
        device_ids=list(compute_devices),
        context_tokens=trial_spec.context_tokens,
        max_output_tokens=trial_spec.predict_tokens,
        concurrency=1,
        minimum_level="E3",
        candidate_plan_sha256=candidate["plan_sha256"],
        runtime_identity_sha256=[
            by_device[device_id].identity_sha256 for device_id in compute_devices
        ],
        require_v2=True,
    )
    if exact.get("applies") is not True:
        raise ValidationError(
            f"emitted E3 does not apply to qualification chain: {exact.get('reasons')}"
        )

    profile = config.profile_map[config.installation.default_profile]
    target_workload = {
        "task": profile.workload.task,
        "context_tokens": profile.workload.context_tokens,
        "max_output_tokens": profile.workload.max_output_tokens,
        "concurrency": profile.workload.max_active_requests,
    }
    e3_workload = dict(evidence.workload)
    workload_exact_for_profile = e3_workload == target_workload

    runtime_manifest_requirements = {
        "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
        "required_provenance": "native-adapter",
        "config_sha256": config.fingerprint,
        "profile": profile.name,
        "model_manifest_sha256": model.manifest_sha256,
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": trial_spec.llama_cli_sha256,
        "required_devices": list(compute_devices),
        "required_nodes": list(candidate.get("compute_nodes", ())),
        "required_workload": target_workload,
        "e3_workload": e3_workload,
        "e3_workload_matches_profile": workload_exact_for_profile,
        "operator_measurement_required": True,
        "physical_pool_memory_measurement_required": True,
    }

    core = {
        "handoff_schema": HANDOFF_SCHEMA,
        "config_sha256": config.fingerprint,
        "planning_input_sha256": planning.fingerprint,
        "candidate_plan_sha256": candidate["plan_sha256"],
        "model_manifest_sha256": model.manifest_sha256,
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "probe_artifact_sha256": probe["artifact_sha256"],
        "native_binding_sha256": native_binding.fingerprint,
        "placement_sha256": placement.fingerprint,
        "trial_spec_sha256": trial_spec.spec_sha256,
        "e3_reference_sha256": e3_reference.fingerprint,
        "qualification_evidence_sha256": evidence.evidence_sha256,
        "e3_evaluation_sha256": evaluation["evaluation_sha256"],
        "runtime_identity_sha256": [
            by_device[device_id].identity_sha256 for device_id in compute_devices
        ],
        "backend_readiness": [
            {
                "device_id": item["tensormeld_device_id"],
                "evidence_sha256": item["evidence_sha256"],
                "runtime_identity_sha256": item["runtime_identity_sha256"],
                "runtime_ready": True,
            }
            for item in readiness
        ],
        "combined_runtime_observation": combined_observation,
        "runtime_manifest_requirements": runtime_manifest_requirements,
        "backend_ready": True,
        "e3_qualified": True,
        "runtime_manifest_required": True,
        "reservation_created": False,
        "launch_authorized": False,
        "executable": False,
        "warnings": [
            "Qualification handoff performs no subprocess launch or hardware measurement.",
            "A native runtime model/operator/memory manifest is still required.",
            "E3 workload must match the target profile before executable admission can succeed.",
            "Memory reservation and launch recheck remain separate mandatory gates.",
        ],
    }
    core["handoff_sha256"] = _canonical_sha256(core)
    return TargetHostQualificationHandoff(core, evidence)
