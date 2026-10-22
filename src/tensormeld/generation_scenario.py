"""Advisory serial phase composition with explicit sampling and cyclic feedback.

This narrow evaluator does not search, admit memory, authorize execution, predict
service TTFT, or claim sustained throughput. It costs one declared ordered chain,
one sampler on a compute owner, and feedback to the first unit's owner only.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import Any

from .config_v2 import Config
from .directional_path_evidence import parse_directional_path_evidence, lookup_transfer_upper_bound
from .generation_phase_evidence import parse_generation_phase_evidence, PHASE_EVIDENCE_SCHEMA_V2
from .legal_model_units import LegalModelUnitsProfile
from .legal_unit_costs import LegalUnitCostsProfile
from .runtime_identity import RuntimeIdentity
from .runtime_model_manifest import RuntimeModelManifest
from .schema import ValidationError
from .selection import resolve_candidates
from .tensor_movability import TensorMovabilityProfile

RESULT_SCHEMA = "tensormeld/generation-scenario-cost-v1"
MAX_COST_US = (1 << 63) - 1


def _cost(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_COST_US:
        raise ValidationError(f"{label}: expected integer microseconds in 0..{MAX_COST_US}")
    return value


def evaluate_generation_scenario(
    *,
    config: Config,
    legal_units: LegalModelUnitsProfile,
    costs: LegalUnitCostsProfile,
    runtime_manifest: RuntimeModelManifest,
    movability: TensorMovabilityProfile,
    runtime_identities: Sequence[RuntimeIdentity],
    phase_data: dict[str, Any],
    path_data: dict[str, Any],
    owners: Sequence[str],
    sampler_device: str,
    prefill_control_us: int,
    decode_control_us: int,
) -> dict[str, Any]:
    """Cost a declared C1 scenario; missing paths return EVIDENCE_INCOMPLETE.

    Controls are explicit additive costs, never inferred as zero. Their measurement
    and non-overlap with operator costs remain the caller's evidence responsibility.
    Raw phase/path inputs are reparsed against current identities on every call.
    """
    prefill_control_us = _cost(prefill_control_us, "prefill_control_us")
    decode_control_us = _cost(decode_control_us, "decode_control_us")
    phase = parse_generation_phase_evidence(
        phase_data, config=config, legal_units=legal_units, costs=costs,
        runtime_manifest=runtime_manifest, movability=movability,
    )
    if phase.schema != PHASE_EVIDENCE_SCHEMA_V2:
        raise ValidationError("phase-specific transfer composition requires v2 phase evidence")
    if phase.concurrency != 1:
        raise ValidationError("serial generation scenario supports concurrency one only")
    paths = parse_directional_path_evidence(
        path_data, config=config, runtime_identities=tuple(runtime_identities),
    )
    provenance_pair = (phase.provenance, paths.provenance)
    if provenance_pair not in {("fixture", "fixture"), ("native-adapter", "native-target")}:
        raise ValidationError("phase/path evidence provenance scopes must agree")
    if not set(paths.runtime_identity_sha256) <= set(movability.runtime_identity_sha256):
        raise ValidationError("path runtime identities differ from the model movement environment")
    if isinstance(owners, (str, bytes)) or not isinstance(owners, Sequence):
        raise ValidationError("owners must be an ordered sequence of device IDs")
    owners = tuple(owners)
    if len(owners) != len(legal_units.units) or any(type(d) is not str for d in owners):
        raise ValidationError("owners must cover every ordered legal unit exactly once")
    if type(sampler_device) is not str or sampler_device not in owners:
        raise ValidationError("initial scenario requires sampler on a compute owner")

    selection = resolve_candidates(config, runtime_manifest.profile)
    eligible = {d["id"]: d for d in selection["eligible_compute_devices"]}
    selected = set(owners)
    if not selected <= set(eligible):
        raise ValidationError("scenario owner is excluded or ineligible under current policy")
    current = {identity.tensormeld_device_id: identity for identity in runtime_identities}
    for device in selected:
        identity = current.get(device)
        if (identity is None
                or identity.identity_sha256 not in movability.runtime_identity_sha256
                or identity.worker_artifact_sha256 != runtime_manifest.worker_artifact_sha256):
            raise ValidationError("scenario runtime identity/worker differs from model measurement environment")
    nodes = {eligible[d]["node"] for d in selected}
    if not (selection["min_compute_devices"] <= len(selected) <= selection["max_compute_devices"]
            and selection["min_compute_nodes"] <= len(nodes) <= selection["max_compute_nodes"]):
        raise ValidationError("scenario violates configured participation limits")
    if not set(selection["required_devices"]) <= selected or not set(selection["required_nodes"]) <= nodes:
        raise ValidationError("scenario omits required resources")
    local = set(selection["required_any_local_gpu"])
    if local and not selected & local:
        raise ValidationError("scenario omits a required local GPU")

    for i, (unit, owner) in enumerate(zip(legal_units.units, owners)):
        if owner not in unit.allowed_devices:
            raise ValidationError(f"unit {unit.id}: owner is not a declared legal device")
        if i and owners[i - 1] != owner and not legal_units.units[i - 1].cut_after:
            raise ValidationError("scenario crosses an undeclared legal cut")

    sampler = next((p for p in phase.sampling_profiles if p.device == sampler_device), None)
    if sampler is None:
        raise ValidationError("scenario lacks explicit sampling evidence for selected device")
    sampling_us = _cost(sampler.sampling_us, "sampling_us")
    phases: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    for label, control in (("prefill", prefill_control_us), ("decode", decode_control_us)):
        compute = 0
        transfers: list[dict[str, Any]] = []
        missing_before = len(missing)
        for i, (unit, owner) in enumerate(zip(phase.units, owners)):
            profile = next((p for p in unit.device_profiles if p.device == owner), None)
            if profile is None:
                raise ValidationError(f"unit {unit.unit_id}: selected device cost is missing")
            compute += _cost(profile.prefill_us if label == "prefill" else profile.decode_step_us,
                             f"{label}.{unit.unit_id}.compute_us")
            if i + 1 < len(owners) and owner != owners[i + 1]:
                payload = (unit.prefill_boundary_output_bytes if label == "prefill"
                           else unit.decode_boundary_output_bytes)
                _append_transfer(paths, label, "boundary", owner, owners[i + 1], payload,
                                 transfers, missing, unit.unit_id)
        _append_transfer(paths, label, "logits", owners[-1], sampler_device,
                         sampler.logits_payload_bytes, transfers, missing, None)
        _append_transfer(paths, label, "token_feedback", sampler_device, owners[0],
                         sampler.feedback_payload_bytes, transfers, missing, None)
        compute = _cost(compute, f"{label}.compute_us")
        transfer_us = _cost(sum(_cost(t["upper_bound_us"], "transfer cost") for t in transfers),
                            f"{label}.transfer_us")
        complete = len(missing) == missing_before
        total = _cost(compute + transfer_us + sampling_us + control, f"{label}.total_us") if complete else None
        phases[label] = {
            "compute_us": compute, "transfer_us": transfer_us if complete else None,
            "sampling_us": sampling_us, "control_us": control,
            "scenario_upper_bound_us": total, "transfers": transfers,
        }
    core = {
        "result_schema": RESULT_SCHEMA,
        "status": "EVIDENCE_INCOMPLETE" if missing else "SCENARIO_COSTED",
        "scope": "serial-ordered-chain-single-sampler-first-owner-feedback",
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal_units.fingerprint,
        "legal_unit_costs_sha256": costs.fingerprint,
        "runtime_manifest_sha256": runtime_manifest.fingerprint,
        "generation_phase_evidence_sha256": phase.fingerprint,
        "directional_path_evidence_sha256": paths.fingerprint,
        "phase_provenance": phase.provenance, "path_provenance": paths.provenance,
        "workload": {"prefill_tokens": phase.prefill_tokens,
                     "decode_context_tokens": phase.decode_context_tokens, "concurrency": 1},
        "owners": list(owners), "sampler_device": sampler_device,
        "phases": phases, "missing_transfers": missing,
        "memory_admission_performed": False, "qualified": False, "executable": False,
        "warnings": [
            "Composition is conditional on declared serial no-overlap component bounds; not a measured end-to-end percentile.",
            "Prefill includes return to next-token start; it is not service TTFT and excludes queue/load/tokenization/client emission.",
            "Feedback covers only the first-unit owner; additional token consumers require a different declared graph contract.",
            "Control must be additive to the supplied compute/path costs; overlap and duplicated timing are not inferred.",
            "No sustained throughput, memory feasibility, execution qualification or globally optimal placement is established.",
        ],
    }
    core["result_sha256"] = hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return core


def _append_transfer(paths, phase, role, source, target, payload, transfers, missing, unit_id):
    transfer = lookup_transfer_upper_bound(
        paths, source_device=source, target_device=target, payload_bytes=payload,
    )
    if transfer is None:
        missing.append({"phase": phase, "role": role, "source_device": source,
                        "target_device": target, "payload_bytes": payload, "unit_id": unit_id})
    else:
        transfers.append({**transfer, "role": role, "unit_id": unit_id})
