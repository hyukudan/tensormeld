"""llama-cli versus llama-server E4 request equivalence.

This gate compares one exact deterministic request across:
- the qualified one-shot llama-cli trial path;
- the same-build persistent llama-server path.

The comparison is intentionally narrow and exact. It binds package, model, pre/post-E3
placement semantics, prompt, context, output length, seed and temperature. Fixture
subprocesses can validate the contract but never become native request authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from .llamacpp_admitted_server import AdmittedLlamaCppServerBinding
from .llamacpp_managed_server import (
    LlamaCppServerLaunchSpec,
    ManagedLlamaCppServer,
    validate_llamacpp_server_launch_spec,
)
from .llamacpp_native_trial import LlamaCppNativeTrialSpec, TRIAL_SCHEMA
from .llamacpp_package import (
    LlamaCppBuildPackage,
    validate_llamacpp_package_identity,
)
from .llamacpp_placement import (
    LlamaCppPlacementTranslation,
    LlamaCppQualificationPlacement,
)
from .model_manifest import _sha256
from .schema import ValidationError, record, text

E4_SPEC_SCHEMA = "tensormeld/llamacpp-server-e4-spec-v1"
E4_RESULT_SCHEMA = "tensormeld/llamacpp-server-e4-result-v1"
E4_EVIDENCE_SCHEMA = "tensormeld/llamacpp-server-e4-evidence-v1"
AUTHORIZED_BINDING_SCHEMA = "tensormeld/llamacpp-authorized-server-binding-v1"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


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


def _placement_semantics(
    qualification: LlamaCppQualificationPlacement,
    execution: LlamaCppPlacementTranslation,
) -> dict[str, Any]:
    if qualification.source_revision != execution.source_revision:
        raise ValidationError("E4 placement revision mismatch")
    q_owners = tuple(qualification.block_owners)
    e_owners = tuple(execution.block_owners)
    if q_owners != e_owners:
        raise ValidationError("E4 pre/post placement block ownership differs")
    if tuple(qualification.device_buffer_types) != tuple(execution.device_buffer_types):
        raise ValidationError("E4 pre/post placement buffer types differ")
    if qualification.override_tensor_value != execution.override_tensor_value:
        raise ValidationError("E4 pre/post override tensor value differs")
    if tuple(qualification.argv_fragment) != tuple(execution.argv_fragment):
        raise ValidationError("E4 pre/post placement argv differs")
    return {
        "block_owners": [list(x) for x in q_owners],
        "device_buffer_types": [list(x) for x in qualification.device_buffer_types],
        "override_tensor_value": qualification.override_tensor_value,
        "argv_fragment": list(qualification.argv_fragment),
    }


@dataclass(frozen=True)
class LlamaCppServerE4Spec:
    package_sha256: str
    model_manifest_sha256: str
    accepted_bundle_sha256: str
    qualification_placement_sha256: str
    execution_placement_sha256: str
    placement_semantics_sha256: str
    trial_spec_sha256: str
    server_spec_sha256: str
    prompt: str
    context_tokens: int
    predict_tokens: int
    seed: int
    temperature: float
    request_body: dict[str, Any]
    fingerprint: str


def build_llamacpp_server_e4_spec(
    *,
    package: LlamaCppBuildPackage,
    trial_spec: LlamaCppNativeTrialSpec,
    qualification_placement: LlamaCppQualificationPlacement,
    server_spec: LlamaCppServerLaunchSpec,
    execution_placement: LlamaCppPlacementTranslation,
) -> LlamaCppServerE4Spec:
    validate_llamacpp_package_identity(package)
    validate_llamacpp_server_launch_spec(server_spec)
    if trial_spec.llama_cli_sha256 != package.llama_cli_sha256:
        raise ValidationError("E4 trial CLI artifact is not package llama-cli")
    if server_spec.server_artifact_sha256 != package.llama_server_sha256:
        raise ValidationError("E4 server artifact is not package llama-server")
    if trial_spec.model_manifest_sha256 != server_spec.model_manifest_sha256:
        raise ValidationError("E4 CLI/server model identity mismatch")
    if trial_spec.source_revision != package.source_revision:
        raise ValidationError("E4 trial revision mismatch")
    if server_spec.source_revision != package.source_revision:
        raise ValidationError("E4 server revision mismatch")
    if trial_spec.placement_sha256 != qualification_placement.fingerprint:
        raise ValidationError("E4 trial qualification placement mismatch")
    if server_spec.placement_sha256 != execution_placement.fingerprint:
        raise ValidationError("E4 server execution placement mismatch")
    if trial_spec.context_tokens != server_spec.context_tokens:
        raise ValidationError("E4 CLI/server context size mismatch")
    semantics = _placement_semantics(
        qualification_placement,
        execution_placement,
    )
    body = {
        "prompt": trial_spec.prompt,
        "n_predict": trial_spec.predict_tokens,
        "seed": 0,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
    }
    placement_semantics_sha256 = _canonical_sha256(semantics)
    canonical = {
        "e4_spec_schema": E4_SPEC_SCHEMA,
        "package_sha256": package.fingerprint,
        "model_manifest_sha256": trial_spec.model_manifest_sha256,
        "accepted_bundle_sha256": server_spec.accepted_bundle_sha256,
        "qualification_placement_sha256": qualification_placement.fingerprint,
        "execution_placement_sha256": execution_placement.fingerprint,
        "placement_semantics_sha256": placement_semantics_sha256,
        "trial_spec_sha256": trial_spec.spec_sha256,
        "server_spec_sha256": server_spec.spec_sha256,
        "prompt": trial_spec.prompt,
        "context_tokens": trial_spec.context_tokens,
        "predict_tokens": trial_spec.predict_tokens,
        "seed": 0,
        "temperature": 0.0,
        "request_body": body,
    }
    return LlamaCppServerE4Spec(
        package.fingerprint,
        trial_spec.model_manifest_sha256,
        server_spec.accepted_bundle_sha256,
        qualification_placement.fingerprint,
        execution_placement.fingerprint,
        placement_semantics_sha256,
        trial_spec.spec_sha256,
        server_spec.spec_sha256,
        trial_spec.prompt,
        trial_spec.context_tokens,
        trial_spec.predict_tokens,
        0,
        0.0,
        body,
        _canonical_sha256(canonical),
    )


def validate_llamacpp_server_e4_spec(
    spec: LlamaCppServerE4Spec,
) -> dict[str, Any]:
    if not isinstance(spec, LlamaCppServerE4Spec):
        raise ValidationError("expected LlamaCppServerE4Spec")
    canonical = {
        "e4_spec_schema": E4_SPEC_SCHEMA,
        "package_sha256": spec.package_sha256,
        "model_manifest_sha256": spec.model_manifest_sha256,
        "accepted_bundle_sha256": spec.accepted_bundle_sha256,
        "qualification_placement_sha256": spec.qualification_placement_sha256,
        "execution_placement_sha256": spec.execution_placement_sha256,
        "placement_semantics_sha256": spec.placement_semantics_sha256,
        "trial_spec_sha256": spec.trial_spec_sha256,
        "server_spec_sha256": spec.server_spec_sha256,
        "prompt": spec.prompt,
        "context_tokens": spec.context_tokens,
        "predict_tokens": spec.predict_tokens,
        "seed": spec.seed,
        "temperature": spec.temperature,
        "request_body": spec.request_body,
    }
    if _canonical_sha256(canonical) != spec.fingerprint:
        raise ValidationError("llama.cpp server E4 spec fingerprint mismatch")
    expected_body = {
        "prompt": spec.prompt,
        "n_predict": spec.predict_tokens,
        "seed": 0,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
    }
    if spec.seed != 0 or spec.temperature != 0 or spec.request_body != expected_body:
        raise ValidationError("llama.cpp server E4 deterministic request contract changed")
    return {**canonical, "spec_sha256": spec.fingerprint}


def validate_llamacpp_server_e4_result(
    result: dict[str, Any],
    *,
    spec: LlamaCppServerE4Spec,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValidationError("E4 server result must be an object")
    expected_fields = {
        "e4_result_schema",
        "e4_spec_sha256",
        "server_spec_sha256",
        "server_artifact_sha256",
        "execution_source",
        "content_sha256",
        "content_bytes",
        "http_status",
        "completed",
        "qualified",
        "real_model_inference",
        "result_sha256",
    }
    if set(result) != expected_fields:
        raise ValidationError("E4 server result fields mismatch")
    if result.get("e4_result_schema") != E4_RESULT_SCHEMA:
        raise ValidationError("E4 server result schema mismatch")
    if result.get("e4_spec_sha256") != spec.fingerprint:
        raise ValidationError("E4 server result spec mismatch")
    if result.get("server_spec_sha256") != spec.server_spec_sha256:
        raise ValidationError("E4 server result launch spec mismatch")
    supplied = result.get("result_sha256")
    core = dict(result)
    core.pop("result_sha256")
    if _canonical_sha256(core) != supplied:
        raise ValidationError("E4 server result fingerprint mismatch")
    _sha256(result.get("content_sha256"), "server_result.content_sha256")
    if result.get("http_status") != 200 or result.get("completed") is not True:
        raise ValidationError("E4 server request did not complete successfully")
    if result.get("qualified") is not False or result.get("real_model_inference") is not False:
        raise ValidationError("E4 request result cannot self-promote qualification")
    return result


def run_llamacpp_server_e4_request(
    *,
    server: ManagedLlamaCppServer,
    spec: LlamaCppServerE4Spec,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    validate_llamacpp_server_e4_spec(spec)
    if server.state != "ready":
        raise ValidationError("E4 server request requires ready managed server")
    if server.spec.spec_sha256 != spec.server_spec_sha256:
        raise ValidationError("E4 spec belongs to another managed server")
    if not 0.1 <= timeout_s <= 120:
        raise ValidationError("E4 request timeout must be within 0.1..120 seconds")

    body = json.dumps(
        spec.request_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    request = urlrequest.Request(
        f"http://{server.spec.host}:{server.spec.port}/completion",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlrequest.urlopen(request, timeout=timeout_s) as response:
            if int(response.status) != 200:
                raise ValidationError(
                    f"E4 server completion returned HTTP {response.status}"
                )
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        raise ValidationError("E4 server completion request failed") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValidationError("E4 server completion response exceeds 4 MiB")
    try:
        parsed = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("E4 server completion response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValidationError("E4 server completion response must be an object")
    content = parsed.get("content")
    if not isinstance(content, str):
        raise ValidationError("E4 server completion response has no string content")
    content_bytes = content.encode("utf-8")
    execution_source = (
        "fixture-server-subprocess"
        if server.spec.launcher_artifact_sha256 is not None
        else "native-server-subprocess"
    )
    core = {
        "e4_result_schema": E4_RESULT_SCHEMA,
        "e4_spec_sha256": spec.fingerprint,
        "server_spec_sha256": server.spec.spec_sha256,
        "server_artifact_sha256": server.spec.server_artifact_sha256,
        "execution_source": execution_source,
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "content_bytes": len(content_bytes),
        "http_status": 200,
        "completed": True,
        "qualified": False,
        "real_model_inference": False,
    }
    core["result_sha256"] = _canonical_sha256(core)
    return core


@dataclass(frozen=True)
class LlamaCppServerE4Evidence:
    e4_spec_sha256: str
    package_sha256: str
    model_manifest_sha256: str
    accepted_bundle_sha256: str
    qualification_placement_sha256: str
    execution_placement_sha256: str
    trial_spec_sha256: str
    server_spec_sha256: str
    cli_output_sha256: str
    server_output_sha256: str
    cli_execution_source: str
    server_execution_source: str
    outputs_equal: bool
    qualified: bool
    fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {
            "e4_evidence_schema": E4_EVIDENCE_SCHEMA,
            "e4_spec_sha256": self.e4_spec_sha256,
            "package_sha256": self.package_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "accepted_bundle_sha256": self.accepted_bundle_sha256,
            "qualification_placement_sha256": self.qualification_placement_sha256,
            "execution_placement_sha256": self.execution_placement_sha256,
            "trial_spec_sha256": self.trial_spec_sha256,
            "server_spec_sha256": self.server_spec_sha256,
            "cli_output_sha256": self.cli_output_sha256,
            "server_output_sha256": self.server_output_sha256,
            "cli_execution_source": self.cli_execution_source,
            "server_execution_source": self.server_execution_source,
            "outputs_equal": self.outputs_equal,
            "level": "E4" if self.qualified else "fixture",
            "qualified": self.qualified,
            "inference_request_authorized": False,
            "fingerprint": self.fingerprint,
        }


def evaluate_llamacpp_server_e4(
    *,
    spec: LlamaCppServerE4Spec,
    cli_trial_result: dict[str, Any],
    server_result: dict[str, Any],
) -> LlamaCppServerE4Evidence:
    if not isinstance(cli_trial_result, dict):
        raise ValidationError("E4 CLI trial result must be an object")
    if cli_trial_result.get("trial_schema") != TRIAL_SCHEMA:
        raise ValidationError("E4 CLI trial result schema mismatch")
    if cli_trial_result.get("spec_sha256") != spec.trial_spec_sha256:
        raise ValidationError("E4 CLI trial spec mismatch")
    if cli_trial_result.get("model_manifest_sha256") != spec.model_manifest_sha256:
        raise ValidationError("E4 CLI trial model mismatch")
    if cli_trial_result.get("placement_sha256") != spec.qualification_placement_sha256:
        raise ValidationError("E4 CLI trial placement mismatch")
    if cli_trial_result.get("process_succeeded") is not True:
        raise ValidationError("E4 CLI trial did not succeed")
    cli_output = _sha256(
        cli_trial_result.get("stdout_sha256"), "cli_trial_result.stdout_sha256"
    )
    cli_source = text(
        cli_trial_result.get("execution_source"),
        "cli_trial_result.execution_source",
    )

    validate_llamacpp_server_e4_spec(spec)
    validate_llamacpp_server_e4_result(server_result, spec=spec)
    server_output = _sha256(
        server_result.get("content_sha256"), "server_result.content_sha256"
    )
    server_source = text(
        server_result.get("execution_source"), "server_result.execution_source"
    )

    outputs_equal = cli_output == server_output
    native = (
        cli_source == "native-subprocess"
        and server_source == "native-server-subprocess"
    )
    qualified = outputs_equal and native
    canonical = {
        "e4_evidence_schema": E4_EVIDENCE_SCHEMA,
        "e4_spec_sha256": spec.fingerprint,
        "package_sha256": spec.package_sha256,
        "model_manifest_sha256": spec.model_manifest_sha256,
        "accepted_bundle_sha256": spec.accepted_bundle_sha256,
        "qualification_placement_sha256": spec.qualification_placement_sha256,
        "execution_placement_sha256": spec.execution_placement_sha256,
        "trial_spec_sha256": spec.trial_spec_sha256,
        "server_spec_sha256": spec.server_spec_sha256,
        "cli_output_sha256": cli_output,
        "server_output_sha256": server_output,
        "cli_execution_source": cli_source,
        "server_execution_source": server_source,
        "outputs_equal": outputs_equal,
        "level": "E4" if qualified else "fixture",
        "qualified": qualified,
        "inference_request_authorized": False,
    }
    return LlamaCppServerE4Evidence(
        spec.fingerprint,
        spec.package_sha256,
        spec.model_manifest_sha256,
        spec.accepted_bundle_sha256,
        spec.qualification_placement_sha256,
        spec.execution_placement_sha256,
        spec.trial_spec_sha256,
        spec.server_spec_sha256,
        cli_output,
        server_output,
        cli_source,
        server_source,
        outputs_equal,
        qualified,
        _canonical_sha256(canonical),
    )


@dataclass(frozen=True)
class AuthorizedAdmittedLlamaCppServerBinding:
    base_binding_sha256: str
    e4_evidence_sha256: str
    e4_spec_sha256: str
    expected_output_sha256: str
    package_sha256: str
    accepted_bundle_sha256: str
    server_spec_sha256: str
    fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {
            "binding_schema": AUTHORIZED_BINDING_SCHEMA,
            "base_binding_sha256": self.base_binding_sha256,
            "e4_evidence_sha256": self.e4_evidence_sha256,
            "e4_spec_sha256": self.e4_spec_sha256,
            "expected_output_sha256": self.expected_output_sha256,
            "package_sha256": self.package_sha256,
            "accepted_bundle_sha256": self.accepted_bundle_sha256,
            "server_spec_sha256": self.server_spec_sha256,
            "server_semantic_equivalence_qualified": True,
            "inference_request_authorized": True,
            "real_model_inference": False,
            "fingerprint": self.fingerprint,
        }


def validate_authorized_llamacpp_server_binding(
    binding: AuthorizedAdmittedLlamaCppServerBinding,
) -> dict[str, Any]:
    if not isinstance(binding, AuthorizedAdmittedLlamaCppServerBinding):
        raise ValidationError("expected AuthorizedAdmittedLlamaCppServerBinding")
    record = binding.as_record()
    supplied = record.pop("fingerprint")
    if _canonical_sha256(record) != supplied:
        raise ValidationError("authorized llama.cpp server binding fingerprint mismatch")
    if (
        record.get("server_semantic_equivalence_qualified") is not True
        or record.get("inference_request_authorized") is not True
        or record.get("real_model_inference") is not False
    ):
        raise ValidationError("authorized llama.cpp server binding state is invalid")
    return {**record, "fingerprint": supplied}


def validate_llamacpp_server_e4_evidence(
    evidence: LlamaCppServerE4Evidence,
) -> dict[str, Any]:
    if not isinstance(evidence, LlamaCppServerE4Evidence):
        raise ValidationError("expected LlamaCppServerE4Evidence")
    record = evidence.as_record()
    supplied = record.pop("fingerprint")
    if _canonical_sha256(record) != supplied:
        raise ValidationError("llama.cpp server E4 evidence fingerprint mismatch")
    if evidence.qualified:
        if (
            evidence.cli_execution_source != "native-subprocess"
            or evidence.server_execution_source != "native-server-subprocess"
            or evidence.outputs_equal is not True
        ):
            raise ValidationError("qualified E4 evidence has invalid native provenance")
    return {**record, "fingerprint": supplied}


def authorize_admitted_llamacpp_server_requests(
    *,
    binding: AdmittedLlamaCppServerBinding,
    spec: LlamaCppServerE4Spec,
    evidence: LlamaCppServerE4Evidence,
) -> AuthorizedAdmittedLlamaCppServerBinding:
    validate_llamacpp_server_e4_spec(spec)
    validate_llamacpp_server_e4_evidence(evidence)
    if evidence.e4_spec_sha256 != spec.fingerprint:
        raise ValidationError("E4 evidence belongs to another equivalence spec")
    if evidence.package_sha256 != spec.package_sha256:
        raise ValidationError("E4 evidence package differs from equivalence spec")
    if evidence.model_manifest_sha256 != spec.model_manifest_sha256:
        raise ValidationError("E4 evidence model differs from equivalence spec")
    if evidence.accepted_bundle_sha256 != spec.accepted_bundle_sha256:
        raise ValidationError("E4 evidence bundle differs from equivalence spec")
    if evidence.qualification_placement_sha256 != spec.qualification_placement_sha256:
        raise ValidationError("E4 evidence qualification placement differs from spec")
    if evidence.execution_placement_sha256 != spec.execution_placement_sha256:
        raise ValidationError("E4 evidence execution placement differs from spec")
    if evidence.trial_spec_sha256 != spec.trial_spec_sha256:
        raise ValidationError("E4 evidence trial spec differs from equivalence spec")
    if evidence.server_spec_sha256 != spec.server_spec_sha256:
        raise ValidationError("E4 evidence server spec differs from equivalence spec")
    if evidence.qualified is not True:
        raise ValidationError("native E4 equivalence is required for request authorization")
    if evidence.outputs_equal is not True:
        raise ValidationError("E4 outputs are not equal")
    if evidence.package_sha256 != binding.package_sha256:
        raise ValidationError("E4 package identity differs from admitted server binding")
    if evidence.server_spec_sha256 != binding.server_spec_sha256:
        raise ValidationError("E4 server spec differs from admitted server binding")
    if evidence.accepted_bundle_sha256 != binding.accepted_bundle_sha256:
        raise ValidationError("E4 accepted bundle differs from admitted server binding")
    canonical = {
        "binding_schema": AUTHORIZED_BINDING_SCHEMA,
        "base_binding_sha256": binding.fingerprint,
        "e4_evidence_sha256": evidence.fingerprint,
        "e4_spec_sha256": spec.fingerprint,
        "expected_output_sha256": evidence.server_output_sha256,
        "package_sha256": binding.package_sha256,
        "accepted_bundle_sha256": binding.accepted_bundle_sha256,
        "server_spec_sha256": binding.server_spec_sha256,
        "server_semantic_equivalence_qualified": True,
        "inference_request_authorized": True,
        "real_model_inference": False,
    }
    return AuthorizedAdmittedLlamaCppServerBinding(
        binding.fingerprint,
        evidence.fingerprint,
        spec.fingerprint,
        evidence.server_output_sha256,
        binding.package_sha256,
        binding.accepted_bundle_sha256,
        binding.server_spec_sha256,
        _canonical_sha256(canonical),
    )
