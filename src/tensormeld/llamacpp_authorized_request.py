"""First E4-authorized persistent llama-server completion request.

This path is intentionally narrow. It executes exactly the deterministic request body
already qualified by the E4 equivalence spec. It accepts no caller-supplied prompt,
sampler, cache or streaming parameters.

Every call revalidates:
- authorized E4 binding fingerprint/state;
- E4 spec fingerprint;
- ready/live managed server and exact server spec;
- active launched lease with exact lease/runtime-manifest identity.

The lease remains active after a successful request because the persistent server remains
alive. Process shutdown remains responsible for lease release.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from .admission import LocalAdmissionController
from .llamacpp_admitted_server import AdmittedManagedLlamaCppServer
from .llamacpp_server_e4 import (
    AuthorizedAdmittedLlamaCppServerBinding,
    LlamaCppServerE4Spec,
    validate_authorized_llamacpp_server_binding,
    validate_llamacpp_server_e4_spec,
)
from .schema import ValidationError

REQUEST_RESULT_SCHEMA = "tensormeld/llamacpp-authorized-completion-v1"
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


@dataclass(frozen=True)
class AuthorizedLlamaCppCompletionResult:
    record: dict[str, Any]
    content: str

    @property
    def fingerprint(self) -> str:
        return self.record["result_sha256"]


def _validate_live_admitted_server(
    *,
    controller: LocalAdmissionController,
    admitted_server: AdmittedManagedLlamaCppServer,
    authorization: AuthorizedAdmittedLlamaCppServerBinding,
    spec: LlamaCppServerE4Spec,
) -> None:
    validate_authorized_llamacpp_server_binding(authorization)
    validate_llamacpp_server_e4_spec(spec)

    if admitted_server.state != "ready":
        raise ValidationError("authorized completion requires admitted server state ready")
    server = admitted_server.server
    if server.state != "ready" or not server.is_alive:
        raise ValidationError("authorized completion requires a live ready server process")
    if server.spec.spec_sha256 != authorization.server_spec_sha256:
        raise ValidationError("authorized completion server spec differs from authorization")
    if server.spec.spec_sha256 != spec.server_spec_sha256:
        raise ValidationError("authorized completion E4 spec differs from running server")
    if authorization.package_sha256 != admitted_server.binding.package_sha256:
        raise ValidationError("authorized completion package differs from admitted server")
    if authorization.accepted_bundle_sha256 != admitted_server.binding.accepted_bundle_sha256:
        raise ValidationError("authorized completion bundle differs from admitted server")
    if authorization.base_binding_sha256 != admitted_server.binding.fingerprint:
        raise ValidationError("authorized completion base binding differs from admitted server")
    if spec.package_sha256 != authorization.package_sha256:
        raise ValidationError("authorized completion E4 package differs from authorization")
    if spec.accepted_bundle_sha256 != authorization.accepted_bundle_sha256:
        raise ValidationError("authorized completion E4 bundle differs from authorization")

    lease_id = admitted_server.binding.lease_id
    active = {lease.lease_id: lease for lease in controller.active_leases()}
    lease = active.get(lease_id)
    if lease is None:
        raise ValidationError("authorized completion lease is no longer active")
    if lease.state != "launched":
        raise ValidationError("authorized completion requires launched lease")
    if lease.lease_sha256 != admitted_server.binding.lease_sha256:
        raise ValidationError("authorized completion lease fingerprint mismatch")
    if lease.runtime_manifest_sha256 != admitted_server.admission.bundle.runtime_manifest_sha256:
        raise ValidationError("authorized completion runtime manifest mismatch")


def execute_authorized_llamacpp_completion(
    *,
    controller: LocalAdmissionController,
    admitted_server: AdmittedManagedLlamaCppServer,
    authorization: AuthorizedAdmittedLlamaCppServerBinding,
    spec: LlamaCppServerE4Spec,
    timeout_s: float = 30.0,
) -> AuthorizedLlamaCppCompletionResult:
    if not 0.1 <= timeout_s <= 120:
        raise ValidationError("authorized completion timeout must be within 0.1..120 seconds")

    _validate_live_admitted_server(
        controller=controller,
        admitted_server=admitted_server,
        authorization=authorization,
        spec=spec,
    )

    server = admitted_server.server
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
                    f"authorized completion returned HTTP {response.status}"
                )
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        if not server.is_alive:
            admitted_server.state = "failed"
            try:
                admitted_server.stop()
            except Exception:
                pass
        raise ValidationError("authorized completion request failed") from exc

    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValidationError("authorized completion response exceeds 4 MiB")
    try:
        parsed = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("authorized completion response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValidationError("authorized completion response must be an object")
    content = parsed.get("content")
    if not isinstance(content, str):
        raise ValidationError("authorized completion response has no string content")

    # Revalidate after the response so a process/lease loss racing the request cannot
    # be reported as a valid persistent completion.
    _validate_live_admitted_server(
        controller=controller,
        admitted_server=admitted_server,
        authorization=authorization,
        spec=spec,
    )

    execution_source = (
        "fixture-server-subprocess"
        if server.spec.launcher_artifact_sha256 is not None
        else "native-server-subprocess"
    )
    real_model_inference = execution_source == "native-server-subprocess"
    content_bytes = content.encode("utf-8")
    core = {
        "request_result_schema": REQUEST_RESULT_SCHEMA,
        "authorized_binding_sha256": authorization.fingerprint,
        "e4_spec_sha256": spec.fingerprint,
        "package_sha256": authorization.package_sha256,
        "accepted_bundle_sha256": authorization.accepted_bundle_sha256,
        "server_spec_sha256": authorization.server_spec_sha256,
        "lease_id": admitted_server.binding.lease_id,
        "lease_sha256": admitted_server.binding.lease_sha256,
        "execution_source": execution_source,
        "request_body_sha256": _canonical_sha256(spec.request_body),
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "content_bytes": len(content_bytes),
        "http_status": 200,
        "completed": True,
        "lease_remains_active": True,
        "inference_request_authorized": True,
        "real_model_inference": real_model_inference,
    }
    core["result_sha256"] = _canonical_sha256(core)
    return AuthorizedLlamaCppCompletionResult(core, content)
