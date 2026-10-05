"""Admitted managed llama-server lifecycle bound through a build package identity.

This is the bridge between:
- E3 qualification on exact llama-cli;
- an AcceptedExecutionBundle whose worker artifact is that qualified llama-cli;
- a sibling llama-server artifact from the same TensorMeld-verified build package;
- a live launch-admitted local lease.

The bridge authorizes process startup only. It does not claim request-semantic equivalence
between llama-cli and llama-server and does not issue inference requests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .admission import LocalAdmissionController
from .llamacpp_managed_server import (
    LlamaCppServerLaunchSpec,
    ManagedLlamaCppServer,
)
from .llamacpp_package import LlamaCppBuildPackage
from .native_session import validate_target_host_admission_result
from .schema import ValidationError
from .target_host_admission import TargetHostAdmissionResult

BINDING_SCHEMA = "tensormeld/llamacpp-admitted-server-binding-v1"


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
class AdmittedLlamaCppServerBinding:
    package_sha256: str
    admission_sha256: str
    accepted_bundle_sha256: str
    lease_id: str
    lease_sha256: str
    llama_cli_sha256: str
    llama_server_sha256: str
    server_spec_sha256: str
    fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {
            "binding_schema": BINDING_SCHEMA,
            "package_sha256": self.package_sha256,
            "admission_sha256": self.admission_sha256,
            "accepted_bundle_sha256": self.accepted_bundle_sha256,
            "lease_id": self.lease_id,
            "lease_sha256": self.lease_sha256,
            "llama_cli_sha256": self.llama_cli_sha256,
            "llama_server_sha256": self.llama_server_sha256,
            "server_spec_sha256": self.server_spec_sha256,
            "qualified_cli_artifact": True,
            "server_same_build_package": True,
            "server_semantic_equivalence_qualified": False,
            "inference_request_authorized": False,
            "fingerprint": self.fingerprint,
        }


def bind_admitted_llamacpp_server(
    *,
    controller: LocalAdmissionController,
    admission: TargetHostAdmissionResult,
    package: LlamaCppBuildPackage,
    server_spec: LlamaCppServerLaunchSpec,
) -> AdmittedLlamaCppServerBinding:
    record = validate_target_host_admission_result(admission)
    bundle = admission.bundle

    if package.source_revision != bundle.engine_revision:
        raise ValidationError("llama.cpp package source revision mismatch")
    if package.llama_cli_sha256 != bundle.worker_artifact_sha256:
        raise ValidationError(
            "qualified bundle worker artifact is not package llama-cli"
        )
    if server_spec.accepted_bundle_sha256 != bundle.bundle_sha256:
        raise ValidationError("llama-server launch spec belongs to another bundle")
    if server_spec.source_revision != package.source_revision:
        raise ValidationError("llama-server launch spec revision mismatch")
    if server_spec.server_artifact_sha256 != package.llama_server_sha256:
        raise ValidationError(
            "llama-server launch artifact is not the package server artifact"
        )
    if server_spec.model_manifest_sha256 != bundle.model_manifest_sha256:
        raise ValidationError("llama-server launch model identity mismatch")

    lease_id = record.get("lease_id")
    lease_sha = record.get("lease_sha256")
    active = {lease.lease_id: lease for lease in controller.active_leases()}
    lease = active.get(lease_id)
    if lease is None:
        raise ValidationError("admitted llama-server lease is no longer active")
    if lease.state != "launched":
        raise ValidationError("admitted llama-server requires launched lease")
    if lease.lease_sha256 != lease_sha:
        raise ValidationError("admitted llama-server lease fingerprint mismatch")
    if lease.runtime_manifest_sha256 != bundle.runtime_manifest_sha256:
        raise ValidationError("admitted llama-server runtime manifest mismatch")

    canonical = {
        "binding_schema": BINDING_SCHEMA,
        "package_sha256": package.fingerprint,
        "admission_sha256": admission.fingerprint,
        "accepted_bundle_sha256": bundle.bundle_sha256,
        "lease_id": lease_id,
        "lease_sha256": lease_sha,
        "llama_cli_sha256": package.llama_cli_sha256,
        "llama_server_sha256": package.llama_server_sha256,
        "server_spec_sha256": server_spec.spec_sha256,
        "qualified_cli_artifact": True,
        "server_same_build_package": True,
        "server_semantic_equivalence_qualified": False,
        "inference_request_authorized": False,
    }
    return AdmittedLlamaCppServerBinding(
        package.fingerprint,
        admission.fingerprint,
        bundle.bundle_sha256,
        lease_id,
        lease_sha,
        package.llama_cli_sha256,
        package.llama_server_sha256,
        server_spec.spec_sha256,
        _canonical_sha256(canonical),
    )


class AdmittedManagedLlamaCppServer:
    """Own a managed server while keeping the admitted lease until process exit."""

    def __init__(
        self,
        *,
        controller: LocalAdmissionController,
        admission: TargetHostAdmissionResult,
        package: LlamaCppBuildPackage,
        server_spec: LlamaCppServerLaunchSpec,
        server: ManagedLlamaCppServer | None = None,
    ) -> None:
        self.controller = controller
        self.admission = admission
        self.binding = bind_admitted_llamacpp_server(
            controller=controller,
            admission=admission,
            package=package,
            server_spec=server_spec,
        )
        self.server = server or ManagedLlamaCppServer(server_spec)
        if self.server.spec.spec_sha256 != server_spec.spec_sha256:
            raise ValidationError("managed server instance has different launch spec")
        self.state = "prepared"
        self._lease_released = False

    def _release_lease_after_process_exit(self) -> dict[str, Any]:
        if self.server.pid is not None:
            raise ValidationError("cannot release admitted lease while server process exists")
        if self._lease_released:
            return {
                "status": "ALREADY_RELEASED",
                "lease_id": self.binding.lease_id,
                "released": True,
            }
        result = self.controller.release(self.binding.lease_id)
        self._lease_released = True
        return {
            "status": result["status"],
            "lease_id": self.binding.lease_id,
            "released": True,
        }

    def start(self) -> dict[str, Any]:
        if self.state != "prepared":
            raise ValidationError(f"admitted server cannot start from state {self.state}")
        try:
            started = self.server.start()
            self.state = "starting"
            return {
                **started,
                "package_sha256": self.binding.package_sha256,
                "admitted_binding_sha256": self.binding.fingerprint,
                "lease_id": self.binding.lease_id,
                "inference_request_authorized": False,
            }
        except BaseException:
            # start() guarantees no usable child on process-spawn failure.
            if self.server.pid is None:
                self._release_lease_after_process_exit()
            self.state = "failed"
            raise

    def wait_ready(self, **kwargs: Any) -> dict[str, Any]:
        if self.state not in {"starting", "ready"}:
            raise ValidationError(f"admitted server readiness invalid from state {self.state}")
        try:
            ready = self.server.wait_ready(**kwargs)
            self.state = "ready"
            return {
                **ready,
                "package_sha256": self.binding.package_sha256,
                "admitted_binding_sha256": self.binding.fingerprint,
                "lease_id": self.binding.lease_id,
                "inference_request_authorized": False,
            }
        except BaseException:
            # ManagedLlamaCppServer stops on readiness timeout, but early-exit state can
            # still hold the completed child object. stop() normalizes it before release.
            try:
                self.server.stop()
            finally:
                if self.server.pid is None:
                    self._release_lease_after_process_exit()
            self.state = "failed"
            raise

    def stop(self, **kwargs: Any) -> dict[str, Any]:
        stopped = self.server.stop(**kwargs)
        if self.server.pid is not None:
            raise ValidationError("managed llama-server process still exists after stop")
        release = self._release_lease_after_process_exit()
        self.state = "stopped"
        return {
            **stopped,
            "package_sha256": self.binding.package_sha256,
            "admitted_binding_sha256": self.binding.fingerprint,
            "lease_id": self.binding.lease_id,
            "lease_release_status": release["status"],
            "inference_request_authorized": False,
            "real_model_inference": False,
        }
