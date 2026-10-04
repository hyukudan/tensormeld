"""Lease-bound native session lifecycle for an admitted execution bundle.

The managed session consumes an already successful TargetHostAdmissionResult and an
approved whole-block backend bound to that exact bundle. It starts no detached/background
processes: execution is synchronous through the existing bounded backend interface.

The session owns the launch-admitted lease lifecycle. Terminal completion, failure or
cancellation releases the lease deterministically. Portable tests use the fixture native
worker and therefore keep real_model_inference=false.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import Lock
from typing import Any

from .admission import LocalAdmissionController
from .schema import ValidationError
from .target_host_admission import TargetHostAdmissionResult
from .whole_block_execution import (
    AcceptedExecutionBundle,
    ReferenceWholeBlockSession,
    WholeBlockBackend,
)

SESSION_SCHEMA = "tensormeld/native-admitted-session-v1"


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


def validate_target_host_admission_result(
    result: TargetHostAdmissionResult,
) -> dict[str, Any]:
    if not isinstance(result, TargetHostAdmissionResult):
        raise ValidationError("expected TargetHostAdmissionResult")
    record = result.record
    if not isinstance(record, dict):
        raise ValidationError("admission result record must be an object")
    if record.get("orchestrator_schema") != "tensormeld/target-host-admission-v1":
        raise ValidationError("admission result schema mismatch")
    supplied = record.get("orchestrator_sha256")
    if not isinstance(supplied, str):
        raise ValidationError("admission result has no fingerprint")
    core = dict(record)
    core.pop("orchestrator_sha256", None)
    if _canonical_sha256(core) != supplied:
        raise ValidationError("admission result fingerprint mismatch")
    if record.get("accepted_execution_bundle_sha256") != result.bundle.bundle_sha256:
        raise ValidationError("admission result bundle fingerprint mismatch")
    if (
        record.get("reservation_created") is not True
        or record.get("launch_authorized") is not True
        or record.get("execution_authorized") is not True
    ):
        raise ValidationError("admission result is not execution-authorized")
    if (
        record.get("inference_started") is not False
        or record.get("real_model_inference") is not False
    ):
        raise ValidationError("admission result cannot pre-claim inference")
    return record


@dataclass(frozen=True)
class NativeSessionResult:
    record: dict[str, Any]
    output: bytes | None

    @property
    def fingerprint(self) -> str:
        return self.record["session_result_sha256"]


class NativeAdmittedSession:
    """Synchronous admitted session with deterministic lease cleanup."""

    def __init__(
        self,
        *,
        controller: LocalAdmissionController,
        admission: TargetHostAdmissionResult,
        backend: WholeBlockBackend,
    ) -> None:
        record = validate_target_host_admission_result(admission)
        bundle = admission.bundle
        backend_bundle_sha = getattr(backend, "bundle_sha256", None)
        if backend_bundle_sha is not None and backend_bundle_sha != bundle.bundle_sha256:
            raise ValidationError("native session backend belongs to another bundle")

        lease_id = record.get("lease_id")
        lease_sha = record.get("lease_sha256")
        if not isinstance(lease_id, str) or not lease_id:
            raise ValidationError("admission result has no lease_id")
        if not isinstance(lease_sha, str) or len(lease_sha) != 64:
            raise ValidationError("admission result has invalid lease_sha256")

        active = {
            lease.lease_id: lease
            for lease in controller.active_leases()
        }
        lease = active.get(lease_id)
        if lease is None:
            raise ValidationError("admitted lease is no longer active")
        if lease.state != "launched":
            raise ValidationError("native session requires a launched lease")
        if lease.lease_sha256 != lease_sha:
            raise ValidationError("native session lease fingerprint mismatch")
        if lease.runtime_manifest_sha256 != bundle.runtime_manifest_sha256:
            raise ValidationError("native session lease/runtime manifest mismatch")

        self.controller = controller
        self.admission = admission
        self.bundle = bundle
        self.backend = backend
        self.lease_id = lease_id
        self.lease_sha256 = lease_sha
        self._lock = Lock()
        self.state = "prepared"
        self._reference = ReferenceWholeBlockSession(bundle, backend)
        self._lease_released = False

    def _release_lease(self) -> dict[str, Any]:
        with self._lock:
            if self._lease_released:
                return {
                    "result_schema": "tensormeld/native-session-release-v1",
                    "status": "ALREADY_RELEASED",
                    "lease_id": self.lease_id,
                    "released": True,
                }
            result = self.controller.release(self.lease_id)
            self._lease_released = True
            return {
                "result_schema": "tensormeld/native-session-release-v1",
                "status": result["status"],
                "lease_id": self.lease_id,
                "released": True,
            }

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            current = self.state
            if current == "released":
                return {
                    "result_schema": "tensormeld/native-session-cancel-v1",
                    "status": "ALREADY_RELEASED",
                    "lease_id": self.lease_id,
                    "cancelled": False,
                }
        self._reference.cancel()
        with self._lock:
            if current == "running":
                return {
                    "result_schema": "tensormeld/native-session-cancel-v1",
                    "status": "CANCEL_REQUESTED",
                    "lease_id": self.lease_id,
                    "cancelled": False,
                    "lease_released": False,
                }
            if self.state == "prepared":
                self.state = "cancelled"
        release = self._release_lease()
        return {
            "result_schema": "tensormeld/native-session-cancel-v1",
            "status": "CANCELLED",
            "lease_id": self.lease_id,
            "cancelled": True,
            "lease_release_status": release["status"],
        }

    def run(self, payload: bytes) -> NativeSessionResult:
        with self._lock:
            if self.state != "prepared":
                raise ValidationError(
                    f"native session cannot run from state {self.state}"
                )
            self.state = "running"

        try:
            run_result = self._reference.run(payload)
            status = run_result["status"]
            with self._lock:
                self.state = "completed" if status == "COMPLETED" else "cancelled"
            release = self._release_lease()
            core = {
                "session_schema": SESSION_SCHEMA,
                "accepted_execution_bundle_sha256": self.bundle.bundle_sha256,
                "lease_id": self.lease_id,
                "lease_sha256": self.lease_sha256,
                "status": status,
                "segments_executed": run_result["segments_executed"],
                "output_sha256": (
                    hashlib.sha256(run_result["output"]).hexdigest()
                    if run_result["output"] is not None
                    else None
                ),
                "lease_release_status": release["status"],
                "inference_started": True,
                "real_model_inference": run_result["real_model_inference"],
                "released": True,
            }
            core["session_result_sha256"] = _canonical_sha256(core)
            return NativeSessionResult(core, run_result["output"])
        except BaseException:
            with self._lock:
                self.state = "failed"
            self._release_lease()
            raise

    def release(self) -> dict[str, Any]:
        with self._lock:
            if self.state == "running":
                raise ValidationError(
                    "running native session cannot be released directly"
                )
        try:
            self._reference.release()
        except ValidationError:
            # The reference session can already be terminal/released. Lease cleanup is
            # the authoritative resource action here.
            pass
        result = self._release_lease()
        with self._lock:
            self.state = "released"
        return result
