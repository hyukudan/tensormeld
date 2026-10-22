"""Bounded C1 consumer of the existing exact authorized completion path.

Cancellation is cooperative: a client acknowledgement does not free the backend slot
or the admitted server lease. Ambiguous request failures fence this consumer until the
owned process is confirmed gone. No arbitrary request bodies or proxy endpoints exist.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import math
import json
from threading import Condition, Lock
from time import monotonic
from typing import Any, Callable
from uuid import uuid4

from .admission import LocalAdmissionController
from .llamacpp_admitted_server import AdmittedManagedLlamaCppServer
from .llamacpp_authorized_request import (
    AuthorizedLlamaCppCompletionResult,
    _canonical_sha256,
    _validate_live_admitted_server,
    execute_authorized_llamacpp_completion,
)
from .llamacpp_server_e4 import AuthorizedAdmittedLlamaCppServerBinding, LlamaCppServerE4Spec
from .schema import MAX_INPUT_BYTES, ValidationError, text

_OWNER_LOCK = Lock()


def _duration(value: float, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.1 <= value <= 120:
        raise ValidationError(f"{label}: expected finite seconds in 0.1..120")
    return float(value)


@dataclass(frozen=True)
class ServingAttempt:
    request_id: str
    attempt_id: str
    worker_epoch: str
    deadline: float


@dataclass(frozen=True)
class C1ServingResult:
    record: dict[str, Any]
    completion: AuthorizedLlamaCppCompletionResult | None


@dataclass
class _OwnedAttempt:
    token: ServingAttempt
    state: str = "queued"
    claimed: bool = False
    cancel_requested: bool = False
    in_client: bool = False
    result: C1ServingResult | None = None


class AuthorizedC1LlamaCppServing:
    """One FIFO consumer bound to one ready admitted process and exact replay spec.

    submit() reserves a bounded pending position; run() consumes it synchronously.
    The caller owns the run thread. No detached work or native family-serving gate
    is introduced. drain() closes admission permanently; restarting needs a freshly
    validated admitted process and a new consumer.
    """

    def __init__(self, *, controller: LocalAdmissionController,
                 admitted_server: AdmittedManagedLlamaCppServer,
                 authorization: AuthorizedAdmittedLlamaCppServerBinding,
                 spec: LlamaCppServerE4Spec, max_pending: int = 32,
                 max_request_bytes: int = MAX_INPUT_BYTES,
                 max_owned_bytes: int = MAX_INPUT_BYTES,
                 clock: Callable[[], float] = monotonic) -> None:
        for label, value, maximum in (("max_pending", max_pending, 128),
                                      ("max_request_bytes", max_request_bytes, MAX_INPUT_BYTES),
                                      ("max_owned_bytes", max_owned_bytes, 128 * MAX_INPUT_BYTES)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValidationError(f"{label}: expected integer in 1..{maximum}")
        self.controller = controller
        self.admitted_server = admitted_server
        self.authorization = authorization
        self.spec = spec
        self._validate()
        self._server = admitted_server.server
        self._process = self._server._child
        if self._process is None:
            raise ValidationError("C1 serving requires an owned server process")
        self._pid = self._server.pid
        self.worker_epoch = uuid4().hex
        self._binding_sha256 = admitted_server.binding.fingerprint
        self._authorization_sha256 = authorization.fingerprint
        self._spec_sha256 = spec.fingerprint
        self._body_bytes = len(json.dumps(spec.request_body, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if self._body_bytes > max_request_bytes or self._body_bytes > max_owned_bytes:
            raise ValidationError("authorized replay body exceeds serving byte limits")
        self.max_pending = max_pending
        self.max_owned_bytes = max_owned_bytes
        self._clock = clock
        self._condition = Condition()
        self._pending: deque[_OwnedAttempt] = deque()
        self._attempts: dict[str, _OwnedAttempt] = {}
        self._terminal: OrderedDict[str, _OwnedAttempt] = OrderedDict()
        self._active: _OwnedAttempt | None = None
        self._owned_bytes = 0
        self._accepting = True
        self._fenced = False
        self._fence_reason: str | None = None
        # One admitted lifecycle cannot acquire two independent C1 schedulers or
        # erase a quarantined owner by constructing a fresh facade.
        with _OWNER_LOCK:
            if getattr(admitted_server, "_c1_serving_owner", None) is not None:
                raise ValidationError("admitted worker already has a C1 serving owner")
            admitted_server._c1_serving_owner = self

    def _validate(self) -> None:
        _validate_live_admitted_server(controller=self.controller, admitted_server=self.admitted_server,
                                       authorization=self.authorization, spec=self.spec)

    def _same_worker(self) -> bool:
        return (self.admitted_server.server is self._server
                and self._server._child is self._process and self._server.pid == self._pid
                and self.admitted_server.binding.fingerprint == self._binding_sha256
                and self.authorization.fingerprint == self._authorization_sha256
                and self.spec.fingerprint == self._spec_sha256)

    def _finish(self, entry: _OwnedAttempt, status: str, *, backend_terminal: bool,
                ownership_released: bool,
                reason_code: str | None = None,
                completion: AuthorizedLlamaCppCompletionResult | None = None) -> C1ServingResult:
        core = {"result_schema": "tensormeld/llamacpp-c1-serving-result-v1",
                "request_id": entry.token.request_id, "attempt_id": entry.token.attempt_id,
                "worker_epoch": entry.token.worker_epoch, "status": status,
                "reason_code": status if reason_code is None else reason_code,
                "backend_terminal": backend_terminal, "request_ownership_released": ownership_released,
                "server_lease_release_performed": False,
                "completion_sha256": None if completion is None else completion.fingerprint}
        core["result_sha256"] = _canonical_sha256(core)
        entry.state = status.lower()
        result = C1ServingResult(core, completion)
        # Retain bounded terminal metadata, never the completion payload. The run
        # caller owns returned content and its lifetime.
        entry.result = C1ServingResult(dict(core), None)
        self._terminal[entry.token.attempt_id] = entry
        while len(self._terminal) > 128:
            old_id, old = self._terminal.popitem(last=False)
            if old is not self._active:
                self._attempts.pop(old_id, None)
        self._condition.notify_all()
        return result

    def _release_active(self, entry: _OwnedAttempt) -> None:
        if self._active is entry:
            self._active = None
            self._owned_bytes -= self._body_bytes
            self._condition.notify_all()

    def _remove_pending(self, entry: _OwnedAttempt, status: str) -> C1ServingResult:
        self._pending.remove(entry)
        self._owned_bytes -= self._body_bytes
        return self._finish(entry, status, backend_terminal=True, ownership_released=True)

    def _refresh(self) -> None:
        if self._process.poll() is not None:
            self._fence("WORKER_PROCESS_EXITED")
        now = self._clock()
        for entry in tuple(self._pending):
            if now >= entry.token.deadline:
                self._remove_pending(entry, "DEADLINE_EXCEEDED")
        # The process handle belongs to this epoch, even if someone installed a
        # replacement in the managed wrapper. Never inspect a successor for cleanup.
        if self._active is not None and not self._active.in_client and self._process.poll() is not None:
            self._release_active(self._active)

    def _fence(self, reason_code: str = "WORKER_IDENTITY_CHANGED") -> None:
        self._fenced = True
        self._accepting = False
        if self._fence_reason is None:
            self._fence_reason = reason_code
        for pending in tuple(self._pending):
            self._pending.remove(pending)
            self._owned_bytes -= self._body_bytes
            self._finish(pending, "WORKER_REVOKED", backend_terminal=True,
                         ownership_released=True, reason_code=self._fence_reason)

    def submit(self, request_id: str, *, timeout_s: float = 30.0) -> ServingAttempt:
        request_id = text(request_id, "request_id")
        timeout_s = _duration(timeout_s, "timeout_s")
        with self._condition:
            self._refresh()
            if not self._accepting:
                raise ValidationError("C1 serving is draining or fenced")
            if not self._same_worker():
                self._fence()
                raise ValidationError("C1 serving worker epoch changed")
            self._validate()
            if len(self._pending) >= self.max_pending:
                raise ValidationError("C1 pending queue count limit exceeded")
            if self._owned_bytes + self._body_bytes > self.max_owned_bytes:
                raise ValidationError("C1 owned request byte limit exceeded")
            token = ServingAttempt(request_id, uuid4().hex, self.worker_epoch, self._clock() + timeout_s)
            entry = _OwnedAttempt(token)
            self._pending.append(entry)
            self._attempts[token.attempt_id] = entry
            self._owned_bytes += self._body_bytes
            self._condition.notify_all()
            return token

    def cancel(self, request_id: str, attempt_id: str) -> dict[str, Any]:
        with self._condition:
            self._refresh()
            entry = self._attempts.get(attempt_id)
            if entry is None or entry.token.request_id != request_id:
                return {"status": "UNKNOWN_ATTEMPT", "cancelled": False}
            if entry.result is not None:
                return {"status": "ALREADY_" + entry.result.record["status"], "cancelled": False}
            if entry is self._active:
                entry.cancel_requested = True
                return {"status": "CANCEL_REQUESTED", "cancelled": False,
                        "request_ownership_released": False, "server_lease_release_performed": False}
            self._remove_pending(entry, "CANCELLED")
            return {"status": "CANCELLED", "cancelled": True,
                    "request_ownership_released": True, "server_lease_release_performed": False}

    def run(self, attempt: ServingAttempt) -> C1ServingResult:
        if not isinstance(attempt, ServingAttempt):
            raise ValidationError("C1 run requires an issued ServingAttempt")
        with self._condition:
            entry = self._attempts.get(attempt.attempt_id)
            if entry is None or entry.token is not attempt:
                raise ValidationError("C1 run requires the issued attempt token")
            if entry.claimed:
                raise ValidationError("C1 attempt already has a run owner")
            entry.claimed = True
            try:
                while True:
                    self._refresh()
                    if entry.result is not None:
                        return entry.result
                    if not self._same_worker():
                        self._fence()
                        return entry.result
                    remaining = entry.token.deadline - self._clock()
                    if remaining < 0.1:
                        return self._remove_pending(entry, "DEADLINE_EXCEEDED")
                    if self._active is None and self._pending[0] is entry:
                        self._validate()
                        if not self._same_worker():
                            self._fence()
                            return entry.result
                        remaining = entry.token.deadline - self._clock()
                        if remaining < 0.1:
                            return self._remove_pending(entry, "DEADLINE_EXCEEDED")
                        self._pending.popleft()
                        self._active = entry
                        entry.state = "running"
                        entry.in_client = True
                        break
                    earliest = min(p.token.deadline for p in self._pending)
                    self._condition.wait(timeout=min(remaining, max(0, earliest - self._clock())))
            except BaseException:
                if entry in self._pending:
                    self._remove_pending(entry, "FAILED")
                raise

        try:
            completion = execute_authorized_llamacpp_completion(
                controller=self.controller, admitted_server=self.admitted_server,
                authorization=self.authorization, spec=self.spec, timeout_s=min(120.0, remaining),
            )
        except BaseException as exc:
            with self._condition:
                entry.in_client = False
                self._fence("CLIENT_EXECUTION_FAILED")
                terminal = self._process.poll() is not None
                if terminal:
                    self._release_active(entry)
                result = self._finish(entry, "FAILED", backend_terminal=terminal, ownership_released=terminal,
                                      reason_code="AUTHORIZED_COMPLETION_FAILED")
            if not isinstance(exc, Exception):
                raise
            return result
        with self._condition:
            entry.in_client = False
            if not self._same_worker():
                self._fence()
                status = "WORKER_REVOKED"
            elif entry.cancel_requested:
                status = "CANCELLED"
            elif self._clock() >= entry.token.deadline:
                status = "DEADLINE_EXCEEDED"
            else:
                status = "COMPLETED"
            self._release_active(entry)
            return self._finish(entry, status, backend_terminal=True, ownership_released=True,
                                completion=completion if status == "COMPLETED" else None)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            self._refresh()
            return {"worker_epoch": self.worker_epoch, "accepting": self._accepting,
                    "fenced": self._fenced, "pending": len(self._pending),
                    "fence_reason": self._fence_reason,
                    "active": self._active is not None, "owned_bytes": self._owned_bytes}

    def drain(self, *, timeout_s: float = 30.0, stop_server: bool = False) -> dict[str, Any]:
        timeout_s = _duration(timeout_s, "drain.timeout_s")
        if type(stop_server) is not bool:
            raise ValidationError("stop_server: expected boolean")
        with self._condition:
            self._accepting = False
            for pending in tuple(self._pending):
                self._remove_pending(pending, "DRAINED")
        # Process stop owns admitted lease cleanup. Do not stop a replacement worker
        # under an old consumer, and do not hold the queue lock while waiting on exit.
        stopped = None
        if stop_server:
            with self._condition:
                if not self._same_worker():
                    self._fence()
                    raise ValidationError("old C1 consumer cannot stop a replacement worker")
            stopped = self.admitted_server.stop(terminate_timeout_s=min(timeout_s, 60),
                                                kill_timeout_s=min(timeout_s, 60))
        end = self._clock() + timeout_s
        with self._condition:
            while True:
                self._refresh()
                if self._active is None:
                    return {"status": "DRAINED", "active": False, "server_stop": stopped}
                remaining = end - self._clock()
                if remaining <= 0:
                    return {"status": "DRAIN_TIMEOUT", "active": True, "server_stop": stopped}
                self._condition.wait(timeout=min(remaining, 0.1))
