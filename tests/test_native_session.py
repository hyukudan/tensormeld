from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from tensormeld.admission import LocalAdmissionController
from tensormeld.native_session import NativeAdmittedSession
from tensormeld.schema import ValidationError
from tensormeld.target_host_admission import orchestrate_target_host_admission
from test_target_host_admission import TargetHostAdmissionOrchestratorTests


class DeterministicBackend:
    def __init__(self, bundle_sha256, *, fail=False, entered=None, proceed=None):
        self.bundle_sha256 = bundle_sha256
        self.fail = fail
        self.entered = entered
        self.proceed = proceed
        self.calls = []

    def execute_segment(self, *, device_id, unit_ids, payload):
        self.calls.append((device_id, unit_ids, payload))
        if self.entered is not None:
            self.entered.set()
        if self.proceed is not None:
            if not self.proceed.wait(timeout=5):
                raise RuntimeError("fixture backend did not receive its terminal gate")
        if self.fail:
            raise ValidationError("fixture backend failure")
        return payload + ("|" + device_id + ":" + ",".join(unit_ids)).encode()


class LockAcquisitionProbe:
    """Expose whether a named contender can acquire a lifecycle lock immediately."""

    def __init__(self, lock, contender):
        self.lock = lock
        self.contender = contender
        self.attempted = threading.Event()
        self.contended = None

    def __enter__(self):
        if threading.current_thread().name == self.contender:
            acquired = self.lock.acquire(blocking=False)
            self.contended = not acquired
            self.attempted.set()
            if not acquired:
                self.lock.acquire()
        else:
            self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()


class NativeAdmittedSessionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TargetHostAdmissionOrchestratorTests(
            methodName="test_reserve_fresh_recheck_builds_accepted_bundle_without_starting_inference"
        )
        self.fixture.setUp()
        self.controller = LocalAdmissionController()
        reserve, launch = self.fixture.snapshots()
        self.admission = orchestrate_target_host_admission(
            controller=self.controller,
            lease_id="session-lease",
            config=self.fixture.config,
            planning=self.fixture.planning,
            candidate=self.fixture.candidate,
            adapter=self.fixture.adapter,
            model=self.fixture.model,
            handoff=self.fixture.handoff,
            collection=self.fixture.collection,
            reservation_snapshot=reserve,
            launch_snapshot=launch,
        )

    def tearDown(self):
        # Session tests release their lease. This is defensive cleanup for failures.
        for lease in self.controller.active_leases():
            self.controller.release(lease.lease_id)
        self.fixture.tearDown()

    def session(self, backend=None):
        backend = backend or DeterministicBackend(
            self.admission.bundle.bundle_sha256
        )
        return NativeAdmittedSession(
            controller=self.controller,
            admission=self.admission,
            backend=backend,
        )

    def start_call(self, name, callback):
        outcomes = {}

        def call():
            try:
                outcomes["result"] = callback()
            except BaseException as exc:
                outcomes["error"] = exc

        thread = threading.Thread(target=call, name=name)
        thread.start()
        return thread, outcomes

    def assert_finished(self, thread):
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "fixture thread did not reach a terminal boundary")

    def test_completed_session_releases_launch_admitted_lease(self):
        session = self.session()
        result = session.run(b"seed")
        self.assertEqual(result.record["status"], "COMPLETED")
        self.assertTrue(result.record["inference_started"])
        self.assertFalse(result.record["real_model_inference"])
        self.assertTrue(result.record["released"])
        self.assertEqual(self.controller.active_leases(), ())
        self.assertEqual(session.state, "completed")
        self.assertEqual(session.release()["status"], "ALREADY_RELEASED")
        self.assertEqual(session.state, "released")

    def test_pre_run_cancel_releases_lease_without_backend_call(self):
        backend = DeterministicBackend(self.admission.bundle.bundle_sha256)
        session = self.session(backend)
        result = session.cancel()
        self.assertEqual(result["status"], "CANCELLED")
        self.assertEqual(backend.calls, [])
        self.assertEqual(self.controller.active_leases(), ())
        self.assertEqual(session.state, "cancelled")
        again = session.cancel()
        self.assertEqual(again["status"], "ALREADY_CANCELLED")
        self.assertTrue(again["cancelled"])
        self.assertTrue(again["lease_released"])

    def test_backend_failure_releases_lease(self):
        backend = DeterministicBackend(
            self.admission.bundle.bundle_sha256,
            fail=True,
        )
        session = self.session(backend)
        with self.assertRaises(ValidationError):
            session.run(b"seed")
        self.assertEqual(session.state, "failed")
        self.assertEqual(self.controller.active_leases(), ())

    def test_running_cancel_is_deferred_until_run_reaches_terminal_state(self):
        entered = threading.Event()
        proceed = threading.Event()
        backend = DeterministicBackend(
            self.admission.bundle.bundle_sha256,
            entered=entered,
            proceed=proceed,
        )
        session = self.session(backend)
        result_holder = {}
        error_holder = {}

        def run():
            try:
                result_holder["result"] = session.run(b"seed")
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(entered.wait(timeout=5))

        cancel = session.cancel()
        self.assertEqual(cancel["status"], "CANCEL_REQUESTED")
        self.assertFalse(cancel["lease_released"])
        self.assertEqual(len(self.controller.active_leases()), 1)

        with self.assertRaisesRegex(ValidationError, "running native session"):
            session.release()
        self.assertEqual(len(self.controller.active_leases()), 1)

        proceed.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", error_holder)
        self.assertEqual(self.controller.active_leases(), ())
        # A one-segment run can win the race with cancellation after the segment starts.
        self.assertIn(
            result_holder["result"].record["status"],
            {"COMPLETED", "CANCELLED"},
        )

    def test_pre_run_cancel_and_concurrent_run_are_serialized(self):
        self.assert_pre_run_control_serializes_run("cancel", "cancelled")

    def test_pre_run_release_and_concurrent_run_are_serialized(self):
        self.assert_pre_run_control_serializes_run("release", "released")

    def assert_pre_run_control_serializes_run(self, operation, terminal_state):
        backend = DeterministicBackend(self.admission.bundle.bundle_sha256)
        session = self.session(backend)
        control_entered = threading.Event()
        resume_control = threading.Event()
        probe = LockAcquisitionProbe(session._lock, "run-contender")
        session._lock = probe
        original_control = getattr(session._reference, operation)

        def paused_control():
            control_entered.set()
            if not resume_control.wait(timeout=5):
                raise RuntimeError("fixture control gate timed out")
            return original_control()

        runner = None
        with patch.object(session._reference, operation, paused_control):
            control, control_outcome = self.start_call(
                "control-owner", getattr(session, operation)
            )
            try:
                self.assertTrue(control_entered.wait(timeout=5))
                runner, run_outcome = self.start_call(
                    "run-contender", lambda: session.run(b"seed")
                )
                self.assertTrue(probe.attempted.wait(timeout=5))
                self.assertTrue(probe.contended)
                self.assertEqual(len(self.controller.active_leases()), 1)
                self.assertEqual(backend.calls, [])
            finally:
                resume_control.set()
                self.assert_finished(control)
                if runner is not None:
                    self.assert_finished(runner)

        self.assertNotIn("error", control_outcome)
        self.assertIsInstance(run_outcome.get("error"), ValidationError)
        self.assertEqual(session.state, terminal_state)
        self.assertEqual(self.controller.active_leases(), ())
        self.assertEqual(backend.calls, [])

    def test_cancel_after_run_owns_state_before_reference_start_is_terminal(self):
        backend = DeterministicBackend(self.admission.bundle.bundle_sha256)
        session = self.session(backend)
        run_owned = threading.Event()
        resume_run = threading.Event()
        original_run = session._reference.run

        def paused_run(payload):
            run_owned.set()
            if not resume_run.wait(timeout=5):
                raise RuntimeError("fixture run gate timed out")
            return original_run(payload)

        with patch.object(session._reference, "run", paused_run):
            runner, outcome = self.start_call("runner", lambda: session.run(b"seed"))
            try:
                self.assertTrue(run_owned.wait(timeout=5))
                cancel = session.cancel()
                self.assertEqual(cancel["status"], "CANCEL_REQUESTED")
                self.assertFalse(cancel["lease_released"])
                self.assertEqual(len(self.controller.active_leases()), 1)
                with self.assertRaises(ValidationError):
                    session.release()
            finally:
                resume_run.set()
                self.assert_finished(runner)

        self.assertNotIn("error", outcome)
        self.assertEqual(outcome["result"].record["status"], "CANCELLED")
        self.assertFalse(outcome["result"].record["inference_started"])
        self.assertEqual(outcome["result"].record["segments_executed"], [])
        self.assertIsNone(outcome["result"].output)
        self.assertEqual(session.state, "cancelled")
        self.assertEqual(backend.calls, [])
        self.assertEqual(self.controller.active_leases(), ())

    def test_cancel_after_completion_preserves_completed_outcome(self):
        session = self.session()
        session.run(b"seed")
        with patch.object(session._reference, "cancel", side_effect=AssertionError):
            cancel = session.cancel()
        self.assertEqual(cancel["status"], "ALREADY_COMPLETED")
        self.assertFalse(cancel["cancelled"])
        self.assertTrue(cancel["lease_released"])
        self.assertEqual(session.state, "completed")

    def test_cancel_after_failure_preserves_failed_outcome(self):
        session = self.session(DeterministicBackend(
            self.admission.bundle.bundle_sha256, fail=True
        ))
        with self.assertRaises(ValidationError):
            session.run(b"seed")
        with patch.object(session._reference, "cancel", side_effect=AssertionError):
            cancel = session.cancel()
        self.assertEqual(cancel["status"], "ALREADY_FAILED")
        self.assertFalse(cancel["cancelled"])
        self.assertTrue(cancel["lease_released"])
        self.assertEqual(session.state, "failed")

    def test_cancel_before_backend_failure_retains_lease_until_backend_raises(self):
        entered = threading.Event()
        proceed = threading.Event()
        session = self.session(DeterministicBackend(
            self.admission.bundle.bundle_sha256,
            fail=True, entered=entered, proceed=proceed,
        ))
        runner, outcome = self.start_call("runner", lambda: session.run(b"seed"))
        try:
            self.assertTrue(entered.wait(timeout=5))
            self.assertEqual(session.cancel()["status"], "CANCEL_REQUESTED")
            self.assertEqual(len(self.controller.active_leases()), 1)
            with self.assertRaises(ValidationError):
                session.release()
            self.assertEqual(len(self.controller.active_leases()), 1)
        finally:
            proceed.set()
            self.assert_finished(runner)
        self.assertIsInstance(outcome.get("error"), ValidationError)
        self.assertEqual(session.state, "failed")
        self.assertEqual(self.controller.active_leases(), ())

    def test_reference_release_rejection_does_not_release_lease(self):
        session = self.session()
        with patch.object(session._reference, "release", side_effect=ValidationError("busy")):
            with self.assertRaisesRegex(ValidationError, "busy"):
                session.release()
        self.assertEqual(session.state, "prepared")
        self.assertEqual(len(self.controller.active_leases()), 1)

    def test_concurrent_release_waits_for_terminal_run_cleanup(self):
        session = self.session()
        cleanup_entered = threading.Event()
        resume_cleanup = threading.Event()
        original_release = self.controller.release
        probe = LockAcquisitionProbe(session._lock, "release-contender")
        session._lock = probe

        def paused_release(lease_id):
            cleanup_entered.set()
            if not resume_cleanup.wait(timeout=5):
                raise RuntimeError("fixture cleanup gate timed out")
            return original_release(lease_id)

        releaser = None
        with patch.object(self.controller, "release", paused_release):
            runner, run_outcome = self.start_call("runner", lambda: session.run(b"seed"))
            try:
                self.assertTrue(cleanup_entered.wait(timeout=5))
                releaser, release_outcome = self.start_call("release-contender", session.release)
                self.assertTrue(probe.attempted.wait(timeout=5))
                self.assertTrue(probe.contended)
                self.assertEqual(len(self.controller.active_leases()), 1)
            finally:
                resume_cleanup.set()
                self.assert_finished(runner)
                if releaser is not None:
                    self.assert_finished(releaser)
        self.assertNotIn("error", run_outcome)
        self.assertNotIn("error", release_outcome)
        self.assertEqual(run_outcome["result"].record["status"], "COMPLETED")
        self.assertEqual(release_outcome["result"]["status"], "ALREADY_RELEASED")
        self.assertEqual(session.state, "released")
        self.assertEqual(self.controller.active_leases(), ())

    def test_session_rejects_backend_from_another_bundle(self):
        backend = DeterministicBackend("0" * 64)
        with self.assertRaises(ValidationError):
            self.session(backend)
        self.assertEqual(len(self.controller.active_leases()), 1)

    def test_session_rejects_missing_or_released_lease(self):
        lease_id = self.admission.record["lease_id"]
        self.controller.release(lease_id)
        with self.assertRaises(ValidationError):
            self.session()
        self.assertEqual(self.controller.active_leases(), ())


if __name__ == "__main__":
    unittest.main()
