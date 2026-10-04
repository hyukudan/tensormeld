from __future__ import annotations

import threading
import unittest

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
            self.proceed.wait(timeout=5)
        if self.fail:
            raise ValidationError("fixture backend failure")
        return payload + ("|" + device_id + ":" + ",".join(unit_ids)).encode()


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
