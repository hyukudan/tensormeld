from __future__ import annotations

from dataclasses import replace
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensormeld.llamacpp_authorized_request import execute_authorized_llamacpp_completion
from tensormeld.llamacpp_c1_serving import AuthorizedC1LlamaCppServing
from tensormeld.schema import ValidationError
import test_llamacpp_authorized_request as fixture_module


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class AuthorizedC1ServingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.AuthorizedPersistentCompletionTests(
            methodName="test_exact_authorized_fixture_completion_runs_and_keeps_lease"
        )
        self.fixture.setUp()
        self.clock = ManualClock()
        self.body_bytes = len(json.dumps(self.fixture.spec.request_body, sort_keys=True,
                                        separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def tearDown(self):
        self.fixture.tearDown()

    def consumer(self, **changes):
        args = dict(controller=self.fixture.controller, admitted_server=self.fixture.server,
                    authorization=self.fixture.authorization, spec=self.fixture.spec, clock=self.clock)
        args.update(changes)
        return AuthorizedC1LlamaCppServing(**args)

    def start(self, callback):
        outcome = {}

        def call():
            try:
                outcome["result"] = callback()
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=call)
        thread.start()
        return thread, outcome

    def finish(self, thread):
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def blocked_client(self, entered, proceed, calls):
        def execute(**kwargs):
            calls.append(kwargs)
            entered.set()
            if not proceed.wait(timeout=5):
                raise RuntimeError("test client gate timed out")
            return execute_authorized_llamacpp_completion(**kwargs)
        return execute

    def test_real_owned_fixture_server_consumes_only_exact_authorized_replay(self):
        gate = self.consumer()
        token = gate.submit("request-one")
        result = gate.run(token)
        self.assertEqual(result.record["status"], "COMPLETED")
        self.assertEqual(result.completion.content, self.fixture.expected_content)
        self.assertTrue(result.record["backend_terminal"])
        self.assertFalse(result.record["server_lease_release_performed"])
        self.assertFalse(result.completion.record["real_model_inference"])
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)
        self.assertEqual(len(self.fixture.controller.active_leases()), 1)
        self.assertEqual(gate.drain(stop_server=True)["status"], "DRAINED")
        self.assertEqual(self.fixture.controller.active_leases(), ())

    def test_count_and_bytes_are_charged_until_active_client_is_terminal(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer(max_pending=1, max_owned_bytes=2 * self.body_bytes)
        first = gate.submit("first")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            thread, outcome = self.start(lambda: gate.run(first))
            try:
                self.assertTrue(entered.wait(timeout=5))
                second = gate.submit("second")
                with self.assertRaisesRegex(ValidationError, "count limit"):
                    gate.submit("third")
                self.assertEqual(gate.snapshot()["owned_bytes"], 2 * self.body_bytes)
                gate.cancel(second.request_id, second.attempt_id)
                self.assertEqual(gate.snapshot()["owned_bytes"], self.body_bytes)
            finally:
                proceed.set()
                self.finish(thread)
        self.assertNotIn("error", outcome)
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)

    def test_byte_limit_is_independent_of_pending_count_limit(self):
        gate = self.consumer(max_pending=2, max_owned_bytes=self.body_bytes)
        token = gate.submit("only")
        with self.assertRaisesRegex(ValidationError, "byte limit"):
            gate.submit("too-many-bytes")
        self.assertEqual(gate.snapshot()["pending"], 1)
        gate.cancel(token.request_id, token.attempt_id)
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)

    def test_two_consumers_cannot_own_one_admitted_worker(self):
        self.consumer()
        with self.assertRaisesRegex(ValidationError, "already has a C1 serving owner"):
            self.consumer()

    def test_fifo_runs_pending_owners_in_submission_order(self):
        entered, proceed = threading.Event(), threading.Event()
        call_threads = []
        gate = self.consumer()

        def execute(**kwargs):
            call_threads.append(threading.current_thread().name)
            entered.set()
            if not proceed.wait(timeout=5):
                raise RuntimeError("test client gate timed out")
            return execute_authorized_llamacpp_completion(**kwargs)

        first = gate.submit("first")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion", execute):
            t1, out1 = self.start(lambda: gate.run(first))
            t2 = t3 = None
            try:
                self.assertTrue(entered.wait(timeout=5))
                second, third = gate.submit("second"), gate.submit("third")
                t3, out3 = self.start(lambda: gate.run(third))
                t3.name = "third-owner"
                t2, out2 = self.start(lambda: gate.run(second))
                t2.name = "second-owner"
                self.assertEqual(len(call_threads), 1)
            finally:
                proceed.set()
                for thread in (t1, t2, t3):
                    if thread is not None:
                        self.finish(thread)
        self.assertEqual(call_threads[1:], ["second-owner", "third-owner"])
        for outcome in (out1, out2, out3):
            self.assertEqual(outcome["result"].record["status"], "COMPLETED")

    def test_interrupted_wait_owner_releases_its_pending_bytes(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer()
        first = gate.submit("active")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            t1, active = self.start(lambda: gate.run(first))
            try:
                self.assertTrue(entered.wait(timeout=5))
                waiting = gate.submit("interrupted")
                with patch.object(gate._condition, "wait", side_effect=InterruptedError):
                    with self.assertRaises(InterruptedError):
                        gate.run(waiting)
                self.assertEqual(gate.snapshot()["pending"], 0)
                self.assertEqual(gate.snapshot()["owned_bytes"], self.body_bytes)
                self.assertEqual(gate.cancel(waiting.request_id, waiting.attempt_id)["status"], "ALREADY_FAILED")
            finally:
                proceed.set()
                self.finish(t1)
        self.assertEqual(active["result"].record["status"], "COMPLETED")

    def test_active_cancel_ack_keeps_ownership_and_suppresses_output_until_return(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer()
        token = gate.submit("cancel-active")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            thread, outcome = self.start(lambda: gate.run(token))
            try:
                self.assertTrue(entered.wait(timeout=5))
                ack = gate.cancel(token.request_id, token.attempt_id)
                self.assertEqual(ack["status"], "CANCEL_REQUESTED")
                self.assertFalse(ack["request_ownership_released"])
                self.assertTrue(gate.snapshot()["active"])
                self.assertEqual(gate.snapshot()["owned_bytes"], self.body_bytes)
                self.assertEqual(len(self.fixture.controller.active_leases()), 1)
            finally:
                proceed.set()
                self.finish(thread)
        self.assertEqual(outcome["result"].record["status"], "CANCELLED")
        self.assertIsNone(outcome["result"].completion)
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)
        self.assertEqual(len(self.fixture.controller.active_leases()), 1)

    def test_fifo_c1_dispatch_and_queued_cancel_never_calls_client(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer()
        first = gate.submit("first")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            t1, first_out = self.start(lambda: gate.run(first))
            t2 = t3 = None
            try:
                self.assertTrue(entered.wait(timeout=5))
                second = gate.submit("second")
                third = gate.submit("third")
                t3, third_out = self.start(lambda: gate.run(third))
                self.assertEqual(gate.cancel(second.request_id, second.attempt_id)["status"], "CANCELLED")
                second_result = gate.run(second)
                self.assertEqual(second_result.record["status"], "CANCELLED")
                self.assertEqual(len(calls), 1)
            finally:
                proceed.set()
                self.finish(t1)
                if t3 is not None:
                    self.finish(t3)
        self.assertEqual(first_out["result"].record["status"], "COMPLETED")
        self.assertEqual(third_out["result"].record["status"], "COMPLETED")
        self.assertEqual(len(calls), 2)
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)

    def test_queued_deadline_and_abandoned_submission_release_without_dispatch(self):
        gate = self.consumer()
        token = gate.submit("expires", timeout_s=1)
        self.clock.now = 2
        self.assertEqual(gate.snapshot()["pending"], 0)
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion") as client:
            self.assertEqual(gate.run(token).record["status"], "DEADLINE_EXCEEDED")
            client.assert_not_called()

    def test_deadline_expiring_in_launch_validation_never_dispatches(self):
        gate = self.consumer()
        token = gate.submit("expires-during-validation", timeout_s=1)
        original_validate = gate._validate

        def validate():
            original_validate()
            self.clock.now = 2

        with patch.object(gate, "_validate", validate):
            with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion") as client:
                result = gate.run(token)
                client.assert_not_called()
        self.assertEqual(result.record["status"], "DEADLINE_EXCEEDED")
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)

    def test_deadline_during_active_client_keeps_ownership_and_discards_late_output(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer()
        token = gate.submit("late", timeout_s=1)
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            thread, outcome = self.start(lambda: gate.run(token))
            try:
                self.assertTrue(entered.wait(timeout=5))
                self.clock.now = 2
                self.assertTrue(gate.snapshot()["active"])
                self.assertEqual(len(self.fixture.controller.active_leases()), 1)
            finally:
                proceed.set()
                self.finish(thread)
        self.assertEqual(outcome["result"].record["status"], "DEADLINE_EXCEEDED")
        self.assertIsNone(outcome["result"].completion)

    def test_transport_timeout_fences_gate_and_holds_owner_until_confirmed_stop(self):
        gate = self.consumer()
        token = gate.submit("uncertain")
        queued = gate.submit("queued")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   side_effect=TimeoutError("fixture request timeout")):
            result = gate.run(token)
        self.assertEqual(result.record["status"], "FAILED")
        self.assertEqual(result.record["reason_code"], "AUTHORIZED_COMPLETION_FAILED")
        self.assertFalse(result.record["backend_terminal"])
        self.assertFalse(result.record["request_ownership_released"])
        self.assertEqual(gate.run(queued).record["status"], "WORKER_REVOKED")
        self.assertTrue(gate.snapshot()["fenced"])
        self.assertTrue(gate.snapshot()["active"])
        self.assertEqual(gate.snapshot()["owned_bytes"], self.body_bytes)
        self.assertEqual(len(self.fixture.controller.active_leases()), 1)
        with self.assertRaises(ValidationError):
            gate.submit("after-timeout")
        self.assertEqual(gate.drain(stop_server=True)["status"], "DRAINED")
        self.assertFalse(gate.snapshot()["active"])
        self.assertEqual(self.fixture.controller.active_leases(), ())

    def test_dead_worker_snapshot_fences_and_releases_pending_without_dispatch(self):
        gate = self.consumer()
        first, second = gate.submit("first"), gate.submit("second")
        process = self.fixture.server.server._child
        process.terminate()
        process.wait(timeout=5)
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion") as client:
            state = gate.snapshot()
            self.assertFalse(state["accepting"])
            self.assertTrue(state["fenced"])
            self.assertEqual(state["fence_reason"], "WORKER_PROCESS_EXITED")
            self.assertEqual(state["pending"], 0)
            self.assertEqual(state["owned_bytes"], 0)
            for token in (first, second):
                result = gate.run(token)
                self.assertEqual(result.record["status"], "WORKER_REVOKED")
                self.assertEqual(result.record["reason_code"], "WORKER_PROCESS_EXITED")
            with self.assertRaises(ValidationError):
                gate.submit("dead-worker")
            client.assert_not_called()
        self.assertEqual(gate.drain(stop_server=True)["status"], "DRAINED")
        self.assertEqual(self.fixture.controller.active_leases(), ())

    def test_process_exit_does_not_drop_an_owner_still_inside_client(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer()
        token = gate.submit("active")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            thread, outcome = self.start(lambda: gate.run(token))
            try:
                self.assertTrue(entered.wait(timeout=5))
                waiting = gate.submit("waiting")
                process = self.fixture.server.server._child
                process.terminate()
                process.wait(timeout=5)
                state = gate.snapshot()
                self.assertTrue(state["active"])
                self.assertEqual(state["owned_bytes"], self.body_bytes)
                self.assertEqual(gate.run(waiting).record["status"], "WORKER_REVOKED")
            finally:
                proceed.set()
                self.finish(thread)
        self.assertEqual(outcome["result"].record["status"], "FAILED")
        self.assertTrue(outcome["result"].record["backend_terminal"])
        self.assertFalse(gate.snapshot()["active"])
        self.assertEqual(gate.snapshot()["owned_bytes"], 0)
        gate.drain(stop_server=True)
        self.assertEqual(self.fixture.controller.active_leases(), ())

    def test_same_pid_replacement_cannot_publish_old_result_or_receive_old_drain(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer()
        token = gate.submit("old-epoch")
        worker = self.fixture.server.server
        original = worker._child
        replacement = SimpleNamespace(pid=original.pid, poll=lambda: None)
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            thread, outcome = self.start(lambda: gate.run(token))
            try:
                self.assertTrue(entered.wait(timeout=5))
                worker._child = replacement
                with self.assertRaisesRegex(ValidationError, "replacement worker"):
                    gate.drain(stop_server=True)
                proceed.set()
                self.finish(thread)
                self.assertEqual(outcome["result"].record["status"], "WORKER_REVOKED")
                self.assertIsNone(outcome["result"].completion)
            finally:
                proceed.set()
                self.finish(thread)
                worker._child = original

    def test_drain_rejects_pending_and_new_work_but_waits_for_active_boundary(self):
        entered, proceed, calls = threading.Event(), threading.Event(), []
        gate = self.consumer()
        first = gate.submit("active")
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion",
                   self.blocked_client(entered, proceed, calls)):
            thread, outcome = self.start(lambda: gate.run(first))
            try:
                self.assertTrue(entered.wait(timeout=5))
                queued = gate.submit("drained")
                # A controlled clock makes drain timeout independent of wall-clock scheduling.
                self.clock.now = 10
                def tick():
                    self.clock.now += 1
                    return self.clock.now
                gate._clock = tick
                drained = gate.drain(timeout_s=0.1)
                self.assertEqual(drained["status"], "DRAIN_TIMEOUT")
                self.assertTrue(drained["active"])
                self.assertEqual(gate.run(queued).record["status"], "DRAINED")
                with self.assertRaises(ValidationError):
                    gate.submit("rejected")
                self.assertEqual(len(self.fixture.controller.active_leases()), 1)
            finally:
                proceed.set()
                self.finish(thread)
        self.assertEqual(outcome["result"].record["status"], "COMPLETED")
        self.assertEqual(gate.drain()["status"], "DRAINED")

    def test_cancel_is_attempt_specific_and_duplicate_run_owner_rejected(self):
        gate = self.consumer()
        old = gate.submit("same-request")
        gate.run(old)
        newer = gate.submit("same-request")
        self.assertNotEqual(old.attempt_id, newer.attempt_id)
        self.assertEqual(gate.cancel(old.request_id, old.attempt_id)["status"], "ALREADY_COMPLETED")
        self.assertEqual(gate.snapshot()["pending"], 1)
        self.assertEqual(gate.cancel("wrong-request", newer.attempt_id)["status"], "UNKNOWN_ATTEMPT")
        with self.assertRaisesRegex(ValidationError, "issued attempt token"):
            gate.run(replace(newer))
        with self.assertRaisesRegex(ValidationError, "run owner"):
            gate.run(old)
        self.assertEqual(gate.run(newer).record["status"], "COMPLETED")

    def test_invalid_authorization_and_byte_limits_fail_before_request(self):
        with patch("tensormeld.llamacpp_c1_serving.execute_authorized_llamacpp_completion") as client:
            with self.assertRaises(ValidationError):
                self.consumer(authorization=replace(self.fixture.authorization, fingerprint="f" * 64))
            with self.assertRaisesRegex(ValidationError, "byte limits"):
                self.consumer(max_request_bytes=self.body_bytes - 1)
            for value in (True, 0, -1, 129):
                with self.subTest(value=value), self.assertRaises(ValidationError):
                    self.consumer(max_pending=value)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
