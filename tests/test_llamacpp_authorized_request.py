from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from tensormeld.llamacpp_admitted_server import AdmittedManagedLlamaCppServer
from tensormeld.llamacpp_authorized_request import (
    execute_authorized_llamacpp_completion,
)
from tensormeld.llamacpp_server_e4 import (
    authorize_admitted_llamacpp_server_requests,
)
from tensormeld.schema import ValidationError
from test_llamacpp_server_e4 import LlamaCppServerE4AuthorizationTests


class AuthorizedPersistentCompletionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = LlamaCppServerE4AuthorizationTests(
            methodName="test_exact_native_e4_authorizes_persistent_requests"
        )
        self.fixture.setUp()
        self.spec = self.fixture.authorization_spec()

        model_path = self.fixture.server_spec.argv[
            self.fixture.server_spec.argv.index("-m") + 1
        ]
        override = self.fixture.server_spec.argv[
            self.fixture.server_spec.argv.index("--override-tensor") + 1
        ]
        prompt = self.spec.request_body["prompt"]
        digest = hashlib.sha256(
            (prompt + "|" + model_path + "|" + override).encode("utf-8")
        ).hexdigest()[:16]
        self.expected_content = "fixture-completion:" + digest
        self.expected_sha = hashlib.sha256(
            self.expected_content.encode("utf-8")
        ).hexdigest()

        evidence = self.fixture.native_evidence(
            self.spec,
            cli_output_sha256=self.expected_sha,
            server_output_sha256=self.expected_sha,
        )
        self.authorization = authorize_admitted_llamacpp_server_requests(
            binding=self.fixture.binding,
            spec=self.spec,
            evidence=evidence,
        )
        self.server = AdmittedManagedLlamaCppServer(
            controller=self.fixture.fixture.controller,
            admission=self.fixture.fixture.admission,
            package=self.fixture.package,
            server_spec=self.fixture.server_spec,
        )
        self.server.start()
        self.server.wait_ready(timeout_s=5, poll_s=0.05)

    def tearDown(self):
        try:
            if self.server.state not in {"stopped", "failed"}:
                self.server.stop(terminate_timeout_s=2, kill_timeout_s=2)
        except Exception:
            pass
        self.fixture.tearDown()

    @property
    def controller(self):
        return self.fixture.fixture.controller

    def test_exact_authorized_fixture_completion_runs_and_keeps_lease(self):
        result = execute_authorized_llamacpp_completion(
            controller=self.controller,
            admitted_server=self.server,
            authorization=self.authorization,
            spec=self.spec,
            timeout_s=5,
        )
        self.assertEqual(result.content, self.expected_content)
        self.assertEqual(result.record["content_sha256"], self.expected_sha)
        self.assertEqual(
            result.record["expected_output_sha256"],
            self.expected_sha,
        )
        self.assertTrue(result.record["completed"])
        self.assertTrue(result.record["inference_request_authorized"])
        self.assertFalse(result.record["real_model_inference"])
        self.assertEqual(
            result.record["execution_source"],
            "fixture-server-subprocess",
        )
        self.assertTrue(result.record["lease_remains_active"])
        self.assertEqual(len(self.controller.active_leases()), 1)

        stopped = self.server.stop(terminate_timeout_s=2, kill_timeout_s=2)
        self.assertEqual(stopped["lease_release_status"], "RELEASED")
        self.assertEqual(self.controller.active_leases(), ())

    def test_output_drift_from_e4_reference_is_rejected_but_live_lease_remains(self):
        wrong = replace(
            self.authorization,
            expected_output_sha256="0" * 64,
        )
        # Recompute the authorized-binding fingerprint so the failure is specifically
        # output drift, not object tampering.
        record = wrong.as_record()
        record.pop("fingerprint")
        from tensormeld.llamacpp_server_e4 import _canonical_sha256
        wrong = replace(wrong, fingerprint=_canonical_sha256(record))

        with self.assertRaises(ValidationError) as cm:
            execute_authorized_llamacpp_completion(
                controller=self.controller,
                admitted_server=self.server,
                authorization=wrong,
                spec=self.spec,
                timeout_s=5,
            )
        self.assertIn("output drifted", str(cm.exception))
        self.assertTrue(self.server.server.is_alive)
        self.assertEqual(len(self.controller.active_leases()), 1)

    def test_released_lease_blocks_request_even_if_server_is_still_ready(self):
        self.controller.release(self.server.binding.lease_id)
        with self.assertRaises(ValidationError):
            execute_authorized_llamacpp_completion(
                controller=self.controller,
                admitted_server=self.server,
                authorization=self.authorization,
                spec=self.spec,
                timeout_s=5,
            )

    def test_stopped_server_cannot_receive_authorized_request(self):
        self.server.stop(terminate_timeout_s=2, kill_timeout_s=2)
        with self.assertRaises(ValidationError):
            execute_authorized_llamacpp_completion(
                controller=self.controller,
                admitted_server=self.server,
                authorization=self.authorization,
                spec=self.spec,
                timeout_s=5,
            )

    def test_tampered_authorized_binding_is_rejected_before_http(self):
        bad = replace(
            self.authorization,
            fingerprint="0" * 64,
        )
        with self.assertRaises(ValidationError):
            execute_authorized_llamacpp_completion(
                controller=self.controller,
                admitted_server=self.server,
                authorization=bad,
                spec=self.spec,
                timeout_s=5,
            )

    def test_another_e4_spec_cannot_reuse_authorization(self):
        bad_spec = replace(
            self.spec,
            prompt="different",
        )
        with self.assertRaises(ValidationError):
            execute_authorized_llamacpp_completion(
                controller=self.controller,
                admitted_server=self.server,
                authorization=self.authorization,
                spec=bad_spec,
                timeout_s=5,
            )


if __name__ == "__main__":
    unittest.main()
