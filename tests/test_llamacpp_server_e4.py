from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from tensormeld.llamacpp_admitted_server import bind_admitted_llamacpp_server
from tensormeld.llamacpp_managed_server import (
    ManagedLlamaCppServer,
    build_llamacpp_server_launch_spec,
)
from tensormeld.llamacpp_native_trial import run_llamacpp_native_trial
from tensormeld.llamacpp_package import build_llamacpp_package_identity
from tensormeld.llamacpp_placement import LlamaCppPlacementTranslation
from tensormeld.llamacpp_server_e4 import (
    LlamaCppServerE4Evidence,
    authorize_admitted_llamacpp_server_requests,
    build_llamacpp_server_e4_spec,
    evaluate_llamacpp_server_e4,
    run_llamacpp_server_e4_request,
)
from tensormeld.native_worker import approved_worker_artifact
from tensormeld.schema import ValidationError
from tensormeld.whole_block_execution import AcceptedExecutionBundle
from test_llamacpp_managed_server import FIXTURE, free_port
from test_llamacpp_native_trial import (
    FIXTURE_CLI,
    LlamaCppNativeTrialTests,
    sha256,
)
from test_llamacpp_package import AdmittedManagedLlamaCppServerTests, probe

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class LlamaCppServerE4FixtureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = LlamaCppNativeTrialTests(
            methodName="test_real_fixture_subprocess_exercises_closed_argv"
        )
        self.fixture.setUp()
        self.trial_spec = self.fixture.spec()
        device_id = self.fixture.candidate["compute_devices"][0]
        node_id = self.fixture.candidate["compute_nodes"][0]
        self.bundle = AcceptedExecutionBundle(
            config_sha256=self.fixture.config.fingerprint,
            profile=self.fixture.config.installation.default_profile,
            planning_input_sha256=self.fixture.planning.fingerprint,
            plan_sha256=self.fixture.candidate["plan_sha256"],
            representability_sha256="1" * 64,
            adapter_id=self.fixture.adapter.adapter_id,
            engine_revision=PIN,
            adapter_capabilities_sha256=self.fixture.adapter.fingerprint,
            model_manifest_sha256=self.fixture.model.manifest_sha256,
            qualification_evidence_sha256="2" * 64,
            runtime_manifest_sha256="3" * 64,
            worker_artifact_sha256=self.fixture.program.sha256,
            backend_readiness=((device_id, "4" * 64, "5" * 64),),
            launch_leases=((node_id, "fixture-lease", "6" * 64),),
            segments=((device_id, 0, len(self.fixture.placement.block_owners)),),
            unit_ids=tuple(
                f"blk.{index}"
                for index, _, _ in self.fixture.placement.block_owners
            ),
            compute_devices=(device_id,),
            compute_nodes=(node_id,),
            bundle_sha256="7" * 64,
        )
        self.execution_placement = LlamaCppPlacementTranslation(
            source_revision=PIN,
            accepted_bundle_sha256=self.bundle.bundle_sha256,
            placement_binding_sha256=self.fixture.placement.placement_binding_sha256,
            block_owners=self.fixture.placement.block_owners,
            device_buffer_types=self.fixture.placement.device_buffer_types,
            override_tensor_value=self.fixture.placement.override_tensor_value,
            argv_fragment=self.fixture.placement.argv_fragment,
            fingerprint="8" * 64,
        )
        self.server_artifact = approved_worker_artifact(
            FIXTURE,
            expected_sha256=sha256(FIXTURE),
            where="fixture llama-server",
        )
        self.package = build_llamacpp_package_identity(
            cli_probe=probe(self.fixture.program.sha256, "llama-cli"),
            server_probe=probe(self.server_artifact.sha256, "llama-server"),
            backend_libraries=[
                {"name": "ggml-fixture", "sha256": "9" * 64}
            ],
        )
        self.server_spec = build_llamacpp_server_launch_spec(
            bundle=self.bundle,
            placement=self.execution_placement,
            model=self.fixture.model,
            llama_server=self.server_artifact,
            launcher=self.fixture.launcher,
            gguf=self.fixture.gguf,
            context_tokens=self.trial_spec.context_tokens,
            port=free_port(),
        )
        self.e4_spec = build_llamacpp_server_e4_spec(
            package=self.package,
            trial_spec=self.trial_spec,
            qualification_placement=self.fixture.placement,
            server_spec=self.server_spec,
            execution_placement=self.execution_placement,
        )

    def tearDown(self):
        self.fixture.tearDown()

    def run_cli(self):
        def runner(argv, timeout):
            completed = subprocess.run(
                [
                    str(self.fixture.launcher.path),
                    str(self.fixture.program.path),
                    *argv[1:],
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                timeout=timeout,
                check=False,
            )
            return completed.returncode, completed.stdout, completed.stderr

        return run_llamacpp_native_trial(
            self.trial_spec,
            timeout_s=5,
            runner=runner,
            execution_source="fixture-subprocess",
        )

    def test_real_fixture_cli_and_server_outputs_are_exactly_equal(self):
        cli_result = self.run_cli()
        server = ManagedLlamaCppServer(self.server_spec)
        try:
            server.start()
            server.wait_ready(timeout_s=5, poll_s=0.05)
            server_result = run_llamacpp_server_e4_request(
                server=server,
                spec=self.e4_spec,
                timeout_s=5,
            )
        finally:
            server.stop(terminate_timeout_s=2, kill_timeout_s=2)

        evidence = evaluate_llamacpp_server_e4(
            spec=self.e4_spec,
            cli_trial_result=cli_result,
            server_result=server_result,
        )
        self.assertTrue(evidence.outputs_equal)
        self.assertFalse(evidence.qualified)
        self.assertEqual(evidence.cli_execution_source, "fixture-subprocess")
        self.assertEqual(
            evidence.server_execution_source,
            "fixture-server-subprocess",
        )
        self.assertEqual(
            cli_result["stdout_sha256"],
            server_result["content_sha256"],
        )
        with self.assertRaises(ValidationError):
            authorize_admitted_llamacpp_server_requests(
                binding=AdmittedManagedLlamaCppServerTests,
                evidence=evidence,
            )

    def test_output_mismatch_never_qualifies(self):
        cli_result = self.run_cli()
        server_result = {
            "e4_result_schema": "tensormeld/llamacpp-server-e4-result-v1",
            "e4_spec_sha256": self.e4_spec.fingerprint,
            "server_spec_sha256": self.server_spec.spec_sha256,
            "server_artifact_sha256": self.server_spec.server_artifact_sha256,
            "execution_source": "fixture-server-subprocess",
            "content_sha256": "0" * 64,
            "content_bytes": 1,
            "http_status": 200,
            "completed": True,
            "qualified": False,
            "real_model_inference": False,
            "result_sha256": "1" * 64,
        }
        evidence = evaluate_llamacpp_server_e4(
            spec=self.e4_spec,
            cli_trial_result=cli_result,
            server_result=server_result,
        )
        self.assertFalse(evidence.outputs_equal)
        self.assertFalse(evidence.qualified)

    def test_different_pre_post_placement_semantics_are_rejected(self):
        bad = replace(
            self.execution_placement,
            override_tensor_value=r"^blk\.0\..*=OTHER",
        )
        with self.assertRaises(ValidationError):
            build_llamacpp_server_e4_spec(
                package=self.package,
                trial_spec=self.trial_spec,
                qualification_placement=self.fixture.placement,
                server_spec=self.server_spec,
                execution_placement=bad,
            )


class LlamaCppServerE4AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = AdmittedManagedLlamaCppServerTests(
            methodName="test_same_build_package_bridges_qualified_cli_and_server_artifacts"
        )
        self.fixture.setUp()
        self.package = self.fixture.package()
        self.server_spec = self.fixture.spec()
        self.binding = bind_admitted_llamacpp_server(
            controller=self.fixture.controller,
            admission=self.fixture.admission,
            package=self.package,
            server_spec=self.server_spec,
        )

    def tearDown(self):
        self.fixture.tearDown()

    def native_evidence(self, **changes):
        output_sha = "a" * 64
        kwargs = {
            "package_sha256": self.binding.package_sha256,
            "model_manifest_sha256": self.fixture.admission.bundle.model_manifest_sha256,
            "accepted_bundle_sha256": self.binding.accepted_bundle_sha256,
            "qualification_placement_sha256": "b" * 64,
            "execution_placement_sha256": "c" * 64,
            "trial_spec_sha256": "d" * 64,
            "server_spec_sha256": self.binding.server_spec_sha256,
            "cli_output_sha256": output_sha,
            "server_output_sha256": output_sha,
            "cli_execution_source": "native-subprocess",
            "server_execution_source": "native-server-subprocess",
            "outputs_equal": True,
            "qualified": True,
        }
        kwargs.update(changes)
        provisional = LlamaCppServerE4Evidence(
            **kwargs,
            fingerprint="",
        )
        record = provisional.as_record()
        record.pop("fingerprint")
        return replace(
            provisional,
            fingerprint=canonical_sha(record),
        )

    def test_exact_native_e4_authorizes_persistent_requests(self):
        authorized = authorize_admitted_llamacpp_server_requests(
            binding=self.binding,
            evidence=self.native_evidence(),
        )
        record = authorized.as_record()
        self.assertTrue(record["server_semantic_equivalence_qualified"])
        self.assertTrue(record["inference_request_authorized"])
        self.assertFalse(record["real_model_inference"])

    def test_non_native_e4_cannot_authorize_requests(self):
        evidence = self.native_evidence(
            cli_execution_source="fixture-subprocess",
            server_execution_source="fixture-server-subprocess",
            qualified=False,
        )
        with self.assertRaises(ValidationError):
            authorize_admitted_llamacpp_server_requests(
                binding=self.binding,
                evidence=evidence,
            )

    def test_tampered_e4_fingerprint_is_rejected(self):
        evidence = replace(
            self.native_evidence(),
            fingerprint="0" * 64,
        )
        with self.assertRaises(ValidationError):
            authorize_admitted_llamacpp_server_requests(
                binding=self.binding,
                evidence=evidence,
            )

    def test_e4_from_another_bundle_is_rejected(self):
        evidence = self.native_evidence(
            accepted_bundle_sha256="0" * 64,
        )
        # Recompute its internal fingerprint so rejection is specifically bundle identity.
        record = evidence.as_record()
        record.pop("fingerprint")
        evidence = replace(evidence, fingerprint=canonical_sha(record))
        with self.assertRaises(ValidationError):
            authorize_admitted_llamacpp_server_requests(
                binding=self.binding,
                evidence=evidence,
            )


if __name__ == "__main__":
    unittest.main()
