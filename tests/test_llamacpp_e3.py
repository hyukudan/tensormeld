from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from tensormeld.llamacpp_e3 import (
    LlamaCppE3Reference,
    evaluate_llamacpp_native_e3,
)
from tensormeld.llamacpp_native_trial import build_llamacpp_native_trial_spec
from tensormeld.native_worker import approved_worker_artifact
from tensormeld.qualification import evidence_applies
from tensormeld.runtime_identity import RuntimeIdentity
from tensormeld.schema import ValidationError
from test_llamacpp_native_trial import (
    FIXTURE_CLI,
    make_model,
    qualification_context,
    sha256,
)
from tensormeld.llamacpp_native_trial import approve_single_file_gguf


def runtime_identity(device_id: str, worker_sha: str, *, driver_version: str = "1"):
    return RuntimeIdentity.parse({
        "runtime_identity_schema": "tensormeld/runtime-identity-v1",
        "node_id": "n0",
        "worker_artifact_sha256": worker_sha,
        "worker_build_id": "llama-cli-fixture",
        "os_name": "fixture-os",
        "os_version": "1",
        "driver_id": "fixture-driver",
        "driver_version": driver_version,
        "runtime_id": "fixture-runtime",
        "runtime_version": "1",
        "tensormeld_device_id": device_id,
        "physical_device_id": f"fixture:{device_id}",
        "topology_sha256": "a" * 64,
    })


class LlamaCppNativeE3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gguf_path = self.root / "fixture.gguf"
        self.gguf_path.write_bytes(b"GGUF" + b"x" * 32)
        self.model = make_model(self.gguf_path)
        (
            self.config,
            self.planning,
            self.candidate,
            self.adapter,
            self.placement,
        ) = qualification_context(self.model)
        self.program = approved_worker_artifact(
            FIXTURE_CLI,
            expected_sha256=sha256(FIXTURE_CLI),
            where="fixture llama cli",
        )
        self.gguf = approve_single_file_gguf(self.model, self.gguf_path)
        self.spec = build_llamacpp_native_trial_spec(
            model=self.model,
            placement=self.placement,
            llama_cli=self.program,
            gguf=self.gguf,
            prompt="TensorMeld trial",
            context_tokens=self.planning.context_tokens,
            predict_tokens=1,
        )
        self.stdout = b"deterministic native completion"
        self.trial = {
            "trial_schema": "tensormeld/llamacpp-native-trial-v1",
            "source_revision": self.spec.source_revision,
            "spec_sha256": self.spec.spec_sha256,
            "llama_cli_sha256": self.spec.llama_cli_sha256,
            "model_manifest_sha256": self.spec.model_manifest_sha256,
            "gguf_sha256": self.spec.gguf_sha256,
            "config_sha256": self.spec.config_sha256,
            "planning_input_sha256": self.spec.planning_input_sha256,
            "candidate_plan_sha256": self.spec.candidate_plan_sha256,
            "placement_sha256": self.spec.placement_sha256,
            "execution_source": "native-subprocess",
            "exit_code": 0,
            "stdout_sha256": hashlib.sha256(self.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_bytes": len(self.stdout),
            "stderr_bytes": 0,
            "process_succeeded": True,
            "stdout_nonempty": True,
            "evidence_level": "E2.5",
            "qualified": False,
            "real_model_inference": False,
            "executable": False,
            "warnings": [],
        }
        self.reference = LlamaCppE3Reference.parse({
            "reference_schema": "tensormeld/llamacpp-e3-reference-v1",
            "reference_id": "fixture-reference",
            "trial_spec_sha256": self.spec.spec_sha256,
            "llama_cli_sha256": self.spec.llama_cli_sha256,
            "model_manifest_sha256": self.model.manifest_sha256,
            "placement_sha256": self.placement.fingerprint,
            "expected_stdout_sha256": self.trial["stdout_sha256"],
            "workload": {
                "context_tokens": self.spec.context_tokens,
                "max_output_tokens": self.spec.predict_tokens,
                "concurrency": 1,
            },
        })
        device_ids = tuple(dict.fromkeys(d for _, d, _ in self.placement.block_owners))
        self.identities = tuple(
            runtime_identity(device, self.spec.llama_cli_sha256)
            for device in device_ids
        )

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self, **changes):
        kwargs = {
            "adapter": self.adapter,
            "model": self.model,
            "placement": self.placement,
            "spec": self.spec,
            "trial_result": self.trial,
            "reference": self.reference,
            "runtime_identities": self.identities,
            "observed_at": "2026-10-03T19:00:00Z",
        }
        kwargs.update(changes)
        return evaluate_llamacpp_native_e3(**kwargs)

    def test_exact_native_trial_emits_e3_v2(self):
        evaluation, evidence = self.evaluate()
        self.assertTrue(evaluation["correct"])
        self.assertTrue(evaluation["qualified"])
        self.assertTrue(evaluation["real_model_inference"])
        self.assertFalse(evaluation["executable"])
        self.assertEqual(evidence.level, "E3")
        self.assertEqual(
            evidence.candidate_plan_sha256,
            self.candidate["plan_sha256"],
        )
        self.assertEqual(evidence.placement_sha256, self.placement.fingerprint)
        self.assertEqual(evidence.trial_spec_sha256, self.spec.spec_sha256)
        self.assertEqual(
            evidence.correctness_contract_sha256,
            self.reference.fingerprint,
        )
        self.assertEqual(
            evidence.runtime_identity_sha256,
            tuple(i.identity_sha256 for i in self.identities),
        )

    def test_emitted_e3_applies_only_to_exact_plan_and_runtime(self):
        _, evidence = self.evaluate()
        result = evidence_applies(
            evidence,
            adapter=self.adapter,
            model=self.model,
            config_sha256=self.config.fingerprint,
            device_ids=list(evidence.device_ids),
            context_tokens=self.spec.context_tokens,
            max_output_tokens=self.spec.predict_tokens,
            concurrency=1,
            minimum_level="E3",
            candidate_plan_sha256=self.candidate["plan_sha256"],
            runtime_identity_sha256=[
                identity.identity_sha256 for identity in self.identities
            ],
            require_v2=True,
        )
        self.assertTrue(result["applies"])

        stale_plan = evidence_applies(
            evidence,
            adapter=self.adapter,
            model=self.model,
            config_sha256=self.config.fingerprint,
            device_ids=list(evidence.device_ids),
            context_tokens=self.spec.context_tokens,
            max_output_tokens=self.spec.predict_tokens,
            concurrency=1,
            minimum_level="E3",
            candidate_plan_sha256="f" * 64,
            runtime_identity_sha256=[
                identity.identity_sha256 for identity in self.identities
            ],
            require_v2=True,
        )
        self.assertFalse(stale_plan["applies"])
        self.assertIn("PLAN_ID_MISMATCH", stale_plan["reasons"])

    def test_fixture_or_injected_trial_cannot_emit_e3(self):
        for source in ("fixture-subprocess", "injected-runner"):
            trial = dict(self.trial)
            trial["execution_source"] = source
            trial["evidence_level"] = "fixture"
            with self.subTest(source=source), self.assertRaises(ValidationError):
                self.evaluate(trial_result=trial)

    def test_wrong_stdout_reference_is_rejected(self):
        reference = LlamaCppE3Reference.parse({
            "reference_schema": "tensormeld/llamacpp-e3-reference-v1",
            "reference_id": "wrong-output",
            "trial_spec_sha256": self.spec.spec_sha256,
            "llama_cli_sha256": self.spec.llama_cli_sha256,
            "model_manifest_sha256": self.model.manifest_sha256,
            "placement_sha256": self.placement.fingerprint,
            "expected_stdout_sha256": "0" * 64,
            "workload": {
                "context_tokens": self.spec.context_tokens,
                "max_output_tokens": self.spec.predict_tokens,
                "concurrency": 1,
            },
        })
        with self.assertRaises(ValidationError):
            self.evaluate(reference=reference)

    def test_runtime_identity_worker_or_device_change_is_rejected(self):
        device = self.identities[0].tensormeld_device_id
        wrong_worker = runtime_identity(device, "0" * 64)
        with self.assertRaises(ValidationError):
            self.evaluate(runtime_identities=(wrong_worker,))

        wrong_device = runtime_identity("other", self.spec.llama_cli_sha256)
        with self.assertRaises(ValidationError):
            self.evaluate(runtime_identities=(wrong_device,))

    def test_trial_plan_or_placement_tampering_is_rejected(self):
        for field, value in (
            ("candidate_plan_sha256", "0" * 64),
            ("placement_sha256", "1" * 64),
            ("spec_sha256", "2" * 64),
        ):
            trial = dict(self.trial)
            trial[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.evaluate(trial_result=trial)

    def test_reference_workload_must_equal_trial_workload(self):
        reference = LlamaCppE3Reference.parse({
            "reference_schema": "tensormeld/llamacpp-e3-reference-v1",
            "reference_id": "wrong-workload",
            "trial_spec_sha256": self.spec.spec_sha256,
            "llama_cli_sha256": self.spec.llama_cli_sha256,
            "model_manifest_sha256": self.model.manifest_sha256,
            "placement_sha256": self.placement.fingerprint,
            "expected_stdout_sha256": self.trial["stdout_sha256"],
            "workload": {
                "context_tokens": self.spec.context_tokens,
                "max_output_tokens": 2,
                "concurrency": 1,
            },
        })
        with self.assertRaises(ValidationError):
            self.evaluate(reference=reference)


if __name__ == "__main__":
    unittest.main()
