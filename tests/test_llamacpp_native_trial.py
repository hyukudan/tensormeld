from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.config_v2 import Config
from tensormeld.llamacpp_native_trial import (
    approve_single_file_gguf,
    build_llamacpp_native_trial_spec,
    run_llamacpp_native_trial,
)
from tensormeld.llamacpp_placement import (
    LlamaCppPlacementBinding,
    translate_candidate_for_llamacpp_qualification,
)
from tensormeld.model_manifest import ModelManifest
from tensormeld.native_worker import approved_worker_artifact
from tensormeld.planner_v2 import plan_v2
from tensormeld.planning_contract import PlanningInput
from tensormeld.schema import ValidationError
from test_planner_v2 import fixture

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"
FIXTURE_CLI = Path(__file__).parent / "fixtures" / "llamacpp_cli_fixture.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def make_model(path: Path) -> ModelManifest:
    return ModelManifest.parse({
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/native-trial",
        "revision": "fixture-rev",
        "format": "gguf",
        "architecture": "llama",
        "tokenizer_ref": "fixture-tokenizer",
        "chat_template_ref": None,
        "files": [{
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }],
        "tensor_index_sha256": "1" * 64,
        "tensor_count": 2,
        "tensor_payload_bytes": 1,
    })


def qualification_context(model: ModelManifest):
    raw_config, raw_planning = fixture(count=1, units=1)
    raw_config["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
    raw_planning["manifest_ref"] = model.manifest_sha256
    raw_planning["units"][0]["id"] = "blk.0"

    config = Config.parse(raw_config)
    planning = PlanningInput.parse(raw_planning, config)
    candidate = plan_v2(config, planning)["best"]
    assert candidate is not None

    device = config.devices[0]
    adapter = AdapterCapabilities.parse({
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "llamacpp-native",
        "engine": "llama.cpp",
        "engine_revision": PIN,
        "placement": {
            "strategies": ["whole_blocks"],
            "exact_owner_binding": True,
            "explicit_unit_ranges": True,
            "mixed_backends": True,
            "remote_compute": False,
            "coordinator_outside_compute": True,
            "max_compute_devices": 4,
            "max_compute_nodes": 1,
            "max_segments": 8,
        },
        "route_modes": ["direct"],
        "coordinator_nodes": [config.nodes[0].id],
        "devices": [{
            "id": device.id,
            "node": device.node,
            "backend": device.backend,
        }],
    })
    binding = LlamaCppPlacementBinding.parse({
        "placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "source_revision": PIN,
        "native_binding_sha256": "d" * 64,
        "devices": [{
            "device_id": device.id,
            "engine_device_name": "CPU",
            "buffer_type": "CPU",
        }],
    }, adapter=adapter)
    bound = {
        "result_schema": "tensormeld/llamacpp-device-binding-result-v1",
        "config_sha256": config.fingerprint,
        "binding_sha256": "d" * 64,
        "resolved_mappings": [{
            "tensormeld_device_id": device.id,
            "engine_device_name": "CPU",
            "backend_from_config": device.backend,
        }],
    }
    index = {
        "index_schema": "tensormeld/gguf-index-v1",
        "architecture": "llama",
        "complete_shard_set": True,
        "tensor_count": 2,
        "index_sha256": model.tensor_index_sha256,
        "files": [{"tensors": [
            {"name": "blk.0.attn_q.weight"},
            {"name": "blk.0.ffn_up.weight"},
        ]}],
    }
    placement = translate_candidate_for_llamacpp_qualification(
        config,
        planning,
        candidate,
        adapter=adapter,
        binding=binding,
        bound_result=bound,
        model=model,
        gguf_index=index,
    )
    return config, planning, candidate, adapter, placement


class LlamaCppNativeTrialTests(unittest.TestCase):
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
        self.launcher = approved_worker_artifact(
            sys.executable,
            expected_sha256=sha256(Path(sys.executable)),
            where="python launcher",
        )
        self.program = approved_worker_artifact(
            FIXTURE_CLI,
            expected_sha256=sha256(FIXTURE_CLI),
            where="fixture llama cli",
        )
        self.gguf = approve_single_file_gguf(self.model, self.gguf_path)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self):
        return build_llamacpp_native_trial_spec(
            model=self.model,
            placement=self.placement,
            llama_cli=self.program,
            gguf=self.gguf,
            prompt="TensorMeld trial",
            context_tokens=self.planning.context_tokens,
            predict_tokens=1,
        )

    def test_pre_e3_candidate_bootstraps_native_trial_without_execution_bundle(self):
        spec = self.spec()
        self.assertEqual(spec.config_sha256, self.config.fingerprint)
        self.assertEqual(spec.planning_input_sha256, self.planning.fingerprint)
        self.assertEqual(spec.candidate_plan_sha256, self.candidate["plan_sha256"])
        self.assertFalse(self.placement.as_record()["qualified"])
        self.assertFalse(self.placement.as_record()["real_model_inference"])

    def test_closed_argv_contains_exact_model_and_placement(self):
        spec = self.spec()
        self.assertEqual(spec.argv[0], str(FIXTURE_CLI.resolve()))
        self.assertEqual(spec.argv[1:3], ("-m", str(self.gguf_path.resolve())))
        self.assertIn("--override-tensor", spec.argv)
        self.assertIn("^blk\\.0\\..*=CPU", spec.argv)
        self.assertIn("--single-turn", spec.argv)
        self.assertIn("--simple-io", spec.argv)
        self.assertNotIn("--rpc", spec.argv)

    def test_real_fixture_subprocess_exercises_closed_argv(self):
        spec = self.spec()

        def runner(argv, timeout):
            import subprocess
            result = subprocess.run(
                [str(self.launcher.path), str(self.program.path), *argv[1:]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                timeout=timeout,
                check=False,
            )
            return result.returncode, result.stdout, result.stderr

        result = run_llamacpp_native_trial(
            spec,
            timeout_s=5,
            runner=runner,
            execution_source="fixture-subprocess",
        )
        self.assertTrue(result["process_succeeded"])
        self.assertTrue(result["stdout_nonempty"])
        self.assertEqual(result["evidence_level"], "fixture")
        self.assertFalse(result["qualified"])
        self.assertFalse(result["real_model_inference"])

    def test_injected_runner_cannot_impersonate_native_trial(self):
        spec = self.spec()

        def runner(argv, timeout):
            return 0, b"some-output", b""

        with self.assertRaises(ValidationError):
            run_llamacpp_native_trial(
                spec,
                runner=runner,
                execution_source="native-subprocess",
            )

    def test_gguf_hash_and_size_are_verified_before_launch(self):
        self.gguf_path.write_bytes(b"GGUFtampered")
        with self.assertRaises(ValidationError):
            approve_single_file_gguf(self.model, self.gguf_path)

    def test_multi_file_manifest_is_rejected_by_initial_trial(self):
        raw = {
            "model_manifest_schema": "tensormeld/model-manifest-v1",
            "model_id": "fixture/split",
            "revision": "r",
            "format": "gguf",
            "architecture": "llama",
            "tokenizer_ref": "tok",
            "chat_template_ref": None,
            "files": [
                {"name": "a.gguf", "size_bytes": 1, "sha256": "1" * 64},
                {"name": "b.gguf", "size_bytes": 1, "sha256": "2" * 64},
            ],
            "tensor_index_sha256": "3" * 64,
            "tensor_count": 1,
            "tensor_payload_bytes": 1,
        }
        with self.assertRaises(ValidationError):
            approve_single_file_gguf(ModelManifest.parse(raw), self.gguf_path)

    def test_prompt_and_token_bounds_fail_closed(self):
        with self.assertRaises(ValidationError):
            build_llamacpp_native_trial_spec(
                model=self.model,
                placement=self.placement,
                llama_cli=self.program,
                gguf=self.gguf,
                prompt="x" * 4097,
                context_tokens=self.planning.context_tokens,
            )
        with self.assertRaises(ValidationError):
            build_llamacpp_native_trial_spec(
                model=self.model,
                placement=self.placement,
                llama_cli=self.program,
                gguf=self.gguf,
                prompt="x",
                context_tokens=0,
            )


if __name__ == "__main__":
    unittest.main()
