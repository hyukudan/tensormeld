from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.llamacpp_native_trial import (
    approve_single_file_gguf,
    build_llamacpp_native_trial_spec,
    run_llamacpp_native_trial,
)
from tensormeld.llamacpp_placement import (
    LlamaCppPlacementBinding,
    translate_whole_blocks_to_llamacpp,
)
from tensormeld.model_manifest import ModelManifest
from tensormeld.native_worker import approved_worker_artifact
from tensormeld.schema import ValidationError
from tensormeld.whole_block_execution import AcceptedExecutionBundle

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"
FIXTURE_CLI = Path(__file__).parent / "fixtures" / "llamacpp_cli_fixture.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def adapter():
    return AdapterCapabilities.parse({
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
        "coordinator_nodes": ["n0"],
        "devices": [{"id": "g0", "node": "n0", "backend": "cpu"}],
    })


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


def bundle(a, m):
    return AcceptedExecutionBundle(
        config_sha256="2" * 64,
        profile="interactive",
        planning_input_sha256="3" * 64,
        plan_sha256="4" * 64,
        representability_sha256="5" * 64,
        adapter_id=a.adapter_id,
        engine_revision=a.engine_revision,
        adapter_capabilities_sha256=a.fingerprint,
        model_manifest_sha256=m.manifest_sha256,
        qualification_evidence_sha256="6" * 64,
        runtime_manifest_sha256="7" * 64,
        worker_artifact_sha256="8" * 64,
        backend_readiness=(("g0", "9" * 64, "a" * 64),),
        launch_leases=(("n0", "lease", "b" * 64),),
        segments=(("g0", 0, 1),),
        unit_ids=("blk.0",),
        compute_devices=("g0",),
        compute_nodes=("n0",),
        bundle_sha256="c" * 64,
    )


def placement(a, b, m):
    binding = LlamaCppPlacementBinding.parse({
        "placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
        "adapter_id": a.adapter_id,
        "adapter_capabilities_sha256": a.fingerprint,
        "source_revision": PIN,
        "native_binding_sha256": "d" * 64,
        "devices": [{
            "device_id": "g0",
            "engine_device_name": "CPU",
            "buffer_type": "CPU",
        }],
    }, adapter=a)
    bound = {
        "result_schema": "tensormeld/llamacpp-device-binding-result-v1",
        "config_sha256": b.config_sha256,
        "binding_sha256": "d" * 64,
        "resolved_mappings": [{
            "tensormeld_device_id": "g0",
            "engine_device_name": "CPU",
            "backend_from_config": "cpu",
        }],
    }
    index = {
        "index_schema": "tensormeld/gguf-index-v1",
        "architecture": "llama",
        "complete_shard_set": True,
        "tensor_count": 2,
        "index_sha256": m.tensor_index_sha256,
        "files": [{"tensors": [
            {"name": "blk.0.attn_q.weight"},
            {"name": "blk.0.ffn_up.weight"},
        ]}],
    }
    return translate_whole_blocks_to_llamacpp(
        b,
        adapter=a,
        binding=binding,
        bound_result=bound,
        model=m,
        gguf_index=index,
    )


class LlamaCppNativeTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gguf_path = self.root / "fixture.gguf"
        self.gguf_path.write_bytes(b"GGUF" + b"x" * 32)
        self.model = make_model(self.gguf_path)
        self.adapter = adapter()
        self.bundle = bundle(self.adapter, self.model)
        self.placement = placement(self.adapter, self.bundle, self.model)
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
        # Fixture process is launched as python <fixture>; the production spec itself
        # binds the approved CLI artifact path. The injected runner below adds the
        # launcher only for the real portable subprocess test.
        return build_llamacpp_native_trial_spec(
            bundle=self.bundle,
            model=self.model,
            placement=self.placement,
            llama_cli=self.program,
            gguf=self.gguf,
            prompt="TensorMeld trial",
            context_tokens=128,
            predict_tokens=1,
        )

    def test_closed_argv_contains_exact_model_and_placement(self):
        spec = self.spec()
        self.assertEqual(spec.argv[0], str(FIXTURE_CLI.resolve()))
        self.assertEqual(spec.argv[1:3], ("-m", str(self.gguf_path.resolve())))
        self.assertIn("--override-tensor", spec.argv)
        self.assertIn("^blk\\.0\\..*=CPU", spec.argv)
        self.assertIn("--single-turn", spec.argv)
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
        self.assertTrue(result["model_output_observed"])
        self.assertEqual(result["evidence_level"], "fixture")
        self.assertFalse(result["qualified"])
        self.assertFalse(result["real_model_inference"])

    def test_native_success_still_does_not_self_qualify(self):
        spec = self.spec()

        def runner(argv, timeout):
            return 0, b"some-output", b""

        result = run_llamacpp_native_trial(
            spec,
            runner=runner,
            execution_source="native-subprocess",
        )
        self.assertEqual(result["evidence_level"], "E2.5")
        self.assertTrue(result["process_succeeded"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["real_model_inference"])
        self.assertFalse(result["executable"])

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
                bundle=self.bundle,
                model=self.model,
                placement=self.placement,
                llama_cli=self.program,
                gguf=self.gguf,
                prompt="x" * 4097,
                context_tokens=128,
            )
        with self.assertRaises(ValidationError):
            build_llamacpp_native_trial_spec(
                bundle=self.bundle,
                model=self.model,
                placement=self.placement,
                llama_cli=self.program,
                gguf=self.gguf,
                prompt="x",
                context_tokens=0,
            )


if __name__ == "__main__":
    unittest.main()
