from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from tensormeld.backend_evidence import retain_llamacpp_backend_evidence
from tensormeld.device_binding import LlamaCppBinding, bind_llamacpp_probe
from tensormeld.llamacpp_e3 import LlamaCppE3Reference
from tensormeld.llamacpp_native_trial import (
    approve_single_file_gguf,
    build_llamacpp_native_trial_spec,
)
from tensormeld.llamacpp_placement import (
    LlamaCppPlacementBinding,
    translate_candidate_for_llamacpp_qualification,
)
from tensormeld.native_worker import approved_worker_artifact
from tensormeld.runtime_identity import RuntimeIdentity
from tensormeld.schema import ValidationError
from tensormeld.target_host_qualification import assemble_target_host_qualification
from test_llamacpp_native_trial import (
    FIXTURE_CLI,
    make_model,
    qualification_context,
    sha256,
)

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"
TEST_SHA = "d" * 64


def runtime_identity(node_id: str, device_id: str, worker_sha: str, *, driver_version="1"):
    return RuntimeIdentity.parse({
        "runtime_identity_schema": "tensormeld/runtime-identity-v1",
        "node_id": node_id,
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


class TargetHostQualificationTests(unittest.TestCase):
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
            _,
        ) = qualification_context(self.model)
        self.device = self.config.devices[0]
        self.node = self.device.node

        self.program = approved_worker_artifact(
            FIXTURE_CLI,
            expected_sha256=sha256(FIXTURE_CLI),
            where="fixture llama cli",
        )
        self.probe = {
            "probe_schema": "tensormeld/llamacpp-probe-v1",
            "engine": "llama.cpp",
            "pinned_source_revision": PIN,
            "artifact_sha256": self.program.sha256,
            "binary_name": FIXTURE_CLI.name,
            "build": {
                "version": "fixture",
                "build": 1,
                "commit": PIN[:8],
                "compiler": "fixture",
                "target": "fixture",
            },
            "devices": [{
                "engine_device_name": "CPU",
                "description": "fixture CPU",
                "total_bytes": 1024 * 1024 * 1024,
                "free_bytes": 512 * 1024 * 1024,
            }],
            "device_identity_mapping": "unresolved",
            "model_loaded": False,
            "listener_started": False,
            "qualified": False,
            "executable": False,
            "warnings": [],
        }
        self.native_binding = LlamaCppBinding.parse({
            "binding_schema": "tensormeld/llamacpp-device-binding-v1",
            "config_sha256": self.config.fingerprint,
            "node_id": self.node,
            "artifact_sha256": self.program.sha256,
            "approval": "explicit",
            "mappings": [{
                "engine_device_name": "CPU",
                "tensormeld_device_id": self.device.id,
                "memory_reporter": True,
            }],
        })
        self.bound = bind_llamacpp_probe(
            self.config, self.probe, self.native_binding
        )

        self.placement_binding = LlamaCppPlacementBinding.parse({
            "placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
            "adapter_id": self.adapter.adapter_id,
            "adapter_capabilities_sha256": self.adapter.fingerprint,
            "source_revision": PIN,
            "native_binding_sha256": self.native_binding.fingerprint,
            "devices": [{
                "device_id": self.device.id,
                "engine_device_name": "CPU",
                "buffer_type": "CPU",
            }],
        }, adapter=self.adapter)
        self.gguf_index = {
            "index_schema": "tensormeld/gguf-index-v1",
            "architecture": "llama",
            "complete_shard_set": True,
            "tensor_count": 2,
            "index_sha256": self.model.tensor_index_sha256,
            "files": [{"tensors": [
                {"name": "blk.0.attn_q.weight"},
                {"name": "blk.0.ffn_up.weight"},
            ]}],
        }
        self.placement = translate_candidate_for_llamacpp_qualification(
            self.config,
            self.planning,
            self.candidate,
            adapter=self.adapter,
            binding=self.placement_binding,
            bound_result=self.bound,
            model=self.model,
            gguf_index=self.gguf_index,
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
        stdout = b"deterministic native completion"
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
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_bytes": len(stdout),
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
        self.identity = runtime_identity(
            self.node,
            self.device.id,
            self.spec.llama_cli_sha256,
        )
        self.self_test = {
            "self_test_schema": "tensormeld/llamacpp-backend-self-test-v1",
            "engine": "llama.cpp",
            "upstream_target": "test-backend-ops",
            "pinned_source_revision": PIN,
            "observed_source_revisions": [PIN[:8]],
            "test_artifact_sha256": TEST_SHA,
            "probe_artifact_sha256": self.program.sha256,
            "binding_sha256": self.native_binding.fingerprint,
            "config_sha256": self.config.fingerprint,
            "node_id": self.node,
            "tensormeld_device_id": self.device.id,
            "engine_device_name": "CPU",
            "backend_from_config": self.device.backend,
            "operation": "ADD",
            "result_rows": 1,
            "supported_rows": 1,
            "passed_rows": 1,
            "backend_initialized": True,
            "backend_executed": True,
            "evidence_level": "E2",
            "model_loaded": False,
            "listener_started": False,
            "reservation_created": False,
            "execution_source": "native-subprocess",
            "qualified": False,
            "executable": False,
            "runtime_observation": {
                "runtime_observation_schema": "tensormeld/runtime-observation-v1",
                "config_sha256": self.config.fingerprint,
                "devices": {
                    self.device.id: {
                        "backend": self.device.backend,
                        "state": "ready",
                    }
                },
                "pools": {},
                "qualified": False,
                "executable": False,
            },
            "warnings": [],
        }
        self.backend_evidence = retain_llamacpp_backend_evidence(
            self.self_test,
            runtime_identity=self.identity,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def assemble(self, **changes):
        kwargs = {
            "config": self.config,
            "planning": self.planning,
            "candidate": self.candidate,
            "adapter": self.adapter,
            "model": self.model,
            "gguf_index": self.gguf_index,
            "probe": self.probe,
            "native_binding": self.native_binding,
            "bound_result": self.bound,
            "placement_binding": self.placement_binding,
            "placement": self.placement,
            "trial_spec": self.spec,
            "trial_result": self.trial,
            "e3_reference": self.reference,
            "runtime_identities": (self.identity,),
            "backend_evidence_by_device": {
                self.device.id: self.backend_evidence
            },
            "backend_test_artifact_sha256_by_device": {
                self.device.id: TEST_SHA
            },
            "observed_at": "2026-10-04T18:00:00Z",
        }
        kwargs.update(changes)
        return assemble_target_host_qualification(**kwargs)

    def test_chain_validates_all_gates_and_emits_non_executable_handoff(self):
        handoff = self.assemble()
        record = handoff.record
        self.assertTrue(record["backend_ready"])
        self.assertTrue(record["e3_qualified"])
        self.assertTrue(record["runtime_manifest_required"])
        self.assertFalse(record["reservation_created"])
        self.assertFalse(record["launch_authorized"])
        self.assertFalse(record["executable"])
        self.assertEqual(
            record["combined_runtime_observation"]["devices"][self.device.id]["state"],
            "ready",
        )
        self.assertEqual(
            record["qualification_evidence_sha256"],
            handoff.qualification_evidence.evidence_sha256,
        )
        self.assertEqual(len(record["handoff_sha256"]), 64)

    def test_handoff_surfaces_e3_profile_workload_gap(self):
        handoff = self.assemble()
        req = handoff.record["runtime_manifest_requirements"]
        self.assertFalse(req["e3_workload_matches_profile"])
        self.assertEqual(req["e3_workload"]["max_output_tokens"], 1)
        self.assertEqual(
            req["required_workload"]["max_output_tokens"],
            self.config.profile_map["interactive"].workload.max_output_tokens,
        )

    def test_stale_supplied_binding_result_is_rejected(self):
        stale = dict(self.bound)
        stale["binding_sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            self.assemble(bound_result=stale)

    def test_e2_runtime_identity_change_is_rejected(self):
        changed = runtime_identity(
            self.node,
            self.device.id,
            self.spec.llama_cli_sha256,
            driver_version="2",
        )
        with self.assertRaises(ValidationError):
            self.assemble(runtime_identities=(changed,))

    def test_fixture_trial_cannot_pass_orchestrator_e3(self):
        trial = dict(self.trial)
        trial["execution_source"] = "fixture-subprocess"
        trial["evidence_level"] = "fixture"
        with self.assertRaises(ValidationError):
            self.assemble(trial_result=trial)

    def test_probe_and_trial_must_use_same_llama_artifact(self):
        probe = dict(self.probe)
        probe["artifact_sha256"] = "f" * 64
        with self.assertRaises(ValidationError):
            self.assemble(probe=probe)

    def test_backend_evidence_must_cover_exact_devices(self):
        with self.assertRaises(ValidationError):
            self.assemble(backend_evidence_by_device={})


if __name__ == "__main__":
    unittest.main()
