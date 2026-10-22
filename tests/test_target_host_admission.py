from __future__ import annotations

import copy
import unittest

from tensormeld.admission import LocalAdmissionController
from tensormeld.native_runtime_collector import NativeRuntimeManifestCollection, _canonical_sha256 as collector_sha
from tensormeld.qualification import QualificationEvidence
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from tensormeld.target_host_admission import orchestrate_target_host_admission
from tensormeld.target_host_qualification import TargetHostQualificationHandoff, _canonical_sha256 as handoff_sha
from test_admission import snapshot
from test_llamacpp_native_trial import qualification_context, make_model
from test_target_host_qualification import runtime_identity
from pathlib import Path
import tempfile


class TargetHostAdmissionOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        gguf = root / "fixture.gguf"
        gguf.write_bytes(b"GGUF" + b"x" * 32)
        self.model = make_model(gguf)
        (
            self.config,
            self.planning,
            self.candidate,
            self.adapter,
            self.placement,
        ) = qualification_context(self.model)
        self.device = self.config.devices[0]
        self.node = self.device.node
        self.worker_sha = "8" * 64
        self.identity = runtime_identity(self.node, self.device.id, self.worker_sha)

        profile = self.config.profile_map[self.config.installation.default_profile]
        self.qualification = QualificationEvidence.parse({
            "qualification_schema": "tensormeld/qualification-evidence-v2",
            "evidence_id": "native-e3-fixture",
            "level": "E3",
            "adapter_id": self.adapter.adapter_id,
            "adapter_capabilities_sha256": self.adapter.fingerprint,
            "engine_revision": self.adapter.engine_revision,
            "worker_artifact_sha256": self.worker_sha,
            "model_manifest_sha256": self.model.manifest_sha256,
            "config_sha256": self.config.fingerprint,
            "device_ids": [self.device.id],
            "workload": {
                "context_tokens": profile.workload.context_tokens,
                "max_output_tokens": profile.workload.max_output_tokens,
                "concurrency": profile.workload.max_active_requests,
            },
            "result": "passed",
            "observed_at": "2026-10-04T18:00:00Z",
            "tests": ["native-subprocess", "exact-placement"],
            "candidate_plan_sha256": self.candidate["plan_sha256"],
            "placement_sha256": self.placement.fingerprint,
            "trial_spec_sha256": "1" * 64,
            "correctness_contract_sha256": "2" * 64,
            "runtime_identity_sha256": [self.identity.identity_sha256],
        })

        raw_manifest = {
            "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
            "provenance": "native-adapter",
            "config_sha256": self.config.fingerprint,
            "profile": profile.name,
            "model_manifest_sha256": self.model.manifest_sha256,
            "adapter_id": self.adapter.adapter_id,
            "adapter_capabilities_sha256": self.adapter.fingerprint,
            "engine_revision": self.adapter.engine_revision,
            "worker_artifact_sha256": self.worker_sha,
            "workload": {
                "task": profile.workload.task,
                "context_tokens": profile.workload.context_tokens,
                "max_output_tokens": profile.workload.max_output_tokens,
                "concurrency": profile.workload.max_active_requests,
            },
            "required_operators": ["ADD"],
            "devices": [{
                "id": self.device.id,
                "node": self.device.node,
                "backend": self.device.backend,
                "operators": ["ADD"],
            }],
            "physical_pool_memory": [{
                "pool_ref": self.device.pool,
                "node": self.device.node,
                "resident_bytes": 100,
                "state_bytes": 20,
                "workspace_peak_bytes": 10,
                "preparation_peak_bytes": 130,
            }],
            "reservation_created": False,
            "qualified": False,
            "executable": False,
        }
        self.manifest = parse_runtime_model_manifest(
            raw_manifest,
            config=self.config,
            model=self.model,
            adapter=self.adapter,
        )

        handoff_core = {
            "handoff_schema": "tensormeld/target-host-qualification-handoff-v1",
            "config_sha256": self.config.fingerprint,
            "planning_input_sha256": self.planning.fingerprint,
            "candidate_plan_sha256": self.candidate["plan_sha256"],
            "model_manifest_sha256": self.model.manifest_sha256,
            "adapter_id": self.adapter.adapter_id,
            "adapter_capabilities_sha256": self.adapter.fingerprint,
            "engine_revision": self.adapter.engine_revision,
            "probe_artifact_sha256": self.worker_sha,
            "native_binding_sha256": "3" * 64,
            "placement_sha256": self.placement.fingerprint,
            "trial_spec_sha256": "1" * 64,
            "e3_reference_sha256": "2" * 64,
            "qualification_evidence_sha256": self.qualification.evidence_sha256,
            "e3_evaluation_sha256": "4" * 64,
            "runtime_identity_sha256": [self.identity.identity_sha256],
            "backend_readiness": [{
                "device_id": self.device.id,
                "evidence_sha256": "5" * 64,
                "runtime_identity_sha256": self.identity.identity_sha256,
                "runtime_ready": True,
            }],
            "combined_runtime_observation": {},
            "runtime_manifest_requirements": {
                "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
                "required_provenance": "native-adapter",
                "config_sha256": self.config.fingerprint,
                "profile": profile.name,
                "model_manifest_sha256": self.model.manifest_sha256,
                "adapter_id": self.adapter.adapter_id,
                "adapter_capabilities_sha256": self.adapter.fingerprint,
                "engine_revision": self.adapter.engine_revision,
                "worker_artifact_sha256": self.worker_sha,
                "required_devices": [self.device.id],
                "required_nodes": [self.node],
                "required_workload": raw_manifest["workload"],
                "e3_workload": {
                    "context_tokens": profile.workload.context_tokens,
                    "max_output_tokens": profile.workload.max_output_tokens,
                    "concurrency": profile.workload.max_active_requests,
                },
                "e3_workload_matches_profile": True,
                "operator_measurement_required": True,
                "physical_pool_memory_measurement_required": True,
            },
            "backend_ready": True,
            "e3_qualified": True,
            "runtime_manifest_required": True,
            "reservation_created": False,
            "launch_authorized": False,
            "executable": False,
            "warnings": [],
        }
        handoff_core["handoff_sha256"] = handoff_sha(handoff_core)
        self.handoff = TargetHostQualificationHandoff(
            handoff_core, self.qualification
        )

        collection_core = {
            "collector_schema": "tensormeld/native-runtime-manifest-collector-v1",
            "handoff_sha256": self.handoff.fingerprint,
            "measurement_sha256": "6" * 64,
            "operator_requirements_sha256": "7" * 64,
            "runtime_manifest_sha256": self.manifest.fingerprint,
            "measurement_source": "native-adapter",
            "operator_requirements_source": "native-adapter",
            "runtime_manifest_provenance": "native-adapter",
            "operator_coverage_complete": True,
            "e3_workload_matches_profile": True,
            "admission_ready_inputs": True,
            "runtime_manifest_summary": {},
            "reservation_created": False,
            "launch_authorized": False,
            "executable": False,
            "warnings": [],
        }
        collection_core["collector_sha256"] = collector_sha(collection_core)
        self.collection = NativeRuntimeManifestCollection(
            self.manifest, collection_core
        )

    def tearDown(self):
        self.tmp.cleanup()

    def snapshots(self):
        return (
            snapshot(self.config, "reserve-1"),
            snapshot(self.config, "launch-2"),
        )

    def orchestrate(self, **changes):
        reserve, launch = self.snapshots()
        kwargs = {
            "controller": LocalAdmissionController(),
            "lease_id": "lease-a",
            "config": self.config,
            "planning": self.planning,
            "candidate": self.candidate,
            "adapter": self.adapter,
            "model": self.model,
            "handoff": self.handoff,
            "collection": self.collection,
            "reservation_snapshot": reserve,
            "launch_snapshot": launch,
        }
        kwargs.update(changes)
        return orchestrate_target_host_admission(**kwargs)

    def test_reserve_fresh_recheck_builds_accepted_bundle_without_starting_inference(self):
        result = self.orchestrate()
        self.assertTrue(result.record["reservation_created"])
        self.assertTrue(result.record["launch_authorized"])
        self.assertTrue(result.record["execution_authorized"])
        self.assertFalse(result.record["inference_started"])
        self.assertFalse(result.record["real_model_inference"])
        self.assertEqual(
            result.record["accepted_execution_bundle_sha256"],
            result.bundle.bundle_sha256,
        )

    def test_same_observation_cannot_be_reused_for_launch(self):
        controller = LocalAdmissionController()
        reserve = snapshot(self.config, "same")
        with self.assertRaises(ValidationError):
            self.orchestrate(
                controller=controller,
                reservation_snapshot=reserve,
                launch_snapshot=copy.deepcopy(reserve),
            )
        self.assertEqual(controller.active_leases(), ())

    def test_launch_rejection_rolls_back_reserved_lease(self):
        controller = LocalAdmissionController()
        reserve = snapshot(self.config, "reserve")
        pool = self.manifest.pools[0].pool
        policy = next(p for p in self.config.resource_policies if p.pool == pool)
        low = {pool: policy.safety_headroom_bytes}
        launch = snapshot(self.config, "launch", available=low)
        with self.assertRaises(ValidationError):
            self.orchestrate(
                controller=controller,
                reservation_snapshot=reserve,
                launch_snapshot=launch,
            )
        self.assertEqual(controller.active_leases(), ())

    def test_non_native_or_non_ready_collection_is_rejected_before_reserve(self):
        for field, value in (
            ("admission_ready_inputs", False),
            ("runtime_manifest_provenance", "fixture"),
        ):
            record = copy.deepcopy(self.collection.record)
            record[field] = value
            record.pop("collector_sha256")
            record["collector_sha256"] = collector_sha(record)
            collection = NativeRuntimeManifestCollection(self.manifest, record)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.orchestrate(collection=collection)

    def test_tampered_collection_fingerprint_is_rejected(self):
        record = copy.deepcopy(self.collection.record)
        record["measurement_sha256"] = "0" * 64
        collection = NativeRuntimeManifestCollection(self.manifest, record)
        with self.assertRaises(ValidationError):
            self.orchestrate(collection=collection)

    def test_stale_candidate_plan_is_rejected(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["plan_sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            self.orchestrate(candidate=candidate)


if __name__ == "__main__":
    unittest.main()
