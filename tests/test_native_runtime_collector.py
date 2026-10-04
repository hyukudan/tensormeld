from __future__ import annotations

import copy
import unittest

from tensormeld.native_runtime_collector import (
    NativeRuntimeMeasurement,
    RuntimeOperatorRequirements,
    collect_native_runtime_manifest,
)
from tensormeld.schema import ValidationError
from tensormeld.target_host_qualification import TargetHostQualificationHandoff
from test_target_host_qualification import TargetHostQualificationTests


class NativeRuntimeCollectorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TargetHostQualificationTests(
            methodName="test_chain_validates_all_gates_and_emits_non_executable_handoff"
        )
        self.fixture.setUp()
        self.handoff = self.fixture.assemble()
        self.req = self.handoff.record["runtime_manifest_requirements"]
        self.device = self.fixture.device
        self.pool = next(
            p for p in self.fixture.config.pools if p.id == self.device.pool
        )

    def tearDown(self):
        self.fixture.tearDown()

    def measurement_raw(self, **changes):
        raw = {
            "measurement_schema": "tensormeld/native-runtime-measurement-v1",
            "measurement_source": "fixture",
            "handoff_sha256": self.handoff.fingerprint,
            "config_sha256": self.fixture.config.fingerprint,
            "model_manifest_sha256": self.fixture.model.manifest_sha256,
            "adapter_capabilities_sha256": self.fixture.adapter.fingerprint,
            "engine_revision": self.fixture.adapter.engine_revision,
            "worker_artifact_sha256": self.fixture.spec.llama_cli_sha256,
            "workload": dict(self.req["required_workload"]),
            "devices": [{
                "id": self.device.id,
                "node": self.device.node,
                "backend": self.device.backend,
                "runtime_identity_sha256": self.fixture.identity.identity_sha256,
                "operators": ["ADD"],
            }],
            "physical_pool_memory": [{
                "pool_ref": self.pool.id,
                "node": self.pool.node,
                "resident_bytes": 100,
                "state_bytes": 20,
                "workspace_peak_bytes": 10,
                "preparation_peak_bytes": 130,
            }],
            "reservation_created": False,
            "qualified": False,
            "executable": False,
        }
        raw.update(changes)
        return raw

    def parse_measurement(self, **changes):
        return NativeRuntimeMeasurement.parse(
            self.measurement_raw(**changes),
            handoff=self.handoff,
        )

    def operator_requirements(self, *, source="fixture", required=("ADD",)):
        return RuntimeOperatorRequirements.parse({
            "operator_requirements_schema": "tensormeld/runtime-operator-requirements-v1",
            "requirements_source": source,
            "handoff_sha256": self.handoff.fingerprint,
            "model_manifest_sha256": self.fixture.model.manifest_sha256,
            "candidate_plan_sha256": self.handoff.record["candidate_plan_sha256"],
            "placement_sha256": self.handoff.record["placement_sha256"],
            "required_operators": list(required),
            "qualified": False,
            "executable": False,
        }, handoff=self.handoff)

    def test_fixture_measurement_builds_non_executable_runtime_manifest(self):
        measurement = self.parse_measurement()
        result = collect_native_runtime_manifest(
            config=self.fixture.config,
            model=self.fixture.model,
            adapter=self.fixture.adapter,
            handoff=self.handoff,
            measurement=measurement,
            operator_requirements=self.operator_requirements(),
        )
        self.assertEqual(result.manifest.provenance, "fixture")
        self.assertTrue(result.manifest.operator_coverage_complete)
        self.assertFalse(result.record["admission_ready_inputs"])
        self.assertFalse(result.record["reservation_created"])
        self.assertFalse(result.record["launch_authorized"])
        self.assertFalse(result.record["executable"])
        self.assertEqual(
            result.record["runtime_manifest_sha256"],
            result.manifest.fingerprint,
        )

    def test_missing_compute_device_pool_is_rejected(self):
        raw = self.measurement_raw()
        raw["physical_pool_memory"] = [{
            "pool_ref": next(
                p.id for p in self.fixture.config.pools if p.id != self.pool.id
            ),
            "node": next(
                p.node for p in self.fixture.config.pools if p.id != self.pool.id
            ),
            "resident_bytes": 1,
            "state_bytes": 0,
            "workspace_peak_bytes": 0,
            "preparation_peak_bytes": 1,
        }]
        measurement = NativeRuntimeMeasurement.parse(raw, handoff=self.handoff)
        with self.assertRaises(ValidationError):
            collect_native_runtime_manifest(
                config=self.fixture.config,
                model=self.fixture.model,
                adapter=self.fixture.adapter,
                handoff=self.handoff,
                measurement=measurement,
            )

    def test_runtime_identity_mismatch_is_rejected_at_measurement_parse(self):
        raw = self.measurement_raw()
        raw["devices"][0]["runtime_identity_sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            NativeRuntimeMeasurement.parse(raw, handoff=self.handoff)

    def test_worker_artifact_mismatch_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.parse_measurement(worker_artifact_sha256="0" * 64)

    def test_incomplete_operator_coverage_remains_non_admission_ready(self):
        measurement = self.parse_measurement()
        result = collect_native_runtime_manifest(
            config=self.fixture.config,
            model=self.fixture.model,
            adapter=self.fixture.adapter,
            handoff=self.handoff,
            measurement=measurement,
            operator_requirements=self.operator_requirements(
                required=("ADD", "MUL")
            ),
        )
        self.assertFalse(result.manifest.operator_coverage_complete)
        self.assertFalse(result.record["admission_ready_inputs"])

    def test_workload_must_match_handoff_profile_exactly(self):
        raw = self.measurement_raw()
        raw["workload"]["context_tokens"] += 1
        with self.assertRaises(ValidationError):
            NativeRuntimeMeasurement.parse(raw, handoff=self.handoff)

    def test_tampered_handoff_is_rejected(self):
        record = copy.deepcopy(self.handoff.record)
        record["model_manifest_sha256"] = "0" * 64
        tampered = TargetHostQualificationHandoff(
            record,
            self.handoff.qualification_evidence,
        )
        with self.assertRaises(ValidationError):
            NativeRuntimeMeasurement.parse(
                self.measurement_raw(),
                handoff=tampered,
            )

    def test_duplicate_pool_records_are_rejected(self):
        raw = self.measurement_raw()
        raw["physical_pool_memory"].append(
            copy.deepcopy(raw["physical_pool_memory"][0])
        )
        with self.assertRaises(ValidationError):
            NativeRuntimeMeasurement.parse(raw, handoff=self.handoff)

    def test_native_label_alone_does_not_override_e3_workload_gap(self):
        raw = self.measurement_raw(measurement_source="native-adapter")
        measurement = NativeRuntimeMeasurement.parse(raw, handoff=self.handoff)
        result = collect_native_runtime_manifest(
            config=self.fixture.config,
            model=self.fixture.model,
            adapter=self.fixture.adapter,
            handoff=self.handoff,
            measurement=measurement,
            operator_requirements=self.operator_requirements(
                source="native-adapter"
            ),
        )
        self.assertEqual(result.manifest.provenance, "native-adapter")
        self.assertTrue(result.manifest.operator_coverage_complete)
        self.assertFalse(
            self.handoff.record["runtime_manifest_requirements"][
                "e3_workload_matches_profile"
            ]
        )
        self.assertFalse(result.record["admission_ready_inputs"])


if __name__ == "__main__":
    unittest.main()
