from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tensormeld.backend_evidence import (
    load_backend_readiness_evidence,
    retain_llamacpp_backend_evidence,
    validate_llamacpp_backend_evidence,
)
from tensormeld.schema import ValidationError
from tensormeld.runtime_identity import RuntimeIdentity

CONFIG_SHA = "a" * 64
PROBE_SHA = "b" * 64
BINDING_SHA = "c" * 64
TEST_SHA = "d" * 64

def runtime_identity(**changes) -> RuntimeIdentity:
    raw = {
        "runtime_identity_schema": "tensormeld/runtime-identity-v1",
        "node_id": "node-a",
        "worker_artifact_sha256": "1" * 64,
        "worker_build_id": "worker-build-1",
        "os_name": "linux",
        "os_version": "fixture-os",
        "driver_id": "fixture-driver",
        "driver_version": "1",
        "runtime_id": "fixture-runtime",
        "runtime_version": "2",
        "tensormeld_device_id": "gpu0",
        "physical_device_id": "pci:0000:01:00.0",
        "topology_sha256": "2" * 64,
    }
    raw.update(changes)
    return RuntimeIdentity.parse(raw)



def self_test_result(source: str = "native-subprocess") -> dict:
    return {
        "self_test_schema": "tensormeld/llamacpp-backend-self-test-v1",
        "engine": "llama.cpp",
        "upstream_target": "test-backend-ops",
        "pinned_source_revision": "552f18f912a32ea86edf82e2b76431cb7131538d",
        "observed_source_revisions": ["552f18f9"],
        "test_artifact_sha256": TEST_SHA,
        "probe_artifact_sha256": PROBE_SHA,
        "binding_sha256": BINDING_SHA,
        "config_sha256": CONFIG_SHA,
        "tensormeld_device_id": "gpu0",
        "engine_device_name": "CUDA0",
        "backend_from_config": "cuda",
        "operation": "ADD",
        "result_rows": 2,
        "supported_rows": 2,
        "passed_rows": 2,
        "backend_initialized": True,
        "backend_executed": True,
        "evidence_level": "E2",
        "model_loaded": False,
        "listener_started": False,
        "reservation_created": False,
        "execution_source": source,
        "qualified": False,
        "executable": False,
        "runtime_observation": {
            "runtime_observation_schema": "tensormeld/runtime-observation-v1",
            "config_sha256": CONFIG_SHA,
            "devices": {
                "gpu0": {"backend": "cuda", "state": "ready"},
            },
            "pools": {},
            "qualified": False,
            "executable": False,
        },
        "warnings": [],
    }


def bound_result(binding_sha: str = BINDING_SHA) -> dict:
    return {
        "result_schema": "tensormeld/llamacpp-device-binding-result-v1",
        "config_sha256": CONFIG_SHA,
        "probe_artifact_sha256": PROBE_SHA,
        "binding_sha256": binding_sha,
        "resolved_mappings": [
            {
                "tensormeld_device_id": "gpu0",
                "engine_device_name": "CUDA0",
                "backend_from_config": "cuda",
            }
        ],
        "runtime_observation": {
            "devices": {
                "gpu0": {"backend": "cuda", "state": "observed"},
            }
        },
    }


class BackendReadinessEvidenceTests(unittest.TestCase):
    def test_native_result_can_be_retained_but_not_reused_as_ready(self):
        identity = runtime_identity()
        evidence = retain_llamacpp_backend_evidence(
            self_test_result(), runtime_identity=identity
        )
        self.assertTrue(evidence["native_execution_recorded"])
        self.assertTrue(evidence["backend_readiness_proven"])
        self.assertFalse(evidence["qualified"])
        self.assertFalse(evidence["executable"])

        applicability = validate_llamacpp_backend_evidence(
            evidence,
            config=SimpleNamespace(fingerprint=CONFIG_SHA),
            bound_result=bound_result(),
            expected_test_artifact_sha256=TEST_SHA,
            tensormeld_device_id="gpu0",
            current_runtime_identity=identity,
        )
        self.assertTrue(applicability["identity_applicable"])
        self.assertFalse(applicability["requires_live_runtime_recheck"])
        self.assertTrue(applicability["runtime_ready"])
        self.assertFalse(applicability["reservation_created"])
        self.assertFalse(applicability["executable"])

    def test_injected_fixture_cannot_be_retained_as_hardware_evidence(self):
        with self.assertRaises(ValidationError):
            retain_llamacpp_backend_evidence(
                self_test_result("injected-runner"),
                runtime_identity=runtime_identity(),
            )

    def test_tampered_record_fingerprint_is_rejected(self):
        evidence = retain_llamacpp_backend_evidence(
            self_test_result(), runtime_identity=runtime_identity()
        )
        evidence["backend"] = "hip"
        with self.assertRaises(ValidationError):
            validate_llamacpp_backend_evidence(
                evidence,
                config=SimpleNamespace(fingerprint=CONFIG_SHA),
                bound_result=bound_result(),
                expected_test_artifact_sha256=TEST_SHA,
                tensormeld_device_id="gpu0",
                current_runtime_identity=runtime_identity(),
            )

    def test_stale_binding_identity_is_rejected(self):
        evidence = retain_llamacpp_backend_evidence(
            self_test_result(), runtime_identity=runtime_identity()
        )
        with self.assertRaises(ValidationError):
            validate_llamacpp_backend_evidence(
                evidence,
                config=SimpleNamespace(fingerprint=CONFIG_SHA),
                bound_result=bound_result("e" * 64),
                expected_test_artifact_sha256=TEST_SHA,
                tensormeld_device_id="gpu0",
                current_runtime_identity=runtime_identity(),
            )

    def test_loader_rejects_duplicate_keys_and_accepts_valid_record(self):
        evidence = retain_llamacpp_backend_evidence(
            self_test_result(), runtime_identity=runtime_identity()
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            valid = root / "valid.json"
            valid.write_text(json.dumps(evidence), encoding="utf-8")
            loaded = load_backend_readiness_evidence(valid)
            self.assertEqual(
                loaded["evidence_sha256"],
                evidence["evidence_sha256"],
            )

            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"evidence_schema":"x","evidence_schema":"y"}',
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                load_backend_readiness_evidence(duplicate)


    def test_runtime_identity_change_invalidates_retained_e2(self):
        identity = runtime_identity()
        evidence = retain_llamacpp_backend_evidence(
            self_test_result(), runtime_identity=identity
        )
        with self.assertRaises(ValidationError):
            validate_llamacpp_backend_evidence(
                evidence,
                config=SimpleNamespace(fingerprint=CONFIG_SHA),
                bound_result=bound_result(),
                expected_test_artifact_sha256=TEST_SHA,
                tensormeld_device_id="gpu0",
                current_runtime_identity=runtime_identity(driver_version="2"),
            )

    def test_runtime_identity_device_must_match_retained_device(self):
        with self.assertRaises(ValidationError):
            retain_llamacpp_backend_evidence(
                self_test_result(),
                runtime_identity=runtime_identity(tensormeld_device_id="gpu1"),
            )


if __name__ == "__main__":
    unittest.main()
