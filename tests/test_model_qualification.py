from __future__ import annotations

import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.model_manifest import ModelManifest, manifest_from_gguf_index
from tensormeld.qualification import QualificationEvidence, evidence_applies
from tensormeld.schema import ValidationError

H = "a" * 64
I = "b" * 64
C = "c" * 64
W = "d" * 64


def index():
    return {
        "index_schema": "tensormeld/gguf-index-v1",
        "reader": "gguf==0.19.0",
        "architecture": "test",
        "complete_shard_set": True,
        "tensor_count": 2,
        "files": [
            {"file_name": "m-00001.gguf", "file_size_bytes": 100, "split_index": 0},
            {"file_name": "m-00002.gguf", "file_size_bytes": 200, "split_index": 1},
        ],
        "tensor_payload_bytes": 250,
        "checkpoint_sha256": None,
        "qualified": False,
        "executable": False,
        "warnings": [],
        "index_sha256": I,
    }


def manifest():
    return manifest_from_gguf_index(
        index(),
        model_id="org/model",
        revision="rev1",
        tokenizer_ref="tokenizer:rev1",
        shard_sha256={"m-00001.gguf": H, "m-00002.gguf": "e" * 64},
    )


def adapter():
    return AdapterCapabilities.parse({
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "native",
        "engine": "engine",
        "engine_revision": "rev-engine",
        "placement": {
            "strategies": ["whole_blocks"],
            "exact_owner_binding": True,
            "explicit_unit_ranges": True,
            "mixed_backends": False,
            "remote_compute": True,
            "coordinator_outside_compute": True,
            "max_compute_devices": 2,
            "max_compute_nodes": 2,
            "max_segments": 2,
        },
        "route_modes": ["direct"],
        "coordinator_nodes": ["n0"],
        "devices": [{"id": "g0", "node": "n0", "backend": "cuda"}],
    })


def evidence(a=None, m=None):
    a = a or adapter()
    m = m or manifest()
    return QualificationEvidence.parse({
        "qualification_schema": "tensormeld/qualification-evidence-v1",
        "evidence_id": "lab-run-1",
        "level": "E3",
        "adapter_id": a.adapter_id,
        "adapter_capabilities_sha256": a.fingerprint,
        "engine_revision": a.engine_revision,
        "worker_artifact_sha256": W,
        "model_manifest_sha256": m.manifest_sha256,
        "config_sha256": C,
        "device_ids": ["g0"],
        "workload": {
            "context_tokens": 8192,
            "max_output_tokens": 1024,
            "concurrency": 1,
        },
        "result": "passed",
        "observed_at": "2026-10-01T12:00:00Z",
        "tests": ["full-model-correctness"],
    })


class ModelManifestTests(unittest.TestCase):
    def test_complete_index_plus_exact_shard_hashes_builds_manifest(self):
        m = manifest()
        self.assertEqual(m.architecture, "test")
        self.assertEqual(len(m.files), 2)
        self.assertEqual(len(m.manifest_sha256), 64)

    def test_partial_index_cannot_become_manifest(self):
        x = index()
        x["complete_shard_set"] = False
        with self.assertRaises(ValidationError):
            manifest_from_gguf_index(
                x, model_id="m", revision="r", tokenizer_ref="t",
                shard_sha256={"m-00001.gguf": H, "m-00002.gguf": H},
            )

    def test_exact_shard_set_required(self):
        with self.assertRaises(ValidationError):
            manifest_from_gguf_index(
                index(), model_id="m", revision="r", tokenizer_ref="t",
                shard_sha256={"m-00001.gguf": H},
            )

    def test_bad_digest_rejected(self):
        raw = {
            "model_manifest_schema": "tensormeld/model-manifest-v1",
            "model_id": "m",
            "revision": "r",
            "format": "gguf",
            "architecture": "a",
            "tokenizer_ref": "t",
            "chat_template_ref": None,
            "files": [{"name": "x.gguf", "size_bytes": 1, "sha256": "bad"}],
            "tensor_index_sha256": I,
            "tensor_count": 1,
            "tensor_payload_bytes": 1,
        }
        with self.assertRaises(ValidationError):
            ModelManifest.parse(raw)

    def test_manifest_identity_changes_with_tokenizer(self):
        m = manifest()
        other = manifest_from_gguf_index(
            index(), model_id="org/model", revision="rev1", tokenizer_ref="other",
            shard_sha256={"m-00001.gguf": H, "m-00002.gguf": "e" * 64},
        )
        self.assertNotEqual(m.manifest_sha256, other.manifest_sha256)


class QualificationEvidenceTests(unittest.TestCase):
    def test_exact_evidence_applies_but_does_not_make_session_executable(self):
        a, m = adapter(), manifest()
        e = evidence(a, m)
        r = evidence_applies(
            e, adapter=a, model=m, config_sha256=C, device_ids=["g0"],
            context_tokens=8192, max_output_tokens=1024, concurrency=1,
        )
        self.assertTrue(r["applies"])
        self.assertTrue(r["qualified"])
        self.assertFalse(r["executable"])

    def test_model_change_invalidates_evidence(self):
        a, m = adapter(), manifest()
        e = evidence(a, m)
        other = manifest_from_gguf_index(
            index(), model_id="org/model", revision="rev2",
            tokenizer_ref="tokenizer:rev1",
            shard_sha256={"m-00001.gguf": H, "m-00002.gguf": "e" * 64},
        )
        r = evidence_applies(
            e, adapter=a, model=other, config_sha256=C, device_ids=["g0"],
            context_tokens=8192, max_output_tokens=1024, concurrency=1,
        )
        self.assertIn("MODEL_MANIFEST_MISMATCH", r["reasons"])

    def test_adapter_capability_change_invalidates_evidence(self):
        a, m = adapter(), manifest()
        e = evidence(a, m)
        changed = AdapterCapabilities.parse({
            "adapter_schema": "tensormeld/adapter-capabilities-v1",
            "adapter_id": "native", "engine": "engine",
            "engine_revision": "rev-engine",
            "placement": {
                "strategies": ["whole_blocks"], "exact_owner_binding": True,
                "explicit_unit_ranges": True, "mixed_backends": False,
                "remote_compute": True, "coordinator_outside_compute": True,
                "max_compute_devices": 3, "max_compute_nodes": 2,
                "max_segments": 2,
            },
            "route_modes": ["direct"], "coordinator_nodes": ["n0"],
            "devices": [{"id": "g0", "node": "n0", "backend": "cuda"}],
        })
        r = evidence_applies(
            e, adapter=changed, model=m, config_sha256=C, device_ids=["g0"],
            context_tokens=8192, max_output_tokens=1024, concurrency=1,
        )
        self.assertIn("ADAPTER_CAPABILITIES_MISMATCH", r["reasons"])

    def test_workload_change_invalidates_evidence(self):
        a, m = adapter(), manifest()
        r = evidence_applies(
            evidence(a, m), adapter=a, model=m, config_sha256=C,
            device_ids=["g0"], context_tokens=4096,
            max_output_tokens=1024, concurrency=1,
        )
        self.assertIn("WORKLOAD_MISMATCH", r["reasons"])

    def test_device_identity_is_part_of_evidence(self):
        a, m = adapter(), manifest()
        r = evidence_applies(
            evidence(a, m), adapter=a, model=m, config_sha256=C,
            device_ids=["other"], context_tokens=8192,
            max_output_tokens=1024, concurrency=1,
        )
        self.assertIn("DEVICE_SET_MISMATCH", r["reasons"])

    def test_failed_evidence_never_applies(self):
        a, m = adapter(), manifest()
        e = QualificationEvidence.parse({
            "qualification_schema": "tensormeld/qualification-evidence-v1",
            "evidence_id": "fail", "level": "E3", "adapter_id": a.adapter_id,
            "adapter_capabilities_sha256": a.fingerprint,
            "engine_revision": a.engine_revision,
            "worker_artifact_sha256": W,
            "model_manifest_sha256": m.manifest_sha256,
            "config_sha256": C, "device_ids": ["g0"],
            "workload": {
                "context_tokens": 8192, "max_output_tokens": 1024,
                "concurrency": 1,
            },
            "result": "failed", "observed_at": "2026-10-01T12:00:00Z",
            "tests": ["full-model-correctness"],
        })
        r = evidence_applies(
            e, adapter=a, model=m, config_sha256=C, device_ids=["g0"],
            context_tokens=8192, max_output_tokens=1024, concurrency=1,
        )
        self.assertIn("EVIDENCE_FAILED", r["reasons"])

    def test_low_evidence_level_rejected_for_e3_requirement(self):
        a, m = adapter(), manifest()
        e = QualificationEvidence.parse({
            "qualification_schema": "tensormeld/qualification-evidence-v1",
            "evidence_id": "low", "level": "E2", "adapter_id": a.adapter_id,
            "adapter_capabilities_sha256": a.fingerprint,
            "engine_revision": a.engine_revision,
            "worker_artifact_sha256": W,
            "model_manifest_sha256": m.manifest_sha256,
            "config_sha256": C, "device_ids": ["g0"],
            "workload": {
                "context_tokens": 8192, "max_output_tokens": 1024,
                "concurrency": 1,
            },
            "result": "passed", "observed_at": "2026-10-01T12:00:00Z",
            "tests": ["operator-self-test"],
        })
        r = evidence_applies(
            e, adapter=a, model=m, config_sha256=C, device_ids=["g0"],
            context_tokens=8192, max_output_tokens=1024, concurrency=1,
        )
        self.assertIn("EVIDENCE_LEVEL_TOO_LOW", r["reasons"])

    def test_timestamp_requires_timezone(self):
        a, m = adapter(), manifest()
        raw = {
            "qualification_schema": "tensormeld/qualification-evidence-v1",
            "evidence_id": "x", "level": "E3", "adapter_id": a.adapter_id,
            "adapter_capabilities_sha256": a.fingerprint,
            "engine_revision": a.engine_revision,
            "worker_artifact_sha256": W,
            "model_manifest_sha256": m.manifest_sha256,
            "config_sha256": C, "device_ids": ["g0"],
            "workload": {
                "context_tokens": 8192, "max_output_tokens": 1024,
                "concurrency": 1,
            },
            "result": "passed", "observed_at": "2026-10-01T12:00:00",
            "tests": ["x"],
        }
        with self.assertRaises(ValidationError):
            QualificationEvidence.parse(raw)
