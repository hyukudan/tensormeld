from __future__ import annotations

import copy
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.config_v2 import Config
from tensormeld.model_manifest import ModelManifest
from tensormeld.planning_contract import PlanningInput
from tensormeld.planner_v2 import plan_v2
from tensormeld.qualification import QualificationEvidence
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from tensormeld.whole_block_execution import (
    ReferenceWholeBlockSession,
    accept_execution_bundle,
)
from test_adapter_contract import capability
from test_planner_v2 import fixture


WORKER_SHA = "f" * 64


def model() -> ModelManifest:
    return ModelManifest.parse({
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/executable",
        "revision": "fixture-rev",
        "format": "gguf",
        "architecture": "fixture",
        "tokenizer_ref": "fixture-tokenizer",
        "chat_template_ref": None,
        "files": [
            {"name": "fixture.gguf", "size_bytes": 1, "sha256": "a" * 64}
        ],
        "tensor_index_sha256": "b" * 64,
        "tensor_count": 1,
        "tensor_payload_bytes": 1,
    })


def setup_fixture():
    raw_config, raw_planning = fixture(count=2, units=3)
    m = model()
    raw_config["profiles"]["interactive"]["model_manifest_ref"] = m.manifest_sha256
    raw_config["profiles"]["interactive"]["execution_mode"] = "distributed"
    raw_planning["manifest_ref"] = m.manifest_sha256

    cfg = Config.parse(raw_config)
    planning = PlanningInput.parse(raw_planning, cfg)
    candidate = plan_v2(cfg, planning)["best"]
    assert candidate is not None

    adapter = AdapterCapabilities.parse(capability(raw_config))
    profile = cfg.profile_map["interactive"]

    used_devices = set(candidate["compute_devices"])
    used_nodes = set(candidate["compute_nodes"])
    runtime_manifest = parse_runtime_model_manifest({
        "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
        "provenance": "fixture",
        "config_sha256": cfg.fingerprint,
        "profile": "interactive",
        "model_manifest_sha256": m.manifest_sha256,
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": WORKER_SHA,
        "workload": {
            "task": profile.workload.task,
            "context_tokens": profile.workload.context_tokens,
            "max_output_tokens": profile.workload.max_output_tokens,
            "concurrency": profile.workload.max_active_requests,
        },
        "required_operators": ["ADD"],
        "devices": [
            {
                "id": d.id,
                "node": d.node,
                "backend": d.backend,
                "operators": ["ADD"],
            }
            for d in cfg.devices
            if d.id in used_devices
        ],
        "physical_pool_memory": [
            {
                "pool_ref": p.id,
                "node": p.node,
                "resident_bytes": 10,
                "state_bytes": 10,
                "workspace_peak_bytes": 10,
                "preparation_peak_bytes": 40,
            }
            for p in cfg.pools
            if p.node in used_nodes
        ],
        "reservation_created": False,
        "qualified": False,
        "executable": False,
    }, config=cfg, model=m, adapter=adapter)

    qualification = QualificationEvidence.parse({
        "qualification_schema": "tensormeld/qualification-evidence-v1",
        "evidence_id": "fixture-e3",
        "level": "E3",
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": WORKER_SHA,
        "model_manifest_sha256": m.manifest_sha256,
        "config_sha256": cfg.fingerprint,
        "device_ids": list(candidate["compute_devices"]),
        "workload": {
            "context_tokens": profile.workload.context_tokens,
            "max_output_tokens": profile.workload.max_output_tokens,
            "concurrency": profile.workload.max_active_requests,
        },
        "result": "passed",
        "observed_at": "2026-10-03T10:00:00Z",
        "tests": ["fixture-full-model-correctness"],
    })

    readiness = [
        {
            "evidence_schema": "tensormeld/backend-readiness-evidence-v2",
            "evidence_sha256": f"{i + 1:064x}",
            "tensormeld_device_id": device,
            "identity_applicable": True,
            "backend_readiness_recorded": True,
            "runtime_identity_sha256": f"{i + 101:064x}",
            "requires_live_runtime_recheck": False,
            "runtime_ready": True,
            "reservation_created": False,
            "qualified": False,
            "executable": False,
        }
        for i, device in enumerate(candidate["compute_devices"])
    ]

    admissions = [
        {
            "result_schema": "tensormeld/local-launch-recheck-v1",
            "status": "LAUNCH_ADMITTED",
            "lease_id": f"lease-{node}",
            "node_id": node,
            "lease_sha256": f"{i + 201:064x}",
            "observation_id": f"obs-{node}",
            "reservation_created": True,
            "launch_authorized": True,
            "qualified": False,
            "executable": False,
        }
        for i, node in enumerate(candidate["compute_nodes"])
    ]
    return (
        cfg, planning, candidate, adapter, m, qualification,
        runtime_manifest, readiness, admissions,
    )


def accepted_bundle():
    args = setup_fixture()
    bundle = accept_execution_bundle(
        config=args[0],
        planning=args[1],
        candidate=args[2],
        adapter=args[3],
        model=args[4],
        qualification_evidence=args[5],
        runtime_manifest=args[6],
        backend_readiness_results=args[7],
        launch_admissions=args[8],
    )
    return args, bundle


class DeterministicBackend:
    def __init__(self):
        self.calls = []

    def execute_segment(self, *, device_id, unit_ids, payload):
        self.calls.append((device_id, unit_ids, payload))
        suffix = ("|" + device_id + ":" + ",".join(unit_ids)).encode("utf-8")
        return payload + suffix


class WholeBlockExecutionTests(unittest.TestCase):
    def test_bundle_is_immutable_identity_of_all_execution_gates(self):
        args, bundle = accepted_bundle()
        record = bundle.as_record()
        self.assertTrue(record["qualified"])
        self.assertTrue(record["execution_authorized"])
        self.assertFalse(record["real_model_inference"])
        self.assertEqual(
            tuple(record["compute_devices"]),
            tuple(args[2]["compute_devices"]),
        )
        self.assertEqual(
            {item["node_id"] for item in record["launch_leases"]},
            set(args[2]["compute_nodes"]),
        )
        self.assertEqual(len(record["bundle_sha256"]), 64)

    def test_reference_adapter_executes_exact_whole_block_segments(self):
        _, bundle = accepted_bundle()
        backend = DeterministicBackend()
        session = ReferenceWholeBlockSession(bundle, backend)
        result = session.run(b"seed")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(session.state, "completed")
        self.assertEqual(
            [(x["device_id"], tuple(x["unit_ids"])) for x in result["segments_executed"]],
            [(d, bundle.unit_ids[first:last]) for d, first, last in bundle.segments],
        )
        self.assertEqual(len(backend.calls), len(bundle.segments))
        self.assertFalse(result["real_model_inference"])
        self.assertEqual(session.release()["status"], "RELEASED")
        self.assertEqual(session.release()["status"], "ALREADY_RELEASED")

    def test_cancel_before_run_blocks_execution(self):
        _, bundle = accepted_bundle()
        backend = DeterministicBackend()
        session = ReferenceWholeBlockSession(bundle, backend)
        session.cancel()
        self.assertEqual(session.state, "cancelled")
        with self.assertRaises(ValidationError):
            session.run(b"seed")
        self.assertEqual(backend.calls, [])
        self.assertEqual(session.release()["status"], "RELEASED")

    def test_tampered_planner_candidate_is_rejected_even_with_old_hash(self):
        args = list(setup_fixture())
        candidate = copy.deepcopy(args[2])
        candidate["owners"][0] = candidate["owners"][-1]
        args[2] = candidate
        with self.assertRaises(ValidationError):
            accept_execution_bundle(
                config=args[0], planning=args[1], candidate=args[2],
                adapter=args[3], model=args[4],
                qualification_evidence=args[5], runtime_manifest=args[6],
                backend_readiness_results=args[7], launch_admissions=args[8],
            )

    def test_e3_must_apply_to_exact_worker_and_workload(self):
        args = list(setup_fixture())
        bad = QualificationEvidence.parse({
            "qualification_schema": "tensormeld/qualification-evidence-v1",
            "evidence_id": "wrong-worker",
            "level": "E3",
            "adapter_id": args[3].adapter_id,
            "adapter_capabilities_sha256": args[3].fingerprint,
            "engine_revision": args[3].engine_revision,
            "worker_artifact_sha256": "0" * 64,
            "model_manifest_sha256": args[4].manifest_sha256,
            "config_sha256": args[0].fingerprint,
            "device_ids": list(args[2]["compute_devices"]),
            "workload": args[6].workload,
            "result": "passed",
            "observed_at": "2026-10-03T10:00:00Z",
            "tests": ["fixture"],
        })
        args[5] = bad
        with self.assertRaises(ValidationError):
            accept_execution_bundle(
                config=args[0], planning=args[1], candidate=args[2],
                adapter=args[3], model=args[4],
                qualification_evidence=args[5], runtime_manifest=args[6],
                backend_readiness_results=args[7], launch_admissions=args[8],
            )

    def test_readiness_must_cover_every_compute_device(self):
        args = list(setup_fixture())
        args[7] = args[7][:-1]
        with self.assertRaises(ValidationError):
            accept_execution_bundle(
                config=args[0], planning=args[1], candidate=args[2],
                adapter=args[3], model=args[4],
                qualification_evidence=args[5], runtime_manifest=args[6],
                backend_readiness_results=args[7], launch_admissions=args[8],
            )

    def test_launch_admission_must_cover_every_compute_node(self):
        args = list(setup_fixture())
        args[8] = args[8][:-1]
        with self.assertRaises(ValidationError):
            accept_execution_bundle(
                config=args[0], planning=args[1], candidate=args[2],
                adapter=args[3], model=args[4],
                qualification_evidence=args[5], runtime_manifest=args[6],
                backend_readiness_results=args[7], launch_admissions=args[8],
            )


if __name__ == "__main__":
    unittest.main()
