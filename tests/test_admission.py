from __future__ import annotations

import copy
import threading
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.admission import LocalAdmissionController
from tensormeld.config_v2 import Config
from tensormeld.model_manifest import ModelManifest
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from test_config_v2 import data


def setup():
    model = ModelManifest.parse({
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/model",
        "revision": "r1",
        "format": "gguf",
        "architecture": "fixture",
        "tokenizer_ref": "tok",
        "chat_template_ref": None,
        "files": [{"name": "m.gguf", "size_bytes": 1, "sha256": "a" * 64}],
        "tensor_index_sha256": "b" * 64,
        "tensor_count": 1,
        "tensor_payload_bytes": 1,
    })
    raw = data()
    raw["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
    cfg = Config.parse(raw)
    adapter = AdapterCapabilities.parse({
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "fixture",
        "engine": "fixture",
        "engine_revision": "r1",
        "placement": {
            "strategies": ["whole_blocks"],
            "exact_owner_binding": True,
            "explicit_unit_ranges": True,
            "mixed_backends": True,
            "remote_compute": True,
            "coordinator_outside_compute": True,
            "max_compute_devices": 128,
            "max_compute_nodes": 64,
            "max_segments": 256,
        },
        "route_modes": ["direct"],
        "coordinator_nodes": [n.id for n in cfg.nodes],
        "devices": [
            {"id": d.id, "node": d.node, "backend": d.backend}
            for d in cfg.devices
        ],
    })
    p = cfg.profile_map["interactive"]
    manifest_raw = {
        "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
        "provenance": "fixture",
        "config_sha256": cfg.fingerprint,
        "profile": "interactive",
        "model_manifest_sha256": model.manifest_sha256,
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": "c" * 64,
        "workload": {
            "task": p.workload.task,
            "context_tokens": p.workload.context_tokens,
            "max_output_tokens": p.workload.max_output_tokens,
            "concurrency": p.workload.max_active_requests,
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
        ],
        "physical_pool_memory": [
            {
                "pool_ref": pool.id,
                "node": pool.node,
                "resident_bytes": 0,
                "state_bytes": 0,
                "workspace_peak_bytes": 0,
                "preparation_peak_bytes": 1024,
            }
            for pool in cfg.pools
        ],
        "reservation_created": False,
        "qualified": False,
        "executable": False,
    }
    manifest = parse_runtime_model_manifest(
        manifest_raw, config=cfg, model=model, adapter=adapter
    )
    return cfg, manifest


def snapshot(cfg, observation_id, available=None, reflected=()):
    pools = {}
    for p in cfg.pools:
        value = p.reported_capacity_bytes or 0
        if available and p.id in available:
            value = available[p.id]
        pools[p.id] = {"available_bytes": value}
    return {
        "admission_snapshot_schema": "tensormeld/admission-snapshot-v1",
        "observation_id": observation_id,
        "runtime_observation": {
            "runtime_observation_schema": "tensormeld/runtime-observation-v1",
            "config_sha256": cfg.fingerprint,
            "devices": {
                d.id: {"backend": d.backend, "state": "ready"}
                for d in cfg.devices
            },
            "pools": pools,
            "qualified": False,
            "executable": False,
        },
        "reflected_lease_ids": list(reflected),
    }


class LocalAdmissionTests(unittest.TestCase):
    def test_atomic_reservation_and_release(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        r = c.reserve(
            lease_id="a", config=cfg, manifest=manifest, snapshot=snapshot(cfg, "o1")
        )
        self.assertEqual(r["status"], "RESERVED")
        self.assertTrue(r["reservation_created"])
        self.assertFalse(r["launch_authorized"])
        self.assertEqual(len(c.active_leases()), 1)
        self.assertEqual(c.release("a")["status"], "RELEASED")
        self.assertEqual(c.release("a")["status"], "ALREADY_RELEASED")
        self.assertEqual(c.active_leases(), ())

    def test_pending_leases_are_subtracted_when_not_reflected(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        pool = manifest.pools[0].pool
        policy = next(p for p in cfg.resource_policies if p.pool == pool)
        required = manifest.pools[0].preparation_peak_bytes
        available = policy.safety_headroom_bytes + required
        obs = {pool: available}
        self.assertEqual(
            c.reserve(
                lease_id="a", config=cfg, manifest=manifest,
                snapshot=snapshot(cfg, "o1", obs),
            )["status"],
            "RESERVED",
        )
        second = c.reserve(
            lease_id="b", config=cfg, manifest=manifest,
            snapshot=snapshot(cfg, "o2", obs),
        )
        self.assertEqual(second["status"], "REJECTED")

    def test_reflected_lease_is_not_subtracted_twice(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        self.assertEqual(
            c.reserve(
                lease_id="a", config=cfg, manifest=manifest,
                snapshot=snapshot(cfg, "o1"),
            )["status"],
            "RESERVED",
        )
        result = c.reserve(
            lease_id="b", config=cfg, manifest=manifest,
            snapshot=snapshot(cfg, "o2", reflected=("a",)),
        )
        self.assertEqual(result["status"], "RESERVED")

    def test_launch_requires_new_observation_and_rechecks_capacity(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        c.reserve(
            lease_id="a", config=cfg, manifest=manifest, snapshot=snapshot(cfg, "o1")
        )
        with self.assertRaises(ValidationError):
            c.launch_recheck(
                lease_id="a", config=cfg, manifest=manifest,
                snapshot=snapshot(cfg, "o1"),
            )

        pool = manifest.pools[0].pool
        policy = next(p for p in cfg.resource_policies if p.pool == pool)
        low = {pool: policy.safety_headroom_bytes}
        rejected = c.launch_recheck(
            lease_id="a", config=cfg, manifest=manifest,
            snapshot=snapshot(cfg, "o2", low),
        )
        self.assertEqual(rejected["status"], "REJECTED")

        admitted = c.launch_recheck(
            lease_id="a", config=cfg, manifest=manifest,
            snapshot=snapshot(cfg, "o3"),
        )
        self.assertEqual(admitted["status"], "LAUNCH_ADMITTED")
        self.assertTrue(admitted["launch_authorized"])
        self.assertFalse(admitted["qualified"])
        self.assertFalse(admitted["executable"])

    def test_unknown_reflected_lease_fails_closed(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        with self.assertRaises(ValidationError):
            c.reserve(
                lease_id="a", config=cfg, manifest=manifest,
                snapshot=snapshot(cfg, "o1", reflected=("ghost",)),
            )

    def test_duplicate_lease_id_is_rejected(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        c.reserve(
            lease_id="a", config=cfg, manifest=manifest, snapshot=snapshot(cfg, "o1")
        )
        with self.assertRaises(ValidationError):
            c.reserve(
                lease_id="a", config=cfg, manifest=manifest,
                snapshot=snapshot(cfg, "o2"),
            )

    def test_concurrent_reservations_are_serialized(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        pool = manifest.pools[0].pool
        policy = next(p for p in cfg.resource_policies if p.pool == pool)
        required = manifest.pools[0].preparation_peak_bytes
        obs = {pool: policy.safety_headroom_bytes + required}
        barrier = threading.Barrier(3)
        statuses = []
        guard = threading.Lock()

        def worker(lease_id):
            barrier.wait()
            result = c.reserve(
                lease_id=lease_id, config=cfg, manifest=manifest,
                snapshot=snapshot(cfg, f"obs-{lease_id}", obs),
            )
            with guard:
                statuses.append(result["status"])

        threads = [
            threading.Thread(target=worker, args=("a",)),
            threading.Thread(target=worker, args=("b",)),
        ]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join()
        self.assertEqual(sorted(statuses), ["REJECTED", "RESERVED"])

    def test_shared_pool_is_charged_once_from_manifest(self):
        cfg, manifest = setup()
        c = LocalAdmissionController()
        lease = c.reserve(
            lease_id="a", config=cfg, manifest=manifest, snapshot=snapshot(cfg, "o1")
        )
        self.assertEqual(
            set(lease["pool_bytes"]),
            {pool.pool for pool in manifest.pools},
        )


if __name__ == "__main__":
    unittest.main()
