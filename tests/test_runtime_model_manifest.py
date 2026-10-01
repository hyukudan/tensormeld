from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.cli import main
from tensormeld.config_v2 import Config
from tensormeld.model_manifest import ModelManifest
from tensormeld.runtime_model_manifest import (
    load_runtime_model_manifest,
    parse_runtime_model_manifest,
    runtime_manifest_summary,
)
from tensormeld.schema import ValidationError
from test_config_v2 import data


def model_data():
    return {
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/model",
        "revision": "fixture-rev",
        "format": "gguf",
        "architecture": "fixture-arch",
        "tokenizer_ref": "fixture-tokenizer",
        "chat_template_ref": None,
        "files": [{"name": "model.gguf", "size_bytes": 1000, "sha256": "a" * 64}],
        "tensor_index_sha256": "b" * 64,
        "tensor_count": 10,
        "tensor_payload_bytes": 900,
    }


def model() -> ModelManifest:
    return ModelManifest.parse(model_data())


def adapter(cfg: Config) -> AdapterCapabilities:
    return AdapterCapabilities.parse({
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "fixture-adapter",
        "engine": "fixture-engine",
        "engine_revision": "fixture-engine-rev",
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
        "route_modes": ["direct", "via_coordinator"],
        "coordinator_nodes": [n.id for n in cfg.nodes],
        "devices": [
            {"id": d.id, "node": d.node, "backend": d.backend}
            for d in cfg.devices
        ],
    })


def setup():
    raw = data()
    m = model()
    raw["profiles"]["interactive"]["model_manifest_ref"] = m.manifest_sha256
    cfg = Config.parse(raw)
    a = adapter(cfg)
    return cfg, m, a


def manifest(cfg: Config, m: ModelManifest, a: AdapterCapabilities):
    devices = []
    used_nodes = set()
    for d in cfg.devices:
        devices.append({
            "id": d.id,
            "node": d.node,
            "backend": d.backend,
            "operators": ["ADD", "MATMUL", "RMS_NORM"],
        })
        used_nodes.add(d.node)
    pools = []
    for p in cfg.pools:
        if p.node not in used_nodes:
            continue
        steady = 300
        pools.append({
            "pool_ref": p.id,
            "node": p.node,
            "resident_bytes": 100,
            "state_bytes": 100,
            "workspace_peak_bytes": 100,
            "preparation_peak_bytes": steady + 100,
        })
    profile = cfg.profile_map["interactive"]
    return {
        "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
        "provenance": "fixture",
        "config_sha256": cfg.fingerprint,
        "profile": "interactive",
        "model_manifest_sha256": m.manifest_sha256,
        "adapter_id": a.adapter_id,
        "adapter_capabilities_sha256": a.fingerprint,
        "engine_revision": a.engine_revision,
        "worker_artifact_sha256": "c" * 64,
        "workload": {
            "task": profile.workload.task,
            "context_tokens": profile.workload.context_tokens,
            "max_output_tokens": profile.workload.max_output_tokens,
            "concurrency": profile.workload.max_active_requests,
        },
        "required_operators": ["ADD", "MATMUL"],
        "devices": devices,
        "physical_pool_memory": pools,
        "reservation_created": False,
        "qualified": False,
        "executable": False,
    }


class RuntimeModelManifestTests(unittest.TestCase):
    def test_exact_tuple_and_physical_pool_memory_are_accepted(self):
        cfg, m, a = setup()
        parsed = parse_runtime_model_manifest(
            manifest(cfg, m, a), config=cfg, model=m, adapter=a
        )
        summary = runtime_manifest_summary(parsed)
        self.assertTrue(summary["operator_coverage_complete"])
        self.assertFalse(summary["reservation_created"])
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])
        self.assertEqual(
            len(summary["physical_pool_memory"]),
            len({x["pool_ref"] for x in summary["physical_pool_memory"]}),
        )

    def test_shared_pool_is_represented_once_not_per_device(self):
        raw_cfg = data()
        m = model()
        raw_cfg["profiles"]["interactive"]["model_manifest_ref"] = m.manifest_sha256
        raw_cfg["devices"][1]["node"] = raw_cfg["devices"][0]["node"]
        raw_cfg["devices"][1]["pool_ref"] = raw_cfg["devices"][0]["pool_ref"]
        cfg = Config.parse(raw_cfg)
        a = adapter(cfg)
        raw = manifest(cfg, m, a)
        pool_ids = [x["pool_ref"] for x in raw["physical_pool_memory"]]
        self.assertEqual(pool_ids.count(raw_cfg["devices"][0]["pool_ref"]), 1)
        parsed = parse_runtime_model_manifest(raw, config=cfg, model=m, adapter=a)
        self.assertEqual(len(parsed.pools), 1)

        raw["physical_pool_memory"].append(copy.deepcopy(raw["physical_pool_memory"][0]))
        with self.assertRaises(ValidationError):
            parse_runtime_model_manifest(raw, config=cfg, model=m, adapter=a)

    def test_model_workload_and_adapter_identities_are_exact(self):
        cfg, m, a = setup()
        for field, mutate in (
            ("model", lambda x: x.__setitem__("model_manifest_sha256", "d" * 64)),
            ("config", lambda x: x.__setitem__("config_sha256", "e" * 64)),
            ("adapter", lambda x: x.__setitem__("adapter_id", "other")),
            ("workload", lambda x: x["workload"].__setitem__("context_tokens", x["workload"]["context_tokens"] + 1)),
        ):
            raw = manifest(cfg, m, a)
            mutate(raw)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                parse_runtime_model_manifest(raw, config=cfg, model=m, adapter=a)

    def test_operator_coverage_can_be_incomplete_without_self_qualification(self):
        cfg, m, a = setup()
        raw = manifest(cfg, m, a)
        raw["devices"][0]["operators"] = ["ADD"]
        parsed = parse_runtime_model_manifest(raw, config=cfg, model=m, adapter=a)
        summary = runtime_manifest_summary(parsed)
        self.assertFalse(summary["operator_coverage_complete"])
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])

    def test_pool_peak_is_bounded_and_preparation_covers_steady_peak(self):
        cfg, m, a = setup()
        raw = manifest(cfg, m, a)
        raw["physical_pool_memory"][0]["preparation_peak_bytes"] = 1
        with self.assertRaises(ValidationError):
            parse_runtime_model_manifest(raw, config=cfg, model=m, adapter=a)

        cap = next(
            p.reported_capacity_bytes
            for p in cfg.pools
            if p.id == raw["physical_pool_memory"][0]["pool_ref"]
        )
        if cap is not None:
            raw = manifest(cfg, m, a)
            raw["physical_pool_memory"][0]["preparation_peak_bytes"] = cap + 1
            with self.assertRaises(ValidationError):
                parse_runtime_model_manifest(raw, config=cfg, model=m, adapter=a)

    def test_fixture_provenance_never_upgrades_claim_strength(self):
        cfg, m, a = setup()
        summary = runtime_manifest_summary(
            parse_runtime_model_manifest(
                manifest(cfg, m, a), config=cfg, model=m, adapter=a
            )
        )
        self.assertEqual(summary["provenance"], "fixture")
        self.assertIn("Fixture provenance", " ".join(summary["warnings"]))
        self.assertFalse(summary["qualified"])

    def test_cli_roundtrip_and_output_safety(self):
        cfg, m, a = setup()
        raw_cfg = data()
        raw_cfg["profiles"]["interactive"]["model_manifest_ref"] = m.manifest_sha256
        raw_adapter = {
            "adapter_schema": "tensormeld/adapter-capabilities-v1",
            "adapter_id": a.adapter_id,
            "engine": a.engine,
            "engine_revision": a.engine_revision,
            "placement": {
                "strategies": sorted(a.strategies),
                "exact_owner_binding": a.exact_owner_binding,
                "explicit_unit_ranges": a.explicit_unit_ranges,
                "mixed_backends": a.mixed_backends,
                "remote_compute": a.remote_compute,
                "coordinator_outside_compute": a.coordinator_outside_compute,
                "max_compute_devices": a.max_compute_devices,
                "max_compute_nodes": a.max_compute_nodes,
                "max_segments": a.max_segments,
            },
            "route_modes": sorted(a.route_modes),
            "coordinator_nodes": sorted(a.coordinator_nodes),
            "devices": [
                {"id": d.id, "node": d.node, "backend": d.backend}
                for d in a.devices
            ],
        }
        raw_runtime = manifest(cfg, m, a)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = root / "config.json"
            model_path = root / "model.json"
            adapter_path = root / "adapter.json"
            runtime_path = root / "runtime.json"
            out = root / "out.json"
            config_path.write_text(json.dumps(raw_cfg), encoding="utf-8")
            model_path.write_text(json.dumps(model_data()), encoding="utf-8")
            adapter_path.write_text(json.dumps(raw_adapter), encoding="utf-8")
            runtime_path.write_text(json.dumps(raw_runtime), encoding="utf-8")
            self.assertEqual(main([
                "validate-runtime-manifest",
                str(config_path),
                str(model_path),
                str(adapter_path),
                str(runtime_path),
                "--out",
                str(out),
            ]), 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(result["reservation_created"])
            before = runtime_path.read_bytes()
            self.assertEqual(main([
                "validate-runtime-manifest",
                str(config_path),
                str(model_path),
                str(adapter_path),
                str(runtime_path),
                "--out",
                str(runtime_path),
            ]), 1)
            self.assertEqual(runtime_path.read_bytes(), before)

    def test_loader_rejects_duplicate_keys(self):
        cfg, m, a = setup()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.json"
            path.write_text(
                '{"runtime_manifest_schema":"x","runtime_manifest_schema":"y"}',
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                load_runtime_model_manifest(
                    path, config=cfg, model=m, adapter=a
                )


if __name__ == "__main__":
    unittest.main()
