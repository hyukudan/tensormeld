from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.cli import main
from tensormeld.predictive_memory import (
    load_predictive_memory_profile,
    parse_predictive_memory_profile,
    predictive_memory_summary,
)
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from test_config_v2 import data
from test_runtime_model_manifest import setup, manifest, model_data


def predictive_raw(runtime):
    pools = []
    for pool in runtime.pools:
        reclaimable = pool.resident_bytes // 2
        pools.append({
            "pool_ref": pool.pool,
            "node": pool.node,
            "hard_resident_bytes": pool.resident_bytes - reclaimable,
            "reclaimable_file_backed_bytes": reclaimable,
            "persistent_state_bytes": pool.state_bytes,
            "workspace_peak_bytes": pool.workspace_peak_bytes,
            "staging_peak_bytes": (
                pool.preparation_peak_bytes - pool.steady_peak_bytes
            ),
        })
    return {
        "predictive_memory_schema": "tensormeld/predictive-memory-v1",
        "provenance": "fixture",
        "runtime_manifest_sha256": runtime.fingerprint,
        "physical_pool_memory": pools,
        "qualified": False,
        "executable": False,
    }


class PredictiveMemoryTests(unittest.TestCase):
    def setUp(self):
        self.config, self.model, self.adapter = setup()
        self.runtime = parse_runtime_model_manifest(
            manifest(self.config, self.model, self.adapter),
            config=self.config,
            model=self.model,
            adapter=self.adapter,
        )

    def parse(self, raw=None):
        return parse_predictive_memory_profile(
            raw or predictive_raw(self.runtime),
            config=self.config,
            runtime_manifest=self.runtime,
        )

    def test_exact_refinement_preserves_runtime_preparation_peak(self):
        profile = self.parse()
        by_pool = {p.pool: p for p in self.runtime.pools}
        for classified in profile.pools:
            runtime = by_pool[classified.pool]
            self.assertEqual(
                classified.worst_case_physical_bytes,
                runtime.preparation_peak_bytes,
            )
            self.assertEqual(
                classified.hard_resident_bytes
                + classified.reclaimable_file_backed_bytes,
                runtime.resident_bytes,
            )
        summary = predictive_memory_summary(profile)
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])

    def test_reclaimable_is_not_treated_as_free_in_worst_case(self):
        profile = self.parse()
        for pool in profile.pools:
            self.assertEqual(
                pool.worst_case_physical_bytes,
                pool.hard_peak_bytes + pool.reclaimable_file_backed_bytes,
            )
            self.assertGreaterEqual(
                pool.worst_case_physical_bytes,
                pool.hard_peak_bytes,
            )

    def test_resident_partition_must_reconcile_exactly(self):
        raw = predictive_raw(self.runtime)
        raw["physical_pool_memory"][0]["hard_resident_bytes"] += 1
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_state_workspace_and_staging_must_match_runtime_manifest(self):
        for field in (
            "persistent_state_bytes",
            "workspace_peak_bytes",
            "staging_peak_bytes",
        ):
            raw = predictive_raw(self.runtime)
            raw["physical_pool_memory"][0][field] += 1
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_every_runtime_pool_must_be_classified_once(self):
        raw = predictive_raw(self.runtime)
        raw["physical_pool_memory"].pop()
        with self.assertRaises(ValidationError):
            self.parse(raw)

        raw = predictive_raw(self.runtime)
        raw["physical_pool_memory"].append(
            copy.deepcopy(raw["physical_pool_memory"][0])
        )
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_runtime_manifest_identity_is_exact(self):
        raw = predictive_raw(self.runtime)
        raw["runtime_manifest_sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_profile_cannot_self_promote(self):
        for field in ("qualified", "executable"):
            raw = predictive_raw(self.runtime)
            raw[field] = True
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_order_does_not_change_fingerprint(self):
        a = predictive_raw(self.runtime)
        b = predictive_raw(self.runtime)
        b["physical_pool_memory"].reverse()
        self.assertEqual(self.parse(a).fingerprint, self.parse(b).fingerprint)

    def test_fixture_provenance_remains_explicit(self):
        profile = self.parse()
        summary = predictive_memory_summary(profile)
        self.assertEqual(summary["provenance"], "fixture")
        self.assertIn("Fixture provenance", " ".join(summary["warnings"]))

    def test_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictive.json"
            path.write_text(
                '{"predictive_memory_schema":"a","predictive_memory_schema":"b"}',
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                load_predictive_memory_profile(
                    path,
                    config=self.config,
                    runtime_manifest=self.runtime,
                )

    def test_cli_roundtrip_and_output_safety(self):
        raw_cfg = data()
        raw_cfg["profiles"]["interactive"]["model_manifest_ref"] = self.model.manifest_sha256
        raw_adapter = {
            "adapter_schema": "tensormeld/adapter-capabilities-v1",
            "adapter_id": self.adapter.adapter_id,
            "engine": self.adapter.engine,
            "engine_revision": self.adapter.engine_revision,
            "placement": {
                "strategies": sorted(self.adapter.strategies),
                "exact_owner_binding": self.adapter.exact_owner_binding,
                "explicit_unit_ranges": self.adapter.explicit_unit_ranges,
                "mixed_backends": self.adapter.mixed_backends,
                "remote_compute": self.adapter.remote_compute,
                "coordinator_outside_compute": self.adapter.coordinator_outside_compute,
                "max_compute_devices": self.adapter.max_compute_devices,
                "max_compute_nodes": self.adapter.max_compute_nodes,
                "max_segments": self.adapter.max_segments,
            },
            "route_modes": sorted(self.adapter.route_modes),
            "coordinator_nodes": [
                n.id for n in self.config.nodes
                if n.id in self.adapter.coordinator_nodes
            ],
            "devices": [
                {"id": d.id, "node": d.node, "backend": d.backend}
                for d in self.adapter.devices
            ],
        }
        raw_runtime = manifest(self.config, self.model, self.adapter)
        raw_predictive = predictive_raw(self.runtime)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = root / "config.json"
            model_path = root / "model.json"
            adapter_path = root / "adapter.json"
            runtime_path = root / "runtime.json"
            predictive_path = root / "predictive.json"
            out = root / "out.json"
            config_path.write_text(json.dumps(raw_cfg), encoding="utf-8")
            model_path.write_text(json.dumps(model_data()), encoding="utf-8")
            adapter_path.write_text(json.dumps(raw_adapter), encoding="utf-8")
            runtime_path.write_text(json.dumps(raw_runtime), encoding="utf-8")
            predictive_path.write_text(json.dumps(raw_predictive), encoding="utf-8")
            self.assertEqual(main([
                "validate-predictive-memory",
                str(config_path),
                str(model_path),
                str(adapter_path),
                str(runtime_path),
                str(predictive_path),
                "--out",
                str(out),
            ]), 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(result["qualified"])
            self.assertFalse(result["executable"])
            before = predictive_path.read_bytes()
            self.assertEqual(main([
                "validate-predictive-memory",
                str(config_path),
                str(model_path),
                str(adapter_path),
                str(runtime_path),
                str(predictive_path),
                "--out",
                str(predictive_path),
            ]), 1)
            self.assertEqual(predictive_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
