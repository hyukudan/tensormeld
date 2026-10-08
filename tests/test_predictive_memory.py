from __future__ import annotations

import copy
import unittest

from tensormeld.predictive_memory import (
    parse_predictive_memory_profile,
    predictive_memory_summary,
)
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from test_runtime_model_manifest import setup, manifest


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


if __name__ == "__main__":
    unittest.main()
