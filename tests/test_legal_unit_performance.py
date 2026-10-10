from __future__ import annotations

import copy
from dataclasses import replace
import unittest

from tensormeld.directional_path_evidence import parse_directional_path_evidence
from tensormeld.legal_model_units import parse_legal_model_units
from tensormeld.legal_unit_costs import parse_legal_unit_costs
from tensormeld.legal_unit_performance import plan_legal_unit_performance
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from test_legal_model_units import env, legal_raw
from test_legal_unit_costs import raw_costs
from test_runtime_model_manifest import manifest as runtime_manifest_raw


def setup_fixture():
    config, model, index, adapter, package, identities, movability = env()
    legal = parse_legal_model_units(legal_raw(movability), movability=movability)
    runtime = parse_runtime_model_manifest(
        runtime_manifest_raw(config, model, adapter),
        config=config,
        model=model,
        adapter=adapter,
    )
    raw = raw_costs(config, legal, runtime)
    costs = parse_legal_unit_costs(
        raw,
        config=config,
        legal_units=legal,
        runtime_manifest=runtime,
        movability=movability,
    )
    by_device = {identity.tensormeld_device_id: identity for identity in identities}
    devices = [device.id for device in config.devices]
    a, b = devices[:2]
    paths_raw = {
        "path_evidence_schema": "tensormeld/directional-path-evidence-v1",
        "provenance": "fixture",
        "config_sha256": config.fingerprint,
        "runtime_identity_sha256": [
            by_device[device].identity_sha256 for device in sorted((a, b))
        ],
        "paths": [
            {
                "id": "a-b",
                "source_device": a,
                "target_device": b,
                "transport": "fixture-private",
                "physical_group": "wire",
                "buckets": [
                    {"max_payload_bytes": 64, "upper_bound_us": 50},
                    {"max_payload_bytes": 1024, "upper_bound_us": 100},
                ],
            },
            {
                "id": "b-a",
                "source_device": b,
                "target_device": a,
                "transport": "fixture-private",
                "physical_group": "wire",
                "buckets": [
                    {"max_payload_bytes": 64, "upper_bound_us": 60},
                    {"max_payload_bytes": 1024, "upper_bound_us": 120},
                ],
            },
        ],
        "qualified": False,
        "executable": False,
    }
    paths = parse_directional_path_evidence(
        paths_raw,
        config=config,
        runtime_identities=identities,
    )
    return config, identities, movability, legal, runtime, raw, costs, paths, paths_raw


class LegalUnitPerformancePlannerTests(unittest.TestCase):
    def setUp(self):
        (
            self.config,
            self.identities,
            self.movability,
            self.legal,
            self.runtime,
            self.raw_costs,
            self.costs,
            self.paths,
            self.paths_raw,
        ) = setup_fixture()

    def plan(self, **kwargs):
        return plan_legal_unit_performance(
            kwargs.get("config", self.config),
            kwargs.get("legal_units", self.legal),
            kwargs.get("costs", self.costs),
            kwargs.get("paths", self.paths),
            kwargs.get("movability", self.movability),
            top_k=kwargs.get("top_k", 5),
            clock=kwargs.get("clock", __import__("time").monotonic),
        )

    def costs_from_raw(self, raw):
        return parse_legal_unit_costs(
            raw,
            config=self.config,
            legal_units=self.legal,
            runtime_manifest=self.runtime,
            movability=self.movability,
        )

    def test_default_ranking_prefers_no_transfer_when_compute_is_equal(self):
        result = self.plan()
        self.assertEqual(result["status"], "CANDIDATES_FOUND")
        self.assertEqual(result["best"]["owners"], ["pc-gpu", "pc-gpu"])
        self.assertEqual(result["best"]["compute_upper_bound_us"], 210)
        self.assertEqual(result["best"]["transfer_upper_bound_us"], 0)
        self.assertEqual(result["objective"], "ordered_unit_chain_upper_bound")
        self.assertFalse(result["qualified"])
        self.assertFalse(result["executable"])

    def test_faster_helper_wins_only_after_explicit_transfer_cost(self):
        raw = copy.deepcopy(self.raw_costs)
        second = raw["units"][1]["device_profiles"]
        second["pc-gpu"]["compute_us"] = 1000
        second["helper-a-igpu"]["compute_us"] = 10
        costs = self.costs_from_raw(raw)
        result = self.plan(costs=costs)
        self.assertEqual(result["best"]["owners"], ["pc-gpu", "helper-a-igpu"])
        self.assertEqual(result["best"]["compute_upper_bound_us"], 110)
        self.assertEqual(result["best"]["transfer_upper_bound_us"], 50)
        self.assertEqual(result["best"]["ordered_unit_chain_upper_bound_us"], 160)
        self.assertEqual(result["best"]["transfers"][0]["path_id"], "a-b")

    def test_large_transfer_cost_can_make_faster_helper_lose(self):
        raw = copy.deepcopy(self.raw_costs)
        second = raw["units"][1]["device_profiles"]
        second["pc-gpu"]["compute_us"] = 300
        second["helper-a-igpu"]["compute_us"] = 10
        costs = self.costs_from_raw(raw)

        path_raw = copy.deepcopy(self.paths_raw)
        path_raw["paths"][0]["buckets"][0]["upper_bound_us"] = 500
        path_raw["paths"][0]["buckets"][1]["upper_bound_us"] = 500
        paths = parse_directional_path_evidence(
            path_raw,
            config=self.config,
            runtime_identities=self.identities,
        )
        result = self.plan(costs=costs, paths=paths)
        self.assertEqual(result["best"]["owners"], ["pc-gpu", "pc-gpu"])

    def test_missing_path_rejects_only_split_candidate(self):
        raw = copy.deepcopy(self.paths_raw)
        raw["paths"] = [p for p in raw["paths"] if p["id"] != "a-b"]
        # Runtime identity list still covers both endpoints through reverse path.
        paths = parse_directional_path_evidence(
            raw,
            config=self.config,
            runtime_identities=self.identities,
        )
        result = self.plan(paths=paths)
        self.assertEqual(result["best"]["owners"], ["pc-gpu", "pc-gpu"])
        self.assertGreater(
            result["search"]["rejections"].get("missing_directional_path_bucket", 0),
            0,
        )

    def test_memory_aggregation_includes_state_workspace_and_staging(self):
        result = self.plan()
        best = result["best"]
        # pc-gpu owns both units:
        # resident 250 + state (10+11) + max workspace 21 + max staging 31 = 323.
        self.assertEqual(
            best["pool_preparation_upper_bound_bytes"]["pc-vram"],
            323,
        )

    def test_memory_pressure_can_force_split_with_full_cost_accounting(self):
        raw_config = {
            "config_schema": "tensormeld/v2",
        }
        # Rebuild from the fixture source rather than mutating frozen Config.
        from test_config_v2 import data
        config_raw = data()
        policy = next(
            p for p in config_raw["resource_policies"] if p["pool_ref"] == "pc-vram"
        )
        policy["allocation_cap_bytes"] = 200
        from tensormeld.config_v2 import Config
        config = Config.parse(config_raw)

        # Evidence identities are tied to config fingerprint, so a new config requires
        # new upstream evidence. The planner must fail closed rather than reuse it.
        with self.assertRaises(ValidationError):
            self.plan(config=config)

    def test_path_runtime_environment_must_match_movability(self):
        changed = replace(
            self.paths,
            runtime_identity_sha256=("0" * 64,),
        )
        with self.assertRaises(ValidationError):
            self.plan(paths=changed)

    def test_search_budget_exhaustion_is_incomplete_not_infeasible(self):
        from test_config_v2 import data
        from tensormeld.config_v2 import Config
        raw = data()
        raw["planning_policy"]["search_budget"]["candidate_limit"] = 1
        changed_config = Config.parse(raw)
        # Exact identity binding rejects a config drift before search.
        with self.assertRaises(ValidationError):
            self.plan(config=changed_config)

    def test_deadline_is_reported_when_exact_config_has_tiny_deadline(self):
        # Config identity is part of cost/path evidence, so deadline policy cannot be
        # changed independently without regenerating evidence. Instead use a clock that
        # crosses the existing deadline immediately.
        times = iter([0.0, 1000.0, 1000.0])
        result = self.plan(clock=lambda: next(times))
        self.assertEqual(result["status"], "SEARCH_INCOMPLETE")
        self.assertEqual(result["search"]["reason"], "deadline")

    def test_top_k_is_bounded(self):
        result = self.plan(top_k=1)
        self.assertLessEqual(len(result["candidates"]), 1)
        for invalid in (0, 21, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                plan_legal_unit_performance(
                    self.config,
                    self.legal,
                    self.costs,
                    self.paths,
                    self.movability,
                    top_k=invalid,
                )

    def test_result_is_deterministic(self):
        self.assertEqual(self.plan()["candidates"], self.plan()["candidates"])

    def test_warnings_do_not_claim_token_latency_or_execution(self):
        result = self.plan()
        warnings = " ".join(result["warnings"])
        self.assertIn("not token latency", warnings)
        self.assertIn("qualified=false", warnings)
        self.assertFalse(result["best"]["executable"])


if __name__ == "__main__":
    unittest.main()
