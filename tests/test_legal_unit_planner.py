from __future__ import annotations

import copy
import unittest

from tensormeld.config_v2 import Config
from tensormeld.legal_model_units import parse_legal_model_units
from tensormeld.legal_unit_planner import plan_legal_units
from tensormeld.schema import ValidationError
from test_config_v2 import data
from test_legal_model_units import env, legal_raw


def legal_profile():
    *_, movability = env()
    return parse_legal_model_units(legal_raw(movability), movability=movability)


def config_raw():
    raw = data()
    # The legal-unit planner consumes only current policy/device/pool identity;
    # it does not claim the synthetic profile model ref is execution-qualified.
    return raw


class LegalUnitPlannerTests(unittest.TestCase):
    def setUp(self):
        self.legal = legal_profile()

    def run(self, raw=None, **kwargs):
        return plan_legal_units(Config.parse(raw or config_raw()), self.legal, **kwargs)

    def test_default_capacity_ranking_prefers_one_device(self):
        result = self.run()
        self.assertEqual(result["status"], "CANDIDATES_FOUND")
        self.assertEqual(result["best"]["compute_device_count"], 1)
        self.assertEqual(result["best"]["owners"], ["pc-gpu", "pc-gpu"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["executable"])
        self.assertEqual(result["objective"], "resident_capacity_only")

    def test_memory_pressure_forces_only_legal_cut(self):
        raw = config_raw()
        policy = next(
            p for p in raw["resource_policies"] if p["pool_ref"] == "pc-vram"
        )
        policy["allocation_cap_bytes"] = 200
        result = self.run(raw)
        self.assertEqual(result["status"], "CANDIDATES_FOUND")
        self.assertEqual(result["best"]["owners"], ["pc-gpu", "helper-a-igpu"])
        self.assertEqual(result["best"]["segments"], [
            {"device": "pc-gpu", "first_unit": 0, "last_unit_exclusive": 1},
            {"device": "helper-a-igpu", "first_unit": 1, "last_unit_exclusive": 2},
        ])

    def test_no_cut_means_memory_fit_cannot_be_invented(self):
        *_, movability = env()
        raw_units = legal_raw(movability)
        raw_units["units"][0]["cut_after"] = False
        no_cut = parse_legal_model_units(raw_units, movability=movability)
        raw = config_raw()
        next(
            p for p in raw["resource_policies"] if p["pool_ref"] == "pc-vram"
        )["allocation_cap_bytes"] = 200
        result = plan_legal_units(Config.parse(raw), no_cut)
        self.assertEqual(result["status"], "NO_CANDIDATE_IN_SEARCH_SPACE")
        self.assertIsNone(result["best"])

    def test_required_helper_device_is_respected(self):
        raw = config_raw()
        raw["selection"]["required_devices"] = ["helper-a-igpu"]
        result = self.run(raw)
        self.assertEqual(set(result["best"]["compute_devices"]), {
            "pc-gpu", "helper-a-igpu"
        })

    def test_policy_filtered_unit_without_legal_device_is_no_candidate(self):
        raw = config_raw()
        raw["selection"]["excluded_devices"] = ["pc-gpu"]
        result = self.run(raw)
        self.assertEqual(result["status"], "NO_CANDIDATE_IN_SEARCH_SPACE")
        self.assertEqual(
            result["search"]["rejections"]["unit_has_no_policy_eligible_device"],
            1,
        )

    def test_shared_pool_is_not_treated_as_two_independent_capacities(self):
        raw = config_raw()
        raw["devices"][1]["node"] = raw["devices"][0]["node"]
        raw["devices"][1]["pool_ref"] = raw["devices"][0]["pool_ref"]
        raw["resource_policies"] = [
            p for p in raw["resource_policies"]
            if p["pool_ref"] != "helper-a-ram"
        ]
        raw["resource_pools"] = [
            p for p in raw["resource_pools"]
            if p["id"] != "helper-a-ram"
        ]
        raw["selection"]["coordinator"]["allowed_nodes"] = ["pc"]
        raw["selection"]["allowed_nodes"] = ["pc"]
        raw["selection"]["required_nodes"] = []
        next(
            p for p in raw["resource_policies"] if p["pool_ref"] == "pc-vram"
        )["allocation_cap_bytes"] = 200
        result = self.run(raw)
        self.assertEqual(result["status"], "NO_CANDIDATE_IN_SEARCH_SPACE")
        self.assertGreater(
            result["search"]["rejections"].get("pool_budget_exceeded", 0),
            0,
        )

    def test_budget_exhaustion_is_search_incomplete_not_infeasible(self):
        raw = config_raw()
        raw["planning_policy"]["search_budget"]["candidate_limit"] = 1
        result = self.run(raw)
        self.assertEqual(result["status"], "SEARCH_INCOMPLETE")
        self.assertFalse(result["search"]["complete"])
        self.assertEqual(result["search"]["reason"], "work_limit")

    def test_deadline_is_search_incomplete(self):
        raw = config_raw()
        raw["planning_policy"]["search_budget"]["deadline_ms"] = 1
        times = iter([0.0, 0.01, 0.02, 0.03])
        result = self.run(raw, clock=lambda: next(times))
        self.assertEqual(result["status"], "SEARCH_INCOMPLETE")
        self.assertEqual(result["search"]["reason"], "deadline")

    def test_result_is_deterministic(self):
        a = self.run()
        b = self.run()
        self.assertEqual(a["candidates"], b["candidates"])

    def test_top_k_is_bounded(self):
        result = self.run(top_k=1)
        self.assertLessEqual(len(result["candidates"]), 1)
        for invalid in (0, 21, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                self.run(top_k=invalid)

    def test_reclaimable_bytes_are_counted_in_full(self):
        result = self.run()
        self.assertEqual(
            result["best"]["pool_tensor_resident_bytes"]["pc-vram"],
            250,
        )
        self.assertIn(
            "Reclaimable/file-backed bytes remain counted in full",
            " ".join(result["warnings"]),
        )

    def test_owner_changes_only_at_declared_cut(self):
        *_, movability = env()
        raw_units = legal_raw(movability)
        # Three units: split tensor.b into two explicit legal units is impossible
        # because every tensor must appear exactly once, so instead assert the
        # existing two-unit ownership boundary corresponds to unit.0 cut_after.
        result = self.run()
        for candidate in result["candidates"]:
            if candidate["owners"][0] != candidate["owners"][1]:
                self.assertTrue(self.legal.units[0].cut_after)


if __name__ == "__main__":
    unittest.main()
