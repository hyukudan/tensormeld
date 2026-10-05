from __future__ import annotations

from dataclasses import replace
import unittest

from tensormeld.config_v2 import Config
from tensormeld.llamacpp_package import build_llamacpp_package_identity
from tensormeld.placement_calibration import (
    next_calibration_round,
    parse_placement_calibration,
    rank_applicable_calibrations,
    validate_placement_calibration,
)
from tensormeld.planner_v2 import plan_v2
from tensormeld.planning_contract import PlanningInput
from tensormeld.schema import ValidationError
from test_llamacpp_package import probe
from test_planner_v2 import fixture
from test_target_host_qualification import runtime_identity


class PlacementCalibrationTests(unittest.TestCase):
    def setUp(self):
        raw_config, raw_planning = fixture(count=3, units=4)
        self.config = Config.parse(raw_config)
        self.planning = PlanningInput.parse(raw_planning, self.config)
        result = plan_v2(self.config, self.planning, top_k=10)
        self.candidates = result["candidates"]
        self.assertGreaterEqual(len(self.candidates), 3)

        self.package = build_llamacpp_package_identity(
            cli_probe=probe("1" * 64, "llama-cli"),
            server_probe=probe("2" * 64, "llama-server"),
            backend_libraries=[
                {"name": "ggml-fixture", "sha256": "3" * 64},
            ],
        )
        self.identities = tuple(
            runtime_identity(
                device.node,
                device.id,
                self.package.llama_server_sha256,
            )
            for device in self.config.devices
        )

    def raw(self, candidate, *, source="fixture", prefill_us=1000, decode_us=4000):
        candidate_nodes = set(candidate["compute_nodes"])
        pool_peaks = [
            {"pool_ref": pool.id, "peak_bytes": 100}
            for pool in self.config.pools
            if pool.node in candidate_nodes
        ]
        by_device = {
            identity.tensormeld_device_id: identity
            for identity in self.identities
        }
        return {
            "calibration_schema": "tensormeld/placement-calibration-v1",
            "measurement_source": source,
            "config_sha256": self.config.fingerprint,
            "planning_input_sha256": self.planning.fingerprint,
            "profile": self.config.installation.default_profile,
            "model_manifest_sha256": self.planning.manifest_ref,
            "package_sha256": self.package.fingerprint,
            "candidate_plan_sha256": candidate["plan_sha256"],
            "runtime_identity_sha256": [
                by_device[device].identity_sha256
                for device in sorted(by_device)
            ],
            "target_workload": {
                "prefill_tokens": self.planning.context_tokens,
                "decode_tokens": self.planning.max_output_tokens,
            },
            "measurement": {
                "prefill_tokens": 1024,
                "decode_tokens": 128,
                "prefill_elapsed_us": prefill_us,
                "decode_elapsed_us": decode_us,
            },
            "physical_pool_peak_bytes": pool_peaks,
            "qualified": False,
            "executable": False,
        }

    def parse(self, candidate, **kwargs):
        return parse_placement_calibration(
            self.raw(candidate, **kwargs),
            config=self.config,
            planning=self.planning,
            candidate=candidate,
            package=self.package,
            runtime_identities=self.identities,
        )

    def test_objective_normalizes_prefill_and_decode_to_target_workload(self):
        calibration = self.parse(
            self.candidates[0],
            prefill_us=1000,
            decode_us=4000,
        )
        # 1000 * 8192/1024 + 4000 * 1024/128 = 8,000 + 32,000.
        self.assertEqual(calibration.objective_us, 40_000)
        self.assertFalse(calibration.as_record()["qualified"])
        self.assertFalse(calibration.as_record()["executable"])
        validate_placement_calibration(calibration)

    def test_native_ranking_ignores_faster_fixture_measurement(self):
        fixture_fast = self.parse(
            self.candidates[0],
            source="fixture",
            prefill_us=10,
            decode_us=10,
        )
        native_slow = self.parse(
            self.candidates[1],
            source="native-target",
            prefill_us=1000,
            decode_us=2000,
        )
        ranked = rank_applicable_calibrations(
            [fixture_fast, native_slow],
            require_native=True,
        )
        self.assertEqual(
            [item.candidate_plan_sha256 for item in ranked],
            [native_slow.candidate_plan_sha256],
        )

    def test_native_ranking_uses_workload_specific_prefill_plus_decode(self):
        a = self.parse(
            self.candidates[0],
            source="native-target",
            prefill_us=1000,
            decode_us=5000,
        )
        b = self.parse(
            self.candidates[1],
            source="native-target",
            prefill_us=4000,
            decode_us=1000,
        )
        ranked = rank_applicable_calibrations([a, b], require_native=True)
        expected = min([a, b], key=lambda item: item.objective_us)
        self.assertEqual(ranked[0].candidate_plan_sha256, expected.candidate_plan_sha256)

    def test_runtime_environment_change_invalidates_measurement(self):
        raw = self.raw(self.candidates[0])
        changed = list(self.identities)
        changed[0] = runtime_identity(
            changed[0].node_id,
            changed[0].tensormeld_device_id,
            self.package.llama_server_sha256,
            driver_version="changed",
        )
        with self.assertRaises(ValidationError):
            parse_placement_calibration(
                raw,
                config=self.config,
                planning=self.planning,
                candidate=self.candidates[0],
                package=self.package,
                runtime_identities=tuple(changed),
            )

    def test_runtime_environment_must_use_package_server_artifact(self):
        wrong = list(self.identities)
        wrong[0] = runtime_identity(
            wrong[0].node_id,
            wrong[0].tensormeld_device_id,
            "f" * 64,
        )
        with self.assertRaises(ValidationError):
            parse_placement_calibration(
                self.raw(self.candidates[0]),
                config=self.config,
                planning=self.planning,
                candidate=self.candidates[0],
                package=self.package,
                runtime_identities=tuple(wrong),
            )

    def test_pool_from_unused_candidate_node_is_rejected(self):
        candidate = next(
            item for item in self.candidates if item["compute_node_count"] == 1
        )
        used_nodes = set(candidate["compute_nodes"])
        unused_pool = next(
            pool for pool in self.config.pools if pool.node not in used_nodes
        )
        raw = self.raw(candidate)
        raw["physical_pool_peak_bytes"].append({
            "pool_ref": unused_pool.id,
            "peak_bytes": 1,
        })
        with self.assertRaises(ValidationError):
            parse_placement_calibration(
                raw,
                config=self.config,
                planning=self.planning,
                candidate=candidate,
                package=self.package,
                runtime_identities=self.identities,
            )

    def test_mixed_calibration_environments_cannot_be_ranked(self):
        a = self.parse(self.candidates[0], source="native-target")
        b = replace(
            self.parse(self.candidates[1], source="native-target"),
            package_sha256="0" * 64,
        )
        # Recompute a valid internal fingerprint to make the failure about mixed
        # environment identity rather than obvious object tampering.
        rec = b.as_record()
        rec.pop("fingerprint")
        import hashlib, json
        b = replace(
            b,
            fingerprint=hashlib.sha256(
                json.dumps(
                    rec,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )
        with self.assertRaises(ValidationError):
            rank_applicable_calibrations([a, b], require_native=True)

    def test_coarse_then_local_refinement_is_bounded_and_deterministic(self):
        candidates = self.candidates
        coarse = next_calibration_round(
            candidates,
            measured_plan_sha256=set(),
            ranked_measurements=[],
            max_coarse_samples=3,
            refine_radius=1,
        )
        self.assertEqual(coarse["phase"], "coarse")
        self.assertLessEqual(len(coarse["selected_plan_sha256"]), 3)
        self.assertIn(candidates[0]["plan_sha256"], coarse["selected_plan_sha256"])
        self.assertIn(candidates[-1]["plan_sha256"], coarse["selected_plan_sha256"])

        measured = set(coarse["selected_plan_sha256"])
        winner_plan = coarse["selected_plan_sha256"][1]
        winner_candidate = next(
            item for item in candidates if item["plan_sha256"] == winner_plan
        )
        winner = self.parse(
            winner_candidate,
            source="native-target",
            prefill_us=100,
            decode_us=100,
        )
        refine = next_calibration_round(
            candidates,
            measured_plan_sha256=measured,
            ranked_measurements=[winner],
            max_coarse_samples=3,
            refine_radius=1,
        )
        self.assertIn(refine["phase"], {"refine", "complete"})
        if refine["phase"] == "refine":
            winner_index = [
                item["plan_sha256"] for item in candidates
            ].index(winner_plan)
            allowed = {
                candidates[i]["plan_sha256"]
                for i in range(
                    max(0, winner_index - 1),
                    min(len(candidates), winner_index + 2),
                )
            }
            self.assertTrue(set(refine["selected_plan_sha256"]) <= allowed)
            self.assertTrue(
                set(refine["selected_plan_sha256"]).isdisjoint(measured)
            )

    def test_coarse_measured_without_applicable_ranking_is_blocked(self):
        first_round = next_calibration_round(
            self.candidates,
            measured_plan_sha256=set(),
            ranked_measurements=[],
            max_coarse_samples=3,
        )
        blocked = next_calibration_round(
            self.candidates,
            measured_plan_sha256=set(first_round["selected_plan_sha256"]),
            ranked_measurements=[],
            max_coarse_samples=3,
        )
        self.assertEqual(blocked["phase"], "blocked")
        self.assertFalse(blocked["complete"])


if __name__ == "__main__":
    unittest.main()
