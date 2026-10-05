from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import unittest

from tensormeld.config_v2 import Config
from tensormeld.llamacpp_package import build_llamacpp_package_identity
from tensormeld.measured_preference import apply_measured_preference
from tensormeld.placement_calibration import parse_placement_calibration
from tensormeld.planner_v2 import plan_v2
from tensormeld.planning_contract import PlanningInput
from tensormeld.schema import ValidationError
from test_llamacpp_package import probe
from test_planner_v2 import fixture
from test_target_host_qualification import runtime_identity


class MeasuredPlannerPreferenceTests(unittest.TestCase):
    def setUp(self):
        raw_config, raw_planning = fixture(count=3, units=4)
        self.config = Config.parse(raw_config)
        self.planning = PlanningInput.parse(raw_planning, self.config)
        self.result = plan_v2(self.config, self.planning, top_k=10)
        self.assertGreaterEqual(len(self.result["candidates"]), 3)

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
        self.by_device = {
            identity.tensormeld_device_id: identity
            for identity in self.identities
        }

    def calibration(self, candidate, *, source="native-target", prefill_us=1000, decode_us=1000):
        candidate_nodes = set(candidate["compute_nodes"])
        raw = {
            "calibration_schema": "tensormeld/placement-calibration-v1",
            "measurement_source": source,
            "config_sha256": self.config.fingerprint,
            "planning_input_sha256": self.planning.fingerprint,
            "profile": self.config.installation.default_profile,
            "model_manifest_sha256": self.planning.manifest_ref,
            "package_sha256": self.package.fingerprint,
            "candidate_plan_sha256": candidate["plan_sha256"],
            "runtime_identity_sha256": [
                self.by_device[device].identity_sha256
                for device in sorted(self.by_device)
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
            "physical_pool_peak_bytes": [
                {"pool_ref": pool.id, "peak_bytes": 100}
                for pool in self.config.pools
                if pool.node in candidate_nodes
            ],
            "qualified": False,
            "executable": False,
        }
        return parse_placement_calibration(
            raw,
            config=self.config,
            planning=self.planning,
            candidate=candidate,
            package=self.package,
            runtime_identities=self.identities,
        )

    def test_native_calibration_can_recommend_non_synthetic_best_without_mutation(self):
        synthetic = self.result["best"]
        target = next(
            candidate
            for candidate in self.result["candidates"]
            if candidate["plan_sha256"] != synthetic["plan_sha256"]
        )
        slow_synthetic = self.calibration(
            synthetic,
            source="native-target",
            prefill_us=5000,
            decode_us=5000,
        )
        fast_target = self.calibration(
            target,
            source="native-target",
            prefill_us=100,
            decode_us=100,
        )
        original = deepcopy(self.result)

        preferred = apply_measured_preference(
            self.result,
            [slow_synthetic, fast_target],
            require_native=True,
        )

        self.assertEqual(
            preferred["synthetic_best"]["plan_sha256"],
            synthetic["plan_sha256"],
        )
        self.assertEqual(
            preferred["measured_best"]["plan_sha256"],
            target["plan_sha256"],
        )
        self.assertEqual(
            preferred["recommended"]["plan_sha256"],
            target["plan_sha256"],
        )
        self.assertEqual(
            preferred["preference"]["recommendation_source"],
            "native-calibration",
        )
        self.assertFalse(preferred["preference"]["qualified"])
        self.assertFalse(preferred["preference"]["executable"])
        self.assertEqual(self.result, original)
        self.assertEqual(preferred["candidates"], original["candidates"])

    def test_fixture_calibration_does_not_override_synthetic_best_in_native_mode(self):
        synthetic = self.result["best"]
        target = next(
            candidate
            for candidate in self.result["candidates"]
            if candidate["plan_sha256"] != synthetic["plan_sha256"]
        )
        fixture_fast = self.calibration(
            target,
            source="fixture",
            prefill_us=1,
            decode_us=1,
        )
        preferred = apply_measured_preference(
            self.result,
            [fixture_fast],
            require_native=True,
        )
        self.assertIsNone(preferred["measured_best"])
        self.assertEqual(
            preferred["recommended"]["plan_sha256"],
            synthetic["plan_sha256"],
        )
        self.assertEqual(
            preferred["preference"]["recommendation_source"],
            "synthetic",
        )

    def test_fixture_mode_can_exercise_measured_reordering_without_native_claim(self):
        target = self.result["candidates"][-1]
        fixture_fast = self.calibration(
            target,
            source="fixture",
            prefill_us=1,
            decode_us=1,
        )
        preferred = apply_measured_preference(
            self.result,
            [fixture_fast],
            require_native=False,
        )
        self.assertEqual(
            preferred["recommended"]["plan_sha256"],
            target["plan_sha256"],
        )
        self.assertEqual(
            preferred["preference"]["recommendation_source"],
            "calibration",
        )
        self.assertFalse(preferred["preference"]["qualified"])

    def test_calibration_for_unretained_candidate_is_rejected(self):
        candidate = deepcopy(self.result["candidates"][0])
        candidate["plan_sha256"] = "f" * 64
        calibration = replace(
            self.calibration(self.result["candidates"][0]),
            candidate_plan_sha256="f" * 64,
        )
        rec = calibration.as_record()
        rec.pop("fingerprint")
        calibration = replace(
            calibration,
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
            apply_measured_preference(
                self.result,
                [calibration],
                require_native=True,
            )

    def test_mismatched_planning_identity_is_rejected(self):
        calibration = replace(
            self.calibration(self.result["candidates"][0]),
            planning_input_sha256="0" * 64,
        )
        rec = calibration.as_record()
        rec.pop("fingerprint")
        calibration = replace(
            calibration,
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
            apply_measured_preference(
                self.result,
                [calibration],
                require_native=True,
            )

    def test_tampered_planner_best_is_rejected(self):
        bad = deepcopy(self.result)
        bad["best"] = deepcopy(bad["best"])
        bad["best"]["estimated_decode_us"] += 1
        with self.assertRaises(ValidationError):
            apply_measured_preference(
                bad,
                [],
                require_native=True,
            )


if __name__ == "__main__":
    unittest.main()
