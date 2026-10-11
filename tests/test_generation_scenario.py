from __future__ import annotations

import copy
from dataclasses import replace
import unittest

from tensormeld.generation_scenario import evaluate_generation_scenario
from tensormeld.schema import ValidationError
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.legal_unit_costs import parse_legal_unit_costs
from test_generation_phase_evidence import raw_phase, raw_phase_v2
from test_legal_unit_performance import setup_fixture
from test_legal_model_units import env
from test_runtime_model_manifest import manifest as runtime_raw


class GenerationScenarioTests(unittest.TestCase):
    def setUp(self):
        (self.config, self.identities, self.movability, self.legal, self.runtime,
         self.raw_costs, self.costs, self.paths, self.path_raw) = setup_fixture()
        _, model, _, adapter, _, _, _ = env()
        manifest = runtime_raw(self.config, model, adapter)
        manifest["worker_artifact_sha256"] = self.identities[0].worker_artifact_sha256
        self.runtime = parse_runtime_model_manifest(manifest, config=self.config,
                                                    model=model, adapter=adapter)
        self.raw_costs["runtime_manifest_sha256"] = self.runtime.fingerprint
        self.costs = parse_legal_unit_costs(self.raw_costs, config=self.config,
                                          legal_units=self.legal, runtime_manifest=self.runtime,
                                          movability=self.movability)
        self.phase_raw = raw_phase_v2(self.config, self.legal, self.runtime, self.costs)
        for path in self.path_raw["paths"]:
            path["buckets"].append({"max_payload_bytes": 4096, "upper_bound_us": 400})

    def evaluate(self, **changes):
        values = dict(config=self.config, legal_units=self.legal, costs=self.costs,
                      runtime_manifest=self.runtime, movability=self.movability,
                      runtime_identities=self.identities, phase_data=self.phase_raw,
                      path_data=self.path_raw, owners=["pc-gpu", "helper-a-igpu"],
                      sampler_device="helper-a-igpu", prefill_control_us=20,
                      decode_control_us=7)
        values.update(changes)
        return evaluate_generation_scenario(**values)

    def test_prefill_decode_use_distinct_payloads_and_close_feedback_cycle(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "SCENARIO_COSTED")
        self.assertEqual(result["phases"]["prefill"]["scenario_upper_bound_us"], 2630)
        self.assertEqual(result["phases"]["decode"]["scenario_upper_bound_us"], 377)
        pre = result["phases"]["prefill"]["transfers"]
        dec = result["phases"]["decode"]["transfers"]
        self.assertEqual(pre[0]["payload_bytes"], 4096)
        self.assertEqual(dec[0]["payload_bytes"], 64)
        self.assertEqual(dec[-1]["role"], "token_feedback")
        self.assertEqual(dec[-1]["path_id"], "b-a")
        self.assertFalse(result["memory_admission_performed"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["executable"])

    def test_sampler_location_counts_logits_instead_of_feedback_transfer(self):
        result = self.evaluate(sampler_device="pc-gpu")
        self.assertEqual(result["phases"]["decode"]["scenario_upper_bound_us"], 437)
        transfers = result["phases"]["decode"]["transfers"]
        self.assertEqual(transfers[-2]["role"], "logits")
        self.assertEqual(transfers[-2]["payload_bytes"], 256)
        self.assertEqual(transfers[-2]["upper_bound_us"], 120)
        self.assertEqual(transfers[-1]["upper_bound_us"], 0)

    def test_local_owner_has_no_invented_transport_cost(self):
        result = self.evaluate(owners=["pc-gpu", "pc-gpu"], sampler_device="pc-gpu")
        self.assertEqual(result["phases"]["decode"]["scenario_upper_bound_us"], 267)
        self.assertEqual(result["phases"]["decode"]["transfer_us"], 0)

    def test_missing_reverse_feedback_is_unknown_not_free_or_infeasible(self):
        raw = copy.deepcopy(self.path_raw)
        raw["paths"] = raw["paths"][:1]
        result = self.evaluate(path_data=raw)
        self.assertEqual(result["status"], "EVIDENCE_INCOMPLETE")
        self.assertIsNone(result["phases"]["decode"]["scenario_upper_bound_us"])
        self.assertIsNone(result["phases"]["decode"]["transfer_us"])
        self.assertEqual({t["role"] for t in result["missing_transfers"]}, {"token_feedback"})

    def test_missing_prefill_bucket_does_not_extrapolate_or_reuse_decode_bytes(self):
        raw = copy.deepcopy(self.path_raw)
        for path in raw["paths"]:
            path["buckets"].pop()
        result = self.evaluate(path_data=raw)
        self.assertEqual(result["status"], "EVIDENCE_INCOMPLETE")
        self.assertIsNone(result["phases"]["prefill"]["scenario_upper_bound_us"])
        self.assertEqual(result["phases"]["decode"]["scenario_upper_bound_us"], 377)

    def test_v1_phase_cannot_supply_transfer_composition(self):
        with self.assertRaisesRegex(ValidationError, "requires v2"):
            self.evaluate(phase_data=raw_phase(self.config, self.legal, self.runtime, self.costs))

    def test_undeclared_cut_is_rejected(self):
        legal = replace(self.legal, units=(replace(self.legal.units[0], cut_after=False), self.legal.units[1]))
        with self.assertRaisesRegex(ValidationError, "undeclared legal cut"):
            self.evaluate(legal_units=legal)

    def test_sampler_cannot_add_unaccounted_device(self):
        with self.assertRaisesRegex(ValidationError, "sampler on a compute owner"):
            self.evaluate(sampler_device="not-a-device")

    def test_identity_mismatch_fails_before_costing(self):
        raw = copy.deepcopy(self.phase_raw)
        raw["runtime_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValidationError, "runtime-manifest identity"):
            self.evaluate(phase_data=raw)

    def test_current_worker_must_match_model_measurement_artifact(self):
        runtime = replace(self.runtime, worker_artifact_sha256="f" * 64)
        with self.assertRaisesRegex(ValidationError, "runtime identity/worker"):
            self.evaluate(runtime_manifest=runtime)

    def test_owner_sequence_is_complete_and_not_a_string(self):
        for owners in (["pc-gpu"], "pc-gpu", ["pc-gpu", "missing-device"]):
            with self.subTest(owners=owners), self.assertRaises(ValidationError):
                self.evaluate(owners=owners, sampler_device="pc-gpu")

    def test_no_implicit_control_zero_negative_or_boolean(self):
        for value in (None, True, -1, 1.5, 1 << 63):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.evaluate(decode_control_us=value)

    def test_composition_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "decode.total_us"):
            self.evaluate(decode_control_us=(1 << 63) - 1)

    def test_incomplete_transfer_does_not_hide_compute_overflow(self):
        raw = copy.deepcopy(self.phase_raw)
        for unit in raw["units"]:
            for profile in unit["device_profiles"].values():
                profile["decode_step_us"] = (1 << 63) - 1
        paths = copy.deepcopy(self.path_raw)
        paths["paths"] = paths["paths"][:1]
        with self.assertRaises(ValidationError):
            self.evaluate(phase_data=raw, path_data=paths)

    def test_fixture_and_native_path_scopes_cannot_be_mixed(self):
        paths = copy.deepcopy(self.path_raw)
        paths["provenance"] = "native-target"
        with self.assertRaises(ValidationError):
            self.evaluate(path_data=paths)

    def test_deterministic_result_and_changed_control_identity(self):
        first = self.evaluate()
        self.assertEqual(first, self.evaluate())
        other = self.evaluate(decode_control_us=8)
        self.assertNotEqual(first["result_sha256"], other["result_sha256"])
        self.assertEqual(other["phases"]["decode"]["scenario_upper_bound_us"], 378)


if __name__ == "__main__":
    unittest.main()
