from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from tensormeld.generation_phase_evidence import (
    generation_phase_summary,
    load_generation_phase_evidence,
    parse_generation_phase_evidence,
)
from tensormeld.legal_model_units import parse_legal_model_units
from tensormeld.legal_unit_costs import parse_legal_unit_costs
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from test_legal_model_units import env, legal_raw
from test_legal_unit_costs import raw_costs
from test_runtime_model_manifest import manifest as runtime_manifest_raw


def fixture():
    config, model, index, adapter, package, identities, movability = env()
    legal = parse_legal_model_units(legal_raw(movability), movability=movability)
    runtime = parse_runtime_model_manifest(
        runtime_manifest_raw(config, model, adapter),
        config=config,
        model=model,
        adapter=adapter,
    )
    costs = parse_legal_unit_costs(
        raw_costs(config, legal, runtime),
        config=config,
        legal_units=legal,
        runtime_manifest=runtime,
        movability=movability,
    )
    return config, movability, legal, runtime, costs


def raw_phase(config, legal, runtime, costs):
    units = []
    for unit in legal.units:
        units.append({
            "id": unit.id,
            "sequence": unit.sequence,
            "device_profiles": {
                device: {
                    "prefill_us": 1000 + unit.sequence * 100,
                    "decode_step_us": 100 + unit.sequence * 10,
                }
                for device in unit.allowed_devices
            },
        })
    sampling = {
        device.id: {
            "sampling_us": 50,
            "logits_payload_bytes": 256,
            "feedback_payload_bytes": 8,
        }
        for device in runtime.devices
    }
    return {
        "generation_phase_schema": "tensormeld/generation-phase-evidence-v1",
        "provenance": legal.provenance,
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal.fingerprint,
        "legal_unit_costs_sha256": costs.fingerprint,
        "runtime_manifest_sha256": runtime.fingerprint,
        "phase_workload": {
            "prefill_tokens": 7168,
            "decode_context_tokens": 7168,
            "concurrency": runtime.workload["concurrency"],
        },
        "units": units,
        "sampling_profiles": sampling,
        "qualified": False,
        "executable": False,
    }


class GenerationPhaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        (
            self.config,
            self.movability,
            self.legal,
            self.runtime,
            self.costs,
        ) = fixture()

    def parse(self, raw=None, **kwargs):
        return parse_generation_phase_evidence(
            raw or raw_phase(self.config, self.legal, self.runtime, self.costs),
            config=kwargs.get("config", self.config),
            legal_units=kwargs.get("legal_units", self.legal),
            costs=kwargs.get("costs", self.costs),
            runtime_manifest=kwargs.get("runtime_manifest", self.runtime),
            movability=kwargs.get("movability", self.movability),
        )

    def test_complete_phase_and_sampling_evidence_is_accepted(self):
        evidence = self.parse()
        self.assertEqual(evidence.prefill_tokens, 7168)
        self.assertEqual(evidence.decode_context_tokens, 7168)
        self.assertEqual(len(evidence.units), len(self.legal.units))
        self.assertGreaterEqual(len(evidence.sampling_profiles), 1)
        summary = generation_phase_summary(evidence)
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])

    def test_phase_profiles_cover_every_legal_device_exactly(self):
        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        movable = next(
            unit for unit in raw["units"]
            if len(unit["device_profiles"]) > 1
        )
        movable["device_profiles"].pop(next(iter(movable["device_profiles"])))
        with self.assertRaises(ValidationError):
            self.parse(raw)

        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        first = raw["units"][0]
        ghost = "ghost-device"
        first["device_profiles"][ghost] = {
            "prefill_us": 1,
            "decode_step_us": 1,
        }
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_prefill_and_decode_costs_must_be_positive(self):
        for field in ("prefill_us", "decode_step_us"):
            raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
            profile = next(iter(raw["units"][0]["device_profiles"].values()))
            profile[field] = 0
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_phase_context_is_explicit_and_bounded(self):
        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        raw["phase_workload"]["prefill_tokens"] = 8192
        with self.assertRaises(ValidationError):
            self.parse(raw)

        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        raw["phase_workload"]["decode_context_tokens"] = 8193
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_concurrency_must_match_runtime_manifest(self):
        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        raw["phase_workload"]["concurrency"] += 1
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_sampling_device_must_exist_in_runtime_manifest(self):
        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        raw["sampling_profiles"]["ghost"] = {
            "sampling_us": 1,
            "logits_payload_bytes": 1,
            "feedback_payload_bytes": 1,
        }
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_sampling_cost_must_be_positive_but_payloads_may_be_zero(self):
        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        sampler = next(iter(raw["sampling_profiles"].values()))
        sampler["sampling_us"] = 0
        with self.assertRaises(ValidationError):
            self.parse(raw)

        raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
        sampler = next(iter(raw["sampling_profiles"].values()))
        sampler["logits_payload_bytes"] = 0
        sampler["feedback_payload_bytes"] = 0
        evidence = self.parse(raw)
        self.assertEqual(evidence.sampling_profiles[0].logits_payload_bytes, 0)

    def test_identity_binding_is_exact(self):
        for field in (
            "config_sha256",
            "legal_model_units_sha256",
            "legal_unit_costs_sha256",
            "runtime_manifest_sha256",
        ):
            raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
            raw[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_profile_never_self_promotes(self):
        for field in ("qualified", "executable"):
            raw = raw_phase(self.config, self.legal, self.runtime, self.costs)
            raw[field] = True
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_input_order_does_not_change_fingerprint(self):
        a = raw_phase(self.config, self.legal, self.runtime, self.costs)
        b = copy.deepcopy(a)
        b["units"].reverse()
        for unit in b["units"]:
            unit["device_profiles"] = dict(
                reversed(list(unit["device_profiles"].items()))
            )
        b["sampling_profiles"] = dict(
            reversed(list(b["sampling_profiles"].items()))
        )
        self.assertEqual(self.parse(a).fingerprint, self.parse(b).fingerprint)

    def test_loader_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "phase.json"
            path.write_text(
                '{"generation_phase_schema":"a","generation_phase_schema":"b"}',
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                load_generation_phase_evidence(
                    path,
                    config=self.config,
                    legal_units=self.legal,
                    costs=self.costs,
                    runtime_manifest=self.runtime,
                    movability=self.movability,
                )

    def test_summary_does_not_claim_ttft_or_token_latency(self):
        summary = generation_phase_summary(self.parse())
        warnings = " ".join(summary["warnings"])
        self.assertIn("not a TTFT", warnings)
        self.assertIn("explicit phase workload", warnings)


if __name__ == "__main__":
    unittest.main()
