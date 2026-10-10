from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from tensormeld.legal_model_units import parse_legal_model_units
from tensormeld.legal_unit_costs import (
    load_legal_unit_costs,
    legal_unit_costs_summary,
    parse_legal_unit_costs,
)
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from test_legal_model_units import env, legal_raw
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
    return config, model, adapter, movability, legal, runtime


def raw_costs(config, legal, runtime):
    devices = {d.id: d for d in config.devices}
    units = []
    for unit in legal.units:
        profiles = {}
        for device_id in unit.allowed_devices:
            device = devices[device_id]
            profiles[device_id] = {
                "compute_us": 100 + unit.sequence * 10,
                "persistent_state_bytes": {device.pool: 10 + unit.sequence},
                "workspace_peak_bytes": {device.pool: 20 + unit.sequence},
                "staging_peak_bytes": {device.pool: 30 + unit.sequence},
            }
        units.append({
            "id": unit.id,
            "sequence": unit.sequence,
            "boundary_output_bytes": 64 if unit.sequence < len(legal.units) - 1 else 0,
            "device_profiles": profiles,
        })
    return {
        "legal_unit_costs_schema": "tensormeld/legal-unit-costs-v1",
        "provenance": legal.provenance,
        "config_sha256": config.fingerprint,
        "legal_model_units_sha256": legal.fingerprint,
        "runtime_manifest_sha256": runtime.fingerprint,
        "units": units,
        "qualified": False,
        "executable": False,
    }


class LegalUnitCostsTests(unittest.TestCase):
    def setUp(self):
        (
            self.config,
            self.model,
            self.adapter,
            self.movability,
            self.legal,
            self.runtime,
        ) = fixture()

    def parse(self, raw=None, **kwargs):
        return parse_legal_unit_costs(
            raw or raw_costs(self.config, self.legal, self.runtime),
            config=kwargs.get("config", self.config),
            legal_units=kwargs.get("legal_units", self.legal),
            runtime_manifest=kwargs.get("runtime_manifest", self.runtime),
            movability=kwargs.get("movability", self.movability),
        )

    def test_complete_explicit_cost_profile_is_accepted(self):
        profile = self.parse()
        self.assertEqual(len(profile.units), len(self.legal.units))
        first = profile.units[0]
        self.assertEqual(first.boundary_output_bytes, 64)
        self.assertEqual(first.device_profiles[0].compute_us, 100)
        summary = legal_unit_costs_summary(profile)
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])

    def test_every_legal_device_requires_explicit_profile(self):
        raw = raw_costs(self.config, self.legal, self.runtime)
        movable = next(
            item for item in raw["units"] if len(item["device_profiles"]) > 1
        )
        movable["device_profiles"].pop(next(iter(movable["device_profiles"])))
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_nonlegal_device_profile_is_rejected(self):
        raw = raw_costs(self.config, self.legal, self.runtime)
        first = raw["units"][0]
        other = next(d.id for d in self.config.devices if d.id not in first["device_profiles"])
        first["device_profiles"][other] = copy.deepcopy(
            next(iter(first["device_profiles"].values()))
        )
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_memory_demands_must_use_device_local_pools(self):
        raw = raw_costs(self.config, self.legal, self.runtime)
        first = raw["units"][0]
        device_id = next(iter(first["device_profiles"]))
        device = next(d for d in self.config.devices if d.id == device_id)
        foreign_pool = next(p.id for p in self.config.pools if p.node != device.node)
        first["device_profiles"][device_id]["persistent_state_bytes"] = {
            foreign_pool: 1
        }
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_last_unit_boundary_payload_must_be_zero(self):
        raw = raw_costs(self.config, self.legal, self.runtime)
        raw["units"][-1]["boundary_output_bytes"] = 1
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_compute_cost_must_be_positive(self):
        raw = raw_costs(self.config, self.legal, self.runtime)
        profile = next(iter(raw["units"][0]["device_profiles"].values()))
        profile["compute_us"] = 0
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_legal_runtime_and_movability_identities_are_exact(self):
        for field in (
            "config_sha256",
            "legal_model_units_sha256",
            "runtime_manifest_sha256",
        ):
            raw = raw_costs(self.config, self.legal, self.runtime)
            raw[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

        raw = raw_costs(self.config, self.legal, self.runtime)
        from dataclasses import replace
        changed_movability = replace(
            self.movability,
            fingerprint="0" * 64,
        )
        with self.assertRaises(ValidationError):
            self.parse(raw, movability=changed_movability)

    def test_profile_never_self_promotes(self):
        for field in ("qualified", "executable"):
            raw = raw_costs(self.config, self.legal, self.runtime)
            raw[field] = True
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_input_order_does_not_change_fingerprint(self):
        a = raw_costs(self.config, self.legal, self.runtime)
        b = copy.deepcopy(a)
        b["units"].reverse()
        for unit in b["units"]:
            unit["device_profiles"] = dict(reversed(list(unit["device_profiles"].items())))
        self.assertEqual(self.parse(a).fingerprint, self.parse(b).fingerprint)

    def test_loader_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "costs.json"
            path.write_text(
                '{"legal_unit_costs_schema":"a","legal_unit_costs_schema":"b"}',
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                load_legal_unit_costs(
                    path,
                    config=self.config,
                    legal_units=self.legal,
                    runtime_manifest=self.runtime,
                    movability=self.movability,
                )

    def test_boundary_payload_is_size_only_not_transfer_time(self):
        summary = legal_unit_costs_summary(self.parse())
        self.assertIn(
            "transfer time still requires explicit directional path evidence",
            " ".join(summary["warnings"]),
        )


if __name__ == "__main__":
    unittest.main()
