from __future__ import annotations

import copy
import unittest

from tensormeld.legal_model_units import (
    legal_model_units_summary,
    parse_legal_model_units,
)
from tensormeld.schema import ValidationError
from tensormeld.tensor_movability import parse_tensor_movability_profile
from test_tensor_movability import raw_profile, setup_env


def env():
    config, model, index, adapter, package, identities = setup_env()
    raw = raw_profile(config, model, index, adapter, package, identities)
    movability = parse_tensor_movability_profile(
        raw,
        config=config,
        model=model,
        tensor_index=index,
        adapter=adapter,
        package=package,
        runtime_identities=identities,
    )
    return config, model, index, adapter, package, identities, movability


def legal_raw(movability):
    by_name = {t.name: t for t in movability.tensors}
    return {
        "legal_units_schema": "tensormeld/legal-model-units-v1",
        "provenance": movability.provenance,
        "tensor_movability_sha256": movability.fingerprint,
        "units": [
            {
                "id": "unit.0",
                "sequence": 0,
                "tensors": ["tensor.a"],
                "allowed_devices": list(by_name["tensor.a"].allowed_devices),
                "cut_after": True,
            },
            {
                "id": "unit.1",
                "sequence": 1,
                "tensors": ["tensor.b"],
                "allowed_devices": list(by_name["tensor.b"].allowed_devices),
                "cut_after": False,
            },
        ],
        "qualified": False,
        "executable": False,
    }


class LegalModelUnitsTests(unittest.TestCase):
    def setUp(self):
        *_, self.movability = env()

    def parse(self, raw=None, *, movability=None):
        return parse_legal_model_units(
            raw or legal_raw(movability or self.movability),
            movability=movability or self.movability,
        )

    def test_complete_ordered_units_and_cut_are_accepted(self):
        profile = self.parse()
        self.assertEqual([u.sequence for u in profile.units], [0, 1])
        self.assertEqual(profile.units[0].hard_resident_bytes, 100)
        self.assertEqual(profile.units[1].reclaimable_file_backed_bytes, 150)
        summary = legal_model_units_summary(profile)
        self.assertEqual(summary["cut_after_sequences"], [0])
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])

    def test_missing_or_duplicate_tensor_is_rejected(self):
        raw = legal_raw(self.movability)
        raw["units"].pop()
        with self.assertRaises(ValidationError):
            self.parse(raw)

        raw = legal_raw(self.movability)
        raw["units"][1]["tensors"] = ["tensor.a", "tensor.b"]
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_allowed_devices_cannot_exceed_member_intersection(self):
        raw = legal_raw(self.movability)
        second_device = next(
            d for d in raw["units"][1]["allowed_devices"]
            if d not in raw["units"][0]["allowed_devices"]
        )
        raw["units"][0]["allowed_devices"].append(second_device)
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_unit_can_conservatively_narrow_movable_device_set(self):
        raw = legal_raw(self.movability)
        raw["units"][1]["allowed_devices"] = [
            raw["units"][1]["allowed_devices"][0]
        ]
        profile = self.parse(raw)
        self.assertEqual(len(profile.units[1].allowed_devices), 1)

    def test_sequence_must_be_contiguous_and_unique(self):
        raw = legal_raw(self.movability)
        raw["units"][1]["sequence"] = 2
        with self.assertRaises(ValidationError):
            self.parse(raw)
        raw = legal_raw(self.movability)
        raw["units"][1]["sequence"] = 0
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_last_unit_cannot_cut_after(self):
        raw = legal_raw(self.movability)
        raw["units"][-1]["cut_after"] = True
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_input_order_does_not_change_explicit_sequence_identity(self):
        a = legal_raw(self.movability)
        b = copy.deepcopy(a)
        b["units"].reverse()
        self.assertEqual(self.parse(a).fingerprint, self.parse(b).fingerprint)

    def test_alias_group_cannot_be_split_across_units(self):
        config, model, index, adapter, package, identities, _ = env()
        raw_move = raw_profile(config, model, index, adapter, package, identities)
        for item in raw_move["tensors"]:
            item["storage_class"] = "hard_resident"
            item["movement_class"] = "pinned"
            item["allowed_devices"] = [config.devices[0].id]
            item["alias_group"] = "tied"
        movability = parse_tensor_movability_profile(
            raw_move,
            config=config,
            model=model,
            tensor_index=index,
            adapter=adapter,
            package=package,
            runtime_identities=identities,
        )
        raw = legal_raw(movability)
        with self.assertRaises(ValidationError):
            self.parse(raw, movability=movability)

        raw["units"] = [{
            "id": "tied-unit",
            "sequence": 0,
            "tensors": ["tensor.a", "tensor.b"],
            "allowed_devices": [config.devices[0].id],
            "cut_after": False,
        }]
        profile = self.parse(raw, movability=movability)
        self.assertEqual(profile.units[0].tensors, ("tensor.a", "tensor.b"))

    def test_provenance_and_movability_identity_are_exact(self):
        raw = legal_raw(self.movability)
        raw["provenance"] = "native-adapter"
        with self.assertRaises(ValidationError):
            self.parse(raw)
        raw = legal_raw(self.movability)
        raw["tensor_movability_sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_profile_never_self_promotes(self):
        for field in ("qualified", "executable"):
            raw = legal_raw(self.movability)
            raw[field] = True
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)


if __name__ == "__main__":
    unittest.main()
