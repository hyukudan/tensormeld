from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tensormeld.directional_path_evidence import (
    directional_path_evidence_summary,
    load_directional_path_evidence,
    lookup_transfer_upper_bound,
    parse_directional_path_evidence,
)
from tensormeld.runtime_identity import RuntimeIdentity
from tensormeld.schema import ValidationError
from test_tensor_movability import setup_env


def fixture():
    config, model, index, adapter, package, identities = setup_env()
    by_device = {i.tensormeld_device_id: i for i in identities}
    devices = [d.id for d in config.devices]
    return config, tuple(identities), by_device, devices


def raw_evidence(config, identities, devices):
    by_device = {i.tensormeld_device_id: i for i in identities}
    endpoint_ids = tuple(by_device[d].identity_sha256 for d in sorted(devices))
    a, b = devices[:2]
    return {
        "path_evidence_schema": "tensormeld/directional-path-evidence-v1",
        "provenance": "fixture",
        "config_sha256": config.fingerprint,
        "runtime_identity_sha256": list(endpoint_ids),
        "paths": [
            {
                "id": "a-to-b-primary",
                "source_device": a,
                "target_device": b,
                "transport": "fixture-private",
                "physical_group": "wire-0",
                "buckets": [
                    {"max_payload_bytes": 64, "upper_bound_us": 100},
                    {"max_payload_bytes": 1024, "upper_bound_us": 200},
                ],
            },
            {
                "id": "b-to-a-primary",
                "source_device": b,
                "target_device": a,
                "transport": "fixture-private",
                "physical_group": "wire-0",
                "buckets": [
                    {"max_payload_bytes": 64, "upper_bound_us": 120},
                    {"max_payload_bytes": 1024, "upper_bound_us": 240},
                ],
            },
        ],
        "qualified": False,
        "executable": False,
    }


class DirectionalPathEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config, self.identities, self.by_device, self.devices = fixture()

    def parse(self, raw=None, identities=None):
        return parse_directional_path_evidence(
            raw or raw_evidence(self.config, self.identities, self.devices),
            config=self.config,
            runtime_identities=identities or self.identities,
        )

    def test_directional_buckets_are_accepted(self):
        evidence = self.parse()
        self.assertEqual(len(evidence.paths), 2)
        summary = directional_path_evidence_summary(evidence)
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])

    def test_lookup_uses_first_covering_upper_bound_bucket(self):
        evidence = self.parse()
        a, b = self.devices[:2]
        small = lookup_transfer_upper_bound(
            evidence, source_device=a, target_device=b, payload_bytes=64
        )
        medium = lookup_transfer_upper_bound(
            evidence, source_device=a, target_device=b, payload_bytes=65
        )
        self.assertEqual(small["upper_bound_us"], 100)
        self.assertEqual(medium["upper_bound_us"], 200)

    def test_no_extrapolation_beyond_largest_bucket(self):
        evidence = self.parse()
        a, b = self.devices[:2]
        self.assertIsNone(
            lookup_transfer_upper_bound(
                evidence, source_device=a, target_device=b, payload_bytes=1025
            )
        )

    def test_direction_is_not_assumed_symmetric(self):
        evidence = self.parse()
        a, b = self.devices[:2]
        forward = lookup_transfer_upper_bound(
            evidence, source_device=a, target_device=b, payload_bytes=64
        )
        reverse = lookup_transfer_upper_bound(
            evidence, source_device=b, target_device=a, payload_bytes=64
        )
        self.assertEqual(forward["upper_bound_us"], 100)
        self.assertEqual(reverse["upper_bound_us"], 120)

    def test_parallel_paths_are_alternatives_not_summed(self):
        raw = raw_evidence(self.config, self.identities, self.devices)
        a, b = self.devices[:2]
        raw["paths"].append({
            "id": "a-to-b-secondary",
            "source_device": a,
            "target_device": b,
            "transport": "fixture-private",
            "physical_group": "wire-1",
            "buckets": [
                {"max_payload_bytes": 64, "upper_bound_us": 80},
                {"max_payload_bytes": 1024, "upper_bound_us": 180},
            ],
        })
        evidence = self.parse(raw)
        chosen = lookup_transfer_upper_bound(
            evidence, source_device=a, target_device=b, payload_bytes=64
        )
        self.assertEqual(chosen["path_id"], "a-to-b-secondary")
        self.assertEqual(chosen["upper_bound_us"], 80)

    def test_same_device_and_zero_payload_need_no_path(self):
        evidence = self.parse()
        a, b = self.devices[:2]
        local = lookup_transfer_upper_bound(
            evidence, source_device=a, target_device=a, payload_bytes=999
        )
        zero = lookup_transfer_upper_bound(
            evidence, source_device=a, target_device=b, payload_bytes=0
        )
        self.assertEqual(local["upper_bound_us"], 0)
        self.assertEqual(zero["upper_bound_us"], 0)

    def test_buckets_must_be_strict_and_nondecreasing(self):
        raw = raw_evidence(self.config, self.identities, self.devices)
        raw["paths"][0]["buckets"][1]["max_payload_bytes"] = 64
        with self.assertRaises(ValidationError):
            self.parse(raw)

        raw = raw_evidence(self.config, self.identities, self.devices)
        raw["paths"][0]["buckets"][1]["upper_bound_us"] = 99
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_unknown_or_same_endpoint_is_rejected(self):
        raw = raw_evidence(self.config, self.identities, self.devices)
        raw["paths"][0]["target_device"] = "ghost"
        with self.assertRaises(ValidationError):
            self.parse(raw)
        raw = raw_evidence(self.config, self.identities, self.devices)
        raw["paths"][0]["target_device"] = raw["paths"][0]["source_device"]
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_runtime_identity_change_invalidates_evidence(self):
        raw = raw_evidence(self.config, self.identities, self.devices)
        changed = list(self.identities)
        rec = changed[0].as_record()
        rec.pop("identity_sha256")
        rec["topology_sha256"] = "9" * 64
        changed[0] = RuntimeIdentity.parse(rec)
        with self.assertRaises(ValidationError):
            self.parse(raw, tuple(changed))

    def test_runtime_identity_list_must_match_exact_endpoint_set_and_order(self):
        raw = raw_evidence(self.config, self.identities, self.devices)
        raw["runtime_identity_sha256"].reverse()
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_profile_never_self_promotes(self):
        for field in ("qualified", "executable"):
            raw = raw_evidence(self.config, self.identities, self.devices)
            raw[field] = True
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "paths.json"
            path.write_text(
                '{"path_evidence_schema":"a","path_evidence_schema":"b"}',
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                load_directional_path_evidence(
                    path,
                    config=self.config,
                    runtime_identities=self.identities,
                )

    def test_lookup_rejects_invalid_payload(self):
        evidence = self.parse()
        a, b = self.devices[:2]
        for payload in (-1, 1.5, True):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                lookup_transfer_upper_bound(
                    evidence,
                    source_device=a,
                    target_device=b,
                    payload_bytes=payload,
                )


if __name__ == "__main__":
    unittest.main()
