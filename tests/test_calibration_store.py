from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.calibration_store import (
    applicable_persisted_calibrations,
    load_calibration_records,
    make_calibration_record,
    persist_calibration,
)
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


class CalibrationStoreTests(unittest.TestCase):
    def setUp(self):
        raw_config, raw_planning = fixture(count=3, units=4)
        self.config = Config.parse(raw_config)
        self.planning = PlanningInput.parse(raw_planning, self.config)
        self.result = plan_v2(self.config, self.planning, top_k=10)
        self.package = build_llamacpp_package_identity(
            cli_probe=probe("1" * 64, "llama-cli"),
            server_probe=probe("2" * 64, "llama-server"),
            backend_libraries=[{"name": "ggml-fixture", "sha256": "3" * 64}],
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
        self.now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)

    def calibration(self, candidate, *, source="native-target", prefill_us=1000, decode_us=1000):
        nodes = set(candidate["compute_nodes"])
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
                if pool.node in nodes
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

    def select(self, records, *, identities=None, package=None, now=None):
        return applicable_persisted_calibrations(
            records,
            now=now or self.now,
            config=self.config,
            planning=self.planning,
            candidates=self.result["candidates"],
            package=package or self.package,
            runtime_identities=identities or self.identities,
        )

    def test_roundtrip_persists_immutable_fingerprint_file(self):
        calibration = self.calibration(self.result["candidates"][0])
        record = make_calibration_record(
            calibration, recorded_at=self.now, max_age_seconds=3600
        )
        with tempfile.TemporaryDirectory() as d:
            path = persist_calibration(d, record)
            self.assertEqual(path.name, calibration.fingerprint + ".json")
            loaded = load_calibration_records(d)
            self.assertEqual(len(loaded), 1)
            selected = self.select(loaded, now=self.now + timedelta(seconds=1))
            self.assertEqual(selected["applicable_count"], 1)
            self.assertEqual(
                selected["applicable"][0].fingerprint,
                calibration.fingerprint,
            )

    def test_identical_write_is_idempotent_conflicting_content_is_rejected(self):
        calibration = self.calibration(self.result["candidates"][0])
        a = make_calibration_record(
            calibration, recorded_at=self.now, max_age_seconds=3600
        )
        b = make_calibration_record(
            calibration,
            recorded_at=self.now + timedelta(seconds=1),
            max_age_seconds=3600,
        )
        with tempfile.TemporaryDirectory() as d:
            first = persist_calibration(d, a)
            second = persist_calibration(d, a)
            self.assertEqual(first, second)
            with self.assertRaises(ValidationError):
                persist_calibration(d, b)

    def test_expired_record_cannot_influence_measured_preference(self):
        synthetic = self.result["best"]
        target = next(
            c for c in self.result["candidates"]
            if c["plan_sha256"] != synthetic["plan_sha256"]
        )
        fast = self.calibration(target, prefill_us=1, decode_us=1)
        record = make_calibration_record(
            fast, recorded_at=self.now, max_age_seconds=60
        )
        selected = self.select(
            [record], now=self.now + timedelta(seconds=61)
        )
        self.assertEqual(selected["applicable"], [])
        self.assertEqual(selected["rejected"][0]["reason"], "EXPIRED")
        preferred = apply_measured_preference(
            self.result, selected["applicable"], require_native=True
        )
        self.assertEqual(
            preferred["recommended"]["plan_sha256"],
            synthetic["plan_sha256"],
        )

    def test_driver_runtime_or_topology_change_invalidates_record(self):
        calibration = self.calibration(self.result["candidates"][0])
        record = make_calibration_record(
            calibration, recorded_at=self.now, max_age_seconds=3600
        )
        for field, value in (
            ("driver_version", "changed-driver"),
            ("runtime_version", "changed-runtime"),
            ("topology_sha256", "9" * 64),
        ):
            changed = list(self.identities)
            original = changed[0]
            raw = original.as_record()
            raw.pop("identity_sha256")
            raw[field] = value
            from tensormeld.runtime_identity import RuntimeIdentity
            changed[0] = RuntimeIdentity.parse(raw)
            with self.subTest(field=field):
                selected = self.select([record], identities=tuple(changed))
                self.assertEqual(selected["applicable_count"], 0)
                self.assertEqual(
                    selected["rejected"][0]["reason"],
                    "IDENTITY_OR_CONTEXT_MISMATCH",
                )

    def test_package_change_invalidates_record(self):
        calibration = self.calibration(self.result["candidates"][0])
        record = make_calibration_record(
            calibration, recorded_at=self.now, max_age_seconds=3600
        )
        changed_package = build_llamacpp_package_identity(
            cli_probe=probe("4" * 64, "llama-cli"),
            server_probe=probe("5" * 64, "llama-server"),
            backend_libraries=[{"name": "ggml-fixture", "sha256": "6" * 64}],
        )
        selected = self.select([record], package=changed_package)
        self.assertEqual(selected["applicable_count"], 0)
        self.assertEqual(
            selected["rejected"][0]["reason"],
            "IDENTITY_OR_CONTEXT_MISMATCH",
        )

    def test_fixture_measurement_is_filtered_in_native_mode(self):
        calibration = self.calibration(
            self.result["candidates"][0], source="fixture"
        )
        record = make_calibration_record(
            calibration, recorded_at=self.now, max_age_seconds=3600
        )
        selected = self.select([record])
        self.assertEqual(selected["applicable_count"], 0)
        self.assertEqual(
            selected["rejected"][0]["reason"], "NON_NATIVE_MEASUREMENT"
        )

    def test_future_record_is_rejected(self):
        calibration = self.calibration(self.result["candidates"][0])
        record = make_calibration_record(
            calibration,
            recorded_at=self.now + timedelta(seconds=1),
            max_age_seconds=3600,
        )
        selected = self.select([record])
        self.assertEqual(
            selected["rejected"][0]["reason"], "RECORDED_IN_FUTURE"
        )

    def test_filename_must_match_fingerprint(self):
        calibration = self.calibration(self.result["candidates"][0])
        record = make_calibration_record(
            calibration, recorded_at=self.now, max_age_seconds=3600
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ("0" * 64 + ".json")
            path.write_text(
                json.dumps(record.as_record()),
                encoding="utf-8",
            )
            with self.assertRaises(ValidationError):
                load_calibration_records(d)

    def test_store_record_never_self_promotes(self):
        calibration = self.calibration(self.result["candidates"][0])
        record = make_calibration_record(
            calibration, recorded_at=self.now, max_age_seconds=3600
        )
        self.assertFalse(record.calibration_record["qualified"])
        self.assertFalse(record.calibration_record["executable"])

    def test_ttl_is_bounded(self):
        calibration = self.calibration(self.result["candidates"][0])
        with self.assertRaises(ValidationError):
            make_calibration_record(
                calibration,
                recorded_at=self.now,
                max_age_seconds=32 * 24 * 60 * 60,
            )


if __name__ == "__main__":
    unittest.main()
