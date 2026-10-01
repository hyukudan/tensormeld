from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.cli import main
from tensormeld.config_v2 import Config
from tensormeld.runtime_observation import load_runtime_observation
from tensormeld.schema import ValidationError
from tensormeld.selection import resolve_runtime_candidates
from test_config_v2 import data


def observation(cfg: Config):
    return {
        "runtime_observation_schema": "tensormeld/runtime-observation-v1",
        "config_sha256": cfg.fingerprint,
        "devices": {
            d.id: {"backend": d.backend, "state": "ready"}
            for d in cfg.devices
        },
        "pools": {
            p.id: {"available_bytes": p.reported_capacity_bytes or 0}
            for p in cfg.pools
        },
        "qualified": False,
        "executable": False,
    }


class RuntimeObservationTests(unittest.TestCase):
    def test_runtime_budget_intersects_free_memory_and_static_policy(self):
        cfg = Config.parse(data())
        obs = observation(cfg)
        obs["pools"]["pc-vram"]["available_bytes"] = 6 * 1024**3
        r = resolve_runtime_candidates(cfg, obs)
        self.assertEqual(r["runtime_pool_budgets"]["pc-vram"], 4 * 1024**3)
        self.assertFalse(r["reservation_created"])
        self.assertFalse(r["qualified"])
        self.assertFalse(r["executable"])

    def test_static_cap_remains_upper_bound(self):
        raw = data()
        raw["resource_policies"][0]["allocation_cap_bytes"] = 4 * 1024**3
        cfg = Config.parse(raw)
        r = resolve_runtime_candidates(cfg, observation(cfg))
        self.assertEqual(r["runtime_pool_budgets"]["pc-vram"], 4 * 1024**3)

    def test_offline_required_device_is_structured_failure(self):
        raw = data()
        raw["selection"]["required_devices"] = ["pc-gpu"]
        cfg = Config.parse(raw)
        obs = observation(cfg)
        obs["devices"]["pc-gpu"]["state"] = "offline"
        r = resolve_runtime_candidates(cfg, obs)
        self.assertEqual(r["status"], "RUNTIME_REQUIREMENTS_UNMET")
        self.assertIn(
            "REQUIRED_DEVICE_NOT_RUNTIME_READY",
            {x["code"] for x in r["reasons"]},
        )

    def test_missing_pool_observation_removes_device_from_runtime_candidates(self):
        cfg = Config.parse(data())
        obs = observation(cfg)
        obs["pools"].pop("pc-vram")
        r = resolve_runtime_candidates(cfg, obs)
        self.assertNotIn(
            "pc-gpu", [x["id"] for x in r["runtime_eligible_compute_devices"]]
        )

    def test_observation_cannot_self_promote(self):
        cfg = Config.parse(data())
        obs = observation(cfg)
        obs["qualified"] = True
        with self.assertRaises(ValidationError):
            resolve_runtime_candidates(cfg, obs)

    def test_config_identity_must_match(self):
        cfg = Config.parse(data())
        obs = observation(cfg)
        obs["config_sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            resolve_runtime_candidates(cfg, obs)

    def test_unknown_device_and_backend_mismatch_are_rejected(self):
        cfg = Config.parse(data())
        obs = observation(cfg)
        obs["devices"]["ghost"] = {"backend": "cuda", "state": "ready"}
        with self.assertRaises(ValidationError):
            resolve_runtime_candidates(cfg, obs)
        obs = observation(cfg)
        obs["devices"]["pc-gpu"]["backend"] = "hip"
        with self.assertRaises(ValidationError):
            resolve_runtime_candidates(cfg, obs)

    def test_available_bytes_cannot_exceed_reported_capacity(self):
        cfg = Config.parse(data())
        obs = observation(cfg)
        obs["pools"]["pc-vram"]["available_bytes"] += 1
        with self.assertRaises(ValidationError):
            resolve_runtime_candidates(cfg, obs)

    def test_loader_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "observation.json"
            p.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_runtime_observation(p)

    def test_cli_runtime_select_and_output_safety(self):
        raw = data()
        cfg = Config.parse(raw)
        obs = observation(cfg)
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / "config.json"
            o = Path(d) / "observation.json"
            out = Path(d) / "result.json"
            c.write_text(json.dumps(raw), encoding="utf-8")
            o.write_text(json.dumps(obs), encoding="utf-8")
            code = main(["runtime-select", str(c), str(o), "--out", str(out)])
            self.assertEqual(code, 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "RUNTIME_CANDIDATES_READY")
            before = o.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["runtime-select", str(c), str(o), "--out", str(o)])
            self.assertEqual(code, 1)
            self.assertEqual(o.read_bytes(), before)
