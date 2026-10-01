from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.cli import main
from tensormeld.config_v2 import Config
from tensormeld.device_binding import LlamaCppBinding, bind_llamacpp_probe
from tensormeld.llamacpp_probe import LLAMACPP_PINNED_COMMIT
from tensormeld.schema import ValidationError
from tensormeld.selection import resolve_runtime_candidates
from test_config_v2 import data

H = "a" * 64


def probe():
    return {
        "probe_schema": "tensormeld/llamacpp-probe-v1",
        "engine": "llama.cpp",
        "pinned_source_revision": LLAMACPP_PINNED_COMMIT,
        "artifact_sha256": H,
        "binary_name": "llama-cli",
        "build": {
            "version": "b1", "build": 1, "commit": "552f18f9",
            "compiler": "cc", "target": "x86_64",
        },
        "devices": [
            {
                "engine_device_name": "CUDA0",
                "description": "GPU",
                "total_bytes": 12 * 1024**3,
                "free_bytes": 8 * 1024**3,
            }
        ],
        "device_identity_mapping": "unresolved",
        "model_loaded": False,
        "listener_started": False,
        "qualified": False,
        "executable": False,
        "warnings": [],
    }


def binding(cfg: Config, *, reporter=True):
    return LlamaCppBinding.parse({
        "binding_schema": "tensormeld/llamacpp-device-binding-v1",
        "config_sha256": cfg.fingerprint,
        "node_id": "pc",
        "artifact_sha256": H,
        "approval": "explicit",
        "mappings": [{
            "engine_device_name": "CUDA0",
            "tensormeld_device_id": "pc-gpu",
            "memory_reporter": reporter,
        }],
    })


class DeviceBindingTests(unittest.TestCase):
    def test_explicit_mapping_produces_observed_not_ready_device(self):
        cfg = Config.parse(data())
        result = bind_llamacpp_probe(cfg, probe(), binding(cfg))
        obs = result["runtime_observation"]
        self.assertEqual(obs["devices"]["pc-gpu"]["state"], "observed")
        self.assertEqual(obs["devices"]["pc-gpu"]["backend"], "cuda")
        self.assertEqual(obs["pools"]["pc-vram"]["available_bytes"], 8 * 1024**3)
        self.assertFalse(result["qualified"])
        self.assertFalse(result["executable"])

    def test_backend_comes_from_config_not_engine_name(self):
        raw = data()
        raw["devices"][0]["backend"] = "custom-backend"
        cfg = Config.parse(raw)
        result = bind_llamacpp_probe(cfg, probe(), binding(cfg))
        self.assertEqual(
            result["resolved_mappings"][0]["backend_from_config"], "custom-backend"
        )

    def test_binding_requires_explicit_approval(self):
        cfg = Config.parse(data())
        raw = {
            "binding_schema": "tensormeld/llamacpp-device-binding-v1",
            "config_sha256": cfg.fingerprint, "node_id": "pc",
            "artifact_sha256": H, "approval": "auto",
            "mappings": [{
                "engine_device_name": "CUDA0",
                "tensormeld_device_id": "pc-gpu",
                "memory_reporter": False,
            }],
        }
        with self.assertRaises(ValidationError):
            LlamaCppBinding.parse(raw)

    def test_artifact_and_config_identity_must_match(self):
        cfg = Config.parse(data())
        b = binding(cfg)
        bad_probe = probe()
        bad_probe["artifact_sha256"] = "b" * 64
        with self.assertRaises(ValidationError):
            bind_llamacpp_probe(cfg, bad_probe, b)
        other_raw = data()
        other_raw["installation"]["id"] = "other"
        with self.assertRaises(ValidationError):
            bind_llamacpp_probe(Config.parse(other_raw), probe(), b)

    def test_unknown_engine_device_rejected(self):
        cfg = Config.parse(data())
        braw = {
            "binding_schema": "tensormeld/llamacpp-device-binding-v1",
            "config_sha256": cfg.fingerprint, "node_id": "pc",
            "artifact_sha256": H, "approval": "explicit",
            "mappings": [{
                "engine_device_name": "CUDA9",
                "tensormeld_device_id": "pc-gpu",
                "memory_reporter": False,
            }],
        }
        with self.assertRaises(ValidationError):
            bind_llamacpp_probe(cfg, probe(), LlamaCppBinding.parse(braw))

    def test_wrong_node_rejected(self):
        cfg = Config.parse(data())
        braw = {
            "binding_schema": "tensormeld/llamacpp-device-binding-v1",
            "config_sha256": cfg.fingerprint, "node_id": "helper-a",
            "artifact_sha256": H, "approval": "explicit",
            "mappings": [{
                "engine_device_name": "CUDA0",
                "tensormeld_device_id": "pc-gpu",
                "memory_reporter": False,
            }],
        }
        with self.assertRaises(ValidationError):
            bind_llamacpp_probe(cfg, probe(), LlamaCppBinding.parse(braw))

    def test_memory_reporter_is_optional(self):
        cfg = Config.parse(data())
        result = bind_llamacpp_probe(cfg, probe(), binding(cfg, reporter=False))
        self.assertEqual(result["runtime_observation"]["pools"], {})

    def test_only_one_reporter_per_physical_pool(self):
        raw = data()
        raw["devices"].append({
            "id": "pc-gpu2", "node": "pc", "kind": "discrete_gpu",
            "backend": "cuda", "pool_ref": "pc-vram", "enabled": True,
        })
        cfg = Config.parse(raw)
        p = probe()
        p["devices"].append({
            "engine_device_name": "CUDA1", "description": "GPU2",
            "total_bytes": 12 * 1024**3, "free_bytes": 7 * 1024**3,
        })
        b = LlamaCppBinding.parse({
            "binding_schema": "tensormeld/llamacpp-device-binding-v1",
            "config_sha256": cfg.fingerprint, "node_id": "pc",
            "artifact_sha256": H, "approval": "explicit",
            "mappings": [
                {"engine_device_name": "CUDA0", "tensormeld_device_id": "pc-gpu", "memory_reporter": True},
                {"engine_device_name": "CUDA1", "tensormeld_device_id": "pc-gpu2", "memory_reporter": True},
            ],
        })
        with self.assertRaises(ValidationError):
            bind_llamacpp_probe(cfg, p, b)

    def test_unmapped_engine_devices_are_reported_not_auto_bound(self):
        cfg = Config.parse(data())
        p = probe()
        p["devices"].append({
            "engine_device_name": "CUDA1", "description": "Other",
            "total_bytes": 4 * 1024**3, "free_bytes": 3 * 1024**3,
        })
        result = bind_llamacpp_probe(cfg, p, binding(cfg))
        self.assertEqual(result["unmapped_engine_devices"], ["CUDA1"])

    def test_observed_device_is_not_runtime_ready(self):
        cfg = Config.parse(data())
        obs = bind_llamacpp_probe(cfg, probe(), binding(cfg))["runtime_observation"]
        result = resolve_runtime_candidates(cfg, obs)
        self.assertNotIn(
            "pc-gpu",
            [d["id"] for d in result["runtime_eligible_compute_devices"]],
        )


class DeviceBindingCLITests(unittest.TestCase):
    def test_cli_binding_roundtrip_and_output_safety(self):
        raw = data()
        cfg = Config.parse(raw)
        p = probe()
        b = {
            "binding_schema": "tensormeld/llamacpp-device-binding-v1",
            "config_sha256": cfg.fingerprint,
            "node_id": "pc",
            "artifact_sha256": H,
            "approval": "explicit",
            "mappings": [{
                "engine_device_name": "CUDA0",
                "tensormeld_device_id": "pc-gpu",
                "memory_reporter": True,
            }],
        }
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = root / "config.json"
            probe_path = root / "probe.json"
            binding_path = root / "binding.json"
            out = root / "bound.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            probe_path.write_text(json.dumps(p), encoding="utf-8")
            binding_path.write_text(json.dumps(b), encoding="utf-8")
            code = main([
                "bind-llamacpp-devices",
                str(config_path), str(probe_path), str(binding_path),
                "--out", str(out),
            ])
            self.assertEqual(code, 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                result["runtime_observation"]["devices"]["pc-gpu"]["state"],
                "observed",
            )
            before = binding_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = main([
                    "bind-llamacpp-devices",
                    str(config_path), str(probe_path), str(binding_path),
                    "--out", str(binding_path),
                ])
            self.assertEqual(code, 1)
            self.assertEqual(binding_path.read_bytes(), before)

    def test_binding_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "binding.json"
            path.write_text('{"binding_schema":"x","binding_schema":"y"}', encoding="utf-8")
            from tensormeld.device_binding import load_llamacpp_binding
            with self.assertRaises(ValidationError):
                load_llamacpp_binding(path)
