from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from tensormeld.engine_describe import device_supports, file_sha256, parse_engine_describe, run_describe
from tensormeld.hetero_hardware import parse_hardware
from tensormeld.hetero_planner import Request, plan_hetero
from tensormeld.schema import ValidationError

FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_describe_engine.py"


def describe(role="amd"):
    out = subprocess.run([sys.executable, str(FAKE), "--describe-json", role], capture_output=True, check=True)
    return json.loads(out.stdout)


class EngineDescribeTests(unittest.TestCase):
    def test_parse(self):
        d = parse_engine_describe(describe())
        self.assertEqual(d.backend, "hip")
        self.assertEqual(d.capabilities, ("toy.attn", "toy.lookup", "toy.moe"))
        self.assertEqual(d.group_modes[0].world, 2)
        self.assertTrue(d.state_export and d.state_import)
        self.assertEqual(d.model("toy").layers, 2)
        self.assertEqual(d.can_run(d.model("toy")), [])
        cuda = parse_engine_describe(describe("cuda"))
        self.assertEqual(cuda.can_run(cuda.model("toy")), ["toy.moe"])

    def test_contract_rules(self):
        raw = describe()
        raw["roles"] = ["whole", "teleport"]
        with self.assertRaisesRegex(ValidationError, "roles"):
            parse_engine_describe(raw)
        raw = describe()
        del raw["parts_rpc"]
        with self.assertRaisesRegex(ValidationError, "parts-worker"):
            parse_engine_describe(raw)
        raw = describe()
        raw["state"] = {}
        with self.assertRaisesRegex(ValidationError, "state.export: required for the prefill role"):
            parse_engine_describe(raw)
        raw = describe()
        raw["state"]["import"] = False
        with self.assertRaisesRegex(ValidationError, "state.import: required for the decode role"):
            parse_engine_describe(raw)
        raw = describe()
        raw["roles"] = ["whole", "prefill"]
        raw["state"]["import"] = False
        self.assertTrue(parse_engine_describe(raw).state_exact)
        raw = describe()
        with self.assertRaisesRegex(ValidationError, "approved artifact"):
            parse_engine_describe(raw, expected_executable_sha256="1" * 64)

    def test_run_describe_checks_the_artifact_hash(self):
        good = file_sha256(FAKE)
        d = run_describe(FAKE, approved_sha256=good, prefix=(sys.executable,))
        self.assertEqual(d.name, "fake")
        with self.assertRaisesRegex(ValidationError, "not the approved artifact"):
            run_describe(FAKE, approved_sha256="0" * 64, prefix=(sys.executable,))

    def test_describes_drive_the_planner(self):
        """Two engines (full kernels on the unified-memory nodes, attention-only on the GPU) -> the
        planner may only put attention on the GPU; experts stay on the nodes."""
        amd = parse_engine_describe(describe("amd"))
        cuda = parse_engine_describe(describe("cuda"))
        model = amd.model("toy")
        hw = {
            "schema": "tensormeld/hetero-hardware-v1",
            "nodes": [{"id": "pc"}, {"id": "n1"}],
            "devices": [
                {"id": "gpu", "node": "pc", "kind": "discrete_gpu", "backend": "cuda", "memory_bytes": 10**9,
                 "mem_bytes_per_s": 10**12, "prefill_ops_per_s": 10**14, "availability": "available",
                 "supports": list(device_supports([amd, cuda], "cuda")), "provenance": "declared"},
                {"id": "apu", "node": "n1", "kind": "integrated_gpu", "backend": "hip", "memory_bytes": 10**10,
                 "mem_bytes_per_s": 10**11, "prefill_ops_per_s": 10**13, "availability": "available",
                 "supports": list(device_supports([amd, cuda], "hip")), "provenance": "declared"},
            ],
            "links": [{"id": "eth", "a": "pc", "b": "n1", "bytes_per_s": 10**8, "latency_us": 40,
                       "provenance": "measured"}],
        }
        r = plan_hetero(model, parse_hardware(hw), Request(prompt_tokens=128, objective="decode"), top_k=10)
        for c in r["candidates"]:
            self.assertNotIn("routed-experts", c["placement"].get("gpu", {"kinds": {}})["kinds"])
        self.assertEqual(r["status"], "OK")
        self.assertIn("whole:gpu", {c["name"] for c in r["infeasible"]})


if __name__ == "__main__":
    unittest.main()
