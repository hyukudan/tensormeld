from __future__ import annotations

import copy
import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch

from tensormeld.cli import main
from tensormeld.diagnostics import MAX_FRAME, loopback, receive_frame, send_frame
from tensormeld.planner import NoRoute, plan, transfer
from tensormeld.probe import nvidia_devices, probe
from tensormeld.schema import Scenario, ValidationError, load


def fixture() -> dict:
    return {
        "schema_version": 1, "name": "synthetic-test", "provenance": "synthetic",
        "workload": {"model_revision": "synthetic-v1", "quantization": "synthetic",
                     "context_tokens": 8192, "concurrency": 1, "phase": "decode"},
        "pools": [{"id": "p-a", "node": "a", "budget_bytes": 120},
                  {"id": "p-b", "node": "b", "budget_bytes": 120}],
        "devices": [{"id": n, "node": n, "pool": f"p-{n}", "backend": "test", "runtime_bytes": 10}
                    for n in ("a", "b")],
        "links": [{"id": f"{a}-{b}", "source": a, "target": b,
                   "payload_bytes_per_s": 1000000, "fixed_latency_us": 100,
                   "physical_group": "one-cable"} for a, b in (("a", "b"), ("b", "a"))],
        "route_mode": "via_coordinator", "coordinators": ["a", "b"],
        "stages": [{"id": f"stage-{i}", "weights_bytes": 40, "state_bytes": 5,
                    "workspace_bytes": 20, "output_bytes": 1000,
                    "decode_ms": {"a": 1.0, "b": 2.0}} for i in range(4)],
        "feedback_bytes": 4,
    }


class ValidationTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(len(Scenario.parse(fixture()).stages), 4)

    def test_hash_stable_key_order(self):
        d = fixture()
        self.assertEqual(Scenario.parse(d).fingerprint,
                         Scenario.parse(dict(reversed(list(d.items())))).fingerprint)

    def test_hash_changes_with_workload(self):
        d = fixture()
        first = Scenario.parse(d).fingerprint
        d["workload"]["context_tokens"] = 42
        self.assertNotEqual(first, Scenario.parse(d).fingerprint)

    def test_unknown_root_field(self):
        d = fixture(); d["ignored_magic_setting"] = 1
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_missing_field(self):
        d = fixture(); del d["feedback_bytes"]
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_unknown_nested_field(self):
        d = fixture(); d["devices"][0]["extra"] = True
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_invalid_numbers(self):
        for value in (-1, True, float("nan"), float("inf"), "96 GB", 10**1000):
            with self.subTest(value=str(value)[:20]):
                d = fixture(); d["pools"][0]["budget_bytes"] = value
                with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_fractional_bytes_rejected(self):
        d = fixture(); d["stages"][0]["weights_bytes"] = 1.5
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_zero_bandwidth_rejected(self):
        d = fixture(); d["links"][0]["payload_bytes_per_s"] = 0
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_nan_compute_rejected(self):
        d = fixture(); d["stages"][0]["decode_ms"]["a"] = float("nan")
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_duplicate_device(self):
        d = fixture(); d["devices"][1] = copy.deepcopy(d["devices"][0])
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_unknown_profile_device(self):
        d = fixture(); d["stages"][0]["decode_ms"]["ghost"] = 1
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_unknown_pool(self):
        d = fixture(); d["devices"][0]["pool"] = "ghost"
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_pool_node_mismatch(self):
        d = fixture(); d["devices"][0]["pool"] = "p-b"
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_unsupported_concurrency(self):
        d = fixture(); d["workload"]["concurrency"] = 2
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_unsupported_prefill(self):
        d = fixture(); d["workload"]["phase"] = "prefill"
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_no_stages(self):
        d = fixture(); d["stages"] = []
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_unknown_route(self):
        d = fixture(); d["route_mode"] = "automatic-magic"
        with self.assertRaises(ValidationError): Scenario.parse(d)

    def test_load_duplicate_json_key(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "case.json"; p.write_text('{"schema_version":1,"schema_version":2}')
            with self.assertRaises(ValidationError): load(p)

    def test_load_size_bound(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "case.json"; p.write_bytes(b" " * (2*1024*1024 + 1))
            with self.assertRaises(ValidationError): load(p)


class PlannerTests(unittest.TestCase):
    def test_capacity_split(self):
        result = plan(Scenario.parse(fixture()))
        self.assertEqual(result["status"], "advisory_candidate")
        self.assertFalse(result["qualified"])
        self.assertEqual(len(result["best"]["ranges"]), 2)
        self.assertEqual(result["best"]["pool_usage_bytes"], {"p-a": 120, "p-b": 120})

    def test_feedback_counted(self):
        result = plan(Scenario.parse(fixture()))["best"]
        self.assertGreater(result["feedback"]["cost_ms"], 0)
        self.assertAlmostEqual(result["estimated_communication_ms"], 1.1 + .104)

    def test_single_gpu_not_forced_to_use_helpers(self):
        d = fixture(); d["pools"][0]["budget_bytes"] = 500
        result = plan(Scenario.parse(d))["best"]
        self.assertEqual([r["device"] for r in result["ranges"]], ["a"])
        self.assertEqual(result["estimated_communication_ms"], 0)

    def test_insufficient_capacity_explicit(self):
        d = fixture()
        for p in d["pools"]: p["budget_bytes"] = 100
        result = plan(Scenario.parse(d))
        self.assertEqual(result["status"], "no_feasible_candidate")
        self.assertIsNone(result["best"])

    def test_shared_pool_not_double_counted(self):
        d = fixture()
        d["pools"] = [{"id": "uma", "node": "shared", "budget_bytes": 200}]
        for dev in d["devices"]: dev["pool"] = "uma"; dev["node"] = "shared"
        # Sum of resident + each active device's own workspace/runtime exceeds 200.
        self.assertIsNone(plan(Scenario.parse(d))["best"])

    def test_workspace_peak_not_sum(self):
        d = fixture(); d["pools"][0]["budget_bytes"] = 210
        result = plan(Scenario.parse(d))["best"]
        self.assertEqual(len(result["ranges"]), 1)
        self.assertEqual(result["pool_usage_bytes"]["p-a"], 180 + 20 + 10)

    def test_unknown_device_subset_rejected(self):
        with self.assertRaises(ValidationError): plan(Scenario.parse(fixture()), ["ghost"])

    def test_duplicate_subset_rejected(self):
        with self.assertRaises(ValidationError): plan(Scenario.parse(fixture()), ["a", "a"])

    def test_restrict_subset(self):
        self.assertIsNone(plan(Scenario.parse(fixture()), ["a"])["best"])

    def test_missing_return_route_rejects(self):
        d = fixture(); d["links"] = d["links"][:1]
        self.assertIsNone(plan(Scenario.parse(d))["best"])

    def test_missing_execution_profile(self):
        d = fixture()
        for stage in d["stages"]: del stage["decode_ms"]["b"]
        self.assertIsNone(plan(Scenario.parse(d))["best"])

    def test_multirail_is_not_summed(self):
        d = fixture()
        extra = copy.deepcopy(d["links"][0]); extra["id"] = "second-cable"
        extra["physical_group"] = "independent-but-unqualified"
        d["links"].append(extra)
        result = transfer(Scenario.parse(d), "a", "b", 1000, "a")
        self.assertAlmostEqual(result["cost_ms"], 1.1)
        self.assertEqual(len(result["hops"]), 1)

    def test_routes_are_directed(self):
        d = fixture(); d["links"] = []
        with self.assertRaises(NoRoute): transfer(Scenario.parse(d), "a", "b", 100, "a")

    def test_relay_cost_is_counted(self):
        d = fixture()
        d["devices"].append({"id": "c", "node": "c", "pool": "p-c", "backend": "test", "runtime_bytes": 0})
        d["pools"].append({"id": "p-c", "node": "c", "budget_bytes": 1000})
        for a, b in (("a", "c"), ("c", "b")):
            d["links"].append({"id": a+b, "source": a, "target": b, "payload_bytes_per_s": 1000000,
                               "fixed_latency_us": 100, "physical_group": a+b})
        d["coordinators"].append("c")
        route = transfer(Scenario.parse(d), "a", "b", 1000, "c")
        self.assertEqual(len(route["hops"]), 2)
        self.assertAlmostEqual(route["cost_ms"], 2.2)

    def test_deterministic(self):
        s = Scenario.parse(fixture())
        self.assertEqual(plan(s), plan(s))

    def test_measured_inputs_do_not_claim_qualification(self):
        d = fixture(); d["provenance"] = "measured"
        self.assertFalse(plan(Scenario.parse(d))["qualified"])


class DiagnosticTests(unittest.TestCase):
    def test_roundtrip_frame(self):
        a, b = socket.socketpair()
        with a, b:
            send_frame(a, b"test")
            self.assertEqual(receive_frame(b), b"test")

    def test_oversize_header_before_allocation(self):
        a, b = socket.socketpair()
        with a, b:
            a.sendall(struct.pack("!I", MAX_FRAME + 1))
            with self.assertRaises(ValueError): receive_frame(b)

    def test_truncated_frame(self):
        a, b = socket.socketpair()
        with a, b:
            a.sendall(struct.pack("!I", 10) + b"short")
            a.shutdown(socket.SHUT_WR)
            with self.assertRaises(EOFError): receive_frame(b)

    def test_empty_frame_rejected(self):
        a, b = socket.socketpair()
        with a, b:
            with self.assertRaises(ValueError): send_frame(a, b"")

    def test_loopback_real_socket(self):
        result = loopback(iterations=2, sizes=(64, 16384))
        self.assertFalse(result["qualified_for_inference"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(len(result["results"][0]["samples_ms"]), 2)

    def test_loopback_limit(self):
        for n in (0, 101, True):
            with self.assertRaises(ValueError): loopback(n)


class ProbeAndCLITests(unittest.TestCase):
    @patch("tensormeld.probe.shutil.which", return_value="nvidia-smi")
    @patch("tensormeld.probe._command", return_value="Example GPU, 96000, 90000, test-driver\n")
    def test_nvidia_inventory_parser(self, *_):
        gpu = nvidia_devices()[0]
        self.assertEqual(gpu["reported_memory_bytes"], 96000 * 1024**2)
        self.assertFalse(gpu["qualified"])

    @patch("tensormeld.probe.shutil.which", return_value=None)
    def test_absent_nvidia_is_not_failure(self, *_):
        self.assertEqual(nvidia_devices(), [])

    def test_cli_file_output(self):
        with tempfile.TemporaryDirectory() as t:
            source, target = Path(t) / "in.json", Path(t) / "out.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            self.assertEqual(main(["plan", str(source), "--out", str(target)]), 0)
            self.assertEqual(json.loads(target.read_text())["status"], "advisory_candidate")

    def test_cli_capacity_exit_code(self):
        with tempfile.TemporaryDirectory() as t:
            source, target = Path(t) / "in.json", Path(t) / "out.json"
            source.write_text(json.dumps(fixture()), encoding="utf-8")
            self.assertEqual(main(["plan", str(source), "--devices", "a", "--out", str(target)]), 2)

    def test_host_probe_json_serializable(self):
        report = probe()
        json.dumps(report, allow_nan=False)
        self.assertIn("host_memory", report)
        self.assertNotIn("hostname", report)


if __name__ == "__main__":
    unittest.main()
