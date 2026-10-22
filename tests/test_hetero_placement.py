from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from tensormeld.hetero_hardware import parse_hardware
from tensormeld.hetero_model import model_parts_summary, parse_model_parts
from tensormeld.hetero_planner import Request, evaluate_decode, plan_hetero, _Ctx
from tensormeld.schema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "hetero"
MB = 10**6
GB = 10**9


def toy_model(layers=4, expert_bytes=40 * GB, dense_bytes=1 * GB):
    """A small MoE: per layer a dense anchor (read fully every token) and big sparse experts."""
    parts = [{"id": "embed", "kind": "embedding", "role": "lookup", "layer": -1, "weight_bytes": 1 * GB,
              "decode": {"read_bytes": 8192}, "io": {"out_bytes_per_row": 16384}}]
    for L in range(layers):
        parts.append({"id": f"L{L}.attn", "kind": "attention", "role": "anchor", "layer": L,
                      "weight_bytes": dense_bytes, "decode": {"read_bytes": dense_bytes, "ops_per_row": 2 * 10**9},
                      "state": {"bytes_per_token": 1000, "bytes_fixed": 1 * MB},
                      "io": {"out_bytes_per_row": 16384}, "shardable": True, "requires": ["toy.attn"]})
        parts.append({"id": f"L{L}.experts", "kind": "routed-experts", "role": "branch", "layer": L,
                      "weight_bytes": expert_bytes,
                      "decode": {"read_bytes": 100 * MB, "read_bytes_extra_row": 90 * MB, "ops_per_row": 10**9},
                      "io": {"in_bytes_per_row": 4096, "out_bytes_per_row": 16384}, "shardable": True,
                      "requires": ["toy.moe"]})
    parts.append({"id": "head", "kind": "lm-head", "role": "head", "layer": layers, "weight_bytes": 1 * GB,
                  "decode": {"read_bytes": 1 * GB, "ops_per_row": 2 * 10**9}, "prefill": {"rows": "last"}})
    return {"schema": "tensormeld/model-parts-v1", "model": {"id": "toy-moe"}, "provenance": "fixture",
            "layers": layers, "stream_bytes_per_row": 65536, "parts": parts}


def toy_hardware(eth=(100 * MB, 50), fast_avail="available"):
    return {
        "schema": "tensormeld/hetero-hardware-v1",
        "nodes": [{"id": "pc", "model_files": ["toy-moe"]}, {"id": "n1", "model_files": ["toy-moe"]},
                  {"id": "n2", "model_files": ["toy-moe"]}],
        "devices": [
            {"id": "fast", "node": "pc", "kind": "discrete_gpu", "backend": "cuda", "memory_bytes": 96 * GB,
             "mem_bytes_per_s": 1000 * GB, "prefill_ops_per_s": 100 * 10**12, "step_overhead_us": 10,
             "crossing_overhead_us": 20, "availability": fast_avail, "supports": ["*"], "provenance": "estimate"},
            *({"id": f"{n}-gpu", "node": n, "kind": "integrated_gpu", "backend": "hip", "memory_bytes": 128 * GB,
               "mem_bytes_per_s": 100 * GB, "prefill_ops_per_s": 5 * 10**12, "step_overhead_us": 50,
               "crossing_overhead_us": 20, "availability": "available", "supports": ["*"],
               "provenance": "measured"} for n in ("n1", "n2")),
        ],
        "links": [
            {"id": "eth-1", "a": "pc", "b": "n1", "bytes_per_s": eth[0], "latency_us": eth[1], "provenance": "measured"},
            {"id": "eth-2", "a": "pc", "b": "n2", "bytes_per_s": eth[0], "latency_us": eth[1], "provenance": "measured"},
            {"id": "fast-12", "a": "n1", "b": "n2", "bytes_per_s": 5 * GB, "latency_us": 10, "provenance": "measured"},
        ],
        "groups": [{"id": "pair", "devices": ["n1-gpu", "n2-gpu"], "link": "fast-12"}],
    }


def plan(model=None, hw=None, **req):
    return plan_hetero(parse_model_parts(model or toy_model()), parse_hardware(hw or toy_hardware()),
                       Request(**{"prompt_tokens": 2048, "output_tokens": 128, **req}), top_k=20)


class ModelPartsTests(unittest.TestCase):
    def test_parse_and_summary(self):
        m = parse_model_parts(toy_model())
        s = model_parts_summary(m)
        self.assertEqual(s["layers"], 4)
        self.assertEqual(s["weight_bytes"], 1 * GB + 4 * (1 * GB + 40 * GB) + 1 * GB)
        self.assertEqual(len(m.fingerprint), 64)
        self.assertEqual(m.part("L0.experts").decode_bytes(3), 100 * MB + 2 * 90 * MB)

    def test_rejects_bad_descriptors(self):
        raw = toy_model()
        raw["parts"][1]["role"] = "boss"
        with self.assertRaisesRegex(ValidationError, "role"):
            parse_model_parts(raw)
        raw = toy_model()
        raw["parts"] = [p for p in raw["parts"] if p["id"] != "L2.attn"]
        with self.assertRaisesRegex(ValidationError, "layer 2: needs at least one anchor"):
            parse_model_parts(raw)
        raw = toy_model()
        raw["parts"][2]["io"] = {"in_bytes_per_row": 1}
        with self.assertRaisesRegex(ValidationError, "out_bytes_per_row"):
            parse_model_parts(raw)
        raw = toy_model()
        raw["surprise"] = 1
        with self.assertRaisesRegex(ValidationError, "unknown"):
            parse_model_parts(raw)
        raw = toy_model()
        raw["parts"].append(copy.deepcopy(raw["parts"][1]))
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            parse_model_parts(raw)

    def test_bounded_prefill_rows(self):
        raw = toy_model()
        raw["parts"][3]["prefill"] = {"rows": {"max": 300}}
        p = parse_model_parts(raw).part("L1.attn")
        self.assertEqual(p.rows_for_prompt(100), 100)
        self.assertEqual(p.rows_for_prompt(8192), 300)


class HardwareTests(unittest.TestCase):
    def test_route_and_overrides(self):
        hw = parse_hardware(toy_hardware())
        self.assertEqual([link.id for link in hw.route("n1", "n2")], ["fast-12"])
        self.assertEqual([link.id for link in hw.route("pc", "n2")], ["eth-2"])
        self.assertEqual(hw.route("pc", "pc"), ())
        h2 = hw.with_overrides({"fast": "unavailable"}, {"eth-1": (10**9, 5.0)})
        self.assertEqual(h2.device("fast").availability, "unavailable")
        self.assertEqual(h2.link("eth-1").bytes_per_s, 10**9)
        self.assertEqual(h2.link("eth-1").provenance, "estimate")
        self.assertNotEqual(h2.fingerprint, hw.fingerprint)

    def test_rejects_bad_hardware(self):
        raw = toy_hardware()
        raw["devices"][0]["availability"] = "sometimes"
        with self.assertRaisesRegex(ValidationError, "availability"):
            parse_hardware(raw)
        raw = toy_hardware()
        raw["links"][0]["b"] = "nowhere"
        with self.assertRaisesRegex(ValidationError, "endpoints"):
            parse_hardware(raw)
        raw = toy_hardware()
        raw["groups"][0]["devices"] = ["n1-gpu"]
        with self.assertRaisesRegex(ValidationError, "2..16"):
            parse_hardware(raw)


class PlannerTests(unittest.TestCase):
    def test_hand_computed_decode_step(self):
        # Whole model on the fast device (1 TB/s): embed 8 KB, 4 x (1 GB attn + 100 MB experts + 10 us floor),
        # 1 GB head; memory fit is not checked by evaluate_decode.
        model = parse_model_parts(toy_model())
        hw = parse_hardware(toy_hardware())
        ctx = _Ctx(model, hw, Request(prompt_tokens=2048))
        assign = {p.id: "fast" for p in model.parts}
        d = evaluate_decode(ctx, assign)
        expected_us = (8192 / 1000e9) * 1e6 + 4 * (1000 + 100 + 10) + 1000  # bytes / 1 TB/s, step floor 10 us
        self.assertAlmostEqual(d["step_ms"], round(expected_us / 1e3, 3), places=3)
        self.assertEqual(d["crossings_per_step"], 0)

    def test_parts_split_puts_dense_on_fast_and_experts_on_capacity(self):
        r = plan(objective="decode")
        self.assertEqual(r["status"], "OK")
        best = r["best"]
        self.assertIn(best["mode"].split("(")[0], ("hetero-parts", "prefill-decode"))
        on_fast = best["placement"]["fast"]["kinds"]
        self.assertEqual(on_fast["attention"], "0-3")
        self.assertIn("routed-experts", best["placement"]["pair"]["kinds"])
        # Remote experts cost one round trip per layer whose experts are not local.
        remote_layers = 4 - len(on_fast.get("routed-experts", "").split(",")) if on_fast.get("routed-experts") else 4
        self.assertGreaterEqual(best["decode"]["crossings_per_step"], 1)
        self.assertLessEqual(best["decode"]["crossings_per_step"], remote_layers + 1)
        self.assertFalse(r["qualified"] or r["executable"])

    def test_whole_model_too_big_for_fast_device_is_infeasible(self):
        r = plan()
        names = {c["name"]: c for c in r["infeasible"]}
        self.assertIn("whole:fast", names)
        self.assertRegex(names["whole:fast"]["problems"][0], "fast: needs")

    def test_slow_link_prefers_fewer_crossings(self):
        slow = plan(hw=toy_hardware(eth=(10 * MB, 2000)), objective="decode")
        fast = plan(hw=toy_hardware(eth=(10 * GB, 5)), objective="decode")
        self.assertLessEqual(slow["best"]["decode"]["crossings_per_step"], fast["best"]["decode"]["crossings_per_step"])
        self.assertLess(fast["best"]["decode"]["step_ms"], slow["best"]["decode"]["step_ms"])

    def test_unavailable_device_and_fallback(self):
        r = plan()
        self.assertIn("fast", r["best"]["devices"])
        down = plan(hw=toy_hardware(fast_avail="unavailable"))
        self.assertNotIn("fast", down["best"]["devices"])
        self.assertEqual(down["best"]["mode"], "engine-group")
        pre = plan(hw=toy_hardware(fast_avail="preemptible"))
        self.assertEqual(pre["fallback"]["without"], ["fast"])
        self.assertEqual(pre["fallback"]["best"]["mode"], "engine-group")
        self.assertIn("co_resident", pre["fallback"])

    def test_prefill_only_device_never_decodes(self):
        r = plan(hw=toy_hardware(fast_avail="prefill-only"))
        best = r["best"]
        self.assertTrue(best["mode"].startswith("prefill-decode"), best["mode"])
        self.assertNotIn("fast", best["placement"])
        self.assertIn("fast", best["prefill_placement"])
        self.assertGreater(best["handoff"]["state_bytes"], 0)

    def test_capabilities_restrict_placement(self):
        hw = toy_hardware()
        hw["devices"][0]["supports"] = ["toy.attn"]  # no MoE kernels on the fast device
        r = plan(hw=hw, objective="decode")
        fast = r["best"]["placement"].get("fast", {"kinds": {}})["kinds"]
        self.assertNotIn("routed-experts", fast)
        hw["groups"][0]["supports"] = ["toy.attn"]  # the pair cannot shard MoE either
        r = plan(hw=hw, objective="decode")
        self.assertNotIn("pair", r["best"]["placement"])

    def test_disk_table_stays_where_the_file_is(self):
        raw = toy_model()
        raw["parts"].append({"id": "L1.table", "kind": "ngram-table", "role": "lookup", "layer": 1,
                             "weight_bytes": 0, "storage_bytes": 100 * GB, "decode": {"read_bytes": 4096},
                             "io": {"out_bytes_per_row": 4096}})
        hw = toy_hardware()
        hw["nodes"][0]["model_files"] = []
        r = plan(model=raw, hw=hw)
        for c in r["candidates"]:
            fast_kinds = c["placement"].get("fast", {"kinds": {}})["kinds"]
            self.assertNotIn("ngram-table", fast_kinds)

    def test_replicas_when_each_device_holds_the_model(self):
        r = plan(model=toy_model(layers=2, expert_bytes=20 * GB), objective="throughput")
        modes = r["best_by_mode"]
        self.assertIn("replicas", modes)
        self.assertGreater(modes["replicas"]["aggregate_tokens_per_s"], modes["single"]["decode"]["tokens_per_s"])

    def test_per_shard_combine_returns_every_partial(self):
        """An exact combine order ((shared + low) + high) sends both halves back to the home."""
        model = parse_model_parts(toy_model())
        hw = parse_hardware(toy_hardware())
        ctx = _Ctx(model, hw, Request(prompt_tokens=2048))
        assign = {p.id: ("pair" if p.kind == "routed-experts" else "fast") for p in model.parts}
        reduce = evaluate_decode(ctx, assign)
        raw = toy_model()
        for p in raw["parts"]:
            if p["kind"] == "routed-experts":
                p["io"]["combine"] = "per-shard"
        ctx2 = _Ctx(parse_model_parts(raw), hw, Request(prompt_tokens=2048))
        per = evaluate_decode(ctx2, assign)
        self.assertEqual(per["link_bytes_per_step"]["eth-1"] - reduce["link_bytes_per_step"]["eth-1"], 4 * 16384)
        self.assertGreater(per["step_ms"], reduce["step_ms"])

    def test_replicated_fraction_of_a_sharded_part(self):
        raw = toy_model()
        raw["parts"][1]["replicated_fraction"] = 0.25
        p = parse_model_parts(raw).part("L0.attn")
        self.assertAlmostEqual(p.shard_share(2), 0.625)
        self.assertEqual(p.shard_share(1), 1.0)
        raw["parts"][1]["replicated_fraction"] = 2
        with self.assertRaisesRegex(ValidationError, "replicated_fraction"):
            parse_model_parts(raw)

    def test_request_validation(self):
        with self.assertRaisesRegex(ValidationError, "objective"):
            plan(objective="fastest")
        with self.assertRaisesRegex(ValidationError, "context_tokens"):
            plan(prompt_tokens=4096, context_tokens=1024)


class ExampleDescriptorTests(unittest.TestCase):
    def test_examples_parse_and_plan(self):
        hw = json.loads((EXAMPLES / "lab-hardware.json").read_text(encoding="utf-8"))
        for name in ("deepseek-v4.1-flash.parts.json", "qwen3.8-flash-next.parts.json"):
            raw = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
            r = plan_hetero(parse_model_parts(raw), parse_hardware(hw), Request(prompt_tokens=4096), top_k=3)
            self.assertEqual(r["status"], "OK", name)
            self.assertTrue(r["unmeasured_inputs"])

    def test_cli_plan_hetero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "plan.json"
            proc = subprocess.run(
                [sys.executable, "-m", "tensormeld", "plan-hetero", str(EXAMPLES / "deepseek-v4.1-flash.parts.json"),
                 str(EXAMPLES / "lab-hardware.json"), "--prompt", "4096", "--unavailable", "rtx",
                 "--link", "eth-pc-c1=1150000000:20", "--out", str(out)],
                cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
                capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["best"]["mode"], "engine-group")
            self.assertNotIn("_best_assigns", result)


class LinkProbeTests(unittest.TestCase):
    def test_loopback_probe_and_token(self):
        from tensormeld import link_probe
        import io
        token = "ab" * 16
        log = io.StringIO()
        port_holder = {}

        def server():
            import socket as s
            probe = s.socket()
            probe.bind(("127.0.0.1", 0))
            port_holder["port"] = probe.getsockname()[1]
            probe.close()
            port_holder["ready"].set()
            link_probe.serve("127.0.0.1", port_holder["port"], 10.0, out=log, token=token)

        port_holder["ready"] = threading.Event()
        t = threading.Thread(target=server, daemon=True)
        t.start()
        port_holder["ready"].wait(5)
        for _ in range(50):
            if "listening" in log.getvalue():
                break
            threading.Event().wait(0.05)
        r = link_probe.run("127.0.0.1", port_holder["port"], token=token, ladder=(64, 4096), rounds=5,
                           bulk_bytes=1 << 20)
        t.join(5)
        self.assertEqual(r["schema"], "tensormeld/link-probe-v1")
        self.assertEqual(len(r["measurements"]), 4)
        self.assertEqual(r["suggested_link"]["provenance"], "measured")
        with self.assertRaisesRegex(ValueError, "private"):
            link_probe.run("8.8.8.8", 1, token=token)
        with self.assertRaisesRegex(ValueError, "token"):
            link_probe.run("127.0.0.1", 1, token="short")


if __name__ == "__main__":
    unittest.main()
