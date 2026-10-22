from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tensormeld.hetero_hardware import parse_hardware
from tensormeld.hetero_model import parse_model_parts
from tensormeld.hetero_planner import Request, plan_hetero
from tensormeld.prefill_route import (
    PrefillRoute, crossover, interpolate, load_ttft_curves, measured_crossover, parse_ttft_curves,
)
from tensormeld.schema import ValidationError

from test_hetero_placement import toy_hardware, toy_model

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "examples" / "hetero" / "ds41-ttft-receipt.json"


def receipt(local, offload):
    return {"schema": "tensormeld/ttft-curve-v1", "routes": {
        "local": {"name": "pair", "points": [{"tokens": n, "ttft_s": t, "provenance": "measured"} for n, t in local]},
        "offload": {"name": "gpu", "points": [{"tokens": n, "ttft_s": t, "provenance": "measured"} for n, t in offload]},
    }}


class CurveTests(unittest.TestCase):
    def test_interpolate_and_extend(self):
        c = parse_ttft_curves(receipt([(1000, 1.0), (2000, 3.0)], [(1000, 4.0), (3000, 5.0)]))
        self.assertAlmostEqual(interpolate(c.local, 1500), 2.0)
        self.assertAlmostEqual(interpolate(c.local, 4000), 7.0)    # last segment extended
        self.assertAlmostEqual(interpolate(c.local, 0), 0.0)       # clamped at 0
        self.assertAlmostEqual(interpolate(c.offload, 2000), 4.5)

    def test_crossover_is_last_token_where_local_is_no_slower(self):
        # local = n/1000 s, offload = 2 + n/10000 s -> equal at n = 2222.2
        x = crossover(lambda n: n / 1000, lambda n: 2 + n / 10000)
        self.assertEqual(x, 2222)
        self.assertEqual(crossover(lambda n: 5.0, lambda n: 1.0), 0)
        self.assertEqual(crossover(lambda n: 1.0, lambda n: 5.0, 1, 100), 100)

    def test_measured_crossover_from_lab_receipt(self):
        c = load_ttft_curves(RECEIPT)
        x = measured_crossover(c)
        self.assertTrue(2048 < x < 4096, x)
        self.assertEqual(c.provenance, "partly-estimated")

    def test_rejects_bad_receipts(self):
        good = receipt([(1, 1.0), (2, 2.0)], [(1, 3.0), (2, 3.0)])
        for mutate in (lambda r: r.update(schema="x"),
                       lambda r: r["routes"].pop("offload"),
                       lambda r: r["routes"]["local"]["points"].pop(),
                       lambda r: r["routes"]["local"]["points"][0].update(tokens=0),
                       lambda r: r["routes"]["local"]["points"][0].update(ttft_s=-1),
                       lambda r: r["routes"]["local"]["points"][1].update(tokens=1),
                       lambda r: r["routes"]["local"]["points"][0].update(provenance="guess")):
            bad = json.loads(json.dumps(good))
            mutate(bad)
            with self.assertRaises(ValidationError):
                parse_ttft_curves(bad)


class DecideTests(unittest.TestCase):
    def test_threshold_and_prefix_cache_hit(self):
        r = PrefillRoute(2048, "measured", "pair", "gpu").validate()
        self.assertEqual(r.decide(2048)["route"], "local")
        self.assertEqual(r.decide(2049)["route"], "offload")
        hit = r.decide(8192, cached_tokens=7168)  # 1024-token suffix after a hit stays local
        self.assertEqual((hit["route"], hit["effective_tokens"]), ("local", 1024))
        self.assertEqual(r.decide(8192, offload_up=False)["route"], "local")
        self.assertEqual(PrefillRoute(0, "configured").decide(1)["route"], "offload")
        with self.assertRaises(ValidationError):
            r.decide(10, cached_tokens=11)
        with self.assertRaises(ValidationError):
            PrefillRoute(-1, "model").validate()


class HitRoutingTests(unittest.TestCase):
    def test_hit_compares_suffix_plus_context_with_full_offload(self):
        c = load_ttft_curves(RECEIPT)
        r = PrefillRoute(measured_crossover(c), "measured", "pair", "gpu", c).validate()
        # 32000-token prompt with 31744 cached: 256-token suffix, ~0.96 s on the pair vs ~11.8 s offloaded
        d = r.decide(32000, 31744)
        self.assertEqual((d["route"], d["rule"]), ("local", "curves"))
        self.assertLess(d["local_s"], 1.2)
        # a 4K suffix at 32K stays local (the offload must redo the whole prompt) although 4K > threshold
        self.assertGreater(32768 - 28672, r.local_max_tokens)
        self.assertEqual(r.decide(32768, 28672)["route"], "local")
        self.assertEqual(r.decide(32768, 16384)["route"], "offload")
        # no hit: the threshold
        self.assertEqual(r.decide(r.local_max_tokens)["rule"], "threshold")
        # without curves a hit uses the threshold on the suffix
        bare = PrefillRoute(r.local_max_tokens, "configured")
        self.assertEqual(bare.decide(32768, 28672)["route"], "offload")

    def test_route_round_trips_through_the_plan_section(self):
        c = load_ttft_curves(RECEIPT)
        r = PlanRouteTests().plan(curves=c)["prefill_route"]
        back = PrefillRoute.from_section(json.loads(json.dumps(r)))
        self.assertEqual(back.local_max_tokens, r["local_max_tokens"])
        self.assertAlmostEqual(back.curves.cached_s_per_token, 6.3e-6)
        self.assertEqual(back.decide(32000, 31744)["route"], "local")
        with self.assertRaises(ValidationError):
            PrefillRoute.from_section({"schema": "x"})


class PlanRouteTests(unittest.TestCase):
    def plan(self, **kw):
        curves = kw.pop("curves", None)
        req = Request(**{"prompt_tokens": 2048, "output_tokens": 128, **kw})
        return plan_hetero(parse_model_parts(toy_model()), parse_hardware(toy_hardware(fast_avail="prefill-only")),
                           req, top_k=5, ttft_curves=curves)

    def test_plan_carries_model_route(self):
        r = self.plan()
        route = r["prefill_route"]
        self.assertEqual(route["schema"], "tensormeld/prefill-route-v1")
        self.assertEqual(route["source"], "model")
        self.assertEqual(route["local_max_tokens"], route["model_crossover_tokens"])
        self.assertIsNone(route["measured_crossover_tokens"])

    def test_precedence_configured_over_measured_over_model(self):
        curves = parse_ttft_curves(receipt([(512, 1.5), (8192, 12.0)], [(512, 4.5), (8192, 5.0)]))
        m = self.plan(curves=curves)["prefill_route"]
        self.assertEqual(m["source"], "measured")
        self.assertEqual(m["local_max_tokens"], m["measured_crossover_tokens"])
        c = self.plan(curves=curves, prefill_local_max_tokens=1000)["prefill_route"]
        self.assertEqual((c["source"], c["local_max_tokens"]), ("configured", 1000))
        self.assertEqual(c["measured_crossover_tokens"], m["measured_crossover_tokens"])

    def test_offload_that_never_wins_keeps_everything_local(self):
        hw = toy_hardware(fast_avail="unavailable")  # only the pair (+ host lookups): no faster prefill elsewhere
        r = plan_hetero(parse_model_parts(toy_model()), parse_hardware(hw), Request(prompt_tokens=2048), top_k=3)
        route = r["prefill_route"]
        if route is not None:
            self.assertEqual(route["local_max_tokens"], 1 << 20)
            self.assertEqual(PrefillRoute(route["local_max_tokens"], "model").decide(1 << 20)["route"], "local")

    def test_request_validation(self):
        with self.assertRaises(ValidationError):
            Request(prefill_local_max_tokens=-1).validate()

    def test_cli_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            hw = Path(tmp) / "hw.json"
            md = Path(tmp) / "model.json"
            hw.write_text(json.dumps(toy_hardware(fast_avail="prefill-only")), encoding="utf-8")
            md.write_text(json.dumps(toy_model()), encoding="utf-8")
            out = Path(tmp) / "plan.json"
            proc = subprocess.run(
                [sys.executable, "-m", "tensormeld", "plan-hetero", str(md), str(hw), "--prompt", "2048",
                 "--ttft-receipt", str(RECEIPT), "--prefill-local-max-tokens", "3000", "--out", str(out)],
                cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
                capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            route = json.loads(out.read_text(encoding="utf-8"))["prefill_route"]
            self.assertEqual((route["source"], route["local_max_tokens"]), ("configured", 3000))
            self.assertTrue(2048 < route["measured_crossover_tokens"] < 4096)


if __name__ == "__main__":
    unittest.main()
