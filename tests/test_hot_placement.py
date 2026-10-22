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
from tensormeld.hot_placement import (
    HOT_PLACEMENT_FORMAT, HotInput, check_against_model, parse_hot_placement, resolve_devices,
)
from tensormeld.schema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
MB = 10**6
GB = 10**9
GIB = 2**30
LAYERS = 4
EXPERTS = 16
TOP_K = 2
EXPERT_BYTES = 100 * MB


def toy_model(moe=True):
    parts = [{"id": "embed", "kind": "embedding", "role": "lookup", "layer": -1, "weight_bytes": 1 * GB,
              "decode": {"read_bytes": 8192}, "io": {"out_bytes_per_row": 16384}}]
    for L in range(LAYERS):
        parts.append({"id": f"L{L}.attn", "kind": "attention", "role": "anchor", "layer": L,
                      "weight_bytes": 1 * GB, "decode": {"read_bytes": 1 * GB, "ops_per_row": 2 * 10**9},
                      "state": {"bytes_per_token": 1000, "bytes_fixed": 1 * MB}, "io": {"out_bytes_per_row": 16384}})
        parts.append({"id": f"L{L}.experts", "kind": "routed-experts", "role": "branch", "layer": L,
                      "weight_bytes": EXPERTS * EXPERT_BYTES,
                      "decode": {"read_bytes": 100 * MB, "read_bytes_extra_row": 90 * MB, "ops_per_row": 10**9},
                      "io": {"in_bytes_per_row": 4096, "out_bytes_per_row": 16384, "combine": "per-shard"},
                      "shardable": True})
    parts.append({"id": "head", "kind": "lm-head", "role": "head", "layer": LAYERS, "weight_bytes": 1 * GB,
                  "decode": {"read_bytes": 1 * GB, "ops_per_row": 2 * 10**9}, "prefill": {"rows": "last"}})
    raw = {"schema": "tensormeld/model-parts-v1", "model": {"id": "toy-moe"}, "provenance": "fixture",
           "layers": LAYERS, "stream_bytes_per_row": 65536, "parts": parts}
    if moe:
        raw["moe"] = {"experts": EXPERTS, "top_k": TOP_K, "expert_bytes": EXPERT_BYTES}
    return raw


def toy_hardware(eth=(100 * MB, 50), fast_gb=96, second_gpu=False, server_step_us=50):
    devices = [
        {"id": "fast", "node": "pc", "kind": "discrete_gpu", "backend": "cuda", "memory_bytes": fast_gb * GB,
         "mem_bytes_per_s": 1000 * GB, "prefill_ops_per_s": 100 * 10**12, "step_overhead_us": 10,
         "crossing_overhead_us": 20, "availability": "available", "supports": ["*"], "provenance": "estimate"},
        {"id": "ram", "node": "pc", "kind": "cpu", "backend": "host", "memory_bytes": 64 * GB,
         "mem_bytes_per_s": 50 * GB, "prefill_ops_per_s": 10**12, "availability": "available",
         "supports": ["toy.lookup"], "provenance": "estimate"},
        *({"id": f"{n}-gpu", "node": n, "kind": "integrated_gpu", "backend": "hip", "memory_bytes": 128 * GB,
           "mem_bytes_per_s": 100 * GB, "prefill_ops_per_s": 5 * 10**12, "step_overhead_us": server_step_us,
           "crossing_overhead_us": 20, "availability": "available", "supports": ["*"],
           "provenance": "measured"} for n in ("n1", "n2")),
    ]
    if second_gpu:
        devices.append({"id": "spare", "node": "pc", "kind": "discrete_gpu", "backend": "cuda",
                        "memory_bytes": 24 * GB, "mem_bytes_per_s": 900 * GB, "prefill_ops_per_s": 50 * 10**12,
                        "availability": "available", "supports": ["*"], "provenance": "estimate"})
    return {
        "schema": "tensormeld/hetero-hardware-v1",
        "nodes": [{"id": "pc", "model_files": ["toy-moe"]}, {"id": "n1", "model_files": ["toy-moe"]},
                  {"id": "n2", "model_files": ["toy-moe"]}],
        "devices": devices,
        "links": [
            {"id": "eth-1", "a": "pc", "b": "n1", "bytes_per_s": eth[0], "latency_us": eth[1], "provenance": "measured"},
            {"id": "eth-2", "a": "pc", "b": "n2", "bytes_per_s": eth[0], "latency_us": eth[1], "provenance": "measured"},
            {"id": "fast-12", "a": "n1", "b": "n2", "bytes_per_s": 5 * GB, "latency_us": 10, "provenance": "measured"},
        ],
        "groups": [{"id": "pair", "devices": ["n1-gpu", "n2-gpu"], "link": "fast-12"}],
    }


def make_placement(hot_per_layer=8, tier_per_layer=0, cache_per_layer=0, expected=None, expert_gib=10.0):
    """A synthetic placement: the first ids on the parts-host, the next on the tier, then the cache."""
    layers = []
    for L in range(LAYERS):
        ids = list(range(EXPERTS))
        entry = {"layer": L, "gpu-main": ids[:hot_per_layer],
                 "gpu-tier": ids[hot_per_layer:hot_per_layer + tier_per_layer]}
        if cache_per_layer:
            entry["host-cache"] = ids[hot_per_layer + tier_per_layer:hot_per_layer + tier_per_layer + cache_per_layer]
        layers.append(entry)
    return {
        "format": HOT_PLACEMENT_FORMAT,
        "model": {"family": "toy-moe", "file": "toy.bin", "model_identity": None, "layers": LAYERS,
                  "experts": EXPERTS, "top_k": TOP_K, "expert_bytes": EXPERT_BYTES, "canonical_split": EXPERTS // 2},
        "dense": {"device": "gpu-main", "layers": "all"},
        "devices": [
            {"id": "gpu-main", "kind": "cuda", "role": "parts-host", "budget_gib": 80.0, "reserve_gib": 9.5,
             "expert_gib": expert_gib},
            {"id": "gpu-tier", "kind": "cuda", "role": "expert-tier", "budget_gib": 20.0, "expert_gib": 20.0},
            {"id": "host-cache", "kind": "host", "role": "expert-cache", "budget_gib": 8.0, "expert_gib": 8.0},
            {"id": "servers", "kind": "hip", "role": "expert-server",
             "servers": [{"node": "n1", "experts": [0, EXPERTS // 2]}, {"node": "n2", "experts": [EXPERTS // 2, EXPERTS]}]},
        ],
        "layers": layers,
        "combine": {"identity": "canonical-halves-v1", "remote_return": "raw-per-expert"},
        "fit": {"objective": "fixture", "seed": 1},
        "expected": expected or {"R1": {"remote_layers_per_token": 2.0, "cold_vectors_per_token": 2.0, "tok_s": 40.0}},
    }


def run_plan(placement=None, hw=None, model=None, overrides=(), **req):
    p = parse_hot_placement(placement or make_placement())
    return plan_hetero(parse_model_parts(model or toy_model()), parse_hardware(hw or toy_hardware()),
                       Request(**{"prompt_tokens": 512, "output_tokens": 64, **req}), top_k=20,
                       hot=HotInput(p, tuple(overrides)))


class LoaderTests(unittest.TestCase):
    def test_parse_and_summary(self):
        p = parse_hot_placement(make_placement(hot_per_layer=5, tier_per_layer=2))
        self.assertEqual(p.expert_count("gpu-main"), 20)
        self.assertEqual(p.expert_count("gpu-tier"), 8)
        self.assertEqual(p.expert_count("host-cache"), 0)
        self.assertEqual(p.expert_count("servers"), LAYERS * EXPERTS)
        self.assertEqual(p.resident_bytes("gpu-main"), 20 * EXPERT_BYTES)
        self.assertEqual(p.per_layer("gpu-main"), (5, 5, 5, 5))
        s = p.summary()
        self.assertEqual(s["devices"]["servers"]["servers"][1]["experts"], [8, 16])
        self.assertEqual(s["expected"]["R1"]["rows"], 1)
        self.assertEqual(len(p.fingerprint), 64)
        self.assertEqual(p.ignored_keys, ())

    def test_rejects_unknown_format(self):
        raw = make_placement()
        raw["format"] = "tensormeld/hot-expert-placement-v2"
        with self.assertRaisesRegex(ValidationError, "format: unknown"):
            parse_hot_placement(raw)

    def test_rejects_bad_ids(self):
        cases = [
            (lambda r: r["layers"][1].__setitem__("gpu-main", [0, EXPERTS]), "out of range"),
            (lambda r: r["layers"][1].__setitem__("gpu-main", [3, 2]), "strictly ascending"),
            (lambda r: r["layers"][1].__setitem__("gpu-main", [2, 2]), "strictly ascending"),
            (lambda r: r["layers"][1].__setitem__("gpu-main", [1.0]), "out of range"),
            (lambda r: r["layers"][2].__setitem__("gpu-tier", [7]), "listed on both gpu-main and gpu-tier"),
            (lambda r: r["layers"][0].__setitem__("servers", [9]), "implicit"),
            (lambda r: r["layers"][0].__setitem__("gpu-nowhere", [9]), "unknown device"),
            (lambda r: r["layers"].pop(), "one entry per layer"),
            (lambda r: r["layers"][2].__setitem__("layer", 3), "layer order"),
        ]
        for mutate, message in cases:
            raw = make_placement()
            mutate(raw)
            with self.subTest(message=message), self.assertRaisesRegex(ValidationError, message):
                parse_hot_placement(raw)

    def test_rejects_bad_structure(self):
        cases = [
            (lambda r: r["devices"][3]["servers"][1].__setitem__("experts", [9, 16]), "tile"),
            (lambda r: r["devices"][3]["servers"][1].__setitem__("experts", [8, 15]), "tile"),
            (lambda r: r["devices"].pop(3), "exactly one parts-host and one expert-server"),
            (lambda r: r["dense"].__setitem__("device", "gpu-tier"), "parts-host"),
            (lambda r: r["dense"].__setitem__("layers", [0, 1]), '"all"'),
            (lambda r: r["expected"].pop("R1") and r["expected"].__setitem__(
                "R4", {"remote_layers_per_token": 1, "cold_vectors_per_token": 1, "tok_s": 1}), "R1: required"),
            (lambda r: r["expected"]["R1"].__setitem__("cold_vectors_per_token", 1.0), "below remote_layers"),
            (lambda r: r["expected"]["R1"].__setitem__("cold_vectors_per_token", 9.0), "top_k"),
            (lambda r: r["expected"]["R1"].__setitem__("remote_layers_per_token", 5.0), "above the 4 layers"),
            (lambda r: r["devices"][0].__setitem__("experts", 31), "declares 31 experts but layers list 32"),
            (lambda r: r["devices"][0].__setitem__("kind", "tpu"), "kind"),
            (lambda r: r["model"].__setitem__("model_identity", "XYZ"), "16 lowercase hex"),
        ]
        for mutate, message in cases:
            raw = make_placement()
            mutate(raw)
            with self.subTest(message=message), self.assertRaisesRegex(ValidationError, message):
                parse_hot_placement(raw)

    def test_memory_check(self):
        # 8 experts x 4 layers x 100 MB = 2.98 GiB: fits 3 GiB, not 2.9 GiB.
        parse_hot_placement(make_placement(expert_gib=3.0))
        with self.assertRaisesRegex(ValidationError, r"gpu-main: 32 experts .* > expert_gib 2.9"):
            parse_hot_placement(make_placement(expert_gib=2.9))
        raw = make_placement(hot_per_layer=2, cache_per_layer=3)
        raw["devices"][2]["budget_gib"] = 0.0
        del raw["devices"][2]["expert_gib"]  # expert_gib defaults to budget - reserve = 0
        with self.assertRaisesRegex(ValidationError, "host-cache: 12 experts"):
            parse_hot_placement(raw)

    def test_additive_keys_are_tolerated(self):
        raw = make_placement()
        raw["future"] = {"x": 1}
        raw["model"]["quant"] = "q2"
        raw["expected"]["R1"]["p99_ms"] = 30
        raw["expected"]["evaluated_on"] = "held-out prompts"
        raw["devices"][0]["experts"] = 32
        p = parse_hot_placement(raw)
        self.assertEqual(sorted(p.expected), ["R1"])
        self.assertIn("expected.evaluated_on", p.ignored_keys)
        self.assertIn("placement.future", p.ignored_keys)
        self.assertIn("model.quant", p.ignored_keys)
        self.assertIn("expected.R1.p99_ms", p.ignored_keys)

    def test_pattern_rows(self):
        raw = make_placement(expected={
            "R1": {"remote_layers_per_token": 1, "cold_vectors_per_token": 1, "tok_s": 1},
            "R4": {"remote_layers_per_token": 2, "cold_vectors_per_token": 5, "tok_s": 1},
            "batch-4": {"remote_layers_per_token": 2, "cold_vectors_per_token": 5, "tok_s": 1},
            "window": {"remote_layers_per_token": 2, "cold_vectors_per_token": 5, "tok_s": 1, "rows": 6},
            "R2": {"remote_layers_per_token": 1, "cold_vectors_per_token": 1, "tok_s": 1, "unit": "token"},
            "W8": {"remote_layers_per_token": 3, "cold_vectors_per_token": 9, "tok_s": 1, "tokens_per_window": 3.2},
        })
        p = parse_hot_placement(raw)
        self.assertEqual((p.expected["R4"].rows, p.expected["R4"].batch), (4, False))
        self.assertEqual((p.expected["batch-4"].rows, p.expected["batch-4"].batch), (4, True))
        self.assertEqual(p.expected["window"].rows, 6)
        self.assertEqual(p.expected["R2"].per_step(1.0), 2.0)
        self.assertEqual((p.expected["W8"].rows, p.expected["W8"].batch, p.expected["W8"].tokens_per_window),
                         (8, False, 3.2))
        raw["expected"]["R4"]["rows"] = 5
        with self.assertRaisesRegex(ValidationError, "contradicts"):
            parse_hot_placement(raw)

    def test_geometry_must_match_model(self):
        p = parse_hot_placement(make_placement())
        check_against_model(p, parse_model_parts(toy_model()))
        model = toy_model()
        model["moe"]["experts"] = 32
        with self.assertRaisesRegex(ValidationError, "16 experts per layer; the model has 32"):
            check_against_model(p, parse_model_parts(model))
        model = toy_model()
        model["moe"]["top_k"] = 4
        with self.assertRaisesRegex(ValidationError, "top_k"):
            check_against_model(p, parse_model_parts(model))
        model = toy_model()
        model["moe"]["expert_bytes"] = 1
        with self.assertRaisesRegex(ValidationError, "expert_bytes"):
            check_against_model(p, parse_model_parts(model))
        model = toy_model()
        model["model"]["id"] = "other"
        with self.assertRaisesRegex(ValidationError, "family"):
            check_against_model(p, parse_model_parts(model))
        raw = make_placement()
        raw["model"]["layers"] = 3
        raw["layers"].pop()
        with self.assertRaisesRegex(ValidationError, "3 layers; the model has 4"):
            check_against_model(parse_hot_placement(raw), parse_model_parts(toy_model()))
        # Without declared geometry only layers and family are checked.
        check_against_model(p, parse_model_parts(toy_model(moe=False)))


class DeviceMapTests(unittest.TestCase):
    def test_default_mapping(self):
        p = parse_hot_placement(make_placement())
        m = resolve_devices(p, parse_hardware(toy_hardware()))
        self.assertEqual(m, {"gpu-main": "fast", "gpu-tier": None, "host-cache": "ram",
                             "servers/n1": "n1-gpu", "servers/n2": "n2-gpu"})

    def test_used_device_needs_a_mapping(self):
        p = parse_hot_placement(make_placement(hot_per_layer=4, tier_per_layer=2))
        with self.assertRaisesRegex(ValidationError, "gpu-tier: holds 8 experts"):
            resolve_devices(p, parse_hardware(toy_hardware()))
        hw = parse_hardware(toy_hardware(second_gpu=True))
        with self.assertRaisesRegex(ValidationError, "ambiguous"):
            resolve_devices(p, hw)
        m = resolve_devices(p, hw, {"gpu-main": "fast"})
        self.assertEqual(m["gpu-tier"], "spare")
        m = resolve_devices(p, hw, {"gpu-main": "spare", "gpu-tier": "fast"})
        self.assertEqual((m["gpu-main"], m["gpu-tier"]), ("spare", "fast"))
        with self.assertRaisesRegex(ValidationError, "unknown placement device"):
            resolve_devices(p, hw, {"nope": "fast"})
        with self.assertRaisesRegex(ValidationError, "unknown hardware device"):
            resolve_devices(p, hw, {"gpu-main": "nope"})
        with self.assertRaisesRegex(ValidationError, "already used"):
            resolve_devices(p, hw, {"gpu-main": "fast", "gpu-tier": "fast"})


class DenseHotPlannerTests(unittest.TestCase):
    def test_no_candidate_without_placement(self):
        r = plan_hetero(parse_model_parts(toy_model()), parse_hardware(toy_hardware()),
                        Request(prompt_tokens=512), top_k=20)
        self.assertIsNone(r["dense_hot"])
        self.assertNotIn("dense+hot", r["best_by_mode"])

    def test_candidate_memory_and_hand_computed_penalty(self):
        r = run_plan()
        c = r["best_by_mode"]["dense+hot"]
        self.assertTrue(c["feasible"], c["problems"])
        self.assertEqual(c["mode"], "dense+hot")
        # fast: embed 1 GB + 4 x (1 GB attn + 1 MB + 8192 x 1000 B state) + head 1 GB + 32 hot experts.
        dense = 1 * GB + 4 * (1 * GB + MB + 8192 * 1000) + 1 * GB
        self.assertAlmostEqual(c["memory_gib"]["fast"], round((dense + 32 * EXPERT_BYTES) / GIB, 2), places=2)
        # Each server holds its id half of every layer, from the placement: 8 x 4 x 100 MB.
        self.assertAlmostEqual(c["memory_gib"]["n1-gpu"], round(32 * EXPERT_BYTES / GIB, 2), places=2)
        hot = c["decode"]["hot"]
        # R1: 2 remote layers, 1 cold vector each -> one server per remote layer (w = 1/2 per server).
        self.assertEqual(hot["servers_per_remote_layer"], 1.0)
        rb = hot["remote_branch_us"]
        self.assertAlmostEqual(rb["send"], 50 + 2048 / 100e6 * 1e6, places=2)
        self.assertAlmostEqual(rb["return"], 50 + 8192 / 100e6 * 1e6, places=2)
        # Server compute: 1 vector x 100 MB at 100 GB/s = 1000 us (> 0.5 G ops at 5 T ops/s) + 50 us step.
        self.assertAlmostEqual(rb["compute"], 1050.0, places=2)
        self.assertAlmostEqual(rb["crossing"], 40.0, places=2)
        # Local branch on fast: 100 MB at 1 TB/s = 100 us.
        self.assertAlmostEqual(hot["local_branch_us_per_layer"], 100.0, places=2)
        exposed = rb["total"] - 100.0
        self.assertAlmostEqual(hot["exposed_us_per_remote_layer"], exposed, places=1)
        self.assertAlmostEqual(c["decode"]["step_ms"], hot["all_local_step_ms"] + 2 * exposed / 1e3, places=2)
        self.assertEqual(c["hot_experts"]["gpu-main"], {"device": "fast", "experts": 32,
                                                        "gib": round(32 * EXPERT_BYTES / GIB, 2)})
        rep = r["dense_hot"]
        self.assertEqual(rep["patterns"]["R1"]["file"]["tok_s"], 40.0)
        self.assertEqual(rep["patterns"]["R1"]["planner"]["step_ms"], c["decode"]["step_ms"])
        self.assertEqual(rep["device_map"]["servers/n2"], "n2-gpu")

    def test_overlap_is_not_counted_twice(self):
        # Fast links and servers: the round trip hides under the local branch, so nothing is added.
        hw = toy_hardware(eth=(100 * GB, 1), server_step_us=0)
        hw["devices"][2]["mem_bytes_per_s"] = hw["devices"][3]["mem_bytes_per_s"] = 10_000 * GB
        hw["devices"][2]["prefill_ops_per_s"] = hw["devices"][3]["prefill_ops_per_s"] = 10**15
        hw["devices"][2]["crossing_overhead_us"] = hw["devices"][3]["crossing_overhead_us"] = 0
        hw["devices"][0]["crossing_overhead_us"] = 0
        c = run_plan(hw=hw)["best_by_mode"]["dense+hot"]
        hot = c["decode"]["hot"]
        self.assertLess(hot["remote_branch_us"]["total"], hot["local_branch_us_per_layer"])
        self.assertEqual(hot["exposed_us_per_remote_layer"], 0.0)
        self.assertEqual(c["decode"]["step_ms"], hot["all_local_step_ms"])

    def test_zero_remote_layers_is_all_local(self):
        placement = make_placement(expected={"R1": {"remote_layers_per_token": 0, "cold_vectors_per_token": 0,
                                                    "tok_s": 99.0}})
        c = run_plan(placement)["best_by_mode"]["dense+hot"]
        self.assertEqual(c["decode"]["step_ms"], c["decode"]["hot"]["all_local_step_ms"])
        self.assertEqual(c["decode"]["hot"]["remote_branch_us"]["total"], 0.0)

    def test_two_servers_share_the_input(self):
        placement = make_placement(expected={"R1": {"remote_layers_per_token": 2, "cold_vectors_per_token": 4,
                                                    "tok_s": 1, "servers_per_remote_layer": 2}})
        hot = run_plan(placement)["best_by_mode"]["dense+hot"]["decode"]["hot"]
        self.assertEqual(hot["servers_per_remote_layer"], 2.0)
        # Separate links: each server gets its input; one vector returns per server.
        self.assertAlmostEqual(hot["remote_branch_us"]["send"], 50 + 4096 / 100e6 * 1e6, places=2)
        self.assertAlmostEqual(hot["remote_branch_us"]["return"], 50 + 16384 / 100e6 * 1e6, places=2)
        # Default estimate for 2 vectors over 2 halves: 2 * (1 - 0.5^2) = 1.5 servers.
        placement["expected"]["R1"].pop("servers_per_remote_layer")
        hot = run_plan(placement)["best_by_mode"]["dense+hot"]["decode"]["hot"]
        self.assertEqual(hot["servers_per_remote_layer"], 1.5)
        # Shared NIC: both inputs serialize on one group.
        hw = toy_hardware()
        for link in hw["links"][:2]:
            link["shared_group"] = "pc-nic"
        placement["expected"]["R1"]["servers_per_remote_layer"] = 2
        hot = run_plan(placement, hw=hw)["best_by_mode"]["dense+hot"]["decode"]["hot"]
        self.assertAlmostEqual(hot["remote_branch_us"]["send"], 50 + 2 * 4096 / 100e6 * 1e6, places=2)

    def test_rows_patterns(self):
        expected = {
            "R1": {"remote_layers_per_token": 2, "cold_vectors_per_token": 2, "tok_s": 40},
            "R4": {"remote_layers_per_token": 3, "cold_vectors_per_token": 6, "tok_s": 100},
            "B4": {"remote_layers_per_token": 4, "cold_vectors_per_token": 8, "tok_s": 90},
        }
        r = run_plan(make_placement(expected=expected))
        self.assertIsNotNone(r["dense_hot"]["patterns"]["R1"]["planner"])
        self.assertIsNone(r["dense_hot"]["patterns"]["R4"]["planner"])
        r = run_plan(make_placement(expected=expected), decode_rows=4)
        pats = r["dense_hot"]["patterns"]
        self.assertEqual(pats["R4"]["planner"]["rows"], 4)
        self.assertEqual(pats["B4"]["file"]["tok_s"], 90)
        self.assertGreater(pats["B4"]["planner"]["step_ms"], pats["R4"]["planner"]["step_ms"])
        self.assertNotIn("tok_s_at_file_tokens_per_window", pats["R4"]["planner"])
        c = r["best_by_mode"]["dense+hot"]
        self.assertEqual(c["decode"]["hot"]["pattern"], "R4")
        self.assertEqual(c["decode"]["step_ms"], pats["R4"]["planner"]["step_ms"])
        expected["W4"] = {"remote_layers_per_token": 3, "cold_vectors_per_token": 6, "tok_s": 50,
                          "tokens_per_window": 2.0}
        pats = run_plan(make_placement(expected=expected), decode_rows=4)["dense_hot"]["patterns"]
        self.assertEqual(pats["W4"]["planner"]["step_ms"], pats["R4"]["planner"]["step_ms"])
        self.assertAlmostEqual(pats["W4"]["planner"]["tok_s_at_file_tokens_per_window"],
                               2.0 * 1e3 / pats["W4"]["planner"]["step_ms"], places=1)
        r = run_plan(make_placement(expected=expected), decode_rows=6)
        self.assertEqual(r["best_by_mode"]["dense+hot"]["decode"]["hot"]["pattern"], "R1")
        self.assertIn("no expected entry for 6 rows", " ".join(r["best_by_mode"]["dense+hot"]["decode"]["hot"]["notes"]))

    def test_host_cache_copies(self):
        placement = make_placement(hot_per_layer=6, cache_per_layer=2, expected={
            "R1": {"remote_layers_per_token": 0, "cold_vectors_per_token": 0, "tok_s": 1,
                   "ram_vectors_per_token": 2.0}})
        c = run_plan(placement)["best_by_mode"]["dense+hot"]
        # 2 x 100 MB at the declared 25 GB/s = 8 ms, not overlapped.
        self.assertAlmostEqual(c["decode"]["hot"]["host_copy_ms"], 8.0, places=3)
        self.assertAlmostEqual(c["memory_gib"]["ram"], round(8 * EXPERT_BYTES / GIB, 2), places=2)
        placement["expected"]["R1"].pop("ram_vectors_per_token")
        c = run_plan(placement)["best_by_mode"]["dense+hot"]
        self.assertIn("not priced", " ".join(c["decode"]["hot"]["notes"]))

    def test_infeasible_when_hardware_is_smaller(self):
        # The file's own budget allows 70 GiB of experts; the hardware device has 8 GB.
        placement = make_placement(hot_per_layer=16, expert_gib=70.0,
                                   expected={"R1": {"remote_layers_per_token": 0, "cold_vectors_per_token": 0,
                                                    "tok_s": 1}})
        r = run_plan(placement, hw=toy_hardware(fast_gb=8))
        self.assertNotIn("dense+hot", r["best_by_mode"])
        self.assertFalse(r["dense_hot"]["feasible"])
        self.assertTrue(any("fast: needs" in x for x in r["dense_hot"]["problems"]))

    def test_paired_prefill_does_not_charge_hot_copies_twice(self):
        from tensormeld.hetero_planner import _Ctx, _Hot, _hot_candidate, _whole
        model, hw = parse_model_parts(toy_model()), parse_hardware(toy_hardware())
        ctx = _Ctx(model, hw, Request(prompt_tokens=512))
        h = _Hot(ctx, HotInput(parse_hot_placement(make_placement())))
        whole = _whole(ctx, "fast")
        c = _hot_candidate(ctx, h, whole, "pd:whole:fast=>dense+hot")
        self.assertEqual(c["mode"], "prefill-decode(single -> dense+hot)")
        # Prefill keeps every routed tensor on fast: the 32 hot copies are inside them.
        self.assertAlmostEqual(c["memory_gib"]["fast"], round(sum(p.weight_bytes for p in model.parts) / GIB
                                                              + 4 * (MB + 8192 * 1000) / GIB, 2), places=2)

    def test_unavailable_server(self):
        hw = toy_hardware()
        hw["devices"][3]["availability"] = "unavailable"
        r = run_plan(hw=hw)
        self.assertFalse(r["dense_hot"]["feasible"])
        self.assertIn("servers/n2 maps to unavailable device n2-gpu", r["dense_hot"]["problems"])

    def test_geometry_mismatch_is_an_error(self):
        model = toy_model()
        model["moe"]["top_k"] = 3
        with self.assertRaisesRegex(ValidationError, "top_k"):
            run_plan(model=model)


class CliTests(unittest.TestCase):
    def _run(self, *args):
        env = {"PYTHONPATH": str(ROOT / "src"), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        return subprocess.run([sys.executable, "-m", "tensormeld", "plan-hetero", *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=120)

    def test_cli_dense_hot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, data in (("model.json", toy_model()), ("hw.json", toy_hardware(second_gpu=True)),
                               ("placement.json", make_placement(hot_per_layer=6, tier_per_layer=2))):
                (tmp / name).write_text(json.dumps(data), encoding="utf-8")
            base = [str(tmp / "model.json"), str(tmp / "hw.json"), "--prompt", "512"]
            proc = self._run(*base, "--mode", "dense+hot", "--placement", str(tmp / "placement.json"),
                             "--placement-device", "gpu-main=fast", "--placement-device", "gpu-tier=spare",
                             "--objective", "decode", "--out", str(tmp / "plan.json"))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((tmp / "plan.json").read_text(encoding="utf-8"))
            self.assertIn("dense+hot", result["best_by_mode"])
            c = result["best_by_mode"]["dense+hot"]
            self.assertAlmostEqual(c["memory_gib"]["spare"], round(8 * EXPERT_BYTES / GIB, 2), places=2)
            self.assertEqual(result["dense_hot"]["source"], "placement.json")
            self.assertEqual(result["dense_hot"]["device_map"]["gpu-tier"], "spare")
            self.assertNotIn('"_hot"', json.dumps(result))
            self.assertNotIn('"_assign"', json.dumps(result))
            proc = self._run(*base, "--mode", "dense+hot")
            self.assertEqual(proc.returncode, 1)
            self.assertIn("needs --placement", proc.stderr)
            proc = self._run(*base, "--placement", str(tmp / "placement.json"))
            self.assertEqual(proc.returncode, 1)
            self.assertIn("need --mode dense+hot", proc.stderr)
            proc = self._run(*base, "--mode", "dense+hot", "--placement", str(tmp / "placement.json"))
            self.assertEqual(proc.returncode, 1)
            self.assertIn("ambiguous", proc.stderr)
            bad = make_placement()
            bad["format"] = "other/v9"
            (tmp / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
            proc = self._run(*base, "--mode", "dense+hot", "--placement", str(tmp / "bad.json"))
            self.assertEqual(proc.returncode, 1)
            self.assertIn("format: unknown", proc.stderr)


if __name__ == "__main__":
    unittest.main()
