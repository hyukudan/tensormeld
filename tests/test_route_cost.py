from __future__ import annotations

import unittest

from tensormeld.route_cost import (
    SEED_SERVICE_PROFILES, CostPolicy, Curve, GpuKey, LocalCandidate, NodeProfile, ServiceCandidate, ServiceProfile,
    choose, local_cost, offload_cost,
)
from tensormeld.schema import ValidationError

NODE = NodeProfile(Curve(((1024, 1.2), (8192, 9.0), (32768, 36.0))), restore_s=0.01)
GPU = ServiceProfile(GpuKey("g", 96, 64, 40), "m", Curve(((2048, 0.8), (8192, 2.5))), Curve(((2048, 0.15), (8192, 0.27))),
                     service_s=0.06, import_pause_s=0.05)


class CurveTests(unittest.TestCase):
    def test_interpolates_and_extrapolates(self):
        c = Curve(((1000, 1.0), (2000, 3.0)))
        self.assertAlmostEqual(c(1500), 2.0)
        self.assertAlmostEqual(c(3000), 5.0)
        self.assertAlmostEqual(c(500), 0.0)
        self.assertEqual(c(0), 0.0)
        self.assertAlmostEqual(Curve(((100, 1.0),))(50), 0.5)

    def test_parse_bounds(self):
        for bad in ([], [[1]], [[-1, 1.0]], [[1, -1.0]], [[1, 1.0], [1, 2.0]], "x"):
            with self.assertRaises(ValidationError):
                Curve.parse(bad, "c")


class ChooseTests(unittest.TestCase):
    def test_cost_terms(self):
        c = LocalCandidate("r", 0.5, 2, 4096, NODE)
        rows_cost = NODE.prefill(8192 - 4096)
        self.assertAlmostEqual(local_cost(8192, c), 0.5 + 0.01 + rows_cost * 3)
        s = ServiceCandidate("s", 1.0, GPU)
        self.assertAlmostEqual(offload_cost(8192, s, c), 1.0 + 0.06 + 2.5 + 0.27 + 2 * 0.05)

    def test_idle_short_prompt_stays_local_and_long_prompt_offloads(self):
        idle = [LocalCandidate("r", 0.0, 0, 0, NODE)]
        svc = [ServiceCandidate("s", 0.0, GPU)]
        self.assertEqual(choose(1500, idle, svc, CostPolicy()).place, "local")           # below the idle floor
        long = choose(8192, idle, svc, CostPolicy())
        self.assertEqual((long.place, long.service), ("offload", "s"))

    def test_active_decodes_make_the_service_a_candidate_below_the_floor(self):
        busy = [LocalCandidate("r", 0.0, 4, 0, NODE)]
        fast = ServiceProfile(GpuKey("g", 96, 64, 40), "m", Curve(((512, 0.1), (2048, 0.3))), Curve(((512, 0.05), (2048, 0.1))))
        self.assertEqual(choose(1500, busy, [ServiceCandidate("s", 0.0, fast)], CostPolicy()).place, "offload")

    def test_cache_hit_beats_the_service_on_the_full_prompt(self):
        warm = [LocalCandidate("r", 0.0, 0, 8192, NODE)]
        self.assertEqual(choose(8300, warm, [ServiceCandidate("s", 0.0, GPU)], CostPolicy()).place, "local")

    def test_queue_and_margin(self):
        idle = [LocalCandidate("r", 0.0, 0, 0, NODE)]
        queued = choose(8192, idle, [ServiceCandidate("s", 20.0, GPU)], CostPolicy())
        self.assertEqual(queued.place, "local")                                            # a long queue loses
        near = choose(8192, idle, [ServiceCandidate("s", 0.0, GPU)], CostPolicy(margin_s=10.0))
        self.assertEqual(near.place, "local")

    def test_best_replica_by_cost(self):
        cands = [LocalCandidate("a", 0.0, 3, 0, NODE), LocalCandidate("b", 0.0, 0, 0, NODE), LocalCandidate("c", 0.0, 0, 4096, NODE)]
        self.assertEqual(choose(4200, cands, [], CostPolicy()).replica, "c")
        self.assertEqual(choose(2000, cands[:2], [], CostPolicy()).replica, "b")

    def test_seed_profiles_are_keyed(self):
        self.assertEqual(sorted(p.gpu.vram_gib for p in SEED_SERVICE_PROFILES), [24.0, 32.0, 48.0, 74.0])
        self.assertTrue(all(p.provenance == "seed" and p.model == "deepseek41" for p in SEED_SERVICE_PROFILES))


if __name__ == "__main__":
    unittest.main()
