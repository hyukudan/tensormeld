from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest

from tensormeld.route_cost import Curve, GpuKey, LocalCandidate, NodeProfile, ServiceCandidate, ServiceProfile
from tensormeld.schema import ValidationError
from tensormeld.serving_policy import (
    JointReservation, ModeController, ModePolicy, Rate, RecordedProfiles, choose_joint, recommend,
)


def profiles():
    # Deliberately synthetic; private recordings are exercised outside Git.
    rates = []
    for n in (1, 2, 4, 8, 16):
        for domain in ("en", "es", "code"):
            for rows in (8, 16):
                for mode, rate in (("pair_ar", 10), ("pair_dspark_batch", 14 if domain == "code" else 10.5),
                                   ("pair_dspark", 15), ("split_dspark", 20),
                                   ("split_adaptive_dspark", 18), ("pair_batch", 8)):
                    rates.append(Rate(mode, n, domain, rows, rate, "synthetic", "unit fixture"))
    # Stay within the bounded profile table.
    return RecordedProfiles(tuple(rates))


class ModeTests(unittest.TestCase):
    def setUp(self):
        self.profiles = profiles()
        self.policy = ModePolicy(enabled=True, split_enabled=True, batch_dspark_enabled=True)

    def test_modes_and_selectivity(self):
        expected = {1: ("split_adaptive_dspark", "split_dspark"),
                    2: ("pair_dspark_batch", "pair_dspark_batch"),
                    4: ("pair_ar", "pair_dspark_batch"), 8: ("pair_batch", "pair_batch"),
                    16: ("pair_batch", "pair_batch")}
        for n, modes in expected.items():
            for domain, mode in zip(("en", "code"), modes):
                with self.subTest(n=n, domain=domain):
                    d = recommend(n, domain, self.profiles, self.policy)
                    self.assertEqual(d.mode, mode)
                    self.assertEqual(d.accelerator_role, "front" if n == 1 else "prefill")
                    self.assertFalse(d.executable)
        p = replace(self.policy, selective_gain=0.01)
        self.assertEqual(recommend(4, "es", self.profiles, p).mode, "pair_dspark_batch")

    def test_default_off_and_fallbacks(self):
        self.assertEqual(recommend(1, "code", self.profiles, ModePolicy()).mode, "pair_ar")
        p = ModePolicy(enabled=True)
        self.assertEqual(recommend(1, "code", self.profiles, p).mode, "pair_dspark")
        self.assertEqual(recommend(4, "code", self.profiles, p).mode, "pair_ar")
        self.assertEqual(recommend(16, "en", self.profiles, self.policy).rows, 8)
        self.assertEqual(recommend(16, "en", self.profiles, replace(self.policy, rows16_enabled=True)).rows, 16)

    def test_missing_profile_is_unknown_and_load_is_bounded(self):
        self.assertIsNone(recommend(3, "en", self.profiles, self.policy).rate)
        for n in (0, 17, True):
            with self.assertRaises(ValidationError):
                recommend(n, "en", self.profiles, self.policy)

    def test_safe_point_hysteresis_and_dwell(self):
        c = ModeController(self.profiles, replace(self.policy, min_dwell_s=5))
        self.assertIsNone(c.observe(1, "code", safe_point=None, now_s=0))
        self.assertTrue(c.transition_pending)
        self.assertEqual(c.observe(1, "code", safe_point="between_requests", now_s=0).mode, "split_dspark")
        for t in (1, 2, 3):
            self.assertEqual(c.observe(2, "code", safe_point=None, now_s=t).mode, "split_dspark")
        self.assertEqual(c.observe(2, "code", safe_point="window_boundary", now_s=4).mode, "split_dspark")
        self.assertEqual(c.observe(2, "code", safe_point="window_boundary", now_s=5).mode, "pair_dspark_batch")
        self.assertFalse(c.transition_pending)
        c.observe(1, "code", safe_point="window_boundary", now_s=6)
        c.observe(2, "code", safe_point="window_boundary", now_s=7)
        self.assertEqual(c.observe(1, "code", safe_point="window_boundary", now_s=11).mode, "pair_dspark_batch")
        self.assertEqual(c.observe(1, "code", safe_point="window_boundary", now_s=12).mode, "split_dspark")
        with self.assertRaises(ValidationError):
            c.observe(1, "code", safe_point="window_boundary", now_s=11)

    def test_profile_rejects_unqualified_promotion_and_invalid_numbers(self):
        raw = {"schema": "tensormeld/serving-rates-v1", "rates": [dict(
            mode="pair_ar", load=1, domain="en", rows=8, per_user_tps=1, provenance="synthetic", source="test")]}
        self.assertEqual(RecordedProfiles.parse(raw).lookup("pair_ar", 1, "en", 8).per_user_tps, 1)
        for key, value in (("qualified", True), ("per_user_tps", float("nan")), ("rows", True),
                           ("mode", []), ("provenance", "qualified")):
            bad = {**raw, "rates": [{**raw["rates"][0], key: value}]}
            with self.subTest(key=key), self.assertRaises(ValidationError):
                RecordedProfiles.parse(bad)


class JointCostTests(unittest.TestCase):
    def setUp(self):
        self.local = LocalCandidate("r", 0, 1, 0, NodeProfile(Curve(((1000, 10.0),))))
        self.service = ServiceCandidate("s", 0, ServiceProfile(
            GpuKey("fixture", 1, 0, 1), "fixture", Curve(((1000, 1.0),)), Curve(((1000, .1),))))

    def decide(self, **kw):
        args = dict(load=1, remaining_tokens=1, pair_tps=10, split_tps=20)
        args.update(kw)
        return choose_joint(1000, self.local, self.service, **args)

    def test_ttft_vs_decode_gain(self):
        self.assertEqual(self.decide().owner, "prefill")
        self.assertEqual(self.decide(remaining_tokens=1000).owner, "front")
        self.assertEqual(self.decide(switch_s=30).owner, "front")
        for n in (2, 4, 8, 16):
            self.assertEqual(self.decide(load=n, remaining_tokens=1000).owner, "prefill")

    def test_joint_cost_changes_single_user_mode(self):
        d = recommend(1, "code", profiles(), ModePolicy(enabled=True, split_enabled=True), self.decide())
        self.assertEqual((d.mode, d.accelerator_role), ("pair_dspark", "prefill"))


class ReservationTests(unittest.TestCase):
    def test_exclusion_safe_release_stale_and_capacity(self):
        r = JointReservation(100, enabled=True)
        self.assertIsNone(r.acquire("front", 80, safe_point=None))
        front = r.acquire("front", 80, safe_point="between_requests")
        self.assertIsNone(r.acquire("prefill", 80, safe_point="window_boundary"))
        self.assertFalse(r.release(front, buffers_released=False, safe_point="window_boundary"))
        with r.use("front") as allowed:
            self.assertTrue(allowed)
            self.assertFalse(r.release(front, buffers_released=True, safe_point="window_boundary"))
        with r.use("prefill") as allowed:
            self.assertFalse(allowed)
        self.assertTrue(r.release(front, buffers_released=True, safe_point="window_boundary"))
        self.assertIsNone(r.acquire("prefill", 101, safe_point="between_requests"))
        prefill = r.acquire("prefill", 80, safe_point="between_requests")
        self.assertFalse(r.release(front, buffers_released=True, safe_point="between_requests"))
        self.assertIs(r.current, prefill)

    def test_atomic_race_and_cancelled_task_does_not_release_residency(self):
        r = JointReservation(100, enabled=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            leases = list(pool.map(lambda i: r.acquire("front" if i % 2 else "prefill", 80,
                                                     safe_point="between_requests"), range(32)))
        self.assertEqual(sum(x is not None for x in leases), 1)
        lease = r.current
        with self.assertRaises(RuntimeError):
            with r.use(lease.owner):
                raise RuntimeError("cancel")
        self.assertIs(r.current, lease)
        self.assertTrue(r.release(lease, buffers_released=True, safe_point="between_requests"))
        self.assertIsNone(JointReservation(100).acquire("front", 80, safe_point="between_requests"))


if __name__ == "__main__":
    unittest.main()
