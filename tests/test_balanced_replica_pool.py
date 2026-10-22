from concurrent.futures import ThreadPoolExecutor
import unittest

from test_replica_router import router_data
from tensormeld.replica_router import ReplicaPool, RouterConfig
from tensormeld.schema import ValidationError


class BalancedPoolTests(unittest.TestCase):
    def pool(self):
        data = router_data([18081, 18082], balance_active_decodes=True)
        # Unequal capacities must not turn four active decodes into 3+1.
        for r, cap in zip(data["replicas"], (8, 16)):
            r["max_inflight"] = cap
        pool = ReplicaPool(RouterConfig.parse(data))
        for spec in pool.specs():
            pool.report_health(spec.id, True, elapsed_ms=0)
        return pool

    def test_simultaneous_cold_arrivals_ignore_stale_scores(self):
        for n in (4, 8, 16):
            pool = self.pool()
            with ThreadPoolExecutor(max_workers=n) as executor:
                assigned = list(executor.map(lambda _: pool.acquire(scores={"r0": 0, "r1": 100}).id, range(n)))
            self.assertEqual(assigned.count("r0"), n // 2)
            self.assertEqual(assigned.count("r1"), n // 2)

    def test_hint_alone_cannot_buy_affinity_and_down_node_is_excluded(self):
        pool = self.pool()
        ids = [pool.acquire(prefer=["r0"]).id for _ in range(4)]
        self.assertEqual(ids.count("r0"), 2)
        self.assertEqual(ids.count("r1"), 2)
        pool.mark_down("r1", "test")
        self.assertEqual(pool.acquire().id, "r0")

    def test_flag_and_cache_defaults_are_explicit(self):
        cfg = RouterConfig.parse(router_data([18081], balance_active_decodes=True))
        self.assertEqual(cfg.prefill.cache_block_tokens, 1024)
        self.assertFalse(RouterConfig.parse(router_data([18081])).balance_active_decodes)
        for bad in (1, "true", None):
            with self.assertRaises(ValidationError):
                RouterConfig.parse(router_data([18081], balance_active_decodes=bad))
