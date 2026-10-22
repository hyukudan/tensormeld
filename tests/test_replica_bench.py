from __future__ import annotations

import unittest

from tensormeld.replica_bench import summarize


def rec(user, turn, sent, chunks, tokens):
    return {"user": user, "turn": turn, "status": 200, "sent_s": sent, "chunks": chunks, "completion_tokens": tokens,
            "prompt_tokens": 100, "cached_tokens": 0, "queue_ms": 0, "prefill_ms": 500, "decode_ms": 1000,
            "mtp_accepted": 3, "mtp_proposed": 4, "replica": "r0", "prefill_place": "local"}


class ClientMetricTests(unittest.TestCase):
    def test_streaming_metrics(self):
        # Two users, one turn: user 0 streams 0.5..1.5 s, user 1 1.0..2.0 s; 4 chunks of equal text each.
        a = rec(0, 0, 0.0, [[0.5, 10], [0.833, 10], [1.167, 10], [1.5, 10]], 8)
        b = rec(1, 0, 0.2, [[1.0, 10], [1.333, 10], [1.667, 10], [2.0, 10]], 8)
        out = summarize([a, b], 2.0, label="t")
        self.assertAlmostEqual(out["client_ttft_s"]["mean"], (0.5 + 0.8) / 2, places=3)
        self.assertAlmostEqual(out["client_decode_tok_s_per_request"]["p50"], 7.0, places=1)   # 7 tokens in 1 s
        self.assertAlmostEqual(out["client_inter_token_s"]["p50"], 0.333 / 2, places=2)        # 2 tokens per chunk
        self.assertAlmostEqual(out["common_window_s"], 0.5, places=3)                           # 1.0 .. 1.5 s
        self.assertEqual(out["mtp_acceptance"], 0.75)
        self.assertEqual(out["prefill_places"], {"local": 2})

    def test_non_streaming_records_have_no_client_metrics(self):
        r = rec(0, 0, 0.0, [], 8)
        out = summarize([r], 1.0)
        self.assertNotIn("client_ttft_s", out)


if __name__ == "__main__":
    unittest.main()
