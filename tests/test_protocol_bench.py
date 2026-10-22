"""Synthetic loopback transport evidence; no tokenizer, model or GPU qualification."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tensormeld.cli import main
from tensormeld.protocol_bench import CATEGORIES, load_corpus, run_protocol
from tensormeld.schema import ValidationError


def corpus(long=False):
    categories = ("synthetic",) if long else CATEGORIES
    return {"schema": "tensormeld/protocol-corpus-v1", "cases": [
        {"id": str(i), "category": category,
         "messages": [{"role": "user", "content": "fixture-a fixture-b fixture-c"}],
         "prompt_tokens": [36, 303], "completion_tokens": [82, 200],
         **({"needles": ["fixture-a", "fixture-b", "fixture-c"]} if long else {})}
        for i, category in enumerate(categories)]}


@contextmanager
def endpoint(*, missing_usage=False, thinking=False, truncated=False, cached=False, bad_length=False, finish_delay=0):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def emit(value):
                self.wfile.write(b"data: " + json.dumps(value).encode() + b"\n\n")
                self.wfile.flush()

            emit({"choices": [{"delta": {"role": "assistant"}}]})
            time.sleep(.001)
            emit({"choices": [{"delta": {"content": "fixture-a fixture-b "}}]})
            time.sleep(.001)
            emit({"choices": [{"delta": {"reasoning_content" if thinking else "content": "fixture-c"}}]})
            emit({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            if not missing_usage:
                emit({"choices": [], "usage": {"prompt_tokens": 100,
                      "completion_tokens": 20 if bad_length else 90,
                      "prompt_tokens_details": {"cached_tokens": 4 if cached else 0}}})
            if not truncated:
                time.sleep(finish_delay)
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ProtocolBenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "corpus.json"

    def write(self, doc=None):
        self.path.write_text(json.dumps(doc or corpus()), encoding="utf-8")
        return self.path

    def test_dry_run_never_connects_or_emits_rates_or_text(self):
        with patch("http.client.HTTPConnection", side_effect=AssertionError("network")):
            result = run_protocol(self.write())
        self.assertEqual(result["planned_requests"], 88)
        self.assertFalse(result["token_bounds_checked"])
        self.assertFalse(result["qualification"])
        self.assertNotIn("summaries", result)
        self.assertNotIn("fixture-a", json.dumps(result))
        hashes = [r["request_sha256"] for w in result["waves"] for r in w["requests"]]
        self.assertEqual(len(set(hashes)), 88)

    def test_stream_usage_packed_deltas_and_wall_aggregation(self):
        with endpoint() as (url, requests):
            result = run_protocol(self.write(), url=url, execute=True, api_key="private-test-key")
        self.assertTrue(result["valid"])
        self.assertEqual(len(requests), 88)
        for body in requests:
            self.assertEqual(body["temperature"], 0)
            self.assertFalse(body["enable_thinking"])
        self.assertEqual(len({b["messages"][0]["content"] for b in requests}), 88)
        self.assertNotIn("private-test-key", json.dumps(result))
        for wave in result["waves"]:
            self.assertEqual(wave["completion_tokens"], 90 * wave["concurrency"])
            self.assertAlmostEqual(wave["aggregate_tok_s"], wave["completion_tokens"] / wave["wall_s"])
            for rec in wave["requests"]:
                self.assertEqual(rec["sse_deltas"], 2)
                self.assertAlmostEqual(rec["per_stream_tok_s"], 89 / (rec["total_s"] - rec["ttft_s"]))
        for summary in result["summaries"]:
            self.assertAlmostEqual(summary["aggregate_tok_s"], summary["completion_tokens"] / summary["wall_s"])

    def test_noncompliant_streams_never_produce_comparable_rates(self):
        self.write(corpus(long=True))
        for options in ({"missing_usage": True}, {"thinking": True}, {"truncated": True},
                        {"cached": True}, {"bad_length": True}):
            with self.subTest(options=options), endpoint(**options) as (url, _):
                result = run_protocol(self.path, mode="long", levels=(1,), url=url, execute=True)
                self.assertFalse(result["valid"])
                self.assertIsNone(result["summaries"][0]["aggregate_tok_s"])

    def test_three_needles_recalled_in_order(self):
        with endpoint() as (url, _):
            result = run_protocol(self.write(corpus(long=True)), mode="long", levels=(1,), url=url, execute=True)
        rec = result["waves"][0]["requests"][0]
        self.assertTrue(result["valid"])
        self.assertEqual(rec["needle_hits"], 3)
        self.assertTrue(rec["needles_in_order"])

    def test_silent_stream_hits_wall_deadline(self):
        with endpoint(finish_delay=.2) as (url, _):
            result = run_protocol(self.write(corpus(long=True)), mode="long", levels=(1,),
                                  url=url, execute=True, timeout_s=.03)
        self.assertFalse(result["valid"])
        self.assertTrue(result["waves"][0]["requests"][0]["error"])

    def test_exhausted_campaign_is_incomplete_without_network(self):
        with patch("http.client.HTTPConnection", side_effect=AssertionError("network")):
            result = run_protocol(self.write(), url="http://127.0.0.1:1", execute=True, budget_s=1e-12)
        self.assertFalse(result["valid"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["waves"], [])

    def test_schema_and_resource_bounds_fail_before_network(self):
        self.write()
        for kwargs in ({"levels": (7,)}, {"levels": (1, 1)}, {"repeats": 0},
                       {"timeout_s": float("nan")}, {"execute": True}, {"run_id": "bad\n"},
                       {"url": "http://secret:key@127.0.0.1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                run_protocol(self.path, **kwargs)
        doc = corpus()
        doc["cases"].pop()
        with self.assertRaises(ValidationError):
            load_corpus(self.write(doc), "short")

    def test_cli_preserves_inputs_and_dry_run_does_not_read_key(self):
        self.write()
        out = Path(self.tmp.name) / "result.json"
        self.assertEqual(main(["bench-protocol", "--corpus", str(self.path),
                               "--api-key-file", str(out.parent / "absent.key"), "--out", str(out)]), 0)
        self.assertTrue(json.loads(out.read_text())["dry_run"])
        original = self.path.read_bytes()
        self.assertEqual(main(["bench-protocol", "--corpus", str(self.path), "--out", str(self.path)]), 1)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
