"""prefill_offload: router-side RTX prefill for the replica the router holds (fake RTX and replicas)."""
from __future__ import annotations

import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest

from tensormeld.prefill_offload import decide
from tensormeld.prefill_route import load_ttft_curves, measured_crossover, PrefillRoute, route_section
from tensormeld.replica_router import ROUTER_CONFIG_SCHEMA, ReplicaRouter, RouterConfig, load_router_config
from tensormeld.schema import ValidationError

RECEIPT = Path(__file__).resolve().parents[1] / "examples" / "hetero" / "qwen38-ttft-receipt.json"


class _JsonHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)) or b"null")


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class FakeQwenReplica:
    """Qwen serve double: /health, /tokenize (one id per character; chat = joined contents),
    /admin/rtx-import(/layout) when `import_flag`, and the completion routes. Every call is logged
    in order in `events`."""

    def __init__(self, name, *, import_flag=True):
        self.name = name
        self.import_flag = import_flag
        self.events: list[tuple[str, object]] = []
        self.raw_bodies: list[bytes] = []
        rep = self

        class Handler(_JsonHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._json(200, {"status": "ok"})
                elif self.path == "/admin/rtx-import/layout" and rep.import_flag:
                    rep.events.append(("layout", None))
                    self._json(200, {"layers": 48})
                else:
                    self._json(404, {})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.loads(raw)
                if self.path == "/tokenize":
                    rep.events.append(("tokenize", body))
                    text = body.get("prompt") or "".join(m.get("content", "") for m in body.get("messages", []))
                    self._json(200, {"tokens": [ord(c) % 1000 for c in text], "count": len(text)})
                elif self.path == "/admin/rtx-import" and rep.import_flag:
                    rep.events.append(("import", body))
                    self._json(200, {"parked": True, "position": body["tokens"] - 1})
                elif self.path in ("/v1/completions", "/v1/chat/completions"):
                    rep.events.append((self.path, body))
                    rep.raw_bodies.append(raw)
                    self._json(200, {"replica": rep.name})
                else:
                    self._json(404, {})

        self.server = _serve(Handler)
        self.port = self.server.server_address[1]

    def kinds(self):
        return [k for k, _ in self.events]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class FakeRtx:
    """RTX prefill service double. On 200 it pushes {session, tokens} to the handoff URL first."""

    def __init__(self):
        self.health = {"status": "ok", "busy": False, "queued": 0, "model": "qwen3.8-flash-next", "max_positions": 65536}
        self.status = 200
        self.delay_s = 0.0
        self.position_offset = -1
        self.calls: list[dict] = []
        self.health_calls = 0
        self.router_port: int | None = None
        self.inflight_seen: list[dict] = []
        rtx = self

        class Handler(_JsonHandler):
            def do_GET(self):
                rtx.health_calls += 1
                self._json(200 if rtx.health.get("status") == "ok" else 503, rtx.health)

            def do_POST(self):
                body = self._body()
                rtx.calls.append({"body": body, "auth": self.headers.get("Authorization")})
                if rtx.router_port is not None:
                    st = json.loads(_call(rtx.router_port, "GET", "/api/status")[2])
                    rtx.inflight_seen.append({r["id"]: r["inflight"] for r in st["replicas"]})
                time.sleep(rtx.delay_s)
                if rtx.status != 200:
                    self._json(rtx.status, {"error": "fake"})
                    return
                url = body["handoff"][len("http://"):]
                hostport, path = url.split("/", 1)
                host, port = hostport.split(":")
                conn = http.client.HTTPConnection(host, int(port), timeout=5)
                n = len(body["tokens"])
                state = json.dumps({"session": body["session"], "tokens": n}).encode()
                conn.request("POST", "/" + path, body=state, headers={"Content-Type": "application/json"})
                reply = json.loads(conn.getresponse().read())
                conn.close()
                self._json(200, {"session": body["session"], "position": n + rtx.position_offset, "tokens": n,
                                 "sha256": "00", "seconds": {"queue": 0.0, "prefill": 0.01, "transfer": 0.0,
                                                             "import": 0.0, "total": 0.01},
                                 "import": reply})

        self.server = _serve(Handler)
        self.port = self.server.server_address[1]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def _call(port, method, path, body=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body).encode()
        conn.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def config_data(ports, offload=None, **extra):
    data = {
        "schema": ROUTER_CONFIG_SCHEMA,
        "listen": {"address": "127.0.0.1", "port": 0},
        "replicas": [
            {"id": f"r{i}", "node": f"node{i}", "address": "127.0.0.1", "port": p, "max_inflight": 1}
            for i, p in enumerate(ports)
        ],
        "health": {"interval_s": 0.05, "timeout_s": 1, "failures_to_down": 1},
        "limits": {"queue_timeout_s": 10, "read_timeout_s": 30},
    }
    if offload is not None:
        data["prefill_offload"] = offload
    data.update(extra)
    return data


def offload_section(rtx_port, **kw):
    return {"endpoint": f"http://127.0.0.1:{rtx_port}", "route": str(RECEIPT), "timeout_s": 5,
            "min_tokens": 1024, **kw}


class ReceiptTests(unittest.TestCase):
    def test_qwen_receipt_parses_and_crosses_between_1k_and_4k(self):
        curves = load_ttft_curves(RECEIPT)
        self.assertEqual([p.tokens for p in curves.local], [1024, 4096, 8192, 16384, 32768])
        self.assertEqual([p.tokens for p in curves.offload], [1024, 4096, 8192, 16384, 32768])
        self.assertTrue(all(p.provenance == "measured" for p in curves.local + curves.offload))
        self.assertEqual(curves.provenance, "measured")
        self.assertTrue(256 < measured_crossover(curves) < 1024)  # offload already wins at 1K (extrapolated)


class DecideTests(unittest.TestCase):
    def setUp(self):
        curves = load_ttft_curves(RECEIPT)
        self.route = PrefillRoute.from_section(route_section("strix-node", "rtx-prefill", None, curves))
        self.ok = {"status": "ok", "busy": False, "queued": 0, "max_positions": 65536}

    def test_threshold_and_min_tokens(self):
        cross = self.route.local_max_tokens
        self.assertEqual(decide(self.route, 500, 1024, self.ok)["reason"], "below_min_tokens")
        self.assertEqual(decide(self.route, cross, 0, self.ok)["reason"], "below_threshold")
        self.assertEqual(decide(self.route, cross + 1, 0, self.ok)["route"], "offload")
        self.assertEqual(decide(self.route, 8000, 10000, self.ok)["reason"], "below_min_tokens")

    def test_service_state(self):
        self.assertEqual(decide(self.route, 8000, 0, None)["reason"], "service_unreachable")
        self.assertEqual(decide(self.route, 8000, 0, {"status": "failed"})["reason"], "service_failed")
        self.assertEqual(decide(self.route, 70000, 0, self.ok)["reason"], "service_max_positions")

    def test_queue_is_priced_by_the_offload_curve(self):
        busy = {**self.ok, "busy": True}
        short = decide(self.route, 1024, 0, busy)  # 2 x 1.36 s > 1.80 s
        self.assertEqual((short["route"], short["reason"]), ("local", "service_queue"))
        self.assertAlmostEqual(short["q_rtx_s"], short["offload_s"], places=3)
        long = decide(self.route, 32768, 0, {**busy, "queued": 1})
        self.assertEqual((long["route"], long["ahead"]), ("offload", 2))  # 3 x 8.99 s < 44.02 s
        self.assertEqual(decide(self.route, 32768, 0, {**busy, "queued": 3})["route"], "local")  # 5 x 8.99 s > 44.02 s

    def test_busy_without_curves_stays_local(self):
        route = PrefillRoute(2000, "configured")
        self.assertEqual(decide(route, 8000, 0, {"status": "ok", "busy": True})["reason"], "service_busy")
        self.assertEqual(decide(route, 8000, 0, {"status": "ok"})["route"], "offload")


class ConfigTests(unittest.TestCase):
    def test_absent_is_off_and_existing_configs_parse_unchanged(self):
        base = config_data([18181, 18182])
        self.assertIsNone(RouterConfig.parse(base).prefill_offload)
        self.assertEqual(RouterConfig.parse(base), RouterConfig.parse(json.loads(json.dumps(base))))

    def test_parse_receipt_override_and_import_urls(self):
        data = config_data([18181, 18182], offload_section(18190, local_max_tokens=3000))
        data["replicas"][1]["import_url"] = "http://10.0.0.7:9000/x/rtx-import"
        cfg = RouterConfig.parse(data).prefill_offload
        self.assertEqual((cfg.route.local_max_tokens, cfg.route.source), (3000, "configured"))
        self.assertIsNotNone(cfg.route.curves)
        self.assertEqual(cfg.import_urls["r0"].url(), "http://127.0.0.1:18181/admin/rtx-import")
        self.assertEqual(cfg.import_urls["r1"].url(), "http://10.0.0.7:9000/x/rtx-import")
        measured = RouterConfig.parse(config_data([18181], offload_section(18190))).prefill_offload.route
        self.assertEqual(measured.source, "measured")
        self.assertEqual(measured.local_max_tokens, measured_crossover(load_ttft_curves(RECEIPT)))

    def test_route_section_and_relative_receipt_path(self):
        section = {"schema": "tensormeld/prefill-route-v1", "local_max_tokens": 4000, "source": "model"}
        cfg = RouterConfig.parse(config_data([18181], offload_section(18190, route=section))).prefill_offload
        self.assertEqual((cfg.route.local_max_tokens, cfg.route.curves), (4000, None))
        with tempfile.TemporaryDirectory() as d:
            Path(d, "receipt.json").write_text(RECEIPT.read_text(encoding="utf-8"), encoding="utf-8")
            Path(d, "router.json").write_text(json.dumps(config_data([18181], offload_section(18190, route="receipt.json"))))
            self.assertEqual(load_router_config(Path(d, "router.json")).prefill_offload.route.source, "measured")

    def test_example_config_loads(self):
        cfg = load_router_config(RECEIPT.parents[1] / "qwen-replicas-router-rtx-prefill.json").prefill_offload
        self.assertEqual(cfg.service.url(), "http://192.0.2.20:18700")
        self.assertEqual(cfg.import_urls["node2"].url(), "http://192.0.2.12:18501/admin/rtx-import")
        self.assertEqual(cfg.route.source, "measured")

    def test_rejections(self):
        bad = [
            offload_section(18190, endpoint="http://rtx.local:8090"),
            offload_section(18190, endpoint="http://8.8.8.8:8090"),
            offload_section(18190, endpoint="https://127.0.0.1:8090"),
            offload_section(18190, endpoint="http://127.0.0.1:8090/v1"),
            offload_section(18190, unknown=1),
            offload_section(18190, route={"schema": "nope"}),
            offload_section(18190, timeout_s=0),
            offload_section(18190, handoff_codec="zstd"),
            offload_section(18190, cooperative_local_rows=[[4096, 4095]]),
            offload_section(18190, cooperative_local_rows=[[4096]]),
            offload_section(18190, cooperative_local_rows={"4096": 2048}),
            {"route": str(RECEIPT)},
        ]
        for section in bad:
            with self.assertRaises(ValidationError, msg=section):
                RouterConfig.parse(config_data([18181], section))
        orphan = config_data([18181])
        orphan["replicas"][0]["import_url"] = "http://127.0.0.1:18181/admin/rtx-import"
        with self.assertRaises(ValidationError):
            RouterConfig.parse(orphan)
        with self.assertRaises(ValidationError):
            RouterConfig.parse(config_data([18181], offload_section(18190), prefill={"policy": "auto"}))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.reps = [FakeQwenReplica("e0"), FakeQwenReplica("e1")]
        self.rtx = FakeRtx()
        for s in self.reps + [self.rtx]:
            self.addCleanup(s.stop)

    def start(self, offload=True, **kw):
        section = offload_section(self.rtx.port, **kw) if offload else None
        router = ReplicaRouter(RouterConfig.parse(config_data([r.port for r in self.reps], section)))
        self.records: list[dict] = []
        router.server.offload_log = self.records.append
        router.start()
        self.addCleanup(router.stop)
        self.assertTrue(router.wait_for_replicas(2, 5))
        self.rtx.router_port = router.address[1]
        return router, router.address[1]

    def complete(self, port, n, **extra):
        return _call(port, "POST", "/v1/completions", {"prompt": [7] * n, "max_tokens": 64, **extra})

    def test_off_forwards_unchanged(self):
        _, port = self.start(offload=False)
        body = {"prompt": [7] * 30000, "max_tokens": 64}
        status, headers, _ = _call(port, "POST", "/v1/completions", body)
        self.assertEqual(status, 200)
        self.assertNotIn("X-TensorMeld-Prefill", headers)
        self.assertEqual(self.reps[0].kinds(), ["/v1/completions"])
        self.assertEqual(self.reps[0].raw_bodies[0], json.dumps(body).encode())
        self.assertEqual((self.rtx.calls, self.rtx.health_calls), ([], 0))
        status, _, raw = _call(port, "GET", "/api/status")
        self.assertNotIn("prefill_offload", json.loads(raw))
        self.assertNotIn(b"prefill_offload", _call(port, "GET", "/metrics")[2])

    def test_short_prompt_stays_local_without_asking_the_service(self):
        _, port = self.start(local_max_tokens=1500)
        status, headers, _ = self.complete(port, 1300)  # below the configured threshold
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "local:below_threshold"))
        status, headers, _ = self.complete(port, 600)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "local:below_min_tokens")
        self.assertEqual((self.rtx.calls, self.rtx.health_calls), ([], 0))
        self.assertEqual([r.kinds() for r in self.reps], [["/v1/completions"], ["/v1/completions"]])

    def test_long_prompt_prefills_on_rtx_for_the_held_replica(self):
        router, port = self.start()
        for i in range(2):  # ties alternate: r0 then r1, each gets its own handoff
            status, headers, raw = self.complete(port, 8000)
            self.assertEqual(status, 200)
            self.assertEqual(headers["X-TensorMeld-Prefill"], "offload:rtx")
            self.assertEqual(headers["X-TensorMeld-Replica"], f"r{i}")
            call = self.rtx.calls[i]["body"]
            self.assertEqual(call["handoff"], f"http://127.0.0.1:{self.reps[i].port}/admin/rtx-import")
            self.assertEqual((len(call["tokens"]), call["reserve_tokens"]), (8000, 64))
            self.assertTrue(call["session"].startswith(f"r{i}-"))
            # The replica was held during the prefill, and got import before its completion.
            self.assertEqual(self.rtx.inflight_seen[i][f"r{i}"], 1)
            self.assertEqual(self.reps[i].kinds(), ["layout", "import", "/v1/completions"])
            self.assertEqual(self.reps[i].events[2][1]["prompt"], [7] * 8000)  # unchanged request
        rec = self.records[0]
        self.assertEqual((rec["route"], rec["tokens"], rec["replica"]), ("offload", 8000, "r0"))
        self.assertIn("prefill", rec["rtx_seconds"])
        status = json.loads(_call(port, "GET", "/api/status")[2])["prefill_offload"]
        self.assertEqual(status["counts"], {"local": 0, "offload": 2, "fallback": 0})
        metrics = _call(port, "GET", "/metrics")[2].decode()
        self.assertIn('tensormeld_router_prefill_offload_total{route="offload"} 2', metrics)

    def test_codec_and_cooperative_rows_reach_the_service(self):
        _, port = self.start(handoff_codec="bp-v1", cooperative_local_rows=[[16384, 4096], [4096, 2048]])
        for n, q in ((3000, None), (8000, 2048), (20000, 4096)):
            self.assertEqual(self.complete(port, n)[0], 200)
            call = self.rtx.calls[-1]["body"]
            self.assertEqual(call["handoff_codec"], "bp-v1")
            self.assertEqual(call.get("local_rows"), q)
            if q:
                self.assertTrue(call["node_completions"].endswith("/v1/completions"))
            self.assertEqual(self.records[-1].get("local_rows"), q)

    def test_chat_is_tokenized_on_the_held_replica_with_the_same_body(self):
        _, port = self.start()
        body = {"messages": [{"role": "user", "content": "x" * 5000}], "max_completion_tokens": 32}
        status, headers, _ = _call(port, "POST", "/v1/chat/completions", body)
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "offload:rtx"))
        self.assertEqual(self.reps[0].kinds(), ["tokenize", "layout", "import", "/v1/chat/completions"])
        self.assertEqual(self.reps[0].events[0][1], body)
        self.assertEqual(self.rtx.calls[0]["body"]["reserve_tokens"], 32)
        # Text completions: tokenized as {"prompt": text}; a one-string list is unwrapped.
        _call(port, "POST", "/v1/completions", {"prompt": ["y" * 3000]})
        self.assertEqual(self.reps[1].events[0], ("tokenize", {"prompt": "y" * 3000}))
        self.assertEqual(self.rtx.calls[1]["body"]["reserve_tokens"], 256)  # router default

    def test_rtx_502_falls_back_to_the_same_replica(self):
        self.rtx.status = 502
        _, port = self.start()
        status, headers, _ = self.complete(port, 8000)
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "fallback:http502"))
        self.assertEqual(self.reps[0].kinds(), ["layout", "/v1/completions"])
        self.assertEqual(self.records[0]["fallback"], "http502")

    def test_rtx_timeout_and_position_mismatch_fall_back(self):
        self.rtx.delay_s = 1.0
        _, port = self.start(timeout_s=0.2)
        _, headers, _ = self.complete(port, 8000)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "fallback:timeout")
        self.rtx.delay_s, self.rtx.position_offset = 0.0, 0
        _, headers, _ = self.complete(port, 8000)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "fallback:position_mismatch")

    def test_busy_rtx_stays_local_when_the_curves_say_so(self):
        self.rtx.health = {**self.rtx.health, "busy": True, "queued": 1}
        _, port = self.start()
        _, headers, _ = self.complete(port, 4000)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "local:service_queue")
        self.assertEqual(self.rtx.calls, [])
        _, headers, _ = self.complete(port, 32000)  # 3 x 9.6 s < 30 s local: still worth queueing
        self.assertEqual(headers["X-TensorMeld-Prefill"], "offload:rtx")

    def test_failed_or_unreachable_service_and_replica_without_import(self):
        self.rtx.health = {"status": "failed"}
        _, port = self.start()
        _, headers, _ = self.complete(port, 8000)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "local:service_failed")
        self.rtx.health = {"status": "ok", "busy": False, "queued": 0}
        self.reps[1].import_flag = False
        _, headers, _ = self.complete(port, 8000)
        self.assertEqual((headers["X-TensorMeld-Replica"], headers["X-TensorMeld-Prefill"]),
                         ("r1", "local:replica_no_import"))
        self.assertEqual(self.rtx.calls, [])
        self.rtx.stop()
        _, headers, _ = self.complete(port, 8000)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "local:service_unreachable")

    def test_messages_route_is_not_offloaded(self):
        _, port = self.start()
        _call(port, "POST", "/v1/messages", {"messages": [{"role": "user", "content": "z" * 5000}], "max_tokens": 8})
        self.assertEqual(self.rtx.calls, [])
        self.assertEqual(self.records, [])


if __name__ == "__main__":
    unittest.main()
