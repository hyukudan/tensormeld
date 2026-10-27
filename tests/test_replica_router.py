from __future__ import annotations

import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from tensormeld.replica_router import (
    ROUTER_CONFIG_SCHEMA,
    AffinityTable,
    ReplicaRouter,
    ReplicaPool,
    RouterConfig,
    affinity_keys,
    load_router_config,
)
from tensormeld.replica_bench import run_bench
from tensormeld.schema import ValidationError
from tensormeld.serving_policy import JointReservation


class FakeEngine:
    """Minimal OpenAI-compatible engine server: one fixed reply, optional SSE stream."""

    def __init__(self, name: str, *, delay_s: float = 0.0):
        self.name = name
        self.delay_s = delay_s
        self.events = 3
        self.healthy = True
        self.requests: list[dict] = []
        self.tokenized = 0
        self.imports: list[dict] = []
        self.active = 0
        self.max_active = 0
        self.stream_gate = threading.Event()
        self.stream_gate.set()
        self.client_gone = threading.Event()
        self.lock = threading.Lock()
        engine = self

        class Handler(BaseHTTPRequestHandler):
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

            def do_GET(self):
                if self.path == "/health":
                    self._json(200 if engine.healthy else 503, {"status": "ok" if engine.healthy else "failed"})
                elif self.path == "/v1/models":
                    self._json(200, {"data": [{"id": "m", "owned_by": engine.name}]})
                else:
                    self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    text = body.get("prompt") or "".join(m.get("content", "") for m in body.get("messages", []))
                    with engine.lock:
                        engine.tokenized += 1
                    self._json(200, {"tokens": [ord(c) % 1000 for c in text], "count": len(text)})
                    return
                if self.path == "/admin/import-state":
                    with engine.lock:
                        engine.imports.append(body)
                    self._json(200, {"imported": True})
                    return
                with engine.lock:
                    engine.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                    engine.active += 1
                    engine.max_active = max(engine.max_active, engine.active)
                try:
                    time.sleep(engine.delay_s)
                    if not body.get("stream"):
                        self._json(200, {"engine": engine.name, "echo": body.get("messages")})
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    try:
                        for i in range(engine.events):
                            event = f'data: {{"engine":"{engine.name}","i":{i}}}\n\n'.encode()
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(event), event))
                            self.wfile.flush()
                            if i == 0:
                                engine.stream_gate.wait(10)
                            time.sleep(0.05)
                        done = b"data: [DONE]\n\n"
                        self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(done), done))
                        self.wfile.flush()
                    except OSError:
                        engine.client_gone.set()
                        self.close_connection = True
                finally:
                    with engine.lock:
                        engine.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def router_data(ports, **extra):
    data = {
        "schema": ROUTER_CONFIG_SCHEMA,
        "listen": {"address": "127.0.0.1", "port": 0},
        "replicas": [
            {"id": f"r{i}", "node": f"node{i}", "address": "127.0.0.1", "port": port, "max_inflight": 1}
            for i, port in enumerate(ports)
        ],
        "health": {"interval_s": 0.05, "timeout_s": 1, "failures_to_down": 1},
        "limits": {"queue_timeout_s": 10, "read_timeout_s": 30},
    }
    data.update(extra)
    return data


def call(port, method, path, body=None, headers=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body).encode()
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def chat(port, text="hi", **kw):
    return call(port, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": text}]}, **kw)


class RouterCase(unittest.TestCase):
    def start(self, engines, **extra):
        router = ReplicaRouter(RouterConfig.parse(router_data([e.port for e in engines], **extra)))
        router.start()
        self.addCleanup(router.stop)
        self.assertTrue(router.wait_for_replicas(sum(e.healthy for e in engines), 5))
        return router, router.address[1]

    def engines(self, n, **kw):
        result = [FakeEngine(f"e{i}", **kw) for i in range(n)]
        for e in result:
            self.addCleanup(e.stop)
        return result


class ConfigTests(unittest.TestCase):
    def test_valid_config_round_trip(self):
        config = RouterConfig.parse(router_data([18081, 18082]))
        self.assertEqual([r.id for r in config.replicas], ["r0", "r1"])
        self.assertEqual(config.listen_address, "127.0.0.1")

    def test_rejects_hostnames_public_and_unknown_fields(self):
        for mutate in (
            lambda d: d["replicas"][0].update(address="localhost"),
            lambda d: d["replicas"][0].update(address="8.8.8.8"),
            lambda d: d["replicas"][0].update(address="169.254.1.1"),
            lambda d: d["replicas"][0].update(max_inflight=0),
            lambda d: d["replicas"][0].update(argv=["x"]),
            lambda d: d["replicas"][1].update(id="r0"),
            lambda d: d["replicas"][1].update(port=d["replicas"][0]["port"]),
            lambda d: d.update(schema="other"),
            lambda d: d["limits"].update(queue_timeout_s=-1),
        ):
            data = router_data([18081, 18082])
            mutate(data)
            with self.assertRaises(ValidationError):
                RouterConfig.parse(data)

    def test_private_listener_requires_api_key(self):
        data = router_data([18081])
        data["listen"]["address"] = "192.168.1.10"
        with self.assertRaises(ValidationError):
            RouterConfig.parse(data)
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key"
            key.write_text("secret-token\n", encoding="ascii")
            data["api_key_file"] = str(key)
            self.assertEqual(RouterConfig.parse(data).api_key, "secret-token")

    def test_load_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.json"
            path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_router_config(path)


class RoutingTests(RouterCase):
    def test_concurrent_requests_go_to_least_loaded_replica(self):
        engines = self.engines(2, delay_s=0.4)
        router, port = self.start(engines)
        results = []
        threads = [threading.Thread(target=lambda: results.append(chat(port))) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual([r[0] for r in results], [200] * 4)
        served = sorted(json.loads(r[2])["engine"] for r in results)
        self.assertEqual(served, ["e0", "e0", "e1", "e1"])
        self.assertEqual([e.max_active for e in engines], [1, 1])
        deadline = time.monotonic() + 5
        while sum(r["inflight"] for r in router.pool.snapshot()["replicas"]) and time.monotonic() < deadline:
            time.sleep(0.02)
        status = router.pool.snapshot()
        self.assertEqual([r["completed"] for r in status["replicas"]], [2, 2])
        self.assertEqual(status["waiting"], 0)

    def test_sequential_requests_alternate_on_ties(self):
        engines = self.engines(2)
        _, port = self.start(engines)
        served = [json.loads(chat(port)[2])["engine"] for _ in range(4)]
        self.assertEqual(served, ["e0", "e1", "e0", "e1"])

    def test_response_names_replica_and_body_is_unchanged(self):
        engines = self.engines(1)
        _, port = self.start(engines)
        status, headers, body = chat(port, "exact bytes")
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-TensorMeld-Replica"], "r0")
        self.assertEqual(json.loads(body)["echo"], [{"role": "user", "content": "exact bytes"}])
        self.assertEqual(engines[0].requests[0]["path"], "/v1/chat/completions")

    def test_utility_routes_hold_no_slot(self):
        engines = self.engines(1)
        _, port = self.start(engines)
        status, _, body = call(port, "GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["owned_by"], "e0")
        self.assertEqual(call(port, "GET", "/nope")[0], 404)
        self.assertEqual(call(port, "POST", "/admin/drain", {})[0], 404)

    def test_queue_full_and_queue_timeout_answer_503(self):
        engines = self.engines(1, delay_s=0.6)
        _, port = self.start(engines, limits={"queue_timeout_s": 0.1, "max_waiting": 0})
        first = []
        t = threading.Thread(target=lambda: first.append(chat(port)))
        t.start()
        time.sleep(0.2)
        status, _, body = chat(port)
        self.assertEqual(status, 503)
        self.assertIn("queue is full", json.loads(body)["error"]["message"])
        t.join()
        self.assertEqual(first[0][0], 200)


class StreamingTests(RouterCase):
    def test_stream_events_arrive_before_upstream_finishes(self):
        engines = self.engines(1)
        engines[0].stream_gate.clear()
        _, port = self.start(engines)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.addCleanup(conn.close)
        body = json.dumps({"stream": True, "messages": []}).encode()
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        first = resp.read1(4096)
        # The engine is still blocked after its first event: the router did not buffer.
        self.assertIn(b'"i":0', first)
        self.assertFalse(engines[0].stream_gate.is_set())
        engines[0].stream_gate.set()
        rest = resp.read()
        self.assertTrue(rest.endswith(b"data: [DONE]\n\n"))
        self.assertIn(b'"i":2', rest)

    def test_client_disconnect_closes_upstream(self):
        engines = self.engines(1)
        engines[0].stream_gate.clear()
        engines[0].events = 200
        router, port = self.start(engines)
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        body = json.dumps({"stream": True, "messages": []}).encode()
        sock.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
        )
        received = b""
        while b'"i":0' not in received:  # headers and the first event may come in one segment or several
            chunk = sock.recv(4096)
            self.assertTrue(chunk, "connection closed before the first event")
            received += chunk
        sock.close()
        time.sleep(0.2)
        engines[0].stream_gate.set()
        self.assertTrue(engines[0].client_gone.wait(5))
        deadline = time.monotonic() + 5
        while router.pool.snapshot()["replicas"][0]["inflight"] and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(router.pool.snapshot()["replicas"][0]["inflight"], 0)


class HealthTests(RouterCase):
    def test_unhealthy_replica_is_not_routed_until_it_recovers(self):
        engines = self.engines(2)
        engines[0].healthy = False
        router, port = self.start(engines)
        self.assertEqual({json.loads(chat(port)[2])["engine"] for _ in range(3)}, {"e1"})
        engines[0].healthy = True
        self.assertTrue(router.wait_for_replicas(2, 5))
        self.assertEqual([json.loads(chat(port)[2])["engine"] for _ in range(4)], ["e0", "e1", "e0", "e1"])

    def test_refused_connection_fails_over_and_all_down_is_503(self):
        engines = self.engines(2)
        router, port = self.start(engines)
        router.monitor.stop()  # passive detection only
        engines[0].stop()
        status, headers, _ = chat(port)
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-TensorMeld-Replica"], "r1")
        self.assertEqual(router.pool.snapshot()["replicas"][0]["state"], "down")
        engines[1].stop()
        self.assertEqual(chat(port)[0], 503)
        status, _, body = call(port, "GET", "/health")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["replicas_up"], 0)

    def test_no_replica_ready_at_start(self):
        port = free_port()
        router = ReplicaRouter(RouterConfig.parse(router_data([port])))
        router.start()
        self.addCleanup(router.stop)
        self.assertEqual(chat(router.address[1])[0], 503)

    def test_status_and_metrics(self):
        engines = self.engines(2)
        _, port = self.start(engines)
        chat(port)
        status, _, body = call(port, "GET", "/api/status")
        snapshot = json.loads(body)
        self.assertEqual(snapshot["schema"], "tensormeld/replica-router-status-v1")
        self.assertEqual([r["state"] for r in snapshot["replicas"]], ["up", "up"])
        status, headers, body = call(port, "GET", "/metrics")
        self.assertEqual(status, 200)
        text = body.decode()
        self.assertIn('tensormeld_router_replica_up{replica="r0",node="node0"} 1', text)
        self.assertIn('tensormeld_router_replica_completed_total{replica="r0",node="node0"} 1', text)


class CredentialTests(RouterCase):
    def test_client_credentials_are_not_forwarded_and_replica_key_is(self):
        engines = self.engines(1)
        with tempfile.TemporaryDirectory() as tmp:
            router_key = Path(tmp) / "router.key"
            router_key.write_text("front-key", encoding="ascii")
            replica_key = Path(tmp) / "replica.key"
            replica_key.write_text("back-key", encoding="ascii")
            data = router_data([engines[0].port])
            data["api_key_file"] = str(router_key)
            data["replicas"][0]["api_key_file"] = str(replica_key)
            router = ReplicaRouter(RouterConfig.parse(data))
        router.start()
        self.addCleanup(router.stop)
        self.assertTrue(router.wait_for_replicas(1, 5))
        port = router.address[1]
        self.assertEqual(chat(port)[0], 401)
        self.assertEqual(chat(port, headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(chat(port, headers={"Authorization": "Bearer front-key"})[0], 200)
        self.assertEqual(chat(port, headers={"x-api-key": "front-key"})[0], 200)
        seen = engines[0].requests[0]["headers"]
        self.assertEqual(seen.get("Authorization"), "Bearer back-key")
        self.assertNotIn("x-api-key", {k.lower() for k in seen})
        # Health stays open for the supervisor.
        self.assertEqual(call(port, "GET", "/health")[0], 200)


def conversation(port, turns, *, user=None, headers=None, prefix="doc"):
    """Send `turns` chat turns of one conversation; returns the replica of every turn."""
    messages = [{"role": "system", "content": f"{prefix} system"}]
    served = []
    for t in range(turns):
        messages.append({"role": "user", "content": f"{prefix} question {t}"})
        body = {"messages": list(messages)}
        if user is not None:
            body["user"] = user
        status, hdrs, _ = call(port, "POST", "/v1/chat/completions", body, headers=headers)
        assert status == 200, status
        served.append(hdrs["X-TensorMeld-Replica"])
        messages.append({"role": "assistant", "content": f"answer {t}"})
    return served


class AffinityKeyTests(unittest.TestCase):
    def test_later_turn_shares_the_earlier_prefix(self):
        m1 = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"}]
        m2 = m1 + [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
        k1 = affinity_keys("/v1/chat/completions", json.dumps({"messages": m1}).encode(), None)
        k2 = affinity_keys("/v1/chat/completions", json.dumps({"messages": m2}).encode(), None)
        self.assertEqual(len(k1), 2)
        self.assertEqual(len(k2), 4)
        self.assertEqual(k2[2:], k1)          # longest first: the shared prefix is the tail
        other = affinity_keys("/v1/chat/completions", json.dumps({"messages": m1, "model": "x"}).encode(), None)
        self.assertFalse(set(other) & set(k1))  # another model never shares a key

    def test_session_ids_and_prompts(self):
        a = affinity_keys("/v1/chat/completions", json.dumps({"messages": [], "user": "u1"}).encode(), None)
        b = affinity_keys("/v1/messages", json.dumps({"messages": [], "metadata": {"user_id": "u1"}}).encode(), None)
        c = affinity_keys("/v1/chat/completions", b"{}", "u1")
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[0], c[0])
        text = affinity_keys("/v1/completions", json.dumps({"prompt": "x" * 1300}).encode(), None)
        self.assertEqual(len(text), 3)
        ids = affinity_keys("/v1/completions", json.dumps({"prompt": list(range(600))}).encode(), None)
        self.assertEqual(len(ids), 3)
        self.assertEqual(affinity_keys("/v1/completions", b"not json", None), [])

    def test_table_is_bounded_lru(self):
        t = AffinityTable(3)
        t.record([b"a", b"b"], "r0")
        t.record([b"c"], "r1")
        self.assertEqual(t.lookup([b"a"]), ["r0"])
        t.record([b"d"], "r1")
        self.assertEqual(len(t), 3)
        self.assertEqual(t.lookup([b"b"]), [])     # least recently used went first
        self.assertEqual(t.lookup([b"c", b"a", b"d"]), ["r1", "r0"])


class AffinityRoutingTests(RouterCase):
    AFF = {"mode": "prefix", "max_extra_inflight": 1}

    def test_conversations_stick_to_their_replica(self):
        engines = self.engines(2)
        router, port = self.start(engines, affinity=self.AFF)
        first = conversation(port, 4, prefix="alpha")
        second = conversation(port, 4, prefix="beta")
        self.assertEqual(len(set(first)), 1)
        self.assertEqual(len(set(second)), 1)
        self.assertNotEqual(first[0], second[0])   # new conversations still spread by load
        aff = router.pool.snapshot()["affinity"]
        self.assertEqual((aff["hits"], aff["none"]), (6, 2))

    def test_without_affinity_turns_alternate(self):
        engines = self.engines(2)
        _, port = self.start(engines)
        self.assertEqual(conversation(port, 4), ["r0", "r1", "r0", "r1"])

    def test_session_header_and_user_field(self):
        engines = self.engines(2)
        _, port = self.start(engines, affinity=self.AFF)
        a = [call(port, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": str(i)}]},
                  headers={"X-Session-Id": "s1"})[1]["X-TensorMeld-Replica"] for i in range(3)]
        self.assertEqual(len(set(a)), 1)
        b = [call(port, "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": str(i)}], "user": "u9"})[1][
            "X-TensorMeld-Replica"] for i in range(3)]
        self.assertEqual(len(set(b)), 1)

    def test_affinity_yields_to_a_full_replica(self):
        engines = self.engines(2, delay_s=0.4)
        router, port = self.start(engines, affinity=self.AFF)
        home = conversation(port, 1, prefix="hot")[0]
        results = []
        # Two turns of the same conversation at once: the second cannot wait for the first.
        body = {"messages": [{"role": "system", "content": "hot system"}, {"role": "user", "content": "hot question 0"},
                             {"role": "assistant", "content": "answer 0"}, {"role": "user", "content": "more"}]}
        threads = [threading.Thread(target=lambda: results.append(call(port, "POST", "/v1/chat/completions", body)))
                   for _ in range(2)]
        for t in threads:
            t.start()
            time.sleep(0.05)
        for t in threads:
            t.join()
        served = sorted(r[1]["X-TensorMeld-Replica"] for r in results)
        self.assertEqual(served, ["r0", "r1"])
        self.assertIn(home, served)
        self.assertGreaterEqual(router.pool.snapshot()["affinity"]["yields"], 1)

    def test_failover_to_the_other_replica_and_back(self):
        engines = self.engines(2)
        router, port = self.start(engines, affinity=self.AFF)
        home = conversation(port, 1, prefix="fo")[0]
        dead = engines[int(home[1])]
        dead.healthy = False
        deadline = time.monotonic() + 5
        while any(r["id"] == home and r["state"] == "up" for r in router.pool.snapshot()["replicas"]):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.02)
        moved = conversation(port, 3, prefix="fo")
        self.assertNotIn(home, moved)              # served by the survivor while the home is down
        dead.healthy = True
        self.assertTrue(router.wait_for_replicas(2, 5))
        self.assertEqual(len(set(conversation(port, 3, prefix="fo"))), 1)

    def test_config_validates_affinity(self):
        for bad in ({"mode": "sticky"}, {"mode": "prefix", "max_entries": 0}, {"mode": "prefix", "x": 1}):
            with self.assertRaises(ValidationError):
                RouterConfig.parse(router_data([18081], affinity=bad))
        self.assertEqual(RouterConfig.parse(router_data([18081])).affinity.mode, "off")


class BenchTests(RouterCase):
    def test_bench_through_the_router_keeps_conversations_sticky(self):
        engines = self.engines(2, delay_s=0.05)
        data = router_data([e.port for e in engines], affinity={"mode": "prefix"})
        for r in data["replicas"]:
            r["max_inflight"] = 2   # room for every conversation on its home replica
        router = ReplicaRouter(RouterConfig.parse(data))
        router.start()
        self.addCleanup(router.stop)
        self.assertTrue(router.wait_for_replicas(2, 5))
        port = router.address[1]
        result = run_bench(f"http://127.0.0.1:{port}", 4, 3, 16, [f"doc {u} " * 50 for u in range(4)], label="t")
        self.assertEqual((result["ok"], result["failed"]), (12, 0))
        self.assertEqual(result["sticky_conversations"], 4)
        self.assertEqual(sum(result["per_replica_requests"].values()), 12)
        self.assertEqual(set(result["per_replica_requests"]), {"r0", "r1"})
        # Every later turn resent the earlier messages verbatim.
        later = [r for r in engines[0].requests + engines[1].requests if len(r["body"]["messages"]) > 2]
        self.assertTrue(all(r["body"]["messages"][0]["role"] == "system" for r in later))


class FakePrefillService:
    """Prefill service double: POST /v1/prefill pushes a small state to the handoff URL."""

    def __init__(self, *, status: int = 200, delay_s: float = 0.0):
        self.status = status
        self.delay_s = delay_s
        self.healthy = True
        self.calls: list[dict] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
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

            def do_GET(self):
                self._json(200 if service.healthy else 503, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                service.calls.append(body)
                time.sleep(service.delay_s)
                if service.status == 200:
                    url = body["handoff"][len("http://"):]
                    hostport, path = url.split("/", 1)
                    host, port = hostport.split(":")
                    conn = http.client.HTTPConnection(host, int(port), timeout=5)
                    state = json.dumps({"tokens": len(body["tokens"]), "session": body["session"]}).encode()
                    conn.request("POST", "/" + path, body=state, headers={"Content-Type": "application/json"})
                    conn.getresponse().read()
                    conn.close()
                self._json(service.status, {"next_token": 7})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class PrefillPlacementTests(RouterCase):
    # Toy profiles: the replica prefills 1 ms per token; the service 0.2 ms per token plus small fixed costs.
    NODE = {"prefill_curve": [[1, 0.001], [1000, 1.0]], "restore_s": 0.0}
    SVC = {"gpu": {"name": "test-gpu", "vram_gib": 8, "pinned_gib": 0, "pcie_gbps": 16}, "model": "toy",
           "prefill_curve": [[1, 0.0002], [1000, 0.2]], "handoff_curve": [[1, 0.01], [1000, 0.02]],
           "service_s": 0.01, "import_pause_s": 0.0}

    def service(self, **kw):
        sv = FakePrefillService(**kw)
        self.addCleanup(sv.stop)
        return sv

    def test_joint_reservation_blocks_front_and_keeps_service_residency(self):
        engines = self.engines(1)
        sv = self.service()
        resource = JointReservation(100, enabled=True)
        front = resource.acquire("front", 90, safe_point="between_requests")
        config = RouterConfig.parse(router_data([engines[0].port], prefill=self.prefill(sv, "rtx")))
        router = ReplicaRouter(config, joint_reservation=resource)
        router.start()
        self.addCleanup(router.stop)
        port = router.address[1]
        status, headers, _ = chat(port, "a" * 1000)
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "fallback:accelerator_reserved")
        self.assertIs(resource.current, front)
        self.assertTrue(resource.release(front, buffers_released=True, safe_point="window_boundary"))
        service_lease = resource.acquire("prefill", 95, safe_point="between_requests")
        status, headers, _ = chat(port, "b" * 1000)
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "offload:rtx")
        self.assertIs(resource.current, service_lease)

    def test_balanced_auto_prefers_paid_cache_only_at_equal_load(self):
        engines = self.engines(2)
        sv = self.service()
        prefill = self.prefill(sv, "auto", cache={"block_tokens": 16})
        del prefill["services"]
        data = router_data([e.port for e in engines], balance_active_decodes=True, prefill=prefill)
        for r in data["replicas"]:
            r["max_inflight"] = 8
        router = ReplicaRouter(RouterConfig.parse(data))
        router.start()
        self.addCleanup(router.stop)
        tokens = list(range(1024))
        router.server.mirrors["r1"].record(tokens)
        body = {"prompt": tokens + [2], "max_tokens": 1}
        status, headers, _ = call(router.address[1], "POST", "/v1/completions", body)
        self.assertEqual(status, 200)
        self.assertEqual(len(engines[1].requests), 1)
        self.assertTrue(router.pool.wait_idle("r1", 2))
        held = router.pool.acquire(scores={"r0": 10, "r1": 0})
        self.assertEqual(held.id, "r1")
        try:
            call(router.address[1], "POST", "/v1/completions", body)
            self.assertEqual(len(engines[0].requests), 1)
        finally:
            router.pool.release(held.id, ok=True)

    def prefill(self, sv, policy, **extra):
        cfg = {"policy": policy, "services": [{"id": "rtx", "node": "pc", "address": "127.0.0.1", "port": sv.port,
                                                "max_inflight": 1, "profile": self.SVC}],
               "node_profile": self.NODE, "margin_s": 0.05, "min_tokens_idle": 100, "cache": {"block_tokens": 16}}
        cfg.update(extra)
        return cfg

    def test_auto_offloads_when_cheaper_and_keeps_short_prompts(self):
        engines = self.engines(1)
        sv = self.service()
        _, port = self.start(engines, prefill=self.prefill(sv, "auto"))
        status, headers, _ = chat(port, "short")
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "local"))     # idle replica, P < floor
        status, headers, _ = chat(port, "x" * 600)                                       # 0.6 s local vs ~0.14 s offload
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "offload:rtx"))
        self.assertIn("offload=", headers["X-TensorMeld-Route-Cost"])
        self.assertEqual(len(sv.calls), 1)
        self.assertEqual(len(sv.calls[0]["tokens"]), 600)
        self.assertTrue(sv.calls[0]["handoff"].endswith(f":{engines[0].port}/admin/import-state"))
        self.assertEqual(len(engines[0].imports), 1)             # the state reached the decode replica first
        status_body = json.loads(call(port, "GET", "/api/status")[2])
        self.assertEqual(status_body["prefill"]["placements"]["offload"], 1)

    def test_saving_within_margin_stays_local(self):
        engines = self.engines(1)
        sv = self.service()
        _, port = self.start(engines, prefill=self.prefill(sv, "auto", margin_s=5.0))
        status, headers, _ = chat(port, "x" * 600)
        self.assertEqual(headers["X-TensorMeld-Prefill"], "local")
        self.assertEqual(sv.calls, [])

    def test_cached_prefix_is_compared_against_the_full_prompt_on_the_service(self):
        engines = self.engines(1)
        sv = self.service()
        _, port = self.start(engines, prefill=self.prefill(sv, "rtx"))
        first = call(port, "POST", "/v1/completions", {"prompt": list(range(640))})
        self.assertEqual(first[1]["X-TensorMeld-Prefill"], "offload:rtx")
        self.assertEqual(engines[0].tokenized, 0)                 # token prompts need no tokenize
        engines2 = self.engines(1)
        sv2 = self.service()
        _, port2 = self.start(engines2, prefill=self.prefill(sv2, "auto"))
        prompt = list(range(640))
        r1 = call(port2, "POST", "/v1/completions", {"prompt": prompt})
        self.assertEqual(r1[1]["X-TensorMeld-Prefill"], "offload:rtx")
        # Offloaded prompts are not in the replica's own cache mirror; a local one is:
        r2 = call(port2, "POST", "/v1/completions", {"prompt": list(range(50))})
        self.assertEqual(r2[1]["X-TensorMeld-Prefill"], "local")
        r3 = call(port2, "POST", "/v1/completions", {"prompt": list(range(50)) + [7, 7]})
        self.assertEqual(r3[1]["X-TensorMeld-Prefill"], "cache")  # 48 of 52 tokens restorable (blocks of 16)

    def test_failing_or_down_service_falls_back(self):
        engines = self.engines(1)
        sv = self.service(status=500)
        _, port = self.start(engines, prefill=self.prefill(sv, "rtx"))
        status, headers, _ = chat(port, "y" * 600)
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "fallback:http500"))
        sv.stop()
        status, headers, _ = chat(port, "w" * 600)
        self.assertEqual(status, 200)
        self.assertTrue(headers["X-TensorMeld-Prefill"].startswith(("fallback:", "local")))

    def test_busy_service_queues_instead_of_refusing(self):
        engines = self.engines(1)
        sv = self.service(delay_s=0.3)
        data = router_data([engines[0].port], prefill=self.prefill(sv, "auto"))
        data["replicas"][0]["max_inflight"] = 4
        router = ReplicaRouter(RouterConfig.parse(data))
        router.start()
        self.addCleanup(router.stop)
        self.assertTrue(router.wait_for_replicas(1, 5))
        port = router.address[1]
        results = []
        threads = [threading.Thread(target=lambda: results.append(chat(port, "z" * 900))) for _ in range(2)]
        for t in threads:
            t.start()
            time.sleep(0.05)
        for t in threads:
            t.join()
        notes = sorted(r[1]["X-TensorMeld-Prefill"] for r in results)
        # 0.9 s local (plus pausing the other request) vs ~0.5 s queued behind the first offload: both offload.
        self.assertEqual(notes, ["offload:rtx", "offload:rtx"])
        self.assertEqual(len(sv.calls), 2)

    def test_auto_without_services_routes_by_cache_and_load(self):
        engines = self.engines(2)
        data = router_data([e.port for e in engines], prefill={"policy": "auto", "node_profile": self.NODE,
                                                               "cache": {"block_tokens": 16}})
        router = ReplicaRouter(RouterConfig.parse(data))
        router.start()
        self.addCleanup(router.stop)
        self.assertTrue(router.wait_for_replicas(2, 5))
        port = router.address[1]
        a = call(port, "POST", "/v1/completions", {"prompt": list(range(100))})[1]["X-TensorMeld-Replica"]
        b = call(port, "POST", "/v1/completions", {"prompt": list(range(500, 600))})[1]["X-TensorMeld-Replica"]
        # Each later turn goes where its prefix is cached, whatever the tie-breaking would say.
        for _ in range(3):
            self.assertEqual(call(port, "POST", "/v1/completions", {"prompt": list(range(100)) + [1]})[1]["X-TensorMeld-Replica"], a)
            self.assertEqual(call(port, "POST", "/v1/completions", {"prompt": list(range(500, 600)) + [2]})[1]["X-TensorMeld-Replica"], b)

    def test_prefill_config_validation(self):
        sv_entry = {"id": "rtx", "node": "pc", "address": "127.0.0.1", "port": 18090, "max_inflight": 1}
        for bad in ({"policy": "gpu"}, {"policy": "rtx"}, {"policy": "auto"},                      # auto needs node_profile
                    {"policy": "auto", "node_profile": self.NODE, "services": [sv_entry]},          # and service profiles
                    {"policy": "nodes", "handoff_path": "relative"},
                    {"policy": "auto", "node_profile": self.NODE, "services": [dict(sv_entry, profile="seed:none:1")]},
                    {"policy": "nodes", "x": 1}):
            with self.assertRaises(ValidationError):
                RouterConfig.parse(router_data([18081], prefill=bad))
        ok = RouterConfig.parse(router_data([18081], prefill={
            "policy": "auto", "node_profile": self.NODE, "services": [dict(sv_entry, profile="seed:deepseek41:74")]}))
        self.assertEqual(ok.prefill.service_profiles[0].gpu.vram_gib, 74.0)
        data = router_data([18081], prefill={"policy": "rtx", "services": [dict(sv_entry, id="r0")]})
        with self.assertRaises(ValidationError):    # a service id may not repeat a replica id
            RouterConfig.parse(data)


if __name__ == "__main__":
    unittest.main()
