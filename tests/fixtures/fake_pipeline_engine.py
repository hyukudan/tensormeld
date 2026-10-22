"""Fake engine for pipeline deployment tests (stdlib only).

Speaks just enough of the pipeline contract to exercise TensorMeld's controller:

* ``--describe-json``: a ``tensormeld/engine-describe-v1`` with the example 40-layer parts
  descriptor, roles ``whole`` and ``pipeline-stage``, the ``ep2-tp`` group mode and
  state export/import (format ``fake-state-v1``);
* TAIL (``--role pipeline-stage``): rank 0 listens on ``--stage-listen``, accepts one FRONT
  whose hello carries the token from ``--stage-token-file``, computes tokens, exports its
  state; rank 1 only stays alive;
* FRONT (``--backend``): connects to the TAIL, prints ``READY {json}``, serves ``/health``,
  ``/v1/completions``, ``/v1/chat/completions``, ``/admin/drain``, ``/admin/export-state``;
* WHOLE (``--http-bind``): rank 0 serves the same HTTP routes alone, prints a JSON
  ``{"event": "READY"}`` line, accepts ``/admin/import-state``; optional API key file.

Tokens are a deterministic function of a running sequence state (position, previous token)
so a test can check that a request served after a switchover continues exactly where the
pipeline stopped.  Wire messages are JSON lines (the real wire is the engine's).
"""
import hashlib
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "..", "..", "examples", "hetero", "deepseek-v4.1-flash.parts.json")


def arg(name, default=None):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def next_token(prev, pos, seed):
    return (prev * 1103515245 + 12345 + pos * 7 + seed) % 50021


def seed_of(prompt):
    return sum(prompt.encode("utf-8")) % 997


def say(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def describe():
    with open(PARTS, encoding="utf-8") as f:
        parts = json.load(f)
    with open(__file__, "rb") as f:
        own_sha = hashlib.sha256(f.read()).hexdigest()
    caps = sorted({r for p in parts["parts"] for r in p.get("requires", [])})
    say(json.dumps({
        "schema": "tensormeld/engine-describe-v1",
        "engine": {"name": "fake-pipeline", "revision": "r1",
                   "executable_sha256": own_sha,
                   "backend": arg("--fake-backend", "hip")},
        "capabilities": caps,
        "group_modes": [{"id": "ep2-tp", "world": 2, "capabilities": caps, "transports": ["usb4auto"]}],
        "roles": ["whole", "pipeline-stage"],
        "state": {"export": True, "import": True, "format": "fake-state-v1", "transports": ["tcp"], "exact": True},
        "models": [parts],
    }))


def read_token():
    with open(arg("--stage-token-file"), encoding="ascii") as f:
        return f.read().strip()


# --- TAIL --------------------------------------------------------------------------

def run_tail():
    rank = int(arg("--rank"))
    first, layers = int(arg("--first-layer")), int(arg("--layers"))
    token = read_token()
    if rank != 0:
        say("connected")
        while True:
            time.sleep(3600)
    host, port = arg("--stage-listen").rsplit(":", 1)
    srv = socket.create_server((host, int(port)))
    say("READY " + json.dumps({"role": "pipeline-stage", "layers": [first, layers], "listen": arg("--stage-listen"),
                               "model_sha256_prefix": "fake"}))
    state = {"pos": 0, "prev": 1}
    while True:
        conn, _ = srv.accept()
        f = conn.makefile("rwb")
        try:
            hello = json.loads(f.readline() or b"{}")
            if hello.get("hello") != token or hello.get("k") != first:
                f.write(b'{"ok": false}\n')
                f.flush()
                conn.close()
                continue
            f.write(b'{"ok": true}\n')
            f.flush()
            for raw in iter(f.readline, b""):
                msg = json.loads(raw)
                if "step" in msg:
                    s = msg["step"]
                    if s["pos"] != state["pos"]:
                        out = {"error": f"position {s['pos']} != {state['pos']}"}
                    else:
                        tok = next_token(state["prev"], state["pos"], s["seed"])
                        state = {"pos": state["pos"] + 1, "prev": tok}
                        out = {"token": tok}
                elif "export" in msg:
                    out = {"pos": state["pos"], "prev": state["prev"], "layers": [first, layers]}
                else:
                    out = {"pong": True}
                f.write(json.dumps(out).encode() + b"\n")
                f.flush()
        except (OSError, ValueError):
            pass
        finally:
            conn.close()


# --- HTTP front end shared by FRONT and WHOLE ----------------------------------------

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.draining = False
        self.inflight = 0

    def health(self):
        return (503, {"status": "draining"}) if self.draining else (200, {"status": "ok"})


def serve_http(engine, address, api_key=None):
    host, port = address

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send(self, status, value, ctype="application/json", headers=None):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.send(*engine.health())
            elif self.path == "/api/status":
                self.send(200, {"inflight": engine.inflight})
            else:
                self.send(404, {"error": "unknown"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                self.send(401, {"error": "auth"})
                return
            if self.path in ("/v1/completions", "/v1/chat/completions"):
                if engine.draining:
                    self.send(503, {"error": "draining"})
                    return
                req = json.loads(body or b"{}")
                prompt = req.get("prompt") or "".join(m.get("content", "") for m in req.get("messages", []))
                engine.inflight += 1
                try:
                    if req.get("sleep_s"):
                        time.sleep(float(req["sleep_s"]))
                    tokens = engine.generate(seed_of(prompt), int(req.get("max_tokens", 4)))
                except Exception as exc:  # stage lost mid-request
                    self.send(502, {"error": str(exc)})
                    return
                finally:
                    engine.inflight -= 1
                self.send(200, {"choices": [{"text": " ".join(map(str, tokens))}], "tokens": tokens})
            elif self.path == "/admin/drain":
                engine.draining = True
                self.send(200, {"status": "draining"})
            elif self.path == "/admin/export-state" and hasattr(engine, "export"):
                blob = json.dumps(engine.export(), sort_keys=True).encode()
                self.send(200, blob, "application/octet-stream",
                          {"X-State-Format": "fake-state-v1", "X-State-Sha256": hashlib.sha256(blob).hexdigest()})
            elif self.path == "/admin/import-state" and hasattr(engine, "import_state"):
                digest = hashlib.sha256(body).hexdigest()
                if self.headers.get("X-State-Sha256") != digest:
                    self.send(400, {"error": "sha256"})
                    return
                engine.import_state(json.loads(body))
                self.send(200, {"imported": True, "sha256": digest, "sessions": 1})
            else:
                self.send(404, {"error": "unknown"})

    srv = ThreadingHTTPServer((host, int(port)), H)
    srv.daemon_threads = True
    return srv


class Front(Engine):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.pos = 0
        self.prev = 1
        host, port = arg("--stage-connect").rsplit(":", 1)
        deadline = time.monotonic() + 30
        while True:
            try:
                self.sock = socket.create_connection((host, int(port)), timeout=5)
                break
            except OSError:
                if time.monotonic() > deadline:
                    sys.stderr.write("stage connect timed out\n")
                    sys.exit(3)
                time.sleep(0.1)
        self.sock.settimeout(10)
        self.f = self.sock.makefile("rwb")
        if self.call({"hello": read_token(), "k": k}).get("ok") is not True:
            sys.stderr.write("stage refused the hello\n")
            sys.exit(4)
        self.connected = True
        threading.Thread(target=self.ping, daemon=True).start()

    def call(self, msg):
        self.f.write(json.dumps(msg).encode() + b"\n")
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise OSError("stage closed")
        return json.loads(line)

    def ping(self):
        while self.connected:
            time.sleep(0.2)
            try:
                with self.lock:
                    self.call({"ping": True})
            except (OSError, ValueError):
                self.connected = False

    def health(self):
        if not self.connected:
            return 503, {"status": "ok", "stage": "disconnected"}
        status, value = super().health()
        return status, {**value, "stage": "connected"}

    def generate(self, seed, n):
        out = []
        with self.lock:
            for _ in range(n):
                r = self.call({"step": {"pos": self.pos, "seed": seed}})
                if "token" not in r:
                    raise OSError(r.get("error", "stage error"))
                self.pos += 1
                self.prev = r["token"]
                out.append(r["token"])
        return out

    def export(self):
        with self.lock:
            tail = self.call({"export": True})
            return {"format": "fake-state-v1", "front": {"pos": self.pos, "layers": [0, self.k]}, "tail": tail}


class Whole(Engine):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.pos = 0
        self.prev = 1

    def generate(self, seed, n):
        out = []
        with self.lock:
            for _ in range(n):
                tok = next_token(self.prev, self.pos, seed)
                self.pos += 1
                self.prev = tok
                out.append(tok)
        return out

    def import_state(self, state):
        front, tail = state["front"], state["tail"]
        if front["layers"][1] != tail["layers"][0] or tail["layers"][1] != self.layers or front["pos"] != tail["pos"]:
            raise ValueError("state does not cover the model consistently")
        with self.lock:
            self.pos, self.prev = tail["pos"], tail["prev"]


def main():
    if "--describe-json" in sys.argv:
        describe()
        return
    if arg("--role") == "pipeline-stage":
        run_tail()
    elif arg("--backend"):
        k = int(arg("--layers"))
        engine = Front(k)
        host, port = arg("--http").rsplit(":", 1)
        srv = serve_http(engine, (host, port))
        say("READY " + json.dumps({"role": "pipeline-stage", "layers": [0, k], "listen": arg("--http"),
                                   "backend": arg("--backend")}))
        srv.serve_forever()
    elif arg("--http-bind"):
        rank = int(arg("--rank"))
        if rank != 0:
            say("connected")
            while True:
                time.sleep(3600)
        key = None
        if arg("--http-api-key-file"):
            with open(arg("--http-api-key-file"), encoding="ascii") as f:
                key = f.read().strip()
        engine = Whole(int(arg("--layers")))
        srv = serve_http(engine, (arg("--http-bind"), arg("--http-port")), key)
        say(json.dumps({"event": "READY", "rank": 0, "http": f"{arg('--http-bind')}:{arg('--http-port')}"}))
        srv.serve_forever()
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
