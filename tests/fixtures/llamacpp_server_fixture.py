from __future__ import annotations

import hashlib
import http.server
import json
import os
import sys


def value(args, flag):
    i = args.index(flag)
    return args[i + 1]


def main() -> int:
    if any(key.startswith("LLAMA_ARG_") for key in os.environ):
        print("inherited LLAMA_ARG environment leaked", file=sys.stderr, flush=True)
        return 63
    args = sys.argv[1:]
    required = [
        "-m",
        "--fit",
        "--device",
        "--override-tensor",
        "--host",
        "--port",
        "--ctx-size",
        "--parallel",
        "--no-webui",
        "--jinja",
    ]
    for flag in required:
        if flag not in args:
            print(f"missing {flag}", file=sys.stderr, flush=True)
            return 64
    if value(args, "--host") != "127.0.0.1":
        print("non-loopback host rejected", file=sys.stderr, flush=True)
        return 65
    if value(args, "--parallel") != "1":
        print("parallel must be one", file=sys.stderr, flush=True)
        return 66
    port = int(value(args, "--port"))
    model = value(args, "-m")
    override = value(args, "--override-tensor")
    print("fixture llama-server starting", file=sys.stderr, flush=True)
    print("fixture backend ready path", file=sys.stderr, flush=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _json(self, status, payload):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in {"/health", "/v1/health"}:
                self._json(200, {"status": "ok"})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/completion":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                self._json(400, {"error": "invalid body"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except Exception:
                self._json(400, {"error": "invalid json"})
                return
            expected_keys = {
                "prompt", "n_predict", "seed", "temperature",
                "stream", "cache_prompt",
            }
            if set(payload) != expected_keys:
                self._json(400, {"error": "unexpected fields"})
                return
            if (
                payload["seed"] != 0
                or payload["temperature"] != 0
                or payload["stream"] is not False
                or payload["cache_prompt"] is not False
            ):
                self._json(400, {"error": "non-deterministic contract"})
                return
            prompt = payload["prompt"]
            digest = hashlib.sha256(
                (prompt + "|" + model + "|" + override).encode("utf-8")
            ).hexdigest()[:16]
            self._json(200, {
                "content": "fixture-completion:" + digest,
                "stop": True,
                "tokens_predicted": int(payload["n_predict"]),
            })

        def log_message(self, fmt, *args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
