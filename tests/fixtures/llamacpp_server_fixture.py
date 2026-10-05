from __future__ import annotations

import http.server
import json
import sys


def value(args, flag):
    i = args.index(flag)
    return args[i + 1]


def main() -> int:
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
    print("fixture llama-server starting", file=sys.stderr, flush=True)
    print("fixture backend ready path", file=sys.stderr, flush=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in {"/health", "/v1/health"}:
                body = json.dumps({"status": "ok"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

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
