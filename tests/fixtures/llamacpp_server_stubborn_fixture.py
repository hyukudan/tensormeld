from __future__ import annotations

import http.server
import signal
import sys


def value(args, flag):
    return args[args.index(flag) + 1]


def main() -> int:
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: None)
    args = sys.argv[1:]
    port = int(value(args, "--port"))

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            return

    print("stubborn fixture ready", file=sys.stderr, flush=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever(poll_interval=0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
