"""Measure a TCP link between two hosts as link-descriptor evidence.

The probe is a bounded request/response ping-pong: the client sends a request
of R bytes and the server answers with S bytes.  Round trips over a ladder of
sizes give a fixed latency (intercept) and a payload rate (slope) that can be
copied into a ``tensormeld/hetero-hardware-v1`` link with ``provenance:
measured``.  A one-way bulk transfer gives the sustained rate.

The module only uses the standard library so the same file runs on a node
without the package installed (``python3 link_probe.py serve ...``).  The
server binds a loopback or private address only, prints a random session
token, accepts one client that presents it (constant-time compare), serves zero
bytes only (it never executes anything) and exits; every size and duration is
bounded.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import secrets
import socket
import statistics
import struct
import sys
import time

LINK_PROBE_SCHEMA = "tensormeld/link-probe-v1"
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_BULK_BYTES = 2 * 1024 * 1024 * 1024
MAX_ROUNDS = 10_000
DEFAULT_LADDER = (64, 4096, 6144, 20480, 81920, 262144, 1048576)
_HEADER = struct.Struct("!4sIQQ")  # magic, op, request bytes, reply bytes
_MAGIC = b"TMLP"
_OP_PING = 1
_OP_BULK = 2
_OP_DONE = 3
_TOKEN_BYTES = 16


def _private_address(text: str) -> str:
    try:
        ip = ipaddress.ip_address(text)
    except ValueError as exc:
        raise ValueError(f"{text!r}: expected a numeric IP address") from exc
    if not (ip.is_loopback or ip.is_private) or ip.is_link_local or ip.is_multicast:
        raise ValueError(f"{text}: only loopback or private addresses are allowed")
    return str(ip)


def _recv_exact(sock: socket.socket, count: int, buffer: bytearray) -> None:
    view = memoryview(buffer)
    got = 0
    while got < count:
        chunk = min(count - got, len(buffer))
        n = sock.recv_into(view[:chunk])
        if n == 0:
            raise ConnectionError("peer closed the connection")
        got += n


def _send_zeros(sock: socket.socket, count: int, block: bytes) -> None:
    left = count
    view = memoryview(block)
    while left > 0:
        n = min(left, len(block))
        sock.sendall(view[:n])
        left -= n


def _configure(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 4 * 1024 * 1024)
        except OSError:
            pass


def serve(bind: str, port: int, idle_timeout_s: float = 60.0, out=sys.stderr, token: str | None = None) -> int:
    """Serve one probe session and return the number of messages handled."""
    address = _private_address(bind)
    token = token or secrets.token_hex(_TOKEN_BYTES)
    if len(token) != 2 * _TOKEN_BYTES:
        raise ValueError("token: expected 32 hex characters")
    listener = socket.socket(socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((address, port))
    listener.listen(1)
    listener.settimeout(idle_timeout_s)
    print(f"link-probe: listening on {address}:{listener.getsockname()[1]} token {token}", file=out, flush=True)
    handled = 0
    try:
        conn, peer = listener.accept()
    except socket.timeout:
        listener.close()
        return 0
    with conn:
        _configure(conn)
        conn.settimeout(idle_timeout_s)
        presented = bytearray(2 * _TOKEN_BYTES)
        _recv_exact(conn, len(presented), presented)
        if not hmac.compare_digest(bytes(presented), token.encode("ascii")):
            listener.close()
            raise ConnectionError("link-probe: wrong session token")
        buffer = bytearray(1024 * 1024)
        block = bytes(1024 * 1024)
        header = bytearray(_HEADER.size)
        while True:
            _recv_exact(conn, _HEADER.size, header)
            magic, op, request_bytes, reply_bytes = _HEADER.unpack(header)
            if magic != _MAGIC or op not in (_OP_PING, _OP_BULK, _OP_DONE):
                raise ConnectionError("bad probe header")
            if op == _OP_DONE:
                break
            limit = MAX_BULK_BYTES if op == _OP_BULK else MAX_MESSAGE_BYTES
            if request_bytes > limit or reply_bytes > MAX_MESSAGE_BYTES:
                raise ConnectionError("probe message above the bound")
            _recv_exact(conn, request_bytes, buffer)
            _send_zeros(conn, max(reply_bytes, 1), block)
            handled += 1
    listener.close()
    print(f"link-probe: session from {peer[0]} done, {handled} messages", file=out, flush=True)
    return handled


def _fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares fit of round-trip microseconds = a + b * bytes."""
    xs = [float(x) for x, _ in points]
    ys = [y for _, y in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    return mean_y - slope * mean_x, slope


def run(
    host: str,
    port: int,
    *,
    token: str,
    ladder: tuple[int, ...] = DEFAULT_LADDER,
    rounds: int = 200,
    bulk_bytes: int = 256 * 1024 * 1024,
    small_reply_bytes: int = 64,
    label: str = "",
) -> dict:
    """Run the client side and return a ``tensormeld/link-probe-v1`` record."""
    address = _private_address(host)
    if not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"rounds: expected 1..{MAX_ROUNDS}")
    if not 0 <= bulk_bytes <= MAX_BULK_BYTES:
        raise ValueError("bulk_bytes above the bound")
    for size in ladder:
        if not 1 <= size <= MAX_MESSAGE_BYTES:
            raise ValueError(f"ladder size {size} out of range")
    if len(token) != 2 * _TOKEN_BYTES:
        raise ValueError("token: expected 32 hex characters")
    sock = socket.create_connection((address, port), timeout=30)
    _configure(sock)
    sock.sendall(token.encode("ascii"))
    buffer = bytearray(1024 * 1024)
    block = bytes(1024 * 1024)
    rows = []
    try:
        # Symmetric round trips (R = S = size) and asymmetric ones (R = size,
        # tiny reply) separate the one-way cost from the reply cost.
        for size in ladder:
            for reply in (size, small_reply_bytes):
                samples = []
                n = max(5, rounds if size <= 81920 else rounds // 10)
                for i in range(n + 2):
                    t0 = time.perf_counter()
                    sock.sendall(_HEADER.pack(_MAGIC, _OP_PING, size, reply))
                    _send_zeros(sock, size, block)
                    _recv_exact(sock, max(reply, 1), buffer)
                    dt = (time.perf_counter() - t0) * 1e6
                    if i >= 2:  # warm-up
                        samples.append(dt)
                samples.sort()
                rows.append({
                    "request_bytes": size,
                    "reply_bytes": reply,
                    "rounds": len(samples),
                    "rtt_us_p50": round(statistics.median(samples), 1),
                    "rtt_us_p90": round(samples[int(0.9 * (len(samples) - 1))], 1),
                    "rtt_us_p99": round(samples[int(0.99 * (len(samples) - 1))], 1),
                    "rtt_us_min": round(samples[0], 1),
                })
        bulk = None
        if bulk_bytes:
            t0 = time.perf_counter()
            sock.sendall(_HEADER.pack(_MAGIC, _OP_BULK, bulk_bytes, 1))
            _send_zeros(sock, bulk_bytes, block)
            _recv_exact(sock, 1, buffer)
            seconds = time.perf_counter() - t0
            bulk = {"bytes": bulk_bytes, "seconds": round(seconds, 4),
                    "bytes_per_s": int(bulk_bytes / seconds)}
        sock.sendall(_HEADER.pack(_MAGIC, _OP_DONE, 0, 0))
    finally:
        sock.close()
    one_way = [(r["request_bytes"], r["rtt_us_p50"]) for r in rows if r["reply_bytes"] == small_reply_bytes]
    intercept, slope = _fit(one_way)
    # The asymmetric round trip carries R bytes one way plus a tiny reply; the
    # intercept is two fixed latencies, the slope is one payload direction.
    latency_us = max(intercept / 2.0, 0.0)
    rate = int(1e6 / slope) if slope > 0 else None
    return {
        "schema": LINK_PROBE_SCHEMA,
        "label": label,
        "peer": address,
        "transport": "tcp",
        "measurements": rows,
        "bulk": bulk,
        "fit": {
            "model": "rtt_us = 2 * latency_us + request_bytes / bytes_per_s (p50, tiny reply)",
            "latency_us": round(latency_us, 1),
            "bytes_per_s": rate,
        },
        "suggested_link": {
            "latency_us": round(latency_us, 1),
            "bytes_per_s": min(x for x in (rate, bulk and bulk["bytes_per_s"]) if x) if (rate or bulk) else None,
            "provenance": "measured",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="link-probe", description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="role", required=True)
    s = subs.add_parser("serve")
    s.add_argument("--bind", required=True)
    s.add_argument("--port", type=int, default=18300)
    s.add_argument("--idle-timeout", type=float, default=60.0)
    s.add_argument("--token", help="32 hex characters; random when omitted")
    c = subs.add_parser("run")
    c.add_argument("--host", required=True)
    c.add_argument("--port", type=int, default=18300)
    c.add_argument("--rounds", type=int, default=200)
    c.add_argument("--bulk-bytes", type=int, default=256 * 1024 * 1024)
    c.add_argument("--label", default="")
    c.add_argument("--token", required=True)
    args = parser.parse_args(argv)
    if args.role == "serve":
        serve(args.bind, args.port, args.idle_timeout, token=args.token)
        return 0
    result = run(args.host, args.port, token=args.token, rounds=args.rounds,
                 bulk_bytes=args.bulk_bytes, label=args.label)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
