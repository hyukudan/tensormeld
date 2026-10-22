"""Bounded LOOPBACK-only echo diagnostic; not a remote agent or tensor protocol.

This intentionally offers no LAN bind address, remote commands, authentication
claims, or one-way bandwidth estimates. Remote diagnostics are blocked until M1.
"""
from __future__ import annotations

import math
import socket
import statistics
import struct
import threading
import time

MAX_FRAME = 8 * 1024 * 1024
HEADER = struct.Struct("!I")


def receive_exact(sock: socket.socket, size: int) -> bytes:
    if not 0 <= size <= MAX_FRAME:
        raise ValueError("receive size outside diagnostic bounds")
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(min(65536, size - len(data)))
        if not chunk:
            raise EOFError("peer closed before a complete diagnostic frame")
        data.extend(chunk)
    return bytes(data)


def send_frame(sock: socket.socket, payload: bytes) -> None:
    if not 0 < len(payload) <= MAX_FRAME:
        raise ValueError("diagnostic payload must contain 1..8 MiB bytes")
    sock.sendall(HEADER.pack(len(payload)))
    sock.sendall(payload)


def receive_frame(sock: socket.socket) -> bytes:
    size, = HEADER.unpack(receive_exact(sock, HEADER.size))
    if not 0 < size <= MAX_FRAME:
        raise ValueError("invalid diagnostic frame length")
    return receive_exact(sock, size)


def loopback(iterations: int = 20, sizes: tuple[int, ...] = (64, 16384, 1048576)) -> dict:
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError("iterations must be an integer between 1 and 100")
    if not sizes or len(sizes) > 8 or any(type(n) is not int or not 1 <= n <= MAX_FRAME for n in sizes):
        raise ValueError("sizes must contain 1..8 valid payload sizes")
    warmups = 2
    total_messages = (iterations + warmups) * len(sizes)
    errors = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(10)
        port = listener.getsockname()[1]

        def worker() -> None:
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    for _ in range(total_messages):
                        send_frame(connection, receive_frame(connection))
            except (OSError, ValueError, EOFError) as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                results = []
                for size in sizes:
                    payload = b"x" * size
                    samples = []
                    for i in range(iterations + warmups):
                        start = time.perf_counter_ns()
                        send_frame(client, payload)
                        received = receive_frame(client)
                        elapsed = time.perf_counter_ns() - start
                        if received != payload:
                            raise ValueError("echo mismatch")
                        if i >= warmups:
                            samples.append(elapsed / 1_000_000)
                    ordered = sorted(samples)
                    results.append({"payload_bytes_each_way": size,
                                    "round_trip_median_ms": statistics.median(samples),
                                    "round_trip_p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
                                    "samples_ms": samples})
        finally:
            thread.join(timeout=10)
        if thread.is_alive() or errors:
            raise RuntimeError(f"loopback worker did not complete cleanly: {errors}")
    return {"schema_version": 1, "transport": "tcp-loopback", "iterations": iterations,
            "qualified_for_inference": False, "results": results,
            "warnings": ["Host-side Python echo RTT, not GPU tensor-path latency.",
                         "Does not measure Ethernet/USB4 or independent one-way payload bandwidth."]}
