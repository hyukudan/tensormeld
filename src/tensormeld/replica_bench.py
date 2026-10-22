"""Concurrent multi-turn chat load against an OpenAI-compatible endpoint (an engine server or
the replica router), for measuring aggregate throughput and time to first token.

Each simulated user holds one conversation: a system message with its own document
excerpt, then `turns` questions asked back to back, every turn resending the whole history
(as chat clients do). All users start together. Per request the client records its
latency, the serving replica (router header), the usage counters (prompt, completion and
cached prompt tokens) and the server's own timings when it reports them (queue, prefill
and decode milliseconds). Requests are greedy (temperature 0).

The summary gives the aggregate completion tokens per second over the whole run, and the
time to first token of first turns and of later turns separately: later turns are where a
prefix state cache on the replica that already holds the conversation can skip work.

With `stream=True` (default) the client also measures its own timings from the event
stream: time to the first content chunk, the gaps between chunks, decode tokens per second
per request (tokens after the first over the time from first to last chunk) and the
aggregate over the common window (from the moment every user has its first token of a turn
until the first user finishes it). Reasoning deltas (thinking models) count as generated
tokens; the first visible answer token is recorded separately (`first_content_s`). Chunks
carry text, not token counts: a request's completion tokens are spread over its chunks in
proportion to their text (approximation).
`doc_seed` shifts the document excerpts, so a new seed gives cold caches without restarting
the engines; `stagger_s` spaces the users' starts.
"""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from .schema import ValidationError

BENCH_SCHEMA = "tensormeld/replica-bench-v1"
QUESTIONS = (
    "Summarize the main points of the document above in a few sentences.",
    "Which part of it would be hardest to implement, and why?",
    "List three concrete questions a reviewer should ask about it.",
    "Rewrite its first paragraph for a non-technical reader.",
    "What would you test first, and how?",
    "Name one risk it does not mention.",
)


def _endpoint(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    if parts.scheme != "http" or not parts.hostname or parts.path not in ("", "/"):
        raise ValidationError("url: expected http://host:port")
    return parts.hostname, parts.port or 80


def documents(paths: list[Path], users: int, chars: int, seed: int = 0) -> list[str]:
    """One excerpt of `chars` characters per user, taken at different offsets of the
    concatenated text files (shifted by `seed`), so every conversation has its own prefix."""
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths)
    if len(text) < chars:
        raise ValidationError(f"documents: need at least {chars} characters, got {len(text)}")
    span = max(1, len(text) - chars)
    return [f"[{seed}:{u}] " + text[((u + 1000 * seed) * 7919 * 13) % span :][:chars] for u in range(users)]


def _post(host: str, port: int, path: str, body: dict, headers: dict, timeout: float) -> tuple[int, dict, Any]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        payload = json.dumps(body).encode("utf-8")
        conn.request("POST", path, body=payload, headers={"Content-Type": "application/json", **headers})
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed = json.loads(data)
        except ValueError:
            parsed = {"raw": data[:300].decode("utf-8", "replace")}
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, parsed
    finally:
        conn.close()


def _post_stream(host: str, port: int, path: str, body: dict, headers: dict, timeout: float,
                 t0: float) -> tuple[int, dict, dict, list[tuple[float, int]], str]:
    """Streaming request: (status, headers, final data with usage/strix, chunks (t, chars), text)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        payload = json.dumps(body).encode("utf-8")
        conn.request("POST", path, body=payload, headers={"Content-Type": "application/json", **headers})
        resp = conn.getresponse()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        if resp.status != 200:
            data = resp.read()
            try:
                return resp.status, hdrs, json.loads(data), [], ""
            except ValueError:
                return resp.status, hdrs, {"raw": data[:300].decode("utf-8", "replace")}, [], ""
        chunks: list[tuple[float, int]] = []
        text: list[str] = []
        final: dict = {}
        first_content: float | None = None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            # Thinking models stream their reasoning first (reasoning_content): those are generated tokens too.
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            piece = delta.get("content") or choice.get("text") or ""
            if thought or piece:
                chunks.append((time.monotonic() - t0, len(thought) + len(piece)))
            if piece:
                if first_content is None:
                    first_content = time.monotonic() - t0
                text.append(piece)
            if event.get("usage") or event.get("strix"):
                final = event
        final = dict(final, _first_content_s=first_content)
        return 200, hdrs, final, chunks, "".join(text)
    finally:
        conn.close()


def run_bench(
    url: str,
    users: int,
    turns: int,
    max_tokens: int,
    docs: list[str],
    *,
    model: str = "default",
    api_key: str | None = None,
    timeout_s: float = 1800.0,
    label: str = "",
    stream: bool = True,
    stagger_s: float = 0.0,
) -> dict[str, Any]:
    host, port = _endpoint(url)
    if users < 1 or turns < 1 or max_tokens < 1 or len(docs) < users:
        raise ValidationError("bench: users, turns, max_tokens >= 1 and one document per user")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    start = threading.Barrier(users)
    t0 = [0.0]

    def user(u: int) -> None:
        messages = [{"role": "system", "content": "You are a careful technical assistant. Document:\n\n" + docs[u]}]
        start.wait()
        if u == 0:
            t0[0] = time.monotonic()
        else:
            while t0[0] == 0.0:
                time.sleep(0.001)
        time.sleep(u * stagger_s)
        for t in range(turns):
            messages.append({"role": "user", "content": QUESTIONS[(u + t) % len(QUESTIONS)]})
            body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
            if stream:
                body.update(stream=True, stream_options={"include_usage": True})
            sent = time.monotonic()
            rec: dict[str, Any] = {"user": u, "turn": t}
            try:
                if stream:
                    status, hdrs, data, chunks, reply = _post_stream(host, port, "/v1/chat/completions", body, headers,
                                                                     timeout_s, t0[0])
                    rec["chunks"] = [[round(tc, 4), n] for tc, n in chunks]
                    if data.get("_first_content_s") is not None:
                        rec["first_content_s"] = round(data["_first_content_s"], 4)
                else:
                    status, hdrs, data = _post(host, port, "/v1/chat/completions", body, headers, timeout_s)
                    reply = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                rec["status"] = status
                rec["replica"] = hdrs.get("x-tensormeld-replica")
                rec["prefill_place"] = hdrs.get("x-tensormeld-prefill")
                usage = data.get("usage") or {}
                details = usage.get("prompt_tokens_details") or {}
                strix = data.get("strix") or {}
                rec.update(
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    cached_tokens=details.get("cached_tokens"),
                    queue_ms=strix.get("queue_ms"),
                    prefill_ms=strix.get("prefill_ms"),
                    decode_ms=strix.get("decode_ms"),
                    mtp_accepted=strix.get("mtp_accepted"),
                    mtp_proposed=strix.get("mtp_proposed"),
                )
                if status == 200:
                    messages.append({"role": "assistant", "content": reply})
                else:
                    rec["error"] = json.dumps(data)[:300]
            except (OSError, http.client.HTTPException) as exc:
                rec["status"] = None
                rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["sent_s"] = round(sent - t0[0], 3)
            rec["latency_s"] = round(time.monotonic() - sent, 3)
            with lock:
                records.append(rec)
            if rec.get("status") != 200:
                return  # the conversation cannot continue without the reply

    threads = [threading.Thread(target=user, args=(u,), daemon=True) for u in range(users)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.monotonic() - t0[0]
    return summarize(records, wall, label=label, users=users, turns=turns, max_tokens=max_tokens, url=url,
                     stream=stream, stagger_s=stagger_s)


def _ttft_s(r: dict[str, Any]) -> float | None:
    if r.get("queue_ms") is not None and r.get("prefill_ms") is not None:
        return (r["queue_ms"] + r["prefill_ms"]) / 1000.0
    return None


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(values[len(values) // 2], 3),
        "p95": round(values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))], 3),
        "max": round(values[-1], 3),
    }


def _client_metrics(ok: list[dict[str, Any]]) -> dict[str, Any]:
    """Client-side timings from the chunk timestamps (streaming runs)."""
    streamed = [r for r in ok if r.get("chunks")]
    if not streamed:
        return {}
    ttft, gaps, decode, events = [], [], [], []
    for r in streamed:
        chunks = r["chunks"]
        total_chars = sum(n for _, n in chunks) or 1
        tokens = r.get("completion_tokens") or len(chunks)
        ttft.append(chunks[0][0] - r["sent_s"])
        for (ta, _), (tb, nb) in zip(chunks, chunks[1:]):
            share = max(tokens * nb / total_chars, 1e-9)
            gaps.append((tb - ta) / share)
        if len(chunks) > 1 and tokens > 1 and chunks[-1][0] > chunks[0][0]:
            decode.append((tokens - 1) / (chunks[-1][0] - chunks[0][0]))
        for tc, n in chunks:
            events.append((tc, tokens * n / total_chars))
    # Common window per turn: every user has its first token of that turn, nobody has finished it yet.
    windows = []
    for t in sorted({r["turn"] for r in streamed}):
        turn = [r for r in streamed if r["turn"] == t]
        if len(turn) < 2:
            continue
        start = max(r["chunks"][0][0] for r in turn)
        end = min(r["chunks"][-1][0] for r in turn)
        if end > start:
            windows.append((start, end))
    window_tokens = sum(n for tc, n in events if any(a <= tc <= b for a, b in windows))
    window_time = sum(b - a for a, b in windows)
    return {
        "client_ttft_s": _stats(ttft),
        "client_inter_token_s": _stats(gaps),
        "client_decode_tok_s_per_request": _stats(decode),
        "common_window_s": round(window_time, 3),
        "common_window_aggregate_tok_s": round(window_tokens / window_time, 2) if window_time > 0 else None,
    }


def summarize(records: list[dict[str, Any]], wall: float, **meta: Any) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == 200]
    tokens = sum(r.get("completion_tokens") or 0 for r in ok)
    by_replica: dict[str, int] = {}
    for r in ok:
        by_replica[str(r.get("replica"))] = by_replica.get(str(r.get("replica")), 0) + 1
    first = [x for x in (_ttft_s(r) for r in ok if r["turn"] == 0) if x is not None]
    later = [x for x in (_ttft_s(r) for r in ok if r["turn"] > 0) if x is not None]
    later_prompt = sum(r.get("prompt_tokens") or 0 for r in ok if r["turn"] > 0)
    later_cached = sum(r.get("cached_tokens") or 0 for r in ok if r["turn"] > 0)
    decode = [
        (r["completion_tokens"] - 1) / (r["decode_ms"] / 1000.0)
        for r in ok
        if r.get("decode_ms") and (r.get("completion_tokens") or 0) > 1
    ]
    # A conversation is "sticky" when every turn ran on the same replica.
    convs: dict[int, set] = {}
    for r in ok:
        convs.setdefault(r["user"], set()).add(r.get("replica"))
    accepted = sum(r.get("mtp_accepted") or 0 for r in ok)
    proposed = sum(r.get("mtp_proposed") or 0 for r in ok)
    places: dict[str, int] = {}
    for r in ok:
        if r.get("prefill_place"):
            places[r["prefill_place"].split(":")[0]] = places.get(r["prefill_place"].split(":")[0], 0) + 1
    return {
        "schema": BENCH_SCHEMA,
        **meta,
        "wall_s": round(wall, 3),
        "requests": len(records),
        "ok": len(ok),
        "failed": len(records) - len(ok),
        "completion_tokens": tokens,
        "aggregate_tok_s": round(tokens / wall, 2) if wall > 0 else 0.0,
        "per_replica_requests": by_replica,
        "sticky_conversations": sum(len(v) == 1 for v in convs.values()),
        "conversations": len(convs),
        "ttft_first_turn_s": _stats(first),
        "ttft_later_turns_s": _stats(later),
        "later_turns_cached_fraction": round(later_cached / later_prompt, 3) if later_prompt else None,
        "decode_tok_s_per_request": _stats(decode),
        "mtp_acceptance": round(accepted / proposed, 3) if proposed else None,
        "prefill_places": places,
        **_client_metrics(ok),
        "records": sorted(records, key=lambda r: (r["sent_s"], r["user"])),
    }


__all__ = ["BENCH_SCHEMA", "documents", "run_bench", "summarize"]
