"""Bounded, opt-in HTTP benchmark; private corpora and recordings stay outside Git.

Usage counters are authoritative. An SSE event is not a token: a speculative
server may deliver several tokens per event. No backend process is launched.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
from pathlib import Path
import socket
import statistics
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from .schema import ValidationError

SCHEMA = "tensormeld/protocol-bench-v1"
CATEGORIES = ("coding", "json", "narrative", "prose", "math", "reasoning", "summary", "format")
MAX_CORPUS_BYTES = 32 * 1024 * 1024
MAX_EVENT_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def load_corpus(path: Path, mode: str) -> tuple[list[dict], str]:
    if mode not in ("short", "long"):
        raise ValidationError("mode: expected short or long")
    with path.open("rb") as source:
        raw = source.read(MAX_CORPUS_BYTES + 1)
    if len(raw) > MAX_CORPUS_BYTES:
        raise ValidationError("corpus exceeds 32 MiB")
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValidationError("corpus must be an object")
    cases = doc.get("cases")
    if doc.get("schema") != "tensormeld/protocol-corpus-v1" or not isinstance(cases, list):
        raise ValidationError("corpus: expected protocol-corpus-v1 and cases")
    if not 1 <= len(cases) <= 32:
        raise ValidationError("corpus: expected 1..32 cases")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValidationError("corpus case must be an object")
        ident = case.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValidationError("corpus: case IDs must be unique nonempty strings")
        ids.add(ident)
        if not isinstance(case.get("category"), str):
            raise ValidationError("case category must be a string")
        messages = case.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 16:
            raise ValidationError("case: expected 1..16 messages")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str)):
                raise ValidationError("case: messages require role and text content only")
        for name in ("prompt_tokens", "completion_tokens"):
            bounds = case.get(name)
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or not all(_integer(n, 1, 1_100_000) for n in bounds)
                    or bounds[0] > bounds[1]):
                raise ValidationError(f"case: {name} requires [minimum, maximum]")
        if mode == "short":
            if case["category"] not in CATEGORIES:
                raise ValidationError("short corpus: unknown category")
            if case["prompt_tokens"][0] < 36 or case["prompt_tokens"][1] > 303:
                raise ValidationError("short corpus: prompt bounds must be within 36..303")
            if case["completion_tokens"][0] < 82 or case["completion_tokens"][1] > 200:
                raise ValidationError("short corpus: output bounds must be within 82..200")
        else:
            needles = case.get("needles")
            if (not isinstance(needles, list) or len(needles) != 3
                    or not all(isinstance(n, str) and n for n in needles)
                    or len(set(needles)) != 3):
                raise ValidationError("long corpus: exactly three distinct answer strings required")
            haystack = "\n".join(m["content"] for m in messages)
            positions = [haystack.find(n) for n in needles]
            if min(positions) < 0 or positions != sorted(positions):
                raise ValidationError("long corpus: needles must occur in input order")
    if mode == "short" and (len(cases) != 8 or {c["category"] for c in cases} != set(CATEGORIES)):
        raise ValidationError("short corpus: exactly one case for each of eight categories")
    return cases, _sha(raw)


def request_body(case: dict, model: str, tag: str) -> dict:
    messages = [dict(m) for m in case["messages"]]
    # Unique front tags prevent identical streams from benchmarking shared prefixes.
    messages[0]["content"] = f"[{tag}] " + messages[0]["content"]
    return {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": case["completion_tokens"][1], "stream": True,
            "stream_options": {"include_usage": True},
            "enable_thinking": False,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}


def _url(url: str) -> tuple[type, str, int, str]:
    parts = urlsplit(url)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in ("", "/", "/v1", "/v1/")):
        raise ValidationError("url: expected http(s)://host[:port][/v1] without credentials")
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    return cls, parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), "/v1/chat/completions"


def _stream(url: str, body: dict, key: str | None, timeout: float) -> dict:
    cls, host, port, path = _url(url)
    conn = cls(host, port, timeout=timeout)
    started = time.perf_counter()
    stream_socket: list[Any] = [None]

    def expire() -> None:
        # HTTP/1.0 may detach conn.sock after headers; retain our own socket
        # reference so a trickling or silent stream cannot extend the deadline.
        active = stream_socket[0] or conn.sock
        if active is not None:
            try:
                active.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    deadline = threading.Timer(timeout, expire)
    deadline.daemon = True
    deadline.start()
    record: dict[str, Any] = {"request_sha256": _sha(json.dumps(body, sort_keys=True).encode()),
                              "ttft_s": None, "last_delta_s": None, "sse_deltas": 0}
    try:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        conn.request("POST", path, json.dumps(body).encode("utf-8"), headers)
        stream_socket[0] = conn.sock
        response = conn.getresponse()
        record["status"] = response.status
        record["replica"] = response.getheader("x-tensormeld-replica")
        if response.status != 200:
            raise ValidationError(f"HTTP status {response.status}")
        if not (response.getheader("Content-Type") or "").lower().startswith("text/event-stream"):
            raise ValidationError("response is not text/event-stream")
        usage: dict = {}
        answer, event_lines = [], []
        total_bytes = 0
        done = finished = thought = False
        while not done:
            remaining = timeout - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("request wall deadline exceeded")
            if conn.sock:
                conn.sock.settimeout(remaining)
            line = response.readline(MAX_EVENT_BYTES + 1)
            total_bytes += len(line)
            if len(line) > MAX_EVENT_BYTES or total_bytes > MAX_RESPONSE_BYTES:
                raise ValidationError("stream exceeds byte bound")
            if not line:
                raise ValidationError("stream ended before [DONE]")
            if line.strip():
                if line.startswith(b"data:"):
                    event_lines.append(line[5:].strip())
                    if sum(map(len, event_lines)) > MAX_EVENT_BYTES:
                        raise ValidationError("SSE event exceeds byte bound")
                continue
            if not event_lines:
                continue
            data = b"\n".join(event_lines)
            event_lines.clear()
            if data == b"[DONE]":
                done = True
                continue
            event = json.loads(data)
            if event.get("error"):
                raise ValidationError("server reported a streaming error")
            if event.get("usage"):
                usage.update(event["usage"])
            for choice in event.get("choices") or []:
                if choice.get("index", 0) != 0:
                    raise ValidationError("multiple completions are unsupported")
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                content = delta.get("content") or ""
                if content or reasoning:
                    elapsed = time.perf_counter() - started
                    if record["ttft_s"] is None:
                        record["ttft_s"] = elapsed
                    record["last_delta_s"] = elapsed
                    record["sse_deltas"] += 1
                    thought |= bool(reasoning)
                    answer.append(content)
                if choice.get("finish_reason") is not None:
                    record["finish_reason"] = choice["finish_reason"]
                    finished = True
        if not finished or record["ttft_s"] is None:
            raise ValidationError("stream lacks content or terminal finish reason")
        if record["finish_reason"] not in ("stop", "length"):
            raise ValidationError("stream did not finish normally")
        if thought or "<think>" in "".join(answer):
            raise ValidationError("thinking output received although disabled")
        if not all(_integer(usage.get(k), 1, 1_100_000) for k in ("prompt_tokens", "completion_tokens")):
            raise ValidationError("authoritative prompt/completion usage is required")
        if usage["completion_tokens"] > body["max_tokens"]:
            raise ValidationError("server exceeded requested output cap")
        details = usage.get("prompt_tokens_details") or {}
        record.update(usage=usage, response_text="".join(answer),
                      cached_tokens=details.get("cached_tokens"), error=None)
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException) as exc:
        # Do not serialize exception messages: headers/endpoints may contain credentials.
        record["error"] = str(exc) if isinstance(exc, ValidationError) else type(exc).__name__
    finally:
        deadline.cancel()
        record["total_s"] = time.perf_counter() - started
        conn.close()
    tokens = record.get("usage", {}).get("completion_tokens")
    ttft, last, total = record["ttft_s"], record["last_delta_s"], record["total_s"]
    record["wall_tok_s"] = tokens / total if tokens else None
    record["prompt_tokens_over_ttft_s"] = record["usage"]["prompt_tokens"] / ttft if tokens and ttft else None
    # Protocol convention includes final flush; packed first deltas make this an estimate.
    record["per_stream_tok_s"] = (tokens - 1) / (total - ttft) if tokens and ttft is not None and total > ttft else None
    record["first_to_last_delta_tok_s_estimate"] = (
        (tokens - 1) / (last - ttft) if tokens and last is not None and last > ttft else None)
    return record


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"mean": statistics.fmean(values), "p50": statistics.median(values),
            "p95": ordered[math.ceil(len(ordered) * .95) - 1], "max": ordered[-1]}


def run_protocol(corpus: Path, *, mode: str = "short", url: str | None = None,
                 model: str = "default", levels: tuple[int, ...] = (1, 4, 6),
                 repeats: int = 1, run_id: str = "campaign-1", execute: bool = False,
                 api_key: str | None = None, timeout_s: float = 900,
                 budget_s: float = 900, label: str = "", identity: dict | None = None) -> dict:
    cases, corpus_sha = load_corpus(corpus, mode)
    if (not levels or len(levels) != len(set(levels))
            or not all(_integer(n, 1, 6) for n in levels)
            or not _integer(repeats, 1, 10) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 1200
            or not math.isfinite(budget_s) or not 0 < budget_s <= 1200):
        raise ValidationError("bench: unique concurrency 1..6, repeats 1..10, timeout (0,1200]")
    if mode == "long" and levels != (1,):
        raise ValidationError("long mode requires --levels 1")
    if not run_id or len(run_id) > 64 or not all(c.isalnum() or c in "-_" for c in run_id):
        raise ValidationError("run-id: expected 1..64 alphanumeric, hyphen or underscore characters")
    if execute and not url:
        raise ValidationError("--execute requires --url")
    if url:
        _url(url)
    waves = []
    for repeat in range(repeats):
        for concurrency in levels:
            for case in cases:
                waves.append((repeat, concurrency, case))
    result: dict[str, Any] = {"schema": SCHEMA, "mode": mode, "dry_run": not execute,
        "qualification": False, "label": label, "identity": identity or {},
        "corpus_sha256": corpus_sha, "run_id": run_id, "levels": list(levels),
        "repeats": repeats, "planned_requests": sum(n for _, n, _ in waves),
        "temperature": 0, "thinking": False, "hidden_warmups": 0,
        "counter_source": "server_usage_required", "per_stream_convention": "(completion_tokens-1)/(total_s-ttft_s)",
        "token_bounds_checked": execute, "budget_s": budget_s, "waves": []}
    campaign_start = time.perf_counter()
    for repeat, concurrency, case in waves:
        remaining = budget_s - (time.perf_counter() - campaign_start)
        if execute and remaining <= 0:
            break
        bodies = [request_body(case, model, f"{run_id}-r{repeat}-c{concurrency}-s{i}-{case['id']}")
                  for i in range(concurrency)]
        wave: dict[str, Any] = {"repeat": repeat, "concurrency": concurrency,
                                "case": case["id"], "category": case["category"]}
        if not execute:
            wave.update(prompt_tokens=case["prompt_tokens"], completion_tokens=case["completion_tokens"],
                        requests=[{"request_sha256": _sha(json.dumps(b, sort_keys=True).encode()),
                                   "max_tokens": b["max_tokens"]} for b in bodies])
        else:
            start = threading.Barrier(concurrency + 1)

            def one(body: dict) -> dict:
                start.wait()
                return _stream(url, body, api_key, min(timeout_s, remaining))

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(one, body) for body in bodies]
                began = time.perf_counter()
                start.wait()
                records = [f.result() for f in futures]
                wall = time.perf_counter() - began
            for record in records:
                violations = []
                if not record["error"]:
                    for name in ("prompt_tokens", "completion_tokens"):
                        low, high = case[name]
                        if not low <= record["usage"][name] <= high:
                            violations.append(name + "_outside_bounds")
                    if record["cached_tokens"]:
                        violations.append("prefix_cache_hit")
                    if mode == "long":
                        offsets = [record["response_text"].find(n) for n in case["needles"]]
                        record["needle_hits"] = sum(p >= 0 for p in offsets)
                        record["needles_in_order"] = min(offsets) >= 0 and offsets == sorted(offsets)
                        if not record["needles_in_order"]:
                            violations.append("needle_recall_or_order")
                record["protocol_violations"] = violations
            valid = all(not r["error"] and not r["protocol_violations"] for r in records)
            tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in records)
            wave.update(wall_s=wall, completion_tokens=tokens, valid=valid,
                        observed_aggregate_tok_s=tokens / wall,
                        aggregate_tok_s=tokens / wall if valid else None,
                        ttft_s=_stats([r["ttft_s"] for r in records if r["ttft_s"] is not None]),
                        per_stream_tok_s=_stats([r["per_stream_tok_s"] for r in records if r["per_stream_tok_s"] is not None]),
                        requests=records)
        result["waves"].append(wave)
    if execute:
        result["complete"] = len(result["waves"]) == len(waves)
        result["campaign_wall_s"] = time.perf_counter() - campaign_start
        summaries = []
        for n in levels:
            group = [w for w in result["waves"] if w["concurrency"] == n]
            wall = sum(w["wall_s"] for w in group)
            tokens = sum(w["completion_tokens"] for w in group)
            valid = len(group) == repeats * len(cases) and all(w["valid"] for w in group)
            summaries.append({"concurrency": n, "valid": valid, "wall_s": wall,
                              "completion_tokens": tokens,
                              "aggregate_tok_s": tokens / wall if valid else None,
                              "category_mean_aggregate_tok_s": statistics.fmean(w["aggregate_tok_s"] for w in group) if valid else None})
        result["summaries"] = summaries
        result["valid"] = all(s["valid"] for s in summaries)
    return result
