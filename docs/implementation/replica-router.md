# Replica router

`tensormeld serve-replicas CONFIG` runs a least-loaded HTTP router in front of
whole-model engine replicas (ADR-0054). It runs until SIGINT/SIGTERM and prints
`tensormeld router: listening on ADDR:PORT` on stderr; at exit it prints the final
status snapshot as JSON.

## Configuration

```json
{
  "schema": "tensormeld/replica-router-v1",
  "listen": {"address": "127.0.0.1", "port": 18190},
  "replicas": [
    {"id": "n1", "node": "node1", "address": "127.0.0.1", "port": 18181, "max_inflight": 1},
    {"id": "n2", "node": "node2", "address": "127.0.0.1", "port": 18182, "max_inflight": 1}
  ],
  "health": {"interval_s": 1, "timeout_s": 2, "failures_to_down": 2},
  "limits": {"queue_timeout_s": 600, "read_timeout_s": 600}
}
```

| Field | Default | Meaning |
|---|---|---|
| `listen.address` / `port` | required | numeric loopback, or private with `api_key_file`; port `0` picks a free one |
| `api_key_file` | none | clients must send `Authorization: Bearer KEY` or `x-api-key: KEY` on API routes |
| `replicas[].address` / `port` | required | numeric loopback or private endpoint of one engine server |
| `replicas[].max_inflight` | required | 1..64: requests the replica runs at once (its batch slots) |
| `replicas[].api_key_file` | none | key sent to that replica; client keys are never forwarded |
| `health.interval_s` / `timeout_s` | 2 / 2 | `GET /health` probe period and timeout |
| `health.failures_to_down` | 2 | consecutive failed probes before leaving rotation |
| `limits.connect_timeout_s` | 5 | upstream connect timeout |
| `limits.read_timeout_s` | 900 | upstream idle read timeout |
| `limits.queue_timeout_s` | 600 | longest wait for a free replica slot |
| `limits.max_waiting` | 64 | requests waiting for a slot before 503 |
| `limits.max_connections` | 128 | open client connections before 503 |
| `limits.max_request_bytes` | 8 MiB | request body bound (`Content-Length` required) |
| `affinity.mode` | `off` | `prefix`: conversations go back to the replica that served their longest known prefix |
| `affinity.max_entries` | 65536 | bounded LRU of prefix digests |
| `affinity.max_extra_inflight` | 1 | affinity yields when the preferred replica has this many more requests than the least loaded free one |
| `prefill.policy` | `nodes` | `nodes`: prefill on the decode replica; `auto`: cost model (below); `rtx`: always offload while a service is up |
| `prefill.node_profile` | required by `auto` | `{"prefill_curve": [[tokens, seconds], ...], "restore_s"}`: the replicas' measured prefill time |
| `prefill.decode_s_per_token` | `[[1, 1/46], [8, 1/6]]` | per-request decode seconds per token by active requests (slot-wait estimates) |
| `prefill.default_output_tokens` | 256 | output length assumed for estimates (capped by the request's `max_tokens`) |
| `prefill.cache.block_tokens` / `max_entries` | 2048 / 65536 | the engine's checkpoint block (its `--prefix-cache-every`) and the mirror's LRU bound |
| `prefill.margin_s` / `margin_frac` | 0.2 / 0.10 | an offload must save more than max(margin_s, margin_frac x local cost) |
| `prefill.min_tokens_idle` | 2048 | below this, a replica without active requests keeps the prefill |
| `prefill.services[]` | none | prefill services: replica fields + `profile` (measured curves keyed by GPU and model, or `seed:<model>:<vram_gib>`) |
| `prefill.handoff_path` / `tokenize_path` / `prefill_path` | `/admin/import-state`, `/tokenize`, `/v1/prefill` | routes of the generic contract |
| `prefill.wait_s` | 600 | longest wait in a service's queue before an offload falls back |

A replica on another node either keeps its server on that node's loopback behind an
authenticated tunnel (for example SSH local forwarding), or listens on the private LAN
with its own API key (`replicas[].api_key_file`; the engine's `/health` stays open for
the probes).

Example for two nodes holding one complete Qwen model each:
`examples/qwen-replicas-router.json`.

## Routes

| Route | Behavior |
|---|---|
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/messages` | one in-flight slot on the chosen replica; body and response relayed unchanged; `X-TensorMeld-Replica` names the replica, `X-TensorMeld-Prefill` where the prompt was prefilled (`local`, `cache`, `offload:<service>`, `fallback:<reason>`) when prefill placement is configured |
| `POST /tokenize`, `/detokenize`, `GET /v1/models` | forwarded to a healthy replica without a slot |
| `GET /health`, `/v1/health` | 200 with at least one replica up, else 503 |
| `GET /api/status` | `tensormeld/replica-router-status-v1`: per replica state, in-flight, assigned, completed, failed, last probe; affinity hits/yields/none; prefill placements and services |
| `GET /metrics` | Prometheus text: per-replica up/in-flight/assigned/completed/failed, waiting, rejected |

`/health`, `/api/status` and `/metrics` need no key.

## Selection

Generation requests take the healthy replica with the lowest `inflight/max_inflight`,
then the least recently assigned, then config order. A request that finds every
healthy replica full waits (bounded) until a slot frees; with no healthy replica it
gets 503 at once. A refused connection marks the replica down and the request moves
to another replica; once a request byte has been sent it is never replayed.

### Session affinity (`affinity.mode = "prefix"`)

Every generation request yields affinity keys, most specific first: an explicit session
id (`X-Session-Id` header, OpenAI `user`, Anthropic `metadata.user_id`), then one digest
per message over everything before it (model, system prompt and tools included) for chat
and messages routes, or one digest per 512 characters / 256 token ids for completions.
A later turn of a conversation resends the earlier messages, so it shares their digests.
The router sends the request to the replica recorded for its most specific known key,
provided that replica is up, admitting, has a free slot and is at most
`max_extra_inflight` requests above the least-loaded free replica; otherwise it takes the
least-loaded replica and records the conversation there. Affinity only changes where a
request runs, never what it returns. It exists so that the replica's prefix state cache
can skip the repeated part of the prompt.

### Prefill placement and cost routing (`prefill`)

With `policy = "auto"` a chat or completions request is placed by a cost model
(`route_cost.py`, ADR-0057):

1. Its token ids come from a replica's `/tokenize` (the same chat template the generation
   uses), or as given for token prompts.
2. Each routable replica r is a local candidate: `J = W_r + R + F(U_r) + N_r * F(U_r)`.
   - `W_r` is the predicted wait for a free slot (0 with one free).
   - `U_r = P - C_r` are the rows left after the cached prefix `C_r`. That is the longest
     checkpoint boundary recorded in r's cache mirror: chain digests of the token blocks of
     prompts r prefilled itself.
   - `F` is the measured prefill curve, `R` the restore cost.
   - `N_r * F` is the pause the prefill imposes on r's active decodes.
3. Each healthy service s with each replica r is an offload candidate:
   `J = max(Q_s + S + F_s(P) + H_s(P), W_r) + N_r * I`.
   - `Q_s` is the service's pending work in seconds.
   - The service prefills the whole prompt (it has no cache); `H_s` is the handoff and `I` the
     import pause.
   - Below `min_tokens_idle`, an idle replica is not offloaded to.
4. The lowest local J wins unless an offload saves more than max(`margin_s`,
   `margin_frac` x J). The output length cancels: both paths end with the same decode.
5. An offload reserves the service and waits in its FIFO: a busy service is a cost, not a
   refusal. It then sends `POST /v1/prefill {"tokens", "handoff": "http://<replica><handoff_path>",
   "session"}`. Only after the state reached the replica does it take the decode slot there
   (no slot is held while waiting for the service) and forward the original request.
6. Any offload failure (service down, HTTP error, wait over `wait_s`) prefills locally on the
   chosen replica: placement never fails a request.

With no service configured, `auto` still picks the replica by that cost: cache-aware and
work-aware routing for whole-model replicas. `X-TensorMeld-Prefill` names the place (`local`,
`cache`, `offload:<service>`, `fallback:<reason>`); `X-TensorMeld-Route-Cost` gives the
compared costs. Profiles are measured curves, keyed by GPU (name, VRAM, pinned host memory,
PCIe) and model; the DeepSeek RTX PRO VRAM-cap measurements ship as seed profiles. The
mirror only approximates the engine's cache (its LRU and capacity): a stale entry costs one
mispredicted placement, never a wrong answer.

The contract is engine-neutral: any model whose decode server implements `/tokenize` and
a state import route, and whose prefill service implements `/health` and `/v1/prefill`,
plugs in without router changes.

### Qwen prefill offload to the RTX service (`prefill_offload`)

Opt-in and off by default: without a `prefill_offload` object the router forwards exactly
as described above (no tokenize call, no extra header, no extra status or metrics). It is
the receipt-driven rule for the Qwen replicas. A request still runs on **one** node
replica. When the rule offloads, the RTX PRO prefill service prefills the prompt for the
replica the router already picked:

1. The router acquires and **holds** the replica first (ordinary selection and affinity).
2. It gets the prompt's token ids:
   - token-id prompts are taken as given;
   - text completions (`"prompt": "..."` or `["..."]`) go through the held replica's
     `/tokenize` as `{"prompt": text}`;
   - chat bodies go to `/tokenize` unchanged. The Qwen serve applies `chat_prompt(body)`,
     the same template as the generation, so the ids are exact.
3. It decides the route (`prefill_offload.decide`).
4. On offload it checks the replica's capability with `GET <import_url>/layout`. This route
   exists only when the replica runs `--rtx-state-import`.
5. It sends `POST <endpoint>/v1/prefill {"tokens", "handoff": "<import_url>", "reserve_tokens", "session"}`
   and waits for the reply. `reserve_tokens` is the request's `max_completion_tokens` or
   `max_tokens`, else `default_reserve_tokens`. The service imports the state into that
   replica and answers with `position`, which must equal `tokens - 1`.
6. It forwards the **original request unchanged** to the same replica. After a successful
   import the replica restores the parked state for a prompt that starts with those tokens.
   After any failure (connect or I/O error, timeout, HTTP 502/503/413, position mismatch)
   the replica simply prefills itself. The offload never fails a request, and a stale
   parked state is harmless because the replica checks it against the prompt.

`/v1/messages` (Anthropic body) is never offloaded: the router cannot ask the replica to
tokenize it the way the generation would. If the held replica's connection is refused, the
request moves to another replica as usual, and the rule runs again for that replica.

```json
"prefill_offload": {
  "endpoint": "http://192.0.2.20:18700",
  "route": "hetero/qwen38-ttft-receipt.json",
  "import_path": "/admin/rtx-import",
  "timeout_s": 120,
  "min_tokens": 1024
}
```

The Qwen receipt is measured on the real path (2026-10-24, 1 GbE, identical tokens on both routes): node replica
1.80 / 5.20 / 10.46 / 21.43 / 44.02 s and RTX offload (service defaults: release by layer, bp-v1, 2 connections)
1.36 / 1.92 / 2.94 / 4.95 / 8.99 s at 1K / 4K / 8K / 16K / 32K tokens. Offload already wins at 1K; the curves'
extrapolated crossover is ~550 tokens, so `min_tokens` (1024) is the effective floor.

| Field | Default | Meaning |
|---|---|---|
| `endpoint` | required | `http://ADDR:PORT` of the RTX prefill service: numeric loopback or private address, no path |
| `route` | required | a `tensormeld/ttft-curve-v1` receipt (path relative to the config file, or inline), or a plan's `tensormeld/prefill-route-v1` section |
| `local_max_tokens` | from `route` | operator override of the threshold (source `configured`) |
| `min_tokens` | 1024 | never offload a prompt shorter than this |
| `import_path` | `/admin/rtx-import` | handoff URL per replica: `http://<replica address>:<port><import_path>`, unless the replica entry has `import_url` |
| `replicas[].import_url` | derived | explicit handoff URL for that replica (numeric private address); only allowed with `prefill_offload` |
| `api_key_file` | none | key sent to the service as `Authorization: Bearer` |
| `handoff_codec` | none (service default) | `identity` or `bp-v1` (lossless byte-plane + Huffman chunks, used only when the replica lists the codec) |
| `cooperative_local_rows` | none | `[[min_tokens, q], ...]`: for prompts of at least `min_tokens`, the service also posts the replica's own completion of prompt[0, q] and sends only rows >= q; the replica needs its prefix state cache (`--prefix-cache-host-bytes`) and q a multiple of its block (2048 in the production server configuration) |
| `timeout_s` / `health_timeout_s` | 120 / 1 | prefill call timeout; `/health` and `/layout` probe timeout |
| `default_reserve_tokens` | `prefill.default_output_tokens` (256) | `reserve_tokens` when the request gives no `max_tokens` |
| `prefill_path` / `health_path` | `/v1/prefill`, `/health` | service routes |

`prefill_offload` needs `prefill.policy = "nodes"`: it does not combine with the cost
placement above.

**Decision rule.** Let `n` be the number of prompt tokens. The rule is evaluated in order:

1. `n < min_tokens`: local.
2. `n <= local_max_tokens`: local. The threshold is the override if one is given. Otherwise
   it is the `local_max_tokens` of the route section, or the measured crossover of the
   receipt curves. Both curves are end-to-end TTFT against prompt tokens, so the
   break-even is the point where the curves cross. Up to this step no network call is made.
3. Otherwise the router asks the service's `GET /health`:
   - no answer: local (`service_unreachable`);
   - `status != "ok"`: local (`service_failed`);
   - `n > max_positions`: local (`service_max_positions`).
4. With `ahead = busy + queued` jobs already at the service, the queue is priced with the
   curves:
   - `ahead == 0`: offload;
   - otherwise offload iff `offload(n) + q_rtx < local(n) + q_local`, where
     `q_rtx = ahead * offload(n)` and `q_local = 0`.

   `q_rtx` counts every job ahead as one prefill of the same length, because the service
   does not report the lengths it holds. `q_local` is 0 because the held replica runs only
   this request (`max_inflight` 1). With a threshold-only route section (no curves), a busy
   service always means local.

**Records.** `X-TensorMeld-Prefill` takes one of three forms:
- `offload:rtx`;
- `local:<reason>`, where the reason is one of `below_min_tokens`, `below_threshold`,
  `service_unreachable`, `service_failed`, `service_max_positions`, `service_queue`,
  `service_busy`, `replica_no_import`, `no_tokens`;
- `fallback:<reason>`, where the reason is one of `http502`, `http503`, `http413`,
  `timeout`, `io`, `position_mismatch`.

`/api/status` has a `prefill_offload` member with:
- counts per route (local, offload, fallback) and per reason;
- the total RTX wall seconds;
- the last 64 per-request records. Each record holds the replica, tokens, route, reason,
  fallback, handoff, session, `rtx_wall_s`, the service's own `seconds`
  (queue/prefill/transfer/import/total), the curve values when they were used, and
  `router_s`.

`/metrics` adds `tensormeld_router_prefill_offload_total{route}`,
`..._reason_total{reason}` and `..._rtx_seconds_total`. `serve-replicas` prints each
record as a `tensormeld router: prefill_offload {...}` line on stderr.

**Receipt.** `examples/hetero/qwen38-ttft-receipt.json` has two measured curves (2026-10-24, one run per
length, tokens identical on both routes):
- `local`: the node replica's own prefill on the production server the router forwards to (2,048-row prompt
  tiles): 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s at 1K / 4K / 8K / 16K / 32K tokens;
- `offload`: end-to-end TTFT through the RTX prefill service with its defaults, including the state transfer
  over 1 GbE, the replica's import and its last row: 1.36 / 1.92 / 2.94 / 4.95 / 8.99 s.

Both curves are `measured`, so the receipt's provenance is `measured`. The router still records `rtx_wall_s` and
the service's `seconds` per request, so the curves can be re-checked against live traffic. An example router
config is in `examples/qwen-replicas-router-rtx-prefill.json`.

**Limits.**
- There is no prefix-cache term. The replica's restorable prefix is unknown to the router,
  so the rule is the threshold on the full prompt. A long prompt that is mostly cached on
  the replica can be offloaded needlessly. The DeepSeek rule's `cached_tokens` path needs a
  Qwen serve endpoint that reports the hit length.
- The queue price is coarse: there are no per-job lengths, and a running job counts as a
  full one.
- `q_local = 0` assumes `max_inflight = 1`.
- The held replica sits idle while the service prefills. Requests that arrive meanwhile go
  to the other replica or wait.
- The service reply is trusted only for `position`. Correctness rests on the replica's own
  check of the parked state against the prompt.
- The curves come from one measured run per length, and the decision is made per request.
  There is no feedback loop that updates the curves from the recorded seconds.

## Load client

`tensormeld bench-replicas --url URL --users N --turns T --max-tokens M --doc FILE`
runs N concurrent multi-turn conversations (each with its own document excerpt as the
system message, every turn resending the history, greedy) against an engine or the
router, and reports aggregate completion tokens per second, time to first token of first
and later turns (server queue + prefill when the engine reports them), the cached share
of later prompts, per-replica request counts and how many conversations stayed on one
replica (`tensormeld/replica-bench-v1`).

## Limits

### Optional active decode balancing

Set `balance_active_decodes: true` in the router configuration to prioritize the
smallest absolute number of outstanding generations among healthy replicas with
space. The router must be their only admission path. Counts conservatively include
prefill/service waits; no backend progress telemetry is inferred. Equal-capacity
replicas receive 2+2 outstanding requests at load four, and unequal slot limits
cannot turn that into 3+1 while both have space. Slot selection and increment are
atomic, including simultaneous cost-placed arrivals.

Within the minimum-load group, token-cache prefill J chooses the cheapest replica.
Message/session affinity alone is ignored in this mode: it is not evidence of a
restorable token block. The default cache mirror interval becomes 1024 tokens;
configure the worker's actual checkpoint interval to match, or override
`prefill.cache.block_tokens`. With no tokenization/profile, ordinary active-count
balancing still applies. The feature defaults off.

Balanced cost routing holds its decode slot while waiting for offloaded prefill.
This intentionally prevents concurrent arrivals from all choosing the same free
replica; its admission estimate includes that pending generation. Shared accelerator
ownership can additionally fence the service, as described in
[serving mode policy](serving-mode-policy.md).

- Router in-flight counts are the load signal: the router is assumed to be the only
  client of its replicas.
- Affinity keys are message prefixes, not token blocks: the engine's cache granularity
  (its block size) decides how much of a matching prefix is actually reused.
- The cache mirror learns from the router's own traffic; a replica restarted with an empty
  cache is mispredicted until its entries age out.
- Pending work is predicted from the profiles; the router does not yet read the engines'
  live progress.
- A client disconnect is noticed while relaying a response; a request that is still
  waiting for a slot or for its first upstream byte finishes on the replica.
- The router does not start, stop or drain engine processes.

## Tests

`tests/test_replica_router.py` runs fake engine servers in-process: config bounds,
least-loaded placement under concurrency, alternation on ties, streaming without
buffering, disconnect propagation, health recovery, passive failover, queue limits,
credential isolation, status and metrics, affinity keys and LRU, sticky conversations,
affinity yielding to load, failover and return, prefill offload with handoff before the
request, token prompts without tokenize, fallback on failing/busy/down services, cache
hits kept local, config validation, and the load client through the router.
`tests/test_prefill_offload.py` covers `prefill_offload` with a fake RTX service and fake
Qwen replicas: off forwards unchanged, short prompts stay local without a service call,
long prompts are prefilled for the held replica (its own handoff URL, held during the
prefill, import before the completion), chat tokenized on that replica with the same body,
502 / timeout / position-mismatch fallback, busy service priced by the curves, failed or
unreachable service, replica without the import route, receipt and config validation.
