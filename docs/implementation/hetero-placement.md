# Heterogeneous part placement

Status: planning, descriptors and engine contract implemented and tested with fixtures (ADR-0055).
A two-stage layer pipeline candidate can be deployed as engine processes from per-process
manifests: [pipeline-deployment.md](pipeline-deployment.md) (ADR-0056), tested with fake engines.
A `dense+hot` candidate reads a hot-expert placement file: [hot-expert-placement.md](hot-expert-placement.md).

## Commands

```bash
# Rank placements for a model on a hardware description
PYTHONPATH=src python -m tensormeld plan-hetero MODEL.parts.json HARDWARE.json \
  [--prompt 8192] [--context N] [--output 256] [--rows 1] [--slots 1] [--draft] \
  [--objective latency|decode|prefill|throughput] [--unavailable DEVICE]... \
  [--link LINK_ID=BYTES_PER_S:LATENCY_US]... [--top-k 5] [--out plan.json]

# Measure a link (server on the far node, client here; same file runs without the package)
python3 src/tensormeld/link_probe.py serve --bind 10.0.0.2 --port 18300     # prints a token
python  src/tensormeld/link_probe.py run --host 10.0.0.2 --port 18300 --token TOKEN

# Regenerate the example descriptors
python examples/hetero/make_examples.py
```

Exit codes of `plan-hetero`: 0 = a feasible placement, 2 = none, 1 = invalid input.

## Model parts (`tensormeld/model-parts-v1`)

| field | meaning |
|---|---|
| `layers`, `stream_bytes_per_row`, `token_bytes` | chain length; bytes of the residual stream per row when a layer's home changes; token id bytes fed back to the embedding |
| `parts[].role` | `anchor` (runs on the layer's home: attention, mixers, norms, routers), `branch` (input from the home, partial result back: experts, FFN), `lookup` (token-id indexed; prefetchable unless at layer -1/0), `head` (layer = `layers`), `draft` (optional) |
| `weight_bytes`, `storage_bytes` | resident bytes; disk-backed table bytes (the node must list the model in `model_files`) |
| `decode.read_bytes`, `decode.read_bytes_extra_row`, `decode.ops_per_row` | bytes read for one row, marginal bytes per extra row (batch / verify window), operations per row |
| `prefill.ops_per_row`, `prefill.rows` | operations per prompt row; `all`, `last` or `{"max": N}` (layers that only see a suffix) |
| `state.bytes_per_token`, `state.bytes_fixed` | persistent request state (KV, recurrent state) per token and per request |
| `io.in_bytes_per_row`, `io.out_bytes_per_row`, `io.combine` | activation bytes into a branch, result bytes out (or a sharded anchor's exchange); `combine: per-shard` when an exact summation order needs every shard's partial at the home (default `reduce`: the group returns one sum) |
| `shardable`, `replicated_fraction` | weights, bytes and operations split across a device group (one exchange of `out_bytes_per_row` per call); `replicated_fraction` is the share every shard repeats (tensor parallelism keeps small projections whole) |
| `requires` | capability strings an engine must implement on the device |
| `cuts[]` | `after_layer`, `side_bytes_per_token`: state later layers need from earlier ones when the home changes there |

## Hardware (`tensormeld/hetero-hardware-v1`)

| object | fields |
|---|---|
| node | `id`, `disk_free_bytes`, `disk_bytes_per_s`, `model_files` |
| device | `id`, `node`, `kind`, `backend`, `memory_bytes`, `reserved_bytes`, `mem_bytes_per_s` (effective), `prefill_ops_per_s` (effective) and optional `prefill_rates` per capability, `decode_ops_per_s`, `step_overhead_us` (per layer step), `crossing_overhead_us` (host-side cost of one remote exchange), `availability`, `supports` (`["*"]` or capabilities), `provenance` |
| link | `id`, `a`, `b` (nodes), `bytes_per_s`, `latency_us` (one way), `shared_group`, `provenance` |
| group | `id`, `devices`, `link` (the intra-group link), `supports` (capabilities the engine runs sharded across the group) |

Availability: `available`; `preemptible` (the plan without it is reported too); `prefill-only`
(a lent device: it may hold weights and run prefill, never decode); `unavailable`.

## Cost model

Decode, per step of `rows` rows:

- **Layer walk:** for each layer, the anchors run on the home: the larger of bytes ÷
  bandwidth and ops ÷ throughput, plus the device's step floor. A sharded anchor also
  pays one exchange on the group link.
- **Branches:** branches on the home add local time. A branch on another set pays a
  round trip: route latency + payload ÷ rate in both directions, forwarding and
  combining inside a group, and the crossing overhead on both ends. Local and remote
  branches overlap (`max`).
- **Home moves:** a home change moves `stream_bytes_per_row × rows`.
- **Head and lookups:** the head adds its time and a token feedback when it is away
  from the embedding. A lookup at layer ≥ 1 on another set costs bytes but no latency.

Prefill, prompt of `P` tokens:

- the same walk with each part's prompt rows;
- weights re-read once per chunk;
- branch payloads of every row, and side bytes at the cuts;
- no overlap between layers.

Prefill-decode: the state of every part whose set differs between the two placements
moves once (`handoff`). The memory of both placements is counted together.

Candidates:

- the whole model on each device or group, and the same with lookups moved to host
  memory on the same node;
- replicas: a greedy disjoint cover;
- a two-stage layer pipeline at every cut;
- part splits for each (fast, large) pair, one per number of leading layers whose
  branches also fit on the fast set, plus a density-greedy fill;
- prefill-decode pairs of the best prefill and decode placements.

Ranking objectives:

| objective | ranks by |
|---|---|
| `latency` | TTFT + output × step |
| `decode` | step per row |
| `prefill` | TTFT |
| `throughput` | aggregate tokens/s |

Every candidate lists its device `sets`; a `hetero-pipeline` candidate also has a `layout`:
its stages in stream order with the layer range `[first, end)`, the exact part ids, the parts
of its layers placed elsewhere (`remote_parts`) and the parts it runs for another stage
(`serves_parts`). `deploy-pipeline` derives the process manifests from it.

The result lists `unmeasured_inputs`. It is never qualified or executable. The model is an
explainable heuristic, not a strict bound: it omits link and PCIe contention, lookup deadlines
and workspace peaks, and it serialises layers.

## Prefill length routing (`prefill_route`, `tensormeld/prefill-route-v1`)

A prefill-decode plan pays a fixed price to prefill elsewhere (offload device setup and weight streaming, state export,
wire, import on the decode set). Short prompts are faster where they decode; long ones on the offload placement. The
plan carries `prefill_route` for the best prefill-decode pair that shares the best placement's decode side (or `null`):

- `local_max_tokens`: a request whose effective length is at or below it prefills on the decode placement (`local`);
  longer ones prefill on the offload placement and move their state once (`offload`). Offload down -> local.
- effective length = prompt tokens - tokens of a prefix-cache hit on the decode placement: a hit leaves only the
  suffix to prefill locally, while the offload placement would redo (or first receive) the cached prefix. No hit:
  the threshold. Hit with a measured receipt: the plan carries `curves` and the TTFTs are compared directly,
  `local(suffix) + cached_s_per_token * cached` against `offload(prompt)` (a long suffix behind a long cached prefix
  can stay local above the threshold). Hit without curves: the threshold on the suffix.
- `source`: `configured` (`--prefill-local-max-tokens N`) > `measured` (crossover of a `tensormeld/ttft-curve-v1`
  receipt given with `--ttft-receipt FILE`: end-to-end TTFT points per prompt length for both routes, piecewise linear,
  end segments extended) > `model` (crossover of this cost model). All three crossovers are reported.
- The cost model has no term for weights streamed per prompt to an offload device that cannot hold them, so its
  crossover can be far too low; use a measured receipt (example: `examples/hetero/ds41-ttft-receipt.json`).
- `tensormeld.prefill_route.PrefillRoute.decide(prompt, cached, offload_up)` is the per-request rule for the process
  that admits requests.

## Engine contract (`tensormeld/engine-describe-v1`)

`run_describe(executable, approved_sha256=...)` hashes the file, runs it with fixed argv
(`--describe-json`), an empty environment (PATH/SYSTEMROOT), no stdin, a time limit and a
4 MiB output bound, checks that the reported `executable_sha256` matches, and validates:

- `engine`: name, revision, executable sha256, backend;
- `capabilities` (what one device of that backend can run), `group_modes`
  (world, capabilities, transports);
- `roles`: `whole`, `prefill`, `decode`, `pipeline-stage` (runs a layer range on an incoming
  stream and returns the stream or the token), `parts-host` (owns the stream and anchors and
  calls remote branches), `parts-worker` (serves branch parts for a layer range);
- `state`: `export`, `import`, `format`, `transports`, `exact`. `prefill` requires `export`
  and `decode` requires `import`;
- `parts_rpc`: `transports`, `max_rows`, `encodings`. Required for `parts-worker`;
- `models[]`: one parts descriptor per model file.

A `parts-worker` call carries, per layer and step: layer id, rows, the branch input
encoding (for example 8-bit codes + scales + expert ids/weights) and returns the partial
sum in a declared encoding. The RPC contract must fix the summation order on the host so a
split can reproduce the engine group's arithmetic exactly when both sides use the same
kernels; that part of the contract is specified here but not yet versioned in code.

## Examples

`examples/hetero/` holds a 40-layer sparse MoE with a causal encoder-decoder split, a
48-layer hybrid linear/full-attention MoE and a lab hardware profile (one discrete GPU
host with host memory, two unified-memory nodes, 1 GbE switch, USB4 between the nodes).
Their numbers are estimates from shapes and engine receipts, marked as such.

## Tests

`tests/test_hetero_placement.py`:

- descriptor parsing and rejection;
- routes and what-if overrides;
- a hand-computed decode step;
- dense parts on the fast device and experts on the large one;
- slow links choose fewer crossings;
- unavailable and preemptible devices, with the fallback;
- `prefill-only` never decodes;
- capability and group restrictions;
- disk tables stay where the file is;
- replicas;
- request validation;
- the examples and the CLI;
- link probe over loopback with its token and address checks.

`tests/test_engine_describe.py`:

- contract rules;
- the artifact hash check through a fake engine script;
- two fake engines with different capabilities driving the planner.

## Not implemented

- executing a split other than a two-stage pipeline: launching `parts-host` / `parts-worker` /
  `prefill` / `decode` roles and routing to a role pair (the pipeline FRONT/TAIL and its
  fallback are deployed by `deploy-pipeline`, ADR-0056);
- N-stage pipelines;
- overlap of transfer and compute;
- shared-link (NIC/PCIe) contention, lookup deadlines and workspace peaks;
- binding a placement to one engine instance and group mode in the planner itself (the planner still
  sees per-device capabilities; `deploy-pipeline` checks each stage against the one engine that runs it);
- measured receipts replacing the model terms.
