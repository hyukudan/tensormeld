# Hot-expert placement and the `dense+hot` mode

Status: format reader, planner candidate and CLI implemented and tested with synthetic fixtures.
Nothing here is measured or executable: the candidate reports `qualified: false, executable: false`
like every `plan-hetero` result, and the remote-layer rates come from the placement file.

## Idea

A large MoE model keeps every dense part (attention, mixers, shared experts, head) on one fast
device, the *parts-host*, together with the routed experts that are selected most often. Experts
that do not fit are held by slower devices: a second GPU on the same node (*expert tier*), host
memory copied on demand (*expert cache*), and a set of *expert servers* reached over the network
that hold every expert. A layer whose top-k selection is fully resident stays local; a layer with
at least one cold expert pays a round trip to the servers that hold the cold ids.

## File format `tensormeld/hot-expert-placement-v1`

| key | meaning |
|---|---|
| `format` | `tensormeld/hot-expert-placement-v1`; readers reject any other value |
| `model` | `family` (must equal the model-parts `model.id`), `layers`, `experts`, `top_k`, `expert_bytes`; optional `file`, `model_identity` (16 hex or null), `canonical_split` (default `experts / 2`) |
| `dense` | `device` (the parts-host id) and `layers` (`"all"`; nothing else in v1) |
| `devices[]` | `id`, `kind` (`cuda`, `hip`, `host`), `role` (`parts-host`, `expert-tier`, `expert-cache`, `expert-server`), `budget_gib`, `reserve_gib`, `expert_gib` (default `budget_gib - reserve_gib`); the single expert server lists `servers[]` of `{node, experts: [first, end)}` ranges that tile `[0, experts)` |
| `layers[]` | one entry per layer, in order: `{"layer": l, "<device>": [ids...]}`; ids ascending, in range, disjoint across devices; a missing device holds nothing; the expert server is implicit and must not be listed |
| `combine` | `identity` of the combine order the engine keeps; `remote_return` |
| `fit` | free-form provenance of how the placement was produced |
| `expected` | producer's out-of-sample numbers per row pattern: `remote_layers_per_token`, `cold_vectors_per_token`, `tok_s`; `R1` is required |

Pattern names: `R<n>` or `W<n>` is a verify window of `n` rows, `B<n>`, `N<n>` or `batch-<n>` is a batch of
`n` independent rows, any other name may declare `rows`. Values are per decode step of that many
rows unless `unit` is `token`.

Optional keys read inside `expected.<pattern>` (additive within v1):

| key | meaning |
|---|---|
| `rows` | rows of a pattern whose name does not say it |
| `unit` | `step` (default) or `token` (the reader multiplies by `rows`) |
| `servers_per_remote_layer` | mean number of servers a remote layer calls; default: the expected number of distinct id ranges hit by the cold vectors, `n (1 - (1 - 1/n)^v)` |
| `ram_vectors_per_token` | expert-cache copies per step; without it cache copies are not priced and the plan says so |
| `tokens_per_window` | accepted tokens per step of a verify window (the file's `tok_s` uses it); the plan then also reports the planner's step at that acceptance (`tok_s_at_file_tokens_per_window`) |

Non-object values inside `expected` (for example a note on which data the numbers were evaluated on)
are metadata, not patterns. A device may declare its total `experts`; the reader checks it against
the layer lists.

Reader checks: the per-device memory rule
`sum(len(layers[l][dev])) * expert_bytes <= expert_gib * 2^30`; `cold_vectors >= remote_layers`;
`cold_vectors <= remote_layers * top_k * rows`; geometry against the model-parts descriptor
(`layers` and family always; `experts`, `top_k`, `expert_bytes` when the descriptor declares the
optional `moe` record). Unknown keys elsewhere are accepted and listed as `ignored_keys`.

## Planner

```bash
PYTHONPATH=src python -m tensormeld plan-hetero MODEL.parts.json HARDWARE.json \
  --mode dense+hot --placement placement.json \
  [--placement-device NAME=HW_DEVICE_ID]... [--host-copy-rate BYTES_PER_S] [other plan-hetero options]
```

Placement devices map to hardware devices by an explicit `--placement-device`, else by equal id,
else by role: the parts-host is the only hardware device whose backend equals its `kind`; a tier or
cache is the only remaining device of its backend on the parts-host's node; server `S/NODE` is the
only device of the server kind on `NODE`. Devices that hold no expert may stay unmapped; ambiguous
mappings are errors. The servers must form one device or one declared device group.

The `dense+hot` candidate:

* places every part on the parts-host except routed-expert parts (kind `routed-experts`), which
  live on the expert-server set; disk-backed lookups follow their tables as elsewhere;
* charges memory from the placement: hot copies on the parts-host, tier and cache, and on each
  server the bytes of its id range for every layer (not the descriptor's monolithic expert tensor);
* prices a decode step as the planner's own walk with every routed expert local on the parts-host,
  plus, per remote layer, only the part of the remote branch that the local branch does not cover:
  `max(0, send + server compute + return + crossing - local branch)`. The walk already takes
  `max(local, remote)` per layer, so the overlap is counted once. Send is the routed part's
  `in_bytes_per_row` per row to each involved server; return is `out_bytes_per_row` per cold
  vector; links that share a `shared_group` (one NIC) serialize their bytes;
* adds expert-cache copies at `--host-copy-rate` (declared, default 25 GB/s), not overlapped;
  prices tier experts as local (reported in `notes`);
* prefills on its base placement, or pairs with other candidates' prefill placements
  (`pd:...=>dense+hot`) with the union of resident bytes.

The plan's `dense_hot` block shows the placement summary, the device map and, per row pattern, the
file's `expected` numbers next to the planner's step time (`R1` always; every pattern with known
rows when `--rows` is above 1). The request's `--rows` picks the pattern with the same rows
(verify window before batch); without one, R1 rates are used and the result says it is optimistic.

Not modelled yet: per-layer remote probabilities (the file gives a mean), server queueing when
several rows hit one server, distinct-expert reuse inside a multi-row step, and tier-GPU transfer
cost.
