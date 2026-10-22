# ADR-0055 — Models are placed by part across unlike devices

Status: **accepted** (planning and contracts; no executable split yet)  
Date: 2026-10-20

## Context

An installation can mix devices that are good at different things: a discrete GPU
with high memory bandwidth and dense matrix throughput but limited capacity, and
unified-memory nodes with large capacity but a fraction of the bandwidth and compute.
A sparse mixture-of-experts model needs both: its dense parts (attention, shared
experts, projections, the output head) are read in full for every token and dominate
decode bytes, while its routed experts hold most of the weights but each token
touches only a few of them. Hybrid models add recurrent state; some models have
layers that see only a bounded suffix of the prompt during prefill; lookup tables
(embeddings, n-gram memories) are indexed by token ids and can be fetched ahead.

ADR-0054 covers models that fit on one device set (replicas). Hard-coding one split
per model and per hardware set would have to be redone for every new model, device
or link. The links also differ by orders of magnitude (a 1 GbE switch, a 10 GbE
link, USB4 host-to-host), and the best split changes with them.

## Decision

1. **Model parts descriptor** `tensormeld/model-parts-v1`. A model is a chain of
   layers. Every part declares its role (`anchor` runs where the layer's residual
   stream lives; `branch` takes an input from the stream and returns a partial result;
   `lookup` depends only on token ids; `head`; optional `draft`) and its costs per row:
   - resident bytes and disk-backed table bytes;
   - decode bytes read, including the marginal bytes per extra row;
   - operations, for decode and for prefill;
   - prefill rows: `all`, `last` or a bounded suffix;
   - persistent state per token and per request;
   - activation bytes at its boundary;
   - whether it can be sharded across a device group, and the share every shard repeats;
   - for a branch, whether its group returns one sum or every shard's partial, which an exact
     summation order needs;
   - the capabilities an engine needs to run it.

   Cuts may declare side bytes per prompt token, for state that later layers need
   from earlier ones. Engines emit this descriptor; hand-written ones are planning
   inputs only.
2. **Hardware descriptor** `tensormeld/hetero-hardware-v1`.
   - Each node declares the model files it holds.
   - Each device declares memory and reserve, effective bandwidth, prefill throughput
     (optionally per capability), a per-layer step floor, a per-crossing host overhead,
     availability (`available`, `preemptible`, `prefill-only`, `unavailable`), the
     capabilities its engine implements, and provenance.
   - Each link declares rate and one-way latency, a shared group and provenance.
   - Each device group declares the capabilities its engine can run sharded.
3. **Placement planner** (`tensormeld plan-hetero`). It generates candidates of
   every mode and returns them ranked, each with its full cost breakdown:
   - **whole model** on one device or group, optionally with lookups in host memory
     next to the device;
   - **replicas** on disjoint sets;
   - **layer pipelines** at any legal cut;
   - **part splits**: the hottest bytes per stored byte on the fastest set, then
     branch parts layer by layer while memory lasts;
   - **prefill-decode** pairs, in which the request state moves once.

   The breakdown covers decode step time, crossings and bytes per link, prefill time,
   handoff, memory per device and unmeasured inputs. When the best plan uses a
   preemptible device, the planner also returns the plan without it, and whether
   both sets of weights can stay resident together.
4. **Engine contract** `tensormeld/engine-describe-v1` (answer to `--describe-json`
   of an approved executable). It declares:
   - backend and capabilities;
   - group modes;
   - roles: `whole`, `prefill` (needs state export), `decode` (needs state import),
     `pipeline-stage`, `parts-host`, `parts-worker`;
   - request-state export/import (format, transports, exactness);
   - the per-layer branch protocol of a `parts-worker`;
   - the parts descriptor of every model it serves.

   Different devices may run different engines; the planner sees only capabilities.
5. **Link evidence.** `tensormeld link-probe` (stdlib, one bounded session,
   loopback or private addresses only, random session token) measures round trips
   over a payload ladder plus bulk rate. It produces `latency_us` and `bytes_per_s`
   values to copy into a link with `provenance: measured`.

## Consequences

- New models and devices are new descriptors, not new code paths. Capability
  strings restrict where a part can run, so an engine backend can grow part by part.
- TensorMeld stays out of the token path. A plan names placements and roles; the
  engines carry activations, sum partial results and move state over their own
  transports.
- The cost model is an explainable heuristic, not a strict bound: it serialises
  layers in prefill and uses store-and-forward routes, but it does not serialise links
  in a shared group or model PCIe contention, lookup deadlines or workspace peaks. Any term
  can be replaced by a measured receipt. Results are always `qualified: false,
  executable: false`.
- Executing a split needs engine work for each role. The first executable split
  will be validated against the whole-model engine group: same tokens when the
  arithmetic is identical; otherwise a new numerical identity with its own quality
  gate.
