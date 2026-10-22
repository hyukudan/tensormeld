# ADR-0051 — Performance-aware legal-unit ranking is an ordered-chain upper bound

Status: **accepted**  
Date: 2026-10-10

## Context

TensorMeld has legal model units, explicit per-unit compute/memory evidence and
directional transfer-path upper bounds. These are enough to compare the ordered
compute chain plus boundary movement, but they are not enough to claim autoregressive
token latency, TTFT, throughput or executable distributed inference.

Missing terms include token feedback/sampling, prefill/decode phase differences,
overlap, queueing, shared-link contention and native finer-grained execution proof.

## Decision

TensorMeld adds a bounded advisory planner with objective:

```text
ordered_unit_chain_upper_bound_us =
    sum(explicit unit compute_us)
  + sum(explicit directed boundary transfer upper_bound_us)
```

The planner:

- uses only adapter-declared legal units and cut boundaries;
- uses only explicit legal-unit device cost evidence;
- uses only conservative directional path buckets;
- rejects an owner-changing candidate when its boundary payload has no covered path;
- counts hard-resident + reclaimable tensor bytes in full;
- adds persistent state across assigned units;
- takes workspace and staging maxima per device/pool, then conservatively sums those
  per-device maxima when several devices share one physical pool;
- intersects all memory with existing static physical-pool owner budgets;
- preserves required/min/max device/node policy and bounded-search semantics;
- returns `SEARCH_INCOMPLETE`, never physical infeasibility, when the search budget expires.

Ranking is by ordered-chain upper bound, then device count, node count and pool
utilization. Results remain `qualified=false` and `executable=false`.

## Consequences

- a faster helper can win only when its explicit compute saving exceeds explicit
  directional transfer cost and all physical-pool memory classes fit;
- a missing path or uncovered payload can invalidate an otherwise legal memory split;
- the objective must not be described as token latency, TTFT or throughput;
- whole-block execution remains the validated executable baseline;
- future work may add feedback/sampling/phase/overlap evidence as distinct contracts
  before stronger performance claims are allowed.
