# 03 — Model, memory and placement

## Physical memory is the admission unit

A physical pool is the resource constraint. CPU and iGPU allocations may reference
the same pool. Workstation RAM and discrete VRAM are normally distinct pools.
The inventory must represent aliases; it must not add an APU's system memory and
GPU-reported shared memory as independent capacity.

For each pool, the planner requires a safe allocatable budget, not merely installed
capacity. Budgets exclude OS, display, driver and user-reserved headroom. For each
candidate it then accounts for resident weights, worst-request persistent state,
worker overhead, peak workspaces, graph arenas, activation and staging buffers,
communication slots, optional speculative state and load-time conversion peaks.

Both steady-state and preparation peaks must fit. A zero-copy alias counts once
only when the backend proves that it is the same physical allocation. Unified
memory is not a license to assume arbitrary external-DMA coherence. Windows driver
budgets and host pressure must be observed, not derived from marketing capacities.

Admission is rechecked at launch. A profile's free-memory observation is not a
reservation. A useful system preserves desktop usability rather than allocating
until the OS swaps or its display workload fails.

## Model manifest

The production manifest must bind exact model revision, tokenizer/chat template,
architecture version, all shard hashes, tensor names/shapes/encodings, embedding
and output-head ownership, tied aliases, routed and shared expert banks, dense
blocks, state layout and memory estimates per supported context/concurrency.

State may be standard KV, compressed latent attention state, sparse-attention
indexing state, convolution/recurrent state or architecture-specific combinations.
A generic parameter-count-times-bitwidth calculation is not a complete memory model.
The first token and long-context execution may have different workspace peaks.

Operators with unsupported encodings or state transitions invalidate a candidate.
Missing operator support must not silently fall back to a slow CPU path. A fallback
may be permitted only when explicit, budgeted, traced and accepted by the plan.

M0 uses an explicit list of ordered stages and supplied costs. It does not parse
model files, infer tied tensors, split MoE internals or validate real architectures.
A fixture stage may represent a group of actual layers; it is not an assertion that
any backend can cut at that boundary.

## Candidate placement families

**Whole blocks/stages:** contiguous complete blocks, including their experts and
persistent state, stay with their owner. This is the initial reference on ordinary
Ethernet because it reduces dependency crossings. A faster device may own fewer
or more parameters depending on cost and memory; equal fractions are not a goal.

**Attention/dense versus expert execution:** keep repeated dense work and state
with one owner while another owner executes resident experts. Return combined
contributions, not full expert weight banks. This is an experimental family whose
communication occurs at many layers and must be explicitly budgeted.

**Within-layer expert parallelism:** dispatch selected expert work to multiple
owners and combine results with correct routing weights and numerical semantics.
The critical path waits for the slowest required contribution. This differs from
putting each layer's entire expert bank on one companion.

**Tensor/operator parallelism:** permitted only when the communication backend,
operator coverage and measured synchronization costs justify it. A vendor-specific
collective API is not a universal cross-vendor execution protocol.

**Separate prefill/decode placement:** requires declaring weights required for each
phase, state migration size/cost, conversion and ownership. It must not be advertised
as free movement of a live conversation. It is not an M0/M3 requirement.

The initial planner should allocate complete blocks to valuable workstation VRAM
rather than insist that all experts leave the GPU. A small hot-expert cache can be
explored later, after stable block ownership is a proven baseline.

## Cost model and objectives

For serial concurrency-one decode the conservative reference is:

```text
T_token = sum(stage_compute)
        + sum(uncovered_transfer_cost)
        + uncovered_synchronization
        + sampling_and_feedback_cost
```

The complete cyclic dependency matters: after the final stage and sampling, the
next token must return to the owner of the first stage. Include control/staging
work that the real backend performs on this path. A host RTT is not GPU-ready RTT.

A transfer profile should be piecewise by payload size, direction and load. A first
approximation may use fixed latency plus bytes divided by measured useful bandwidth.
It must declare what is included in the fixed term and what is omitted. Profiles
must not derive effective bandwidth from nominal link speed alone.

Prefill requires different compute and tensor sizes. The production model must
compare chunking/microbatch choices and display their first-token impact separately.
Multiple requests introduce contention, shared-link scheduling and per-request
fairness; serial decode estimates must not be reused as throughput predictions.

Sequential block placement does not automatically parallelize a single autoregressive
stream. Pipeline utilization improvements require independent microbatches/requests
or other proven work overlap. No benefit may be claimed from hypothetical overlap.

## Search and ranking

The production planner should first prune incompatible kernels, insufficient pools
and unavailable directed routes. Then optimize useful device subsets, valid graph
cuts, device assignment/order, coordinator choices and qualified route policies.
The model's dependency order is immutable; changing device order does not reorder
model layers. Exact enumeration is restricted to a small bounded reference solver. It should return multiple Pareto candidates when prefill, decode,
capacity and power objectives conflict.

Always compare feasible baselines: local GPU(s), explicit local CPU offload,
companion-only, and useful eligible subsets with or without the local GPU. Counts
come from the registry and policy, never a fixed one/two-companion list. Compare the
same checkpoint/encoding/context before attributing a speed difference to hardware.
A changed quantization is an alternative workload, not a same-workload speedup.
A required local GPU is a user constraint, not a built-in assumption.

The engine must never select a plan solely because it uses every device. Rejections
must explain insufficient state headroom, missing kernels, unknown transfer costs,
unqualified layout or latency constraints. Unknown inputs must not masquerade as
measured zeros. Estimates and measured results use separate fields and confidence.

## Profiling and invalidation

Profiles bind device, backend binary hash, driver/runtime, exact model/encoding,
context class, active requests, phase, batch/microbatch, transfer route and power
profile. Thermal state and sample distributions must be recorded. A single timing
sample is not a sustainable performance claim.

Changes in topology, link negotiation, backend version, graph configuration or
memory budget invalidate relevant profile entries. Link profiles include simultaneous
traffic on shared resources. The planner must not sum isolated measurements from
two cables or treat logical streams as independent physical links.

A candidate becomes qualified only after actual execution confirms placement,
correctness, memory and end-to-end metrics. M0 never reaches this state.


## Generic scalability contract

Raising the M0 `3` constant is NOT the production scaling plan. Subset/permutation/
cut enumeration grows rapidly and must remain a small-case oracle. A production
search first computes eligibility, required-resource constraints, memory lower
bounds, permitted cut points and adapter-valid routes. It then explores bounded
candidate orders/partitions using dynamic programming, beam search or another
measured deterministic heuristic. No algorithm is selected solely by its name.

Search has explicit time, candidate and memory budgets plus cancellation. Results
include explored scope, algorithm/version, termination reason and optimality status.
Time exhaustion without a candidate means SEARCH_INCOMPLETE, not proof of
NO_FEASIBLE_PLAN. A feasible but unqualified result is advisory. A heuristic does
not claim global optimality; compare small cases to the exact oracle.

The inventory can contain many enrolled nodes without selecting all of them. Registry
limits, search limits, adapter limits and physically qualified scale are distinct.
Configured node/device maxima count actual compute owners, not front-door/relay-only
roles. An adapter's smaller supported envelope is disclosed before load.

Offline tests cover 1, 2, 3, 4, 8 and 16 nodes, including multiple devices on some
nodes, irregular routes and disconnected subsets. These are planner correctness/
scalability fixtures, NOT distributed-inference performance claims. Planning must
not rely on 128 being the maximum possible number of model layers or graph units.
Input bounds remain explicit for safety; grouping requires adapter-valid cut points.

## Model fit is not a scalar memory test

A manifest maps exact encoded tensors and tied aliases to legal indivisible or
splittable units. Each placement variant has per-pool demand for weights, host
metadata, graph/compute buffers, persistent state, staging, cache and load/repack peaks.
A single tensor/bank or mandatory local workspace can defeat a plan even when total
installed memory exceeds the checkpoint size. Report that unit and constraint.

Output heads, tied embeddings, tokenization/sampling ownership, recurrent reset
boundaries, shared experts and optional projectors must be explicit. Local multi-GPU
routing and inter-host routing can have different restrictions; do not flatten them
into a memory-ratio vector. CPU host RAM is not an implicit GPU-accessible weight pool.

Increasing context can change the chosen placement even when weights alone fit. A
backend-supported alternative KV encoding or smaller context requires explicit
approval and a new identity/profile. Do not present a silent fit adjustment as
satisfying the original request. See spec 10 for alternatives and refusal reasons.

## Forecasts versus actionable recommendations

Return a small set of non-dominated feasible alternatives: placement/compute owners,
per-pool peaks, load cost, first-token/decode estimates, measurement confidence and
reason each enrolled node was omitted. User latency constraints can be hard or soft,
but a hard constraint with unknown cost is not certified satisfied.

Hypothetical hardware or link upgrades belong to a clearly labeled what-if view.
They cannot populate qualification profiles or promise that buying hardware produces
an estimated speedup. Membership-minimizing plans are not called energy-optimal
without comparable energy measurements.
