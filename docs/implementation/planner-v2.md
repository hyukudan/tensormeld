# v2 synthetic whole-block planner

Implemented in `planning_contract.py` and `planner_v2.py` since 0.2.0a2.

## Usage

```bash
python -m tensormeld plan-v2 docs/examples/12gb-pc-one-companion.config.json examples/v2/12gb-dense.planning.json
python -m tensormeld plan-v2 examples/v2/high-capacity.config.json examples/v2/high-capacity.planning.json --top-k 3
```

These are synthetic costs and illustrative resource ceilings, not RTX/Strix benchmarks.

## Contract

Input schema: `tensormeld/planning-v1`; requires a matching v2 installation/profile,
exact manifest reference, context, output bound and C1 text-generation decode workload.
Every unit lists eligible device profiles, decode cost in integer microseconds,
`resident_bytes` (weights + persistent state) and `workspace_bytes` maps by physical
pool. Both maps may include local VRAM and local system RAM. Remote-pool demands
are rejected. Explicit maps also specify worker and coordinator overhead; zero is an
explicit synthetic assumption, not an observed absence of overhead.

Links refer to `device:<id>` and `node:<id>`, with payload bytes/second, fixed latency
and a physical-group label. `direct` requires direct device links. `via_coordinator`
requires device-to-coordinator and coordinator-to-device hops. Even same-host copies
need explicit profiles. No general graph relaying, implicit USB4 aggregate, GPUDirect
claim or frontend-stream latency is inferred.

The parser is bounded to 256 units and 2048 links inside a 2 MiB JSON document. Config
registry bounds remain 64 nodes, 128 devices and 192 pools. These are parser/search
limits, not hardware scaling claims.

## Search and scoring

The planner tries single owners before a lazy DFS. A device owns at most one nonempty
contiguous segment. Work-unit and deadline budgets cover partial/rejected attempts,
with at most 100,000 work units and 60,000 ms configurable in this build. A user can
lower these bounds. Retained results are 1..20; the DFS stack is bounded by unit count.
This is not a globally optimal scalable solver on budget exhaustion. Single-owner
seeding and coordinator ordering can bias which candidates are found first.

`interactive_latency` minimizes the modeled sequential decode cost. `capacity` first
prefers fewer compute nodes/devices among placements of the SAME requested model,
then modeled decode cost. Throughput/balanced scoring and expert/tensor/phase partition
requests are rejected rather than quietly treated as this objective/strategy.

Memory uses static ceilings intersected with capacity-minus-reserve, not live allocation
leases. Each worker retains its per-pool peak arena, so those peaks are summed across
workers sharing a pool. No two workers are assumed to alias allocator workspaces.

## Outcomes and exit status

- Exit 0: `CANDIDATES_FOUND`, complete within this search family.
- Exit 2: `NO_CANDIDATE_IN_SEARCH_SPACE`, no candidate in this restricted input.
- Exit 3: `SEARCH_INCOMPLETE`, with or without a best-so-far candidate.
- Exit 1: invalid input or unsupported contract; no automatic semantic fallback.

Outputs include config/input/plan hashes, segment boundaries, physical-pool usage,
chosen links, cyclic feedback, rejection counts and the exact exhausted budget.
They are never executable plans. Adapters, model-specific support, real measurements,
loading/transient admission and active-session reservations remain unimplemented.
