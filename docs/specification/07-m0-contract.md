# 07 — Implemented M0 contract

This document describes existing code, not the full production design.

## Inputs

`schema_version` is the integer 1. `name` is a nonempty label. `provenance` is
`synthetic` or `measured`. User-supplied measured status does not certify hardware
or the resulting plan. `workload` identifies model revision, quantization, context,
concurrency and phase. Only phase `decode` and concurrency 1 are accepted.

`pools` contains unique IDs, owning node IDs and positive integer `budget_bytes`.
These are allocatable limits after safety reservations. `devices` identifies node,
pool, backend label and nonnegative runtime bytes. Up to three devices are supported;
multiple same-node devices may reference one pool. A backend label is descriptive,
not evidence that a driver or operator exists.

`stages` contains 1..128 unique, ordered stage IDs, resident weight bytes, worst-
workload state bytes, per-stage workspace bytes, output-boundary payload bytes and
positive `decode_ms` costs keyed by eligible device. The last stage includes output
and sampling cost. Missing device cost means that placement is unavailable. Stage
weights must not contain unaccounted tied aliases; M0 does not resolve them.

`links` contains zero or more directed profiles. Each has unique ID, source/target
device IDs, positive useful payload bytes per second, nonnegative fixed latency in
microseconds and a physical sharing-group label. Link cost includes whatever GPU,
CPU and protocol path its input measurement represents. Do not mix incomparable
profile boundaries. M0 cannot detect such a measurement mistake.

`route_mode=direct` requires an explicit direct profile for each transfer.
`route_mode=via_coordinator` uses source→coordinator→target when neither endpoint is
the coordinator. Each hop needs its own profile. `coordinators` lists eligible
existing device IDs. A coordinator may be enabled without owning compute stages;
its control-plane resource use must already be included in pool safety reservations.
Control work that is not represented in stage or transfer costs is not modeled.

`feedback_bytes` is a positive cyclic payload from final-stage sampling to the first
stage for the next decode step. The solver charges this path even if it is small.
All byte fields are integers, not GB/GiB strings. Unknown fields, duplicate JSON keys,
nonfinite numbers and oversized inputs are rejected. Scenario file size is at most
2 MiB. Input costs and memory estimates are supplied, not inferred from nominal specs.

## Search

For every nonempty subset of enabled devices, enumerate device permutations and
nonempty contiguous stage ranges, at most one range per selected device. For each
eligible enabled coordinator, check stage profiles, memory and routes. The model's
stage order never changes. Device order may vary. The search is exhaustive only
within this bounded candidate family, not a globally optimal heterogeneous solver.

Pool usage is the sum, for each active device referencing that pool, of its weights,
persistent state, runtime bytes and the maximum workspace of its assigned stages.
Concurrent per-device peak workspaces are conservatively summed in a shared pool.
It does not model loading transients, tied tensor aliases or future transport slots;
those must be included by the supplied stage/runtime budget or remain unqualified.

Each boundary uses the last stage's output size. The cost is serial stage time plus
boundary costs plus cyclic feedback. If multiple links exist for a hop, choose the
cheapest SINGLE link for that payload. Sharing-group labels are preserved in the
trace, but simultaneous contention is not simulated. No multirail sum is performed.

Stable tie-breaking prefers lower cost, fewer compute devices, then deterministic
lexical device/cut/coordinator order. The result records scenario SHA-256, workload,
search counts, ranges, pool usage, hop traces and separate compute/communication
estimates. Exit code 2 means no feasible candidate. A parse/operational error is 1.

## Deliberate limitations

No actual GPU inference, model-file parsing, native worker launch, HTTP inference
server, paired remote agent, concurrent scheduling, automatic memory admission,
remote profiling, expert-level split, direct GPU transfer or performance claim.
Only a proposed candidate is returned; a backend must later validate its ability
to implement it. All results are `qualified=false`.

The loopback diagnostic is separate from the model transport. It opens only
127.0.0.1, uses length-prefixed bounded frames, verifies echo data and reports host
round-trip sample distributions. It must never populate production link profiles
as if it measured Ethernet, USB4 or a GPU-ready tensor path.
