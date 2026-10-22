# ADR-0009 — Synthetic planning remains separate from executable manifests

**Status:** Accepted
**Date:** 2026-10-01

## Decision

Implement a bounded whole-block planner against explicit synthetic workload profiles
before integrating native workers. A directory of checkpoint tensors is not a complete
runtime memory manifest: state, workspace, aliasing, loading transients, host allocations
and protocol buffers still require backend evidence.

The synthetic input accepts only `provenance=synthetic`. The result always states
`qualified=false` and `executable=false`. A require-qualified execution policy is not
relaxed; this command produces advisory candidates only and cannot launch a worker.

Start with single-owner seeds, then a lazy depth-first search of contiguous block
assignments. Charge partial and rejected expansions against the search budget; bound
retained results and stack depth. Deadline checks are cooperative, not hard realtime.
Return `SEARCH_INCOMPLETE` on exhaustion even when a best-so-far candidate exists.
`NO_CANDIDATE_IN_SEARCH_SPACE` means none in this finite synthetic search family,
not that inference is physically impossible using another adapter or partition strategy.

Count persistent allocations, one worker overhead per selected device, coordinator
allocations and retained per-worker peak workspace per physical pool. Do not add
capacities for multiple views of the same pool. Preserve cyclic token-feedback cost.
Select one explicit directed link per hop: multiple cables never imply summed capacity.
