# ADR-0041 — Placement estimates generate candidates; calibration ranks measured candidates

Status: **accepted**  
Date: 2026-10-05

## Context

TensorMeld's synthetic planner estimates decode cost and produces bounded whole-block
candidates. Those estimates are useful for candidate generation but are not target-hardware
measurements.

The TensorFold/Strata prior-art audit reinforced a useful principle: placement should be
measured on the actual runtime, with prefill and decode treated separately, while avoiding
machine-specific constants becoming global policy.

## Decision

TensorMeld introduces `tensormeld/placement-calibration-v1`.

Calibration is a separate evidence layer over already hashed planner candidates. It never
creates or mutates a plan.

A calibration record binds:
- exact Config, PlanningInput, profile and model identity;
- exact llama.cpp build package;
- exact candidate plan SHA;
- one common stable runtime environment for the sweep;
- exact package llama-server artifact in every runtime identity;
- target prefill/decode workload;
- measured prefill/decode tokens and elapsed time;
- optional physical-pool peaks for candidate-used nodes.

The runtime environment is a superset of the candidate's device set. This allows a
one-device candidate and a two-device candidate to be compared under the same hardware
snapshot.

The workload objective is normalized with integer arithmetic:
`ceil(prefill_elapsed * target_prefill / measured_prefill)
+ ceil(decode_elapsed * target_decode / measured_decode)`.

No generic tok/s score is promoted across workloads.

TensorMeld also introduces a bounded coarse→refine helper over the planner's existing
ordered candidate list. The coarse pass samples evenly across that list. Once applicable
measurements exist, refinement considers nearby unmeasured candidates around the current
winner.

Fixture measurements are never eligible for native ranking when native evidence is
required.

## Consequences

- Planner estimates remain estimates and candidate identities remain immutable.
- Hardware calibration can override planner ordering without rewriting plans.
- Different device subsets can be compared under one common runtime environment.
- The calibration objective is explicitly workload-specific.
- Coarse→refine is a bounded heuristic, not proof of a global optimum.
- Real target measurements are required before any performance claim or measured
  placement preference is used.
