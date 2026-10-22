# ADR-0053 — Prefill and decode require distinct inter-unit boundary payloads

Status: **accepted**  
Date: 2026-10-10

## Context

`generation-phase-evidence-v1` separates prefill/decode compute and sampling, but it
does not distinguish the activation payload crossing a legal-unit boundary during
prefill from the payload for one decode step.

Those payloads can differ materially because prefill processes multiple tokens while
decode advances one autoregressive step. Reusing one generic boundary size would make
future TTFT/decode-cycle estimates unsound.

## Decision

TensorMeld adds backward-compatible `tensormeld/generation-phase-evidence-v2`.

v2 keeps all v1 identity and phase-workload requirements and additionally requires for
every legal unit:

- `prefill_boundary_output_bytes`;
- `decode_boundary_output_bytes`.

The final legal unit must declare both values as zero because no following legal unit
exists.

v1 remains parseable and retains its original canonical fingerprint behavior. It is
reported with `phase_boundary_payloads_complete=false` and must not be used for
phase-transfer modeling.

v2 reports `phase_boundary_payloads_complete=true` and is the minimum evidence version
for future TTFT-scenario or closed decode-cycle transfer calculations.

## Consequences

- existing v1 records remain auditable and valid for their original scope;
- prefill and decode transfer costs cannot silently share one payload size;
- future planners must require v2 before combining phase compute with directional path
  evidence;
- no TTFT/token-latency/execution claim is enabled merely by adopting v2.
