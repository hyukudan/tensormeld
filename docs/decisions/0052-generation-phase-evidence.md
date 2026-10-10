# ADR-0052 — Generation-phase evidence is scenario-specific and separates sampling/feedback

Status: **accepted**  
Date: 2026-10-10

## Context

An ordered legal-unit compute chain plus boundary transfers is not yet a generation-cycle
model. Prefill and decode have different costs, decode cost depends on context position,
sampling may occur on a different device, and token feedback closes the autoregressive
cycle.

Reusing one generic unit compute number for all phases would overstate what was measured.

## Decision

TensorMeld introduces `tensormeld/generation-phase-evidence-v1`.

The evidence is bound to exact config, legal-unit, legal-unit-cost and runtime-manifest
identities, plus the tensor-movability/model/adapter identity chain already enforced by
those contracts.

The measured/declared scenario includes:

- exact `prefill_tokens`;
- exact `decode_context_tokens` for one decode-step observation;
- exact concurrency matching the runtime manifest.

For every legal unit and every legal allowed device it requires explicit:

- `prefill_us` for the stated prefill scenario;
- `decode_step_us` for one decode step at the stated context position.

Sampling is represented separately for one or more runtime-manifest devices:

- `sampling_us`;
- `logits_payload_bytes` from the final model owner to the sampler;
- `feedback_payload_bytes` from the sampler back toward the next-token start owner.

Transfer time for those payloads is not inferred here; it still requires exact
directional path evidence.

All results remain `qualified=false` and `executable=false`.

## Consequences

- an 8K decode-step observation cannot silently stand in for 32K;
- prefill and decode can be ranked independently later;
- sampling location can be chosen explicitly rather than assumed to be the final GPU;
- logits and feedback transfers can close a future autoregressive cycle model;
- this evidence alone is not TTFT, token latency, sustained decode rate or throughput;
- native execution remains independently gated.
