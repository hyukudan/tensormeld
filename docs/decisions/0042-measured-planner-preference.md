# ADR-0042 — Native calibration may change recommendation, not plan identity

Status: **accepted**  
Date: 2026-10-05

## Context

Planner-v2 produces immutable synthetic candidates with deterministic plan hashes. Placement
calibration can provide stronger target-hardware evidence about which retained candidate is
faster for a specific workload.

Rewriting planner candidates or synthetic estimates from measurements would blur the
boundary between search assumptions and observed evidence, and would invalidate downstream
identity contracts.

## Decision

TensorMeld introduces a separate measured-preference overlay.

The overlay consumes:
- one intact planner-v2 result;
- zero or more placement-calibration records.

It preserves the planner result and all retained candidates unchanged.

The result explicitly exposes:
- `synthetic_best`;
- `measured_best`;
- `recommended`;
- recommendation source and measured objective.

When `require_native=true`, only `native-target` calibration may produce
`measured_best`. If no applicable native calibration exists, the recommendation is the
original synthetic best.

Every calibration must reference a retained candidate and match the planner result's
config/planning/profile identity. Incompatible or unretained measurements fail closed.

The overlay never sets `qualified=true` or `executable=true`.

## Consequences

- Synthetic estimates remain auditable and unchanged.
- Real measurements can improve recommendation quality without changing plan identity.
- Downstream qualification/admission continues to operate on the exact same candidate.
- Fixture measurements remain useful for contract testing but cannot influence native-mode
  recommendations.
