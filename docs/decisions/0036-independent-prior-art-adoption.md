# ADR-0036 — Independent architecture and explicit component adoption

Status: accepted. Original decision date: 2026-10-05.
Editorial neutralization: 2026-10-11; technical ownership policy is unchanged.

## Decision

TensorMeld owns its heterogeneous resource model, planning, admission, transport and
session semantics. Private evaluations may inform hypotheses, but are not runtime
dependencies or qualification. Project documentation describes its own contracts.

Machine-specific constants and heuristics are not transferable defaults. The pinned
llama.cpp adapter remains a replaceable compute backend. Native paths may replace
individual operations when profiling and correctness evidence justify them.

Source reuse remains an explicit ADR-0008 decision with exact provenance, license,
notices, local modifications and qualification. Actual dependency notices are retained;
private research-only entries do not belong in the public dependency registry.

## Consequences

- The control plane and planner remain hardware-neutral.
- Native implementations satisfy project-owned numerical and evidence contracts.
- Comparative records stay outside Git; supported component provenance remains public.
