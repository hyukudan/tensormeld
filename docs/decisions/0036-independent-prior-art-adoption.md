# ADR-0036 — External engines inform TensorMeld; they do not own its architecture

Status: **accepted**  
Date: 2026-10-05

## Context

TensorMeld is deliberately studying current inference projects such as Strata and
TensorFold because they contain useful measured lessons around placement, memory,
scheduling, caching and execution.

There is a risk that "reuse before reimplementation" could be read too broadly: importing
another project's planner, scheduler or memory model would make TensorMeld dependent on
that project's architecture and hardware assumptions.

## Decision

TensorMeld remains an independent inference orchestration and execution architecture.

External projects may contribute:

- prior-art ideas;
- benchmark methodology;
- public interoperability interfaces;
- evidence about useful optimization directions.

Those concepts must be re-expressed as TensorMeld-owned contracts and implemented against
TensorMeld's heterogeneous node/device/physical-pool model.

The default policy is **independent implementation informed by prior art**.

In particular:

- Strata's planner, tuner, scheduler and process manager are not TensorMeld dependencies;
- TensorFold's engine, scheduler and memory manager are not TensorMeld dependencies;
- their machine-specific constants and heuristics are not TensorMeld defaults;
- llama.cpp is initially a revision-pinned replaceable compute backend, not the owner of
  TensorMeld planning, resource accounting, transport or session semantics;
- TensorMeld-native HIP/CUDA/other kernels may replace individual llama.cpp paths when
  profiling and correctness evidence justify doing so.

Direct source reuse remains possible only as an explicit exception under ADR-0008. It must
record exact provenance/license obligations and explain why independent implementation is
not preferable.

## Consequences

- TensorMeld can learn from fast-moving engines without inheriting their architecture.
- The control plane and planner remain backend-neutral.
- Optimization work is evaluated by TensorMeld's own correctness/evidence gates.
- llama.cpp can remain a reference/fallback engine while native paths are introduced
  incrementally.
- External research notes are non-normative until a separate TensorMeld decision adopts a
  principle into the architecture.
