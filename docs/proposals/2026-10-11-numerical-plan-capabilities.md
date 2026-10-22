# Phase specific numerical capability proposal

Status: Draft. Requirement IDs `NPC`. Extends adapter/evidence contracts without hard-coding one GPU vendor or changing current execution qualification.

NPC-001: Workers describe supported numerical profiles by model encoding, shape/phase, activation preparation, accumulation, state/wire ABI and implementation identity. A storage-bit label alone does not establish representability or compatibility.

NPC-002: Profiles bind codebook/affine interpretation, weight and activation group geometry, rounding/clipping/finite-input rules, signedness/saturation, scale-combination order, outlier/rotation/calibration identity, row invariance, prefill/decode/verify policies and persistent-state consequences.

NPC-003: Model selection and user policy approve allowed numerical modes. The planner cannot switch activation precision, re-quantize weights, quantize state or compress a boundary merely to improve a score. Within an approved profile, workers may select equivalent qualified kernel realizations.

NPC-004: Distinguish model fidelity, within-plan exactness and approximate quality-qualified execution. Exact speculative verification is bound to its authoritative target semantics and state. A tolerant numerical comparison is not proof of an identical sampled distribution.

NPC-005: Kernel calibration uses complete phase cost and actual row/encoding scope. Include input quantization, layout/lookup, grouping/scatter, scales, output, snapshots and transient peaks; capability discovery and theoretical arithmetic throughput never create measured planner evidence.

NPC-006: Cache/session/evidence identity includes phase-specific numerical profile, compatible state history and realization as required. A profile change invalidates affected caches/calibrations/qualification explicitly. Separate durable model/profile evidence from per-launch admission/canaries.

NPC-007: Cross-node boundary dtype, quantization and state compatibility are explicit pair capabilities. Local INT4 computation does not imply INT4 transport or direct GPU DMA. Every backend and architecture needs its own native applicability record.

NPC-008: Unsupported shape/encoding/mode returns a structured rejection. Fallback is permitted only within approved equivalent semantics or after an explicit new plan/policy choice. User-visible outputs identify approximate numerical modes and their qualification scope.

## Acceptance

- An adapter rejects a nonlinear nibble codebook offered as affine integer operands.
- A phase-profile change cannot reuse incompatible recurrent/KV/prefix state.
- A W4A4 verifier cannot authorize exact W4A16 target commits.
- A missing row/phase calibration cannot be represented as zero cost or a measured speedup.
- Kernel-only savings do not bypass full memory, quality, transport and committed-token gates.
- Backend-neutral planning retains local-only and companion-only baselines and policy exclusions.
