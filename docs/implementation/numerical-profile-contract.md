# Numerical profile declarations and static applicability

Implemented in `src/tensormeld/numerical_profile.py`, with portable regressions
in `tests/test_numerical_profile.py`. This is a bounded contract slice of the
[draft numerical capability proposal](../proposals/2026-10-11-numerical-plan-capabilities.md),
not a worker, execution manifest, numerical kernel or qualification artifact.

## Profile identity

`NumericalProfile.parse(record)` and `parse_numerical_profile(record)` accept the
strict `tensormeld/numerical-profile-v1` schema. `load_numerical_profile(path)`
adds the existing 2 MiB bound, duplicate-key rejection and malformed/nonfinite
JSON rejection. Unknown/missing fields, unsupported phases and self-supplied
`qualified=true` or `executable=true` fail closed.

A frozen profile separates these declarations:

- exact model manifest and target semantics identities, plus `declared_target`,
  `approximate_target` or `proposal_only` mode;
- checkpoint encoding, storage bits, stored code dtype/signedness, physical
  block geometry, scale-group geometry, affine/codebook interpretation,
  scale/offset formula, codebook hash and packed-layout hash;
- explicit per-phase activation encoding/grouping, rounding, clipping,
  finite-input, zero-group, denormal and division rules;
- decoded weight and activation operand dtypes, accumulation/saturation,
  group-scale order and reduction identity;
- output dtype, row-invariance declaration, input layout, optional outlier,
  compatible rotation and activation-calibration identities;
- KV/recurrent dtype, state layout/update semantics, and a separate wire
  dtype/quantization/layout/ABI contract.

The current finite activation vocabulary permits floating input or symmetric
A8/A4 preparation. Integer preparation requires a complete declared contract;
it is never inferred from weight-storage width. A4 can explicitly use signed
I8 operands without changing its quantization. Raw stored codes cannot narrow
or change signedness. A nonlinear four-bit codebook can use signed I8 lookup
or local floating conversion; its nibble indices cannot be raw I4 operands.
The generic encoding label alone grants no capability. This parser does not
decode a checkpoint or verify that its tensors match the declaration; the
model/adapter manifest gate must establish that separately.

V1 supports affine integer four/eight-bit stored codes, four-bit nonlinear
indices with an exact signed-I8 lookup declaration, and scalar floating
16/32-bit storage. It does **not** support scaled FP4 E2M1/NVFP4, FP8 E4M3/E5M2,
or hierarchical/microscaled floating formats. Their code dtypes and additional
scale-hierarchy fields are rejected; labeling their floating code values as
affine integers would contradict the checkpoint contract and must also be
rejected by the model/adapter manifest gate. Those formats need a later schema
with exact scale hierarchy, grouping, decoding and operand semantics. This
v1 contract does not cover every low-precision backend format.

`profile.fingerprint` is SHA-256 over UTF-8 canonical JSON with sorted object
keys and phases sorted by name. All semantic/layout/calibration fields and
the unqualified markers are included. `profile.as_record()` returns an
independent JSON-compatible record with `profile_sha256`; parsing verifies
that supplied hash. Nested contracts are frozen and contain no mutable maps.
Applicability functions reparse records to reject tampering through manual
dataclass construction or replacement. Hashes provide identity/integrity,
not source authentication or native correctness evidence.

One profile aggregates explicitly declared phases. A request-family envelope
can bind this aggregate fingerprint as its `numerical_plan_sha256`. That
opaque envelope binding does not itself establish phase/shape support.

## Exact applicability

`NumericalScope` binds profile hash, exact phase, actual rows, K/N, backend,
device architecture, stable runtime hash and implementation hash.
`NumericalCapability.parse(record)` declares precisely one such scope.
There are no wildcard shape ranges, implicit backend mappings, automatic
precision relaxation or replacement of verify evidence by prefill evidence.
Backend and architecture are explicit strings; no vendor is privileged.

`validate_numerical_applicability(profile, scope, capability, ...)` also
requires the exact profile in the caller's immutable approved-policy set.
The caller must derive that set from the request's model/user policy rather
than from the worker's advertisement. M is actual rows for this operation or
expert, not installation concurrency. Decode is one actual row in this
contract. K must contain exact physical storage blocks and activation groups;
K=640 cannot silently receive a 256-element storage tail.

Results are `REPRESENTABLE` or `REJECTED`, with structured reasons and always
`qualified=false`, `executable=false`. Successful declarations must still
pass model fidelity, exactness/quality, native applicability, resource,
transport, admission and complete-workload gates before execution.

An optional `NumericalCalibration` must match the entire scope. Its six
explicit cost categories cover preparation, layout/lookup, grouping/scatter,
compute, scales/output and snapshots; transient peak bytes are separate.
Missing categories are errors. Complete cost must be positive and bounded;
individual zero categories mean explicitly declared no work, not unknown
work. Missing or mismatched calibration produces no usable cost, and fixture
records do not satisfy a required native calibration. Native source labels
and parsed counters do not prove measurement authenticity, freshness,
quality or E4: existing trusted evidence gates must establish those separately.
The module supplies no measured preference, speedup or cost extrapolation.

## Retained state and authoritative verification

`StateHistoryBinding` carries profile and derived state-history hashes.
The history hash includes the complete aggregate profile and state contract.
A change in any phase, quantization/calibration, reduction, layout or state
semantics conservatively invalidates this binding. Matching storage dtype
alone cannot authorize recurrent/KV/prefix reuse. Input content, history
position, session authorization, ownership and epochs remain separate
request/session gates; this record does not replace them.

For `exact_commit=true`, applicability requires an explicit approved
authoritative serial target and a verify request. The verifier must bind the
**same complete aggregate profile** as its authoritative target, including
prefill/decode history, wire precision and every layout/calibration field.
V1 has no independent profile-conversion/equivalence evidence, so a different
profile cannot borrow authority even when it binds state from its own history.
Model, target semantics, weight interpretation and state contracts must match;
both phases must
declare row invariance, and verify arithmetic must match serial decode
arithmetic. Only phase name and physical input layout are excluded from the
arithmetic comparison. Thus W4A4 verification cannot authorize W4A16 target
commit even if both records repeat the same target label. Approximate and
proposal-only modes cannot self-authorize exact commits. A passing static
comparison is still a declaration and not proof of identical decisions,
sampled distributions or accepted state.

## Explicit boundary pairs

`BoundaryCapability` binds directional source/target profile/runtime hashes,
exact wire contract and a mechanism (`host_copy`, `network_stream`, `gpu_dma`).
`validate_boundary_applicability` requires both endpoint wire contracts to
match and the exact pair record to exist. Local A4/I4 execution does not
imply integer wire encoding, remote compute, DMA import/export or system
visibility. A stream pair record cannot satisfy a GPU-DMA request. This is
only static representability; native transfer ownership/completion and
trusted pair qualification remain independent requirements.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_numerical_profile.py -v
```

The suite covers canonical ordering, detached immutable records, all bound
identity categories, tampered fingerprints, malformed/duplicate/oversized
JSON, codebook/raw-I4 incompatibility, signedness/precision rejection, exact
phase/shape/runtime scopes, K=640 storage incompatibility, explicit policy,
retained phase-history invalidation, A4/A16 authoritative mismatch,
row-invariance, missing/fixture/invalid calibration and directional
wire/runtime/mechanism rejection. No GPU or model execution is performed.
