# Implementation roadmap

## M0.2 — General contracts and policy

Status: **advanced**.

Implemented: heterogeneous registry/policy, bounded synthetic whole-block planning,
adapter representability and advisory runtime availability. Runtime observations remain
snapshots, not reservations.

## M1 — Agent and enrollment

Implemented:

- cross-platform local agent skeleton without a remote listener;
- explicit enrollment/node/key identity supplied at runtime;
- authenticated versioned capability envelope with replay rejection;
- drain/disable/revoke lifecycle and fixed allowlisted operations;
- host-owned physical-pool lease authority over the existing local admission primitive.

Next:

- real multi-machine private-LAN integration using provisioned certificates and enrolled endpoint records;
- loopback/private-LAN integration tests through that channel;
- OS-protected key/certificate storage integration and rotation/revocation persistence.

Implemented transport foundation:

- strict client/server TLS policy builders with CA verification and client certificates;
- dedicated ALPN and exact peer-certificate SHA-256 pinning;
- bounded versioned control frames with epoch/sequence replay protection;
- only the existing agent allowlist is representable as a control operation;
- real loopback mTLS handshake with ephemeral CI-generated CA/server/client certificates;
- exact peer-certificate fingerprint bound to enrolled node identity before control exchange;
- explicit private/loopback endpoint policy plus bounded remote agent method dispatch;
- request/response correlation and local-object registry for reserve/recheck references;
- end-to-end remote health/reserve dispatch through the real loopback mTLS channel;
- explicit private endpoint connect/serve helpers validated across separate OS processes.

Exit gate: two machines can establish a secure control relationship without exposing an
arbitrary execution surface.

## M2 — Evidence, topology and model manifests

Status: **in progress**.

Implemented:

- exact model/checkpoint identity;
- qualification evidence/applicability contracts;
- advisory runtime physical-pool observations;
- pinned llama.cpp trusted-local no-model probe;
- explicit approved engine-device ↔ TensorMeld-device binding;
- one explicit memory reporter per physical pool;
- observed-but-not-ready native device state;
- pinned upstream `test-backend-ops` readiness contract with target-execution proof;
- narrow observed → ready promotion after strict E2 backend self-test evidence;
- native-only retained E2 backend-readiness record with exact identity fingerprints;
- retained E2 v2 records bound to stable worker/OS/driver/runtime/device/topology identity;
- exact live identity reuse of the narrow backend-ready fact after a fresh binding;
- exact runtime model/operator/memory manifest contract;
- per-device operator declarations plus one memory record per physical pool;
- fixture/native-adapter provenance separation and non-executable CLI validation;
- atomic in-process physical-pool leases for exact manifest preparation peaks;
- launch-time recheck with explicit telemetry/reflected-lease accounting and release.

Next:

- real target-host backend-readiness/runtime-manifest records using the implemented handoff-bound collector;
- directional path profiling and evidence provenance.

No real native GPU backend has yet been qualified by TensorMeld; portable fixtures do not
satisfy the real E2/E3 gates.

## M3 — First executable distributed inference

- run the implemented target-host qualification chain on real hardware and record real E2/E3/runtime-manifest evidence;
- real target-host llama.cpp capability probe and device binding;
- real backend self-test and live adapter qualification;
- runtime memory/admission manifest for exact model/workload;
- atomic/leased resource admission and launch-time recheck on real target hardware using the implemented admission orchestrator;
- secure enrolled agent and authenticated private transport;
- native whole-block executable adapter across one or more nodes;
- immutable accepted plan;
- stable local streaming API;
- native-session cancellation and deterministic lease release implemented for the admitted local worker lifecycle; real llama.cpp process cancellation remains to be proven on target hardware;
- reference correctness suite;
- same-model comparisons: local vs companion vs distributed.

Reference execution foundation implemented:

- immutable accepted-execution bundle that recomputes planner integrity and adapter representability;
- exact E3 qualification applicability and worker-artifact match;
- exact per-device backend-ready proofs and per-node launch-admitted leases;
- bounded reference whole-block session with deterministic fixture backend, cancellation and release.

The reference path deliberately reports `real_model_inference=false`.

Native subprocess foundation implemented:

- exact local launcher/program artifact SHA-256 verification;
- engine-revision and accepted-bundle binding;
- fixed argv with no shell or caller-supplied command line;
- bounded JSON stdin/stdout protocol carrying exact segment/device/unit identity;
- real cross-platform subprocess execution using a fixture worker;
- fail-closed request/response hash checks and rejection of worker self-claims of real inference.

Model-aware llama.cpp qualification/execution placement foundation implemented:

- exact `blk.N` unit naming and contiguous full-GGUF block coverage;
- exact current native binding SHA and engine-device identity checks;
- local primary buffer-type ownership only; RPC/remote devices fail closed;
- generated anchored `--override-tensor` rules with `--fit off` and no user regex;
- deterministic post-E3 placement fingerprint bound to the accepted execution bundle and pinned llama.cpp revision;
- separate pre-E3 qualification placement from exact planner candidate + representability + current native binding + complete GGUF block coverage.

Native trial foundation implemented:

- approved local llama-cli artifact SHA-256 and exact single-file GGUF SHA/size/name verification;
- closed deterministic llama-cli argv using prompt, one-shot generation, seed 0, temperature 0, simple IO, no prompt echo/timings/color and the exact qualification placement fragment;
- inherited `LLAMA_ARG_*` variables removed before subprocess launch;
- injected runners cannot impersonate native-subprocess evidence;
- successful process/stdout can produce trial evidence only and never self-promotes E3, `qualified`, `executable` or `real_model_inference`.

Native E3 correctness foundation implemented:

- approved deterministic reference contract bound to exact trial spec, llama-cli SHA, model, placement and workload;
- native-subprocess-only evaluator with exact stdout SHA-256 comparison;
- exact runtime worker/device identity coverage;
- QualificationEvidence v2 carrying candidate-plan, placement, trial, reference and runtime fingerprints;
- execution admission requires the E3 v2 plan/runtime identities to match current readiness.

Target-host qualification handoff foundation implemented:

- recompute probe/binding identity instead of trusting a supplied bound record;
- require exact retained E2 and current runtime identity for every candidate device;
- recompute pre-E3 placement from config/planning/candidate/GGUF/binding;
- rerun native E3 correctness evaluation from retained trial/reference artifacts;
- combine per-device E2 promotions into one ready runtime observation;
- emit exact runtime-manifest requirements while keeping reservation/launch/executable false;
- surface an explicit E3-vs-profile workload mismatch instead of silently widening qualification.

Native runtime-manifest collection foundation implemented:

- runtime measurement bound to exact qualification handoff/worker/config/model/adapter/runtime identities;
- operator requirements are a separate handoff-bound contract and cannot be self-declared by measurement observations;
- exact profile workload required;
- one runtime identity per compute device;
- at least each compute device's physical pool measured, with no duplicate shared-pool records;
- existing RuntimeModelManifest parser remains the canonical capacity/operator invariant gate;
- native manifest provenance requires both measurement and operator-requirement provenance to be native;
- collector remains non-reserving/non-executable and only reports whether inputs are admission-ready.

Target-host admission orchestration foundation implemented:

- require intact admission-ready native runtime-manifest collection;
- one explicit compute-node authority for the current llama.cpp local shim;
- reserve exact physical-pool preparation peaks;
- require a distinct second runtime observation for launch recheck;
- rollback lease on launch rejection or later bundle-construction failure;
- construct AcceptedExecutionBundle only from the same plan/E3/readiness/manifest tuple;
- keep inference_started=false even after execution authorization.

Native admitted-session lifecycle foundation implemented:

- session creation requires an intact execution-authorized admission result;
- backend must be bound to the exact AcceptedExecutionBundle;
- launch-admitted lease must still be active, launched and fingerprint-matched;
- synchronous completion releases the lease;
- pre-run cancellation releases without executing backend work;
- in-flight cancellation is deferred until execution reaches a terminal boundary, so the lease is not released under a running worker;
- backend failures release the lease;
- explicit release is idempotent.

A real target-host run must still supply the actual llama.cpp worker backend and prove real process cancellation/termination behavior before production native inference claims.

### Prior-art-informed M3 priorities

The 2026-10-05 TensorFold/Strata audit changes implementation order, not ownership:

**P0 — measured llama.cpp backend baseline**
- managed child-process ownership with bounded logs, readiness and terminate/kill is implemented and portable-tested with a real fixture server;
- exact llama-cli/llama-server/backend-library build package identity and admitted lease bridge are implemented;
- next add server-vs-cli E4 request equivalence for the exact package/model/placement before enabling inference requests through the persistent server;
- qualify the first real target machines through the existing E2/E3/manifest/admission chain;
- TensorMeld-owned placement calibration records are implemented for exact planner candidates, build package and common runtime environment;
- separate prefill/decode timings are normalized to the target context/output workload with integer arithmetic;
- bounded coarse→local-refinement candidate selection is implemented without inventing placements or treating synthetic estimates as measurements;
- measured preference overlay is implemented: applicable native-target calibration can supersede synthetic ordering without mutating planner candidates or plan identity;
- local calibration persistence/aging is implemented with immutable fingerprint files and current package/runtime/topology revalidation;
- next collect real native-target calibration records on RTX/Strix hosts and compare only currently applicable records across runs.

**P1 — predictive memory and equivalence**
- physical-pool predictive memory classes are implemented as an exact refinement of runtime-manifest-v1: hard-resident, state, workspace, staging and reclaimable/file-backed;
- current admission remains conservative because worst-case physical bytes exactly reproduce the existing preparation peak;
- exact tensor storage/movability evidence is implemented with complete GGUF-index coverage, explicit allowed devices and current build/runtime identity binding;
- next connect verified tensor movability to predictive pool classes and planner legal-unit generation without changing current whole-block admission by default;
- add measured session/cache growth and prefill/decode/shared-round working-set contracts;
- add E4 equivalence for fresh/resumed, solo/concurrent and serial/optimized execution before enabling advanced scheduling or cache reuse;
- define cache/snapshot identity over model, runtime, kernel/precision route, placement and chunk/prefill plan.

These are independent TensorMeld implementations informed by prior art. Strata and
TensorFold are research references, not planner/runtime dependencies.

Exit gate: a model larger than the entrypoint GPU can execute through a verified native plan.

## M4 — Planner quality and product UX

- objective-specific scoring;
- plan comparison/explanations;
- profile presets;
- model/session queue;
- local web/desktop UI using the same control API;
- cache/catalog management.

## M5 — Advanced heterogeneous execution

Prior-art-informed later work:
- session-prefix/KV checkpoint retention and optional disk spill before generic disk-backed weight streaming;
- directional storage→host→device and node→node path profiling;
- benchmark mmap/page-cache versus direct/pinned/read-ahead separately for discrete GPUs and unified-memory Strix Halo;
- adaptive speculative depth only after exact AMD-native equivalence evidence;
- selective replacement of llama.cpp paths only where profiling proves a material bottleneck and a TensorMeld-native implementation passes correctness/equivalence gates.

Core M5 placement/execution work:
- expert-aware placement;
- tensor/operator strategies;
- prefill/decode phase placement;
- compute/communication overlap;
- multirail qualification;
- optimized Strix Halo worker integration;
- speculative/MTP strategies where measurable and semantically safe.
