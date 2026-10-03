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

- real target-host backend-readiness/runtime-manifest records;
- directional path profiling and evidence provenance.

No real native GPU backend has yet been qualified by TensorMeld; portable fixtures do not
satisfy the real E2/E3 gates.

## M3 — First executable distributed inference

- correctness-qualified revision-pinned llama.cpp worker implementation on top of the implemented placement shim and subprocess protocol;
- real target-host llama.cpp capability probe and device binding;
- real backend self-test and live adapter qualification;
- runtime memory/admission manifest for exact model/workload;
- atomic/leased resource admission and launch-time recheck;
- secure enrolled agent and authenticated private transport;
- native whole-block executable adapter across one or more nodes;
- immutable accepted plan;
- stable local streaming API;
- cancellation and deterministic release for native sessions;
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

Model-aware llama.cpp placement foundation implemented:

- exact `blk.N` unit naming and contiguous full-GGUF block coverage;
- exact current native binding SHA and engine-device identity checks;
- local primary buffer-type ownership only; RPC/remote devices fail closed;
- generated anchored `--override-tensor` rules with `--fit off` and no user regex;
- deterministic placement fingerprint bound to the accepted execution bundle and pinned llama.cpp revision.

A real llama.cpp worker using this placement spec must still pass native E3 correctness before `real_model_inference=true` can exist.

Exit gate: a model larger than the entrypoint GPU can execute through a verified native plan.

## M4 — Planner quality and product UX

- objective-specific scoring;
- plan comparison/explanations;
- profile presets;
- model/session queue;
- local web/desktop UI using the same control API;
- cache/catalog management.

## M5 — Advanced heterogeneous execution

- expert-aware placement;
- tensor/operator strategies;
- prefill/decode phase placement;
- compute/communication overlap;
- multirail qualification;
- optimized Strix Halo worker integration;
- speculative/MTP strategies where measurable and semantically safe.
