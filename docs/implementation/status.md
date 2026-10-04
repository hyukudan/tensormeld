# Current implementation status

Version: **0.2.0a2** — TensorMeld. Specification baseline: 0.2.0.

The canonical repository is public for engineering transparency. This does not change
the pre-alpha status or qualify any execution path.

## Implemented

- strict heterogeneous installation/policy contracts and bounded synthetic planner;
- exact static adapter representability gate;
- exact GGUF-backed model identity and qualification-evidence applicability contracts;
- advisory runtime-observation and conservative runtime-pool budget resolution;
- optional psutil host inventory and upstream GGUF catalog reuse;
- trusted-local pinned llama.cpp no-model probe with artifact/source identity;
- explicit approved one-to-one llama.cpp engine-device ↔ TensorMeld-device binding;
- backend identity taken from TensorMeld config, never inferred from engine labels;
- optional single memory reporter per physical pool to avoid shared-memory double counting;
- bound native devices enter runtime observations as `observed`, not `ready`;
- strict pinned llama.cpp `test-backend-ops` readiness adapter for one bound device;
- positive target-backend execution proof is required in addition to exit code;
- successful narrow E2 readiness can promote only that bound device to runtime `ready`;
- native-only retained E2 backend-readiness records with deterministic fingerprints;
- exact config/probe/binding/test-artifact/device/backend applicability checks for retained E2;
- retained E2 v2 records bind stable worker/OS/driver/runtime/device/topology identity;
- exact current runtime identity plus fresh binding can reuse only the narrow retained E2 backend-ready fact;
- strict runtime model/operator/memory manifests tied to exact config/model/profile/workload/adapter identities;
- runtime memory is represented once per physical pool with resident/state/workspace/preparation peaks;
- operator coverage is explicit per device and cannot self-promote qualification/execution;
- CLI validation for runtime manifests without creating resource reservations;
- in-process atomic physical-pool leases for exact runtime manifests;
- launch-time re-admission requires a newly identified runtime observation;
- active leases can be explicitly marked reflected in telemetry to avoid double subtraction;
- deterministic local lease release and concurrent admission serialization;
- enrolled local host-agent skeleton with explicit node/key identity and signed capability envelopes;
- replay-safe monotonic capability envelopes using standard-library HMAC-SHA256;
- host agents reserve/recheck only their own node's physical pools;
- drain/disable/revoke lifecycle without arbitrary peer-supplied execution surface;
- TLS/mTLS context policy builders using Python ssl/OpenSSL with CA verification and client certificates;
- exact peer-certificate SHA-256 pinning plus dedicated control-channel ALPN;
- bounded private control framing with connection epoch and monotonic replay rejection;
- real ephemeral-certificate mTLS loopback integration in Linux CI;
- enrolled peer identity can be bound to an exact TLS certificate SHA-256;
- explicit private/loopback endpoint policy rejects public, hostname and link-local targets;
- bounded remote HostAgent dispatch over the private control channel with request correlation;
- remote reserve/recheck accepts only locally registered manifest/snapshot identities, never peer-supplied paths or payload objects;
- real loopback mTLS integration now exercises RemoteAgentClient/Dispatcher end-to-end, including host-local reserve;
- explicit private endpoint client/server helpers run the authenticated control stack across separate OS processes;
- immutable accepted-execution bundles combine exact plan integrity, adapter representability, E3 model qualification, runtime manifest identity, per-device backend readiness and per-node launch admission;
- first bounded whole-block executable reference session iterates exact plan segments with cancel/release lifecycle;
- revision-pinned local subprocess worker foundation validates launcher/program SHA-256, engine revision, accepted-bundle identity and fixed no-shell argv;
- bounded native-worker JSON protocol binds request, segment, device/unit ownership and payload identity across a real subprocess boundary;
- pinned llama.cpp model-aware placement shim maps exact `blk.N` units to generated `--override-tensor` rules only after current native device binding and complete GGUF tensor-index checks;
- initial llama.cpp placement shim is single-node/local-device only and rejects RPC devices, non-primary buffers, partial block coverage and caller-supplied patterns;
- pre-E3 llama.cpp qualification placement translates an exact representable planner candidate without requiring an AcceptedExecutionBundle, avoiding E3 bootstrap circularity;
- strict llama.cpp native trial spec binds approved llama-cli/GGUF SHA-256 identities, qualification placement, deterministic prompt/context/predict controls and scrubbed `LLAMA_ARG_*` environment;
- QualificationEvidence v2 binds E3 to candidate plan, placement, trial spec, correctness contract and exact runtime-identity fingerprints while preserving v1 parsing compatibility;
- native llama.cpp E3 correctness evaluator requires genuine `native-subprocess` trial provenance, exact stdout SHA-256 match to an approved reference contract and exact runtime worker/device identity before emitting E3;
- AcceptedExecutionBundle now requires E3 v2 applicability to the same plan and same runtime identities as current backend-readiness evidence;
- target-host qualification orchestrator recomputes current probe/binding and pre-E3 placement, validates retained E2 for every compute device, reruns E3 evaluation, combines ready-state observations and emits a deterministic non-executable runtime-manifest/admission handoff;
- native runtime-manifest collector validates an intact qualification handoff, independent operator-requirement contract, per-device runtime identities/operator observations and one memory record per physical pool before producing RuntimeModelManifest;
- runtime manifest provenance is `native-adapter` only when both measurement and operator-requirement sources are native; mixed/fixture inputs degrade to fixture provenance;
- target-host admission orchestrator requires an intact admission-ready native collection, distinct reservation/launch observations, performs atomic reserve + fresh launch recheck, and constructs AcceptedExecutionBundle only after both gates pass;
- lease-bound native admitted session requires a live launched lease plus exact bundle/backend identity and deterministically releases the lease on completion, failure or cancellation cleanup;
- unmapped engine devices remain visible but are never auto-bound;
- dependency/license register and alternate pinned native-engine candidate;
- legacy analytical planner and bounded loopback diagnostics retained.

## Tested here / in portable CI

Contract tests cover policy, planner, adapter representability, exact model/evidence
identity, runtime availability, llama.cpp probe parsing/safety, explicit device binding,
strict backend-self-test parsing/promotion behavior, retained-E2 identity rules and
runtime-manifest identity/pool/operator invariants.

GitHub Actions separately validates portable Windows/Linux Python behavior and the real
optional GGUF dependency integration. Hosted CI is not GPU qualification.

The llama.cpp probe, binding, backend-self-test and retained-evidence portable tests use
harmless fixtures or synthetic records where appropriate. Runtime-manifest tests use
fixture provenance and do not claim native runtime measurements. Injected-runner
self-test results remain rejected from retained hardware evidence.

No real llama.cpp CUDA/HIP `test-backend-ops` execution, real llama-cli GGUF trial, or real native runtime-model
manifest has been recorded by TensorMeld in this development environment. Runtime-identity
CI uses fixtures and therefore proves invalidation/control semantics, not real driver or
GPU identity capture.

## Not tested / not implemented

- real target-host llama.cpp probe/binding/backend self test on CUDA/HIP hardware;
- real native-adapter model/operator/memory manifest capture on target hardware;
- native Windows GPU inference;
- native CUDA/HIP/distributed inference on real target hardware; the correctness evaluator exists but no real target-host E3 has yet been recorded;
- retained hardware/backend E2-E5 evidence from real target devices;
- expert/tensor/phase placement, multirail, GUI/API.

Runtime manifests are evidence inputs, not reservations. The local admission controller can
now reserve their exact preparation peaks atomically within one process and recheck before
launch, but this is not yet a distributed/agent lease service. Explicit reflected-lease IDs
separate telemetry that already includes an allocation from pending logical reservations so
committed bytes are not necessarily subtracted twice.

No measured RTX, Strix or model speed is claimed.
