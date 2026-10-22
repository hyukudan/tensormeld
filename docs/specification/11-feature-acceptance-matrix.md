# 11 — Feature priorities and acceptance matrix

Specification 0.2.0-draft. Every test below is a FUTURE acceptance test; none is
claimed as part of the unchanged 54-test M0 suite. P0 includes several milestones,
not a promise that all P0 features ship in the next commit.

| ID | Priority | Feature | Contract | Acceptance | Gate | Current reality |
|---|---|---|---|---|---|---|
| F01 | P0 | Generic nodes/devices/pools | No fixed companion count or 96 GB/MoE eligibility | T01–T05 | M0.2 | v1 has only device records; limit 3 |
| F02 | P0 | Auto/manual subset policy | Required/allowed/excluded; compute node and device counts separate | T06–T09 | M0.2 | Not implemented |
| F03 | P0 | Multi-GPU and control-only roles | Coordinator is node/process; alias detection | T04–T05 | M0.2 | Partial aliases; coordinator tied to device |
| F04 | P0 | Bounded scalable search | Deadline/cancellation; incomplete != infeasible; exact small oracle | T10–T12 | M0.2 | Only bounded exhaustive v1 |
| F05 | P0 | Exact model/semantic contract | Dense/MoE; no silent precision/context/model change | T13–T16 | M0.2/M2 | Supplied synthetic stages only |
| F06 | P0 | Safe multi-pool memory | Load peaks, pinned RAM, state, live budget and local lease | T17–T20 | M0.2/M1/M2 | Partial analytical shared-pool accounting |
| F07 | P0 | Secure node lifecycle | Pairing, revocation, owner limits, bounded probes | T21–T24 | M1 | Not implemented |
| F08 | P0 | Native worker/placement proof | Pinned engine; exact plan lowering or rejection | T25–T26 | M2 | Not implemented |
| F09 | P0 | Local and companion-only | Local GPU can be omitted; CPU explicit | T27–T29 | M2 | Only advisory subset exploration |
| F10 | P0 | Real distributed model | Same-model capacity and correctness, first 2 then 3 hosts | T30–T32 | M3 | Not implemented |
| F11 | P0 | Actual-route network costing | Multiple interfaces, slow link, relays and shared resources | T33–T35 | M1/M3 | Only supplied direct/via-device costs |
| F12 | P0 | Bounded serving/API | Streaming/cancel/errors; stable endpoint; bounded queue | T36–T38 | M3 | Not implemented |
| F13 | P0 | Verified cache/load lifecycle | Hashes, resume, quotas; no per-token bank streaming | T39–T40 | M3 | Not implemented |
| F14 | P0 | Drain/failure/resource release | No silent continuation after state loss | T41–T44 | M3 | Not implemented |
| F15 | P0 | Evidence and compatibility scope | Predicted vs measured; build/OS/model/scale tuple | T45–T47 | All gates | Advisory labeling and local unit tests only |
| F16 | P1 | Desktop profiles and idle unload | Foreground headroom, safe release/warm reload | T48–T49 | M4 | Not implemented |
| F17 | P1 | Dashboard/client integration | Simple and advanced controls use same API; unsupported options fail | T50–T51 | M4 | Not implemented |
| F18 | P1 | Bounded multi-session fairness | Shared weights/state, multiple clients, no over-admission | T52–T53 | M4 | Not implemented |
| F19 | P1 | Catalog/download/discovery UX | Approved revision, local offline path, discovery never trust | T54–T55 | M4 | Not implemented |
| F20 | P1 | Packaging/update/diagnostics | Native services; provenance; no secrets in export | T56–T57 | M4 | Not implemented |
| F21 | P2 | Fine-grained MoE/tensor/phase splits | Independent backend/numeric/path qualification | T58 | M5 | Not implemented |
| F22 | P2 | Speculation/MTP | Committed-token benefit with correct rollback | T59 | M5 | Not implemented |
| F23 | P2 | Multirail/direct transport | Proven concurrency, ownership and secure fallback | T60 | M6 | Not implemented |
| F24 | P2 | Additional tasks/adapters/replicas | Gated APIs; service routing != tensor cooperation | T61 | M5 | Not implemented |
| F25 | P2 | Energy and advanced availability | Measured energy; wake policies never imply failover | T62 | M5+ | Not implemented |

## Concrete acceptance tests

T01: v2 registry/config round trips with 1, 2, 3, 4, 8 and 16 nodes; no discarded IDs.
T02: 12 GB and 96 GB synthetic capacity profiles follow the same schema/solver path.
T03: dense and MoE manifests are admitted/rejected by operators/bytes, never family size.
T04: two local GPUs plus one remote GPU count as two compute nodes and three devices.
T05: CPU-only control/coordinator host; GPU alias views and UMA pools are not duplicated.
T06: auto excludes an eligible but slower/unnecessary helper with an explanation.
T07: manual exact compute set cannot omit a selected owner; UI-only role counts correctly.
T08: contradictory filters, empty allowlist, unknown ID and min/max violations fail early.
T09: required offline/disabled/draining node fails; local GPU exclusion is honored.
T10: heuristic plans are checked against a bounded exhaustive oracle on small fixtures.
T11: timeout with no solution is SEARCH_INCOMPLETE; no false infeasibility/optimality.
T12: bounded planning/profiling on large registries; adapter scale limits remain explicit.
T13: unsupported operator, encoding, graph cut or required adapter feature fails pre-load.
T14: weights fit but requested context does not; no automatic truncation/precision change.
T15: total memory fits but an indivisible tensor/mandatory workspace does not; explain it.
T16: tied tensors, output heads, shared experts and recurrent state preserve dependencies.
T17: host+VRAM demand, load/repack peak and UMA aliases do not over-admit memory.
T18: two competing plans atomically reserve the same node/pool; only a fitting set succeeds.
T19: lowered live/owner budget invalidates new admission; own usage is not deducted twice.
T20: 12 GB budget on a larger GPU is labeled synthetic/constrained, not a 12 GB benchmark.
T21: unpaired/wrong/revoked certificates and incompatible versions are rejected.
T22: node-local resource limits cannot be raised by remote user profile overrides.
T23: bounded malformed/frame/path/epoch tests; no peer-supplied code or shell launch.
T24: bounded probes respect active work, local caps and actual enrolled paths.
T25: pinned CUDA/HIP workers execute tiny fixtures; build success alone gives no qualification.
T26: adapter proves exact cut/route/encoding or rejects; no approximate memory-ratio claim.
T27: local-only excludes all remote compute; no mandatory helper.
T28: companion-only leaves PC GPU unused and keeps the stable client endpoint.
T29: CPU offload requires explicit permission, declared memory and recorded execution.
T30: low-VRAM dense target runs on real relevant hardware, with complete comparison evidence.
T31: large sparse/MoE target exceeds local capacity and executes with verified state/correctness.
T32: 2-host then 3-host real execution; beyond tested scale remains unqualified.
T33: slow PC link plus fast companion link; trace reveals every actual relay/copy.
T34: two same-host GPUs and remote GPU paths remain distinct; missing paths are not invented.
T35: no additive dual-link bandwidth without concurrent measurement/implementation; shared uplink counted.
T36: stable chat stream, usage, model identity and feature-specific errors pass client fixtures.
T37: cancellation/client disconnect/queue limits release bounded resources and return clear outcomes.
T38: administrative scopes, local origin/host checks and privacy defaults prevent unintended access.
T39: interrupted/corrupt model transfer resumes/verifies or fails; quota and load peak respected.
T40: warm cache avoids unnecessary load traffic; active shards cannot be evicted mid-request.
T41: drain blocks new ownership and releases only after safe completion/cancellation.
T42: worker loss after emitted tokens terminates explicitly; no invisible restart concatenation.
T43: sleep/wake/new address gives authenticated fresh epoch; no stale tensor commits.
T44: node/control-owner revocation or expired lease fences stale work; partial prepare rolls back.
T45: unknown/estimated/measured/qualified states remain distinct after export/import/UI rendering.
T46: profiles invalidate on model/backend/driver/route/state/budget changes as applicable.
T47: release publishes physical platform/model/scale envelope; software tests do not upgrade it.
T48: foreground-preservation cap reduces admission safely without unsupported GPU-share promises.
T49: idle unload frees model resources on every worker; reload is cold/warm labeled.
T50: CLI and dashboard apply identical config precedence and show reasons for omitted nodes.
T51: unsupported tool/JSON/multimodal/client fields fail, not silently ignored.
T52: concurrent clients share only proven allocations; state isolation and fairness hold.
T53: model switch/replicas do not double-spend resources or imply combined model capacity.
T54: approved downloads are pinned, quota-limited, cancellable and usable offline after preparation.
T55: discovery does not auto-enroll; address changes preserve identity checks.
T56: approved artifact update is staged and compatible; active sessions are not hot-swapped.
T57: diagnostic/config export strips secrets/prompt content and distinguishes synthetic examples.
T58: optional fine split preserves output/state contract and measures complete-path benefit.
T59: MTP/speculation records accepted tokens, rollback, temporary memory and net user benefit.
T60: optional multirail/direct buffers prove ordering/visibility and qualified secure failure behavior.
T61: added task/adapter/replica advertises precise support; whole-model endpoint is not operator RPC.
T62: energy/wake features require measured coverage and platform permission; no inferred failover.

## Decision rule

A feature may be specified before implemented. It may be implemented before hardware
qualified. All three states are separate. A failing optional optimization stays off;
a failing P0 safety/semantic invariant blocks qualification of the affected path.
