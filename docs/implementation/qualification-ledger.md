# Implementation and qualification ledger

Updated: 2026-10-12. This is the current cross-cutting summary. Older slice notes
and decisions remain historical records. A portable test pass does not establish
native model, transport or performance qualification.

| Area | Implemented and portable-tested | Native/product gate still open |
| --- | --- | --- |
| Registry and policy | Heterogeneous nodes/devices, aliased physical pools, bounded synthetic search, exclusions and participation rules | Live inventory and exact backend realization on target hosts |
| Qualification and admission | Artifact-bound E2/E3 chain, current runtime applicability, host-local leases and launch recheck | Actual target GPU/model evidence and persistent cross-process authority |
| Sessions | Synchronous admitted-worker lifecycle; serialized run/cancel/release; lease retained through backend termination | Real model process cancellation and descendant cleanup |
| Generation cost | C1 ordered-chain prefill/decode composition, directional boundaries, sampler and first-owner token feedback | Native component measurements, additional token consumers, end-to-end generation and capacity admission |
| Request family | Durable declaration-only identity/limit/feature preflight, with variable rendered input digest | Trusted rendered-token correctness producer, request-family evidence and actual serving consumer |
| C1 serving consumer | Actual existing authorized completion path with bounded FIFO count/bytes, deadlines, attempt cancellation, original-process fencing, quarantine and drain | Exact replay scope only; arbitrary-prompt family serving, public HTTP/SSE and native cancellation remain unqualified |
| Numerical profiles | Canonical identity and exact phase/shape/runtime/state/wire applicability; same-profile authoritative verify gate | Native arithmetic/quality/equivalence and measured complete-cost evidence; scaled FP4/FP8 formats remain unsupported in v1 |
| Local mature-engine placement | Exact block mapping through the pinned local adapter; remote/RPC devices rejected | Target-native tensor/state ownership and qualified serving |
| Native distributed worker | Separate engine implements placement/control, a compiled experimental Linux DMA-BUF extension, concrete syscall/HIP providers and C1 GPU boundary staging | Same-family two-host model/state continuity, external-DMA coherence and boundary qualification |
| Numerical experiments | Native repository has arithmetic references, offline instruction emission, a tiled integer LM-head adapter and complete C1 graph capture/replay with compiled native fixtures | Native fragment correctness, full-model quality, state equivalence and committed-token benefit |
| CUDA and heterogeneous pairs | Core resource model and adapter boundaries accommodate distinct backends | Each CUDA device/backend qualified independently; mixed pair compatibility and performance |
| Hardware campaign preparation | Private commissioning templates, repeated imported-pool sessions and paired receipt/oracle checks, native full-kernel staging, eager/graph/integer RAW collection and same-plan restored-state import | Actual target configuration/boot and native evidence; matching recordings never authorize production and paired schedules do not prove physical DMA overlap |

The last completed Windows TensorMeld portable suite ran 686 tests, with five optional
tests skipped; Linux ran the same 686 tests with six optional skips. Skips do not
count as native qualification. The native repository's
portable Release integration passed 143 CTest entries. Results are local engineering
checks, not a public model-performance claim.

## Immediate development order

1. Distinguish durable numerical/request-family declarations from exact replay
   qualification and per-launch admission. The declaration parsers are implemented;
   the exact-replay C1 consumer is implemented. Next extend it only with trusted
   rendering/tokenization, family evidence and a bounded public terminal protocol.
   Keep unsupported requests non-executable and require genuine family evidence.
2. Qualify the implemented experimental direct-transport ABI and pair capability
   preflight on actual hosts. Device discovery alone must not imply GPU import,
   visibility or DMA.
3. On the configured target hosts, collect artifact-bound operator/model and
   transport evidence. Preserve negative results and exact provenance.
4. Qualify local-only and whole-model companion baselines before selecting real
   split execution. PC front-door ownership does not require relaying all activations.

See [generation scenario](generation-scenario.md), the
[closed local campaign runner](qualification-campaign.md), the
[current status](status.md) and [roadmap](roadmap.md). Personal deployment inventory,
comparative research and review correspondence remain outside Git.
