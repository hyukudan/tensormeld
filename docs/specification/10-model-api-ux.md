# 10 — Models, client API and useful controls

Status: proposed. This does not implement any HTTP endpoint or graphical interface.

## Exact model catalog and safe import

A model entry resolves to an immutable manifest: checkpoint revision/shard hashes,
architecture/config, tokenizer and chat template hashes, weight encodings, state
layout, tied aliases, optional adapters and projectors, task/modalities and licensing
metadata. An alias is a user label, not permission to replace that manifest silently.
Model licensing/redistribution permission is not granted by this software.

Local files are the P0 import path. Optional authorized registry downloads are P1,
with confirmation, size/quota checks, pinned revision, resumable transfer and hashes.
Model code, pickle or peer-provided commands are not executed. Backend-specific cache
formats are declared, bounded and integrity-checked. Do not promise that only shards
are needed when the selected coordinator's loader needs a full checkpoint.

Data transfer is resume-safe, cancellable and separately scheduled from interactive
inference. Cache quotas, available disk and temporary duplication are checked before
transfer. A user can pin/evict an unused model; active in-use content cannot vanish.
Transfers remain within permitted nodes and exact authorized model scope.

## Fit assistant and semantic contract

The assistant answers whether an exact workload fits, which resources enable it and
what prevents alternatives. Evaluate weights AND context state, graph workspaces,
load peaks, per-unit indivisibility and backend support. Report insufficient device
VRAM, host RAM, disk/cache, unsupported unit/cut, unavailable node and missing profile
as distinct causes.

When the request cannot fit, propose alternatives such as a different approved
checkpoint/encoding, fewer concurrent requests, different context, another device
subset or explicit CPU offload. Each is separately labeled. Automatic mode may tune
microbatching or placement only within user-approved semantics. It MUST NOT silently
change weight/KV precision, chat templates, truncate history or lower output/context.

The context bound includes tokenized template/input plus permitted output and any
architecture-specific state budget. Reject oversize requests or request explicit
user action. No automatic infinite-context shifting/truncation is part of P0.
A larger context plan can use helpers even when weights fit locally; isolated KV
migration is a distinct optional strategy requiring adapter support.

Dense versus MoE is a capability/placement dimension, not a model catalog hierarchy
where only MoE is useful. Models below a fixed parameter count are not excluded.
A machine with 12 GB needs an exact byte/context analysis, not a blanket 'small model'
label. User-defined models can be imported without changing scheduler source code.

## Public inference surface

The first stable surface is text inference using a documented subset of common
chat-completion semantics: `/health`, `/v1/models`, `/v1/chat/completions`, streaming,
usage, bounded sampling controls, cancellation and structured errors. Compatibility
is a published field/behavior matrix validated against clients, NOT a claim of
complete compatibility with every provider endpoint. Control operations use a
separate versioned namespace such as `/companion/v1/`.

Model listing exposes tasks, input/output modalities, context bounds, loaded status
and exact feature gates. Unsupported options are rejected; they are not silently
ignored or stripped. A model being present on disk does not mean ready or qualified.
Private backend endpoints are never a supported external integration interface.

Control actions cover inventory, enrollment, policy/config revision, dry-run plans,
profiles, prepare/load/unload, cancellation, drain, model/cache lifecycle and safe
export. Admin and inference scopes differ. Client authentication is not a node
certificate and cannot grant arbitrary worker execution.

Streaming preserves request/attempt identity, ordered deltas, finish reason, usage
and terminal errors. Client disconnect triggers bounded cleanup by default unless
an explicitly supported detached-job mode is enabled later. Token counting and
sampling use the selected model contract, not a generic tokenizer approximation.

## Optional API capabilities

P1/P2 features are separately gated by model AND adapter: structured/schema output,
tool-call formatting, embeddings, reranking, LoRA/adapters, multimodal projection and
model-specific reasoning-output fields. Text support does not certify these paths.
No agent is allowed to execute returned tool calls; execution belongs to the client
under its own permissions. Speculation/MTP requires acceptance/rollback tests.

A compatible external chat/editor UI may use the stable endpoint. A preexisting
whole-model server cannot automatically become an operator/tensor worker; it needs
a different adapter contract. Endpoint interoperability is not distributed execution.

## Minimal useful UI

Provide three main views rather than a large monitoring suite:

1. Machines/resources: enrolled versus online/eligible/draining, physical pools,
   budgets, devices, selected interfaces and per-model availability.
2. Models/plans: exact workload, local/companion/distributed alternatives, expected
   load and first-token/decode metrics, omissions and actionable rejection reasons.
3. Running work: stream state, selected nodes, measured resource/traffic timeline,
   stop/drain/release controls and optional redacted diagnostics export.

Simple controls: automatic/manual selection, maximum participating nodes, local GPU
participation, usable VRAM/RAM ceiling, context/output limit and preserve-desktop
profile. Advanced controls: device filters, legal graph cuts, route preferences,
CPU fallback, state dtype, concurrency and profile validity. Disabled features show
why they are unavailable instead of offering nonfunctional switches.

Named profiles are intent/resource policies, not fixed speed promises. Examples are
interactive, capacity and preserve-desktop. A future energy profile requires measured
power coverage; using fewer hosts alone is not proof of lower energy per token.

## Planning result and error UX

Each proposed plan explains participating/omitted nodes, coordinator and routes,
per-pool resident/load/state peaks, model contract, source/confidence of estimates,
search completeness, supported scale and required approvals. Compare cold load,
warm first-token and sustained generation separately. Do not turn synthetic data
into an installation benchmark by displaying it on a dashboard.

Core error classes also include MODEL_UNSUPPORTED, ENCODING_UNSUPPORTED,
CONTEXT_EXCEEDED, CPU_FALLBACK_DISALLOWED, NO_FEASIBLE_PLAN (only when established),
UNQUALIFIED_PLAN, RESOURCE_PRESSURE, REQUEST_QUEUE_FULL, NODE_LOST and
SESSION_STATE_LOST. Messages propose relevant actions without asserting that a
hardware purchase will fix an unmeasured bottleneck.
