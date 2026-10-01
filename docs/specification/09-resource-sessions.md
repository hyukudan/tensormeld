# 09 — Resource sharing, model sessions and recovery

Status: normative design. No distributed lease/scheduler is implemented yet.

## Per-host authority and memory admission

Every host owner sets local ceilings. The agent performs atomic reservation against
ALL physical pools required by the worker, not just its GPU. This includes pinned
host pages, tensor staging, mapped/repacked file pages, graph arenas, CPU experts,
worker metadata and state. OS paging is not a planned fast memory tier. Explicit
bounded CPU offload is different from unplanned driver eviction or OS swapping.

A practical budget calculation reconciles a fresh OS/backend observation, the
worker's current allocation, the policy cap, safety headroom and reservations not
already reflected in observed usage. Implementations MUST document this accounting
to avoid subtracting committed allocations twice. Atomically charge the incremental
allocation required by the new session; do not assume a telemetry sample is a lease.

The first fit check precedes loading, a second precedes READY, and runtime pressure
is monitored. Memory for the maximum admitted context/output/concurrency is reserved
or guaranteed by an explicitly bounded growth policy. A weights-only successful load
is not admission. Loading transients and coexisting old/new model weights count.

Windows budget sampling must run in or be attributable to the actual worker where
process budgets apply. A controller's global memory reading may be insufficient.
The resource guard preserves configured headroom; it does not promise instantaneous
reclamation from every application or a fixed GPU execution-time share.

## Safe response to pressure

Default actions: stop new admissions, report pressure, evict unused caches where
safe, drain/release idle loaded sessions, then cancel under the configured deadline
when remaining usage is unsafe. Do not silently shrink context, drop past tokens,
change quantization, kill unrelated applications or migrate live state.

A foreground-preservation toggle applies to new plans immediately and drains active
sessions. A cancel-and-release action is explicit and reports when buffers/workers
are actually released. It must not claim memory is free merely because cancellation
was requested. Driver/runtime failures may require a process restart; report it.

## Admission and bounded queue

P0 admits one active conversation per qualified loaded model configuration and uses
a bounded queue with timeout, cancellation and request-size limits. P1 can add bounded
concurrency/fairness when shared weights, per-request state and workspace interactions
are correctly modeled. Reject excess load rather than buffer prompts indefinitely.

Distinguish registered models, loaded models, model replicas, sessions and requests.
Two HTTP clients using one model do not imply two weight copies; two isolated workers
may require them. Shared prefixes/weights count once only when ownership and lifetime
are proven. Node-local lease arbitration protects against two plan preparations
spending the same memory. Distributed preparation rolls back partial reservations.

Request priority cannot starve interactive work or exceed owner limits. An explicit
batch/throughput profile never overrides an active desktop-preservation policy.
Multi-model routing is a later scheduling feature, not a substitute for model splitting.

## Session lifecycle and idle behavior

A loaded session binds model/adapter/plan/config identities and leases. Requests bind
input, sampling, context, epoch and token sequence. The API distinguishes queued,
preparing, loading, ready, running, draining, cancelled, failed and released.
Warm/cold estimates distinguish disk cache, RAM residency, GPU residency and prefix
cache. An idle unload can release weights/state; an on-disk verified cache may remain.

Idle unload/warm reload is P1. It must coordinate every worker in a split model and
report the reload penalty. A sleeping OS host is not merely an unloaded model.
No automatic OS suspend or BIOS wake-on-LAN setup is performed. Optional future
wake features require explicit ownership, platform support and re-qualification.

Node membership changes affect new plans. Removing or lowering the budget of an
active owner requires drain/cancel. A request does not gain capacity merely because
a new machine was enrolled. No token-level resharding or seamless failover is implied.

## Failure semantics and streaming commitment

Each request carries a stable ID, attempt ID, plan ID, epoch and monotonic output
sequence. Receiving a transport frame is not committing model state or client output.
A worker crash invalidates its state and fails dependent requests. A recovered node
must be revalidated before new work. Revocation fences the node immediately.

Before any output is delivered, the service may propose/perform only a bounded
retry permitted by policy, with a new attempt and complete re-admission. After output
has begun, the default is a structured terminal error and explicit client restart;
never append a fresh generation as if it continued lost state. Reproducible sampling
is separately qualified; identical seeds do not guarantee cross-backend bit identity.

Link failure can fall back only where the transport preserves ordering, byte delivery,
request identity and buffer ownership. Otherwise fail cleanly. Link redundancy cannot
restore lost GPU state. Retries/backpressure are bounded and never free owned buffers.

## Local privacy and observability

Prompts and outputs are not persistently logged by default. Optional history is local,
explicitly enabled and has retention/export/deletion settings. Prefix/state caches
are scoped to model, tokenizer, adapter, numeric layout and authorized session scope;
no cross-user cache disclosure. Cache sharing is not presumed safe merely on a LAN.

Every request records non-content metrics: queue delay, model load, prefill, first
output, inter-token percentiles, accepted output tokens, pool peaks, transfer waits,
selected routes, cancellation latency and resource release. Record unavailable
metrics explicitly. Power/noise readings are optional and cannot be fabricated.
