# TensorFold and Strata audit for TensorMeld

Date: **2026-10-05**

This is a research note, not a qualification record and not a performance claim for
TensorMeld hardware.

Reviewed revisions:

- **Strata**: `mehbul/strata@e8647a84bac93cc2fae5f620726621e56fbc9659`
  (Apache-2.0).
- **TensorFold**: `ashhart/TensorFold@609ca419abecebdc5a059498a613680bd3aa847f`
  (TensorFold 0.6.5, Apache-2.0; older TensorFold code retains its stated MIT notice).

No source from either project is vendored or copied by this audit.

## Executive decision

TensorMeld should keep **llama.cpp as the first production compute engine** while the
TensorMeld control plane owns model identity, hardware identity, placement, qualification,
resource admission, topology, session lifecycle and eventually multi-node transport.

The architectural target is not "TensorMeld as a llama.cpp wrapper". The backend boundary
must stay replaceable so that measured bottlenecks can later move to TensorMeld-native HIP
or other kernels one path at a time.

The most useful external patterns divide cleanly:

- **Strata** is strongest as evidence for measured placement around llama.cpp on AMD:
  model facts from GGUF, context-aware VRAM/RAM trade-offs, empirical placement tuning and
  owned child-process lifecycle.
- **TensorFold** is strongest as evidence for exactness-first optimized execution:
  predictive memory admission, explicit resident/staging/mapped memory classes, cache
  identity, prefix/checkpoint reuse, read-ahead/streaming primitives and hardware-qualified
  optimization paths.

TensorMeld should combine those ideas under its stronger identity/evidence/admission
contracts rather than copying either architecture wholesale.

## What Strata actually does

The current Strata code explicitly separates its responsibilities from llama.cpp:
Strata parses GGUF, detects hardware, plans/tunes placement, chooses context, owns the
llama.cpp process, schedules requests and exposes the API. llama.cpp performs the matrix
multiplications.

That is close to the role TensorMeld should give llama.cpp in M3.

### 1. Compute-runtime-first hardware observation — ADOPT

`hardware.rs` asks the actual llama.cpp runtime for `--list-devices`. It deliberately
prefers the compute process's view over a second vendor SDK view.

This matches TensorMeld's existing probe/binding design and reinforces it:

- keep the engine observation as one evidence source;
- map engine-local names explicitly to TensorMeld device identities;
- never infer backend identity from the engine label;
- keep stable physical-pool identity separate from transient free memory.

Strata has one AMD-specific heuristic that TensorMeld should **not** copy: classifying an
integrated GPU from the device's human-readable name. TensorMeld already has the stronger
physical-pool model and should represent shared/unified memory explicitly.

### 2. Model facts from the file, not a registry — ADOPT / ALREADY ALIGNED

`gguf.rs` sizes tensors from the GGUF tensor index and reads architecture facts from GGUF
metadata. `placement.rs` separates routed-expert bytes from dense bytes and computes KV
cost from model geometry.

TensorMeld already treats the exact checkpoint/index as an identity boundary. The useful
extension is to add **movability classes** to future planning:

- always-hot / always-used weights;
- routed/sparse weights;
- KV/recurrent state;
- workspace;
- reclaimable file-backed data.

This is more general than Strata's "routed experts versus everything else" and applies to
dense, hybrid SSM and future architectures.

### 3. Estimate first, then measure placement — ADOPT

Strata's most important idea is in `tune.rs`.

Its static capacity estimate only supplies starting candidates. It then launches the real
engine repeatedly and measures candidate placements. It measures **prefill and decode
separately**, and ranks a candidate by wall time for a declared workload:

`prefill_tokens / prefill_rate + decode_tokens / decode_rate`.

The search is intentionally cheap: a coarse grid plus neighbouring values around the best
candidate. Tuning results are stored per context and KV type because changing the KV
reservation changes the best weight placement.

TensorMeld should implement the same *principle*, with stronger evidence:

- a `PlacementCalibrationRecord` bound to model, engine, worker/runtime identity, context,
  KV format, topology and workload class;
- separate prefill/decode measurements;
- a workload-weighted objective rather than a universal "tok/s";
- bounded coarse search followed by local refinement;
- measurements expire when any invalidating identity changes;
- planner estimates provide candidates, not truth.

Do **not** import Strata's fixed candidate fractions or its reported RX 7600 XT optimum as
TensorMeld coefficients. They are machine/model measurements, not transferable constants.

### 4. Context is part of placement — ADOPT

Strata chooses context before tuning because the full KV allocation competes with expert
residency. It stores tuning results per context/KV type.

TensorMeld should make this relationship first-class. A future calibrated plan key should
include at least:

- context window;
- reply reservation;
- concurrency;
- KV/state precision and layout;
- placement;
- physical-pool budgets.

The current TensorMeld runtime-manifest/workload contracts already provide most of the
identity surface needed for this.

Do **not** copy Strata's "KV <= 10% of VRAM" rule. It is an empirical rule from one tested
machine. TensorMeld should measure or model the trade-off per target.

### 5. Owned llama.cpp process lifecycle — ADAPT NOW

`runner.rs` owns the llama.cpp child, drains stderr continuously, retains a bounded log
tail, health-checks startup, notices early child exit and kills the child when its owner is
dropped.

TensorMeld's admitted-session/lease lifecycle is already stricter on resource ownership,
but the next native backend should adopt the missing process mechanics:

- explicit child handle;
- bounded stderr/stdout draining so pipes cannot deadlock the child;
- readiness deadline plus early-exit detection;
- bounded diagnostic tail attached to failed evidence;
- terminate, bounded wait, then kill fallback;
- release the TensorMeld lease only after process termination is confirmed.

The process environment must remain generated/allowlisted by TensorMeld; Strata's flexible
runtime discovery/environment behavior is useful operationally but is not sufficient as
TensorMeld evidence identity.

### 6. Stable-prefix compaction — ADAPT LATER

Strata's compaction deliberately makes the cut from a stable position near the **start** of
the conversation instead of "keep the newest N tokens". The useful property is that later
turns preserve the same prefix, allowing llama.cpp's prefix/KV cache to remain reusable.

TensorMeld should keep this as a future API/cache policy:

- context management should optimize semantic retention **and** cache-key stability;
- the rendered-token prefix identity, template identity and compaction identity belong in
  a session/cache fingerprint;
- compaction itself must not be mixed into low-level placement.

### 7. Disk is not automatically a useful weight tier — ADOPT AS POLICY

Strata refuses heavily spilled MoE plans because routed experts create scattered,
low-locality page faults. Its "disk_experts" is explicitly a warning/accounting class, not
an implemented fast disk tier.

TensorMeld should therefore **not** advertise NVMe as a generic third memory tier.

Disk/file streaming must be qualified by access pattern:

- sequential/read-ahead-friendly tables can be candidates;
- sparse random MoE expert traffic may be catastrophically unsuitable;
- a disk-backed placement needs its own E3/E4 correctness and measured path profile.

This becomes particularly important when borrowing TensorFold streaming ideas below.

## What TensorFold actually contributes

TensorFold 0.6.5 is no longer simply an MLX experiment. It contains separate MLX and CUDA
families, model-specific kernels, exact draft verification, concurrent-stream rules,
memory admission, checkpoint/prefix state and file-backed data paths.

The useful concepts are mostly backend-neutral even though their implementations are not.

### 1. Project future memory, not just current free bytes — ADOPT

`engine/memory.py` models:

- current held memory;
- each live stream's growth to its longest allowed reply;
- the new stream's projected cache;
- prefill working memory;
- the widest shared-round working set.

Admission uses the larger of prefill and shared-round work. It includes stepped cache
growth/slack rather than pretending KV grows byte-perfectly per token.

This should be TensorMeld's next major resource-model extension after real target
qualification.

The TensorMeld equivalent should add a **session growth contract** to the native adapter:

- fixed resident bytes;
- state/KV bytes per token or per allocation step;
- temporary resize-copy overhead;
- prefill working-set function;
- decode/shared-round working set;
- reply reservation;
- concurrency geometry.

LocalAdmissionController can then reserve projected session growth, not only startup
preparation peaks.

### 2. Resident, staging and mapped/reclaimable are different — ADOPT

TensorFold's CUDA capacity model distinguishes:

- resident weights;
- staging/loading peak;
- mapped read-only tables that the OS may reclaim;
- cache/workspace geometry.

TensorMeld currently has resident/state/workspace/preparation peaks, which is a good
baseline, but a future physical-pool record should distinguish **reclaimable/file-backed**
bytes from hard resident bytes.

On Strix Halo/unified memory this matters especially: host allocations, GPU allocations and
file cache compete for the same physical pool. Counting "GPU VRAM" and host RAM separately
would be wrong.

### 3. Measure memory geometry — ADOPT

TensorFold probes real cache growth and working-set peaks rather than relying only on
configuration formulas. It also repeats startup probes where peaks vary and sizes to the
worst observation.

TensorMeld should add native measurement provenance for:

- cache/state growth curve;
- prefill peak;
- decode/shared-round peak;
- startup staging peak;
- repeated-sample maximum or a declared percentile where appropriate.

Those measurements should feed RuntimeModelManifest/SessionGrowth rather than become
planner constants.

### 4. Exactness contract is broader than "same final answer" — ADOPT STRONGLY

TensorFold requires, within one engine/runtime/checkpoint/settings identity:

- serial versus drafted equality;
- one-row versus batched equality;
- resumed-prefix versus fresh equality;
- concurrent stream versus solo equality;
- rollback/state restoration correctness;
- memory admission before release claims.

TensorMeld's E3 contract currently proves one deterministic model output. It should grow an
**E4 session equivalence suite** with these dimensions before advanced scheduling,
speculation or cache reuse is enabled.

This is one of the strongest ideas to port.

### 5. Cache/snapshot identity includes execution details — ADOPT

TensorFold snapshots include model/runtime/kernel/chunk-plan identity, and some family paths
also include resolved matmul/prefill route and GPU identity.

TensorMeld should follow the same rule for any persisted KV/prefix state. A cache key should
include at least:

- model/checkpoint;
- tokenizer/chat-template;
- engine revision + worker artifact;
- kernel/precision route;
- placement;
- runtime/device identity as required by the backend;
- KV/state format;
- chunk/prefill plan;
- context/position semantics.

Changing arithmetic or layout must invalidate the cache rather than "try it and see".

### 6. Prefix checkpoint spill is not weight streaming — ADOPT LATER

TensorFold can evict conversation-prefix state from the in-memory cache to disk and reload
it. This is a much more attractive first disk use than random MoE weight streaming:
prefix checkpoints are coarse, explicitly addressed objects and can be checksummed.

TensorMeld should keep two separate concepts:

- **weight tiering/streaming**, which is access-pattern-sensitive;
- **session state spill**, which can be an explicit cache object with identity and checksum.

The latter should arrive first.

### 7. Direct read + pinned double-buffer read-ahead — ADAPT LATER

TensorFold's CUDA loader uses aligned direct reads where available, pinned host buffers,
non-blocking device copies, side streams/events and grouped read-ahead over neighbouring
tensor ranges.

This is highly relevant to future TensorMeld streaming, especially for discrete GPUs and
multi-machine staging. The portable idea is:

`storage -> bounded host staging -> async device/peer transfer -> consumer event`.

For Strix Halo, however, the best path may differ because CPU and GPU share physical
memory. TensorMeld must benchmark:

- mmap/page-cache;
- direct I/O;
- pinned/staged copies;
- zero-/low-copy ROCm paths;

rather than transplanting the CUDA implementation.

### 8. Adaptive speculative depth from measured costs — DEFER

TensorFold 0.6.5 measures draft/verify costs at startup and stops a draft chain when the
next verify row is expected to cost more than the draft can save.

This is a good future planner pattern: use measured marginal cost/benefit rather than a
fixed speculative depth.

It is not M3 work. TensorMeld should defer it until the native execution path and E4
equivalence suite are real on AMD.

## Adopt / adapt / defer / reject matrix

| Pattern | Source | TensorMeld action | Priority |
| --- | --- | --- | --- |
| compute-runtime device observation | Strata | keep current explicit probe/binding model | already |
| GGUF-derived size/architecture facts | Strata | keep exact index; extend to movability classes | P1 |
| measured placement tuning | Strata | **adopt** as calibrated planner layer | **P0 after first real target run** |
| separate prefill/decode objective | Strata | **adopt** | **P0** |
| tuning key includes context/KV | Strata | **adopt** | **P0** |
| owned child + health/log tail | Strata | **adapt** into real llama.cpp session backend | **P0** |
| fixed 10% KV rule | Strata | reject as global constant; measure per target | reject |
| fixed expert-split fractions | Strata | reject as transferable policy | reject |
| stable-prefix compaction | Strata | adapt at API/cache layer | P2 |
| generic NVMe weight tier | neither supports generically | require access-pattern qualification | defer |
| projected live-stream memory | TensorFold | **adopt** as SessionGrowth contract | **P1** |
| resident/staging/mapped classes | TensorFold | **adapt** physical-pool accounting | **P1** |
| measured cache/workspace growth | TensorFold | **adopt** into runtime measurement | **P1** |
| serial/draft/batch/concurrent equality | TensorFold | **adopt** as E4 equivalence | **P1** |
| kernel/runtime/chunk cache identity | TensorFold | **adopt** before persisted KV/prefix state | **P1** |
| checkpoint/prefix spill | TensorFold | adapt before weight streaming | P2 |
| direct-read pinned read-ahead | TensorFold | benchmark per backend/physical-pool type | P2 |
| adaptive speculative depth | TensorFold | adopt later after native AMD correctness | P3 |
| MLX/CUDA kernels themselves | TensorFold | do not port directly to HIP | reject direct copy |
| llama.cpp RPC for TensorMeld WAN/LAN execution | Strata/llama.cpp | keep disabled; use TensorMeld authenticated transport | reject for control/tensor transport today |

## Resulting TensorMeld sequence

### P0 — finish the real llama.cpp engine path

1. Implement a real child-process backend with explicit terminate/kill and bounded logs.
2. Run the existing TensorMeld qualification chain on the actual target machines.
3. Record first real E2/E3/runtime-manifest/admission/session evidence.
4. Add a calibrated placement record:
   - candidate placement;
   - context/KV settings;
   - prefill rate;
   - decode rate;
   - workload-weighted turn time;
   - memory peaks;
   - exact runtime/topology identity.
5. Use bounded coarse + local-refinement search. Never make benchmark results universal constants.

### P1 — make planning stateful and memory-predictive

1. Add tensor/memory movability classes to the model index.
2. Add SessionGrowth / cache-growth measurements.
3. Split hard-resident, staging and reclaimable/file-backed physical-pool bytes.
4. Add E4 equivalence:
   - fresh == resumed;
   - solo == concurrent;
   - serial == optimized/drafted where relevant;
   - rollback/cancel state exactness.
5. Define cache-state identity before any persistent prefix/KV cache.

### P2 — exploit I/O and cache hierarchy

1. Prefix/KV checkpoint retention and optional disk spill.
2. Directional path profiling for storage→host→GPU and node→node.
3. Benchmark mmap versus direct/pinned/read-ahead on discrete GPUs and Strix Halo separately.
4. Only then consider file-backed weight streaming for access patterns with demonstrated locality.

### P3 — replace llama.cpp selectively

Only replace a llama.cpp path when all of these are true:

1. profiling shows it is a material bottleneck;
2. TensorMeld has an AMD/HIP implementation candidate;
3. it passes exact/reference correctness;
4. it passes E4 session equivalence;
5. target hardware demonstrates a useful end-to-end improvement.

This allows llama.cpp to remain the reference/fallback engine while TensorMeld-native HIP
paths grow incrementally.

## Specific implication for the two Strix Halo machines

The external projects reinforce, rather than weaken, TensorMeld's physical-pool model.

Strata's discrete-GPU measurements should **not** be applied directly to Strix Halo.
TensorFold's unified-memory handling is more relevant conceptually: CPU allocations, GPU
allocations and file-backed pages can contend for one physical memory pool.

Therefore the Strix qualification profile should prioritize:

- one shared physical-pool budget per machine;
- measured live headroom rather than "VRAM + RAM";
- context/KV cost measured against that shared pool;
- mmap versus direct/staged I/O benchmarks on the actual Linux/ROCm stack;
- inter-node transfer profiling separately from local unified-memory bandwidth;
- no assumption that maximizing GPU-visible residency maximizes throughput.

## Licensing / reuse boundary

Both reviewed projects are permissively licensed at the reviewed revisions, but this audit
does not justify copying source.

TensorMeld should first reuse **ideas and public interfaces**. If a later implementation
wants to copy or adapt source, record that separately with:

- exact upstream revision and file;
- license/notice obligations;
- local modifications;
- reason reuse is preferable to independent implementation;
- tests proving that the imported path respects TensorMeld's security and evidence
  boundaries.
