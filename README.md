# TensorMeld

> [!CAUTION]
> **TensorMeld is in very early, active development.**
>
> This repository is public **for transparency and open technical discussion only**. It is **not an alpha release**, is **not ready for general use**, and should not currently be relied on for production workloads, benchmarking claims, security-sensitive deployments, or stable APIs.
>
> Expect incomplete features, breaking changes, experimental code paths, incorrect planning decisions, unfinished documentation, and repository history that may change substantially while the architecture is being developed.

## TensorMeld in 30 seconds

**TensorMeld is the planning and routing layer for heterogeneous local AI inference.** It aims to match compute, memory and interconnect constraints to a suitable execution placement across different devices and hosts—without pretending that remote RAM is local GPU memory.

| If you are looking for… | Start here |
|---|---|
| Project readiness | [Current status](#current-status) and [What not to expect yet](#what-not-to-expect-yet) |
| Why different devices need different placements | [Why TensorMeld?](#why-tensormeld) |
| Actual measured planning experiments | [Heterogeneous placement](#heterogeneous-placement-working-decisions) |
| Security boundaries | [Security model](#security-model) |
| Implementation map | [Repository layout](#repository-layout) |
| Inference execution on Strix Halo | [Strix Halo Engine Lab](https://github.com/hyukudan/strix-halo-engine-lab) |

**Important distinction:** TensorMeld is **not** the engine producing the standalone Strix Halo token-generation numbers, and it is **not yet a stable distributed inference product**. [Strix Halo Engine Lab](https://github.com/hyukudan/strix-halo-engine-lab) owns native execution; [Hyperloom](https://github.com/hyukudan/hyperloom) is the integration/control project. Components and experimental adapters should not be mistaken for an accepted end-to-end deployment.

## AMD-first implementation, independent inference roadmap

**AMD-first in engineering priority; hardware-neutral in the orchestration core.**
Our first native implementation and performance-qualification target is AMD Ryzen AI
Max / Strix Halo (`gfx1151`), with ROCm/HIP, unified memory and direct USB4
interconnects. We design and measure for this hardware first, while keeping
resource discovery, placement and control contracts capable of accommodating
CUDA and other devices later. AMD-first is a development priority, not a claim
that every AMD GPU or distributed topology is already supported.

**Our own inference engine is the intended primary AMD execution path.**
[Strix Halo Engine Lab](https://github.com/hyukudan/strix-halo-engine-lab)
develops the native C++/HIP model execution and token-generation kernels used
for the published single-node Strix Halo experiments. TensorMeld is the
companion orchestration project: it is **not** currently the engine executing
those native generation benchmarks, and integration of that engine as a
qualified TensorMeld backend and real multi-node inference remain future work.

**No mandatory third-party inference runtime for the intended native path.**
The current pinned llama.cpp adapter serves bounded compatibility,
qualification and reference experiments; it is not the dependency behind
our standalone Strix Halo engine's generation results. External inference
engines may remain optional adapters, comparison baselines or sources of
technical ideas, but they should not be required to keep our own native
AMD-first execution path running. This does not mean eliminating necessary
AMD ROCm/HIP, Linux, compiler, model-format or other infrastructure
dependencies. We distinguish independently implemented execution code
from adopted components and preserve third-party attribution.

## What TensorMeld is trying to become

TensorMeld is an experimental local AI inference orchestration project for combining heterogeneous compute resources across one or more machines.

The long-term goal is to let a user start with the hardware they already have—for example a PC with a discrete GPU—and add one or more companion systems to expand usable model capacity and, where the topology makes it worthwhile, inference performance.

TensorMeld is intended to remain hardware-neutral at its core. Early development is focusing on combinations such as:

- Windows and Linux hosts;
- NVIDIA CUDA devices;
- AMD ROCm/HIP devices;
- unified-memory systems such as AMD Strix Halo;
- multiple GPUs in one host;
- multiple compute nodes;
- different interconnects with very different latency and bandwidth characteristics.

The project is not limited to very large models. The same architecture should eventually support anything from a 12 GB desktop GPU assisted by one companion node to much larger heterogeneous systems.

## Why TensorMeld?

Local AI hardware is increasingly heterogeneous.

A desktop GPU may provide very high compute throughput and fast local VRAM, but its memory capacity can limit the models, quantizations, contexts, or workloads it can run. A companion system may have much more usable memory—especially unified-memory machines such as Strix Halo—while being slower for some operations than the discrete GPU.

Those resources should not necessarily be treated as interchangeable, and TensorMeld is not trying to pretend that remote memory is simply extra VRAM.

The idea is instead to **place each part of a workload where it makes the most sense**.

For example, a future TensorMeld plan could decide to:

- keep compute-heavy or latency-sensitive operations on a fast discrete GPU;
- place model blocks, experts, state, or other memory-heavy components on a high-capacity companion;
- use more than one companion when the additional capacity or parallelism justifies the communication cost;
- keep an entire model on a companion when splitting it would be slower;
- leave a device unused when involving it would make the workload worse;
- reserve part of the main GPU for the desktop or another application;
- choose a different placement for capacity, latency, throughput, or long-context goals.

A simple example is a PC with a 12 GB GPU plus a 128 GB unified-memory companion. The discrete GPU may be substantially faster for suitable kernels, while the companion makes models possible that cannot fit in 12 GB at all.

At the other end of the spectrum, a workstation with a 96 GB discrete GPU may already run many models locally, but one or more high-memory companions could make larger models, higher-precision quantizations, longer contexts, or different placement strategies possible.

The hard part is that **more aggregate memory and more devices do not automatically mean better inference**. Every split introduces communication, synchronization, memory-placement, backend-compatibility, and scheduling costs. A mathematically balanced partition can still be much slower than a simpler one.

TensorMeld therefore aims to answer a more useful question than “how do we use every GPU?”:

> **Given this model, this workload, these devices, these memory pools, and these links, which resources should participate, what should each one own, and is the resulting plan actually worthwhile?**

That decision should eventually be based on measured capabilities and explicit constraints rather than fixed assumptions about vendor names or device classes.

## Current status

**Status: pre-alpha / architecture and implementation work in progress.**

At this stage, TensorMeld is primarily concerned with getting the foundations right:

- resource discovery and capability descriptions;
- explicit node, device, and physical-memory-pool modelling;
- memory-aware placement planning;
- configurable participation of local and remote devices;
- bounded planning across multiple nodes;
- topology and communication-cost modelling;
- backend adapter boundaries;
- correctness, observability, recovery, and security contracts;
- evaluation of existing open-source runtimes that can be reused instead of reimplemented.

Some components may already execute or have automated tests, but that **does not mean the complete distributed inference system is ready or validated**.

The [local qualification campaign runner](docs/implementation/qualification-campaign.md)
supports pinned operator experiments, selected CTest suites and exact approved
model trials. It retains bounded results privately; running a campaign does not
grant model qualification, resource admission or serving authorization.

In particular, current code and documentation must not be interpreted as proof that a given model, GPU combination, network topology, or placement strategy will work correctly or outperform a simpler setup.

## Heterogeneous placement: working decisions

This section records the placement decisions we are currently working under, and the evidence behind each. They are working decisions, not accepted ADRs: they may change as measurements come in. The execution code lives in the [engine](https://github.com/hyukudan/strix-halo-engine-lab) branches and is not yet a qualified TensorMeld backend.

**Test system.** One Windows workstation with a 96 GB RTX PRO 6000 and an RTX 4090, plus two Strix Halo nodes (128 GB unified memory each). The two nodes are joined by two direct USB4 cables. The workstation reaches them over 1 GbE.

**First target model.** DeepSeek V4.1 Flash 1017. Its experts use the Q2 GGUF quantization published by [@kyuz0](https://huggingface.co/kyuz0). Today the model runs on the two Strix Halo nodes, with experts split between nodes and tensor-parallel dense parts.

**Second model.** Qwen3.8-Flash-Next, in a mostly 4-bit GGUF quantization (111 GB: routed experts in Q4_K / Q5_1 with a few layers kept at higher precision, per-layer token embeddings in IQ4_NL, attention and shared parts in Q8_0), plus a distilled Q8_0 draft head for speculative decoding. Each Strix Halo node holds a complete copy, and the router sends each request to one of them (decision 24).

**Third model (bring-up).** MiniMax M3.1 Flash, EXL3 trellis weights at 3.06 bits per weight, as a tensor-parallel pair over USB4 with its own engine, following the same distribution recipe as DeepSeek (one request on the whole pair; replicas first when more copies fit). Its numbers are preliminary.

### Measured on 2026-10-24

Every number below is measured, greedy, and exact against its reference unless the row says otherwise
(token ids, logits digests or state bytes, as stated). *Release* is the public engine release of 2026-10-23;
*pre-release* means an integration build measured after it, whose code lands in the next engine release.
Older rows keep their own date. The decisions table further down keeps the evidence of 2026-10-23 and earlier.

**DeepSeek V4.1 Flash, two Strix Halo nodes, expert tensor parallel (TP2) over USB4.**

| What | Value | Date | Build | Conditions |
|---|---|---|---|---|
| Greedy decode, 1 stream | **30.6 ms/token = 32.7 tok/s** (production build in the same session: 34.7 ms/token) | 2026-10-24 | pre-release | interleaved A/B ×3; both ranks identical to the stored reference in every run |
| Verify window of 6 rows | **58.92 ms** (production 63.25 ms) | 2026-10-24 | pre-release | same runs |
| Batched decode, 8 streams, no drafting | **78.97 ms/step = 101.3 tok/s** aggregate (release: 95.9) | 2026-10-24 | pre-release | 8 prompts × 48 tokens; tokens identical; interleaved A/B ×3, ranges disjoint |
| Prefill, time to first token | 1K **2.20 s** (465 tok/s); 8K **10.78 s** (760 tok/s; production 11.21 s in the same session); 16K **20.40 s** (production 21.36 s) | 2026-10-24 | pre-release | logits identical on both ranks and between builds; 16K: A/B ×3 |
| Prefill, time to first token, 32K | **39.55 s** (828 tok/s), 2 runs (41.37 s without the new prefill defaults) | 2026-10-23 | pre-release | logits identical |
| DSpark speculative decode, 24 prompts × 256 tokens | **62.3 tok/s** on code; 37.5 English, 39.3 Spanish; 46.4 over all 24 | 2026-10-22 | release | ids equal plain greedy |
| Split with the RTX PRO (layers 0-21 on the RTX PRO, the rest on the pair), DSpark | **81.1 tok/s** on code; 61.8 over all 24 (×1.33 against the pair alone); greedy 38.3 tok/s | 2026-10-22 | release | declared composed identity with the trellis-coded sites (KL 1.7e-3 against the pair); bitwise equal to the pair without them |
| Exact prefill on the RTX PRO, state handed to the pair | time to first token **3.64 / 4.39 / 12.20 s** at 2K / 8K / 32K | 2026-10-22 | release | warm resident service, load excluded; state identical to the exact reference |

**Qwen3.8-Flash-Next.**

| What | Value | Date | Build | Conditions |
|---|---|---|---|---|
| RTX PRO prefill offload, time to first token, 1K / 4K / 8K / 16K / 32K | **1.36 / 1.92 / 2.94 / 4.95 / 8.99 s**; the same node server alone: 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s | 2026-10-24 | pre-release | exact CUDA prefill that reproduces the node's prefill bit for bit; state streamed to the node layer by layer over 1 GbE with a lossless codec on 2 connections; 64 generated tokens identical to the node-only run at every length; one run per length |
| One node, prompt reading, 1K / 4K / 16K / 32K | 1.13 / **3.72 / 14.46 / 29.60 s** (production build, same session: 1.15 / 3.83 / 15.24 / 31.24 s) | 4K-32K 2026-10-24; 1K 2026-10-22 | pre-release (1K: release) | fresh process per run, A/B ×4, median of 4; prefix state and logits identical |
| One node server (1 replica, 8 slots), 8-category short-prompt set, 4 concurrent | **61.2 → 100.7 tok/s** wall aggregate (+65 %) | 2026-10-24 | pre-release | the running prompts' tiled blocks grouped into one batch, their tails run as windows, the windows of all prompts run together; 32/32 requests token-identical to each request alone |
| Same server, coding chat, 8 concurrent | **79.3 → 107.4 tok/s** wall aggregate with prompt grouping alone (+36 %, 2 runs each); **76.1 → 125.0** with all three changes (median of 3, +64 %); longest time to first token 10.6-11.3 → 2.0-3.4 s | 2026-10-24 | pre-release | every request's tokens equal the request alone |

**MiniMax M3.1 Flash (bring-up, preliminary), two nodes, TP2 over USB4.**

| What | Value | Date | Build | Conditions |
|---|---|---|---|---|
| Greedy decode, 1K / 8K context | **55.14 ms/token = 18.1 tok/s** / 57.06 ms/token = 17.5 tok/s (152.9 ms/token on the first optimized build the same morning) | 2026-10-24 | pre-release, all defaults | both ranks agree; batch of 8: 43.0 tok/s aggregate, rows identical to each request alone |
| Later the same day | 53.1 ms/token (next build, A/B ×3), then 55.4 → 51.0 ms/token with a one-launch index projection (in-process A/B ×3), since made the default | 2026-10-24 | pre-release | 1K tokens identical in every arm |
| Prefill, time to first token | 1K 5.94 s, 8K 45.0 s | 2026-10-24 | pre-release, all defaults | |
| RTX PRO prefill hand-off gate, 1K | **PASS**: the attention state computed on the RTX PRO (K/V and index keys of 60 layers, ~78 MB per rank) is byte-identical to the node's FP32 reference prefill on 3 prompts × 2 ranks, and the 257 greedy tokens decoded after the import equal the node-only run | 2026-10-24 | pre-release | gate only: hand-off timing and the 8K gate are not measured yet |

The MiniMax quality gate (teacher-forced, 7 pre-registered rules) is not declared passed yet: one prompt passes 7 of
7, the other 6 of 7 (top-1 agreement 71.7 % against 73.8 % required). A gate over more prompts is planned.

**Orchestration.** The deployment controller that launches these engines was accepted on the DeepSeek pair with a
golden-render check, a token gate, negative tests and an interleaved A/B/A serve check (within 3 %; measured +0.3 to
+2.2 %, never slower): see [deployment-controller.md](docs/implementation/deployment-controller.md). The replica
router's RTX prefill offload for the Qwen replicas is in this repository (`prefill_offload` in
[replica-router.md](docs/implementation/replica-router.md)).

### Decisions

| # | Decision | Status / evidence (2026-10-23) |
|---|---|---|
| 1 | **Prefill and decode may use different placements.** The plan is chosen per phase, and the request state is handed over between them. | The handoff format is implemented. A 2,048-token prompt exports to 4.72 MB of state. |
| 2 | **Prefill runs on the discrete GPU.** Its dense parts are resident, experts are streamed from host RAM over PCIe, and the resulting state is exported to the Strix Halo pair for decode. | The CUDA prefill is bit-exact with the AMD reference at 2K and 8K tokens. The pair decodes from the imported state with the same tokens and logits at every step. Live timing of the resident GPU service with exact arithmetic (warm, load excluded, 73 GB of experts pinned in host RAM and the rest read from NVMe, median of 3): prefill **605 / 2,029 / 2,838 tok/s** at 2K / 8K / 32K tokens (3.39 / 4.04 / 11.55 s), state identical to the exact reference at every length. The AMD pair alone: 734 tok/s at 8K on the current production build; 458 / 640 / 758 tok/s at 1K / 4K / 16K on the campaign build of decision 20. Up to 8K the GPU prompt is bound by the ~80 GB of expert bytes crossing PCIe, not by compute. A faster approximate path exists but fails the quality gate (decision 22). |
| 3 | **Decode keeps the dense parts and the hot experts on the discrete GPU**, chosen by how experts are routed *together*. A layer only stays local when all its top-6 experts are local, so the placement objective is supermodular. We optimize complete co-activated sets, not per-expert frequency, and validate on held-out whole prompts. | Measured end to end, one user, greedy, exact (256/256 steps equal to the nodes-only reference): the GPU holds the dense part of all 40 layers, ~80 GB of hot experts and a 10 GiB dynamic expert cache fed from pinned host RAM; the remaining experts are served by the two AMD nodes over 1 GbE (~11 remote layers per token, 0.37 ms each). **40.1 tok/s vs 27.2 on the two nodes alone (×1.47)**, both measured on an earlier pair build. A pipeline split (GPU runs layers 0-21 with everything resident, the nodes run the rest) reaches 38.4 tok/s; 22 layers is the most the card's memory allows. Speculative decoding through the split: decision 21. |
| 4 | **The Strix Halo nodes act as expert servers for the GPU host**, with an exact combine. | One hybrid MoE layer, with 50 % or 80 % of its experts on the GPU and the rest served by an AMD node over 1 GbE, matches the all-resident reference bit for bit at 1, 2, 4 and 8 rows, for any residency split. One remote expert costs about 0.46 ms round trip over 1 GbE (network 0.35 + AMD compute 0.11); windows of 4-8 rows are link-bound. |
| 5 | **No expert replicas across the AMD nodes.** The free memory is kept for long context (up to 512K tokens, two requests). | The measured gain was about 1–3 %, or ~4 % at 30 GiB per node. Not worth the memory. |
| 6 | **Expert quantization stays as published (Q2).** Our work goes into the dense parts and the kernels instead. A different expert format is adopted only if it is at least as fast on the AMD nodes. | Tested on 4 layers against today's experts (12,759 paired rows): better rounding at the same bytes gives no measurable gain (ΔKL −0.00004); a lookup-table 2-bit format keeps quality (+0.0002) with 7.6 % fewer bytes but its kernel takes 210 µs per layer against 103–109 µs; a 2-bit trellis improves quality (ΔKL −0.0018) but runs at ~198 µs. Q2 stays, and the expert-format work is closed (2026-10-22): it reopens only if a lookup-table kernel proves clearly faster than Q2's. The trellis expert kernel reaches 182–205 GB/s on wide matrices but only 133 GB/s at the per-expert shapes. |
| 7 | **Converge on one trellis (QTIP/EXL3-style) weight format on both sides.** Existing CUDA kernels serve the NVIDIA GPU; our own `gfx1151` kernels serve the AMD nodes. Trellis is used mainly for **decode**, which is memory-bound. **Bit width:** 5 bits for the dense parts as the baseline; 4 or 3 bits per tensor class where the quality gate holds. Experts stay as published. **Prefill:** if the AMD nodes run a prefill (for example short prompts, or no discrete GPU) and trellis decoding makes it slower than the current format, they keep a second copy of the dense weights in the faster prefill format. The cost is memory, so it is decided per measurement and per memory budget. The discrete-GPU prefill uses whatever format is fastest there. | Fused trellis GEMV on `gfx1151` reads the real model head at 218 GB/s at 1 row (about the memory roof) and 164–189 GB/s at 8 rows. Prefill-shape trellis GEMMs reach 11–15 useful TOPS, so a separate prefill copy may pay off. Full-model KL against the original checkpoint (104K tokens, experts at 2 bits as in production): dense at 5-bit trellis 0.0444 vs 0.0447 with our 6.5-bit format, no category worse by more than +0.004, while reading 2.79 instead of 3.63 GB per token per node. A per-class sweep shows every dense class holds at 4 bits; **adopted: every dense class at 4 bits with the head at 5** (2.27 GB per token per node, full-set KL equal to our 6.5-bit format, paired −0.0001). Pushing three classes to 3 bits (1.92 GB) costs +0.0024; all at 3 bits fails the gate. Most of our remaining distance to the original comes from the 2-bit experts, not the dense part. In the engine, a split-K trellis kernel for the dense projections is bit-exact with the previous one and gives, on the two nodes, 27.0 → 28.3 tok/s at 1 row, 69.4 → 65.7 ms per 6-row speculative window and 79.6 → 82.6 tok/s aggregate at 8 requests. |
| 8 | **Training and distillation run on the discrete GPU.** The AMD nodes only generate data and run inference. | — |
| 9 | **Correctness before speed.** Every split or fusion must reproduce the single-placement tokens and logits bit for bit before it is measured. **Exception: prefill on the discrete GPU** may use faster, non-identical arithmetic (FP8/BF16 tensor cores, other summation orders). There it is gated on quality instead: KL against the exact prefill, and agreement of the decode that continues from the handed-over state. | Decode and expert serving stay bit-exact. The prefill exception was adopted on 2026-10-20. |
| 10 | **No single-GPU assumptions.** The discrete-GPU paths must adapt to the GPU they find: residency, streaming slots, chunk sizes and kernel tiles come from its VRAM, measured PCIe bandwidth and architecture (Blackwell and Ada first). Our 96 GB card is the development machine, not a requirement. | A first sweep with VRAM capped by software (24 / 32 / 48 / 74 GiB, expert-granularity residency) ran on the fast prefill path, which has since failed the quality gate (decision 22), so its numbers are not quoted here. The sweep will be repeated on the exact path. What it did show holds for both paths: short prompts are bound by the expert bytes crossing PCIe, so more VRAM helps them most. |
| 11 | **Flexible prefill placement.** For each request the router picks where the prefill runs from what is present and healthy: a resident prefill service on a discrete GPU, or the AMD nodes splitting it between them, or a prefix-cache hit that skips most of the prefill. A policy setting chooses `auto`, `gpu`, `nodes` or per-length thresholds, and the router falls back automatically when the GPU service is down or busy. Each model plugs in through the same engine contract. | The GPU service keeps its weights loaded between requests (0.05-0.07 s overhead per request) and hands the state to the nodes in 0.15-0.27 s. The two-node prefill and the prefix cache are both bit-exact. |
| 12 | **Integrate first, measure once.** Changes are gated for exactness individually, then measured together in one campaign with ablations. This avoids many partial benchmark runs. | — |
| 13 | **Per-request routing by predicted cost, for any GPU.** For each request the router estimates, per candidate path (prefix-cache resume on the nodes, prefill on the nodes, prefill on the GPU then decode on the nodes), the time to first token and the time to finish: queue wait in seconds of pending work, prefill time from the measured curve of *that* GPU and of the node pair, state handoff, and decode. It picks the lowest total cost, counting the delay it causes to requests already decoding, with a small margin against noise. Even a small saving is taken. GPU profiles (prefill rate by prompt length, VRAM, host-pinned memory, handoff cost) are measured per machine, not hard-coded to one card. Split decode stays out of `auto` until it beats single-placement decode end to end. | With one request, the decode cost is the same on both prefill paths, so the choice depends only on the time to first token. Before the interleaved prefill (decision 17), the nodes paused other decodes while they ran a prefill, so with N active streams a local prefill of B seconds costs about N×B seconds to the others; that favours the GPU under load. Initial rules: nodes below 2,048 prompt tokens when idle (the 512-2,047 band is being measured), GPU eligible from 2,048 tokens or whenever the nodes are busy decoding, cache hits stay on the nodes. Measured on the 96 GB card with exact arithmetic (resident service, real experts, state export included): time to first token 3.64 s at 2K, 4.39 s at 8K and 12.20 s at 32K with the experts that don't fit in VRAM pinned in host RAM. The measured crossover against the nodes is 2,323 tokens of uncached prompt; it is a configurable threshold set from a measured receipt, and it moves up when those experts have to come from NVMe. The router reads which tier the service got and uses that curve. Measured with concurrency: a prefill on the nodes blocks every other stream for its whole duration (0.45 / 1.3 / 4.4 / 14.5 s at 128 / 512 / 2K / 8K tokens), while importing a GPU-prefilled state pauses them for only 17-48 ms. So the GPU path wins earlier as load grows. |
| 14 | **Prefix and session state spill to NVMe in blocks; the KV state stays replicated, not sharded.** Each node writes its prefix-cache blocks through to local NVMe (crash-safe files, all-or-nothing hits across nodes), so a long session survives a restart and can be forked or rewound. Our per-token state is small (~890 B), so replicating it costs ~1.8 GB per node for two 1M-token sessions; sharding it by position would add traffic between the nodes at every step. | A 237K-token session resumes after a server restart in 0.16 s (disk read at 4.9 GB/s plus import) instead of 334 s of prefill; 32K in 18 ms. RAM hits, disk hits and disk hits after a restart reproduce the fresh prefill bit for bit, including the next 32 decoded tokens. ~896 B/token on disk. |
| 15 | **Speculative decoding also with several requests.** The draft model runs on both nodes, and each request keeps its own controller; verification stays exact. | Through the server, 8 concurrent chats of 128 tokens, token ids identical to plain decoding: **2 slots 33.5 → 52.3 tok/s (+56 %), 4 slots 47.8 → 56.2 tok/s (+18 %)**. On code it gains 30–127 % at every batch size; on prose from 3 requests up it falls back to plain rows (neutral). Costs ~9.6 GiB per node for the draft model. In serve, drafting pauses while more than 4 requests are in flight (decision 20). |
| 16 | **Expert down projections are tensor-parallel across the two nodes,** with a kernel that reads each routed expert once per step. | Bit-exact on both nodes at 1, 2, 4 and 8 requests. 6-row window 69.60 → 67.54 ms (−3.0 %); routed experts 179 → 197 GB/s. |
| 17 | **A long prefill on the nodes no longer blocks the other streams.** The prefill is resumable and interleaved with decode steps (minimum 64 rows per slice). | Bit-exact. With 4 streams decoding and an 8K prompt arriving, the longest decode stall goes from 14.69 s to 0.19 s. This shifts the GPU/nodes crossover of decision 13 back toward the nodes under load; the router curves will be re-measured. |
| 18 | **The node-to-node link pre-posts its receive buffers.** | Sends that found no posted receive waited 100–200 ms for link credits. With receives pre-posted, 0 stalls in 9,000 exchanges (14 in 3,000 before), and an 8-request expert-parallel step became stable (91 ms instead of 105/115 ms bimodal). |
| 19 | **Weight formats are judged on quality and speed together.** A format that improves quality but slows decode is not adopted; a small, measured quality loss is acceptable for a clear speed gain. Quality is always measured with the same fixed metric and reference set. | Applied in decision 6 (2-bit trellis experts: better quality, slower: not adopted yet) and to the second model's expert formats, which are being evaluated on the whole model. |
| 20 | **Production defaults for the DeepSeek pair, set by one timing campaign with ablations (decision 12).** On by default: segmented batch steps; tensor-parallel expert down projections (decision 16); speculative decoding in the batch and in serve, with drafting paused while more than 4 requests are in flight; interleaved prefill in serve (decision 17); eager sends for messages up to 64 KiB on the node link; fused staging (the attention output product writes the pinned send buffer, and the add reads the pinned receive buffer). Off: tensor-parallel prefill, the prepared weight load and 16-row batch steps. | 132 timed runs, 3 interleaved rounds per arm, every run's tokens checked against its reference. Each item, measured against all on: segmented steps, batch of 8 87.0 → 95.8 tok/s (+9.1 %); expert TP down, 6-row window 66.0 → 64.0 ms (−3.0 %); speculative batch +82 % at 1 request and +53 % at 2, neutral at 4-8 (the controller stops drafting there); speculative serve +62 % at 1 request, +52 % at 2 and +19 % at 4 (8-category short-prompt set). Uncapped, it cost +45 % time to first token at 8 requests; the cap of 4 cuts that to +24 % (3.26 vs 2.63 s) and keeps +7 % throughput. Eager small messages: one-user decode 27.78 → 28.48 tok/s (+2.5 %), bit-identical. Fused staging: −0.2 to −0.4 ms per token and −0.25 ms per step for batches of 1-2, bit-identical. The interleaved prefill has a cost of its own: with 4 streams decoding, an 8K prompt's first token comes at 28.8 s instead of 12.8 s, so long prompts go to the GPU prefill when it is up. Tensor-parallel prefill was 1.4-6.1 % slower than expert-parallel. Serve with these defaults, 64/64 requests token-identical to the campaign reference: 38.4 / 52.7 / 59.3 / 66.8 tok/s at 1 / 2 / 4 / 8 requests (8-category short-prompt set). Added since, each bit-exact and on by default: the attention hidden-state mix runs while the output projection's exchange is in flight (6-row window −0.35 ms, batch steps −0.6 to −1.5 ms), and the prefill writes its final KV state by owner (8K prefill +1.8 %; an 8K prompt's first token in serve after idle 17.35 → 15.78 s). Latest measured: one-user greedy decode 28.48 tok/s; on the current build, gated bit for bit against the references, a batch of 8 at 95.9 tok/s and an 8K prefill at 734 tok/s. |
| 21 | **Split decode with speculative decoding.** The GPU runs the first 22 layers with everything resident and the nodes run the rest (the pipeline split of decision 3); the draft model's windows are verified through the split. | Measured live, 24 prompts (8 English, 8 Spanish, 8 code) × 256 tokens: **61.8 tok/s against 46.4 on the two nodes alone with speculative decoding (×1.33)**; English 50.3 vs 37.5, Spanish 54.2 vs 39.3, code 81.1 vs 62.3. With the bit-exact identity (without the trellis mix of decision 7) it gives 61.7 tok/s, and all 24 outputs (6,144 tokens) equal the nodes-only reference. The end-to-end run in one GPU process (GPU prefill, state handoff to the nodes, split decode) was measured only with the fast prefill, which fails the quality gate (decision 22), so it is not quoted; it will be repeated with the exact prefill. |
| 22 | **The GPU prefill uses exact arithmetic at every length.** A faster approximate path (INT8 tensor-core dense and MoE, fused attention) stays experimental until it passes the quality gate (mean KL ≤ 0.002 against the exact prefill, and agreement of the decode that continues from its state); the work now is to find which part breaks it and repair that part. | Exact, live on the resident GPU service (warm, load excluded, 73 GB of experts pinned in host RAM, the rest read from NVMe, median of 3): time to first token including state export **3.64 / 4.39 / 12.20 s** at 2K / 8K / 32K tokens. Up to 8K the exact and fast paths are equal within the load-to-load variance (~0.5 s), because both are bound by the expert bytes crossing PCIe. The fast path is 1.9× faster at 32K, but a re-check over 256 positions on 20 prompt cells (2K-32K; English, Spanish, agent and older chat formats) failed 9 of them, the worst at 32K (KL 0.19, top-1 247/256), and at 32K its state once produced end-of-sequence as the first token. An earlier pass over only 64 positions had passed at 32K; that result is withdrawn. |
| 23 | **The node-to-node USB4 link: measured limits and what we changed.** In use: pre-posted receive buffers (decision 18); bounded waits with abort and a failure report (60 s, below the serve watchdog) instead of indefinite blocking; eager sends for messages up to 64 KiB; fused staging. Contiguous DMA buffers in the driver are available as an opt-in. | The two cables are one 40 Gb/s and one 20 Gb/s link: ~56 Gb/s per direction of usable data after encoding and headers (calculated). Measured raw with 4 streams, one process per stream: 41.0 Gb/s one way and 39.8 per direction both ways as the sum of the four streams' own rates; over the common window in which all streams are running, ~37 Gb/s per direction, because the stream on the slower cable finishes last. A ring reading host memory gives ~9-11 Gb/s. Contiguous TX and RX buffers raise a ring on the faster cable from ~11.2 to ~12.1 Gb/s, the rate of a GPU-direct ring, with the same bits (the sum of per-stream rates, 4 streams both ways, goes from ~79 to ~85 Gb/s; the common-window rate stays ~37 Gb/s per direction, set by the slower cable). The model's step time does not change (6-row window 63.90 vs 63.95 ms), so it stays opt-in. The engine's large messages were held at ~13-15 Gb/s per direction by its staging copies; an opt-in pipelined host path (1 MiB chunks, checksum overlapped with the copy) moves 13 MiB in 3.19 ms and 21 MiB in 5.15 ms (~34 Gb/s per direction, bytes exact), worth +1.2-1.3 % on 4K-16K prefills. Small messages: a 20 KB exchange takes 18-22 µs back to back, but 70-90 µs on average after ≥ 100 µs idle, which is the model's real cadence; through the engine's host path, ~100-130 µs after a 300 µs gap. Most of that idle cost is the wake-up chain from interrupt to reader: per-CPU high-priority completion work made it worse (decision 25) and IRQ pinning was neutral. Opt-in busy polling in the driver, together with per-message ready flags in the engine, makes a 20 KB message available ~32 µs after a 300 µs gap (p50; p99 113-151 µs). It is not on by default: it has not passed a model gate, and full busy polling lowers a ring's bandwidth by 7.5-8 % (12.1 → 11.2 Gb/s). GPU-direct exchanges in the production apps (same bits as the host path), against host streams: 6-row window +2.9 %, 8K prefill +1.6 %, serve at 2-8 requests +4-7 %, one-user decode −1 %; the layout choice is pending. |
| 24 | **The second model runs as two full replicas, one per node, behind the router; no tensor parallelism across the link.** Each request runs entirely on one node, and the router balances active decodes. Two server fixes are adopted: the draft controller's learned state is reset for each request, and the speculative windows of concurrent requests are planned and verified together. | Measured on 2026-10-22 with both fixes, servers restarted before each of 3 rounds (median), greedy, token ids through the router equal to the single-node run (570/570). **Coding chat** (~70-token prompt, 187-token answers), decode rate summed over requests: **90.7 / 181.5 / 201.6 / 255.0 tok/s at 1 / 2 / 4 / 8 users** (wall aggregate including time to first token 68.1 / 136.2 / 146.9 / 166.7). **8-category short-prompt set**, wall aggregate including time to first token: **42.8 / 85.8 / 102.9 / 110.6 / 120.4 tok/s at 1 / 2 / 4 / 6 / 8 users** (time to first token p50 0.89 s at 1 user, 2.55 s at 8). Before the fixes the same set gave 42.4 / 84.6 / 94.8 / 86.3 at 1 / 2 / 4 / 6, and coding chat 89.1 / 179.0 / 180.7 at 1 / 2 / 4. **Per-request reset:** chat traffic used to leave a slow draft schedule behind (a 1K code prompt 59 → 42 tok/s, persistent); with the reset, after 80 chat requests it runs at 58.9 tok/s (prose 54.7 instead of 41.3), with the same tokens. **Joint windows:** on one node, the 1K code prompt 64.1 → 71.0 tok/s at 2 requests (+10.8 %) and 78.1 → 83.1 at 4 (+6.3 %); coding chat +12.5 % at 2 and +36.2 % at 4; one request unchanged (59.86 → 59.89). One node, one request: 59.9 tok/s on a 1K code prompt and 55.5 on 1K prose (256 tokens). |
| 25 | **Measured and not adopted.** | **Tensor parallelism for one Qwen user over USB4:** an exchange costs ~200 µs today against a budget of ~50 µs (96 exchanges per window, calculated); it reopens only if exchanges at the real cadence get close to their back-to-back time. **Per-CPU high-priority completion work in the USB4 driver:** p99 of a 20 KB exchange 230 µs → 2.9 ms, 6-row window +3.7 %, decode −3.8 %. **2-bit trellis or lookup-table experts:** slower than Q2 on the AMD nodes (decision 6). **Overlapping the attention all-reduce in K chunks:** exact but slower (6-row window +1.2 / +3 / +5 ms at K = 2 / 4 / 8): each extra host exchange costs ~21 µs net, and nothing independent runs after attention to hide it. **Tensor-parallel prefill on the nodes:** 1.4-6.1 % slower than expert-parallel. |
| 26 | **Qwen prompts of 1K tokens or more go to the RTX PRO prefill service when it is up**; shorter prompts, and every prompt when the service is down, priced out by its queue or refusing, are read on the node. The router holds the replica, tokenizes on it, asks the service to prefill and import the state into that replica, then forwards the original request; any failure falls back to the replica's own prefill. | Measured 2026-10-24 on the real path (identical tokens on both routes): time to first token 1.36 / 1.92 / 2.94 / 4.95 / 8.99 s with the offload against 1.80 / 5.20 / 10.46 / 21.43 / 44.02 s on the node at 1K-32K; 4.9× at 32K. The offload already wins at the shortest measured length, so the 1K floor is the configured minimum, not a measured crossover. |
| 27 | **MiniMax M3.1 Flash runs on its own engine as a TP2 pair**, with the same distribution recipe as DeepSeek and its own kernels. An RTX PRO prefill is admitted only with a byte-identical state hand-off. | Bring-up, preliminary (2026-10-24): greedy decode 18.1 tok/s at 1K and 17.5 at 8K with all defaults, 51.0 ms/token after the latest exact change; 1K hand-off gate PASS (state bytes identical, 257 tokens equal to node-only). |

## Security model

A local network is not treated as trusted. What is implemented and tested today:

- **Enrolled host agents.** Each host agent has an explicit node and key identity. Control requests travel as signed, replay-safe capability envelopes (HMAC-SHA256, monotonic sequence). An agent opens no listener until it is enrolled ([ADR-0020](docs/decisions/0020-agent-enrollment-before-listener.md)).
- **Mutual TLS control channel.** CA verification, client certificates, exact SHA-256 pinning of the enrolled peer's certificate, a dedicated ALPN, bounded framing, and rejection of stale connections and replays ([ADR-0021](docs/decisions/0021-private-tls-control-channel.md)).
- **Private or loopback endpoints only.** Public addresses, hostnames and link-local targets are rejected.
- **Nothing executable crosses the channel.** A peer can only refer to objects the host already registered. It cannot send paths, payload objects, commands or executables for the host to trust ([ADR-0023](docs/decisions/0023-remote-control-local-object-refs.md)).
- **Replica router.**
  - It listens on loopback or a private address. A non-loopback listener refuses to start without an API key.
  - It forwards no client credentials to the replicas. Only content and protocol headers pass through, and each replica gets its own configured key.
  - The router speaks plain HTTP, so keep it on a trusted segment or behind a TLS terminator.

**Tested:**
- real mutual TLS with ephemeral certificates generated in CI, on loopback ([ADR-0022](docs/decisions/0022-ephemeral-mtls-loopback.md));
- end to end through remote dispatch ([ADR-0024](docs/decisions/0024-mtls-remote-dispatch-e2e.md));
- across separate OS processes ([ADR-0025](docs/decisions/0025-process-separated-private-control.md)).

**Not yet:**
- a qualified multi-machine deployment;
- OS-protected secret storage;
- key rotation and revocation in operation.

See [SECURITY.md](SECURITY.md) and the [network and security contract](docs/specification/04-network-security.md).

## Development principles

TensorMeld is being developed around a few rules:

1. **Do not require every available device to participate.** More hardware is useful only when it improves the requested objective or enables the workload.
2. **Treat capacity, latency, and throughput as different goals.**
3. **Model physical memory honestly.** Shared or unified memory must not be counted more than once.
4. **Account for communication.** Remote memory is not local VRAM, and link bandwidth alone does not describe distributed inference cost.
5. **Prefer measured placement decisions over hardware-name heuristics.**
6. **Keep the core hardware-neutral.** Vendor-specific execution belongs behind explicit backend/worker interfaces.
7. **Reuse good open-source components where appropriate.** TensorMeld should not reimplement mature model readers, runtimes, kernels, or system libraries without a reason.
8. **Make fallback and uncertainty explicit.** An incomplete search or unsupported placement must not be presented as proof that no solution exists.
9. **Preserve user control.** Automatic planning must respect resource limits, excluded devices, privacy boundaries, and workload requirements.

## Open-source reuse

TensorMeld will deliberately build on suitable open-source projects when their licenses, behaviour, and architecture fit the project.

Potential and current areas of reuse include model-format tooling, hardware discovery, inference runtimes, GPU backends, networking primitives, and observability libraries.

Third-party code or dependencies should be recorded with their source, version or revision, license, integration status, and any relevant limitations. Reuse does not imply that TensorMeld inherits the upstream project's support guarantees or that an upstream feature has been validated in TensorMeld.

## What not to expect yet

Please do **not** expect, at this stage:

- a stable installation procedure;
- a stable configuration schema;
- stable CLI or API compatibility;
- supported releases;
- production security guarantees;
- validated Windows/Linux feature parity;
- validated CUDA/ROCm heterogeneous execution across arbitrary hardware;
- reliable performance comparisons;
- automatic optimal placement;
- compatibility with every GGUF/model architecture;
- migration guarantees between development snapshots.

If you experiment with the repository, assume that you may need to read the source, inspect plans manually, and rebuild configurations after changes.

## Repository layout

The repository is being organised so that architecture decisions remain separate from implementation details:

```text
docs/
  architecture/      System contracts and architecture
  decisions/         Architecture Decision Records (ADRs)
  implementation/    Roadmap, implementation notes and status
  specification/     Product and behavioural specifications
  examples/          Example configurations and scenarios

src/                 TensorMeld implementation
tests/               Automated tests
research/            Evaluations and non-normative technical research
third_party/         Third-party dependency and license records
```

Documentation under `docs/architecture/` and accepted ADRs should be treated as the intended design contract. Research notes and experimental results are non-normative unless promoted into an explicit architecture decision.

## Why publish this so early?

Because distributed local inference is full of hidden assumptions: memory may be shared, links may not aggregate, execution backends may differ across operating systems, and a mathematically balanced partition can still be slower because of communication.

Publishing the project during development makes those assumptions, experiments, mistakes, and design changes visible rather than presenting a polished result after the fact.

**The repository is public to make the engineering process transparent—not to imply that TensorMeld is already a usable product.**

## Contributions and discussion

Technical discussion, issue reports, references to relevant research, and pointers to reusable open-source work are welcome while the architecture is evolving.

Before investing significant effort in an implementation, please assume interfaces may change and check the current architecture and ADRs first.

## License

TensorMeld's original code is released under the [MIT License](LICENSE). Third-party components keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `third_party/` for provenance and retained notices.
