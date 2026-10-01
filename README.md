# TensorMeld

> [!CAUTION]
> **TensorMeld is in very early, active development.**
>
> This repository is public **for transparency and open technical discussion only**. It is **not an alpha release**, is **not ready for general use**, and should not currently be relied on for production workloads, benchmarking claims, security-sensitive deployments, or stable APIs.
>
> Expect incomplete features, breaking changes, experimental code paths, incorrect planning decisions, unfinished documentation, and repository history that may change substantially while the architecture is being developed.

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

In particular, current code and documentation must not be interpreted as proof that a given model, GPU combination, network topology, or placement strategy will work correctly or outperform a simpler setup.

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
