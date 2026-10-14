# TensorMeld

> [!CAUTION]
> TensorMeld is in early, active development. This public repository documents an experimental engineering project, not a supported release. APIs, schemas and execution paths may change. It is not ready for production or security-sensitive deployment.

TensorMeld is a local-first, hardware-neutral orchestration project for heterogeneous AI inference resources. Its goal is to choose useful placements across devices and machines while respecting physical memory, communication costs and owner policies.

The current package is **0.2.0a2**, with a **0.2.0 specification baseline**. The version identifier does not establish product readiness. The usable starting point is the control-plane prototype: configuration validation, resource selection, advisory planning and explicit evidence contracts. **End-to-end distributed model inference is not implemented or qualified.**

## Purpose and scope

A fast discrete GPU may lack capacity; a companion may offer more memory but different compute and transfer costs. TensorMeld models those resources separately rather than treating remote memory as extra local VRAM.

The intended scope includes dense and mixture-of-experts workloads, low-VRAM hosts, multiple devices per node and variable node counts. Strix Halo is an initial companion profile, not a product dependency. Windows/Linux, CUDA/HIP and heterogeneous execution are design targets whose actual support must be established per adapter and workload.

A valid future placement may use the local device, companions only, a subset of available resources or a genuine split. More nodes, memory or cables do not automatically improve latency or throughput. Requested model semantics, context, precision and owner limits must remain explicit.

## Architecture

The design separates three responsibilities:

1. **Control:** installation identity, enrollment, resource policy, authenticated communication and lifecycle.
2. **Placement:** model units, physical-pool accounting, directional paths and bounded candidate search.
3. **Execution:** approved backend artifacts, exact model/runtime bindings, admission and session ownership.

Nodes, physical devices, backend views and physical memory pools have distinct identities. Shared RAM or unified memory is counted once. Planning preserves execution order and cyclic token feedback rather than considering only one-way transfers. Search bounds and incomplete evidence remain visible in the result.

An advisory plan is not an executable manifest. Native execution requires applicable readiness and model-correctness evidence, exact adapter representation, runtime memory/operator coverage and fresh admission. Local atomic leases implement an in-process ownership boundary; they are not a distributed lock service.

Read the [architecture overview](docs/architecture/system-overview.md), [planning pipeline](docs/architecture/planning-pipeline.md) and [configuration contract](docs/specification/08-configuration.md) for the normative boundaries.

## Current capabilities and limits

| Area | Available implementation | Boundary |
| --- | --- | --- |
| Configuration and policy | Strict v2 parser, static selection and advisory runtime-snapshot intersection | Configuration does not prove availability or compatibility |
| Placement | Bounded synthetic whole-block search; legal-unit and generation-scenario evidence contracts | Synthetic costs and ordered-chain estimates are not measured token performance |
| Inventory and models | Local inventory, optional system discovery, optional upstream GGUF indexing and exact model identities | Discovery and parsing do not qualify inference |
| Agents and control | Enrollment, signed capability envelopes, bounded authenticated control, TLS/mTLS helpers and local pool leases | Protocol/fixture tests do not qualify a deployment or GPU path |
| Native adapter foundations | Pinned local probes, explicit device binding, narrow backend self-test and model-trial contracts | The initial placement shim is local/single-node; arbitrary remote execution is not enabled |
| Request lifecycle | Accepted bundles, bounded worker/session seams, managed local server primitive and exact C1 completion consumer | Arbitrary-prompt family serving and public HTTP/SSE remain separate gates |
| Qualification tooling | Closed local campaign runner with pinned artifacts and bounded private records | Running a campaign does not grant qualification, admission or serving authorization |

The [current status](docs/implementation/status.md) and [qualification ledger](docs/implementation/qualification-ledger.md) distinguish implemented contracts, portable tests and outstanding native evidence. Hosted CI and synthetic fixtures establish software behavior, not CUDA/HIP execution, model fidelity, topology independence or speedups.

## Developer setup

Use Python **3.11 or newer**. The base package has no runtime dependencies; install from a local checkout in an isolated environment. This is a developer workflow, not a stable end-user installer.

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install and inspect the current command surface:

```bash
python -m pip install -e .
python -m tensormeld --version
python -m tensormeld --help
```

The installed `tensormeld` command exposes the same CLI. Optional integrations have pinned dependencies:

```bash
python -m pip install -e ".[system,gguf]"
```

The `system` extra adds psutil discovery; `gguf` adds the upstream GGUF reader. Neither installs a GPU inference engine, drivers or model weights. Native adapters require separately approved, pinned local artifacts; follow the [adapter guide](docs/implementation/open-source-adapters.md).

## First control-plane experiment

Run from the repository root:

```bash
python -m tensormeld validate-config docs/examples/12gb-pc-one-companion.config.json
python -m tensormeld select docs/examples/12gb-pc-one-companion.config.json
python -m tensormeld plan-v2 docs/examples/12gb-pc-one-companion.config.json examples/v2/12gb-dense.planning.json --top-k 3
```

These examples illustrate policy and synthetic placement. Their capacities, names and costs are not measurements of your host. See [configuration examples](docs/examples/README.md) for companion-only and multi-node/multi-GPU policies and the [v2 planner contract](docs/implementation/planner-v2.md) for inputs and result semantics.

Read-only inventory and a bounded local socket diagnostic are also available:

```bash
python -m tensormeld probe
python -m tensormeld bench-loopback --iterations 20
```

Loopback timing does not measure a physical network, USB4 path or model inference. Commands support `--out` for JSON output; keep personal inventories and experimental records outside Git. For optional GGUF indexing of an approved local file:

```bash
python -m tensormeld inspect-gguf /path/to/model.gguf --trusted-local-file
```

Use `python -m tensormeld COMMAND --help` before more advanced operations. Legacy `validate` and `plan` commands retain the v1 analytical oracle; they do not replace the v2 configuration workflow.

## Tests and evidence

Run the portable suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Tests cover policy, memory aliasing, bounded search, identity invalidation, admission, cancellation and transport contracts. Some integrations need optional packages or platform tools. A skipped optional test is not a passed integration; consult the [testing strategy](docs/implementation/testing.md) and CI definitions for prerequisites.

Native qualification is separate: exact backend, model, workload, numerical contract and runtime identity must be bound to genuine target evidence. No README command authorizes driver/firmware changes, network reconfiguration or large model downloads. See the [qualification campaign runner](docs/implementation/qualification-campaign.md) for the explicit operator workflow.

## Repository and documentation

| Path | Purpose |
| --- | --- |
| `src/tensormeld/` | Python control plane, planners, adapters and runtime contracts |
| `tests/` | Portable contracts, fixtures and bounded integrations |
| `docs/architecture/` | System architecture |
| `docs/specification/` | Numbered normative product/runtime contracts |
| `docs/decisions/` | Architecture Decision Records |
| `docs/implementation/` | Current support, tests, roadmap and qualification boundaries |
| `docs/examples/`, `examples/` | Configuration and analytical planning inputs |
| `third_party/` | Dependency registry, provenance and licenses |

Start at the [documentation index](docs/README.md), [roadmap](docs/implementation/roadmap.md), [implementation status](IMPLEMENTATION_STATUS.md) and [security policy](SECURITY.md). Proposals and research are not adopted execution capabilities.

## Contributing and publication policy

Read [AGENTS.md](AGENTS.md), the implementation ledger and the relevant specification before changing behavior. Maintained code, tests and documentation use English. Preserve the legacy analytical fixtures until explicitly migrated, and report implemented, tested and qualified separately.

Comparative research, external benchmark records, personal inventories, prompts, credentials, caches and model weights stay outside Git. Public documents describe project-owned contracts and qualified evidence. Adopted components retain their required provenance, licenses and notices; reuse does not inherit upstream support guarantees.

Technical discussion and focused issue reports are welcome. Interfaces remain unstable; check current architecture decisions before investing in a large implementation.

## License

Original TensorMeld code is released under the [MIT License](LICENSE). Third-party components retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `third_party/`.
