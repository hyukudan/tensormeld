# Documentation

This directory separates normative product decisions from implementation notes and
research. Comparative evaluations and third-party benchmark records remain outside Git.
Public documentation records project-owned contracts; actual adopted components retain
required provenance and license notices.

## Architecture

- [System overview](architecture/system-overview.md)
- [Resource model](architecture/resource-model.md)
- [Planning pipeline](architecture/planning-pipeline.md)
- [Execution and adapters](architecture/execution-and-adapters.md)
- [Security model](architecture/security-model.md)

## Decisions

- [ADR-0001 — Separate control, placement and execution](decisions/0001-separate-control-placement-and-execution.md)
- [ADR-0002 — Model nodes, devices and physical pools separately](decisions/0002-resource-identity-and-memory-pools.md)
- [ADR-0003 — Use backend adapters instead of one universal GPU process](decisions/0003-backend-adapters.md)
- [ADR-0004 — Bounded search with explicit incompleteness](decisions/0004-bounded-planning-search.md)
- [ADR-0005 — Whole-block placement is the first executable distributed baseline](decisions/0005-whole-block-first.md)
- [ADR-0006 — Strix Halo is a qualification profile, not a product dependency](decisions/0006-hardware-neutral-core.md)

- [ADR-0007 — TensorMeld identity](decisions/0007-tensormeld-identity.md)
- [ADR-0008 — Reuse before reimplementation](decisions/0008-reuse-before-reimplementation.md)
- [ADR-0009 — Synthetic planning boundary](decisions/0009-synthetic-v2-planning.md)
- [ADR-0010 — Public transparency publication](decisions/0010-public-transparency-publication.md)
- [ADR-0036 — External engines inform TensorMeld; they do not own its architecture](decisions/0036-independent-prior-art-adoption.md)

## Normative specification

The numbered documents under [`specification/`](specification/) define current product
and runtime requirements. Start with:

- [01 — Product](specification/01-product.md)
- [02 — Architecture](specification/02-architecture.md)
- [03 — Placement](specification/03-placement.md)
- [04 — Network and security](specification/04-network-security.md)
- [05 — Platforms and validation](specification/05-platforms-validation.md)
- [08 — Configuration](specification/08-configuration.md)
- [09 — Resources and sessions](specification/09-resource-sessions.md)
- [10 — Model API and UX](specification/10-model-api-ux.md)
- [11 — Feature acceptance matrix](specification/11-feature-acceptance-matrix.md)

`07-m0-contract.md` documents the retained legacy analytical planner. It is not the v2
installation contract.

## Implementation

- [Roadmap](implementation/roadmap.md)
- [Current status](implementation/status.md)
- [Implementation and qualification ledger](implementation/qualification-ledger.md)
- [Testing strategy](implementation/testing.md)

- [v2 planner contract](implementation/planner-v2.md)
- [Open-source adapter status](implementation/open-source-adapters.md)
- [Serial generation scenario evaluator](implementation/generation-scenario.md)
- [Durable request-family preflight](implementation/request-envelope.md)
- [Bounded C1 exact replay consumer](implementation/llamacpp-c1-serving.md)
- [Closed local qualification campaign runner](implementation/qualification-campaign.md)
- [Private direct-GPU commissioning preparation](implementation/direct-gpu-commissioning.md)
- [Numerical profile contract](implementation/numerical-profile-contract.md)

## Research

Public investigation notes describe project-owned optimization principles. Comparative
records and external benchmark/source observations remain private outside Git.

- [Runtime optimization principles](research/runtime-optimization-principles.md)

## Draft numerical capabilities

- [Phase specific numerical capability proposal](proposals/2026-10-11-numerical-plan-capabilities.md).

This is an unqualified draft and does not change current execution support.

## Examples

`examples/*.config.json` are accepted v2 control-plane configuration examples. Values
are illustrative resource policies, not hardware performance measurements.
