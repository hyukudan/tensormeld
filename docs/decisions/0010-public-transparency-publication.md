# ADR-0010 — Publish the development repository for transparency

**Status:** Accepted
**Date:** 2026-10-01

## Decision

Publish the canonical TensorMeld development repository publicly while the project is
still pre-alpha. The purpose is engineering transparency and open technical discussion,
not to imply a supported release, stable API, production readiness, or validated
distributed inference.

This decision supersedes only the private-publication clause of ADR-0007. It does not
change TensorMeld's name, package identity, implementation status, security gates,
qualification requirements, or the license already present in the public repository.

## Why

The project exists to reason explicitly about heterogeneous local inference: a fast
discrete GPU may be the best place for some compute, while a higher-capacity companion
may make larger model/state placement possible. Those trade-offs involve hidden
assumptions about memory ownership, communication, backends and topology. Publishing
the engineering work early makes those assumptions, failures and changes inspectable.

## Consequences

- README status warnings remain prominent while the project is not generally usable;
- synthetic planning results remain clearly separated from measured or executable plans;
- credentials, model files, prompts, personal inventories and private network details are
  never committed;
- third-party provenance and license notices are maintained alongside integrations;
- public CI results are evidence for their exact hosted environments only, not hardware
  qualification;
- old history documents may describe the earlier private-publication plan and remain
  historical rather than normative.
