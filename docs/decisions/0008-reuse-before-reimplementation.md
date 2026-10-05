# ADR-0008 — Reuse before reimplementation

**Status:** Accepted
**Date:** 2026-10-01

## Decision

Prefer a maintained dependency or native-engine adapter when it satisfies the required
contract. Custom code is justified for placement policy, resource accounting and the
coordination contracts that distinguish this product, not for duplicating mature parsers,
OS introspection libraries, cryptography, tokenizers or GPU kernels.

"Reuse" does not mean adopting another inference project's planner, scheduler or memory
architecture as TensorMeld's core. Prior-art ideas are normally re-derived into
TensorMeld-owned contracts and independently implemented against TensorMeld's heterogeneous
resource model. Direct source reuse is an explicit, provenance-recorded exception.

Prefer in order: a pinned library API, an isolated subprocess adapter, a small documented
patch set, then narrowly vendored source with preserved notices. A large fork is not the
default way to acquire a small feature. Keep generic inference engines outside the
Python token-critical path; never round-trip every tensor through Python merely to
standardize an interface.

## Admission gates

Before adopting a component, record source identity, license and notice obligations,
version/artifact identity, platform/dependency requirements, update strategy, security
boundary and an executable integration test. Distinguish reviewed, implemented, tested
and hardware-qualified. Do not advertise a dependency integration merely because an
adapter passes a mock test. Dependencies may not silently weaken precision, memory,
privacy, network or exact placement contracts.

Candidate native engines must be pinned independently. Two forks sharing an ancestor
are not assumed RPC-compatible. No downloaded binary is implicitly trusted. Future
bundled installers need complete transitive inventories, licenses and artifact hashes.
The component register is maintained outside normative architecture documentation.
