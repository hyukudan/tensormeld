# ADR-0007 — TensorMeld identity

**Status:** Accepted
**Date:** 2026-10-01

## Decision

The project, Python package, command and repository are named
`TensorMeld`, `tensormeld`, `tensormeld` and `tensormeld`, respectively.
The tagline is **Different machines. One model.**

Keep the existing Git history. Do not rewrite past commits or archived specifications.
The previous working title was Inference Companion. No upstream endorsement, trademark
clearance, domain ownership or published package ownership is implied by this name.

Canonical installation files use `tensormeld/v2`. Old
`inference-companion/v2` files require the explicit `migrate-config` command.
The command produces a new file and never silently rewrites the source.
Version-one analytical fixtures retain their integer `schema_version`.

The original publication helper targeted a new private repository. That publication
clause is superseded by ADR-0010; the canonical repository is now public for
engineering transparency. Renaming locally did not itself create or publish a repository.
