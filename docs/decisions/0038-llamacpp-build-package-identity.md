# ADR-0038 — Same-build provenance does not imply server semantic equivalence

Status: **accepted**  
Date: 2026-10-05

## Context

Native E3 currently qualifies an exact llama-cli artifact. Persistent serving uses a
different executable, llama-server. ADR-0037 therefore kept server process ownership
separate from qualification.

To attach the persistent process to an admitted lease safely, TensorMeld needs stronger
provenance than "same source commit", while still avoiding the false claim that two
sibling binaries necessarily implement identical request semantics.

## Decision

TensorMeld introduces `tensormeld/llamacpp-build-package-v1`.

The package binds:
- pinned source revision;
- exact llama-cli SHA-256;
- exact llama-server SHA-256;
- identical observed version/build/commit/compiler/target metadata from both executables;
- deterministic backend-library name/SHA identities.

llama-cli and llama-server must remain distinct artifact identities.

TensorMeld also introduces an admitted-server binding. It permits process ownership only
when:
- package llama-cli equals the E3-qualified worker artifact in AcceptedExecutionBundle;
- launch spec llama-server equals the package server artifact;
- launch spec/bundle/model/revision identities match;
- the launch-admitted lease is still active and fingerprint-matched.

This bridge allows the persistent server process to own the already admitted lease.
It does **not** authorize inference requests and does not claim server semantic
equivalence to the CLI trial.

A future E4 equivalence gate must compare server requests against the qualified reference
under the same package/model/placement/workload before
`inference_request_authorized=true` is possible.

## Consequences

- Build provenance is stronger than source-revision equality.
- Individual executable hashes remain first-class identities.
- Backend libraries participate in package invalidation.
- Persistent server process ownership can be integrated with lease lifecycle without
  weakening E3.
- Request serving remains blocked until an explicit equivalence gate exists.
