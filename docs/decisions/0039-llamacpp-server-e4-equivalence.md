# ADR-0039 — Persistent server requests require E4 equivalence, not merely same-build provenance

Status: **accepted**  
Date: 2026-10-05

## Context

ADR-0038 permits llama-server process ownership under an admitted lease when llama-cli and
llama-server are exact sibling artifacts in one verified build package.

That provenance is still insufficient to assume the server's request path produces the
same model result as the qualified CLI path. HTTP request parsing, server defaults, cache
behavior and slot execution add semantics not covered by CLI E3.

## Decision

TensorMeld introduces a narrow llama-cli ↔ llama-server E4 equivalence gate.

The E4 spec binds:
- exact build package;
- exact model and AcceptedExecutionBundle;
- pre-E3 and post-E3 placement fingerprints;
- exact equality of block ownership, buffer mapping, override-tensor rules and placement
  argv;
- exact CLI trial and server launch specs;
- prompt, context, output length, seed and temperature;
- a closed non-streaming server request with prompt caching disabled.

E4 executes the server request against loopback /completion and compares the exact SHA-256
of generated content with the retained CLI trial stdout SHA-256.

Fixture CLI/server processes may validate protocol behavior, but only the provenance pair
`native-subprocess` + `native-server-subprocess` can produce qualified E4 evidence.

An admitted server binding can become request-authorized only when the E4 evidence and E4
spec are intact and match the exact package, bundle and server spec of that admitted
binding.

E4 authorization does not itself execute an inference request, so
`real_model_inference` remains false until the later request path actually runs.

## Consequences

- Same-build provenance is necessary but not sufficient for persistent request serving.
- Server defaults/cache behavior cannot silently bypass the CLI correctness baseline.
- Fixture equality is useful CI evidence but cannot become hardware/native qualification.
- The next execution step can be extremely narrow: one authorized completion contract,
  rather than a general server proxy.
