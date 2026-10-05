# ADR-0037 — Persistent llama-server is a distinct execution artifact

Status: **accepted**  
Date: 2026-10-05

## Context

TensorMeld's native E3 path currently qualifies a pinned llama-cli artifact. A persistent
local API would naturally use llama-server, which is a different executable.

Treating both binaries as one worker solely because they came from the same llama.cpp
source revision would weaken the artifact identity boundary.

At the same time, TensorMeld needs process ownership mechanics before a real target-host
persistent session can be trusted.

## Decision

TensorMeld introduces a managed llama-server child-process primitive.

The server launch is bound to:
- exact pinned llama.cpp source revision;
- approved server artifact SHA-256;
- optional approved launcher SHA-256 for test or wrapper execution;
- exact ModelManifest and approved GGUF;
- exact post-E3 placement translation and AcceptedExecutionBundle;
- loopback-only host;
- explicit unprivileged port and bounded context;
- exactly one server slot.

TensorMeld owns the child process, drains stderr into a bounded diagnostic tail, polls the
loopback /health endpoint, detects early child exit and shuts down using
terminate → bounded wait → kill fallback.

Inherited LLAMA_ARG_* variables are removed. No caller-supplied argv or environment,
public listener or llama.cpp RPC is represented.

The managed server is **not yet** treated as the E3-qualified execution worker. A later
TensorMeld build/package identity must explicitly bind the qualified llama-cli artifact
and persistent llama-server artifact before the server can inherit execution authorization.

## Consequences

- Process ownership and cancellation mechanics can be qualified independently of model correctness.
- TensorMeld preserves executable-level identity rather than collapsing binaries by source revision.
- Persistent server startup cannot yet consume the admitted lease as production inference.
- The next slice is a package/build identity bridge, not a relaxation of E3.
