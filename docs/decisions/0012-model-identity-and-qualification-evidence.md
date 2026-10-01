# ADR-0012 — Exact model identity and qualification evidence are separate contracts

Status: **accepted**  
Date: 2026-10-01

## Context

A GGUF tensor directory is useful for inspection, but file size and tensor metadata do not
prove runtime memory requirements, operator coverage, model correctness, or performance.

Likewise, a static adapter capability declaration does not prove that a particular worker
artifact, model revision, device set, workload, and configuration have actually passed a
qualification run.

## Decision

TensorMeld separates:

1. **Model identity** — `tensormeld/model-manifest-v1`
2. **Qualification evidence** — `tensormeld/qualification-evidence-v1`
3. **Evidence applicability** — an exact binding check against current identities.

The initial GGUF-backed model manifest requires:

- a complete shard set;
- exact shard file names, byte sizes, and full SHA-256 hashes supplied for every shard;
- model/revision identity;
- architecture;
- tokenizer reference and optional chat-template reference;
- the metadata-only GGUF tensor-index fingerprint.

The manifest deliberately contains no inferred RAM/VRAM requirement.

Qualification evidence binds a successful or failed observed run to:

- adapter ID and exact adapter-capabilities fingerprint;
- engine revision;
- worker artifact SHA-256;
- exact model-manifest fingerprint;
- installation/config fingerprint;
- ordered device identities;
- workload bounds;
- evidence level E0..E5;
- timestamp and named tests.

A successful applicability check may state `qualified=true` for that exact evidence
scope, but **must remain `executable=false`**. Live resource admission, secure session
preparation, and runtime state are separate gates.

## Consequences

- Changing a shard, tokenizer, model revision, adapter capability envelope, engine revision,
  config, device identity, workload, or required evidence level invalidates applicability.
- Partial GGUF sets cannot become exact model manifests.
- Payload byte size cannot be reused as a runtime memory estimate.
- Qualification records become traceable and invalidatable rather than generic
  "this GPU/model works" labels.
