# ADR-0019 — Local admission reserves exact physical-pool peaks atomically

Status: **accepted**  
Date: 2026-10-02

## Context

Runtime observations are snapshots, not leases. Two concurrent preparations can otherwise
spend the same apparent free memory. Runtime manifests now provide exact preparation peaks
per physical pool, including shared/unified pools only once.

Telemetry may already reflect allocations created by TensorMeld. Blindly subtracting every
active logical lease from that telemetry can therefore charge committed memory twice.

## Decision

TensorMeld introduces an in-process local admission controller that:

1. serializes reservation changes under one process-local lock;
2. reserves the exact preparation peak for every physical pool in the runtime manifest;
3. intersects current runtime availability with owner policy before reservation;
4. treats active leases not listed as reflected in the observation as additional committed bytes;
5. does not subtract active lease bytes again when the snapshot explicitly lists that lease ID as already reflected in telemetry;
6. requires a newly identified observation for launch-time recheck;
7. retains the lease on a failed launch recheck so the caller can retry or release it;
8. releases deterministically and idempotently.

The primitive is intentionally local. It is not a distributed lock, agent protocol,
remote reservation service or proof that a backend allocation actually occurred.

## Consequences

- Concurrent local admissions cannot consume the same unreflected pool budget.
- Shared physical pools remain charged once because the runtime manifest is pool-based.
- The telemetry accounting rule is explicit rather than implicitly double-subtracting.
- Host agents can later wrap this primitive while keeping node-owner policy authoritative.
- Successful launch admission still leaves model correctness, backend qualification,
  worker lifecycle and distributed execution as separate gates.
