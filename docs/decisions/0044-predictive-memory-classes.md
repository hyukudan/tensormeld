# ADR-0044 — Predictive memory classes refine, but do not relax, runtime admission

Status: **accepted**  
Date: 2026-10-08

## Context

A single resident-byte total hides materially different memory behavior. Some bytes must
remain committed to execute a request, while mapped/file-backed weight pages may be
reclaimable under pressure. Load/repack staging has different lifetime from persistent
state and decode workspace.

Treating all bytes as identical prevents useful future planning. Treating reclaimable
bytes as free would be unsafe.

## Decision

TensorMeld introduces `tensormeld/predictive-memory-v1` as an exact refinement of an
existing `RuntimeModelManifest v1`.

For each physical pool the refinement declares:

- `hard_resident_bytes`;
- `reclaimable_file_backed_bytes`;
- `persistent_state_bytes`;
- `workspace_peak_bytes`;
- `staging_peak_bytes`.

The refinement must reconcile exactly with the existing runtime manifest:

```text
hard_resident + reclaimable_file_backed == resident_bytes
persistent_state == state_bytes
workspace_peak == workspace_peak_bytes
staging_peak == preparation_peak_bytes - steady_peak_bytes
```

Derived values are:

```text
hard_peak = hard_resident + persistent_state + workspace_peak + staging_peak
worst_case_physical = hard_peak + reclaimable_file_backed
```

`worst_case_physical` must equal the current runtime manifest preparation peak. Therefore
this contract cannot reduce current admission requirements.

The predictive profile is bound to the exact runtime-manifest fingerprint and must classify
every runtime physical pool exactly once. Fixture and native-adapter provenance remain
separate.

## Consequences

- current v1 admission behavior is unchanged;
- reclaimable/file-backed bytes remain visible as physical-memory pressure and are never
  silently treated as zero-cost;
- future planners can reason about hard lower bounds and page-cache-sensitive pressure
  without losing the conservative worst case;
- GGUF file size still does not become a runtime memory estimate;
- native predictive memory evidence must still be collected on target hardware before any
  performance/capacity claim.
