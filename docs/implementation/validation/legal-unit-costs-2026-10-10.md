# Legal-unit cost evidence validation — 2026-10-10

Implementation slice: exact per-legal-unit compute/memory/boundary evidence.

## Scope

The contract binds every legal model unit and every legal device to explicit compute time,
persistent-state bytes, workspace peak, staging peak and boundary output payload bytes.

It is bound to exact config, legal-unit, tensor-movability and runtime-manifest identities.
No cost is inferred from tensor size, name, model family or backend label.

The final legal unit must expose zero boundary payload. Memory maps may reference only
physical pools local to the profiled device node.

## CI

Pull request #34 completed with all seven portable jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

## Result

PR #34 merged as 9fb0ce92be7f262c16d5d81fc551a09be483afb9.

This is evidence-contract validation only. It does not enable finer-grained execution,
performance ranking, or real hardware claims.
