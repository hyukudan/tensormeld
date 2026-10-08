# Tensor movability validation — 2026-10-08

Implementation slice: complete explicit tensor storage/movement evidence.

## Scope

The tensor-movability profile classifies every tensor from one complete GGUF index
without inferring movability from tensor names, size, backend labels or model family.

The profile is bound to the exact model/tensor-index, adapter capability fingerprint,
llama.cpp build package and current runtime identities. The GGUF index fingerprint is
recomputed before use, so modified index contents cannot retain an old hash.

## CI

Pull request #30 workflow run 37780644654 completed with all seven jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

The test suite covers complete tensor coverage, exact byte sizes, pinned/owner-local/
placement-movable constraints, alias-group consistency, package/runtime invalidation,
GGUF-index tamper detection, persisted package loading and CLI output safety.

## Result

PR #30 merged as 335914cea8354a9d0f7a7410592fe97991c0be46.

This remains portable contract/fixture validation. No real target-host tensor movability,
CUDA/HIP movement cost, RTX/Strix Halo memory behavior or fine-grained execution claim
was produced.
