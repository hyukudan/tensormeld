# Publication reconciliation and CI fixes — 2026-10-01

## Reconciliation

The public branch advanced while the prepared snapshot was being transferred. The
publication follow-up is based on `9e259c7c5f98a23653e8acc5d8d8ba4ff82fc20e` and preserves
those intervening commits, including the public-publication guards and removal of the
obsolete private-publication helper. It does not replace the public source tree with
an older archive or overwrite the public README and owner-selected MIT LICENSE.

The prepared archive at local snapshot `c719b76` was independently rerun: 140 tests,
139 passed and 1 optional upstream GGUF integration skipped on Linux/Python 3.13.5.
That historical suite count must not be reused as the current public CI test count:
the public branch replaced private-publication tests with public-transparency guards.

## Failures observed in hosted CI

Run `36861702279` at commit `be60b67ab2cd543a66f79abb9d4da2e08b3dd304` failed.
The Ubuntu core job `110367222971` exposed an outdated third-party notice claiming
no original-code license was selected, contradicting the existing root LICENSE and
publication guard. The Windows optional-adapter job `110367223388` exposed an
injected-reader test comparing normalized and unnormalized path strings. That made
two distinct fixture files appear to have the same mock shard/tensor identity.

The Windows job separately passed the actual upstream GGUF writer/reader round trip.
That narrow observation does not turn the failed overall run into a pass and does
not qualify distributed inference, GPU kernels or arbitrary model architectures.

## Changes

- Defer original-code licensing to the repository root `LICENSE`, without modifying it.
- Compare fixture file identity with `Path.samefile`, not raw spelling of the path.
- Add a noncanonical-path regression using an existing directory and `..`.
- Preserve duplicate tensor rejection in the real inspector and the negative tests.
- Leave runtime implementation, user-facing README, existing ADRs and prior commits intact.

## Validation of this patch

Locally reran the changed adapter test module against the prepared runtime:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_open_source_adapters.py -v
```

Result: 15 tests, 14 passed and 1 optional upstream integration skipped, no failures.
The fixture test file was verified against the current public blob before editing.
This is targeted Linux validation, not a native Windows rerun. The new public commit
must pass its own complete Windows/Linux workflows before claiming those results.

## Handoff

The initial code/docs publication must not be repeated. Inspect CI on the latest commit,
then continue the bounded adapter/capability work in the implementation roadmap.
Use the existing ADR-0010 for public-transparency policy. Do not restore the deleted
private-publication helper or relax tests to conceal failures. A periodic assistant
watchdog is not a daemon shipped in TensorMeld and does not run continuously.
