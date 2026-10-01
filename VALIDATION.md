# TensorMeld 0.2.0a2 — Validation record

Date: 2026-10-01. Environment: available Linux x86_64, Python 3.13.5.
The test environment is not the user's workstation or Strix Halo hardware.

## Observed results

`PYTHONPATH=src python -m unittest discover -s tests -v`

**140 tests discovered: 139 passed, 1 skipped, 0 failures/errors.** The skipped test is
`GGUFUpstreamIntegrationTests.test_upstream_writer_reader_roundtrip`: the actual
`gguf` package is unavailable in this environment. Adapter tests using injected readers
passed; they are not a substitute for this integration test.

Coverage includes legacy regression, v2 configuration, manual/automatic policy,
required resources, 1/2/3/4/8/16 simulated-node planning, multi-GPU node accounting,
multi-pool admission, shared pools, independent coordinator overhead, cyclic feedback,
one-link path selection, work/deadline exhaustion, a small independent exhaustive cost
oracle, migration safeguards, output/input alias protection, optional-provider contracts
and real psutil 7.2.2 memory/interface calls.

## Command and packaging checks

- Python source compilation: passed.
- `--version`: reports `0.2.0a2`.
- Low-VRAM example, synthetic v2 planning: `CANDIDATES_FOUND`, exit 0.
- High-capacity three-node example, synthetic v2 planning: `CANDIDATES_FOUND`, exit 0.
- Wheel built locally, offline, without dependency installation.
- Wheel installed into a separate target directory; CLI commands tested outside the
  source tree. Version, v2 planning and read-only inventory succeed.
- Wheel contains THIRD_PARTY_NOTICES and the retained psutil/ggml license texts.
- Active documentation local links checked: none unresolved.
- Source ZIP extracted into a clean directory: the full suite repeated with the same
  140 discovered / 139 passed / 1 explicitly skipped result. No local caches, inventories,
  model files or Git-internal files are included in the source ZIP.

No observed cost in a synthetic fixture is a hardware benchmark. Declared capacities,
per-unit memory and timings in these examples are assumptions for contract tests.

## Not performed / not qualified

Native Windows testing, GitHub Actions runs, real upstream GGUF parsing, CUDA/HIP worker
builds, native engine integration, authenticated remote enrollment, runtime pool leases,
real model-state/workspace manifests, Ethernet/USB4 tensor execution, multirail,
large-model inference and GPU performance comparisons were not performed here.

The two native-engine revisions in the third-party registry were identified but not
downloaded/compiled. Dependency hashes/versions do not constitute a transitive security
audit. GGUF parsing remains trusted-local-only, not an adversarial-file sandbox.

## Repository publication

TensorMeld is now published in a public GitHub repository for engineering transparency.
Publication does not change the pre-alpha status or qualify any untested runtime path.
The validation results above remain tied to the recorded local Linux environment and must
not be interpreted as GitHub-hosted, Windows, CUDA, ROCm, or distributed-inference qualification.
