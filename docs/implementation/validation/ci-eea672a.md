# Hosted CI validation — eea672a

Observed on 2026-10-01. This record describes an actual completed GitHub Actions run,
not a proposed workflow or an inference/hardware qualification.

- Source commit: `eea672a7eab43a5551715d66a7a177b1d1be7c42`.
- Workflow: `Portable Python tests`, run 33, attempt 1.
- Run ID: `36863594044`.
- Event: push to main.
- Final status: completed / success.
- Completed at: `2026-10-01T12:43:37Z` (run update timestamp).
- Evidence: https://github.com/hyukudan/tensormeld/actions/runs/36863594044

| Job | ID | Result |
|---|---|---|
| test (ubuntu-latest, 3.11) | 110373518625 | success |
| test (ubuntu-latest, 3.13) | 110373518603 | success |
| test (windows-latest, 3.11) | 110373518605 | success |
| test (windows-latest, 3.13) | 110373518627 | success |
| optional-adapter-tests (ubuntu-latest) | 110373518173 | success |
| optional-adapter-tests (windows-latest) | 110373518571 | success |

All six jobs completed successfully. The four core jobs passed the unit suite,
configuration and selection commands, synthetic v1/v2 planning examples and source
compilation. Optional-dependency jobs installed the `system,gguf` extras, verified
imports and the pinned GGUF version, and passed the adapter module, including the
actual upstream writer/reader round trip and the noncanonical-path regression.

The original local archive test count (140) is not reused as the current public test
count: private-publication guards were replaced and a regression was added. Core jobs
without optional dependencies may skip those integrations; the separate installed-
dependency jobs provide the integration evidence. No skip is represented as a pass.

This supersedes the pending hosted-CI status for this exact code commit in the initial
publication follow-up. Historical local skips remain valid descriptions of their own
environment. A later commit must be assessed on its own changes and CI coverage.

## Explicit exclusions

No CUDA/HIP worker build, discrete/integrated GPU execution, real language-model
inference, inter-host tensor transport, model fit/performance benchmark, secure remote
agent or arbitrary model-format support is established by this run. TensorMeld remains
in very early active development, published for transparency rather than production use.
