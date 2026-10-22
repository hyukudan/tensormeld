# llama.cpp native E3 correctness validation — 2026-10-03

Implementation slice: deterministic native E3 correctness evaluator and evidence v2.

## Correctness contract

`tensormeld/llamacpp-e3-reference-v1` binds the exact trial spec, llama-cli artifact,
ModelManifest, placement translation, tested workload and expected raw stdout SHA-256.

## Native provenance gate

The evaluator accepts only a retained trial whose execution source is
`native-subprocess`, whose exit code is zero, whose stdout is non-empty and whose exact
trial/spec/config/model/plan/placement identities match the supplied trial specification.
Fixture or injected provenance is rejected.

## Runtime identity

Runtime identities must cover exactly the placement devices. Each runtime identity must
name the same worker artifact SHA-256 as the trial's llama-cli artifact.

## QualificationEvidence v2

Successful evaluation emits E3 evidence that additionally carries:
- candidate-plan SHA;
- placement SHA;
- trial-spec SHA;
- correctness-contract SHA;
- ordered runtime-identity SHAs.

AcceptedExecutionBundle requires this v2 identity set to match the exact candidate and
current backend-readiness runtime identities.

## Portable coverage

Tests use synthetic/native-shaped records and verify exact E3 emission, strict
applicability, fixture/injected rejection, output-reference mismatch, runtime worker/device
changes, plan/spec/placement tampering and workload mismatch.

No real llama.cpp binary, GGUF inference, CUDA/HIP kernel or physical GPU is executed by
this validation.

GitHub Actions result: PR #16, workflow `Portable Python tests`, run #178 (37147238835) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. Hosted CI validates E3 evaluator semantics and QualificationEvidence v2 compatibility using synthetic/native-shaped records only; it is not target-host llama.cpp/GPU correctness evidence.
