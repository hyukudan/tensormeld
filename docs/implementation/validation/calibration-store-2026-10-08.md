# Calibration persistence and aging validation — 2026-10-08

Implementation slice: immutable placement-calibration persistence with explicit age
limits and current-environment revalidation.

## Scope

This slice persists already-validated placement calibrations across process runs. It
does not create measurements, run a model, qualify hardware, or make a planner candidate
executable.

A persisted calibration is eligible for measured preference only after:

- bounded age/future-date checks;
- exact config/planning/candidate identity;
- exact llama.cpp build-package identity;
- exact current runtime identities, which include worker artifact, OS, driver/runtime,
  physical device and topology fingerprints;
- recomputation of the derived normalized objective;
- optional native-target provenance requirement.

## CI

Pull request #28 portable workflow run 37778748286 completed with all seven jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

An earlier branch run failed before the persisted objective was removed from parser input.
Commit 08eab78c4ebe5c53ba2bbb7d7ac9a07f33d331b5 fixed this by recomputing and checking
the derived objective on load. Subsequent branch and PR CI passed.

## Result

PR #28 merged as 013752df732d71e97118b6b46881c4e445b56750.

This is portable contract/fixture evidence only. No RTX, Strix Halo, CUDA, HIP or real
model performance calibration was collected in this slice.
