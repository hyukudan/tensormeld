# Adapter representability validation — 2026-10-01

Implementation slice: static native-adapter representability gate.

## Scope

This slice validates only whether a declared adapter capability envelope can represent
a synthetic planner candidate without changing its whole-block placement semantics.

It does **not** launch a native engine, reserve memory, qualify a model, authenticate a
remote node, or make any plan executable.

## Local validation

Available Linux/Python 3.13 development environment:

- 155 tests discovered;
- 154 passed;
- 1 skipped because the optional upstream `gguf` package is unavailable in the local
  environment;
- 0 failures/errors.

The 15 new adapter-contract tests cover exact ownership/range requirements, remote and
mixed-backend capability, device/backend identity, coordinator constraints, route mode,
adapter limits, candidate tamper rejection and deterministic report identity.

## GitHub Actions

Workflow run 36865853335 for commit
`e61f69b54f0e6c7547f93d12b695de0e5b185e76` completed successfully.

All six jobs passed:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows.

The optional jobs installed the pinned `gguf==0.19.0` package and ran the upstream
GGUF integration tests. Hosted CI remains CPU/software validation and is not CUDA,
ROCm, Strix Halo, network, or distributed-inference qualification.

## Result

The static representability contract is accepted as an M0.2 control-plane component.
A `REPRESENTABLE` result remains `qualified=false` and `executable=false`.
