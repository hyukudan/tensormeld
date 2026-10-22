# ADR-0030 — Native E3 bootstraps from a pre-E3 placement, not an execution bundle

Status: **accepted**  
Date: 2026-10-03

## Context

The AcceptedExecutionBundle intentionally requires applicable E3 model qualification.
The first model-aware llama.cpp placement translator originally consumed that bundle.
That is correct for post-E3 execution, but using the same requirement for the native trial
that must produce E3 creates a circular dependency.

A qualification run needs the same exact model/device/block placement guarantees without
claiming the workload is already qualified or executable.

## Decision

TensorMeld keeps the post-E3 AcceptedExecutionBundle translator and adds a separate
pre-E3 qualification placement path.

The qualification placement starts from:

- exact Config and PlanningInput;
- a planner candidate whose canonical plan SHA is recomputed;
- exact adapter representability;
- the pinned llama.cpp revision;
- current native device-binding SHA and engine device mapping;
- the exact ModelManifest;
- a complete matching GGUF tensor index;
- exact full `blk.N` coverage on one local compute node.

It produces only a non-qualified placement fingerprint and closed placement argv fragment.

TensorMeld also introduces `tensormeld/llamacpp-native-trial-v1`.

The initial native trial requires:

- an approved local llama-cli artifact SHA-256;
- exactly one approved local GGUF whose file name, size and SHA-256 match ModelManifest;
- the pre-E3 qualification placement;
- bounded prompt/context/predict values.

The generated llama-cli argv fixes deterministic subprocess-oriented controls:
`--seed 0`, `--temp 0`, `--simple-io`, `--single-turn`,
`--no-display-prompt`, `--no-show-timings`, and `--color off`.

Inherited `LLAMA_ARG_*` environment variables are removed before the default subprocess
launch so they cannot override the command line.

A successful process is trial evidence only. Exit code zero or non-empty stdout cannot
create QualificationEvidence. Injected runners cannot be labeled as native-subprocess
evidence.

## Consequences

- Native E3 correctness can now be produced without an impossible pre-existing E3 bundle.
- Post-E3 execution still requires AcceptedExecutionBundle and all later gates.
- The first trial supports a single-file GGUF only; split checkpoints remain a later
  extension.
- Portable CI validates the process/argv/evidence boundary using a fixture CLI, not
  llama.cpp model execution.
