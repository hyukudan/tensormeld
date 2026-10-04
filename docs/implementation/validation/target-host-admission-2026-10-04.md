# Target-host admission orchestration validation — 2026-10-04

Implementation slice: reserve → fresh launch recheck → AcceptedExecutionBundle.

## Gates

The orchestrator requires:
- intact target-host qualification handoff;
- intact admission-ready native runtime-manifest collection;
- native-adapter manifest provenance;
- exact config/planning/candidate/model/adapter/worker identities;
- complete operator coverage;
- exact backend-readiness coverage;
- one explicit compute node for the current local llama.cpp shim;
- distinct reservation and launch observation IDs.

It reserves manifest preparation peaks, performs launch recheck on the second observation
and then calls the existing AcceptedExecutionBundle gate.

## Rollback

Launch rejection releases the reservation. Any later bundle-construction exception also
attempts deterministic release.

## Output

A successful result reports:
- reservation_created=true;
- launch_authorized=true;
- execution_authorized=true;
- inference_started=false;
- real_model_inference=false.

No inference process is launched by this orchestration layer.

GitHub Actions result: PR #19, workflow `Portable Python tests`, run #203 (37228884230) passed all 7 jobs after one earlier run exposed a collection-provenance consistency gap. The fix now requires the collection record's `runtime_manifest_provenance` to equal the parsed manifest provenance. Linux and Windows Python 3.11/3.13, optional-adapter integration on both operating systems, and the dedicated real-mTLS-loopback job all passed.
