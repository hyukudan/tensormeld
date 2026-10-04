# Native admitted session lifecycle validation — 2026-10-04

Implementation slice: execution-session lifecycle after AcceptedExecutionBundle.

## Session admission

The session validates:
- intact execution-authorized TargetHostAdmissionResult;
- exact AcceptedExecutionBundle fingerprint;
- backend/bundle identity;
- active launched lease;
- matching lease SHA;
- matching runtime-manifest SHA.

## Lease lifecycle

Portable tests prove:
- successful run releases the lease;
- cancellation before run invokes no backend and releases the lease;
- backend failure releases the lease;
- in-flight cancel returns CANCEL_REQUESTED and leaves the lease active until run
  completion/cancellation boundary;
- wrong-bundle backend is rejected;
- a released/missing lease cannot start a session;
- explicit release after automatic cleanup is idempotent.

The managed session reports `inference_started=true` only after its run path begins, but
fixture execution keeps `real_model_inference=false`.

No real llama.cpp subprocess or GPU inference is executed by this validation; real
subprocess protocol coverage remains in the native-worker fixture tests.

GitHub Actions result: PR #20, workflow `Portable Python tests`, run #208 (37230869251) passed all 7 jobs: Linux and Windows on Python 3.11/3.13, optional-adapter integration on both operating systems, plus the dedicated real-mTLS-loopback job. Portable lifecycle tests validate lease ownership, cancellation/failure cleanup and admitted-bundle identity using deterministic backend execution; they do not establish real llama.cpp/GPU process cancellation.
