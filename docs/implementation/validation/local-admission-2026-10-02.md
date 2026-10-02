# Local reservation/admission validation — 2026-10-02

Implementation slice: atomic local physical-pool leases and launch-time re-admission.

## Contract

`LocalAdmissionController` consumes an exact runtime model manifest plus an admission
snapshot containing a runtime observation, an observation identity and the active lease
IDs already reflected in telemetry.

All manifest preparation peaks are checked as one atomic reservation. Active leases not
reflected in the snapshot are subtracted conservatively. Reflected leases are not charged
again, making the double-subtraction rule explicit.

Launch requires a different observation identity from the reservation observation and
re-runs readiness/pool checks. Rejection leaves the reservation active; release is
deterministic and idempotent.

## Portable coverage

Tests cover reservation/release, concurrent contenders, pending-lease subtraction,
reflected-lease accounting, launch-time recheck under reduced capacity, duplicate IDs,
unknown reflected IDs and physical-pool charging.

These are process-local software tests. They do not prove cross-process arbitration,
remote host authority, CUDA/HIP allocation, Windows GPU behavior, model correctness,
performance or distributed inference.

GitHub Actions result: PR #4, workflow `Portable Python tests`, run #72 (36976422657) passed all 6 jobs: Linux and Windows on Python 3.11/3.13 plus optional-adapter integration jobs on both operating systems. Hosted CI remains portable software evidence, not GPU qualification.
