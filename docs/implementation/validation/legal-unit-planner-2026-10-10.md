# Legal-unit advisory planner validation — 2026-10-10

Implementation slice: bounded advisory resident-capacity planning over adapter-declared
legal model units and explicit cut boundaries.

## Scope

The planner assigns complete legal units only to current policy-eligible allowed devices,
permits owner changes only after declared cut boundaries, and charges hard-resident plus
reclaimable/file-backed tensor bytes to the owning physical pool.

It deliberately does not model compute, persistent state, workspace, staging,
communication or native execution. Results remain qualified=false and executable=false.

## CI

Pull request #33 completed with all seven portable jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

An earlier test commit failed only because a unittest helper was named run and shadowed
unittest.TestCase.run. Commit 84888e4745ea70aa810a73f969fe3c0727e362f6
renamed the helper; the PR CI then passed.

## Result

PR #33 merged as af7281016996f79fd7f97a8a0be593e496e32157.

No real GPU/model performance measurement or finer-grained native execution is claimed.
