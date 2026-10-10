# Directional path evidence validation — 2026-10-10

Implementation slice: exact conservative directional device-to-device path evidence.

## Scope

The contract binds directed paths to the current TensorMeld config and endpoint runtime
identities. Transfer lookup uses explicit payload-size buckets with conservative upper
bounds only.

No nominal link-speed inference, interpolation, extrapolation or automatic multirail
aggregation is permitted. Parallel paths are treated as alternatives.

## CI

Pull request #35 completed with all seven portable jobs green:

- Linux / Python 3.11;
- Linux / Python 3.13;
- Windows / Python 3.11;
- Windows / Python 3.13;
- optional adapter dependencies on Linux;
- optional adapter dependencies on Windows;
- real mTLS loopback integration.

## Result

PR #35 merged as 661fbd3d7a88d8af27bec5b4ce73f5825b5f5d6f.

This validates the software contract only. No real Ethernet/USB4/CUDA/HIP path
measurement or finer-grained native execution is claimed.
