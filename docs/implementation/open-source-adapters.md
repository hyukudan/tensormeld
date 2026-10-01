# Open-source integration boundaries

## Host inventory: integrated

`probe` uses optional `psutil==7.2.2` for host RAM/interface observations. Interface
speed is not measured inference payload bandwidth, and host available memory is not a
GPU allocation limit.

## GGUF catalog: upstream adapter integrated in portable CI

TensorMeld reuses upstream `gguf==0.19.0` for trusted local GGUF inspection. A metadata
index is not a checkpoint hash, runtime memory manifest or execution qualification.

## llama.cpp native probe, binding and backend readiness

The third-party registry pins llama.cpp source revision
`552f18f912a32ea86edf82e2b76431cb7131538d`.

The trusted-local probe executes only upstream version/device discovery. Engine-local
names remain unresolved until an explicit approved binding is applied.

A successful binding emits runtime state `observed`, not `ready`. Backend identity
comes from TensorMeld config, never from labels such as `CUDA0` or `ROCm0`. At most
one mapped device may report free memory for a physical pool.

The next narrow gate reuses the pinned upstream `test-backend-ops` target. TensorMeld
fixes the invocation to test mode, the explicitly bound engine device, the `ADD`
operation, SQL output and one worker. The SQL text is parsed as data and never executed.

At the pinned revision, selecting a backend name that is absent can still return process
success after every available device is skipped. TensorMeld therefore does not trust exit
code alone. It requires at least one successful supported `ADD` result row for the
exact bound backend, verifies the reported pinned source revision, and rejects any failed
supported target row.

Only a strict pass changes that device from `observed` to `ready`; it still creates no
reservation and leaves `qualified=false` and `executable=false`.

Portable tests use injected output and are not hardware evidence. No real llama.cpp
CUDA/HIP backend self-test, native GPU inference or distributed inference is claimed by
this repository snapshot.

## Alternate native engines

The pinned `llama-halo-hybrid` candidate remains a separate evaluation. Do not assume
protocol interoperability with upstream workers.

Unauthenticated experimental RPC is not TensorMeld's LAN security boundary.
