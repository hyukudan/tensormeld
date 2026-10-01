# Open-source integration boundaries

## Host inventory: integrated

`probe` uses optional `psutil==7.2.2` for host RAM/interface observations. Interface
speed is not measured inference payload bandwidth, and host available memory is not a
GPU allocation limit.

## GGUF catalog: upstream adapter integrated in portable CI

TensorMeld reuses upstream `gguf==0.19.0` for trusted local GGUF inspection. A metadata
index is not a checkpoint hash, runtime memory manifest or execution qualification.

## llama.cpp native probe and explicit identity binding

The third-party registry pins llama.cpp source revision
`552f18f912a32ea86edf82e2b76431cb7131538d`.

The trusted-local probe reuses upstream discovery:

```bash
python -m tensormeld probe-llamacpp /path/to/llama-cli \
  --trusted-local-binary \
  --expected-sha256 <artifact-sha256> \
  --out local/llamacpp-probe.json
```

It executes only `--version` and `--list-devices`, with no shell, stdin, model or
listener.

Engine-local names are deliberately unresolved after probing. TensorMeld requires an
explicit approved mapping document:

```json
{
  "binding_schema": "tensormeld/llamacpp-device-binding-v1",
  "config_sha256": "<exact-config-fingerprint>",
  "node_id": "pc",
  "artifact_sha256": "<exact-probed-binary-sha256>",
  "approval": "explicit",
  "mappings": [
    {
      "engine_device_name": "CUDA0",
      "tensormeld_device_id": "pc-gpu",
      "memory_reporter": true
    }
  ]
}
```

Apply it with:

```bash
python -m tensormeld bind-llamacpp-devices \
  config.json local/llamacpp-probe.json binding.json \
  --out local/llamacpp-bound.json
```

Backend identity comes from TensorMeld config, not from labels such as `CUDA0` or
`ROCm0`. Unmapped engine devices are reported but not auto-bound.

When multiple configured devices alias one physical memory pool, at most one mapped
engine device may be selected as `memory_reporter`; TensorMeld never sums duplicated
reports for a shared pool.

A successful binding emits a runtime observation with state `observed`, not `ready`.
The ordinary runtime selector accepts the state syntactically but excludes it from
runtime-ready candidates. A later backend self-test/qualification is required before
promotion to ready.

Current native-probe/binding tests use fixtures. No real llama.cpp CUDA/HIP binary has
been downloaded, compiled or run in this development environment.

## Alternate native engines

The pinned `llama-halo-hybrid` candidate remains a separate evaluation. Do not assume
protocol interoperability with upstream workers.

Unauthenticated experimental RPC is not TensorMeld's LAN security boundary. No native
GPU inference or distributed execution is included in this snapshot.
