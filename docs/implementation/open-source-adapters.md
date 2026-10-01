# Open-source integration boundaries

## Host inventory: integrated

`probe` uses the optional `psutil==7.2.2` provider for host RAM and interface statistics.
Read-only platform fallbacks remain available without this dependency. Observations
identify the provider/version and remain unqualified. Interface speed is not measured
inference payload bandwidth, and host available memory is not a GPU allocation limit.

```bash
python -m pip install ".[system]"
python -m tensormeld probe --out local/inventory.json
```

## GGUF catalog: upstream adapter integrated in portable CI

```bash
python -m pip install ".[gguf]"
python -m tensormeld inspect-gguf /path/to/model.gguf --trusted-local-file --out local/model-index.json
```

For split files, supply each local path explicitly. Partial sets are marked incomplete;
inconsistent metadata, duplicate names/indexes and out-of-file tensor ranges fail.
No automatic model download or remote Python execution is allowed.

The reader is the upstream package, not a rewritten binary parser. Inspection maps
local files read-only; an index hash is NOT a checkpoint hash. The trusted-local flag is
an explicit precondition, not a sandbox. Hostile-file resource limits/process isolation
are still pending, so agents must not expose this routine to untrusted remote input.

Portable optional-dependency CI installs `gguf==0.19.0` and exercises the real upstream
writer/reader roundtrip on Windows and Linux. That validates the catalog adapter only;
it does not declare model execution support.

## llama.cpp native probe: implemented, no real engine binary qualified yet

The third-party registry pins llama.cpp source revision
`552f18f912a32ea86edf82e2b76431cb7131538d`.

The trusted-local probe deliberately reuses the upstream executable's own discovery
surface instead of reproducing backend enumeration:

```bash
python -m tensormeld probe-llamacpp /path/to/llama-cli \
  --trusted-local-binary \
  --expected-sha256 <artifact-sha256> \
  --out local/llamacpp-probe.json
```

The probe computes the artifact SHA-256 and executes only:

- `--version`;
- `--list-devices`.

It uses no shell, no stdin, bounded output and a bounded timeout. The observed source
commit must match the pinned revision. The device parser exports engine-local name,
description, total memory and transient free memory.

No model is loaded and no listener is started. Engine-local names such as `CUDA0` or
`ROCm0` are **not** automatically mapped to TensorMeld identities or backend policy.
A successful probe remains non-qualified and non-executable.

Current tests use a harmless injected runner/fixture. No real llama.cpp CUDA/HIP binary
was downloaded, built or run in the development environment for this milestone.

## Alternate native engines

The registry also records the pinned `llama-halo-hybrid` candidate. It remains a
separate adapter evaluation. Never assume protocol interoperability between upstream and
fork workers simply because they share ancestry.

Do not reuse unauthenticated experimental RPC directly as the product's LAN security
boundary. A secure wrapper's overhead must eventually be measured. No GPU kernel,
CUDA/HIP binary or native inference execution is included in this snapshot.
