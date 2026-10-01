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

## GGUF catalog: adapter implemented; upstream integration pending in this environment

```bash
python -m pip install ".[gguf]"
python -m tensormeld inspect-gguf /path/to/model.gguf --trusted-local-file --out local/model-index.json
```

For split files, supply each local path explicitly. Partial sets are marked incomplete;
inconsistent metadata, duplicate names/indexes and out-of-file tensor ranges fail.
No automatic model download or remote Python execution is allowed. Only a small metadata
allowlist is exported; tensor values and tokenizer text are not exported.

The reader is the upstream package, not a rewritten binary parser. Inspection maps
local files read-only; an index hash is NOT a checkpoint hash. The trusted-local flag is
an explicit precondition, not a sandbox. Hostile-file resource limits/process isolation
are still pending, so agents must not expose this routine to untrusted remote input.

The injected-reader tests exercise this adapter contract. A separate real upstream
writer/reader integration test exists and is skipped when the package is unavailable.
During this session, network/package download was unavailable; do not record that skip
as a successful GGUF parse. This release does not declare new model/quantization support
merely because metadata can be indexed.

## Native engines: identified and pinned, not vendored or compiled

The third-party registry records two native candidates. The next integration step is
an isolated, revision-pinned worker adapter with platform/backend smoke tests and exact
placement rejection. Keep alternate engine protocols distinct; never mix clients and
workers from different forks without an interoperability qualification record.

Do not reuse unauthenticated experimental RPC directly as the product's LAN security
boundary. A secure wrapper's overhead must eventually be measured. No GPU kernel,
CUDA/HIP binary or native inference execution is included in this snapshot.
