# Pinned vocabulary tokenization

`tensormeld.llamacpp_tokenization` is a bounded trusted-local adapter to the
registered MIT-licensed llama.cpp `llama-tokenize` tool at revision
`552f18f912a32ea86edf82e2b76431cb7131538d`. It tokenizes explicitly supplied
rendered UTF8 bytes using an exact selected GGUF vocabulary. It does not render
chat messages, decode generated tokens, execute inference or qualify hardware.
No upstream source is vendored; the existing registry and ggml MIT notice apply.

The reviewed implementation loads vocabulary only, rereads `--file` as binary
without stripping its final newline, and passes explicit string length to the
tokenizer. UTF8 bytes, including literal escapes and embedded NUL, are preserved
with `--no-escape`. BOS policy is `model_default` (the tool's decision based on
the selected vocabulary) or `disabled` (`--no-bos`). Special-token parsing true
uses the tool's reviewed default; false adds `--no-parse-special`. The separate
`--parse-special` flag belongs to another upstream example and is never passed.
These semantics are pinned to the [tokenizer source](https://github.com/ggml-org/llama.cpp/blob/552f18f912a32ea86edf82e2b76431cb7131538d/tools/tokenize/tokenize.cpp),
[argument parser](https://github.com/ggml-org/llama.cpp/blob/552f18f912a32ea86edf82e2b76431cb7131538d/common/arg.cpp)
and [common tokenizer helper](https://github.com/ggml-org/llama.cpp/blob/552f18f912a32ea86edf82e2b76431cb7131538d/common/common.cpp).

The closed request schema is `tensormeld/llamacpp-tokenization-spec-v1`:

```json
{
  "schema":"tensormeld/llamacpp-tokenization-spec-v1",
  "model":{"path":"/private/model.gguf","sha256":"ACTUAL_SHA256","bytes":123},
  "tool":{"path":"/private/llama-tokenize","sha256":"ACTUAL_SHA256","bytes":123},
  "source_manifest":{"path":"/private/tokenizer-sources.json","sha256":"ACTUAL_SHA256","bytes":123},
  "build_receipt":{"path":"/private/tokenizer-build.json","sha256":"ACTUAL_SHA256","bytes":123},
  "prompt":{"path":"/private/rendered-prompt.txt","sha256":"ACTUAL_SHA256","bytes":123},
  "prompt_contract_sha256":"ACTUAL_RENDERED_REQUEST_OR_CASE_SHA256",
  "controls":{"bos_policy":"model_default","parse_special":true,"vocabulary_size":153600,"maximum_tokens":32768}
}
```

Placeholders intentionally fail. Model/tool/prompt/manifest/receipt/result paths
are absolute resolved private paths outside every Git checkout. Source files and
installed dependency files may live in their actual checkouts/installations.
GGUF v3 is checked; the full model hash is streamed before and after execution.
The source manifest has schema `tensormeld/llamacpp-tokenizer-source-v1`, exact
`source_revision`, `license:"MIT"` and four `files` objects containing
`upstream_path` plus `artifact` pin. Required reviewed paths are
`tools/tokenize/tokenize.cpp`, `common/arg.cpp`, `common/common.cpp` and `LICENSE`;
their exact bytes/SHA256 are checked against the adapter's reviewed constants.
Keep source-inspection receipts and copies privately.

The build receipt has schema `tensormeld/llamacpp-tokenizer-build-v1`, exact
`source_revision`, `source_manifest_sha256`, `tool_sha256`,
`cpu_only_declared:true`, bounded `compiler`, and `dependencies` pins. Source
revision, CPU-only compilation and completeness of declared dependencies are
trusted operator associations, not attestations derived from an executable.
An observed short build commit is recorded separately; it cannot substitute the
full pinned source declaration. This tool sets `vocab_only` but otherwise uses
default loader/context parameters: accepted `--device none` and
`--n-gpu-layers 0` flags alone do not prove GPU inactivity. Supply an explicitly
trusted CPU-only build. No actual tokenizer/model invocation occurred in tests.

The command is fixed to `--log-disable --offline --model PIN --file PIN --ids
--no-escape --device none --n-gpu-layers 0` plus only the two approved policy
flags. No caller argv, shell, model URL or network worker is accepted. Environment
is narrowed to ordinary path/runtime variables, logging/download/user options
are excluded, and GPU visibility variables are empty. Environment identity is
hashed without publishing its values. A gated Python launcher waits until the
Windows Job or Linux process group is established before starting the tool.
The wrapper stays alive to preserve parent-death/group cleanup, including on
Windows where an exec overlay could exit before its child completes. Timeouts,
output overflow and uncertain pipe/child cleanup fail clearly.

Limits: prompt1MiB, model512GiB, tool2GiB, source/spec/build JSON2MiB, evidence16MiB,
combined stdout/stderr8MiB, declared dependencies128/aggregate2GiB, and at most
1048576 token IDs. IDs must be integers within the explicitly supplied actual
vocabulary; empty, negative, boolean, floating, truncated and log-contaminated
output fails. Exact reviewed ID formatting accepts LF/Windows CRLF and is bound
to the captured stdout hash. Timeout applies to each child invocation (version,
tokenization), within0.1..120seconds. Full-file identity hashing is separate and
can dominate preparation time; this is not a throughput benchmark.

```sh
python -m tensormeld.llamacpp_tokenization /private/tokenizer-spec.json \
  --trusted-local-binary --timeout 120 --out /private/new-token-evidence.json
python -B -m unittest discover -s tests -p test_llamacpp_tokenization.py
```

The Python API exposes `TokenizationSpec.parse`, `tokenization_argv`,
`tokenize_llamacpp`, `validate_tokenization_receipt` and
`load_tokenization_receipt`. Output schema `tensormeld/llamacpp-tokenization-v1`
binds immutable spec identity, actual tokens and little-endian u32 token hash,
observed build version, exact argv, environment hash, adapter/runner/Python file
pins and bounded output/cleanup observations. Supplied test runners always emit
`execution_kind:"injected-fixture"`; `require_observed=True` rejects them.
Actual trusted-local execution emits `trusted-local-upstream-tool` while keeping
`qualified:false`, `native_inference_qualified:false`, `detokenization:"unknown"`
and `chat_template_rendering:"external-unvalidated"`.

Inputs, loaded code/tool/dependencies and environment are trusted and quiescent.
Pre/post pins detect lasting mutation; they cannot prove atomic immutable
executed bytes or detect a malicious rewrite restored between observations.
Receipts are consistency evidence from this trust boundary, not cryptographic
authentication of a host or a model/API qualification claim. Generated output
text needs a separate actual detokenizer or remains an operator assertion.

The native numerical RAW importer accepts an optional pinned
`input_token_evidence` receipt. It requires observed upstream execution and
exact GGUF/rendered-case/UTF8/prefix-ID/vocabulary bindings for both fresh and
restored records. Fixture output cannot satisfy this input binding. The receipt
does not verify an independently supplied decoded output string.
