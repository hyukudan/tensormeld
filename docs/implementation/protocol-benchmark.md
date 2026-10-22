# Fixed-corpus HTTP benchmark

`tensormeld bench-protocol` targets an already running compatible chat endpoint,
including a pair server, a split front endpoint or a replica router. It does not
start workers, acquire GPU turns or grant execution/qualification. Default mode
is a network-free dry run. External comparisons, corpora, prompts, credentials,
endpoint configurations and recordings must stay outside Git.

The short corpus has exactly eight cases: coding, JSON, narrative, prose, math,
reasoning, summary and format. Each case specifies actual acceptable input bounds
within 36–303 tokens and completion bounds within 82–200 tokens. These are measured
tokens, including the selected endpoint's chat template and unique front tag;
word or character counts cannot establish them. All requests use temperature zero,
streaming usage and explicit thinking-off fields. Ignored thinking-off settings,
missing usage, truncation of SSE, errors and out-of-range lengths invalidate a wave.
An ordinary early EOS below the requested minimum is recorded as a length violation;
the harness never pads, suppresses EOS or silently changes sampling to force a result.

Use an original private UTF-8 JSON corpus with schema
`tensormeld/protocol-corpus-v1` and a `cases` array. Each case requires:

- `id`, `category`, `messages` (objects with text `content` and `role`);
- `prompt_tokens: [minimum, maximum]`;
- `completion_tokens: [minimum, maximum]` (maximum is the request output cap).

No third-party prompts or code are required. Optional corpus metadata can record
authorship, local tokenizer hashes and independent offline length checks. Dry runs
validate structure and limits, emit request hashes, and explicitly leave token bounds
unchecked: only a real endpoint's usage validates its rendered input lengths.

```powershell
$env:PYTHONPATH = 'src'
python -m tensormeld bench-protocol --corpus D:/private/short.json --out D:/private/plan.json
# Only inside the owner's authorized final timing campaign:
python -m tensormeld bench-protocol --execute --url http://127.0.0.1:8000/v1 --corpus D:/private/short.json --levels 1,4,6 --run-id final-a --identity D:/private/identity.json --out D:/private/result.json
```

Each category runs a synchronized wave at C1/C4/C6, sequentially across categories
and concurrency levels (88 requests per short sweep). Each stream has a deterministic
unique tag at the beginning; use a new run ID for each recording to avoid prefix-cache
reuse. No hidden warmups or retries occur. Known nonzero cached usage invalidates the
cold wave; absent cached usage remains unknown. Keep weights warm independently and
record that preparation. Set endpoint capacity explicitly: an unsupported C6 yields
an invalid observation, never a silently reduced load. `--repeats` is bounded to ten.

The timing contract records:

- Per-wave wall aggregate: total usage completion tokens divided by elapsed wall time
  from barrier release through the slowest stream and final usage/flush, including
  TTFT, admission, prefill and network. Thread startup is excluded.
- Per-stream TTFT: request start to first nonempty generated delta, ignoring role-only
  chunks. Per-stream wall rate is also recorded.
- Protocol per-stream rate: `(completion_tokens - 1) / (total_s - ttft_s)`. It includes
  terminal flush. An additional first-to-last-delta estimate excludes that flush.
  Speculative multi-token deltas make both first-token-subtraction rates estimates;
  the wall aggregate uses authoritative usage without counting chunks as tokens.
- Per-concurrency ratio of total tokens to summed category wall times, and a separately
  named arithmetic mean of category aggregates. These averages can differ materially.
  Failed or noncompliant waves have no comparable aggregate; partial observations remain
  visible. CLI exit 2 means an executed run was invalid; exit 1 means setup failed.

`--mode long --levels 1` accepts up to 32 private context cases with exactly three
distinct `needles` answer strings, occurring in input order, and explicit token bounds.
Prepare diverse natural text with a real local tokenizer and plant facts around 10%,
50% and 90%; ask for their values in order without repeating them in the final question.
Record actual token offsets and tokenizer/template identity in private metadata.
The harness reports three-needle recall/order, full answer, TTFT, wall and decode rates.
`prompt_tokens_over_ttft_s` includes admission/network/first decode and is an apparent
prefill rate, not a device prefill measurement. Per-segment engine timings require a
separate native recording. Large contexts require explicit endpoint/resource configuration;
the harness does not chunk, resume, reduce context or schedule extra pair turns.

Maximum corpus size is 32 MiB; each stream has 1 MiB events and an 8 MiB response
bound, with a configurable request timeout up to 1,200 seconds. At most six streams
are active. A shared `--budget-s` (900 seconds by default, maximum 1,200) also limits
the campaign; incomplete sweeps are invalid and never have comparable summaries.
HTTPS uses system certificate verification. API keys are read only for
execution and never serialized. Results include private text/optional identity metadata:
write `--out` outside the repository. Identity should bind the exact model/quantization,
template, backend build, precision, speculative policy, transport and cache state.

Portable tests and dry runs prove measurement plumbing only. They do not establish
native inference, model quality, hardware performance or split commit/rollback correctness.
