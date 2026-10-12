# Closed local qualification campaign runner

`python -m tensormeld.qualification_campaign /absolute/private/campaign.json`
executes an explicitly configured sequence of existing local TensorMeld commands,
selected portable test suites and three fixed native experiment invocations. It
returns a small JSON summary and writes bounded raw evidence and a hashed report
to a new private directory outside Git. Python callers can use
`load_campaign()` and `run_qualification_campaign()` from
`tensormeld.qualification_campaign`.

The campaign result records orchestration. It never grants native qualification,
hardware readiness, resource reservations or inference authorization. A successful
portable test, an executable hash, an output saying `PASS`, or a supplied native
provenance label cannot establish E2–E5. Existing native commands keep their own
narrow proof requirements. E3/E4 acceptance and the exact authorized completion
path remain independent.

## Explicit inputs

Configuration uses `tensormeld/qualification-campaign-v1` and rejects unknown keys,
duplicate JSON keys, arbitrary commands/arguments and peer endpoints. All artifact
references contain an absolute local `path` and the SHA-256 of its complete bytes.
The top-level keys are:

| Key | Contract |
|---|---|
| `campaign_schema` | `tensormeld/qualification-campaign-v1` |
| `mode` | `fixture` or `native` |
| `host` | Exact local `hostname`, `platform` and configured `node_id` |
| `artifacts` | `config`, `model_manifest`, `adapter_capabilities`, `runtime_manifest`, and list `runtime_identities` |
| `results_directory` | Absolute private directory outside every Git checkout |
| `stages` | Ordered list of 1–32 closed stages |

The existing loaders check the exact config/profile/model/adapter/runtime tuple.
Runtime identities must cover exactly the manifest devices, including declared
remote devices, and match their configured nodes and runtime worker artifact.
Backend self-test bindings must also match the configured host node and runtime
worker artifact; a result from another node or worker fails before execution.
These files are explicit bindings, not fresh driver/device measurements. The
model manifest is bound; this runner does not independently open or hash model
weights. A native campaign requires a `native-adapter` runtime manifest, but that
declaration alone remains insufficient to qualify hardware or execution.

Each stage has `id`, `kind`, boolean `enabled`, `timeout_s` in 0.1–300,
`max_output_bytes` in 1–4,194,304, and a closed `parameters` object. IDs contain
only letters, digits, `_` and `-`, with at most 64 characters. Disabled stages
require empty parameters and need no absent native executable to be invented.

For example, a configured validation stage is:

```json
{
  "id": "runtime-manifest",
  "kind": "validate_runtime_manifest",
  "enabled": true,
  "timeout_s": 30,
  "max_output_bytes": 262144,
  "parameters": {}
}
```

## Closed stages

| Kind | Reused command or fixed invocation | Parameters |
|---|---|---|
| `validate_config` | `tensormeld validate-config` | `{}` |
| `host_probe` | `tensormeld probe`, read-only inventory | `{}` |
| `validate_runtime_manifest` | `tensormeld validate-runtime-manifest` | `{}` |
| `portable_tests` | Standard-library unittest discovery for one fixed suite | `suite`: `configuration`, `numerical`, `request_envelope`, `native_session` or `c1_serving` |
| `runtime_probe` | Existing trusted-local `probe-llamacpp`, fixed probe flags | Pinned `binary`, equal to bound worker artifact |
| `backend_self_test` | Existing trusted-local `self-test-llamacpp-backend`, bound ADD test | Pinned `binary`, pinned `bound_result`, local-host `device_id`; timeout at most 120 seconds |
| `native_platform_probe` | Trusted configured platform-probe executable, no arguments | Pinned `binary` and `source_record` |
| `native_projection_lab` | Trusted configured integer-projection executable, fixed `--run` | Pinned `binary` and `source_record` |
| `native_graph_fixture` | Trusted configured graph-input fixture executable, no arguments | Pinned `binary` and `source_record` |

The native stages require `mode: native`; fixture campaigns cannot launch them.
Their binary is an explicit local trust decision made by the configured operator,
not executable content supplied by a peer. A native experiment's `source_record`
pins a private build/source provenance artifact. It is retained as a reference in
the copied campaign input, rather than interpreted as native qualification proof.
Native lab output may be plain text; its bounded raw bytes and hashes are stored.
Names, hashes and exit codes do not prove that a binary implemented the intended
experiment. Review its build/source record and the raw result independently.

## Lifetime and provenance

Each stage runs in a new gated Python worker with no shell. Windows assigns that
worker to a kill-on-close Job before granting its stdin gate. Linux starts a new
session, installs `PR_SET_PDEATHSIG` before the gate, checks the parent identity
after installation, and kills its process group if the campaign parent dies.
These paths contain ordinary inherited descendants through timeout, output
overflow, normal worker exit and campaign-parent death. Containment failure
prevents the gate. Other operating systems fail closed for campaign execution.
Trusted programs that deliberately detach or change containment are outside this
contract; this is not a sandbox for hostile local executables.

Stdout and stderr have a combined retained-byte limit. Native capture reserves
one quarter of that limit for raw native bytes, leaving space for base64 evidence
and report metadata inside the bounded outer worker protocol. Both pipes drain
concurrently; timeout or overflow stops the worker and descendants, confirms its
exit, closes the pipes, and joins the readers. The report records observed and
retained byte counts, timeout/overflow flags, confirmed child exit and requested
tree cleanup. If cleanup cannot be confirmed, the stage fails.

Source identity covers TensorMeld Python source, all test Python/JSON files,
example JSON fixtures, the Python executable and Python version. The campaign
and pinned artifacts are revalidated before and after each stage; changed inputs
fail the stage and stop later enabled stages. Native binary identity is checked
immediately before and after subprocess execution. These checks require trusted,
quiescent local inputs. They do not make imports or path-based execution atomic,
nor attest every loaded shared library; a concurrent edit followed by restoration
can evade snapshot checks. They also do not prove declared runtime identities are
current measurements.

The stage environment uses a fixed allowlist for OS paths, locale, library
discovery, selected device visibility and supported runtime controls. It excludes
credentials, `LLAMA_ARG_*`, Python startup hooks, loader injection and
`HSA_OVERRIDE_GFX_VERSION`. The report records the configured environment values
and digest privately; worker/native records include the actual effective digest.
Library paths and selected devices remain visible in this evidence for review.

Each result row separates `executed` from `outcome`: `PASSED`, `FAILED`, `SKIPPED`
or `NOT_RUN`. `executed` requires the worker's validated stage-start record. A
spawn alone does not establish execution. Disabled stages are skipped, and an
all-skipped portable suite is skipped. A failure or timeout prevents subsequent
enabled stages, which remain not run. Malformed/mismatched worker records fail
closed. Portable output is labeled `portable-fixture-tests`; native experiments
use `trusted-local-experiment`; existing commands use `local-cli`.

Every run creates a unique directory containing `campaign-input.json`, bounded
stage `.stdout`/`.stderr`, any native raw evidence files, and `campaign.json`.
The result includes canonical campaign identity, exact manifest/runtime bindings,
code/environment identity, raw evidence hashes, UTC observation time and a
canonical result hash. No previous result is overwritten. These private records
can contain host/runtime details and are never placed in Git by the runner.

## Tested scope and remaining gates

Portable tests exercise actual local stage progression, malformed configuration,
fixture/native separation, output provenance, artifact/code drift, native pin
checks, failure/timeout progression, bounded output and real Windows process
cleanup, including descendant survival attempts and campaign-parent death.
The same campaign suite also passed on Linux through a local WSL environment,
including real process-group parent-death cleanup; its Windows Job failure test
is explicitly skipped there. That validates the Linux software lifecycle without
claiming native companion GPU execution or hardware qualification.

This slice does not orchestrate CTest, arbitrary shell tools, E3 model correctness,
E4 performance campaigns, multi-host transport, sustained serving, fresh runtime
identity collection, or independent family-serving authorization. Those need
separate closed input contracts and concrete qualification evidence. Hardware
execution remains explicitly configured future work.
