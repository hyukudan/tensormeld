# Closed local qualification campaign runner

`python -m tensormeld.qualification_campaign /absolute/private/campaign.json`
executes an explicitly configured sequence of existing local TensorMeld commands,
selected portable test suites, three fixed native experiment invocations, pinned
CTest selections and an exact reference-bound native model trial. It
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
model manifest is bound; stages other than `native_model_trial` do not independently
open or hash model weights. The model trial also validates the exact local GGUF
through the existing approval contract. A native campaign requires a `native-adapter` runtime manifest, but that
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
| `native_ctest` | Pinned CTest inventory followed by serial execution of exact selected tests | Pinned `binary`, typed pinned `source_record`, allowlisted `tests` |
| `native_model_trial` | Existing closed llama.cpp native trial specification and runner, exact stdout reference comparison | Pinned `binary`, `gguf`, `planning`, `candidate`, `placement_binding`, `bound_result`, `gguf_index`, `controls`, `reference` |

The native stages require `mode: native`; fixture campaigns cannot launch them.
Their binary is an explicit local trust decision made by the configured operator,
not executable content supplied by a peer. A native experiment's `source_record`
pins a private build/source provenance artifact. It is retained as a reference in
the copied campaign input, rather than interpreted as native qualification proof.
Native lab output may be plain text; its bounded raw bytes and hashes are stored.
Names, hashes and exit codes do not prove that a binary implemented the intended
experiment. Review its build/source record and the raw result independently.

`native_ctest` accepts only these fixed names: `strix-integer-projection-tests`,
`strix-graph-inputs-tests`, `strix-host-upload-tests`,
`strix-projection-reduction-tests`, `strix-low-precision-tests`,
`strix-low-precision-vectors`, `strix-hip-graph-inputs-tests`,
`strix-integer-projection-execution-tests`, `strix-graph-attention-state-tests`,
`strix-hip-graph-attention-state-tests`, `strix-copied-stream-transport-tests`,
`strix-integer-projection-runtime-tests`, `strix-graph-operator-control-tests`,
`strix-graph-ple-tests`, `strix-direct-transport-gpu-export-tests`,
`strix-hip-integer-projection-runtime-tests`, `strix-hip-direct-transport-gpu-export-tests`,
`strix-hip-direct-gpu-pool-allocator-tests`,
`strix-hip-gdn-launcher-tests`, `strix-hip-indexed-attention-launcher-tests`,
`strix-hip-serial-graph-tests`, `strix-hip-launch-context-tests`,
`strix-linux-direct-gpu-stream-uapi-tests`, `strix-linux-direct-gpu-stream-logic-tests`,
`strix-linux-direct-transport-driver-tests`, `strix-direct-gpu-boundary-staging-tests`
`strix-hip-direct-gpu-stream-gpu-tests` and
`strix-direct-gpu-stream-commissioning-tests`, `strix-raw-baseline-recording-tests`
and `strix-direct-gpu-stream-driver-observation-tests`. Caller-supplied
regular expressions, labels, test arguments, presets, dashboard scripts and
fixture setup are not supported. The runner generates a new private CTest file
from only the exact pinned selected binary records, disables automatic fixture
additions and validates that generated file's JSON inventory before running tests.
It never evaluates the original `CTestTestfile.cmake`, `CTestCustom.cmake` or
included scripts, even during inventory: CTest's original show-only mode can run
CMake commands. Commands must use the exact pinned local executable with no
arguments, except the fixed `vectors` argument for `strix-low-precision-vectors`.
Environment, fixture and result-inversion controls cannot enter generated tests.

Its `source_record` contains exactly `record_schema:
"tensormeld/ctest-build-record-v1"`, an absolute `build_directory`, lists of
SHA-pinned `source_files` and `build_files`, and `test_binaries` records with
`name`, pinned `artifact` and optional closed `properties`. The properties accept
only finite `timeout_s` in (0,300], boolean `disabled`, explicit
`skip_return_code: 77`, and an absolute existing `working_directory` within the
private build directory. `WILL_FAIL`, pass/fail/skip regular expressions and every
unknown property are rejected. Original CMake properties are provenance only and
are never copied; an original `WILL_FAIL` cannot turn a failing selected binary
into a pass. Build files and test binaries remain inside the
build directory. The build directory must be a private out-of-source build
outside every Git checkout, and a nested working directory cannot itself be a
Git checkout. CTest runs with the fresh private metadata directory as its current
and test directory, so its automatic logs also remain outside Git.
Every `CTestTestfile.cmake` beneath the declared build directory must be
pinned for provenance, without evaluating it. The binary records cover exactly
the selected names. These are trusted local binary/build inputs, not a sandbox
for hostile test executables or a proof that a
declared source list reproduces every linked dependency. They are revalidated
between inventory and dispatch and after execution. The generated metadata is
also checked before dispatch and after execution, then retained with its hash.

CTest's bounded JUnit XML is retained with a hash alongside raw CTest output.
Per-test outcomes and timeouts remain explicit. A passed aggregate requires
nonzero executed tests, complete selected-test coverage, and no failures. Empty
selections fail. All-disabled or all-skipped selections remain `SKIPPED`, even
when CTest returns its no-tests error for a disabled selection. Skips are never
counted as successful test execution.
Unsupported JUnit roots and error-tag cases cannot pass. Retained stdout, stderr
and JUnit are bounded; this does not establish a bound on all CTest internal
collection memory or temporary logs. The trusted executable/process lifetime
boundary and stage deadline remain separate from host memory qualification.

`native_model_trial` rebuilds placement and the native trial specification through
the existing planning, candidate-integrity, exact representability, placement
binding, approved-worker and single-file GGUF contracts. It accepts no serialized
argv. The pinned `controls` JSON contains only `prompt`, `context_tokens` and
`predict_tokens`; context matches the exact planned workload and output length
stays within its limit. The pinned `reference` uses the existing
`tensormeld/llamacpp-e3-reference-v1` parser and must bind the rebuilt spec, worker,
model, placement and exact C1 trial workload. The bound worker/node and selected
local owners must match campaign runtime bindings before dispatch.

The existing trial runner invokes the actual bounded subprocess and records its
unchanged narrow trial provenance. Success requires zero exit, nonempty stdout
and an exact match to the configured reference stdout hash. The result retains
the trial and comparison; it does not emit E3 `QualificationEvidence`, upgrade
the campaign to hardware readiness, create admission, or authorize a server
request. E3 evaluation and E4 CLI/server equivalence keep their independent
existing gates. An approved literal reference is a correctness input, not a
new execution authorization path.

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
all-skipped portable or CTest suite is skipped. A failure or timeout prevents subsequent
enabled stages, which remain not run. Malformed/mismatched worker records fail
closed. Portable output is labeled `portable-fixture-tests`; native experiments
and CTest use `trusted-local-experiment`; the bound model trial uses
`trusted-local-model-trial`; existing commands use `local-cli`.

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
New integration controls execute actual pinned local CTest processes, validate
nonzero passed counts, disabled and empty selections, per-test timeout and hidden
command/environment omission. Marker controls verify original `execute_process`,
includes and `CTestCustom` files never execute; failing binaries cannot be
promoted through original `WILL_FAIL`, and typed result transformers are rejected.
Tiny model-shaped fixtures validate trial
reconstruction, reference/control/GGUF drift and a real failing subprocess;
mocked success remains portable contract evidence, not model execution.
The same campaign suite also passed on Linux through a local WSL environment,
including real process-group parent-death cleanup; its Windows Job failure test
is explicitly skipped there. The extension's actual CTest and failed model-trial
subprocess controls also passed on Windows and Linux. That validates the software lifecycle without
claiming native companion GPU execution or hardware qualification.

This slice does not orchestrate arbitrary shell tools, produce E3 qualification,
run E4 performance campaigns, multi-host transport, sustained serving, fresh runtime
identity collection, or independent family-serving authorization. Those need
separate closed input contracts and concrete qualification evidence. Hardware
execution remains explicitly configured future work.
