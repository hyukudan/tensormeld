# Private direct-GPU commissioning preparation

The native engine's experimental commissioning executable is separate from
production inference. Its default profile exercises one opaque GPU payload,
with independent controller completion, GPU acquire, full-byte readback,
consumer credit and receive repost. Repeat with reversed roles in a fresh epoch.
Explicit session-v2 profiles additionally reuse one import for sequential,
alternating-direction or bounded paired TX/RX operations.
No measured path is available until the exact target observations exist.

`python -m tensormeld.transport_commissioning` prepares private plans and checks
paired native receipts. It launches no process, changes no host configuration and
does not emit E2/E3 qualification, `DirectGpuStreamQualification`, resource leases
or execution authorization. Matching receipts are `OBSERVATIONS_MATCH`, requiring
independent review of actual native execution and the exact coherence path.

## Plan contract

The closed schema is `tensormeld/direct-gpu-commissioning-plan-v1`. It contains
`campaign_id`, exactly two `nodes`, 1–128 `runs`, and both
`production_qualified: false` and `launch_authorized: false`. Unknown fields,
boolean integers, duplicate node/run identities and repeated connection epochs
are rejected. Every v1 run uses sequence 1 and a fresh peer-agreed epoch, with
exactly one transfer per invocation. A direction reversal
is a separate invocation in v1. Session-v2 semantics are explicit below and are
never silently applied to a v1 plan or receipt.

Each node contains `id`, `expected` and `artifacts`. `expected` binds
`local_endpoint_identity` and `gpu_uuid` as nonzero lowercase 32-digit hex,
`gpu_architecture: "gfx1151"`, positive integer `gpu_runtime_version` and
`gpu_driver_version`, `kernel_identity`, `driver_build_id`,
`driver_identity: "thunderbolt_stream:gnu-build-id:<driver_build_id>"`, and
`abi_identity`. The driver's loaded GNU build ID is observed independently by
the native executable; an operator-supplied module hash cannot replace it.

`artifacts` pins `harness_binary`, `source_record` and `kernel_build_record` by
absolute local `path` and complete-file `sha256`. The full kernel record uses
`strix-private-full-kernel-build-v1`; its release and stream-module GNU build ID
must match `expected`. Copies of remote build artifacts must be reviewed and
retained locally before preparing the plan. A source list or binary hash is
provenance, not an attestation of execution or source completeness.

Each run contains `run_id`, `sender`, `receiver`, `physical_link_id` (string),
`connection_epoch`, `sequence`, `payload_bytes`, `pattern_seed`, `model_plan_id`,
`request_id`, `generation`, `boundary_id`, `shape_fingerprint`, and `dtype_id`.
These are exact native descriptor identities, not a serialized model request.
The native executable's host-only config check determines its fingerprint from
the declared campaign/run/pattern/payload inputs. Pin that value in the plan.
Payloads are bounded to 4095 × 4096 bytes, leaving a dedicated terminal frame.

Prepare from explicit, reviewed host observations:

Before hosts arrive, `template --out /private/new-input-template` writes 28
unresolved cases covering both directions, two seeds, 1 byte, frame boundaries,
64 KiB, 1 MiB and the maximum payload. Hardware/build/epoch/fingerprint inputs
remain `null`; the template deliberately fails executable-plan validation until
the operator supplies actual bindings. No fixture identity is substituted.

```sh
python -m tensormeld.transport_commissioning prepare \
  --plan /private/commissioning-plan.json --out /private/new-campaign
```

Output is an immutable copy of the plan and a preparation record with
`executed: false`. Missing hardware identities are unresolved inputs; do not
replace them with fixture identities. Output paths must be outside Git and
fresh. Personal host inventory, prompts, receipts and build products stay private.

`prepare-from-configs --inputs INPUTS --out NEW_PRIVATE_DIRECTORY` can assemble
the plan from SHA-pinned native `--check-config` JSON files. The closed input
schema `tensormeld/direct-gpu-commissioning-config-inputs-v1` contains
`campaign_id`, two `{id, artifacts}` node bindings and runs containing
`{run_id, sender, receiver, sender_config, receiver_config}`. Each config reference
is an absolute path/hash artifact. Both role reports must agree on every
descriptor field and independently reconstructed pattern hash; endpoint identities
must be reciprocal and stable across cases. Host-only configs declare operator
expectations. They are neither hardware observations nor execution receipts.

## Paired checks

```sh
python -m tensormeld.transport_commissioning review-pair \
  --plan /private/new-campaign/plan.json --run-id forward-small \
  --sender /private/sender.json --receiver /private/receiver.json \
  --out /private/forward-small-review.json
```

The exact schema `strix-direct-gpu-commissioning-v1` requires both endpoints,
current runtime/module identities, the whole descriptor, one completed transfer,
and released cleanup. Sender DMA completion, remote ACK and remote consumer
credit are distinct required stages. Receiver full acquire, unpack/consumer
completion, full-byte verification, credit and actual repost are independently
required. Inapplicable stages must remain false. Partial execution, retained
cleanup, a missing stage, another peer/epoch/shape or any native error cannot pass.

The checker independently reconstructs the wrapping 64-bit splitmix byte
pattern v1 from position, seed and sequence, then recomputes its SHA-256. It
compares the receiver's full readback hash and both expected hashes against that
oracle. Two matching forged checksums are insufficient. The native receiver also
checks every acquired/unpacked byte; the Python hash check cannot prove that
those GPU operations occurred.

`review-campaign --plan PLAN --pairs PAIRS --out REPORT` requires exactly one
`{run_id, sender, receiver}` receipt pair per declared run. It preserves every
failure and never averages away a broken direction or payload size. CLI review
returns nonzero for failed observations or malformed artifacts. Artifact pins
are rechecked before and after evaluation; trusted local inputs must be quiescent.

## First campaign order

1. Run local operator/state fixtures, floating eager/graph comparison and the
   eager integer projection fixtures independently on each APU.
2. Establish single-host model/token/state references and safe memory limits.
3. Measure the copied CPU transport baseline over the configured physical route.
4. Commission the direct GPU path in both directions with small, partial-frame
   and large payloads, first in fresh epochs and then with stable imported-pool
   reuse, alternating directions and explicit bounded duplex pairs. Use loopback control through an
   operator-established authenticated tunnel; do not open an unauthenticated LAN
   listener or let a peer supply executables.
5. Independently review exact driver/GPU/pool identity, release/acquire coherence,
   peer lifecycle and cleanup evidence before production path qualification.
6. Run model continuity and preparation-inclusive end-to-end measurements only
   after the local model and transport gates pass.

Workstation-only, each companion alone, copied two-companion execution and
qualified direct two-companion execution are separate comparison rows. A CUDA
workstation uses its own qualified adapter; this commissioning executable does
not implement a mixed HIP/CUDA direct-memory path. The PC can own the front door
without relaying every activation. No placement is selected from nominal cable
rate or GPU arithmetic throughput.

## Imported-pool session-v2 checks

`tensormeld/direct-gpu-commissioning-session-plan-v2` has the v1 root fields plus
`session_id`, `schedule` and `concurrency`. A session holds one exact physical
link/epoch and each endpoint's exact FD/device/pool/import owner. It contains
2–128 ordered transfers with run IDs `SESSION.s0`, `SESSION.s1`, etc., sequences
1..N and independent per-transfer fingerprints/pattern hashes. Total payload is
bounded to 128 MiB. `sequential` preserves initial roles, `alternating` reverses
them each time, and both have concurrency 1. `duplex_pairs` requires an even
count, opposite roles within each adjacent pair and concurrency 2. A pair has
one TX and one RX; both retire before the next pair. This does not qualify a
production duplex backend or authorize a larger pipeline.
`concurrency:2` describes the bounded owned-operation capacity and paired
lifecycle, not a measurement of simultaneous physical DMA. Completion receipts
do not contain matched controller start/end intervals, so session review always
reports `physical_overlap_status:"UNKNOWN"` and
`controller_overlap_qualified:false`, even when both directions complete.
Physical overlap needs independently collected timing/traces on the actual path;
neither host submit ordinals nor completion-poll order can establish it.

The native `--iterations N` profile creates a session; add
`--alternate-directions` or `--duplex-pairs` as appropriate and optionally
`--payload-sizes 1,4097,5120,...` for exact varying per-transfer geometry. The
first size equals `--payload-bytes`, both hosts declare the same schedule and
descriptors, and local roles remain reciprocal. `--check-config` emits
`strix-direct-gpu-commissioning-session-config-v2` without native work.

`prepare-session-from-configs --inputs INPUTS --out PRIVATE_DIRECTORY` accepts
the closed `tensormeld/direct-gpu-commissioning-session-config-inputs-v2` schema:
`campaign_id`, `session_id`, two `{id, artifacts}` `nodes`, `node_a`, `node_b`,
and SHA-pinned `node_a_config`/`node_b_config` files. It verifies reciprocal
roles/peers, the entire ordered descriptor schedule, independently reconstructed
sequence-dependent payload hashes and stable local endpoint observations.
`prepare-session --plan PLAN --out PRIVATE_DIRECTORY` accepts a reviewed
session plan directly. Both preparation paths remain private and nonexecuting.

```sh
python -m tensormeld.transport_commissioning review-session \
  --plan /private/reuse/plan.json --node-a node-a --node-b node-b \
  --receipt-a /private/node-a.json --receipt-b /private/node-b.json \
  --out /private/reuse-review.json
```

The exact `strix-direct-gpu-commissioning-session-v2` receipt stores final
released/retained cleanup and ordered full
`strix-direct-gpu-commissioning-transfer-v2` records. Their
`cleanup:"import_held"` records intentional retention between operations, not
release; every transfer must keep the same endpoint, actual device instance,
pool identity and import generation, matching the outer import fields. Review
requires complete exact schedule coverage, every role-specific completion and
full readback hash, and final released cleanup on both hosts. A missing suffix,
retained cleanup, an owner replacement, replay, reordered pair or failed
direction fails the whole session; no averaging or inferred qualification occurs.
Legacy receipt-v1 requires per-run release and rejects these v2 schemas.

These profiles do not implement HIP/CUDA or arbitrary mixed-device native
importers. Independent GPU export/import, visibility/event, driver mapping and
allocator ownership adapters remain software prerequisites for those devices,
followed by their own physical qualification. Changing `gpu_architecture` in a
plan cannot enable an unimplemented native path.
