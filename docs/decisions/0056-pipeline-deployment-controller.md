# ADR-0056 — A layer pipeline is deployed from per-process manifests bound to engine instances

Status: **accepted** (controller tested with fake engines; no real engine runs the roles yet)  
Date: 2026-10-20

## Context

ADR-0055 ranks placements but executes none. The first split to execute is a two-stage
layer pipeline: a FRONT stage (embedding and the first layers, the HTTP API, tokenizer and
sampler) on a fast but preemptible device, and a TAIL stage (the remaining layers and the
head) on a device group, with the engines carrying the residual stream over their own
TCP wire. The fast device can be taken back by its owner at any time, so the deployment
also needs a planned way back to a placement without it.

Two findings of the ADR-0055 review apply directly: a placement must be bound to one
engine instance and its group mode (capabilities of different binaries must not be
unioned), and launching must not open a path for a peer to supply executables or shell.

## Decision

1. **Plan layout.** A `hetero-pipeline` candidate carries `layout.stages[]`: set, devices,
   layer range `[first, end)`, the exact part ids, parts of its layers placed elsewhere
   (`remote_parts`) and parts it runs for another stage (`serves_parts`). Every candidate
   carries `sets` (set id → devices).
2. **Inventory** `tensormeld/pipeline-inventory-v1` (operator-owned, local file): nodes
   (numeric private/loopback address, work directory, `local` or `ssh` launcher), engines
   (binary path, approved sha256, backends, model path, a closed argv form, allowed
   environment, how its description is obtained: recorded file, probe of the approved
   binary, or an explicit `unchecked` reason), and the FRONT, TAIL and fallback bindings
   (set, engine, ranks → devices, group mode, transport, ports).
3. **Manifest** `tensormeld/deployment-manifest-v1`, one per process: deployment id and
   epoch, plan sha256 and candidate, process (position, role, rank, world), node, devices,
   engine (binary, sha256, backend, argv form, group mode, transport, description record),
   model, layer range and parts, endpoints, secret *file locations*, environment and the
   argv. The manifest hashes itself and validation re-derives the argv from its fields.
   The binding is strict: the stage set maps to one binding whose ranks cover exactly the
   set's devices; the engine's own description must declare the role, the group mode
   (world, transport) and every capability the stage's parts require.
4. **Argv forms are code.** `pipeline-tail-v1`, `pipeline-front-v1` and `group-serve-v1`
   are the engine contract's command lines; an inventory selects a form, never writes argv.
5. **Launch.** Local processes start without a shell and with a minimal environment.
   Remote processes start through the operator's own SSH client in batch mode; the remote
   side runs a fixed `sh` wrapper (`env -i`, the manifest's variables only) whose stdin is a
   dead-man switch: when the controller goes away, the remote process is terminated.
   A random 32-byte session token per deployment is written to `0600` files (remote:
   `umask 077`, content over the SSH channel, never in argv); a non-loopback fallback HTTP
   listener also gets a generated API key. The FRONT HTTP API binds loopback only.
6. **Order and health.** TAIL ranks start, rank 0 prints `READY` (role and layers checked),
   other ranks must stay alive; then the FRONT starts, prints `READY` and answers
   `/health` with `stage: connected`. Liveness and FRONT health are watched continuously.
7. **Route.** The TensorMeld router (ADR-0054) is the optional stable front end: it proxies
   HTTP to the FRONT and later to the fallback. It relays HTTP bytes only; activations and
   state move between engines. Without it, clients use the reported FRONT URL.
8. **Switchover.** A planned retirement holds new requests in the router's bounded queue,
   waits for the FRONT's in-flight requests, drains the FRONT, asks it for the request
   state (`/admin/export-state`, its layers plus the TAIL's), stops both stages, starts
   (cold) or reuses (standby) the fallback, imports the state there (`/admin/import-state`,
   sha256 checked), and routes to the fallback. A stage process that dies loses its
   layers' state: the controller either stops or starts the fallback empty, and reports
   `state_lost: true`. Nothing is migrated silently.

## Consequences

- Every running process can be traced to one plan candidate, one approved binary and one
  description of that binary. A capability missing from the engine that runs a stage is a
  deployment error, not a runtime surprise.
- The SSH launcher depends on the operator's SSH configuration and keys; TensorMeld adds
  no agent, listener or credential of its own on the nodes.
- The state handoff routes are a proposal to the engine (`tensormeld/state-handoff-v1`);
  until an engine implements them, a deployment reports `state_handoff: false` and a
  retirement starts the fallback empty.
- Only two-stage pipelines deploy. N stages, prefill/decode pairs and part splits need
  their own forms and roles.
- A deployment is not an execution qualification: results stay `qualified: false`.
