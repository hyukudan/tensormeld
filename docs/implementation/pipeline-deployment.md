# Pipeline deployment

Status: controller, manifests, launchers, router integration and switchover implemented and
tested end-to-end on loopback with fake engines; not yet run with a real engine (the engine
roles are in progress). ADR-0056.

## Commands

```bash
# 1. Plan (the candidate must be a hetero-pipeline; its layout is in the plan)
PYTHONPATH=src python -m tensormeld plan-hetero MODEL.parts.json HARDWARE.json --prompt 8192 --out plan.json

# 2. Print the process manifests without starting anything
PYTHONPATH=src python -m tensormeld deploy-pipeline plan.json MODEL.parts.json HARDWARE.json INVENTORY.json \
    [--candidate NAME] [--unavailable DEV]... [--link ID=B/s:us]... --dry-run

# 3. Run it (until SIGINT/SIGTERM, --run-seconds, or a failure)
PYTHONPATH=src python -m tensormeld deploy-pipeline plan.json MODEL.parts.json HARDWARE.json INVENTORY.json \
    [--on-failure stop|fallback] [--retire-file PATH] [--run-seconds S]

# Planned retirement of the FRONT device (router configured):
curl -X POST http://ROUTER/admin/pipeline/retire          # 202; progress in GET /api/status
# or create the --retire-file
```

`--unavailable`/`--link` must repeat those given to `plan-hetero` (the plan is bound to the
hardware fingerprint). Exit codes: 0 = dry run or a clean stop while serving (or after a
switchover), 2 = the deployment failed, 1 = invalid input.

## Inventory (`tensormeld/pipeline-inventory-v1`)

```json
{
  "schema": "tensormeld/pipeline-inventory-v1",
  "nodes": [
    {"id": "pc", "address": "127.0.0.1", "work_dir": "D:/tm-run", "launcher": {"kind": "local"}},
    {"id": "node1", "address": "10.0.0.11", "work_dir": "/home/op/tm-run", "launcher": {"kind": "ssh", "target": "op@node1"}},
    {"id": "node2", "address": "10.0.0.12", "work_dir": "/home/op/tm-run", "launcher": {"kind": "ssh", "target": "op@node2"}}
  ],
  "engines": [
    {"id": "front", "binary": "D:/engine/pipeline-front.exe", "sha256": "…", "backends": ["cuda", "hip", "host"],
     "model_path": "D:/models/m.gguf", "argv_form": "pipeline-front-v1", "describe": {"probe": true},
     "env": {"CUDA_VISIBLE_DEVICES": "0"}},
    {"id": "tail", "binary": "/opt/engine/rank", "sha256": "…", "backends": ["hip"], "model_path": "/models/m.gguf",
     "argv_form": "pipeline-tail-v1", "describe": {"file": "tail-describe.json"}},
    {"id": "serve", "binary": "/opt/engine/serve", "sha256": "…", "backends": ["hip"], "model_path": "/models/m.gguf",
     "argv_form": "group-serve-v1", "describe": {"unchecked": "no describe in this build"}}
  ],
  "front": {"set": "gpu", "engine": "front", "http_port": 18080},
  "tail": {"set": "pair", "engine": "tail", "group_mode": "ep2-tp", "transport": "usb4auto", "stage_port": 18400,
           "ranks": ["node1-gpu", "node2-gpu"]},
  "fallback": {"set": "pair", "engine": "serve", "group_mode": "ep2-tp", "transport": "usb4auto",
               "http_port": 18081, "start": "cold", "ranks": ["node1-gpu", "node2-gpu"]},
  "router": {"listen": {"address": "127.0.0.1", "port": 18000}, "max_inflight": 4},
  "timeouts": {"tail_ready_s": 900, "front_ready_s": 900, "fallback_ready_s": 900, "drain_s": 120}
}
```

- `nodes[].id` are the hardware descriptor's node ids; a process runs on the node of its device.
- `describe`: exactly one of `file` (recorded `--describe-json` output; its sha256 must be the
  approved one), `probe` (run `--describe-json` on the node after checking the binary's sha256)
  or `unchecked` (an explicit reason, recorded in every manifest; no capability check).
- `env`: up to 32 upper-case names; `PATH`, `LD_PRELOAD`, `LD_AUDIT`, `PYTHONPATH`, … are refused.
- `launch_prefix`: an interpreter before the binary (test fixtures only).
- `fallback.start`: `cold` (started after the stages stop: their memory is free) or `standby`
  (started with the pipeline; needs the plan's `co_resident.fits`, which counts one engine
  instance holding both placements).
- Optional `front.backend` substitutes the FRONT backend (`hip`/`host` instead of `cuda`) for
  tests; it is reported as a warning (not the planned numerics or speed).
- Timeouts (seconds unless noted): `describe_s`, `tail_ready_s`, `front_ready_s`,
  `fallback_ready_s`, `health_interval_s`, `health_timeout_s`, `failures_to_down` (count),
  `stop_s`, `drain_s`, `export_s`, `import_s`, `max_state_bytes` (bytes).

## Manifest (`tensormeld/deployment-manifest-v1`)

One per process (`tail-r0`, `tail-r1`, `front`, `fallback-r0`, `fallback-r1`): deployment id
and epoch, plan sha256/candidate/model/hardware fingerprints, process (position, role, rank,
world), node (address, launcher, work dir), set and device, engine (binary, approved sha256,
backend, argv form, group mode, transport, description record), model id and path, layer
range, `has_embedding`/`has_head`, `parts`/`remote_parts`/`serves_parts`, endpoints, secret
file locations (never values), environment, launch prefix, argv, readiness rule and
`manifest_sha256`. `validate_manifest` checks the closed key set, the hash, endpoint
addresses and that the argv is exactly the form applied to the manifest's fields.

Argv forms (engine contract, pipeline v1):

| form | command line |
|---|---|
| `pipeline-tail-v1` | `BIN MODEL --rank R --transport T --role pipeline-stage --first-layer K --layers L --stage-listen H:P --stage-token-file F [--max-positions N]` |
| `pipeline-front-v1` | `BIN MODEL --backend cuda\|hip\|host --layers K --stage-connect H:P --stage-token-file F --http 127.0.0.1:P [--max-positions N]` |
| `group-serve-v1` | `BIN MODEL --rank R --transport T --layers L --http-bind H --http-port P [--http-api-key-file F] [--context N]` |

## Run-time behaviour

1. Pre-flight: sha256 of every binary on its node (`sha256sum` over SSH).
2. Secrets: one session token per deployment (32 random bytes, hex) in each stage process's
   token file; a fallback API key when its listener is not loopback. Files are `0600` in a
   `0700` per-deployment directory and are removed at stop.
3. TAIL ranks start; rank 0 must print `READY {json}` (role/layers checked when present),
   the other ranks must be alive. Then the FRONT: `READY`, then `GET /health` = 200 with
   `stage: connected`. A `standby` fallback starts next. Then the router, if configured,
   with the FRONT routable and the fallback registered but not admitting.
4. Watch: every `health_interval_s`, process liveness (exit, `POISONED`, SSH channel loss)
   and FRONT health (`failures_to_down` consecutive failures).
5. Retirement (`/admin/pipeline/retire`, `--retire-file`, or `PipelineController.retire`):
   router hold + FRONT not admitting → wait for its in-flight requests (`drain_s`) →
   FRONT `POST /admin/drain` → `POST /admin/export-state` → stop FRONT and TAIL → fallback
   ready (cold start or standby) → `POST /admin/import-state` → fallback routable, hold
   released. Requests that arrive meanwhile wait in the router queue (bounded by
   `queue_timeout_s`). The record lists every step with its time, state bytes and sha256.
6. A dead stage: `--on-failure stop` (default) stops everything and reports the reason;
   `--on-failure fallback` starts the fallback without the lost state (`state_lost: true`).
7. Stop: router, FRONT, TAIL, fallback (graceful, then kill after `stop_s`), then secrets.

Remote processes run under a fixed `sh` wrapper: `env -i PATH=/usr/local/bin:/usr/bin:/bin
VARS… BIN ARGS </dev/null &`, `TM-PID`/`TM-EXIT` lines, and a watcher that terminates the
program when the SSH channel's stdin closes (controller exit or stop).

## State handoff (proposed to the engine: `tensormeld/state-handoff-v1`)

| route | on | request | answer |
|---|---|---|---|
| `POST /admin/drain` | FRONT | — | 200 `{"status":"draining"}`; running requests finish, new ones get 503 |
| `POST /admin/export-state` | FRONT | — | 200 `application/octet-stream`; headers `X-State-Format`, `X-State-Sha256`; the FRONT collects the TAIL's layers over its wire |
| `POST /admin/import-state` | fallback rank 0 | body = exported bytes; headers `X-State-Format`, `X-State-Sha256` | 200 `{"imported":true,"sha256":…,"sessions":N}` |

TensorMeld treats the state as opaque bytes, bounded by `max_state_bytes`, and checks the
sha256 on both sides.

## Tests

`tests/test_pipeline_deploy.py` (fake engine `tests/fixtures/fake_pipeline_engine.py`, fake
SSH client `tests/fixtures/fake_ssh.py`):

- plan layout: stages, ranges, remote/served lookups, part coverage;
- manifests follow the contract argv; tampering (argv, keys, hash) is rejected;
- binding rejections: wrong set, missing group mode, wrong transport, ranks not covering the
  set, wrong argv form, unknown backend, non-loopback FRONT HTTP, remote FRONT, non-pipeline
  candidate, other hardware fingerprint, forbidden environment, a missing capability in the
  engine's own description, a description of another binary;
- CLI dry run and a short real run;
- loopback end to end: plan → manifests → launch → health → HTTP through the router →
  deterministic tokens → stop (secrets removed);
- a wrong session token never connects; a TAIL death fails clearly (router 503);
- retirement with state (cold fallback): a request issued during the switchover waits and is
  served by the fallback, and tokens continue exactly where the pipeline stopped;
  the admin route with a standby fallback; the retire file without a router;
- FRONT death with `--on-failure fallback`: the fallback starts empty (`state_lost`);
- the SSH wrapper through a local `sh`: secret file creation (no overwrite), remote sha256,
  `READY`, environment, dead-man stop and cleanup.

## Not implemented

- a real engine run (the engine's pipeline roles and state routes are in progress);
- periodic state checkpoints for an abrupt loss of the FRONT device;
- N-stage pipelines, prefill/decode pairs and part splits;
- restart or re-placement policies beyond the single fallback.
