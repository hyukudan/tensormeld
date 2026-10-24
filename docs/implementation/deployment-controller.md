# Deployment controller: phases 2-4 (capability notes)

Status as of 2026-10-24. This page describes what our deployment controller does today and how it was accepted
on real hardware. The controller is a separate, not yet public tool that launches the native engines on the
Strix Halo nodes and the RTX PRO workstation and puts TensorMeld's replica router in front of them. Its code
depends on unpublished engine builds and on our machines' GPU scheduling, so it is not part of this repository.
What is generic lives here: the replica router, the prefill route and the per-request RTX prefill offload
([replica-router.md](replica-router.md), [hetero-placement.md](hetero-placement.md)).

Hardware names: **node A** and **node B** are the two Strix Halo machines (128 GB unified memory each) joined by
two USB4 cables; the **workstation** is the Windows PC with the RTX PRO 6000 (96 GB), on 1 GbE to the nodes.

## Placement and routing

One request runs on one decode placement. More load adds replicas first; a model is split across machines only
when no further copy fits, and the choice is made from measured speed.

| Model family | Decode placement | Prefill placement |
|---|---|---|
| Qwen3.8-Flash-Next | two whole-model replicas, one per node, behind the router (least-loaded, affinity yields to load) | node prefill, or the RTX PRO prefill service with the state streamed to the replica the router holds (`prefill_offload`, from 1K tokens) |
| DeepSeek V4.1 Flash | one tensor-parallel pair (TP2 over USB4); rank 0 is the router's single replica | pair prefill, or the RTX PRO prefill service with the state imported by rank 0 (`prefill.policy auto` with measured curves) |
| MiniMax M3.1 Flash (bring-up) | one TP2 pair; rank 0 binds loopback only, so the router runs on rank 0's machine and is the only authenticated entry | pair prefill (RTX PRO hand-off under qualification) |

The RTX PRO prefill service is resident: it keeps its weights loaded, binds loopback only and is pinned to the
RTX PRO by its PCI UUID. It runs inside a Windows Job Object (killed with its parent) under a watchdog that stops it,
and only it, when free commit falls below a floor, when its VRAM exceeds the configured cap, when it becomes visible
on any other GPU, or when its reservation window ends. After it reports ready, `/health` is checked every 10 s and
three failures stop it. Stopping a profile stops the router first, then the service, then the engines; a service that
fails to stop never keeps the engines running.

## Acceptance gates

Every engine family goes through the same gates before the controller may launch it as production.

1. **Golden render.** The command line and environment the controller renders for each rank or replica are compared
   byte for byte with the engine's documented production line for that build (one golden file per preset, checked by
   unit tests). A diagnostic switch or a non-default tuning knob marks the launch `production: false`.
2. **Token gate.** N greedy prompts produce identical token ids through the controller's launch and through a hand
   launch of the same binary. First pass (DeepSeek pair): 8 of 8 prompts, 290 tokens, identical.
3. **Negative tests**, all refused before any GPU work unless stated:
   - nodes on different releases: refused by the controller, and, when forced, by the engine's own rank hello
     (`the engine executables differ (SHA-256)`), so neither rank becomes ready;
   - a same-size wrong model file: caught by the asset manifest (size, mtime, inode and SHA-256, compared across the
     pair's ranks), first as a stale entry, then as `manifest sha256 mismatch between pair ranks` after re-hashing;
   - a wrong USB4 layout: the start is aborted one second after the GPU lease is acquired, before any engine starts,
     with no process left behind.
4. **Interleaved A/B/A serve check.** One GPU lease holds three arms in order: A = hand launch, B = the controller's
   launch with its clean environment, A again. B must stay within 3 % of mean(A) in every cell, otherwise the
   difference is investigated. Measured on the DeepSeek pair (8 fixed prompts, greedy, 192 tokens each, every request
   complete):

   | tok/s | A1 | B | A2 | B vs mean(A) |
   |---|---:|---:|---:|---:|
   | C1 aggregate | 33.0 | 34.7 | 36.1 | +0.3 % |
   | C1 decode per request | 37.8 | 39.1 | 40.4 | 0.0 % |
   | C4 aggregate | 56.9 | 57.8 | 57.5 | +1.0 % |
   | C4 decode per request | 21.6 | 21.7 | 21.0 | +2.2 % |

   The controller's launch was not slower in any cell; the A1-to-A2 spread (9 % on C1) is larger than any B
   difference. The full path (supervisor, layout check, readiness, owner watch, router, then stop) gave C1 36.5 and
   C4 55.9 tok/s aggregate. These are serve-path smoke numbers on the public engine release of 2026-10-23, measured
   to show that orchestration adds no measurable overhead; the engine's own figures are in the README.
5. **RTX prefill service gate.** The service launched by the controller reproduces the exact reference states at
   512 / 2K / 8K / 32K tokens (state SHA-256, next token, logits digest, state size), is visible only on the RTX PRO
   in the driver's per-process list, and gives the same states as the same executable launched by hand.

## Supervisor lifecycle

- `up` checks, before asking for any GPU lease: configuration, the deployed node tool against the controller's
  checkout, every asset against the manifest, and that every node runs the same release.
- On the nodes, a supervisor starts each engine in its own process group with a clean environment (`PATH`, `HOME`,
  the release's loader path and the identity variables), checks the USB4 layout for pair families, and waits for the
  family's readiness signal (stdout/stderr line or JSON event, then `/health`).
- The second rank of a pair is watched by a dead-man peer: if the owner's lease ends, the peer stops.
- `down` drains (an HTTP drain where the engine has one), stops leader-first where the family needs it, and ends the
  lease. Every lease has a deadline; a queued lease can be cancelled, and `up` takes a queue timeout.

## Release manifests

- A release is built on each node from pinned component commits with a recorded build recipe; its id is a hash of
  the manifest (component commits, recipe, binary SHA-256). Releases are staged and published once, never rebuilt in
  place; nodes in one profile must run the same release id.
- Capability checks are static: every flag the controller will pass is looked up as a string in the built binary,
  which is never executed for the check.
- A release built from a commit that is not public yet is marked unpublished in its manifest.
- The controller's own release archives are reproducible (two builds give the same SHA-256 per file) and are verified
  before install; rollback re-activates an earlier release.

## Diagnostics, doctor and host keys

- **Diagnostic runs.** Named, observation-only engine switches per build (each read in the engine source: clocks and
  log lines only, no change to what is computed or which code path runs). They only add environment variables, never
  replace a production one, and mark the run `production: false` everywhere it is recorded. Numbers from a diagnostic
  run are never published as production measurements.
- **Doctor.** A read-only check of a node (Python, git, CMake, Ninja, ROCm and compiler, `gfx1151` visibility, disk,
  USB4 driver and layout for pair families) that lists every command it ran. It bootstraps its own small helper, so it
  works on a fresh machine.
- **Host keys.** `hosts scan` and `hosts pin` record each node's SSH host key; the fingerprint is checked against the
  machine before pinning. On Windows the stock `ssh-keyscan` cannot negotiate the default post-quantum key exchange of
  current OpenSSH servers, so the scan retries with another client; `ssh` and `scp` keep using the system client.

## Metrics

A Prometheus exporter re-exports what the router and the engines already publish (router `/api/status` and each
replica's `/metrics` and status), with no new measurements. Missing or invalid values are left out, never written as
0. Each source reports its own up flag and scrape time; one bad source marks only that source down. Every fetch is
bounded in time and size, the whole scrape has one 8 s deadline (below Prometheus' default timeout), at most 2 scrapes
run at once (the next gets 503), and the listener is loopback-only. Engines bound to loopback on another machine are
never probed from the control machine.

## Long-running services (`systemd --user`)

Optional and off by default, for machines that are not shared with other GPU work: the same node supervisor runs as
a `systemd --user` unit without a deadline. It refuses root, refuses any machine that shows signs of a shared GPU
scheduler, and never shares an install root with a GPU lease (a unit and a lease cannot both be live). It is limited
to the Qwen replicas today; the pairs need a coordinated peer on the second node and stay lease-only. The unit uses
`KillMode=mixed`, so the supervisor drains the engine before the final kill.

## Per-family adapters

Each engine family has one adapter; presets stay pure data (argv, environment, assets, pins).

| | Qwen replicas | DeepSeek pair | MiniMax pair |
|---|---|---|---|
| topology | one process per node | TP2, rank 0 + rank 1 | TP2, rank 0 + rank 1 |
| USB4 layout check after acquire | no | both nodes | both nodes |
| readiness | `listening` line on stderr | JSON `READY` event | rank 0 `READY` line; rank 1 load line |
| health | `/health` per replica | `/health` on rank 0 | `/health` on rank 0, checked on rank 0's machine |
| stop | SIGTERM | HTTP drain on rank 0, then SIGTERM | rank 0 first, then rank 1 |
| bind and auth | private address + key file | private address + key file | loopback, no key; router on rank 0's machine with its own key |
| router | every replica | rank 0 as the single replica | rank 0 as the single replica |

## Not public, and why

The node tool, the GPU lease integration, the presets (they pin unpublished engine builds), the cluster configuration
schema, key handling and the web portal stay private: they encode our machines and engine builds rather than generic
orchestration. Generic pieces move into TensorMeld when they are needed here; the replica router's RTX prefill offload
is the first.
