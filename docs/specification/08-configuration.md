# 08 — Configurable nodes, policy and versioned data

Status: normative v2 contract. The 0.2.0a2 parser, selector and synthetic planner
implement a documented subset; consult the implementation ledger for the remaining
runtime/agent requirements. Canonical configurations use `tensormeld/v2`.

## Identity and data ownership

| Entity | Stable identity and meaning |
|---|---|
| Installation | Trust/configuration owner and registry namespace |
| Node | Enrolled agent/OS identity; contains zero or more eligible devices |
| Device | Physical CPU/GPU identity; backend views are aliases, not extra capacity |
| Worker | Approved artifact/process, owning one or more devices as declared |
| Physical pool | Unique RAM/VRAM allocation domain; multiple views may alias it |
| Interface/physical link | Host interface and physical path/resource sharing domain |
| Profile | Evidence-bound compute/transfer/allocation observation |
| Model manifest | Exact checkpoint, tokenizer, state and transformation contract |
| Plan/session/request | Immutable placement snapshot, loaded resources and stream lifecycle |

The registry stores nodes as a variable-length collection. No `node1`, `node2`,
`main_gpu=0`, or array position is a persistent identity. Node display names and
addresses are editable labels; trust uses keys and UUID-like IDs. Device binding
uses platform-stable identity; uncertain replacement requires re-qualification.

## Configuration layers and precedence

Security and physical limits are intersected, never overridden by a convenience
profile. The effective resource ceiling is the minimum of local host policy,
installation policy, approved workload policy, live adapter/OS limits and existing
reservations. A CLI/request override may choose a stricter limit; it cannot enlarge
another owner's limit. Invalid combinations fail rather than silently clipping
requested semantic guarantees.

Persisted defaults < named workload profile < explicitly approved request overrides
for ordinary non-security settings. Installation and node-local ceilings remain
constraints across all layers. The planner outputs the resolved configuration and
its hash. Read-only hardware observations never become editable config facts.

Schema/config/protocol/adapter versions are independent. A new required field or
feature needs negotiated support. Unknown required fields, duplicate IDs, invalid
units and incompatible versions fail clearly. Migration is explicit and reversible;
an imported v1 scenario remains synthetic/advisory and cannot authorize workers.

## Membership and compute selection

`nodes[]` references enrolled nodes, desired enabled state and allowed roles.
A node may offer a GPU, CPU (opt-in computation), storage, control, or combinations.
Availability is an observation, not inferred from `enabled=true`.

`selection.mode` is `auto` or `manual`.

- Auto: choose a feasible subset within `allowed_nodes`, `required_nodes`,
  `excluded_nodes`, minimum/maximum compute-node counts, and device constraints.
- Manual: `selected_nodes` is the exact required COMPUTE owner set. Each must own
  nonempty useful compute work. For UI/relay-only participation use role assignment.

Lists have explicit semantics: absent `allowed_nodes` means all eligible enrolled
nodes; an empty list means none. Absent or empty `required_nodes`/`excluded_nodes`
means no such constraint. `required_nodes` is a subset of allowed and disjoint from
excluded. Unknown IDs, an excluded required device or impossible minima are errors.
Required devices imply their host is a required compute node. Disabled/offline/
draining required owners make a new plan unavailable, not automatically optional.

`min_compute_nodes` and `min_compute_devices` default to 1. The default maximum
for each is `auto`. `max_compute_nodes` is a positive integer or
`auto`; auto resolves against eligible inventory and declared implementation/adapter
limits. Node count is NOT GPU count: equivalent `min_compute_devices` and
`max_compute_devices` govern compute devices. A relay-only/controller-only node
consumes resources but is not a compute owner. The planner reports both counts.

The execution coordinator is selected from allowed NODE identities with appropriate
backend capabilities. It may be a CPU-only node/process. It is not implicitly
restricted to compute owners; all of its host/runtime allocations still count.
`participation.local_gpu` can be auto, required or excluded. Companion-only mode
requires it excluded; local-only limits compute ownership to the control host.
Conflicting mode and membership constraints fail before search.

## Limits: configurable does not mean unbounded

A build advertises `registry_limits`, `planning_limits`, `adapter_limits` and
`qualified_envelopes`, including model/encoding/platform constraints. Limits cover
nodes, devices, graph size, message/frame bytes, cache bytes and concurrent probes.
Operators may lower limits; increases cannot bypass hard safety limits. No silent
truncation is permitted. Exceeding a search budget is not physical infeasibility.
Larger registry capacity is not proof that the adapter runs a single model on it.

## Resource and availability policy

Host policies set allocatable pool caps/reserves, CPU thread limits, storage/cache
quotas, number of workers and participation state. Pool headroom may be an absolute
amount or a percentage; when both apply use the stricter reservation, not an
ambiguous sum. Units are integers in bytes internally with explicitly typed MiB/GiB
UI input. Never equate decimal GB, GiB, installed memory and usable budget.

An absolute allocation cap and free-memory headroom are different controls. Dynamic
budget calculation is described in spec 09. The local owner may disable sharing or
request drain. Availability windows are optional P1 policy and affect NEW admissions;
they do not grant transparent migration. Local foreground-preservation profiles
must not claim a guaranteed percentage of GPU compute when the backend cannot
provide that isolation. CPU thread limits are not whole-system CPU quotas.

## Model/workload and placement controls

The `execution_mode` setting is `auto`, `local_only`, `companion_only` or
`distributed`. Auto compares the compatible families and can select a single owner.
Distributed means execution across multiple devices as declared by the adapter; it
does not itself require multiple physical nodes. To require inter-host computation,
set `min_compute_nodes` to at least 2. To require same-host multi-GPU, set one allowed
compute node and `min_compute_devices` to at least 2. Strategy and mode must agree.

Profiles fix exact model manifest and task, requested context (prompt plus generated
state, including template tokens), output bound, concurrency, state dtype, approved
precision variants and sampler limits. Automatic adjustments are confined to
backend-safe performance knobs within those bounds. Checkpoint switching, reduced
context and KV/weight encoding changes require explicit approval and a new profile.

Placement strategy is auto or an explicit supported family: whole-block baseline,
then optional expert/tensor/phase strategies. Advanced users may pin legal units or
owners, but the adapter must validate every constraint. No hard-coded per-family
flags are normative. A rejected strategy returns the missing capability.

Selection, privacy and precision policy are enforced BEFORE cost ranking. A fast
node outside the model's allowed trust set is never a candidate. A manual plan
cannot bypass memory admission or backend compatibility.

## Network and diagnostics controls

Allow or pin interfaces; choose automatic/measured paths or explicit approved routes;
limit bulk bandwidth, probe concurrency and staging; choose one link by default.
Multirail is an explicit capability/policy, not `bandwidth *= cable_count`.
A directional profile includes backend/copy/security overhead and payload class.

Profiling levels are read-only inventory, lightweight connectivity, host-path probe,
GPU-path probe and actual model qualification. Intrusive probes require approval and
are bounded/interruptible. A busy installation defers probes rather than contaminating
latency evidence or occupying every node in an all-pairs benchmark.

## Configuration outcomes

Expose config revision, resolved constraints, participating/omitted identities,
limit sources, missing data and feature availability. Suggested error classes:
INVALID_CONFIG, UNKNOWN_NODE, POLICY_CONFLICT, REQUIRED_NODE_UNAVAILABLE,
ADAPTER_LIMIT_EXCEEDED, FEATURE_UNSUPPORTED, PROFILE_MISSING,
MEMORY_BUDGET_EXCEEDED, UNSPLITTABLE_UNIT_TOO_LARGE, SEARCH_INCOMPLETE.
Model/request errors additionally follow spec 10.
