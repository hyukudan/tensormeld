# Serving mode policy and shared accelerator ownership

`serving_policy.py` implements opt-in Python control-plane contracts. It does not
launch a split worker, change a running model, or qualify an execution path. All
feature flags default off. Existing router configuration remains compatible.

## Mode selection

`recommend(load, domain, profiles, policy, joint=None)` accepts an explicit load
of 1..16 requests and domain `code`, `en` or `es`. Domain classification belongs
to the caller; the router does not inspect or retain prompts to infer it.

| Load | Code | Prose |
| --- | --- | --- |
| 1 | Split decode with DSpark | Split decode with adaptive DSpark |
| 2 | Pair with batch DSpark | Pair with batch DSpark |
| 3–4 | Pair with batch DSpark | Pair AR; batch DSpark only when its exact profile exceeds AR by the configured selective gain |
| 5–16 | Pair batch | Pair batch |

Split availability is explicit; its fallback is pair DSpark. Batch DSpark has a
separate availability flag, with AR fallback. Enabling 16 rows applies to batch
modes. An eight-row plan at larger loads represents alternating bounded groups,
not an unsupported larger forward. The 1..16 range is this policy's tested bound,
not a core registry/device limit or execution qualification.

`ModeController.observe` retains the selected mode until a safe boundary has
observed the same candidate for `stable_observations` and `min_dwell_s` has
elapsed. The initial mode also requires a safe boundary. Accepted boundaries are
`between_requests` and `window_boundary`; the latter means every affected worker
has completed commit/rollback. Unsafe observations cannot earn debounce credit.
The caller supplies monotonic seconds and serializes distinct boundary events.
It must hold new admissions while `transition_pending` is true and preserve existing
requests' model, context, precision, state identity and token feedback. These
decisions never authorize state migration or an in-flight mode change.

## Joint prefill/decode cost

`choose_joint` extends the prefill-placement J cost with a bounded remaining
decode horizon. It compares:

- FRONT: local prefill J + N × remaining output tokens / split tok/s;
- prefill: service prefill J + N × remaining output tokens / pair tok/s + switch cost.

The service queue, restore, handoff and import pauses retain `route_cost.py`
semantics. Prefill must win by the configured absolute/fractional margin. Supply
the complete switch cost, including unload/load, drain and state handoff as
applicable. A zero switch cost is only an explicit simulation assumption.
At N >= 2 the shared accelerator is reserved for prefill, and split is excluded.
At N = 1 pass the resulting `JointChoice` to `recommend` or `observe`; a prefill
win changes the requested mode to pair DSpark. No joint input means decode-only
mode selection, as in the simulator.

## Reservation and service integration

`JointReservation(capacity_bytes, enabled=True)` is an atomic process-local
authority for one physical accelerator domain. FRONT and prefill are mutually
exclusive even when the sum of their declared bytes would fit. The caller must
provide owner-approved byte limits and complete allocation requirements, including
workspace and transients. This reservation supplements host memory admission;
it does not replace it or infer free capacity from telemetry.

A supervisor must acquire a lease **before loading**, drain/commit the current
owner before switching, and acknowledge actual buffer release through `release`.
Cancellation, elapsed time, a completed prefill and a safe boundary alone never
free resident memory. A task pins the resident lease with `use(owner)`; release
fails while a task uses it. Stale/foreign lease objects cannot free a newer lease.
State loss must fail/drain dependent requests; there is no automatic recovery.

Pass the same authority as `ReplicaRouter(config, joint_reservation=resource)`.
Service offloads then require a live prefill-owned lease and pin it through the
service queue and handoff. A FRONT-owned, absent or disabled lease causes an
explicit `fallback:accelerator_reserved` local prefill. Completing the HTTP call
retains the service's residency lease. The supervisor must also use the authority
for FRONT tasks and must be the only loader of these workers. Cross-process or
unmanaged accelerator users are outside this authority's guarantees. The optional
Python argument is deliberately not a configuration claim of available execution.

## Recorded profiles and dry run

Recorded rates are exact `(mode, load, domain, rows)` lookups. Unknown entries stay
unknown; no interpolation or extrapolation is performed. The bounded
`tensormeld/serving-rates-v1` input has `rates`, each containing `mode`, `load`,
`domain`, `rows`, `per_user_tps`, `provenance` and `source`. Provenance can be
`measured`, `replay`, `estimate` or `synthetic`. A recording's label is an input
claim only; every mode decision is `executable=False`. Private source records,
deployment inventories and external measurement tables stay outside Git.

Run without a GPU:

```text
python -m tensormeld.serving_simulator --profiles PATH --enable-split --enable-batch-dspark
python -m tensormeld.serving_simulator --profiles PATH --enable-split --enable-batch-dspark --rows16
```

The simulator prints selected modes, required accelerator roles, expected per-user
and aggregate tok/s, and provenance at N=1/2/4/8/16. It neither reserves resources
nor includes prefill time in decode throughput. Missing profiles print `unknown`.

Portable tests cover the policy matrix, selective gains, missing evidence,
safe boundaries, dwell/debounce, J tradeoffs, exclusive ownership races, stale
release and cancellation-safe task pins. HTTP tests exercise FRONT exclusion,
prefill handoff and retention of service residency. These are control-plane tests;
native performance and split worker exactness remain separate gates.
