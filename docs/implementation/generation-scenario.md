# Serial generation scenario evaluator

`tensormeld.generation_scenario.evaluate_generation_scenario` costs one declared
ordered legal-unit placement for concurrency one. It consumes current configuration,
model/runtime/movability contracts, generation phase evidence v2 and directional path
evidence. Raw evidence is parsed against these identities on each invocation.

Prefill and decode use their respective unit timings and boundary payload sizes.
The selected sampler must be a compute owner with explicit sampling evidence. The
last owner sends logits to the sampler, which returns the selected token to the first
owner. Local transfers have zero link cost. Additional token consumers, overlapping
stages and concurrent requests require a different graph contract.

Callers supply nonnegative integer additive control costs for both phases. The
evaluator cannot establish that separately measured components are independent,
non-overlapping upper bounds. Composed values are conditional scenario bounds, not
measured end-to-end percentiles. Prefill includes return to the next-token start;
it does not represent service time to first token.

Missing directional paths or payload buckets produce `EVIDENCE_INCOMPLETE` with
unknown totals for affected phases. They are never extrapolated or treated as free.
Illegal cuts, excluded owners, participation-policy violations, identity mismatches
and invalid costs raise `ValidationError`.

The deterministic `tensormeld/generation-scenario-cost-v1` result records input
fingerprints, workload, ownership, sampler location, component costs, transfer roles
and missing evidence. It remains advisory: no placement search, physical-pool memory
admission, execution authorization, throughput claim or hardware qualification is
performed. Fixture provenance remains fixture provenance.

Portable tests cover phase payload separation, asymmetric feedback, sampler location,
local movement, incomplete evidence, illegal cuts, identity validation and bounded
cost composition. Native measurements and full generation-loop qualification remain
required before this calculation can inform an executable plan.
