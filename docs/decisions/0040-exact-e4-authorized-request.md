# ADR-0040 — The first persistent request is exactly the E4-qualified request

Status: **accepted**  
Date: 2026-10-05

## Context

TensorMeld can now prove native E4 equivalence between one deterministic llama-cli trial
and one same-build llama-server completion contract, and can derive a request-authorized
server binding.

Allowing arbitrary prompts or sampler changes immediately after that gate would broaden
qualification beyond what E4 actually tested.

## Decision

The first persistent completion path executes exactly the request body fingerprinted by
the E4 spec. It does not accept caller-provided prompt, context, output length, sampling,
cache or streaming changes.

Before and after each request TensorMeld verifies:
- intact request-authorized E4 binding;
- intact E4 spec;
- exact package, bundle, base binding and server launch identities;
- ready/live managed server process;
- active launched lease;
- exact lease and runtime-manifest fingerprints.

The authorized binding retains the expected E4 output SHA-256. Because this request
contract is deterministic, every execution must reproduce that exact output hash. Drift
fails closed.

A successful request leaves the lease active because the persistent server remains alive.
Lease release remains tied to confirmed process shutdown.

The request result reports `real_model_inference=true` only when the actual execution
source is `native-server-subprocess`. Fixture/wrapper execution remains false.

## Consequences

- Qualification is not silently widened from one E4 request into a general API.
- Runtime drift is detected even after initial E4 authorization.
- Process/lease liveness is checked on every call.
- A later request-family contract must explicitly define which dimensions may vary while
  preserving qualification.
