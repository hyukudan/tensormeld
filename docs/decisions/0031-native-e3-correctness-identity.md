# ADR-0031 — Native E3 binds correctness to plan, placement and runtime identity

Status: **accepted**  
Date: 2026-10-03

## Context

The pre-E3 llama.cpp trial can prove that a pinned local process ran with an exact model
and placement, but process success and stdout presence are not correctness evidence.

The historical QualificationEvidence v1 contract binds adapter, worker, model, config,
device set and workload, but it does not bind the exact planner candidate, placement
translation, native trial specification, correctness reference or stable runtime identity.
Reusing such evidence for execution could therefore admit a different placement or a
changed runtime under otherwise identical high-level inputs.

## Decision

TensorMeld introduces `tensormeld/qualification-evidence-v2` while retaining v1 parsing
compatibility.

V2 adds:
- candidate plan SHA-256;
- placement SHA-256;
- trial-spec SHA-256;
- correctness-contract SHA-256;
- one stable runtime-identity SHA-256 per device.

The llama.cpp E3 evaluator also introduces
`tensormeld/llamacpp-e3-reference-v1`.

The reference contract binds:
- exact native-trial spec;
- exact llama-cli artifact;
- exact ModelManifest;
- exact placement translation;
- exact workload;
- exact expected raw stdout SHA-256.

E3 evaluation succeeds only when:
1. the trial is marked `native-subprocess`;
2. all retained trial/spec/model/config/plan/placement identities match;
3. exit code is zero and stdout is non-empty;
4. stdout SHA-256 exactly matches the approved reference;
5. runtime identities cover exactly the placement devices;
6. every runtime identity names the same worker artifact used by the trial.

The evaluator then emits QualificationEvidence v2 at E3.

AcceptedExecutionBundle now requires E3 v2 and compares the evidence's candidate-plan and
runtime-identity fingerprints against the exact current candidate and backend-readiness
results.

## Consequences

- E3 cannot be silently reused after a planner or runtime-identity change.
- Historical v1 evidence remains parseable but is insufficient for executable E3
  authorization.
- Exact raw stdout comparison is intentionally strict; later reference-contract versions
  may define other canonical correctness metrics explicitly.
- Hosted fixture tests validate semantics only. A real E3 claim requires a real
  target-host native trial and retained runtime identities.
