# Contributor instructions

Read docs/README.md, IMPLEMENTATION_STATUS.md and the relevant specification before
changing code. Normative spec is 0.2.0; runtime is 0.2.0a2. The v2 configuration parser and
policy-only candidate resolver are implemented, but distributed inference is not.
Do not treat a requirement or configuration example as execution qualification.

Use English for maintained code, tests and documentation. Keep research provenance
and third-party benchmark notes outside normative specs. Pin external adapters and
record licenses/security/provenance before adopting code. No model weights, personal
inventories, prompts, keys, caches or credentials belong in Git.

The product is NOT restricted to GLM/DeepSeek, sparse models, 96 GB GPUs, a GPU in
the control PC, or two companions. Low-VRAM dense workloads and large MoE workloads
are first-class. Strix Halo is the first companion profile, not the core identity.
Support variable nodes and multiple devices per node with explicit tested limits.
Do not merely raise the v1 three-device cap: implement spec 08 and M0.2 first.

Keep v1 source fixtures and spec 07 as an analytical oracle until explicitly migrated.
Do not interpret synthetic costs as measurements, a GPU budget cap as real low-VRAM
hardware, or hosted tests as GPU qualification. Report implemented/tested/qualified
separately. A new configuration cannot mark itself measured/qualified.

Preserve physical memory aliasing, multi-pool demand, actual routes, graph order,
cyclic token feedback, precise layout/state identities and explicit commit semantics.
A node/cable is not inherently a speedup. Local/companion-only are legitimate plans.
Never silently relax requested model/context/precision or owner resource limits.

No unauthenticated LAN worker; no peer-supplied executable/shell; ordinary-user agents
and bounded messages. No unapproved driver/firmware/network/clock changes. Drain or
fail clearly on state loss; do not claim automatic migration or background work.
Use bounded queues, local atomic leases and cancellation-safe buffers.

Implementation order: finish M0.2 contracts/tests and bounded v2 planning, then M1
secure agents, M2 evidence/model manifests, M3 first genuine split, M4 UX, then optional
M5/M6 optimizations.
Run available tests and update the ledger honestly. Hardware/paid CI and large model
downloads require explicit configuration. Do not change other private projects.

TensorMeld is the canonical package/CLI name. Prefer documented open-source adapters
over rebuilding mature infrastructure. Keep third_party/registry.json and notices
current. Synthetic planning is not an executable manifest. A skipped optional
upstream integration is never a passed test. Preserve old commits/history documents.
