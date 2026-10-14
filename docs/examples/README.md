# Configuration examples

These JSON files are accepted by the current v2 control-plane parser. They illustrate
policy and topology structure only; their capacities and names are examples, not measured
hardware profiles or performance claims.

- `12gb-pc-one-companion.config.json`: a desktop with one 12 GB discrete GPU plus one
  large unified-memory companion.
- `companion-only.config.json`: the desktop remains the entrypoint but its GPU is excluded
  from compute.
- `four-nodes-multi-gpu.config.json`: four nodes and five accelerators, with a policy that
  allows at most three compute nodes and four compute devices for a request.

The parser validates references and policy invariants. The `select` command resolves
legal candidates. Neither action proves model compatibility or execution performance.

## Try an example

After the [developer setup](../../README.md#developer-setup), run from the repository root:

```bash
python -m tensormeld validate-config docs/examples/12gb-pc-one-companion.config.json
python -m tensormeld select docs/examples/12gb-pc-one-companion.config.json
python -m tensormeld plan-v2 docs/examples/12gb-pc-one-companion.config.json examples/v2/12gb-dense.planning.json --top-k 3
```

An installation configuration describes policy/topology; the separate planning input
supplies model-unit demands and synthetic costs. Keep their identities and selected
profile consistent. See the [v2 planner contract](../implementation/planner-v2.md) for
outcomes, search bounds and exit codes, and [configuration specification](../specification/08-configuration.md)
for absent-versus-empty lists and manual selection semantics.

`adapter-capabilities.example.json` illustrates an adapter declaration, not a measured
backend capability or authorization to execute. Do not substitute example declarations
for native evidence, live resource observations or admission.
