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
