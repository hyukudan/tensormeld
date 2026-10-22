"""Fake engine for tests: answers --describe-json with a small parts descriptor."""
import hashlib
import json
import sys

ROLE = sys.argv[2] if len(sys.argv) > 2 else "amd"

if "--describe-json" not in sys.argv:
    sys.exit(2)
layers = 2
parts = [{"id": "embed", "kind": "embedding", "role": "lookup", "layer": -1, "weight_bytes": 1000,
          "decode": {"read_bytes": 10}, "io": {"out_bytes_per_row": 16}, "requires": ["toy.lookup"]}]
for L in range(layers):
    parts.append({"id": f"L{L}.attn", "kind": "attention", "role": "anchor", "layer": L, "weight_bytes": 10**6,
                  "decode": {"read_bytes": 10**6}, "requires": ["toy.attn"]})
    parts.append({"id": f"L{L}.experts", "kind": "routed-experts", "role": "branch", "layer": L,
                  "weight_bytes": 10**8, "decode": {"read_bytes": 10**6},
                  "io": {"in_bytes_per_row": 64, "out_bytes_per_row": 64}, "requires": ["toy.moe"]})
parts.append({"id": "head", "kind": "lm-head", "role": "head", "layer": layers, "weight_bytes": 1000,
              "requires": ["toy.lookup"]})
caps = ["toy.attn", "toy.lookup", "toy.moe"] if ROLE == "amd" else ["toy.attn", "toy.lookup"]
print(json.dumps({
    "schema": "tensormeld/engine-describe-v1",
    "engine": {"name": "fake", "revision": "r1", "executable_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               "backend": "hip" if ROLE == "amd" else "cuda"},
    "capabilities": caps,
    "group_modes": [{"id": "ep2", "world": 2, "capabilities": ["toy.moe"], "transports": ["usb4"]}],
    "roles": ["whole", "prefill", "decode", "pipeline-stage", "parts-worker"],
    "state": {"export": True, "import": True, "format": "toy-state-v1", "transports": ["tcp"], "exact": True},
    "parts_rpc": {"transports": ["tcp"], "max_rows": 64},
    "models": [{"schema": "tensormeld/model-parts-v1", "model": {"id": "toy"}, "provenance": "engine-describe",
                "layers": layers, "stream_bytes_per_row": 64, "parts": parts}],
}))
