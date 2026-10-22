"""Generate the example heterogeneous-placement descriptors in this directory.

    python examples/hetero/make_examples.py

Writes:
  deepseek-v4.1-flash.parts.json   40-layer MoE with a causal encoder-decoder split
  qwen3.8-flash-next.parts.json    48-layer hybrid linear/full-attention MoE
  lab-hardware.json                one discrete GPU host + two unified-memory nodes

Every number is an engineering estimate derived from tensor shapes and from the
engine's own receipts (bytes per token, measured tok/s); descriptors carry
``provenance: estimate`` or ``measured`` per device/link and are NOT
qualification evidence.  An engine's ``--describe-json`` is expected to emit
the parts descriptor directly; this script stands in for it until it does.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GB = 10**9
GIB = 2**30


def deepseek() -> dict:
    layers = 40
    enc = 20  # layers 0..19 see every prompt row; 20..39 see a bounded suffix (cone)
    hidden = 5120
    a8_in = 5808  # A8 codes + scales + expert ids/weights per row
    half_sum = hidden * 4
    parts = [{
        "id": "embed", "kind": "embedding", "role": "lookup", "layer": -1, "weight_bytes": int(1.32 * GB),
        "decode": {"read_bytes": hidden * 2}, "io": {"out_bytes_per_row": half_sum},
        "requires": ["ds41.lookup"],
    }]

    def rows(layer):
        if layer < enc:
            return "all"
        if layer == layers - 1:
            return "last"
        return {"max": 128 + 127 * (layers - 2 - layer)}

    for L in range(layers):
        r = rows(L)
        state = {"bytes_fixed": 136 * 528}  # per-layer FP8 sliding-window ring
        if L in (2, 8, 14):
            state["bytes_per_token"] = 178  # ratio-2 compressed KV + indexer keys
        if L == enc:
            state["bytes_per_token"] = 356  # ratio-1 KV + indexer keys shared by layers 20..39
        parts.append({
            "id": f"L{L}.attn", "kind": "attention", "role": "anchor", "layer": L, "weight_bytes": 135_000_000,
            "decode": {"read_bytes": 135_000_000, "ops_per_row": 253_000_000},
            "prefill": {"ops_per_row": 337_000_000, "rows": r}, "state": state,
            "io": {"out_bytes_per_row": half_sum}, "shardable": True, "replicated_fraction": 0.07,
            "requires": ["ds41.attention"],
        })
        parts.append({
            "id": f"L{L}.mix", "kind": "router-mix", "role": "anchor", "layer": L, "weight_bytes": 10_000_000,
            "decode": {"read_bytes": 10_000_000, "ops_per_row": 6_000_000},
            "prefill": {"ops_per_row": 6_000_000, "rows": r}, "requires": ["ds41.core"],
        })
        if L in (1, 14):
            parts.append({
                "id": f"L{L}.engram_kv", "kind": "engram-projection", "role": "anchor", "layer": L,
                "weight_bytes": 315_000_000, "decode": {"read_bytes": 315_000_000, "ops_per_row": 315_000_000},
                "prefill": {"rows": r}, "requires": ["ds41.engram"],
            })
            parts.append({
                "id": f"L{L}.engram_table", "kind": "engram-table", "role": "lookup", "layer": L, "weight_bytes": 0,
                "storage_bytes": int(188.83 * GIB / 2), "decode": {"read_bytes": 24 * 264},
                "prefill": {"rows": r}, "io": {"out_bytes_per_row": 24 * 264}, "requires": ["ds41.lookup"],
            })
        if L == enc - 1:
            parts.append({
                "id": f"L{L}.ced_bridge", "kind": "ced-bridge", "role": "anchor", "layer": L, "weight_bytes": 4_000_000,
                "decode": {"read_bytes": 4_000_000, "ops_per_row": 8_000_000},
                "prefill": {"ops_per_row": 8_000_000, "rows": "all"}, "requires": ["ds41.attention"],
            })
        parts.append({
            "id": f"L{L}.shared", "kind": "shared-expert", "role": "branch", "layer": L, "weight_bytes": 37_600_000,
            "decode": {"read_bytes": 37_600_000, "ops_per_row": 70_800_000}, "prefill": {"rows": r},
            "io": {"in_bytes_per_row": a8_in, "out_bytes_per_row": half_sum}, "shardable": True,
            "requires": ["ds41.moe-shared"],
        })
        parts.append({
            "id": f"L{L}.experts", "kind": "routed-experts", "role": "branch", "layer": L,
            "weight_bytes": int(142 * GIB / layers),
            "decode": {"read_bytes": 60_000_000, "read_bytes_extra_row": 56_000_000, "ops_per_row": 425_000_000},
            "prefill": {"rows": r},
            "io": {"in_bytes_per_row": a8_in, "out_bytes_per_row": half_sum, "combine": "per-shard"},
            "shardable": True, "requires": ["ds41.moe-routed"],
        })
    parts.append({
        "id": "head", "kind": "lm-head", "role": "head", "layer": layers, "weight_bytes": 703_000_000,
        "decode": {"read_bytes": 703_000_000, "ops_per_row": 1_324_000_000}, "prefill": {"rows": "last"},
        "shardable": True, "requires": ["ds41.core"],
    })
    parts.append({
        "id": "draft", "kind": "speculative-drafter", "role": "draft", "layer": layers,
        "weight_bytes": int(7.42 * GIB), "optional": True, "requires": ["ds41.draft"],
    })
    # Cuts inside a KV-sharing range need the producer's state on the far side;
    # every cut at or after the encoder end needs the decoder's shared pool.
    cuts = []
    for after in range(layers - 1):
        side = 0
        if 2 <= after < enc - 1 and after not in (7, 13):
            side = 178
        if after >= enc - 1:
            side = 356
        if side:
            cuts.append({"after_layer": after, "side_bytes_per_token": side})
    return {
        "schema": "tensormeld/model-parts-v1",
        "model": {"id": "deepseek-v4.1-flash", "revision": "q2-experts/q8-dense", "numerics": "engine-default",
                  "family": "moe-ced-mhc"},
        "provenance": "estimate",
        "layers": layers,
        "stream_bytes_per_row": 4 * hidden * 4,
        "token_bytes": 4,
        "moe": {"experts": 384, "top_k": 6, "expert_bytes": 9_953_280},
        "parts": parts,
        "cuts": cuts,
        "notes": "Shapes from the model card; bytes/token from engine receipts; prefill rows of layers 20..39 "
                 "follow the exact causal cone (layer 38-m computes 128+127m rows).",
    }


def qwen() -> dict:
    layers = 48
    hidden = 2560
    a8_in = 2700
    half_sum = hidden * 4
    parts = [{
        "id": "embed", "kind": "embedding", "role": "lookup", "layer": -1, "weight_bytes": 248320 * hidden * 2,
        "decode": {"read_bytes": hidden * 2}, "io": {"out_bytes_per_row": half_sum}, "requires": ["qwen.lookup"],
    }]
    for L in range(layers):
        full = L % 4 == 3
        if full:
            parts.append({
                "id": f"L{L}.attn", "kind": "full-attention", "role": "anchor", "layer": L,
                "weight_bytes": 56_200_000, "decode": {"read_bytes": 56_200_000, "ops_per_row": 106_000_000},
                "prefill": {"ops_per_row": 156_000_000}, "state": {"bytes_per_token": 2304},
                "requires": ["qwen.attention"],
            })
        else:
            parts.append({
                "id": f"L{L}.gdn", "kind": "linear-attention", "role": "anchor", "layer": L,
                "weight_bytes": 65_600_000, "decode": {"read_bytes": 65_600_000, "ops_per_row": 116_000_000},
                "state": {"bytes_fixed": 3_300_000}, "requires": ["qwen.gdn"],
            })
        parts.append({
            "id": f"L{L}.mix", "kind": "router-mix", "role": "anchor", "layer": L, "weight_bytes": 19_700_000,
            "decode": {"read_bytes": 19_700_000, "ops_per_row": 8_000_000}, "requires": ["qwen.core"],
        })
        if L == 2:
            parts.append({
                "id": "L2.ple", "kind": "per-layer-embedding", "role": "lookup", "layer": 2,
                "weight_bytes": int(26.8 * GIB), "decode": {"read_bytes": 35_000_000},
                "io": {"out_bytes_per_row": half_sum}, "requires": ["qwen.lookup"],
            })
        parts.append({
            "id": f"L{L}.shared", "kind": "shared-expert", "role": "branch", "layer": L, "weight_bytes": 5_200_000,
            "decode": {"read_bytes": 5_200_000, "ops_per_row": 9_800_000},
            "io": {"in_bytes_per_row": a8_in, "out_bytes_per_row": half_sum}, "requires": ["qwen.moe"],
        })
        parts.append({
            "id": f"L{L}.experts", "kind": "routed-experts", "role": "branch", "layer": L,
            "weight_bytes": 77 * GB // layers,
            "decode": {"read_bytes": 31_250_000, "read_bytes_extra_row": 18_300_000, "ops_per_row": 98_000_000},
            "io": {"in_bytes_per_row": a8_in, "out_bytes_per_row": half_sum}, "shardable": True,
            "requires": ["qwen.moe"],
        })
    parts.append({
        "id": "head", "kind": "lm-head", "role": "head", "layer": layers, "weight_bytes": 675_000_000,
        "decode": {"read_bytes": 675_000_000, "ops_per_row": 1_271_000_000}, "prefill": {"rows": "last"},
        "requires": ["qwen.core"],
    })
    parts.append({
        "id": "draft", "kind": "speculative-drafter", "role": "draft", "layer": layers,
        "weight_bytes": 2_786_568_256, "optional": True, "requires": ["qwen.draft"],
    })
    return {
        "schema": "tensormeld/model-parts-v1",
        "model": {"id": "qwen3.8-flash-next", "revision": "q4-experts/q8-dense", "numerics": "engine-default",
                  "family": "moe-hybrid-linear-attention"},
        "provenance": "estimate",
        "layers": layers,
        "stream_bytes_per_row": 4 * hidden * 4,
        "token_bytes": 4,
        "parts": parts,
        "notes": "Bytes per token from the engine's decode traffic model; linear-attention state is fixed per "
                 "request (recurrence), full-attention state grows per token.",
    }


def hardware() -> dict:
    return {
        "schema": "tensormeld/hetero-hardware-v1",
        "nodes": [
            {"id": "pc", "disk_free_bytes": 1_700 * GB, "disk_bytes_per_s": 6 * GB, "model_files": []},
            {"id": "node1", "disk_free_bytes": 300 * GB, "disk_bytes_per_s": 6 * GB,
             "model_files": ["deepseek-v4.1-flash", "qwen3.8-flash-next"]},
            {"id": "node2", "disk_free_bytes": 300 * GB, "disk_bytes_per_s": 6 * GB,
             "model_files": ["deepseek-v4.1-flash", "qwen3.8-flash-next"]},
        ],
        "devices": [
            {"id": "rtx", "node": "pc", "kind": "discrete_gpu", "backend": "cuda",
             "memory_bytes": 97_887 * 2**20, "reserved_bytes": 6 * GB,
             "mem_bytes_per_s": 1_380 * GB, "prefill_ops_per_s": 150 * 10**12,
             "step_overhead_us": 40, "crossing_overhead_us": 65,
             "availability": "preemptible", "supports": ["*"], "provenance": "measured",
             "notes": "Measured 2026-10-20 on Windows (WDDM), GPU shared with other jobs (minimums used). "
                      "mem_bytes_per_s: Q8_0 x A8 decode GEMV prototype over one layer's dense projections "
                      "(attention + shared expert, 172 MB, CUDA graph) = 124 us/layer = 1.38 TB/s (big GEMVs "
                      "1.43-1.49 TB/s; 1.79 nominal). crossing_overhead_us: GPU->TCP->GPU minus TCP alone, "
                      "p50 ~65 us (D2H 11-26 + H2D after idle 45-75); p99 adds ~150-300 us. step_overhead_us "
                      "(derived): graph node 0.43 us + small dependent kernels (~3 us fixed each). "
                      "prefill_ops_per_s stays an estimate: dense INT8 GEMM measures 560-690 TOPS, FP8 500-575, "
                      "BF16 290-310; 150 assumes quantized grouped kernels at ~22 % of that. "
                      "'supports' is the PLANNED backend: today no engine runs these parts here."},
            {"id": "pc-host", "node": "pc", "kind": "cpu", "backend": "host",
             "memory_bytes": 128 * GIB, "reserved_bytes": 80 * GIB, "mem_bytes_per_s": 60 * GB,
             "prefill_ops_per_s": 10**12, "step_overhead_us": 5, "crossing_overhead_us": 5,
             "availability": "available", "supports": ["ds41.lookup", "qwen.lookup"], "provenance": "estimate",
             "notes": "PC host RAM for token-id lookup tables next to the GPU; 48 GiB allowed (the PC runs other work)."},
            *({"id": f"{n}-gpu", "node": n, "kind": "integrated_gpu", "backend": "hip",
               "memory_bytes": 128 * GIB, "reserved_bytes": 10 * GIB,
               "mem_bytes_per_s": 165 * GB, "prefill_ops_per_s": int(6.5 * 10**12),
               "prefill_rates": {"qwen.core": 12 * 10**12, "qwen.gdn": 12 * 10**12, "qwen.attention": 12 * 10**12,
                                 "qwen.moe": 12 * 10**12},
               "step_overhead_us": 115, "crossing_overhead_us": 30,
               "availability": "available", "supports": ["*"], "provenance": "measured",
               "notes": "Rates fitted to engine receipts: decode 20 tok/s two-node (49.8 ms), prefill 567 tok/s at "
                        "8K two-node; ~1000 tok/s one-node for the hybrid model."} for n in ("node1", "node2")),
        ],
        "links": [
            {"id": "eth-pc-c1", "a": "pc", "b": "node1", "bytes_per_s": 117_000_000, "latency_us": 42,
             "shared_group": "pc-nic", "provenance": "measured",
             "notes": "1 GbE switch; NICs are 2.5 GbE. link-probe 2026-10-20: RTT p50 84 us, 118.6 MB/s bulk."},
            {"id": "eth-pc-c2", "a": "pc", "b": "node2", "bytes_per_s": 117_000_000, "latency_us": 42,
             "shared_group": "pc-nic", "provenance": "measured"},
            {"id": "eth-c1-c2", "a": "node1", "b": "node2", "bytes_per_s": 117_000_000, "latency_us": 42,
             "provenance": "estimate"},
            {"id": "usb4-c1-c2", "a": "node1", "b": "node2", "bytes_per_s": 4_700_000_000, "latency_us": 14,
             "provenance": "measured", "notes": "two cables, host path; 32 us RTT p50 at 20 KB, 4.7 GB/s at 42 MB."},
        ],
        "groups": [{"id": "amd-pair", "devices": ["node1-gpu", "node2-gpu"], "link": "usb4-c1-c2",
                    "supports": ["ds41.attention", "ds41.core", "ds41.draft", "ds41.engram", "ds41.lookup",
                                 "ds41.moe-routed", "ds41.moe-shared"],
                    "notes": "engine group modes today: the 40-layer MoE as expert + tensor parallel over USB4; "
                             "the hybrid model has no group mode yet"}],
        "notes": "Lab profile. Override link rates with --link to ask what-if questions (2.5/10/25 GbE, USB4).",
    }


def main() -> None:
    for name, data in (("deepseek-v4.1-flash.parts.json", deepseek()), ("qwen3.8-flash-next.parts.json", qwen()),
                       ("lab-hardware.json", hardware())):
        (HERE / name).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
