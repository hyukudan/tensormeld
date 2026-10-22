"""Model parts descriptor: a model as an ordered graph of parts with costs.

``tensormeld/model-parts-v1`` describes what each part of a model needs so a
placement planner can split it across unlike devices.  It is engine-declared
(an engine's ``--describe-json`` emits it for a model file and numerical
profile) or hand-written for planning; it never makes a plan executable.

Structure.  A model is a chain of layers ``0..layers-1``.  Every part belongs
to a layer (``-1`` = before layer 0, ``layers`` = after the last layer) and has
a role:

``anchor``
    Runs where the layer's residual stream lives (attention, mixers, norms,
    routers).  All anchors of one layer share a device set; that set is the
    layer's *home*.  Moving the home between consecutive layers moves
    ``stream_bytes_per_row`` per row.
``branch``
    Consumes ``io.in_bytes_per_row`` from the home and returns
    ``io.out_bytes_per_row`` (shared or routed experts, dense FFN).  It may run
    on another device set; it then costs a round trip per layer.  Branches of
    one layer run in parallel with each other.
``lookup``
    Input depends only on token ids (embeddings, n-gram tables).  It can run
    ahead of the stream, so a remote lookup costs bytes but no latency, except
    at layer ``-1``/``0`` where the next token waits for it.
``head``
    Consumes the final stream and emits a token (``token_bytes`` back to the
    embedding).
``draft``
    Optional speculative drafter; counted in memory when requested.

Costs are per row (a row is one token position in a step): bytes read from
device memory in decode, operations, prefill rows (``all`` | ``last`` |
``{"max": N}`` for layers that only see a bounded suffix of the prompt),
persistent state, and activation bytes at the part's boundary.  ``cuts`` add
side bytes per prompt token when the home changes after a given layer (state
that later layers need from earlier ones).

Optional ``moe`` geometry (``experts`` per layer, ``top_k``, ``expert_bytes``)
lets per-expert inputs, such as a hot-expert placement file, be checked
against the model they claim to describe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, number, record, text

MODEL_PARTS_SCHEMA = "tensormeld/model-parts-v1"
MAX_LAYERS = 512
MAX_PARTS = 8192
MAX_CAPABILITIES = 64
ROLES = frozenset({"anchor", "branch", "lookup", "head", "draft"})
PROVENANCE = frozenset({"declared", "engine-describe", "fixture", "estimate"})


@dataclass(frozen=True)
class Part:
    id: str
    kind: str
    role: str
    layer: int
    weight_bytes: int
    storage_bytes: int
    decode_read_bytes: int
    decode_read_bytes_extra_row: int
    decode_ops_per_row: int
    prefill_ops_per_row: int
    prefill_rows: str  # "all" | "last" | "max"
    prefill_rows_max: int
    state_bytes_per_token: int
    state_bytes_fixed: int
    in_bytes_per_row: int
    out_bytes_per_row: int
    shardable: bool
    requires: tuple[str, ...]
    optional: bool
    replicated_fraction: float = 0.0  # share of bytes/ops every shard repeats (e.g. TP keeps small projections whole)
    combine: str = "reduce"  # "reduce": a group returns one summed result; "per-shard": each shard's partial returns

    def shard_share(self, shards: int) -> float:
        """Fraction of this part's bytes/ops that one of ``shards`` devices carries."""
        if shards <= 1 or not self.shardable:
            return 1.0
        return self.replicated_fraction + (1.0 - self.replicated_fraction) / shards

    def rows_for_prompt(self, prompt_tokens: int) -> int:
        if self.prefill_rows == "all":
            return prompt_tokens
        if self.prefill_rows == "last":
            return 1
        return min(prompt_tokens, self.prefill_rows_max)

    def decode_bytes(self, rows: int) -> int:
        return self.decode_read_bytes + max(rows - 1, 0) * self.decode_read_bytes_extra_row

    def state_bytes(self, context_tokens: int) -> int:
        return self.state_bytes_fixed + self.state_bytes_per_token * context_tokens


@dataclass(frozen=True)
class Cut:
    after_layer: int
    side_bytes_per_token: int


@dataclass(frozen=True)
class MoeGeometry:
    experts: int
    top_k: int
    expert_bytes: int  # 0 = not declared


@dataclass(frozen=True)
class ModelParts:
    model_id: str
    revision: str
    numerics: str
    provenance: str
    layers: int
    stream_bytes_per_row: int
    token_bytes: int
    parts: tuple[Part, ...]
    cuts: tuple[Cut, ...]
    fingerprint: str
    moe: MoeGeometry | None = None

    def part(self, part_id: str) -> Part:
        for p in self.parts:
            if p.id == part_id:
                return p
        raise KeyError(part_id)

    def by_layer(self, layer: int) -> tuple[Part, ...]:
        return tuple(p for p in self.parts if p.layer == layer)

    def anchors(self, layer: int) -> tuple[Part, ...]:
        return tuple(p for p in self.parts if p.layer == layer and p.role == "anchor")

    def side_bytes_after(self, layer: int) -> int:
        return sum(c.side_bytes_per_token for c in self.cuts if c.after_layer == layer)

    def total_weight_bytes(self, include_optional: bool = False) -> int:
        return sum(p.weight_bytes for p in self.parts if include_optional or not p.optional)


def _int(value, where: str, minimum: int = 0) -> int:
    return int(number(value, where, minimum, True))


def _bool(value, where: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{where}: expected boolean")
    return value


def _part(raw, where: str, layers: int) -> Part:
    r = record(raw, where, {"id", "kind", "role", "layer", "weight_bytes"},
               {"storage_bytes", "decode", "prefill", "state", "io", "shardable", "requires", "optional",
                "replicated_fraction"})
    pid = text(r["id"], f"{where}.id")
    role = r["role"]
    if role not in ROLES:
        raise ValidationError(f"{where}.role: expected one of {sorted(ROLES)}")
    layer = int(number(r["layer"], f"{where}.layer", -1, True))
    if layer > layers:
        raise ValidationError(f"{where}.layer: above layers")
    if role in ("anchor", "branch") and not 0 <= layer < layers:
        raise ValidationError(f"{where}: {role} parts must belong to a layer 0..{layers - 1}")
    if role == "head" and layer != layers:
        raise ValidationError(f"{where}: head parts use layer = layers ({layers})")
    decode = record(r.get("decode", {}), f"{where}.decode", set(),
                    {"read_bytes", "read_bytes_extra_row", "ops_per_row"})
    prefill = record(r.get("prefill", {}), f"{where}.prefill", set(), {"ops_per_row", "rows"})
    state = record(r.get("state", {}), f"{where}.state", set(), {"bytes_per_token", "bytes_fixed"})
    io = record(r.get("io", {}), f"{where}.io", set(), {"in_bytes_per_row", "out_bytes_per_row", "combine"})
    combine = io.get("combine", "reduce")
    if combine not in ("reduce", "per-shard"):
        raise ValidationError(f"{where}.io.combine: expected reduce or per-shard")
    replicated = float(number(r.get("replicated_fraction", 0), f"{where}.replicated_fraction"))
    if replicated > 1:
        raise ValidationError(f"{where}.replicated_fraction: expected 0..1")
    rows = prefill.get("rows", "all")
    rows_max = 0
    if isinstance(rows, dict):
        rows_max = _int(record(rows, f"{where}.prefill.rows", {"max"})["max"], f"{where}.prefill.rows.max", 1)
        rows = "max"
    elif rows not in ("all", "last"):
        raise ValidationError(f"{where}.prefill.rows: expected all, last or {{max: N}}")
    requires = r.get("requires", [])
    if not isinstance(requires, list) or len(requires) > MAX_CAPABILITIES:
        raise ValidationError(f"{where}.requires: expected a bounded list")
    requires = tuple(sorted(text(x, f"{where}.requires[]") for x in requires))
    if role in ("branch", "lookup") and "out_bytes_per_row" not in io:
        raise ValidationError(f"{where}.io.out_bytes_per_row: required for {role} parts")
    return Part(
        id=pid,
        kind=text(r["kind"], f"{where}.kind"),
        role=role,
        layer=layer,
        weight_bytes=_int(r["weight_bytes"], f"{where}.weight_bytes"),
        storage_bytes=_int(r.get("storage_bytes", 0), f"{where}.storage_bytes"),
        decode_read_bytes=_int(decode.get("read_bytes", 0), f"{where}.decode.read_bytes"),
        decode_read_bytes_extra_row=_int(decode.get("read_bytes_extra_row", 0), f"{where}.decode.read_bytes_extra_row"),
        decode_ops_per_row=_int(decode.get("ops_per_row", 0), f"{where}.decode.ops_per_row"),
        prefill_ops_per_row=_int(prefill.get("ops_per_row", decode.get("ops_per_row", 0)), f"{where}.prefill.ops_per_row"),
        prefill_rows=rows,
        prefill_rows_max=rows_max,
        state_bytes_per_token=_int(state.get("bytes_per_token", 0), f"{where}.state.bytes_per_token"),
        state_bytes_fixed=_int(state.get("bytes_fixed", 0), f"{where}.state.bytes_fixed"),
        in_bytes_per_row=_int(io.get("in_bytes_per_row", 0), f"{where}.io.in_bytes_per_row"),
        out_bytes_per_row=_int(io.get("out_bytes_per_row", 0), f"{where}.io.out_bytes_per_row"),
        shardable=_bool(r.get("shardable", False), f"{where}.shardable"),
        requires=requires,
        optional=_bool(r.get("optional", False), f"{where}.optional"),
        replicated_fraction=replicated,
        combine=combine,
    )


def parse_model_parts(data) -> ModelParts:
    r = record(data, "model parts",
               {"schema", "model", "provenance", "layers", "stream_bytes_per_row", "parts"},
               {"token_bytes", "cuts", "notes", "moe"})
    if r["schema"] != MODEL_PARTS_SCHEMA:
        raise ValidationError(f"schema: expected {MODEL_PARTS_SCHEMA}")
    if r["provenance"] not in PROVENANCE:
        raise ValidationError(f"provenance: expected one of {sorted(PROVENANCE)}")
    model = record(r["model"], "model", {"id"}, {"revision", "numerics", "family"})
    layers = _int(r["layers"], "layers", 1)
    if layers > MAX_LAYERS:
        raise ValidationError(f"layers: at most {MAX_LAYERS}")
    raw_parts = r["parts"]
    if not isinstance(raw_parts, list) or not 1 <= len(raw_parts) <= MAX_PARTS:
        raise ValidationError(f"parts: expected 1..{MAX_PARTS} entries")
    parts = tuple(_part(p, f"parts[{i}]", layers) for i, p in enumerate(raw_parts))
    ids = [p.id for p in parts]
    if len(set(ids)) != len(ids):
        raise ValidationError("parts: duplicate id")
    for layer in range(layers):
        if not any(p.role == "anchor" and p.layer == layer for p in parts):
            raise ValidationError(f"layer {layer}: needs at least one anchor part")
    if not any(p.role == "head" for p in parts):
        raise ValidationError("parts: a head part is required")
    cuts = []
    raw_cuts = r.get("cuts", [])
    if not isinstance(raw_cuts, list) or len(raw_cuts) > MAX_LAYERS:
        raise ValidationError("cuts: expected a bounded list")
    for i, c in enumerate(raw_cuts):
        c = record(c, f"cuts[{i}]", {"after_layer", "side_bytes_per_token"})
        after = _int(c["after_layer"], f"cuts[{i}].after_layer")
        if after >= layers - 1:
            raise ValidationError(f"cuts[{i}].after_layer: must be below the last layer")
        cuts.append(Cut(after, _int(c["side_bytes_per_token"], f"cuts[{i}].side_bytes_per_token")))
    moe = None
    if "moe" in r:
        g = record(r["moe"], "moe", {"experts", "top_k"}, {"expert_bytes"})
        experts = _int(g["experts"], "moe.experts", 1)
        top_k = _int(g["top_k"], "moe.top_k", 1)
        if experts > 1 << 16 or top_k > experts:
            raise ValidationError("moe: expected 1 <= top_k <= experts <= 65536")
        moe = MoeGeometry(experts, top_k, _int(g.get("expert_bytes", 0), "moe.expert_bytes"))
    digest = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return ModelParts(
        model_id=text(model["id"], "model.id"),
        revision=str(model.get("revision", "")),
        numerics=str(model.get("numerics", "")),
        provenance=r["provenance"],
        layers=layers,
        stream_bytes_per_row=_int(r["stream_bytes_per_row"], "stream_bytes_per_row", 1),
        token_bytes=_int(r.get("token_bytes", 4), "token_bytes", 1),
        parts=parts,
        cuts=tuple(cuts),
        fingerprint=digest,
        moe=moe,
    )


def _load_json(path: Path):
    raw = Path(path).read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError(f"{path}: above {MAX_INPUT_BYTES} bytes")
    return json.loads(raw, object_pairs_hook=_no_duplicates)


def load_model_parts(path: Path) -> ModelParts:
    return parse_model_parts(_load_json(path))


def model_parts_summary(model: ModelParts) -> dict:
    by_role: dict[str, int] = {}
    for p in model.parts:
        by_role[p.role] = by_role.get(p.role, 0) + p.weight_bytes
    return {
        "schema": "tensormeld/model-parts-summary-v1",
        "model": model.model_id,
        "fingerprint": model.fingerprint,
        "layers": model.layers,
        "parts": len(model.parts),
        "weight_bytes": model.total_weight_bytes(),
        "weight_bytes_by_role": dict(sorted(by_role.items())),
        "storage_bytes": sum(p.storage_bytes for p in model.parts),
        "decode_read_bytes_per_token": sum(p.decode_read_bytes for p in model.parts if not p.optional),
        "state_bytes_per_token": sum(p.state_bytes_per_token for p in model.parts if not p.optional),
    }
