"""Strict model-aware placement translation for pinned llama.cpp.

This module translates only TensorMeld whole-block units named exactly ``blk.N`` into
pinned llama.cpp tensor-buffer override rules. It does not launch a model, infer tensor
names beyond the documented ``blk.N...`` namespace, or approximate placement with
tensor-split ratios.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .adapter_contract import AdapterCapabilities
from .llamacpp_probe import LLAMACPP_PINNED_COMMIT
from .llamacpp_selftest import BOUND_RESULT_SCHEMA
from .model_manifest import ModelManifest
from .schema import ValidationError, items, record, text, unique
from .whole_block_execution import AcceptedExecutionBundle

PLACEMENT_SCHEMA = "tensormeld/llamacpp-placement-binding-v1"
TRANSLATION_SCHEMA = "tensormeld/llamacpp-whole-block-placement-v1"
_MAX_DEVICES = 128
_BLOCK_RE = re.compile(r"^blk\.(0|[1-9][0-9]*)$")
_BUFT_RE = re.compile(r"^[A-Za-z0-9_.:\[\]()/+-]{1,128}$")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LlamaCppPlacementBinding:
    adapter_id: str
    adapter_capabilities_sha256: str
    source_revision: str
    native_binding_sha256: str
    device_bindings: tuple[tuple[str, str, str], ...]
    fingerprint: str

    @classmethod
    def parse(cls, data: Any, *, adapter: AdapterCapabilities) -> "LlamaCppPlacementBinding":
        r = record(data, "llama.cpp placement binding", {"placement_binding_schema", "adapter_id", "adapter_capabilities_sha256", "source_revision", "native_binding_sha256", "devices"})
        if r["placement_binding_schema"] != PLACEMENT_SCHEMA:
            raise ValidationError(f"placement_binding_schema: expected {PLACEMENT_SCHEMA}")
        if r["adapter_id"] != adapter.adapter_id:
            raise ValidationError("placement binding adapter_id mismatch")
        if r["adapter_capabilities_sha256"] != adapter.fingerprint:
            raise ValidationError("placement binding adapter fingerprint mismatch")
        if r["source_revision"] != LLAMACPP_PINNED_COMMIT:
            raise ValidationError("placement binding source revision mismatch")
        native_binding_sha256 = text(
            r["native_binding_sha256"], "native_binding_sha256"
        ).lower()
        if len(native_binding_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in native_binding_sha256
        ):
            raise ValidationError("native_binding_sha256: expected SHA-256 hex")
        exposed = {d.id for d in adapter.devices}
        pairs: list[tuple[str, str, str]] = []
        for i, raw in enumerate(items(r["devices"], "devices", _MAX_DEVICES, 1)):
            d = record(raw, f"devices[{i}]", {"device_id", "engine_device_name", "buffer_type"})
            device_id = text(d["device_id"], f"devices[{i}].device_id")
            if device_id not in exposed:
                raise ValidationError(f"devices[{i}]: device is not exposed by adapter")
            engine_name = text(d["engine_device_name"], f"devices[{i}].engine_device_name")
            if "," in engine_name or "=" in engine_name or not _BUFT_RE.fullmatch(engine_name):
                raise ValidationError(f"devices[{i}].engine_device_name: unsupported spelling")
            buft = text(d["buffer_type"], f"devices[{i}].buffer_type")
            if not _BUFT_RE.fullmatch(buft) or "," in buft or "=" in buft:
                raise ValidationError(f"devices[{i}].buffer_type: unsupported buffer-type spelling")
            if engine_name.startswith("RPC") or buft.startswith("RPC"):
                raise ValidationError("remote llama.cpp RPC devices are not permitted by this local shim")
            pairs.append((device_id, engine_name, buft))
        unique([d for d, _, _ in pairs], "devices.device_id")
        unique([e for _, e, _ in pairs], "devices.engine_device_name")
        unique([b for _, _, b in pairs], "devices.buffer_type")
        canonical = {"placement_binding_schema": PLACEMENT_SCHEMA, "adapter_id": adapter.adapter_id, "adapter_capabilities_sha256": adapter.fingerprint, "source_revision": LLAMACPP_PINNED_COMMIT, "native_binding_sha256": native_binding_sha256, "devices": [{"device_id": d, "engine_device_name": e, "buffer_type": b} for d, e, b in sorted(pairs)]}
        return cls(adapter.adapter_id, adapter.fingerprint, LLAMACPP_PINNED_COMMIT, native_binding_sha256, tuple(sorted(pairs)), _canonical_sha256(canonical))

    @property
    def engine_device_by_device(self) -> dict[str, str]:
        return {d: e for d, e, _ in self.device_bindings}

    @property
    def buffer_type_by_device(self) -> dict[str, str]:
        return {d: b for d, _, b in self.device_bindings}


@dataclass(frozen=True)
class LlamaCppPlacementTranslation:
    source_revision: str
    accepted_bundle_sha256: str
    placement_binding_sha256: str
    block_owners: tuple[tuple[int, str, str], ...]
    device_buffer_types: tuple[tuple[str, str], ...]
    override_tensor_value: str
    argv_fragment: tuple[str, ...]
    fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {"translation_schema": TRANSLATION_SCHEMA, "source_revision": self.source_revision, "accepted_bundle_sha256": self.accepted_bundle_sha256, "placement_binding_sha256": self.placement_binding_sha256, "block_owners": [{"block_index": i, "device_id": d, "buffer_type": b} for i, d, b in self.block_owners], "device_buffer_types": [{"device_id": d, "buffer_type": b} for d, b in self.device_buffer_types], "override_tensor_value": self.override_tensor_value, "argv_fragment": list(self.argv_fragment), "real_model_inference": False, "fingerprint": self.fingerprint}


def translate_whole_blocks_to_llamacpp(
    bundle: AcceptedExecutionBundle,
    *,
    adapter: AdapterCapabilities,
    binding: LlamaCppPlacementBinding,
    bound_result: dict[str, Any],
    model: ModelManifest,
    gguf_index: dict[str, Any],
) -> LlamaCppPlacementTranslation:
    if model.manifest_sha256 != bundle.model_manifest_sha256:
        raise ValidationError("model manifest does not match accepted bundle")
    if gguf_index.get("index_schema") != "tensormeld/gguf-index-v1":
        raise ValidationError("gguf index schema mismatch")
    if gguf_index.get("complete_shard_set") is not True:
        raise ValidationError("llama.cpp shim requires a complete GGUF shard index")
    if gguf_index.get("index_sha256") != model.tensor_index_sha256:
        raise ValidationError("GGUF tensor index identity does not match model manifest")
    if gguf_index.get("architecture") != model.architecture:
        raise ValidationError("GGUF architecture does not match model manifest")
    if gguf_index.get("tensor_count") != model.tensor_count:
        raise ValidationError("GGUF tensor count does not match model manifest")

    if len(bundle.compute_nodes) != 1:
        raise ValidationError("initial llama.cpp shim supports only one compute node")
    if bundle.adapter_id != adapter.adapter_id:
        raise ValidationError("accepted bundle adapter_id mismatch")
    if bundle.engine_revision != adapter.engine_revision:
        raise ValidationError("accepted bundle engine revision mismatch")
    if bundle.adapter_capabilities_sha256 != adapter.fingerprint:
        raise ValidationError("accepted bundle adapter fingerprint mismatch")
    if binding.adapter_id != adapter.adapter_id or binding.adapter_capabilities_sha256 != adapter.fingerprint:
        raise ValidationError("placement binding adapter identity mismatch")
    if binding.source_revision != LLAMACPP_PINNED_COMMIT:
        raise ValidationError("placement binding source revision mismatch")
    if bound_result.get("result_schema") != BOUND_RESULT_SCHEMA:
        raise ValidationError("current llama.cpp bound result schema mismatch")
    if bound_result.get("config_sha256") != bundle.config_sha256:
        raise ValidationError("current llama.cpp bound result config mismatch")
    if bound_result.get("binding_sha256") != binding.native_binding_sha256:
        raise ValidationError("placement binding native binding fingerprint mismatch")
    mappings = bound_result.get("resolved_mappings")
    if not isinstance(mappings, list):
        raise ValidationError("current llama.cpp bound result mappings must be a list")
    engine_by_bound_device = {}
    for item in mappings:
        if isinstance(item, dict):
            device_id = item.get("tensormeld_device_id")
            engine_name = item.get("engine_device_name")
            if isinstance(device_id, str) and isinstance(engine_name, str):
                if device_id in engine_by_bound_device:
                    raise ValidationError("duplicate current native mapping for device")
                engine_by_bound_device[device_id] = engine_name
    if set(engine_by_bound_device) != set(bundle.compute_devices):
        raise ValidationError("current native mappings must cover exactly the bundle compute devices")
    for device_id, engine_name, _ in binding.device_bindings:
        if engine_by_bound_device.get(device_id) != engine_name:
            raise ValidationError("placement binding engine device differs from current native mapping")

    actual_block_indices: set[int] = set()
    files = gguf_index.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("GGUF index files must be a non-empty list")
    for shard in files:
        tensors = shard.get("tensors") if isinstance(shard, dict) else None
        if not isinstance(tensors, list):
            raise ValidationError("GGUF shard tensors must be a list")
        for tensor in tensors:
            name = tensor.get("name") if isinstance(tensor, dict) else None
            if not isinstance(name, str):
                raise ValidationError("GGUF tensor name is invalid")
            if name.startswith("blk."):
                match = re.match(r"^blk\.(0|[1-9][0-9]*)\.", name)
                if not match:
                    raise ValidationError(
                        "GGUF contains an unsupported blk.* tensor namespace"
                    )
                actual_block_indices.add(int(match.group(1)))
    if not actual_block_indices:
        raise ValidationError("GGUF index contains no transformer block tensors")
    actual_sorted = sorted(actual_block_indices)
    if actual_sorted != list(range(actual_sorted[-1] + 1)):
        raise ValidationError("GGUF transformer block indices are not contiguous from zero")

    indices: list[int] = []
    for unit_id in bundle.unit_ids:
        match = _BLOCK_RE.fullmatch(unit_id)
        if not match:
            raise ValidationError("llama.cpp shim accepts only unit IDs named exactly blk.N")
        indices.append(int(match.group(1)))
    if len(indices) != len(set(indices)):
        raise ValidationError("duplicate transformer block unit index")
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValidationError("llama.cpp shim requires contiguous ascending transformer blocks")
    if indices != actual_sorted:
        raise ValidationError(
            "accepted bundle must cover exactly every transformer block in the GGUF index"
        )

    owner_by_unit: list[str | None] = [None] * len(bundle.unit_ids)
    for device, first, last in bundle.segments:
        if device not in bundle.compute_devices:
            raise ValidationError("bundle segment references unknown compute device")
        if not (0 <= first < last <= len(bundle.unit_ids)):
            raise ValidationError("bundle segment range is invalid")
        for pos in range(first, last):
            if owner_by_unit[pos] is not None:
                raise ValidationError("bundle segments overlap")
            owner_by_unit[pos] = device
    if any(owner is None for owner in owner_by_unit):
        raise ValidationError("bundle segments do not cover every unit exactly once")

    engine_by_device = binding.engine_device_by_device
    buft_by_device = binding.buffer_type_by_device
    if set(bundle.compute_devices) != set(buft_by_device):
        raise ValidationError("placement binding must cover exactly the bundle compute devices")

    block_owners: list[tuple[int, str, str]] = []
    overrides: list[str] = []
    for pos, block_index in enumerate(indices):
        device = owner_by_unit[pos]
        assert device is not None
        buft = buft_by_device[device]
        block_owners.append((block_index, device, buft))
        overrides.append(rf"^blk\.{block_index}\..*={buft}")

    override_value = ",".join(overrides)
    argv = ("--fit", "off", "--device", ",".join(engine_by_device[d] for d in bundle.compute_devices), "--override-tensor", override_value)
    canonical = {"translation_schema": TRANSLATION_SCHEMA, "source_revision": LLAMACPP_PINNED_COMMIT, "accepted_bundle_sha256": bundle.bundle_sha256, "placement_binding_sha256": binding.fingerprint, "block_owners": [[i, d, b] for i, d, b in block_owners], "engine_devices": [[d, engine_by_device[d]] for d in bundle.compute_devices], "device_buffer_types": [[d, buft_by_device[d]] for d in bundle.compute_devices], "override_tensor_value": override_value, "argv_fragment": list(argv), "real_model_inference": False}
    return LlamaCppPlacementTranslation(LLAMACPP_PINNED_COMMIT, bundle.bundle_sha256, binding.fingerprint, tuple(block_owners), tuple((d, buft_by_device[d]) for d in bundle.compute_devices), override_value, argv, _canonical_sha256(canonical))
