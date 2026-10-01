"""Exact checkpoint identity contract.

A model manifest identifies bytes and semantic references. It deliberately does not
infer runtime RAM/VRAM, operator coverage, or execution qualification from file size.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, items, number, record, text, unique

MAX_MODEL_FILES = 128
_HEX = set("0123456789abcdef")


def _sha256(value: Any, where: str) -> str:
    s = text(value, where).lower()
    if len(s) != 64 or any(c not in _HEX for c in s):
        raise ValidationError(f"{where}: expected 64-character lowercase SHA-256 hex")
    return s


@dataclass(frozen=True)
class ModelFile:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    revision: str
    format: str
    architecture: str
    tokenizer_ref: str
    chat_template_ref: str | None
    files: tuple[ModelFile, ...]
    tensor_index_sha256: str
    tensor_count: int
    tensor_payload_bytes: int
    manifest_sha256: str

    @classmethod
    def parse(cls, data: Any) -> "ModelManifest":
        r = record(data, "model manifest", {
            "model_manifest_schema", "model_id", "revision", "format", "architecture",
            "tokenizer_ref", "chat_template_ref", "files", "tensor_index_sha256",
            "tensor_count", "tensor_payload_bytes",
        })
        if r["model_manifest_schema"] != "tensormeld/model-manifest-v1":
            raise ValidationError("model_manifest_schema: expected tensormeld/model-manifest-v1")
        if r["format"] != "gguf":
            raise ValidationError("format: only gguf is supported by the current manifest adapter")
        files = []
        for i, raw in enumerate(items(r["files"], "files", MAX_MODEL_FILES, 1)):
            f = record(raw, f"files[{i}]", {"name", "size_bytes", "sha256"})
            files.append(ModelFile(
                text(f["name"], f"files[{i}].name"),
                int(number(f["size_bytes"], f"files[{i}].size_bytes", 1, True)),
                _sha256(f["sha256"], f"files[{i}].sha256"),
            ))
        unique([f.name for f in files], "files")
        tokenizer = text(r["tokenizer_ref"], "tokenizer_ref")
        chat = r["chat_template_ref"]
        if chat is not None:
            chat = text(chat, "chat_template_ref")
        canonical = {
            "model_manifest_schema": r["model_manifest_schema"],
            "model_id": text(r["model_id"], "model_id"),
            "revision": text(r["revision"], "revision"),
            "format": "gguf",
            "architecture": text(r["architecture"], "architecture"),
            "tokenizer_ref": tokenizer,
            "chat_template_ref": chat,
            "files": [f.__dict__ for f in files],
            "tensor_index_sha256": _sha256(r["tensor_index_sha256"], "tensor_index_sha256"),
            "tensor_count": int(number(r["tensor_count"], "tensor_count", 1, True)),
            "tensor_payload_bytes": int(number(r["tensor_payload_bytes"], "tensor_payload_bytes", 1, True)),
        }
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            canonical["model_id"], canonical["revision"], "gguf", canonical["architecture"],
            tokenizer, chat, tuple(files), canonical["tensor_index_sha256"],
            canonical["tensor_count"], canonical["tensor_payload_bytes"], digest,
        )


def manifest_from_gguf_index(
    index: dict[str, Any], *, model_id: str, revision: str, tokenizer_ref: str,
    shard_sha256: dict[str, str], chat_template_ref: str | None = None,
) -> ModelManifest:
    if index.get("index_schema") != "tensormeld/gguf-index-v1":
        raise ValidationError("index: expected tensormeld/gguf-index-v1")
    if index.get("complete_shard_set") is not True:
        raise ValidationError("index: complete shard set is required for a model manifest")
    if index.get("qualified") is not False or index.get("executable") is not False:
        raise ValidationError("index: inspection output must remain non-qualified/non-executable")
    files = index.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("index.files: expected non-empty shard list")
    names = [f.get("file_name") for f in files]
    if set(names) != set(shard_sha256) or len(shard_sha256) != len(names):
        raise ValidationError("shard_sha256 must cover the exact indexed shard set")
    raw = {
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": model_id,
        "revision": revision,
        "format": "gguf",
        "architecture": index.get("architecture"),
        "tokenizer_ref": tokenizer_ref,
        "chat_template_ref": chat_template_ref,
        "files": [
            {
                "name": f["file_name"],
                "size_bytes": f["file_size_bytes"],
                "sha256": shard_sha256[f["file_name"]],
            }
            for f in files
        ],
        "tensor_index_sha256": index.get("index_sha256"),
        "tensor_count": index.get("tensor_count"),
        "tensor_payload_bytes": index.get("tensor_payload_bytes"),
    }
    return ModelManifest.parse(raw)


def load_model_manifest(path: str | Path) -> ModelManifest:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("model manifest exceeds 2 MiB")
    try:
        return ModelManifest.parse(json.loads(raw, object_pairs_hook=_no_duplicates))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid model manifest JSON: {exc}") from exc
