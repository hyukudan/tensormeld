"""Durable declaration-only request-family applicability.

This contract neither widens exact E4 replay nor authorizes serving. Trusted adapters
must render/tokenize the actual input before supplying its digest and counts. Hashes
and fixture declaration references are identity inputs, never native qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .model_manifest import _sha256
from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, items, number, record, text, unique

ENVELOPE_SCHEMA = "tensormeld/serving-request-envelope-v1"
REQUEST_SCHEMA = "tensormeld/rendered-text-request-v1"
IDENTITY_FIELDS = frozenset({
    "package_sha256", "model_manifest_sha256", "tokenizer_sha256",
    "chat_template_sha256", "realization_sha256", "numerical_plan_sha256",
    "state_layout_sha256", "prefill_plan_sha256", "adapter_capabilities_sha256",
})
DECLARATION_PURPOSES = frozenset({
    "representative_correctness", "boundary_limits", "rendered_input_contract",
})
FEATURES = {
    "text": True, "greedy": True, "sampling": False, "sse": False, "tools": False,
}


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _integer(value: Any, label: str, maximum: int = (1 << 63) - 1) -> int:
    result = int(number(value, label, 1, True))
    if result > maximum:
        raise ValidationError(f"{label}: exceeds supported bound {maximum}")
    return result


@dataclass(frozen=True)
class ServingRequestIdentity:
    hashes: tuple[tuple[str, str], ...]
    runtime_identity_sha256: tuple[str, ...]

    @classmethod
    def parse(cls, data: Any) -> "ServingRequestIdentity":
        root = record(data, "serving identity", set(IDENTITY_FIELDS) | {"runtime_identity_sha256"})
        runtime = [_sha256(value, "runtime_identity_sha256[]") for value in items(
            root["runtime_identity_sha256"], "runtime_identity_sha256", 128, 1,
        )]
        unique(runtime, "runtime_identity_sha256")
        return cls(tuple((key, _sha256(root[key], key)) for key in sorted(IDENTITY_FIELDS)),
                   tuple(sorted(runtime)))

    def as_record(self) -> dict[str, Any]:
        return {**dict(self.hashes), "runtime_identity_sha256": list(self.runtime_identity_sha256)}


@dataclass(frozen=True)
class ServingRequestEnvelope:
    identity: ServingRequestIdentity
    context_windows: tuple[int, ...]
    max_input_tokens: int
    max_output_tokens: int
    max_input_bytes: int
    declarations: tuple[tuple[str, str], ...]
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> "ServingRequestEnvelope":
        root = record(data, "serving request envelope", {
            "envelope_schema", "identity", "limits", "features", "cache_modes",
            "declaration_references", "native_qualified", "executable",
        })
        if root["envelope_schema"] != ENVELOPE_SCHEMA:
            raise ValidationError(f"envelope_schema: expected {ENVELOPE_SCHEMA}")
        if root["native_qualified"] is not False or root["executable"] is not False:
            raise ValidationError("request envelope declarations cannot self-promote qualification or execution")
        features = record(root["features"], "features", set(FEATURES))
        if any(type(features[key]) is not bool or features[key] is not expected
               for key, expected in FEATURES.items()):
            raise ValidationError("initial envelope supports only nonstreaming greedy text")
        if root["cache_modes"] != ["disabled"]:
            raise ValidationError("initial envelope requires cache disabled")
        identity = ServingRequestIdentity.parse(root["identity"])
        limits = record(root["limits"], "limits", {
            "context_windows", "max_input_tokens", "max_output_tokens", "max_input_bytes", "concurrency",
        })
        if type(limits["concurrency"]) is not int or limits["concurrency"] != 1:
            raise ValidationError("initial envelope requires concurrency one")
        windows = [_integer(value, "context_windows[]") for value in items(
            limits["context_windows"], "context_windows", 128, 1,
        )]
        unique(windows, "context_windows")
        max_input = _integer(limits["max_input_tokens"], "max_input_tokens")
        max_output = _integer(limits["max_output_tokens"], "max_output_tokens")
        max_bytes = _integer(limits["max_input_bytes"], "max_input_bytes", MAX_INPUT_BYTES)
        if max(max_input, max_output) > max(windows):
            raise ValidationError("input/output limits exceed every declared context window")
        declarations = []
        for i, value in enumerate(items(root["declaration_references"], "declaration_references", 256, 3)):
            declaration = record(value, f"declaration_references[{i}]", {
                "kind", "purpose", "provenance", "record_sha256",
            })
            if (declaration["kind"] != "request_family_declaration"
                    or declaration["provenance"] != "fixture"):
                raise ValidationError("declaration reference is not native qualification or legacy replay evidence")
            purpose = text(declaration["purpose"], "declaration.purpose")
            if purpose not in DECLARATION_PURPOSES:
                raise ValidationError("unsupported request-family declaration purpose")
            declarations.append((purpose, _sha256(declaration["record_sha256"], "declaration.record_sha256")))
        unique([digest for _, digest in declarations], "declaration_references.record_sha256")
        if {purpose for purpose, _ in declarations} != DECLARATION_PURPOSES:
            raise ValidationError("request family requires correctness, boundary and rendered-input declarations")
        envelope = cls(identity, tuple(sorted(windows)), max_input, max_output, max_bytes,
                       tuple(sorted(declarations)), "")
        return cls(identity, envelope.context_windows, max_input, max_output, max_bytes,
                   envelope.declarations, _fingerprint(envelope.as_record()))

    def as_record(self) -> dict[str, Any]:
        return {
            "envelope_schema": ENVELOPE_SCHEMA, "identity": self.identity.as_record(),
            "limits": {"context_windows": list(self.context_windows),
                       "max_input_tokens": self.max_input_tokens, "max_output_tokens": self.max_output_tokens,
                       "max_input_bytes": self.max_input_bytes, "concurrency": 1},
            "features": dict(FEATURES), "cache_modes": ["disabled"],
            "declaration_references": [
                {"kind": "request_family_declaration", "purpose": purpose,
                 "provenance": "fixture", "record_sha256": digest}
                for purpose, digest in self.declarations
            ],
            "native_qualified": False, "executable": False,
        }


def preflight_serving_request(
    envelope: ServingRequestEnvelope, *, request: Any, current_identity: Any,
) -> dict[str, Any]:
    """Compare an adapter-rendered descriptor with a durable declaration.

    The caller supplies exact rendered token counts/digest and current identity from
    trusted local inspection. This function does not tokenize, attest to that inspection,
    reserve resources, run a canary, establish native correctness, or invoke a backend.
    """
    if not isinstance(envelope, ServingRequestEnvelope):
        raise ValidationError("expected ServingRequestEnvelope")
    validated = ServingRequestEnvelope.parse(envelope.as_record())
    if validated.fingerprint != envelope.fingerprint:
        raise ValidationError("serving request envelope fingerprint mismatch")
    current = ServingRequestIdentity.parse(current_identity)
    raw = record(request, "rendered request", {
        "request_schema", "identity", "rendered_input_sha256", "input_bytes", "input_tokens",
        "context_window_tokens", "max_output_tokens", "concurrency", "task", "sampler", "stream", "cache_mode",
    })
    if raw["request_schema"] != REQUEST_SCHEMA:
        raise ValidationError(f"request_schema: expected {REQUEST_SCHEMA}")
    requested = ServingRequestIdentity.parse(raw["identity"])
    digest = _sha256(raw["rendered_input_sha256"], "rendered_input_sha256")
    numeric = {key: _integer(raw[key], key) for key in (
        "input_bytes", "input_tokens", "context_window_tokens", "max_output_tokens", "concurrency",
    )}
    for key in ("task", "sampler", "cache_mode"):
        text(raw[key], key)
    if type(raw["stream"]) is not bool:
        raise ValidationError("stream: expected boolean")
    reasons = []
    expected_identity = validated.identity.as_record()
    for scope, identity in (("CURRENT", current), ("REQUEST", requested)):
        values = identity.as_record()
        for key in sorted(expected_identity):
            if values[key] != expected_identity[key]:
                reasons.append(f"{scope}_{key.removesuffix('_sha256').upper()}_MISMATCH")
    checks = (
        (raw["task"] == "text_completion", "UNSUPPORTED_TASK"),
        (raw["sampler"] == "greedy", "UNSUPPORTED_SAMPLER"),
        (raw["stream"] is False, "STREAMING_NOT_DECLARED"),
        (raw["cache_mode"] == "disabled", "CACHE_MODE_NOT_DECLARED"),
        (numeric["concurrency"] == 1, "CONCURRENCY_NOT_DECLARED"),
        (numeric["context_window_tokens"] in validated.context_windows, "CONTEXT_WINDOW_NOT_DECLARED"),
        (numeric["input_bytes"] <= validated.max_input_bytes, "INPUT_BYTE_LIMIT_EXCEEDED"),
        (numeric["input_tokens"] <= validated.max_input_tokens, "INPUT_TOKEN_LIMIT_EXCEEDED"),
        (numeric["max_output_tokens"] <= validated.max_output_tokens, "OUTPUT_TOKEN_LIMIT_EXCEEDED"),
        (numeric["input_tokens"] + numeric["max_output_tokens"] <= numeric["context_window_tokens"],
         "INPUT_PLUS_OUTPUT_EXCEEDS_CONTEXT"),
    )
    reasons.extend(code for ok, code in checks if not ok)
    canonical_request = {
        **raw, **numeric, "identity": requested.as_record(), "rendered_input_sha256": digest,
    }
    core = {
        "result_schema": "tensormeld/serving-request-preflight-v1",
        "envelope_sha256": validated.fingerprint, "request_sha256": _fingerprint(canonical_request),
        "current_identity_sha256": _fingerprint(current.as_record()),
        "status": "DECLARATION_APPLICABLE" if not reasons else "DECLARATION_INAPPLICABLE",
        "applies": not reasons, "reasons": reasons, "declaration_only": True,
        "native_qualified": False, "inference_request_authorized": False,
        "resource_admission_performed": False, "launch_canary_performed": False, "executable": False,
    }
    core["result_sha256"] = _fingerprint(core)
    return core


def load_serving_request_envelope(path: str | Path) -> ServingRequestEnvelope:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("serving request envelope exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError("invalid serving request envelope JSON") from exc
    return ServingRequestEnvelope.parse(value)
