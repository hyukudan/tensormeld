"""Trusted control-plane import for bounded runtime availability snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates


def load_runtime_observation(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        raw = f.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("runtime observation exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid runtime observation JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("runtime observation: expected object")
    return value
