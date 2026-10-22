"""Canonical planner candidate identity shared by qualification and execution gates."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .config_v2 import Config
from .planning_contract import PlanningInput
from .schema import ValidationError


def validate_candidate_hash(
    config: Config,
    planning: PlanningInput,
    profile_name: str,
    candidate: dict[str, Any],
) -> str:
    if not isinstance(candidate, dict):
        raise ValidationError("planner candidate must be an object")
    core = dict(candidate)
    supplied = core.pop("plan_sha256", None)
    identity = {
        "config": config.fingerprint,
        "planning": planning.fingerprint,
        "profile": profile_name,
        "plan": core,
    }
    expected = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if supplied != expected:
        raise ValidationError(
            "candidate plan_sha256 does not match candidate contents"
        )
    return expected
