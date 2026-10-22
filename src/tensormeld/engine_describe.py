"""Engine self-description for heterogeneous placement.

An engine (any backend: one per device type) answers ``--describe-json`` with
``tensormeld/engine-describe-v1``:

* ``engine``: name, revision, executable sha256 (checked against the approved
  artifact), ``backend`` (``hip``, ``cuda``, ``cpu``, ...);
* ``capabilities``: the part capabilities it can execute on one device of its
  backend (the ``requires`` strings of ``tensormeld/model-parts-v1``);
* ``group_modes``: modes in which several instances run one model as a sharded
  group (world size, capabilities, transports);
* ``roles``: which deployment roles a process can take
  (``whole``, ``prefill`` [needs state export], ``decode`` [needs state import],
  ``pipeline-stage`` [runs a layer range on an incoming stream], ``parts-host``,
  ``parts-worker``);
* ``state``: request-state export/import (format id, transports, exactness);
* ``parts_rpc``: the per-layer branch protocol for ``parts-worker`` (transport,
  max rows per call, activation encodings);
* ``models``: for each model file it can serve, the parts descriptor.

TensorMeld runs the probe as a bounded child process of an approved
executable (fixed argv, no shell, empty environment except PATH/SYSTEMROOT,
size and time limits), validates it, and derives device capabilities and the
model descriptor from it.  A description is evidence about intent, not a
qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .hetero_model import ModelParts, parse_model_parts
from .schema import ValidationError, _no_duplicates, number, record, text

ENGINE_DESCRIBE_SCHEMA = "tensormeld/engine-describe-v1"
ROLES = frozenset({"whole", "prefill", "decode", "pipeline-stage", "parts-host", "parts-worker"})
MAX_DESCRIBE_BYTES = 4 * 1024 * 1024
MAX_ITEMS = 256


@dataclass(frozen=True)
class GroupMode:
    id: str
    world: int
    capabilities: tuple[str, ...]
    transports: tuple[str, ...]


@dataclass(frozen=True)
class EngineDescribe:
    name: str
    revision: str
    executable_sha256: str
    backend: str
    capabilities: tuple[str, ...]
    group_modes: tuple[GroupMode, ...]
    roles: tuple[str, ...]
    state_export: bool
    state_import: bool
    state_format: str
    state_transports: tuple[str, ...]
    state_exact: bool
    parts_rpc_transports: tuple[str, ...]
    parts_rpc_max_rows: int
    parts_rpc_encodings: tuple[str, ...]
    models: tuple[ModelParts, ...]
    fingerprint: str

    def model(self, model_id: str) -> ModelParts:
        for m in self.models:
            if m.model_id == model_id:
                return m
        raise ValidationError(f"engine does not describe model {model_id!r}")

    def can_run(self, model: ModelParts) -> list[str]:
        """Capabilities the model needs that this engine lacks (empty = can run it whole)."""
        have = set(self.capabilities)
        return sorted({r for p in model.parts for r in p.requires if not p.optional} - have)


def _strings(value, where: str, allowed: frozenset | None = None) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise ValidationError(f"{where}: expected a list of at most {MAX_ITEMS} strings")
    out = tuple(sorted({text(v, f"{where}[]") for v in value}))
    if allowed is not None and set(out) - allowed:
        raise ValidationError(f"{where}: unknown {sorted(set(out) - allowed)}")
    return out


def parse_engine_describe(data, *, expected_executable_sha256: str | None = None) -> EngineDescribe:
    r = record(data, "engine describe",
               {"schema", "engine", "capabilities", "roles", "models"},
               {"group_modes", "state", "parts_rpc", "argv_template", "notes"})
    if r["schema"] != ENGINE_DESCRIBE_SCHEMA:
        raise ValidationError(f"schema: expected {ENGINE_DESCRIBE_SCHEMA}")
    e = record(r["engine"], "engine", {"name", "revision", "executable_sha256", "backend"})
    sha = text(e["executable_sha256"], "engine.executable_sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValidationError("engine.executable_sha256: expected 64 hex characters")
    if expected_executable_sha256 and sha != expected_executable_sha256.lower():
        raise ValidationError("engine.executable_sha256: does not match the approved artifact")
    modes = []
    for i, raw in enumerate(r.get("group_modes", [])):
        g = record(raw, f"group_modes[{i}]", {"id", "world", "capabilities", "transports"})
        modes.append(GroupMode(text(g["id"], "group_modes.id"), int(number(g["world"], "world", 2, True)),
                               _strings(g["capabilities"], "group_modes.capabilities"),
                               _strings(g["transports"], "group_modes.transports")))
    state = record(r.get("state", {}), "state", set(), {"export", "import", "format", "transports", "exact"})
    for key in ("export", "import", "exact"):
        if key in state and type(state[key]) is not bool:
            raise ValidationError(f"state.{key}: expected boolean")
    if (state.get("export") or state.get("import")) and not state.get("format"):
        raise ValidationError("state.format: required when export/import is declared")
    rpc = record(r.get("parts_rpc", {}), "parts_rpc", set(), {"transports", "max_rows", "encodings"})
    roles = _strings(r["roles"], "roles", ROLES)
    if "parts-worker" in roles and not rpc.get("transports"):
        raise ValidationError("parts_rpc.transports: required for the parts-worker role")
    if "prefill" in roles and not state.get("export"):
        raise ValidationError("state.export: required for the prefill role")
    if "decode" in roles and not state.get("import"):
        raise ValidationError("state.import: required for the decode role")
    raw_models = r["models"]
    if not isinstance(raw_models, list) or not 1 <= len(raw_models) <= 16:
        raise ValidationError("models: expected 1..16 entries")
    models = tuple(parse_model_parts(m) for m in raw_models)
    digest = hashlib.sha256(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return EngineDescribe(
        name=text(e["name"], "engine.name"), revision=text(e["revision"], "engine.revision"),
        executable_sha256=sha, backend=text(e["backend"], "engine.backend"),
        capabilities=_strings(r["capabilities"], "capabilities"), group_modes=tuple(modes), roles=roles,
        state_export=bool(state.get("export", False)), state_import=bool(state.get("import", False)),
        state_format=str(state.get("format", "")), state_transports=_strings(state.get("transports", []), "state.transports"),
        state_exact=bool(state.get("exact", False)),
        parts_rpc_encodings=_strings(rpc.get("encodings", []), "parts_rpc.encodings"),
        parts_rpc_transports=_strings(rpc.get("transports", []), "parts_rpc.transports"),
        parts_rpc_max_rows=int(number(rpc.get("max_rows", 1), "parts_rpc.max_rows", 1, True)),
        models=models, fingerprint=digest)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_describe(executable: Path, *, approved_sha256: str, argv: tuple[str, ...] = ("--describe-json",),
                 timeout_s: float = 60.0, prefix: tuple[str, ...] = ()) -> EngineDescribe:
    """Run an approved engine executable's describe probe and validate its answer.

    ``prefix`` lets a test run a script through an interpreter; the hash is
    always checked on ``executable`` itself before it runs.
    """
    executable = Path(executable)
    actual = file_sha256(executable)
    if actual != approved_sha256.lower():
        raise ValidationError(f"{executable.name}: sha256 {actual[:12]}... is not the approved artifact")
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}
    proc = subprocess.run([*prefix, str(executable), *argv], capture_output=True, timeout=timeout_s, env=env,
                          stdin=subprocess.DEVNULL, check=False)
    if proc.returncode != 0:
        raise ValidationError(f"describe probe exited {proc.returncode}: {proc.stderr[-400:].decode(errors='replace')}")
    if len(proc.stdout) > MAX_DESCRIBE_BYTES:
        raise ValidationError("describe probe output above the bound")
    data = json.loads(proc.stdout, object_pairs_hook=_no_duplicates)
    return parse_engine_describe(data, expected_executable_sha256=actual)


def device_supports(describes: list[EngineDescribe], backend: str) -> tuple[str, ...]:
    """Union of capabilities of the engines that run on ``backend`` (what a device can host)."""
    caps: set[str] = set()
    for d in describes:
        if d.backend == backend:
            caps |= set(d.capabilities)
    return tuple(sorted(caps))
