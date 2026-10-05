"""Exact llama.cpp build/package identity.

A source revision alone is not enough to bridge qualification performed with llama-cli
to persistent execution through llama-server. This contract binds the exact sibling
executables, their identical observed build metadata, and any backend-library artifacts
that materially participate in the runtime.

The package proves build provenance/identity only. It does not prove that llama-server
request semantics are equivalent to the llama-cli E3 correctness trial.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .llamacpp_probe import LLAMACPP_PINNED_COMMIT
from .model_manifest import _sha256
from .schema import ValidationError, items, record, text, unique

PACKAGE_SCHEMA = "tensormeld/llamacpp-build-package-v1"
MAX_BACKEND_LIBS = 64


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validated_probe(probe: Any, where: str) -> dict[str, Any]:
    if not isinstance(probe, dict):
        raise ValidationError(f"{where}: expected probe object")
    if probe.get("probe_schema") != "tensormeld/llamacpp-probe-v1":
        raise ValidationError(f"{where}: probe schema mismatch")
    if probe.get("engine") != "llama.cpp":
        raise ValidationError(f"{where}: engine mismatch")
    if probe.get("pinned_source_revision") != LLAMACPP_PINNED_COMMIT:
        raise ValidationError(f"{where}: source revision mismatch")
    artifact = _sha256(probe.get("artifact_sha256"), f"{where}.artifact_sha256")
    build = record(
        probe.get("build"),
        f"{where}.build",
        {"version", "build", "commit", "compiler", "target"},
    )
    commit = text(build["commit"], f"{where}.build.commit").lower()
    if not LLAMACPP_PINNED_COMMIT.startswith(commit):
        raise ValidationError(f"{where}: observed commit is not the pinned revision")
    canonical_build = {
        "version": text(build["version"], f"{where}.build.version"),
        "build": int(build["build"]),
        "commit": commit,
        "compiler": text(build["compiler"], f"{where}.build.compiler"),
        "target": text(build["target"], f"{where}.build.target"),
    }
    return {
        "artifact_sha256": artifact,
        "binary_name": text(probe.get("binary_name"), f"{where}.binary_name"),
        "build": canonical_build,
    }


@dataclass(frozen=True)
class LlamaCppBuildPackage:
    source_revision: str
    build: dict[str, Any]
    llama_cli_sha256: str
    llama_server_sha256: str
    backend_libraries: tuple[tuple[str, str], ...]
    fingerprint: str

    def as_record(self) -> dict[str, Any]:
        return {
            "package_schema": PACKAGE_SCHEMA,
            "source_revision": self.source_revision,
            "build": dict(self.build),
            "artifacts": {
                "llama_cli_sha256": self.llama_cli_sha256,
                "llama_server_sha256": self.llama_server_sha256,
                "backend_libraries": [
                    {"name": name, "sha256": sha}
                    for name, sha in self.backend_libraries
                ],
            },
            "package_sha256": self.fingerprint,
        }


def build_llamacpp_package_identity(
    *,
    cli_probe: dict[str, Any],
    server_probe: dict[str, Any],
    backend_libraries: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> LlamaCppBuildPackage:
    cli = _validated_probe(cli_probe, "cli_probe")
    server = _validated_probe(server_probe, "server_probe")
    if cli["build"] != server["build"]:
        raise ValidationError("llama-cli and llama-server build metadata differ")
    if cli["artifact_sha256"] == server["artifact_sha256"]:
        raise ValidationError(
            "llama-cli and llama-server must remain distinct artifact identities"
        )

    libs: list[tuple[str, str]] = []
    for i, raw in enumerate(items(
        list(backend_libraries),
        "backend_libraries",
        MAX_BACKEND_LIBS,
        0,
    )):
        lib = record(raw, f"backend_libraries[{i}]", {"name", "sha256"})
        name = text(lib["name"], f"backend_libraries[{i}].name")
        sha = _sha256(lib["sha256"], f"backend_libraries[{i}].sha256")
        libs.append((name, sha))
    unique([name for name, _ in libs], "backend_libraries.name")
    unique([sha for _, sha in libs], "backend_libraries.sha256")
    libs.sort()

    canonical = {
        "package_schema": PACKAGE_SCHEMA,
        "source_revision": LLAMACPP_PINNED_COMMIT,
        "build": cli["build"],
        "artifacts": {
            "llama_cli_sha256": cli["artifact_sha256"],
            "llama_server_sha256": server["artifact_sha256"],
            "backend_libraries": [
                {"name": name, "sha256": sha} for name, sha in libs
            ],
        },
    }
    return LlamaCppBuildPackage(
        LLAMACPP_PINNED_COMMIT,
        dict(cli["build"]),
        cli["artifact_sha256"],
        server["artifact_sha256"],
        tuple(libs),
        _canonical_sha256(canonical),
    )
