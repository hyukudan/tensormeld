from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .diagnostics import loopback
from .config_v2 import load_config
from .selection import resolve_candidates, resolve_runtime_candidates
from .runtime_observation import load_runtime_observation
from .planning_contract import load_planning_input
from .planner_v2 import plan_v2
from .migration import migrate_config
from .gguf_catalog import inspect_gguf
from .llamacpp_probe import probe_llamacpp
from .device_binding import bind_llamacpp_probe, load_llamacpp_binding, load_llamacpp_probe_report
from .llamacpp_selftest import (
    load_llamacpp_bound_result,
    self_test_llamacpp_backend,
)
from .model_manifest import load_model_manifest
from .adapter_contract import load_adapter_capabilities
from .runtime_model_manifest import (
    load_runtime_model_manifest,
    runtime_manifest_summary,
)
from .predictive_memory import (
    load_predictive_memory_profile,
    predictive_memory_summary,
)
from .llamacpp_package import load_llamacpp_package_identity
from .runtime_identity import load_runtime_identity
from .tensor_movability import (
    load_gguf_tensor_index,
    load_tensor_movability_profile,
    tensor_movability_summary,
)
from .planner import plan
from .probe import probe
from .schema import ValidationError, load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tensormeld",
        description=(
            "Control-plane prototype: inventory, policy/runtime resolution and advisory "
            "placement; NO GPU inference yet."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("probe", help="Read-only local hardware inventory")
    p.add_argument("--out", type=Path)
    p = subs.add_parser("validate", help="Validate one bounded scenario")
    p.add_argument("scenario", type=Path)
    p.add_argument("--out", type=Path)
    p = subs.add_parser("plan", help="Advisory whole-stage C1 decode plan, not executable")
    p.add_argument("scenario", type=Path)
    p.add_argument("--devices", nargs="+", help="Restrict candidate devices, e.g. strix-a strix-b")
    p.add_argument("--out", type=Path)
    p = subs.add_parser("validate-config", help="Validate a TensorMeld v2 installation config")
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path)
    p = subs.add_parser("select", help="Resolve legal v2 compute/coordinator candidates before placement")
    p.add_argument("config", type=Path)
    p.add_argument("--profile", help="Profile name; defaults to installation.default_profile")
    p.add_argument("--out", type=Path)
    p = subs.add_parser(
        "runtime-select",
        help="Intersect static policy with one advisory runtime snapshot; no reservation is created",
    )
    p.add_argument("config", type=Path)
    p.add_argument("runtime_observation", type=Path)
    p.add_argument("--profile", help="Profile name; defaults to installation.default_profile")
    p.add_argument("--out", type=Path)
    p = subs.add_parser("plan-v2", help="Bounded synthetic multi-node whole-block planner; not executable")
    p.add_argument("config", type=Path)
    p.add_argument("planning_input", type=Path)
    p.add_argument("--profile")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", type=Path)
    p = subs.add_parser("migrate-config", help="Explicit migration from the previous working title")
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--out", type=Path)
    p = subs.add_parser("inspect-gguf", help="Index trusted local GGUF shards using optional upstream gguf")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--trusted-local-file", action="store_true")
    p.add_argument("--out", type=Path)
    p = subs.add_parser("probe-llamacpp", help="Trusted-local no-model probe of the pinned llama.cpp revision")
    p.add_argument("binary", type=Path)
    p.add_argument("--trusted-local-binary", action="store_true")
    p.add_argument("--expected-sha256")
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--out", type=Path)
    p = subs.add_parser("bind-llamacpp-devices", help="Apply an explicit approved llama.cpp device mapping; outputs observed-not-ready runtime data")
    p.add_argument("config", type=Path)
    p.add_argument("probe_report", type=Path)
    p.add_argument("binding", type=Path)
    p.add_argument("--out", type=Path)
    p = subs.add_parser(
        "self-test-llamacpp-backend",
        help=(
            "Run pinned upstream test-backend-ops on one explicitly bound backend; "
            "successful evidence promotes only that runtime observation to ready"
        ),
    )
    p.add_argument("config", type=Path)
    p.add_argument("bound_result", type=Path)
    p.add_argument("test_binary", type=Path)
    p.add_argument("tensormeld_device_id")
    p.add_argument("--trusted-local-binary", action="store_true")
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--out", type=Path)
    p = subs.add_parser(
        "validate-runtime-manifest",
        help="Validate an exact runtime model/operator/memory manifest; no reservation is created",
    )
    p.add_argument("config", type=Path)
    p.add_argument("model_manifest", type=Path)
    p.add_argument("adapter_capabilities", type=Path)
    p.add_argument("runtime_manifest", type=Path)
    p.add_argument("--profile")
    p.add_argument("--out", type=Path)
    p = subs.add_parser(
        "validate-predictive-memory",
        help="Validate predictive physical-pool memory classes against an exact runtime manifest",
    )
    p.add_argument("config", type=Path)
    p.add_argument("model_manifest", type=Path)
    p.add_argument("adapter_capabilities", type=Path)
    p.add_argument("runtime_manifest", type=Path)
    p.add_argument("predictive_memory", type=Path)
    p.add_argument("--profile")
    p.add_argument("--out", type=Path)
    p = subs.add_parser(
        "validate-tensor-movability",
        help="Validate complete explicit tensor storage/movement evidence against exact model/build/runtime identities",
    )
    p.add_argument("config", type=Path)
    p.add_argument("model_manifest", type=Path)
    p.add_argument("tensor_index", type=Path)
    p.add_argument("adapter_capabilities", type=Path)
    p.add_argument("package_identity", type=Path)
    p.add_argument("movability", type=Path)
    p.add_argument("--runtime-identity", dest="runtime_identities", action="append", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p = subs.add_parser("bench-loopback", help="Loopback-only bounded TCP echo diagnostic")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out:
            inputs = [
                getattr(args, name, None)
                for name in (
                    "config", "runtime_observation", "planning_input", "scenario",
                    "source", "destination", "binary", "probe_report", "binding",
                    "bound_result", "test_binary", "model_manifest",
                    "adapter_capabilities", "runtime_manifest", "predictive_memory",
                    "tensor_index", "package_identity", "movability",
                )
            ]
            inputs += getattr(args, "paths", [])
            inputs += getattr(args, "runtime_identities", []) or []
            if any(
                isinstance(p, Path)
                and (
                    p.resolve() == args.out.resolve()
                    or (p.exists() and args.out.exists() and p.samefile(args.out))
                )
                for p in inputs
            ):
                raise ValidationError(
                    "--out must not overwrite an input file or migration destination"
                )
        code = 0
        if args.command == "probe":
            result = probe()
        elif args.command == "bench-loopback":
            result = loopback(args.iterations)
        elif args.command == "migrate-config":
            result = migrate_config(args.source, args.destination)
        elif args.command == "inspect-gguf":
            result = inspect_gguf(args.paths, trusted_local_file=args.trusted_local_file)
        elif args.command == "probe-llamacpp":
            result = probe_llamacpp(
                args.binary,
                trusted_local_binary=args.trusted_local_binary,
                expected_artifact_sha256=args.expected_sha256,
                timeout_s=args.timeout,
            )
        elif args.command == "bind-llamacpp-devices":
            config = load_config(args.config)
            probe_report = load_llamacpp_probe_report(args.probe_report)
            binding = load_llamacpp_binding(args.binding)
            result = bind_llamacpp_probe(config, probe_report, binding)
        elif args.command == "self-test-llamacpp-backend":
            config = load_config(args.config)
            bound_result = load_llamacpp_bound_result(args.bound_result)
            result = self_test_llamacpp_backend(
                args.test_binary,
                config=config,
                bound_result=bound_result,
                tensormeld_device_id=args.tensormeld_device_id,
                trusted_local_binary=args.trusted_local_binary,
                expected_artifact_sha256=args.expected_sha256,
                timeout_s=args.timeout,
            )
        elif args.command == "validate-runtime-manifest":
            config = load_config(args.config)
            model_manifest = load_model_manifest(args.model_manifest)
            adapter = load_adapter_capabilities(args.adapter_capabilities)
            runtime_manifest = load_runtime_model_manifest(
                args.runtime_manifest,
                config=config,
                model=model_manifest,
                adapter=adapter,
                profile_name=args.profile,
            )
            result = runtime_manifest_summary(runtime_manifest)
        elif args.command == "validate-predictive-memory":
            config = load_config(args.config)
            model_manifest = load_model_manifest(args.model_manifest)
            adapter = load_adapter_capabilities(args.adapter_capabilities)
            runtime_manifest = load_runtime_model_manifest(
                args.runtime_manifest,
                config=config,
                model=model_manifest,
                adapter=adapter,
                profile_name=args.profile,
            )
            predictive = load_predictive_memory_profile(
                args.predictive_memory,
                config=config,
                runtime_manifest=runtime_manifest,
            )
            result = predictive_memory_summary(predictive)
        elif args.command == "validate-tensor-movability":
            config = load_config(args.config)
            model_manifest = load_model_manifest(args.model_manifest)
            tensor_index = load_gguf_tensor_index(args.tensor_index)
            adapter = load_adapter_capabilities(args.adapter_capabilities)
            package = load_llamacpp_package_identity(args.package_identity)
            runtime_identities = tuple(
                load_runtime_identity(path) for path in args.runtime_identities
            )
            movability = load_tensor_movability_profile(
                args.movability,
                config=config,
                model=model_manifest,
                tensor_index=tensor_index,
                adapter=adapter,
                package=package,
                runtime_identities=runtime_identities,
            )
            result = tensor_movability_summary(movability)
        elif args.command == "runtime-select":
            config = load_config(args.config)
            observation = load_runtime_observation(args.runtime_observation)
            result = resolve_runtime_candidates(config, observation, args.profile)
            code = 0 if result["status"] == "RUNTIME_CANDIDATES_READY" else 2
        elif args.command == "plan-v2":
            config = load_config(args.config)
            model = load_planning_input(args.planning_input, config)
            result = plan_v2(config, model, args.profile, args.top_k)
            code = 3 if result["status"] == "SEARCH_INCOMPLETE" else (2 if result["best"] is None else 0)
        elif args.command in ("validate-config", "select"):
            config = load_config(args.config)
            if args.command == "validate-config":
                result = {
                    "valid": True,
                    "config_schema": "tensormeld/v2",
                    "config_sha256": config.fingerprint,
                    "nodes": len(config.nodes),
                    "devices": len(config.devices),
                    "profiles": [p.name for p in config.profiles],
                }
            else:
                result = resolve_candidates(config, args.profile)
        else:
            scenario = load(args.scenario)
            if args.command == "validate":
                result = {
                    "valid": True,
                    "scenario_sha256": scenario.fingerprint,
                    "provenance": scenario.provenance,
                }
            else:
                result = plan(scenario, args.devices)
                code = 2 if result["best"] is None else 0
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        return code
    except (ValidationError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
