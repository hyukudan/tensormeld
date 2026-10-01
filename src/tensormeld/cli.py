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
                    "source", "destination",
                )
            ]
            inputs += getattr(args, "paths", [])
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
