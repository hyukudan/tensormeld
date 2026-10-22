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
from .replica_router import load_router_config, serve_replicas
from .replica_bench import documents as bench_documents, run_bench
from .protocol_bench import run_protocol
from .probe import probe
from .schema import ValidationError, load


def _serve_replicas(path: Path) -> dict:
    import signal
    import threading

    config = load_router_config(path)
    stop = threading.Event()
    previous = {}
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        return serve_replicas(config, stop, log=sys.stderr)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv[:1] == ["link-probe"]:
        # Own argument grammar (serve | run); the same module also runs as a standalone file on a node.
        from .link_probe import main as link_probe_main

        try:
            return link_probe_main(raw_argv[1:])
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
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
    p = subs.add_parser(
        "serve-replicas",
        help="Least-loaded HTTP router over whole-model engine replicas; runs until SIGINT/SIGTERM",
    )
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path)
    p = subs.add_parser(
        "plan-hetero",
        help="Place a model's parts across unlike devices (descriptors in, ranked modes out); not executable",
    )
    p.add_argument("model_parts", type=Path)
    p.add_argument("hardware", type=Path)
    p.add_argument("--prompt", type=int, default=8192)
    p.add_argument("--context", type=int)
    p.add_argument("--output", type=int, default=256)
    p.add_argument("--rows", type=int, default=1, help="rows per decode step (batch or verify window)")
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--draft", action="store_true", help="count the optional drafter in memory")
    p.add_argument("--objective", choices=("latency", "decode", "prefill", "throughput"), default="latency")
    p.add_argument("--unavailable", action="append", default=[], help="device id treated as unavailable")
    p.add_argument("--link", action="append", default=[], help="what-if: LINK_ID=BYTES_PER_S:LATENCY_US")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mode", choices=("auto", "dense+hot"), default="auto",
                   help="dense+hot: also plan dense parts on the parts-host with hot experts per --placement")
    p.add_argument("--placement", type=Path, help="tensormeld/hot-expert-placement-v1 file (with --mode dense+hot)")
    p.add_argument("--placement-device", action="append", default=[],
                   help="map a placement device (or SERVER/NODE) to a hardware device: NAME=HW_DEVICE_ID")
    p.add_argument("--host-copy-rate", type=float, default=25e9,
                   help="declared host->device bytes/s for expert-cache misses (dense+hot)")
    p.add_argument("--prefill-local-max-tokens", type=int,
                   help="length routing: prompts (minus a prefix-cache hit) up to N tokens prefill on the decode "
                        "placement, longer ones on the prefill-decode offload placement (default: measured or model)")
    p.add_argument("--ttft-receipt", type=Path,
                   help="tensormeld/ttft-curve-v1: measured end-to-end TTFT per prompt length for both routes")
    p.add_argument("--out", type=Path)
    p = subs.add_parser(
        "deploy-pipeline",
        help="Launch a hetero-pipeline plan as FRONT + TAIL engine processes (or --dry-run: print the manifests)",
    )
    p.add_argument("plan", type=Path, help="plan-hetero --out JSON")
    p.add_argument("model_parts", type=Path)
    p.add_argument("hardware", type=Path)
    p.add_argument("inventory", type=Path, help="tensormeld/pipeline-inventory-v1")
    p.add_argument("--candidate", help="candidate name (default: the plan's best)")
    p.add_argument("--unavailable", action="append", default=[], help="as given to plan-hetero")
    p.add_argument("--link", action="append", default=[], help="as given to plan-hetero")
    p.add_argument("--dry-run", action="store_true", help="validate and print the manifests; start nothing")
    p.add_argument("--on-failure", choices=("stop", "fallback"), default="stop",
                   help="when a stage process dies: stop, or start the fallback without the lost state")
    p.add_argument("--retire-file", type=Path, help="switch to the fallback when this file appears")
    p.add_argument("--run-seconds", type=float, help="stop after this many seconds (tests, short GPU turns)")
    p.add_argument("--out", type=Path)
    subs.add_parser("link-probe", help="Measure a TCP link as link evidence: link-probe serve|run ... (token-authenticated)")
    p = subs.add_parser(
        "bench-replicas",
        help="Concurrent multi-turn chat load against an OpenAI-compatible endpoint (engine or router)",
    )
    p.add_argument("--url", required=True)
    p.add_argument("--users", type=int, default=1)
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--doc", dest="docs", action="append", type=Path, required=True)
    p.add_argument("--doc-chars", type=int, default=6000)
    p.add_argument("--doc-seed", type=int, default=0)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--stagger-s", type=float, default=0.0)
    p.add_argument("--model", default="default")
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--timeout-s", type=float, default=1800.0)
    p.add_argument("--label", default="")
    p.add_argument("--out", type=Path)
    p = subs.add_parser("bench-protocol", help="Dry-run or explicitly execute a private fixed-corpus streaming benchmark")
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--mode", choices=("short", "long"), default="short")
    p.add_argument("--levels", default="1,4,6")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--run-id", default="campaign-1")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--url")
    p.add_argument("--model", default="default")
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--identity", type=Path)
    p.add_argument("--timeout-s", type=float, default=900)
    p.add_argument("--budget-s", type=float, default=900)
    p.add_argument("--label", default="")
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
                    "model_parts", "hardware", "plan", "inventory", "placement",
                    "corpus", "identity", "api_key_file",
                )
            ]
            inputs += getattr(args, "paths", [])
            inputs += getattr(args, "runtime_identities", []) or []
            inputs += getattr(args, "docs", []) or []
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
        elif args.command == "serve-replicas":
            result = _serve_replicas(args.config)
        elif args.command == "bench-replicas":
            key = args.api_key_file.read_text(encoding="ascii").strip() if args.api_key_file else None
            result = run_bench(
                args.url, args.users, args.turns, args.max_tokens,
                bench_documents(args.docs, args.users, args.doc_chars, args.doc_seed),
                model=args.model, api_key=key, timeout_s=args.timeout_s, label=args.label,
                stream=not args.no_stream, stagger_s=args.stagger_s,
            )
        elif args.command == "bench-protocol":
            key = args.api_key_file.read_text(encoding="ascii").strip() if args.execute and args.api_key_file else None
            identity = json.loads(args.identity.read_text(encoding="utf-8")) if args.identity else None
            if identity is not None and not isinstance(identity, dict):
                raise ValidationError("identity must be an object")
            result = run_protocol(
                args.corpus, mode=args.mode, url=args.url, model=args.model,
                levels=tuple(int(n) for n in args.levels.split(",")), repeats=args.repeats,
                run_id=args.run_id, execute=args.execute, api_key=key,
                timeout_s=args.timeout_s, budget_s=args.budget_s, label=args.label, identity=identity,
            )
            code = 2 if args.execute and not result["valid"] else 0
        elif args.command == "bench-loopback":
            result = loopback(args.iterations)
        elif args.command == "deploy-pipeline":
            result = _deploy_pipeline(args)
            code = 0 if result.get("final_state", result.get("status")) in ("DRY_RUN", "serving", "fallback") else 2
        elif args.command == "plan-hetero":
            result = _plan_hetero(args)
            code = 2 if result["best"] is None else 0
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


def _hardware_with_overrides(args):
    from .hetero_hardware import load_hardware

    hardware = load_hardware(args.hardware)
    links = {}
    for item in args.link:
        try:
            link_id, rest = item.split("=", 1)
            rate, latency = rest.split(":", 1)
            links[link_id] = (int(float(rate)), float(latency))
        except ValueError as exc:
            raise ValidationError(f"--link {item!r}: expected LINK_ID=BYTES_PER_S:LATENCY_US") from exc
    if args.unavailable or links:
        hardware = hardware.with_overrides({d: "unavailable" for d in args.unavailable}, links)
    return hardware


def _deploy_pipeline(args) -> dict:
    import secrets as _secrets
    import signal
    import threading
    import time as _time

    from .hetero_model import load_model_parts
    from .pipeline_controller import PipelineController, engine_nodes, launchers_for, resolve_describes
    from .pipeline_deploy import build_deployment, load_inventory, load_plan

    plan = load_plan(args.plan)
    model = load_model_parts(args.model_parts)
    hardware = _hardware_with_overrides(args)
    inventory = load_inventory(args.inventory)
    launchers = launchers_for(inventory)
    if args.dry_run:
        describes = resolve_describes(inventory, launchers, probe=False)
        deployment = build_deployment(plan, model, hardware, inventory, candidate=args.candidate, describes=describes)
        return {"status": "DRY_RUN", **deployment.summary()}
    describes = resolve_describes(inventory, launchers, probe=True, engine_nodes=engine_nodes(inventory, hardware))
    deployment_id = _time.strftime("pl-%Y%m%d-%H%M%S-") + _secrets.token_hex(2)
    deployment = build_deployment(plan, model, hardware, inventory, candidate=args.candidate,
                                  describes=describes, deployment_id=deployment_id)
    controller = PipelineController(deployment, launchers, on_failure=args.on_failure,
                                    retire_file=args.retire_file, log=sys.stderr)
    stop = threading.Event()
    previous = {}
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.signal(signum, lambda *_: stop.set())
    try:
        controller.start()
        print(f"tensormeld pipeline: serving at {controller.url()}", file=sys.stderr, flush=True)
        deadline = None if args.run_seconds is None else _time.monotonic() + args.run_seconds
        while not stop.is_set() and controller.state not in ("failed", "stopped"):
            if deadline is not None and _time.monotonic() >= deadline:
                break
            stop.wait(0.5)
        return controller.stop()
    except ValidationError:
        if controller.state != "failed":
            controller.stop()
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _plan_hetero(args) -> dict:
    from .hetero_model import load_model_parts
    from .hetero_planner import Request, plan_hetero
    from .prefill_route import load_ttft_curves

    model = load_model_parts(args.model_parts)
    hardware = _hardware_with_overrides(args)
    request = Request(prompt_tokens=args.prompt, context_tokens=args.context or args.prompt,
                      output_tokens=args.output, decode_rows=args.rows, slots=args.slots,
                      with_draft=args.draft, objective=args.objective,
                      prefill_local_max_tokens=args.prefill_local_max_tokens)
    curves = load_ttft_curves(args.ttft_receipt) if args.ttft_receipt else None
    hot = None
    if args.mode == "dense+hot":
        if not args.placement:
            raise ValidationError("--mode dense+hot needs --placement FILE")
        from .hot_placement import HotInput, load_hot_placement

        overrides = []
        for item in args.placement_device:
            name, sep, device = item.partition("=")
            if not sep or not name or not device:
                raise ValidationError(f"--placement-device {item!r}: expected NAME=HW_DEVICE_ID")
            overrides.append((name, device))
        if not args.host_copy_rate > 0:
            raise ValidationError("--host-copy-rate: expected a positive rate")
        hot = HotInput(load_hot_placement(args.placement), tuple(overrides), int(args.host_copy_rate),
                       source=args.placement.name)
    elif args.placement or args.placement_device:
        raise ValidationError("--placement/--placement-device need --mode dense+hot")
    result = plan_hetero(model, hardware, request, top_k=args.top_k, hot=hot, ttft_curves=curves)
    result.pop("_best_assigns", None)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
