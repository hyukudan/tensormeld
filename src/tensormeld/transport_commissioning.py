"""Private two-host commissioning plans and paired observation checks.

This module never launches a process or grants qualification/admission. Exact
artifact hashes are provenance, not evidence that a kernel or GPU executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, items, record, text, unique

PLAN_SCHEMA = "tensormeld/direct-gpu-commissioning-plan-v1"
RECEIPT_SCHEMA = "strix-direct-gpu-commissioning-v1"
MAX_RUNS = 128
MAX_PAYLOAD = 4095 * 4096
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
UUID = re.compile(r"[0-9a-f]{32}\Z")
ENDPOINT_FIELDS = {"local_endpoint_identity", "kernel_identity", "driver_identity", "driver_build_id", "abi_identity",
                   "gpu_uuid", "gpu_architecture", "gpu_runtime_version", "gpu_driver_version"}
RUN_FIELDS = {"run_id", "sender", "receiver", "physical_link_id", "connection_epoch", "sequence",
              "payload_bytes", "pattern_seed", "model_plan_id", "request_id", "generation",
              "boundary_id", "shape_fingerprint", "dtype_id"}
FLAGS = {"full_payload_verified", "controller_frames_complete", "gpu_acquire_complete",
         "consumer_complete", "receive_reposted", "local_tx_complete", "transport_ack", "consumer_credit"}
RECEIPT_FIELDS = (ENDPOINT_FIELDS | FLAGS | (RUN_FIELDS - {"sender", "receiver"}) |
                  {"schema", "campaign_id", "role", "executed", "production_qualified", "iterations",
                   "completed_count", "peer_endpoint_identity", "device_identity", "device_instance_generation",
                   "expected_payload_sha256", "observed_payload_sha256", "cleanup", "error",
                   "pool_binding_identity", "import_generation", "pattern_version"})
CONFIG_FIELDS = (ENDPOINT_FIELDS | (RUN_FIELDS - {"sender", "receiver"}) |
                 {"schema", "campaign_id", "role", "executed", "production_qualified",
                  "pattern_version", "expected_payload_sha256", "peer_endpoint_identity"})


def _uint(value, where, maximum=(1 << 64) - 1, minimum=1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValidationError(f"{where}: expected integer in [{minimum}, {maximum}]")
    return value


def _sha(value, where):
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValidationError(where + ": expected lowercase SHA-256")
    return value


def _uuid(value, where):
    if not isinstance(value, str) or UUID.fullmatch(value) is None or value == "0" * 32:
        raise ValidationError(where + ": expected nonzero lowercase 16-byte identity")
    return value


def _identifier(value, where):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValidationError(where + ": expected bounded native identifier")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _json(path):
    if not Path(path).is_file():
        raise ValidationError("JSON artifact must be a regular file")
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("JSON artifact exceeds 2 MiB")
    def reject_constant(_):
        raise ValidationError("nonfinite JSON")
    try:
        return json.loads(raw, object_pairs_hook=_no_duplicates,
                          parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError("invalid JSON artifact") from exc


def _artifact(raw, where):
    value = record(raw, where, {"path", "sha256"})
    supplied_path = value["path"]
    if not isinstance(supplied_path, str) or not supplied_path or len(supplied_path) > 4096:
        raise ValidationError(where + ": invalid artifact path")
    path = Path(supplied_path)
    if not path.is_absolute():
        raise ValidationError(where + ": absolute path required")
    path = path.resolve(strict=True)
    expected = _sha(value["sha256"], where + ".sha256")
    if not path.is_file():
        raise ValidationError(where + ": regular file required")
    before = path.stat()
    if not 0 < before.st_size <= MAX_ARTIFACT_BYTES:
        raise ValidationError(where + ": artifact extent outside bounds")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValidationError(where + ": artifact identity mismatch")
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValidationError(where + ": artifact changed while hashing")
    return path


def _private(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValidationError("private output must be absolute")
    resolved = path.resolve()
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValidationError("private output must remain outside Git")
    return resolved


def expected_shape_fingerprint(campaign_id, run):
    canonical = (campaign_id + "\n" + run["run_id"] + "\n1\n" + str(run["pattern_seed"]) +
                 "\n" + str(run["payload_bytes"]) + "\n" + str(run["sequence"]))
    return int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8], "little") or 1


def validate_plan(raw):
    root = record(raw, "commissioning plan", {"schema", "campaign_id", "nodes", "runs",
                                              "production_qualified", "launch_authorized"})
    if root["schema"] != PLAN_SCHEMA:
        raise ValidationError("unsupported commissioning plan")
    _identifier(root["campaign_id"], "campaign_id")
    if root["production_qualified"] is not False or root["launch_authorized"] is not False:
        raise ValidationError("commissioning plan cannot self-promote")
    nodes = {}
    for value in items(root["nodes"], "nodes", 2, 2):
        node = record(value, "node", {"id", "expected", "artifacts"})
        node_id = text(node["id"], "node.id")
        if node_id in nodes:
            raise ValidationError("duplicate node")
        endpoint = record(node["expected"], "expected endpoint", ENDPOINT_FIELDS)
        _uuid(endpoint["local_endpoint_identity"], "local_endpoint_identity")
        _uuid(endpoint["gpu_uuid"], "gpu_uuid")
        for field in ("kernel_identity", "driver_identity", "abi_identity"):
            text(endpoint[field], field)
        build_id = text(endpoint["driver_build_id"], "driver_build_id")
        if re.fullmatch(r"[0-9a-f]{2,128}", build_id) is None or len(build_id) % 2:
            raise ValidationError("invalid GNU build ID")
        if endpoint["driver_identity"] != "thunderbolt_stream:gnu-build-id:" + build_id:
            raise ValidationError("driver build identity mismatch")
        if endpoint["gpu_architecture"] != "gfx1151":
            raise ValidationError("initial commissioning profile requires observed gfx1151")
        for field in ("gpu_runtime_version", "gpu_driver_version"):
            _uint(endpoint[field], field, (1 << 31) - 1)
        artifacts = record(node["artifacts"], "node artifacts",
                           {"harness_binary", "source_record", "kernel_build_record"})
        resolved = {field: _artifact(artifact, node_id + "." + field) for field, artifact in artifacts.items()}
        kernel = _json(resolved["kernel_build_record"])
        if not isinstance(kernel, dict) or kernel.get("schema") != "strix-private-full-kernel-build-v1":
            raise ValidationError("full private kernel build record required")
        modules = kernel.get("modules")
        if not isinstance(modules, dict) or not isinstance(modules.get("thunderbolt_stream"), dict):
            raise ValidationError("stream module build record required")
        module = modules["thunderbolt_stream"]
        if kernel.get("kernel_release") != endpoint["kernel_identity"] or module.get("gnu_build_id") != build_id:
            raise ValidationError("expected driver/kernel differs from pinned build record")
        _sha(module.get("sha256"), "stream module sha256")
        nodes[node_id] = node
    if len({node["expected"]["local_endpoint_identity"] for node in nodes.values()}) != 2:
        raise ValidationError("endpoints must differ")
    if len({node["expected"]["gpu_uuid"] for node in nodes.values()}) != 2:
        raise ValidationError("two-host GPU identities must differ")
    run_ids, epochs = [], []
    for raw_run in items(root["runs"], "runs", MAX_RUNS):
        run = record(raw_run, "run", RUN_FIELDS)
        run_ids.append(_identifier(run["run_id"], "run_id"))
        if run["sender"] not in nodes or run["receiver"] not in nodes or run["sender"] == run["receiver"]:
            raise ValidationError("run requires two known distinct nodes")
        text(run["physical_link_id"], "physical_link_id")
        for field in RUN_FIELDS - {"run_id", "sender", "receiver", "physical_link_id", "payload_bytes", "pattern_seed", "dtype_id"}:
            _uint(run[field], field)
        _uint(run["payload_bytes"], "payload_bytes", MAX_PAYLOAD)
        _uint(run["pattern_seed"], "pattern_seed", minimum=0)
        _uint(run["dtype_id"], "dtype_id", (1 << 32) - 1)
        if run["sequence"] != 1:
            raise ValidationError("first commissioning campaign uses one transfer per fresh epoch")
        if run["shape_fingerprint"] != expected_shape_fingerprint(root["campaign_id"], run):
            raise ValidationError("shape fingerprint differs from exact native campaign/run/pattern binding")
        epochs.append(run["connection_epoch"])
    unique(run_ids, "run IDs")
    unique(epochs, "connection epochs")
    if epochs != sorted(epochs):
        raise ValidationError("connection epochs must increase in campaign order")
    return root


def expected_payload_sha256(payload_bytes, seed, sequence=1):
    """Independent byte oracle for the native commissioning pattern v1."""
    _uint(payload_bytes, "payload_bytes", MAX_PAYLOAD)
    _uint(seed, "pattern_seed", minimum=0)
    _uint(sequence, "sequence")
    mask = (1 << 64) - 1
    def byte(offset):
        value = offset ^ seed ^ ((sequence * 0x9e3779b97f4a7c15) & mask)
        value = (value + 0x9e3779b97f4a7c15) & mask
        value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & mask
        value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & mask
        return (value ^ (value >> 31)) & 255
    digest = hashlib.sha256()
    for start in range(0, payload_bytes, 65536):
        digest.update(bytes(byte(offset)
                            for offset in range(start, min(start + 65536, payload_bytes))))
    return digest.hexdigest()


def _validate_receipt(raw):
    receipt = record(raw, "native commissioning receipt", RECEIPT_FIELDS)
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["role"] not in {"sender", "receiver"}:
        raise ValidationError("unsupported native receipt schema/role")
    if receipt["production_qualified"] is not False:
        raise ValidationError("commissioning receipt cannot qualify production")
    for field in FLAGS | {"executed"}:
        if type(receipt[field]) is not bool:
            raise ValidationError(field + ": boolean required")
    for field in ("campaign_id", "run_id", "physical_link_id", "device_identity", "kernel_identity", "driver_identity", "driver_build_id", "abi_identity", "gpu_architecture"):
        text(receipt[field], field)
    for field in ("local_endpoint_identity", "peer_endpoint_identity", "gpu_uuid"):
        _uuid(receipt[field], field)
    for field in (RUN_FIELDS - {"run_id", "sender", "receiver", "physical_link_id"}) | {
            "device_instance_generation", "pool_binding_identity", "import_generation",
            "gpu_runtime_version", "gpu_driver_version", "iterations", "completed_count"}:
        _uint(receipt[field], field, minimum=0 if field in {"pattern_seed", "completed_count", "import_generation"} else 1)
    _sha(receipt["expected_payload_sha256"], "expected_payload_sha256")
    if receipt["observed_payload_sha256"] is not None:
        _sha(receipt["observed_payload_sha256"], "observed_payload_sha256")
    if receipt["cleanup"] not in {"released", "retained"}:
        raise ValidationError("invalid cleanup status")
    if not isinstance(receipt["error"], str) or len(receipt["error"]) > 4096:
        raise ValidationError("invalid native error")
    if type(receipt["pattern_version"]) is not int or receipt["pattern_version"] != 1:
        raise ValidationError("unsupported pattern version")
    return receipt


def evaluate_pair(plan, run_id, sender, receiver):
    plan = validate_plan(plan)  # Rehash all bound local artifacts on every evaluation.
    runs = [run for run in plan["runs"] if run["run_id"] == run_id]
    if len(runs) != 1:
        raise ValidationError("unknown run ID")
    run = runs[0]
    nodes = {node["id"]: node for node in plan["nodes"]}
    receipts = {"sender": _validate_receipt(sender), "receiver": _validate_receipt(receiver)}
    expected_sha = expected_payload_sha256(run["payload_bytes"], run["pattern_seed"])
    blockers = []
    for role, receipt in receipts.items():
        endpoint = nodes[run[role]]["expected"]
        peer = nodes[run["receiver" if role == "sender" else "sender"]]["expected"]
        checks = {
            "ROLE": receipt["role"] == role,
            "CAMPAIGN": receipt["campaign_id"] == plan["campaign_id"],
            "RUN": all(receipt[field] == run[field] for field in RUN_FIELDS - {"sender", "receiver"}),
            "ENDPOINT": all(receipt[field] == value for field, value in endpoint.items()),
            "PEER": receipt["peer_endpoint_identity"] == peer["local_endpoint_identity"],
            "EXECUTED": receipt["executed"] is True,
            "COMPLETE": receipt["iterations"] == receipt["completed_count"] == 1,
            "PATTERN": receipt["expected_payload_sha256"] == expected_sha,
            "CONTROLLER": receipt["controller_frames_complete"] is True,
            "CLEANUP": receipt["cleanup"] == "released",
            "ERROR": receipt["error"] == "",
            "IMPORT": receipt["import_generation"] > 0,
        }
        required = {"local_tx_complete", "transport_ack", "consumer_credit"} if role == "sender" else {
            "full_payload_verified", "gpu_acquire_complete", "consumer_complete", "receive_reposted", "consumer_credit"}
        checks["STAGES"] = all(receipt[field] is True for field in required)
        checks["INAPPLICABLE_STAGES"] = all(receipt[field] is False for field in FLAGS - required - {"controller_frames_complete"})
        checks["READBACK"] = (receipt["observed_payload_sha256"] == expected_sha if role == "receiver"
                              else receipt["observed_payload_sha256"] is None)
        blockers.extend(role.upper() + "_" + code for code, passed in checks.items() if not passed)
    # Revalidate all provenance after computing results; local inputs must be quiescent.
    validate_plan(plan)
    result = {"schema": "tensormeld/direct-gpu-commissioning-pair-review-v1",
              "campaign_id": plan["campaign_id"], "run_id": run_id,
              "outcome": "OBSERVATIONS_MATCH" if not blockers else "FAILED",
              "blockers": blockers, "plan_sha256": _digest(plan),
              "sender_receipt_sha256": _digest(sender), "receiver_receipt_sha256": _digest(receiver),
              "production_qualified": False, "launch_authorized": False,
              "external_dma_visibility_qualified": False,
              "requires_independent_native_review": True}
    result["review_sha256"] = _digest(result)
    return result


def evaluate_campaign(plan, pairs):
    """Require complete exact run coverage; never average away a failed direction."""
    plan = validate_plan(plan)
    values = items(pairs, "receipt pairs", MAX_RUNS)
    pairs_by_id = {}
    for value in values:
        pair = record(value, "receipt pair", {"run_id", "sender", "receiver"})
        run_id = text(pair["run_id"], "run_id")
        if run_id in pairs_by_id:
            raise ValidationError("duplicate receipt pair")
        pairs_by_id[run_id] = pair
    if set(pairs_by_id) != {run["run_id"] for run in plan["runs"]}:
        raise ValidationError("complete exact campaign coverage required")
    results = [evaluate_pair(plan, run["run_id"], pairs_by_id[run["run_id"]]["sender"],
                             pairs_by_id[run["run_id"]]["receiver"]) for run in plan["runs"]]
    return {"schema": "tensormeld/direct-gpu-commissioning-campaign-review-v1",
            "plan_sha256": _digest(plan), "runs": results,
            "outcome": "OBSERVATIONS_MATCH" if all(result["outcome"] == "OBSERVATIONS_MATCH" for result in results) else "FAILED",
            "production_qualified": False, "launch_authorized": False,
            "external_dma_visibility_qualified": False, "requires_independent_native_review": True}


def prepare_campaign(plan, directory):
    plan = validate_plan(plan)
    destination = _private(directory)
    # mkdir is exclusive: never merge or overwrite an older campaign.
    destination.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(json.dumps(plan, allow_nan=False))
    validate_plan(frozen)
    (destination / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    prepared = {"schema": "tensormeld/direct-gpu-commissioning-prepared-v1",
                "plan_sha256": _digest(frozen), "runs": [run["run_id"] for run in frozen["runs"]],
                "executed": False, "production_qualified": False, "launch_authorized": False,
                "next_step": "Explicit operator configuration and separate native commissioning invocation on both hosts"}
    (destination / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n", encoding="utf-8")
    return prepared


def assemble_plan(inputs):
    """Bind the native host-only config reports without treating them as observations."""
    inputs = record(inputs, "configuration input bundle", {"schema", "campaign_id", "nodes", "runs"})
    if inputs["schema"] != "tensormeld/direct-gpu-commissioning-config-inputs-v1":
        raise ValidationError("unsupported configuration input bundle")
    campaign_id = _identifier(inputs["campaign_id"], "campaign_id")
    nodes = {}
    for value in items(inputs["nodes"], "nodes", 2, 2):
        node = record(value, "node artifact bindings", {"id", "artifacts"})
        node_id = text(node["id"], "node ID")
        if node_id in nodes:
            raise ValidationError("duplicate node")
        nodes[node_id] = {**node, "expected": None}
    runs, pins = [], []
    for value in items(inputs["runs"], "config pairs", MAX_RUNS):
        pair = record(value, "config pair", {"run_id", "sender", "receiver", "sender_config", "receiver_config"})
        if pair["sender"] not in nodes or pair["receiver"] not in nodes or pair["sender"] == pair["receiver"]:
            raise ValidationError("config pair requires two known distinct nodes")
        configs = {}
        for role in ("sender", "receiver"):
            artifact = pair[role + "_config"]
            path = _artifact(artifact, role + " config")
            config = record(_json(path), "native host-only config", CONFIG_FIELDS)
            if (config["schema"] != "strix-direct-gpu-commissioning-config-v1" or config["role"] != role
                    or config["campaign_id"] != campaign_id or config["run_id"] != pair["run_id"]
                    or config["executed"] is not False or config["production_qualified"] is not False
                    or type(config["pattern_version"]) is not int or config["pattern_version"] != 1):
                raise ValidationError("host-only config identity/schema/status mismatch")
            endpoint = {field: config[field] for field in ENDPOINT_FIELDS}
            node = nodes[pair[role]]
            if node["expected"] is not None and node["expected"] != endpoint:
                raise ValidationError("node expectation changed between runs")
            node["expected"] = endpoint
            configs[role] = config
            pins.append(artifact)
        a, b = configs["sender"], configs["receiver"]
        if any(a[field] != b[field] for field in RUN_FIELDS - {"sender", "receiver"}):
            raise ValidationError("config pair descriptor mismatch")
        if (a["peer_endpoint_identity"] != b["local_endpoint_identity"] or
                b["peer_endpoint_identity"] != a["local_endpoint_identity"]):
            raise ValidationError("config pair peer mismatch")
        sha = expected_payload_sha256(a["payload_bytes"], a["pattern_seed"], a["sequence"])
        if a["expected_payload_sha256"] != sha or b["expected_payload_sha256"] != sha:
            raise ValidationError("config pair pattern hash mismatch")
        runs.append({**{field: a[field] for field in RUN_FIELDS - {"sender", "receiver"}},
                     "sender": pair["sender"], "receiver": pair["receiver"]})
    plan = validate_plan({"schema": PLAN_SCHEMA, "campaign_id": campaign_id, "nodes": list(nodes.values()),
                          "runs": runs, "production_qualified": False, "launch_authorized": False})
    for artifact in pins:
        _artifact(artifact, "host-only config after assembly")
    return plan


def write_template(directory):
    """Write unresolved operator inputs, never fixture identities masquerading as hosts."""
    destination = _private(directory)
    destination.mkdir(parents=True, exist_ok=False)
    nodes = [{"id": name, "expected": {field: None for field in sorted(ENDPOINT_FIELDS)},
              "artifacts": {field: {"path": None, "sha256": None} for field in
                            ("harness_binary", "source_record", "kernel_build_record")}} for name in ("node-a", "node-b")]
    runs = []
    for sender, receiver, direction in (("node-a", "node-b", "forward"), ("node-b", "node-a", "reverse")):
        for size in (1, 4095, 4096, 4097, 65536, 1048576, MAX_PAYLOAD):
            for seed in (0, (1 << 64) - 1):
                runs.append({"run_id": f"{direction}-b{size}-s{seed}", "sender": sender, "receiver": receiver,
                             "physical_link_id": None, "connection_epoch": None, "sequence": 1,
                             "payload_bytes": size, "pattern_seed": seed, "model_plan_id": None,
                             "request_id": None, "generation": None, "boundary_id": None,
                             "shape_fingerprint": None, "dtype_id": None})
    plan = {"schema": PLAN_SCHEMA, "campaign_id": None, "nodes": nodes, "runs": runs,
            "production_qualified": False, "launch_authorized": False}
    (destination / "plan.template.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    pending = {"schema": "tensormeld/direct-gpu-commissioning-inputs-needed-v1", "status": "UNRESOLVED",
               "executed": False, "production_qualified": False, "launch_authorized": False,
               "run_count": len(runs), "required": [
                   "Exact observed host/GPU/module/ABI/endpoint identities on both target hosts",
                   "Reviewed harness/source/full-kernel build artifact pins available locally",
                   "Fresh peer-agreed monotonically increasing epoch for every invocation",
                   "Exact descriptor IDs/fingerprint from native --check-config",
                   "Operator-established authenticated tunnel for loopback-only control",
                   "Explicit local target configuration and separate native execution"]}
    (destination / "inputs-needed.json").write_text(json.dumps(pending, indent=2) + "\n", encoding="utf-8")
    return pending


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template")
    template.add_argument("--out", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--out", required=True)
    config_prepare = sub.add_parser("prepare-from-configs")
    config_prepare.add_argument("--inputs", required=True)
    config_prepare.add_argument("--out", required=True)
    review = sub.add_parser("review-pair")
    review.add_argument("--plan", required=True)
    review.add_argument("--run-id", required=True)
    review.add_argument("--sender", required=True)
    review.add_argument("--receiver", required=True)
    review.add_argument("--out", required=True)
    whole = sub.add_parser("review-campaign")
    whole.add_argument("--plan", required=True)
    whole.add_argument("--pairs", required=True)
    whole.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            result = write_template(args.out)
        elif args.command == "prepare":
            result = prepare_campaign(_json(args.plan), args.out)
        elif args.command == "prepare-from-configs":
            result = prepare_campaign(assemble_plan(_json(args.inputs)), args.out)
        else:
            result = (evaluate_pair(_json(args.plan), args.run_id, _json(args.sender), _json(args.receiver))
                      if args.command == "review-pair" else evaluate_campaign(_json(args.plan), _json(args.pairs)))
            output = _private(args.out)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result.get("outcome", "OBSERVATIONS_MATCH") == "OBSERVATIONS_MATCH" else 1
    except (ValidationError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
