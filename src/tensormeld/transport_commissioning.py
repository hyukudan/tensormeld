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
SESSION_PLAN_SCHEMA = "tensormeld/direct-gpu-commissioning-session-plan-v2"
SESSION_RECEIPT_SCHEMA = "strix-direct-gpu-commissioning-session-v2"
SESSION_CONFIG_SCHEMA = "strix-direct-gpu-commissioning-session-config-v2"
TRANSFER_RECEIPT_SCHEMA = "strix-direct-gpu-commissioning-transfer-v2"
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


def _validate_nodes(raw):
    nodes = {}
    for value in items(raw, "nodes", 2, 2):
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
    return nodes


def validate_plan(raw):
    root = record(raw, "commissioning plan", {"schema", "campaign_id", "nodes", "runs",
                                              "production_qualified", "launch_authorized"})
    if root["schema"] != PLAN_SCHEMA:
        raise ValidationError("unsupported commissioning plan")
    _identifier(root["campaign_id"], "campaign_id")
    if root["production_qualified"] is not False or root["launch_authorized"] is not False:
        raise ValidationError("commissioning plan cannot self-promote")
    nodes = _validate_nodes(root["nodes"])
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


def _validate_receipt(raw, *, session_transfer=False):
    receipt = record(raw, "native commissioning receipt", RECEIPT_FIELDS)
    if receipt["schema"] != (TRANSFER_RECEIPT_SCHEMA if session_transfer else RECEIPT_SCHEMA) or receipt["role"] not in ("sender", "receiver"):
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
    if receipt["cleanup"] not in (("import_held",) if session_transfer else ("released", "retained")):
        raise ValidationError("invalid cleanup status")
    if not isinstance(receipt["error"], str) or len(receipt["error"]) > 4096:
        raise ValidationError("invalid native error")
    if type(receipt["pattern_version"]) is not int or receipt["pattern_version"] != 1:
        raise ValidationError("unsupported pattern version")
    return receipt


def _pair_blockers(plan, run, sender, receiver, *, session_transfer=False):
    nodes = {node["id"]: node for node in plan["nodes"]}
    receipts = {"sender": _validate_receipt(sender, session_transfer=session_transfer),
                "receiver": _validate_receipt(receiver, session_transfer=session_transfer)}
    expected_sha = expected_payload_sha256(run["payload_bytes"], run["pattern_seed"], run["sequence"])
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
            "CLEANUP": receipt["cleanup"] == ("import_held" if session_transfer else "released"),
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
    return blockers


def evaluate_pair(plan, run_id, sender, receiver):
    plan = validate_plan(plan)  # Rehash all bound local artifacts on every evaluation.
    runs = [run for run in plan["runs"] if run["run_id"] == run_id]
    if len(runs) != 1:
        raise ValidationError("unknown run ID")
    blockers = _pair_blockers(plan, runs[0], sender, receiver)
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


def validate_session_plan(raw):
    """Explicit v2 C1 reuse/paired C2; v1 fresh-epoch semantics stay unchanged."""
    root = record(raw, "commissioning session plan", {"schema", "campaign_id", "session_id", "schedule",
                  "concurrency", "nodes", "runs", "production_qualified", "launch_authorized"})
    if root["schema"] != SESSION_PLAN_SCHEMA or root["schedule"] not in ("sequential", "alternating", "duplex_pairs"):
        raise ValidationError("unsupported commissioning session schema/schedule")
    _identifier(root["campaign_id"], "campaign_id")
    session_id = _identifier(root["session_id"], "session_id")
    concurrency = 2 if root["schedule"] == "duplex_pairs" else 1
    if len(session_id) > 122 or type(root["concurrency"]) is not int or root["concurrency"] != concurrency:
        raise ValidationError("session requires exact bounded schedule concurrency")
    if root["production_qualified"] is not False or root["launch_authorized"] is not False:
        raise ValidationError("commissioning session cannot self-promote")
    nodes = _validate_nodes(root["nodes"])
    runs = items(root["runs"], "session transfers", MAX_RUNS, 2)
    if concurrency == 2 and len(runs) % 2:
        raise ValidationError("duplex_pairs requires complete opposite-direction pairs")
    total = 0
    for index, value in enumerate(runs):
        run = record(value, "session transfer", RUN_FIELDS)
        if run["run_id"] != session_id + ".s" + str(index):
            raise ValidationError("session transfer ID/order mismatch")
        text(run["sender"], "sender")
        text(run["receiver"], "receiver")
        if run["sender"] not in nodes or run["receiver"] not in nodes or run["sender"] == run["receiver"]:
            raise ValidationError("transfer requires two distinct known endpoints")
        text(run["physical_link_id"], "physical_link_id")
        for field in RUN_FIELDS - {"run_id", "sender", "receiver", "physical_link_id", "payload_bytes", "pattern_seed", "dtype_id"}:
            _uint(run[field], field)
        total += _uint(run["payload_bytes"], "payload_bytes", MAX_PAYLOAD)
        _uint(run["pattern_seed"], "pattern_seed", minimum=0)
        _uint(run["dtype_id"], "dtype_id", (1 << 32) - 1)
        if run["sequence"] != index + 1 or total > 128 * 1024 * 1024:
            raise ValidationError("session sequence/aggregate payload bound violated")
        if run["shape_fingerprint"] != expected_shape_fingerprint(root["campaign_id"], run):
            raise ValidationError("session transfer fingerprint mismatch")
        if index:
            first = runs[0]
            if any(run[field] != first[field] for field in ("connection_epoch", "physical_link_id")):
                raise ValidationError("session cannot replace its physical epoch/link")
            roles = (first["receiver"], first["sender"]) if root["schedule"] in {"alternating", "duplex_pairs"} and index % 2 else (first["sender"], first["receiver"])
            if (run["sender"], run["receiver"]) != roles:
                raise ValidationError("session direction schedule mismatch")
    return root


def _session_receipt(raw):
    receipt = record(raw, "native session receipt", {"schema", "campaign_id", "session_id", "schedule", "concurrency",
                     "requested_transfers", "completed_count", "pool_binding_identity", "import_generation",
                     "cleanup", "error", "production_qualified", "transfers"})
    if receipt["schema"] != SESSION_RECEIPT_SCHEMA or receipt["schedule"] not in ("sequential", "alternating", "duplex_pairs"):
        raise ValidationError("unsupported session receipt schema/schedule")
    _identifier(receipt["campaign_id"], "campaign_id")
    _identifier(receipt["session_id"], "session_id")
    if receipt["production_qualified"] is not False or type(receipt["concurrency"]) is not int or receipt["concurrency"] != (2 if receipt["schedule"] == "duplex_pairs" else 1):
        raise ValidationError("session receipt cannot qualify production or change bounded schedule concurrency")
    _uint(receipt["requested_transfers"], "requested_transfers", MAX_RUNS, 2)
    if receipt["schedule"] == "duplex_pairs" and receipt["requested_transfers"] % 2:
        raise ValidationError("duplex receipt requires complete requested pairs")
    _uint(receipt["completed_count"], "completed_count", receipt["requested_transfers"], 0)
    _uint(receipt["pool_binding_identity"], "pool_binding_identity")
    _uint(receipt["import_generation"], "import_generation", minimum=0)
    if receipt["cleanup"] not in ("released", "retained") or not isinstance(receipt["error"], str) or len(receipt["error"]) > 4096:
        raise ValidationError("invalid session cleanup/error")
    transfers = items(receipt["transfers"], "native session transfers", receipt["requested_transfers"], 0)
    for transfer in transfers:
        _validate_receipt(transfer, session_transfer=True)
    return receipt


def evaluate_session(plan, node_a, receipt_a, node_b, receipt_b):
    """Require exact ordered coverage and stable imported owners on both hosts."""
    plan = validate_session_plan(plan)
    text(node_a, "node_a")
    text(node_b, "node_b")
    ids = {node["id"] for node in plan["nodes"]}
    if node_a == node_b or {node_a, node_b} != ids:
        raise ValidationError("session review requires both exact node IDs")
    receipts = {node_a: _session_receipt(receipt_a), node_b: _session_receipt(receipt_b)}
    blockers, results = [], []
    count = len(plan["runs"])
    for node_id, receipt in receipts.items():
        checks = {"IDENTITY": all(receipt[field] == plan[field] for field in ("campaign_id", "session_id", "schedule", "concurrency")),
                  "COVERAGE": receipt["requested_transfers"] == receipt["completed_count"] == len(receipt["transfers"]) == count,
                  "CLEANUP": receipt["cleanup"] == "released", "ERROR": receipt["error"] == "", "IMPORT": receipt["import_generation"] > 0}
        stable_fields = ENDPOINT_FIELDS | {"peer_endpoint_identity", "device_identity", "device_instance_generation", "pool_binding_identity", "import_generation"}
        first = receipt["transfers"][0] if receipt["transfers"] else None
        for transfer in receipt["transfers"]:
            if any(transfer[field] != first[field] for field in stable_fields) or any(transfer[field] != receipt[field] for field in ("pool_binding_identity", "import_generation")):
                checks["IMPORT_OWNER_STABLE"] = False
        blockers.extend(node_id + "_" + code for code, passed in checks.items() if not passed)
    for index, run in enumerate(plan["runs"]):
        a, b = receipts[run["sender"]]["transfers"], receipts[run["receiver"]]["transfers"]
        failures = ["MISSING_TRANSFER"] if index >= len(a) or index >= len(b) else _pair_blockers(plan, run, a[index], b[index], session_transfer=True)
        blockers.extend(run["run_id"] + "_" + code for code in failures)
        results.append({"run_id": run["run_id"], "outcome": "FAILED" if failures else "OBSERVATIONS_MATCH", "blockers": failures})
    validate_session_plan(plan)  # Provenance must remain stable across the whole independent oracle.
    result = {"schema": "tensormeld/direct-gpu-commissioning-session-review-v2", "campaign_id": plan["campaign_id"],
              "session_id": plan["session_id"], "outcome": "FAILED" if blockers else "OBSERVATIONS_MATCH", "blockers": blockers,
              "transfers": results, "plan_sha256": _digest(plan), "receipts_sha256": {node: _digest(receipt) for node, receipt in receipts.items()},
              "production_qualified": False, "launch_authorized": False, "external_dma_visibility_qualified": False,
              # Two owned operations/complete directions are lifecycle evidence.
              # The current receipt has no matched controller DMA start/end
              # intervals and cannot establish physical overlap from capacity2.
              "physical_overlap_status": "UNKNOWN", "controller_overlap_qualified": False,
              "requires_independent_native_review": True}
    result["review_sha256"] = _digest(result)
    return result


def assemble_session_plan(inputs):
    inputs = record(inputs, "session config input bundle", {"schema", "campaign_id", "session_id", "nodes",
                    "node_a", "node_b", "node_a_config", "node_b_config"})
    if inputs["schema"] != "tensormeld/direct-gpu-commissioning-session-config-inputs-v2":
        raise ValidationError("unsupported session config inputs")
    nodes = {}
    for raw in items(inputs["nodes"], "nodes", 2, 2):
        node = record(raw, "node artifacts", {"id", "artifacts"})
        if text(node["id"], "node ID") in nodes:
            raise ValidationError("duplicate session node")
        nodes[node["id"]] = {**node, "expected": None}
    node_a, node_b = inputs["node_a"], inputs["node_b"]
    text(node_a, "node_a")
    text(node_b, "node_b")
    if node_a == node_b or {node_a, node_b} != set(nodes):
        raise ValidationError("both exact session nodes required")
    configs, pins = {}, []
    for node_id, name in ((node_a, "node_a_config"), (node_b, "node_b_config")):
        pin = inputs[name]; path = _artifact(pin, name); pins.append(pin)
        config = record(_json(path), "native session config", {"schema", "campaign_id", "session_id", "schedule", "concurrency", "executed", "production_qualified", "transfers"})
        if config["schema"] != SESSION_CONFIG_SCHEMA or config["executed"] is not False or config["production_qualified"] is not False:
            raise ValidationError("session config schema/status mismatch")
        if any(config[field] != inputs[field] for field in ("campaign_id", "session_id")):
            raise ValidationError("session config identity mismatch")
        transfers = items(config["transfers"], "session config transfers", MAX_RUNS, 2)
        for transfer in transfers:
            record(transfer, "host-only transfer config", CONFIG_FIELDS)
            if transfer["schema"] != "strix-direct-gpu-commissioning-config-v1" or transfer["executed"] is not False or transfer["production_qualified"] is not False or type(transfer["pattern_version"]) is not int or transfer["pattern_version"] != 1 or transfer["role"] not in ("sender", "receiver") or transfer["campaign_id"] != inputs["campaign_id"]:
                raise ValidationError("invalid host-only transfer config")
            endpoint = {field: transfer[field] for field in ENDPOINT_FIELDS}
            if nodes[node_id]["expected"] is not None and nodes[node_id]["expected"] != endpoint:
                raise ValidationError("session config endpoint changed")
            nodes[node_id]["expected"] = endpoint
        configs[node_id] = config
    a, b = configs[node_a], configs[node_b]
    if any(a[field] != b[field] for field in ("schedule", "concurrency")) or len(a["transfers"]) != len(b["transfers"]):
        raise ValidationError("session config schedule mismatch")
    runs = []
    for left, right in zip(a["transfers"], b["transfers"]):
        if left["role"] == right["role"] or any(left[field] != right[field] for field in RUN_FIELDS - {"sender", "receiver"}):
            raise ValidationError("session config direction/descriptor mismatch")
        if left["peer_endpoint_identity"] != right["local_endpoint_identity"] or right["peer_endpoint_identity"] != left["local_endpoint_identity"]:
            raise ValidationError("session config peer mismatch")
        sha = expected_payload_sha256(left["payload_bytes"], left["pattern_seed"], left["sequence"])
        if left["expected_payload_sha256"] != sha or right["expected_payload_sha256"] != sha:
            raise ValidationError("session config independent pattern hash mismatch")
        runs.append({**{field: left[field] for field in RUN_FIELDS - {"sender", "receiver"}},
                     "sender": node_a if left["role"] == "sender" else node_b,
                     "receiver": node_b if left["role"] == "sender" else node_a})
    plan = validate_session_plan({"schema": SESSION_PLAN_SCHEMA, "campaign_id": inputs["campaign_id"], "session_id": inputs["session_id"],
                "schedule": a["schedule"], "concurrency": a["concurrency"], "nodes": list(nodes.values()), "runs": runs,
                "production_qualified": False, "launch_authorized": False})
    for pin in pins:
        _artifact(pin, "session config after assembly")
    return plan


def prepare_session(plan, directory):
    plan = validate_session_plan(plan)
    destination = _private(directory)
    destination.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(json.dumps(plan, allow_nan=False))
    validate_session_plan(frozen)
    (destination / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    result = {"schema": "tensormeld/direct-gpu-commissioning-session-prepared-v2", "plan_sha256": _digest(frozen),
              "session_id": plan["session_id"], "transfers": len(plan["runs"]), "concurrency": plan["concurrency"],
              "executed": False, "production_qualified": False, "launch_authorized": False}
    (destination / "prepared.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


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
    session_prepare = sub.add_parser("prepare-session")
    session_prepare.add_argument("--plan", required=True)
    session_prepare.add_argument("--out", required=True)
    session_configs = sub.add_parser("prepare-session-from-configs")
    session_configs.add_argument("--inputs", required=True)
    session_configs.add_argument("--out", required=True)
    session_review = sub.add_parser("review-session")
    session_review.add_argument("--plan", required=True)
    session_review.add_argument("--node-a", required=True)
    session_review.add_argument("--node-b", required=True)
    session_review.add_argument("--receipt-a", required=True)
    session_review.add_argument("--receipt-b", required=True)
    session_review.add_argument("--out", required=True)
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
        elif args.command == "prepare-session":
            result = prepare_session(_json(args.plan), args.out)
        elif args.command == "prepare-session-from-configs":
            result = prepare_session(assemble_session_plan(_json(args.inputs)), args.out)
        else:
            if args.command == "review-session":
                result = evaluate_session(_json(args.plan), args.node_a, _json(args.receipt_a), args.node_b, _json(args.receipt_b))
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
