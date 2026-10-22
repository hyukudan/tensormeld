from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tensormeld.cli import main
from tensormeld.config_v2 import Config
from tensormeld.device_binding import bind_llamacpp_probe
from tensormeld.llamacpp_selftest import (
    MAX_OUTPUT_BYTES,
    _SQL_FIELDS,
    _SQL_HEADER,
    load_llamacpp_bound_result,
    self_test_llamacpp_backend,
)
from tensormeld.schema import ValidationError
from tensormeld.selection import resolve_runtime_candidates
from test_device_binding import binding, probe
from test_config_v2 import data


def sql_output(
    *,
    backend: str = "CUDA0",
    commit: str = "552f18f9",
    op: str = "ADD",
    mode: str = "test",
    supported: str = "1",
    passed: str = "1",
    error: str = "",
) -> bytes:
    values = {
        "test_time": "2026-10-01T20:00:00Z",
        "build_commit": commit,
        "backend_name": backend,
        "op_name": op,
        "op_params": "type=f32,ne=[1,1,1,1],nr=[1,1,1,1]",
        "test_mode": mode,
        "supported": supported,
        "passed": passed,
        "error_message": error,
        "time_us": "0.000000",
        "flops": "0.000000",
        "bandwidth_gb_s": "0.000000",
        "memory_kb": "0",
        "n_runs": "0",
        "device_description": "",
        "backend_reg_name": "",
    }
    escaped = [
        "'" + values[field].replace("'", "''") + "'"
        for field in _SQL_FIELDS
    ]
    insert = (
        "INSERT INTO test_backend_ops ("
        + ", ".join(_SQL_FIELDS)
        + ") VALUES ("
        + ", ".join(escaped)
        + ");"
    )
    return ("\n".join((*_SQL_HEADER, "", insert)) + "\n").encode()


class FixtureRunner:
    def __init__(self, output: bytes | None = None, rc: int = 0, stderr: bytes = b""):
        self.output = output if output is not None else sql_output()
        self.rc = rc
        self.stderr = stderr
        self.calls = []

    def __call__(self, argv, timeout):
        self.calls.append((tuple(argv), timeout))
        return self.rc, self.output, self.stderr


class LlamaCppBackendSelfTestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.binary = Path(self.tmp.name) / "test-backend-ops"
        self.binary.write_bytes(b"fixture-test-backend-ops")
        self.sha = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.cfg = Config.parse(data())
        self.bound = bind_llamacpp_probe(self.cfg, probe(), binding(self.cfg))

    def run_self_test(self, runner=None, **kwargs):
        return self_test_llamacpp_backend(
            self.binary,
            config=self.cfg,
            bound_result=self.bound,
            tensormeld_device_id="pc-gpu",
            trusted_local_binary=True,
            expected_artifact_sha256=self.sha,
            runner=runner or FixtureRunner(),
            **kwargs,
        )

    def test_strict_success_promotes_only_bound_device_to_ready(self):
        runner = FixtureRunner()
        result = self.run_self_test(runner)
        self.assertEqual(
            runner.calls[0][0][1:],
            ("test", "-b", "CUDA0", "-o", "ADD", "--output", "sql", "-j", "1"),
        )
        self.assertTrue(result["backend_initialized"])
        self.assertTrue(result["backend_executed"])
        self.assertEqual(result["evidence_level"], "E2")
        self.assertEqual(result["node_id"], self.bound["node_id"])
        self.assertEqual(result["execution_source"], "injected-runner")
        self.assertEqual(result["passed_rows"], 1)
        self.assertEqual(
            result["runtime_observation"]["devices"]["pc-gpu"]["state"], "ready"
        )
        self.assertFalse(result["reservation_created"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["executable"])

        runtime = resolve_runtime_candidates(
            self.cfg, result["runtime_observation"]
        )
        self.assertIn(
            "pc-gpu",
            [d["id"] for d in runtime["runtime_eligible_compute_devices"]],
        )
        self.assertFalse(runtime["reservation_created"])

    def test_zero_exit_without_target_rows_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.run_self_test(FixtureRunner(sql_output(backend="CPU")))

    def test_zero_exit_without_executed_supported_row_is_rejected(self):
        output = sql_output(supported="0", passed="0", error="not supported")
        with self.assertRaises(ValidationError):
            self.run_self_test(FixtureRunner(output))

    def test_supported_failure_is_rejected_even_with_zero_exit(self):
        output = sql_output(supported="1", passed="0", error="compare failed")
        with self.assertRaises(ValidationError):
            self.run_self_test(FixtureRunner(output))

    def test_wrong_mode_operation_or_source_revision_is_rejected(self):
        for output in (
            sql_output(mode="support"),
            sql_output(op="MUL"),
            sql_output(commit="deadbee"),
        ):
            with self.subTest(output=output[:80]):
                with self.assertRaises(ValidationError):
                    self.run_self_test(FixtureRunner(output))

    def test_exit_code_is_required_but_not_sufficient(self):
        with self.assertRaises(ValidationError):
            self.run_self_test(FixtureRunner(rc=1))

    def test_exact_test_artifact_identity_and_explicit_trust_are_required(self):
        with self.assertRaises(ValidationError):
            self_test_llamacpp_backend(
                self.binary,
                config=self.cfg,
                bound_result=self.bound,
                tensormeld_device_id="pc-gpu",
                expected_artifact_sha256=self.sha,
                runner=FixtureRunner(),
            )
        with self.assertRaises(ValidationError):
            self_test_llamacpp_backend(
                self.binary,
                config=self.cfg,
                bound_result=self.bound,
                tensormeld_device_id="pc-gpu",
                trusted_local_binary=True,
                expected_artifact_sha256="0" * 64,
                runner=FixtureRunner(),
            )

    def test_self_test_rejects_stale_or_pre_promoted_binding_state(self):
        stale = json.loads(json.dumps(self.bound))
        stale["config_sha256"] = "0" * 64
        with self.assertRaises(ValidationError):
            self_test_llamacpp_backend(
                self.binary,
                config=self.cfg,
                bound_result=stale,
                tensormeld_device_id="pc-gpu",
                trusted_local_binary=True,
                expected_artifact_sha256=self.sha,
                runner=FixtureRunner(),
            )

        ready = json.loads(json.dumps(self.bound))
        ready["runtime_observation"]["devices"]["pc-gpu"]["state"] = "ready"
        with self.assertRaises(ValidationError):
            self_test_llamacpp_backend(
                self.binary,
                config=self.cfg,
                bound_result=ready,
                tensormeld_device_id="pc-gpu",
                trusted_local_binary=True,
                expected_artifact_sha256=self.sha,
                runner=FixtureRunner(),
            )

    def test_output_is_bounded_and_sql_contract_is_strict(self):
        with self.assertRaises(ValidationError):
            self.run_self_test(FixtureRunner(b"x" * (MAX_OUTPUT_BYTES + 1)))
        with self.assertRaises(ValidationError):
            self.run_self_test(
                FixtureRunner(
                    b"CREATE TABLE IF NOT EXISTS test_backend_ops (\n"
                    b"unexpected\n);\n"
                )
            )


class LlamaCppBackendSelfTestCLITests(unittest.TestCase):
    def test_cli_roundtrip_and_output_safety(self):
        raw = data()
        cfg = Config.parse(raw)
        bound = bind_llamacpp_probe(cfg, probe(), binding(cfg))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config_path = root / "config.json"
            bound_path = root / "bound.json"
            binary = root / "test-backend-ops"
            out = root / "self-test.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            bound_path.write_text(json.dumps(bound), encoding="utf-8")
            binary.write_bytes(b"fixture-test-backend-ops")
            sha = hashlib.sha256(binary.read_bytes()).hexdigest()

            with patch(
                "tensormeld.llamacpp_selftest._default_runner",
                FixtureRunner(),
            ):
                code = main([
                    "self-test-llamacpp-backend",
                    str(config_path),
                    str(bound_path),
                    str(binary),
                    "pc-gpu",
                    "--trusted-local-binary",
                    "--expected-sha256",
                    sha,
                    "--out",
                    str(out),
                ])
            self.assertEqual(code, 0)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                result["runtime_observation"]["devices"]["pc-gpu"]["state"], "ready"
            )

            before = bound_path.read_bytes()
            with patch(
                "tensormeld.llamacpp_selftest._default_runner",
                FixtureRunner(),
            ), contextlib.redirect_stderr(io.StringIO()):
                code = main([
                    "self-test-llamacpp-backend",
                    str(config_path),
                    str(bound_path),
                    str(binary),
                    "pc-gpu",
                    "--trusted-local-binary",
                    "--expected-sha256",
                    sha,
                    "--out",
                    str(bound_path),
                ])
            self.assertEqual(code, 1)
            self.assertEqual(bound_path.read_bytes(), before)

    def test_bound_result_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bound.json"
            path.write_text('{"result_schema":"x","result_schema":"y"}', encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_llamacpp_bound_result(path)


if __name__ == "__main__":
    unittest.main()
