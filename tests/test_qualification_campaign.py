from __future__ import annotations

import base64
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from tensormeld.qualification_campaign import (
    QualificationCampaign, REPO, SCHEMA, _digest, _experiment_argv, _file_sha,
    _run_process, _worker, code_identity, load_campaign, main, run_qualification_campaign,
)
from tensormeld.schema import ValidationError
import tensormeld.qualification_campaign as campaign_module
from test_config_v2 import data
from test_runtime_model_manifest import manifest, model_data, setup
from test_target_host_qualification import runtime_identity


def stage(ident, kind="validate_config", **overrides):
    result = {"id": ident, "kind": kind, "enabled": True, "timeout_s": 30,
              "max_output_bytes": 256 * 1024, "parameters": {}}
    result.update(overrides)
    return result


def fixture(root: Path):
    cfg, model, adapter = setup()
    raw_cfg = data()
    raw_cfg["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
    raw_adapter = {
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": adapter.adapter_id, "engine": adapter.engine,
        "engine_revision": adapter.engine_revision,
        "placement": {"strategies": sorted(adapter.strategies),
                      **{name: getattr(adapter, name) for name in (
                          "exact_owner_binding", "explicit_unit_ranges", "mixed_backends",
                          "remote_compute", "coordinator_outside_compute", "max_compute_devices",
                          "max_compute_nodes", "max_segments")}},
        "route_modes": sorted(adapter.route_modes),
        "coordinator_nodes": [node.id for node in cfg.nodes],
        "devices": [{"id": device.id, "node": device.node, "backend": device.backend}
                    for device in adapter.devices],
    }

    def artifact(name, raw):
        path = root / name
        path.write_text(json.dumps(raw), encoding="utf-8")
        return {"path": str(path), "sha256": _file_sha(path)}

    artifacts = {"config": artifact("config.json", raw_cfg),
                 "model_manifest": artifact("model.json", model_data()),
                 "adapter_capabilities": artifact("adapter.json", raw_adapter),
                 "runtime_manifest": artifact("runtime.json", manifest(cfg, model, adapter)),
                 "runtime_identities": [artifact(f"identity-{index}.json", runtime_identity(
                     device.node, device.id, "c" * 64).as_record())
                     for index, device in enumerate(cfg.devices)]}
    return {"campaign_schema": SCHEMA, "mode": "fixture",
            "host": {"node_id": "pc", "hostname": socket.gethostname(), "platform": platform.system()},
            "artifacts": artifacts, "results_directory": str(root / "private-results"),
            "stages": [stage("config"), stage("portable", "portable_tests", parameters={"suite": "request_envelope"}),
                       stage("runtime", "validate_runtime_manifest"),
                       stage("unconfigured", "native_graph_fixture", enabled=False)]}


def captured(ident="one", kind="validate_config", *, returncode=0, reason=None,
             mutation=None, native=None, all_skipped=False):
    started = {"stage_started_schema": "tensormeld/campaign-stage-started-v1", "stage_id": ident, "kind": kind}
    payload = {"stage_result_schema": "tensormeld/campaign-stage-result-v1", "stage_id": ident,
               "kind": kind, "exit_code": returncode, "detail": {}, "native_invocations": native or [],
               "execution_source": "portable-fixture-tests" if kind == "portable_tests" else "local-cli",
               "campaign_mode": "fixture", "qualified": False, "executable": False,
               "all_tests_skipped": all_skipped}
    if mutation:
        mutation(payload)
    raw = (json.dumps(started) + "\n" + json.dumps(payload) + "\n").encode()
    return {"returncode": returncode, "stdout": raw, "stderr": b"", "failure_reason": reason,
            "timed_out": reason == "TIMEOUT", "output_limit_exceeded": reason == "OUTPUT_LIMIT",
            "output_bytes_observed": len(raw), "child_exit_confirmed": True,
            "process_tree_stop_requested": True, "pipes_closed": True, "elapsed_s": 0.01}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.raw = fixture(self.root)
        self.path = self.root / "campaign-input.json"

    def save(self):
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        return self.path

    def test_actual_stage_progression_private_provenance_and_cli(self):
        report = run_qualification_campaign(self.save())
        self.assertEqual([item["outcome"] for item in report["stages"]], ["PASSED", "PASSED", "PASSED", "SKIPPED"])
        self.assertEqual([item["executed"] for item in report["stages"]], [True, True, True, False])
        self.assertGreater(report["stages"][1]["stage_result"]["detail"]["tests_executed"], 0)
        self.assertEqual(report["stages"][1]["stage_result"]["detail"]["tests_failed"], 0)
        self.assertIn("OK", report["stages"][1]["stage_result"]["portable_test_output"])
        for key in ("native_qualified", "hardware_readiness_established", "inference_request_authorized", "executable"):
            self.assertIs(report[key], False)
        self.assertEqual(report["binding"]["runtime_manifest_provenance"], "fixture")
        output = Path(report["results_directory"])
        for item in report["stages"][:3]:
            self.assertTrue(item["input_identity_unchanged_after_stage"])
            self.assertTrue(item["child_exit_confirmed"])
            self.assertTrue(item["pipes_closed"])
            self.assertTrue(item["process_tree_stop_requested"])
            self.assertEqual(_file_sha(output / (item["id"] + ".stdout")), item["stdout_sha256"])
        stored = json.loads((output / "campaign.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, report)
        claimed = stored.pop("result_sha256")
        self.assertEqual(_digest(stored), claimed)
        self.assertEqual(load_campaign(output / "campaign-input.json").fingerprint, report["campaign_sha256"])
        # The callable module also has a user-facing CLI with a bounded summary.
        with patch("tensormeld.qualification_campaign.run_qualification_campaign", return_value=report), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main([str(self.path)]), 0)
        self.assertEqual(json.loads(out.getvalue())["result_sha256"], claimed)

    def test_code_identity_binds_test_sources_and_examples(self):
        identity = code_identity()
        for name in ("src/tensormeld/qualification_campaign.py", "tests/test_request_envelope.py",
                     "tests/test_qualification_campaign.py", "docs/examples/12gb-pc-one-companion.config.json"):
            self.assertEqual(identity["source_files"][name], _file_sha(REPO / name))
        self.assertEqual(identity["source_sha256"], _digest(identity["source_files"]))
        self.assertEqual(identity["python_artifact_sha256"], _file_sha(Path(sys.executable)))

    def test_immutable_parse_has_detached_config_and_identity(self):
        campaign = QualificationCampaign.parse(self.raw)
        self.raw["stages"].clear()
        campaign.config["stages"].clear()
        self.assertEqual(len(campaign.config["stages"]), 4)
        self.assertEqual(_digest(campaign.config), campaign.fingerprint)

    def test_fail_stops_later_enabled_stages_but_disabled_is_skipped(self):
        self.raw["stages"] = [stage("one"), stage("two"), stage("disabled", enabled=False)]
        with patch("tensormeld.qualification_campaign._run_process", return_value=captured(returncode=1)) as runner:
            report = run_qualification_campaign(self.save())
        self.assertEqual(runner.call_count, 1)
        self.assertEqual([item["outcome"] for item in report["stages"]], ["FAILED", "NOT_RUN", "SKIPPED"])
        self.assertTrue(report["stages"][0]["executed"])
        self.assertFalse(report["stages"][1]["executed"])

    def test_timeout_and_partial_result_preserve_started_observation(self):
        self.raw["stages"] = [stage("one"), stage("later")]
        outcome = captured(returncode=-1, reason="TIMEOUT")
        outcome["stdout"] = outcome["stdout"].splitlines()[0] + b"\n{\"truncated\""
        with patch("tensormeld.qualification_campaign._run_process", return_value=outcome):
            report = run_qualification_campaign(self.save())
        first = report["stages"][0]
        self.assertEqual(first["outcome"], "FAILED")
        self.assertEqual(first["reason"], "TIMEOUT")
        self.assertTrue(first["executed"])
        self.assertTrue(first["timed_out"])
        self.assertEqual(report["stages"][1]["outcome"], "NOT_RUN")

    def test_cleanup_failure_is_fail_closed(self):
        self.raw["stages"] = [stage("one"), stage("later")]
        with patch("tensormeld.qualification_campaign._run_process", side_effect=ValidationError("cleanup failed")):
            report = run_qualification_campaign(self.save())
        self.assertEqual(report["stages"][0]["reason"], "STAGE_PROCESS_OR_CLEANUP_FAILED")
        self.assertEqual(report["stages"][1]["outcome"], "NOT_RUN")
        self.assertFalse(report["hardware_readiness_established"])

    def test_all_skipped_portable_suite_is_never_passed(self):
        self.raw["stages"] = [stage("one", "portable_tests", parameters={"suite": "numerical"})]
        with patch("tensormeld.qualification_campaign._run_process", return_value=captured(kind="portable_tests", all_skipped=True)):
            report = run_qualification_campaign(self.save())
        self.assertEqual(report["stages"][0]["outcome"], "SKIPPED")
        self.assertEqual(report["stages"][0]["reason"], "ALL_TESTS_SKIPPED")

    def test_mismatched_or_promoted_stage_output_is_rejected(self):
        self.raw["stages"] = [stage("one")]
        mutations = [lambda value: value.update(qualified=True), lambda value: value.update(executable=True),
                     lambda value: value.update(stage_id="another"), lambda value: value.update(kind="host_probe"),
                     lambda value: value.update(campaign_mode="native"), lambda value: value.update(exit_code=True),
                     lambda value: value.update(exit_code=1), lambda value: value.update(execution_source="native"),
                     lambda value: value.update(native_invocations=["malformed"])]
        for mutation in mutations:
            with self.subTest(mutation=mutation), patch("tensormeld.qualification_campaign._run_process", return_value=captured(mutation=mutation)):
                report = run_qualification_campaign(self.save())
            self.assertEqual(report["stages"][0]["outcome"], "FAILED")
            self.assertFalse(report["native_qualified"])

    def test_duplicate_stage_protocol_keys_are_invalid(self):
        self.raw["stages"] = [stage("one")]
        outcome = captured()
        outcome["stdout"] = outcome["stdout"].replace(b'"qualified": false', b'"qualified": true, "qualified": false')
        with patch("tensormeld.qualification_campaign._run_process", return_value=outcome):
            report = run_qualification_campaign(self.save())
        self.assertEqual(report["stages"][0]["outcome"], "FAILED")

    def test_mutated_artifact_during_stage_is_failure_and_stops_progression(self):
        self.raw["stages"] = [stage("one"), stage("later")]
        def mutate(*args, **kwargs):
            Path(self.raw["artifacts"]["model_manifest"]["path"]).write_text("changed", encoding="utf-8")
            return captured()
        with patch("tensormeld.qualification_campaign._run_process", side_effect=mutate):
            report = run_qualification_campaign(self.save())
        self.assertEqual(report["stages"][0]["reason"], "INPUT_IDENTITY_CHANGED_DURING_STAGE")
        self.assertFalse(report["stages"][0]["input_identity_unchanged_after_stage"])
        self.assertEqual(report["stages"][1]["outcome"], "NOT_RUN")

    def test_code_or_python_identity_drift_after_stage_is_failure(self):
        self.raw["stages"] = [stage("one")]
        actual = code_identity()
        changed = dict(actual, python_artifact_sha256="f" * 64)
        with patch("tensormeld.qualification_campaign.code_identity", side_effect=[actual, actual, changed]), patch(
                "tensormeld.qualification_campaign._run_process", return_value=captured()):
            report = run_qualification_campaign(self.save())
        self.assertEqual(report["stages"][0]["reason"], "INPUT_IDENTITY_CHANGED_DURING_STAGE")

    def test_pre_dispatch_identity_change_does_not_execute(self):
        self.raw["stages"] = [stage("one")]
        with patch("tensormeld.qualification_campaign._inputs_unchanged", return_value=False), patch(
                "tensormeld.qualification_campaign._run_process") as runner:
            report = run_qualification_campaign(self.save())
        runner.assert_not_called()
        self.assertEqual(report["stages"][0]["reason"], "INPUT_IDENTITY_CHANGED_BEFORE_STAGE")
        self.assertFalse(report["stages"][0]["executed"])

    def test_raw_native_text_is_preserved_with_hashes_without_promoting_qualification(self):
        self.raw["stages"] = [stage("one")]
        stdout, stderr = b"native fixture PASS text\n", b"diagnostic\n"
        invocation = {"stdout_base64": base64.b64encode(stdout).decode(), "stderr_base64": base64.b64encode(stderr).decode(),
                      "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                      "timed_out": False, "output_limit_exceeded": False}
        with patch("tensormeld.qualification_campaign._run_process", return_value=captured(native=[invocation])):
            report = run_qualification_campaign(self.save())
        output = Path(report["results_directory"])
        self.assertEqual((output / "one.native-0.stdout").read_bytes(), stdout)
        self.assertEqual((output / "one.native-0.stderr").read_bytes(), stderr)
        record = report["stages"][0]["stage_result"]["native_invocations"][0]
        self.assertNotIn("stdout_base64", record)
        self.assertEqual(record["stdout_retained_bytes"], len(stdout))
        self.assertFalse(report["native_qualified"])

    def test_bad_native_raw_hash_fails(self):
        self.raw["stages"] = [stage("one")]
        invocation = {"stdout_base64": "YQ==", "stderr_base64": "", "stdout_sha256": "f" * 64,
                      "stderr_sha256": hashlib.sha256(b"").hexdigest(), "timed_out": False,
                      "output_limit_exceeded": False}
        with patch("tensormeld.qualification_campaign._run_process", return_value=captured(native=[invocation])):
            report = run_qualification_campaign(self.save())
        self.assertEqual(report["stages"][0]["outcome"], "FAILED")

    def test_invalid_closed_configuration_and_limits(self):
        mutations = [lambda value: value.update(command="whoami"), lambda value: value.update(mode=True),
                     lambda value: value["host"].update(hostname="another-local-host"),
                     lambda value: value["host"].update(node_id="not-configured"),
                     lambda value: value.update(results_directory=str(REPO / "results")),
                     lambda value: value["stages"][0].update(kind="shell"),
                     lambda value: value["stages"][0].update(parameters={"argv": ["whoami"]}),
                     lambda value: value["stages"][0].update(id="../escape"),
                     lambda value: value["stages"][0].update(enabled=1),
                     lambda value: value["stages"][0].update(timeout_s=True),
                     lambda value: value["stages"][0].update(timeout_s=301),
                     lambda value: value["stages"][0].update(max_output_bytes=True),
                     lambda value: value["stages"][0].update(max_output_bytes=4 * 1024 * 1024 + 1),
                     lambda value: value["stages"].append(copy.deepcopy(value["stages"][0])),
                     lambda value: value["artifacts"].update(runtime_identities=[]),
                     lambda value: value["stages"][1].update(parameters={"suite": "test_arbitrary.py"})]
        for mutation in mutations:
            candidate = copy.deepcopy(self.raw)
            mutation(candidate)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                QualificationCampaign.parse(candidate)

    def test_hash_mismatch_and_native_fixture_separation(self):
        candidate = copy.deepcopy(self.raw)
        candidate["artifacts"]["config"]["sha256"] = "f" * 64
        with self.assertRaises(ValidationError):
            QualificationCampaign.parse(candidate)
        candidate = copy.deepcopy(self.raw)
        candidate["mode"] = "native"
        with self.assertRaisesRegex(ValidationError, "fixture runtime"):
            QualificationCampaign.parse(candidate)
        binary = {"path": str(Path(sys.executable)), "sha256": _file_sha(Path(sys.executable))}
        candidate = copy.deepcopy(self.raw)
        candidate["stages"] = [stage("native", "native_graph_fixture", parameters={
            "binary": binary, "source_record": candidate["artifacts"]["config"]})]
        with self.assertRaisesRegex(ValidationError, "fixture campaigns"):
            QualificationCampaign.parse(candidate)

    def test_result_directory_in_other_git_checkout_is_rejected(self):
        other = self.root / "another-repo"
        other.mkdir()
        (other / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        self.raw["results_directory"] = str(other / "private")
        with self.assertRaises(ValidationError):
            QualificationCampaign.parse(self.raw)

    def test_loader_rejects_duplicate_json_keys(self):
        self.path.write_text('{"campaign_schema":"a","campaign_schema":"b"}', encoding="utf-8")
        with self.assertRaises(ValidationError):
            load_campaign(self.path)

    def test_native_experiment_argv_is_fixed_and_projection_requires_run(self):
        for kind, args in (("native_projection_lab", ["--run"]), ("native_graph_fixture", []), ("native_platform_probe", [])):
            with self.subTest(kind=kind):
                self.assertEqual(_experiment_argv(stage("native", kind, parameters={"binary": {"path": "trusted.exe"}})), ["trusted.exe"] + args)

    def test_stage_environment_is_closed_recorded_and_reproducible(self):
        self.raw["stages"] = [stage("one")]
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "2", "LLAMA_ARG_MODEL": "unexpected",
                                    "PRIVATE_TEST_CREDENTIAL": "secret", "LD_PRELOAD": "injected",
                                    "PYTHONSTARTUP": "injected.py", "HSA_OVERRIDE_GFX_VERSION": "11.5.1"}), patch(
                "tensormeld.qualification_campaign._run_process", return_value=captured()) as runner:
            report = run_qualification_campaign(self.save())
        env = runner.call_args.kwargs["env"]
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        for name in ("PRIVATE_TEST_CREDENTIAL", "LLAMA_ARG_MODEL", "LD_PRELOAD", "PYTHONSTARTUP", "HSA_OVERRIDE_GFX_VERSION"):
            self.assertNotIn(name, env)
        self.assertEqual(report["stage_environment"]["variables"], env)
        self.assertEqual(report["stage_environment"]["sha256"], _digest(env))

    def native_worker_fixture(self):
        # Declared native metadata is deliberately still synthetic: every campaign
        # result must stay unqualified even when a supplied label says native.
        runtime_path = Path(self.raw["artifacts"]["runtime_manifest"]["path"])
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["provenance"] = "native-adapter"
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        self.raw["artifacts"]["runtime_manifest"]["sha256"] = _file_sha(runtime_path)
        self.raw["mode"] = "native"
        binary = self.root / "trusted-fixture.exe"
        binary.write_bytes(Path(sys.executable).read_bytes())
        self.raw["stages"] = [stage("native", "native_projection_lab", parameters={
            "binary": {"path": str(binary), "sha256": _file_sha(binary)},
            "source_record": self.raw["artifacts"]["config"]})]
        self.save()
        return load_campaign(self.path), binary

    def test_native_worker_has_expected_before_after_pins_and_fixed_run_argument(self):
        campaign, binary = self.native_worker_fixture()
        outcome = captured()
        outcome["stdout"] = b"projection fixtures passed; unqualified\n"
        with patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                "tensormeld.qualification_campaign._run_process", return_value=outcome) as runner:
            code = _worker(self.path, "native", campaign.fingerprint, code_identity()["source_sha256"])
        self.assertEqual(code, 0)
        self.assertEqual(runner.call_args.args[0], [str(binary), "--run"])
        payload = json.loads(out.getvalue().splitlines()[1])
        invocation = payload["native_invocations"][0]
        expected = campaign.config["stages"][0]["parameters"]["binary"]["sha256"]
        self.assertEqual(invocation["expected_artifact_sha256"], expected)
        self.assertEqual(invocation["artifact_sha256_before"], expected)
        self.assertEqual(invocation["artifact_sha256_after"], expected)
        self.assertEqual(base64.b64decode(invocation["stdout_base64"]), outcome["stdout"])
        self.assertEqual(payload["execution_source"], "trusted-local-experiment")
        self.assertFalse(payload["qualified"])

    def test_native_worker_binary_changed_during_execution_fails(self):
        campaign, binary = self.native_worker_fixture()
        def mutate(*args, **kwargs):
            binary.write_bytes(b"changed executable")
            return dict(captured(), stdout=b"PASS\n")
        with patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                "tensormeld.qualification_campaign._run_process", side_effect=mutate):
            code = _worker(self.path, "native", campaign.fingerprint, code_identity()["source_sha256"])
        self.assertEqual(code, 1)
        invocation = json.loads(out.getvalue().splitlines()[1])["native_invocations"][0]
        self.assertNotEqual(invocation["artifact_sha256_before"], invocation["artifact_sha256_after"])

    def test_native_worker_known_drift_before_launch_does_not_execute(self):
        campaign, binary = self.native_worker_fixture()
        original = campaign_module._artifact
        def mutate_after_artifact_check(value, where):
            result = original(value, where)
            if where == "native invocation binary":
                binary.write_bytes(b"known drift before launch")
            return result
        with patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                "tensormeld.qualification_campaign._artifact", side_effect=mutate_after_artifact_check), patch(
                "tensormeld.qualification_campaign._run_process") as runner:
            code = _worker(self.path, "native", campaign.fingerprint, code_identity()["source_sha256"])
        runner.assert_not_called()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue().splitlines()[1])["native_invocations"], [])

    def test_backend_bound_result_must_match_host_and_runtime_worker(self):
        self.native_worker_fixture()
        binary = self.raw["stages"][0]["parameters"]["binary"]
        bound_path = self.root / "bound-result.json"
        self.raw["stages"] = [stage("backend", "backend_self_test", parameters={
            "binary": binary, "device_id": "pc-gpu", "bound_result": {"path": str(bound_path), "sha256": "0" * 64}})]
        for node, worker, valid in (("pc", "c" * 64, True), ("helper-a", "c" * 64, False), ("pc", "f" * 64, False)):
            with self.subTest(node=node, worker=worker):
                bound_path.write_text(json.dumps({"result_schema": "tensormeld/llamacpp-device-binding-result-v1",
                                                 "node_id": node, "probe_artifact_sha256": worker}), encoding="utf-8")
                self.raw["stages"][0]["parameters"]["bound_result"]["sha256"] = _file_sha(bound_path)
                if valid:
                    QualificationCampaign.parse(self.raw)
                else:
                    with self.assertRaisesRegex(ValidationError, "host/runtime worker"):
                        QualificationCampaign.parse(self.raw)

    def ctest_fixture(self, *, properties="", command_extra="", passive=None, failing=False):
        executable = shutil.which("ctest")
        if executable is None:
            self.skipTest("explicit installed local CTest fixture executable is unavailable")
        self.native_worker_fixture()
        directory = self.root / "ctest-build"
        directory.mkdir()
        empty = directory / "empty"
        empty.mkdir()
        child = directory / ("fixture-test.exe" if os.name == "nt" else "fixture-test")
        # CTest's executable may locate resources relative to its installation;
        # a copied CTest is not a portable successful child process fixture.
        fixture_name = ("where.exe" if failing else "hostname.exe") if os.name == "nt" else ("false" if failing else "true")
        test_executable = shutil.which(fixture_name)
        if test_executable is None:
            self.skipTest("explicit local system process fixture executable is unavailable")
        shutil.copy2(test_executable, child)
        ctest_file = directory / "CTestTestfile.cmake"
        name = "strix-integer-projection-tests"
        ctest_file.write_text(f'add_test({name} "{child.as_posix()}" {command_extra})\n'
                             f'set_tests_properties({name} PROPERTIES WORKING_DIRECTORY "{empty.as_posix()}" {properties})\n', encoding="utf-8")
        ref = lambda path: {"path": str(path), "sha256": _file_sha(path)}
        controls = {"working_directory": str(empty)}
        if properties == "DISABLED TRUE":
            controls["disabled"] = True
        if properties == "TIMEOUT 0.000001":
            controls["timeout_s"] = 0.000001
        controls.update(passive or {})
        build = {"record_schema": "tensormeld/ctest-build-record-v1", "build_directory": str(directory),
                 "source_files": [ref(Path(__file__))], "build_files": [ref(ctest_file)],
                 "test_binaries": [{"name": name, "artifact": ref(child), "properties": controls}]}
        source_record = self.root / "ctest-build-record.json"
        source_record.write_text(json.dumps(build), encoding="utf-8")
        self.raw["stages"] = [stage("ctest", "native_ctest", parameters={
            "binary": ref(Path(executable)), "source_record": ref(source_record), "tests": [name]})]
        return directory, build, source_record

    def test_actual_ctest_success_has_nonzero_execution_and_private_junit(self):
        self.ctest_fixture()
        report = run_qualification_campaign(self.save())
        item = report["stages"][0]
        self.assertEqual(item["outcome"], "PASSED", item)
        detail = item["stage_result"]["detail"]
        self.assertEqual(detail["tests_executed"], 1)
        self.assertEqual(detail["tests_failed"], 0)
        self.assertEqual(detail["tests"][0]["outcome"], "PASSED")
        self.assertEqual(_file_sha(Path(report["results_directory"]) / detail["junit_file"]), detail["junit_sha256"])
        self.assertEqual(len(item["stage_result"]["native_invocations"]), 2)
        self.assertFalse(report["native_qualified"])

    def test_actual_ctest_all_disabled_is_skipped_not_passed(self):
        self.ctest_fixture(properties="DISABLED TRUE")
        report = run_qualification_campaign(self.save())
        item = report["stages"][0]
        self.assertEqual(item["outcome"], "SKIPPED", item)
        detail = item["stage_result"]["detail"]
        self.assertEqual(detail["tests_executed"], 0)
        self.assertEqual(detail["tests_skipped"], 1)

    def test_actual_ctest_timeout_is_failure_and_preserves_test_timeout(self):
        self.ctest_fixture(properties="TIMEOUT 0.000001")
        report = run_qualification_campaign(self.save())
        item = report["stages"][0]
        self.assertEqual(item["outcome"], "FAILED", item)
        detail = item["stage_result"]["detail"]
        self.assertEqual(detail["tests_failed"], 1)
        self.assertTrue(detail["tests"][0]["timed_out"])
        self.assertTrue(item["timed_out"])

    def test_ctest_original_arguments_and_environment_are_never_evaluated(self):
        for extra, properties in (("--version", ""), ("", 'ENVIRONMENT "HSA_OVERRIDE_GFX_VERSION=11.5.1"')):
            with self.subTest(extra=extra, properties=properties):
                # Each configuration needs distinct fixture directories.
                if (self.root / "ctest-build").exists():
                    shutil.rmtree(self.root / "ctest-build")
                self.ctest_fixture(command_extra=extra, properties=properties)
                report = run_qualification_campaign(self.save())
                item = report["stages"][0]
                self.assertEqual(item["outcome"], "PASSED", item)
                self.assertEqual(len(item["stage_result"]["native_invocations"]), 2)
                detail = item["stage_result"]["detail"]
                metadata = (Path(report["results_directory"]) / detail["generated_metadata_file"]).read_text()
                self.assertNotIn("--version", metadata)
                self.assertNotIn("ENVIRONMENT", metadata)
                self.assertEqual(hashlib.sha256(metadata.encode()).hexdigest(), detail["generated_metadata_sha256"])

    def test_ctest_build_metadata_binaries_and_names_are_pinned(self):
        directory, build, source_record = self.ctest_fixture()
        for mutation in (lambda params: params.update(tests=["arbitrary-test"]),
                         lambda params: params.update(tests=[]),
                         lambda params: params.update(tests=["strix-integer-projection-tests"] * 2)):
            candidate = copy.deepcopy(self.raw)
            mutation(candidate["stages"][0]["parameters"])
            with self.assertRaises(ValidationError):
                QualificationCampaign.parse(candidate)
        (directory / "CTestTestfile.cmake").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValidationError):
            QualificationCampaign.parse(self.raw)

    def test_actual_ctest_empty_generated_inventory_never_passes(self):
        self.ctest_fixture()
        def empty_inventory(argv, **kwargs):
            if "--show-only=json-v1" in argv:
                metadata = Path(argv[argv.index("--test-dir") + 1]) / "CTestTestfile.cmake"
                metadata.write_text("# injected missing selection\n", encoding="utf-8")
            return original(argv, **kwargs)
        original = campaign_module._run_process
        campaign = QualificationCampaign.parse(self.raw)
        with patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                "tensormeld.qualification_campaign._run_process", side_effect=empty_inventory):
            code = _worker(self.save(), "ctest", campaign.fingerprint, code_identity()["source_sha256"])
        self.assertEqual(code, 1)
        self.assertEqual(len(json.loads(out.getvalue().splitlines()[1])["native_invocations"]), 1)

    def test_ctest_original_execute_process_include_and_custom_files_never_execute(self):
        directory, build, source_record = self.ctest_fixture()
        marker = directory / "must-not-exist"
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("explicit local CMake marker fixture unavailable")
        command = f'execute_process(COMMAND "{Path(cmake).as_posix()}" -E touch "{marker.as_posix()}")\n'
        included = directory / "hidden-include.cmake"
        custom = directory / "CTestCustom.cmake"
        included.write_text(command, encoding="utf-8")
        custom.write_text(command, encoding="utf-8")
        metadata = directory / "CTestTestfile.cmake"
        metadata.write_text(command + f'include("{included.as_posix()}")\n' + metadata.read_text(), encoding="utf-8")
        build["build_files"] = [{"path": str(path), "sha256": _file_sha(path)} for path in (metadata, included, custom)]
        source_record.write_text(json.dumps(build), encoding="utf-8")
        self.raw["stages"][0]["parameters"]["source_record"]["sha256"] = _file_sha(source_record)
        report = run_qualification_campaign(self.save())
        self.assertEqual(report["stages"][0]["outcome"], "PASSED", report["stages"][0])
        self.assertFalse(marker.exists(), "original CMake must never execute during inventory or test dispatch")

    def test_ctest_original_will_fail_cannot_invert_native_failure(self):
        self.ctest_fixture(properties="WILL_FAIL TRUE", failing=True)
        report = run_qualification_campaign(self.save())
        item = report["stages"][0]
        self.assertEqual(item["outcome"], "FAILED", item)
        self.assertEqual(item["stage_result"]["detail"]["tests_failed"], 1)
        self.assertEqual(item["stage_result"]["detail"]["tests_executed"], 1)

    def test_ctest_outcome_transformers_or_other_unclosed_properties_rejected(self):
        _, build, source_record = self.ctest_fixture()
        original = copy.deepcopy(build)
        for property_name, value in (("WILL_FAIL", True), ("PASS_REGULAR_EXPRESSION", ".*"),
                                     ("FAIL_REGULAR_EXPRESSION", ".*"), ("SKIP_REGULAR_EXPRESSION", ".*"),
                                     ("ENVIRONMENT", "HSA_OVERRIDE_GFX_VERSION=11.5.1"),
                                     ("skip_return_code", 0), ("disabled", 1), ("timeout_s", True)):
            build = copy.deepcopy(original)
            build["test_binaries"][0]["properties"][property_name] = value
            source_record.write_text(json.dumps(build), encoding="utf-8")
            self.raw["stages"][0]["parameters"]["source_record"]["sha256"] = _file_sha(source_record)
            with self.subTest(property_name=property_name), self.assertRaises(ValidationError):
                QualificationCampaign.parse(self.raw)

    def test_ctest_unsupported_junit_root_or_error_case_cannot_pass(self):
        self.ctest_fixture()
        original = campaign_module._run_process
        for root, inner in (("unsupported", ""), ("testsuite", '<error message="fixture error"/>')):
            def mutate_junit(argv, **kwargs):
                outcome = original(argv, **kwargs)
                if "--output-junit" in argv:
                    path = Path(argv[argv.index("--output-junit") + 1])
                    path.write_text(f'<{root}><testcase name="strix-integer-projection-tests" status="run">{inner}</testcase></{root}>', encoding="utf-8")
                return outcome
            campaign = QualificationCampaign.parse(self.raw)
            with self.subTest(root=root), patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                    "tensormeld.qualification_campaign._run_process", side_effect=mutate_junit):
                code = _worker(self.save(), "ctest", campaign.fingerprint, code_identity()["source_sha256"])
            self.assertEqual(code, 1)

    def test_ctest_known_generated_metadata_drift_never_dispatches_tests(self):
        self.ctest_fixture()
        original = campaign_module._run_process
        def mutate_after_inventory(argv, **kwargs):
            outcome = original(argv, **kwargs)
            if "--show-only=json-v1" in argv:
                path = Path(argv[argv.index("--test-dir") + 1]) / "CTestTestfile.cmake"
                path.write_text("execute_process(COMMAND ignored)\n", encoding="utf-8")
            return outcome
        campaign = QualificationCampaign.parse(self.raw)
        with patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                "tensormeld.qualification_campaign._run_process", side_effect=mutate_after_inventory) as runner:
            code = _worker(self.save(), "ctest", campaign.fingerprint, code_identity()["source_sha256"])
        self.assertEqual(code, 1)
        self.assertEqual(runner.call_count, 1)

    def test_ctest_nested_git_working_directory_is_rejected(self):
        directory, _, _ = self.ctest_fixture()
        (directory / "empty" / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "outside Git"):
            QualificationCampaign.parse(self.raw)

    def test_ctest_build_directory_must_remain_outside_git(self):
        directory, _, _ = self.ctest_fixture()
        (directory / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "outside Git"):
            QualificationCampaign.parse(self.raw)

    def model_trial_fixture(self):
        from test_llamacpp_native_trial import make_model, qualification_context, PIN
        from test_planner_v2 import fixture as planner_fixture
        from tensormeld.llamacpp_native_trial import approve_single_file_gguf, build_llamacpp_native_trial_spec
        from tensormeld.native_worker import approved_worker_artifact
        gguf = self.root / "fixture.gguf"
        gguf.write_bytes(b"GGUF" + b"x" * 32)
        model = make_model(gguf)
        config, planning, candidate, adapter, placement = qualification_context(model)
        raw_config, raw_planning = planner_fixture(count=1, units=1)
        raw_config["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
        raw_planning["manifest_ref"] = model.manifest_sha256
        raw_planning["units"][0]["id"] = "blk.0"
        raw_adapter = {
            "adapter_schema": "tensormeld/adapter-capabilities-v1", "adapter_id": adapter.adapter_id,
            "engine": adapter.engine, "engine_revision": adapter.engine_revision,
            "placement": {"strategies": sorted(adapter.strategies), **{name: getattr(adapter, name) for name in (
                "exact_owner_binding", "explicit_unit_ranges", "mixed_backends", "remote_compute", "coordinator_outside_compute",
                "max_compute_devices", "max_compute_nodes", "max_segments")}},
            "route_modes": sorted(adapter.route_modes), "coordinator_nodes": [node.id for node in config.nodes],
            "devices": [{"id": device.id, "node": device.node, "backend": device.backend} for device in adapter.devices]}
        raw_model = asdict(model)
        raw_model.pop("manifest_sha256")
        raw_model["model_manifest_schema"] = "tensormeld/model-manifest-v1"
        ref = lambda path: {"path": str(path), "sha256": _file_sha(path)}
        def artifact(name, raw):
            path = self.root / (name + ".json")
            path.write_text(json.dumps(raw), encoding="utf-8")
            return ref(path)
        worker = ref(Path(sys.executable))
        raw_runtime = manifest(config, model, adapter)
        raw_runtime.update(provenance="native-adapter", worker_artifact_sha256=worker["sha256"])
        self.raw["artifacts"] = {"config": artifact("model-config", raw_config), "model_manifest": artifact("model-manifest", raw_model),
            "adapter_capabilities": artifact("model-adapter", raw_adapter), "runtime_manifest": artifact("model-runtime", raw_runtime),
            "runtime_identities": [artifact("model-identity", runtime_identity("n0", "g0", worker["sha256"]).as_record())]}
        self.raw["mode"] = "native"
        self.raw["host"]["node_id"] = "n0"
        controls = {"prompt": "bounded fixture trial", "context_tokens": planning.context_tokens, "predict_tokens": 1}
        spec = build_llamacpp_native_trial_spec(model=model, placement=placement,
            llama_cli=approved_worker_artifact(sys.executable, expected_sha256=worker["sha256"], where="fixture launcher"),
            gguf=approve_single_file_gguf(model, gguf), **controls)
        expected = b"bounded fixture stdout\n"
        reference = {"reference_schema": "tensormeld/llamacpp-e3-reference-v1", "reference_id": "campaign-fixture-reference",
            "trial_spec_sha256": spec.spec_sha256, "llama_cli_sha256": spec.llama_cli_sha256,
            "model_manifest_sha256": model.manifest_sha256, "placement_sha256": placement.fingerprint,
            "expected_stdout_sha256": hashlib.sha256(expected).hexdigest(),
            "workload": {"context_tokens": spec.context_tokens, "max_output_tokens": 1, "concurrency": 1}}
        params = {"binary": worker, "gguf": ref(gguf), "planning": artifact("trial-planning", raw_planning),
            "candidate": artifact("trial-candidate", candidate), "controls": artifact("trial-controls", controls),
            "reference": artifact("trial-reference", reference),
            "placement_binding": artifact("trial-placement-binding", {"placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
                "adapter_id": adapter.adapter_id, "adapter_capabilities_sha256": adapter.fingerprint, "source_revision": PIN,
                "native_binding_sha256": "d" * 64, "devices": [{"device_id": "g0", "engine_device_name": "CPU", "buffer_type": "CPU"}]}),
            "bound_result": artifact("trial-bound", {"result_schema": "tensormeld/llamacpp-device-binding-result-v1", "config_sha256": config.fingerprint,
                "binding_sha256": "d" * 64, "node_id": "n0", "probe_artifact_sha256": worker["sha256"],
                "resolved_mappings": [{"tensormeld_device_id": "g0", "engine_device_name": "CPU", "backend_from_config": config.devices[0].backend}]}),
            "gguf_index": artifact("trial-index", {"index_schema": "tensormeld/gguf-index-v1", "architecture": "llama", "complete_shard_set": True,
                "tensor_count": 2, "index_sha256": model.tensor_index_sha256,
                "files": [{"tensors": [{"name": "blk.0.attn_q.weight"}, {"name": "blk.0.ffn_up.weight"}]}]})}
        self.raw["stages"] = [stage("model", "native_model_trial", parameters=params)]
        return spec, expected

    def test_model_trial_rebuilds_existing_spec_and_checks_literal_reference_without_qualification(self):
        from tensormeld import llamacpp_native_trial
        original_runner = llamacpp_native_trial._default_runner
        spec, expected = self.model_trial_fixture()
        campaign = QualificationCampaign.parse(self.raw)
        outcome = dict(captured(), stdout=expected)
        with patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                "tensormeld.qualification_campaign._run_process", return_value=outcome) as runner:
            code = _worker(self.save(), "model", campaign.fingerprint, code_identity()["source_sha256"])
        self.assertEqual(code, 0)
        self.assertIs(llamacpp_native_trial._default_runner, original_runner)
        self.assertEqual(runner.call_args.args[0], list(spec.argv))
        detail = json.loads(out.getvalue().splitlines()[1])["detail"]
        self.assertTrue(detail["reference_stdout_match"])
        self.assertFalse(detail["trial"]["qualified"])
        self.assertFalse(detail["trial"]["real_model_inference"])

    def test_actual_model_trial_failed_fixture_process_cannot_pass(self):
        self.model_trial_fixture()
        # The pinned interpreter is deliberately not a model engine; its closed
        # trial arguments fail without loading a model or touching hardware.
        report = run_qualification_campaign(self.save())
        item = report["stages"][0]
        self.assertEqual(item["outcome"], "FAILED")
        self.assertFalse(item["stage_result"]["detail"]["trial"]["process_succeeded"])
        self.assertEqual(len(item["stage_result"]["native_invocations"]), 1)
        self.assertFalse(report["inference_request_authorized"])

    def test_model_trial_reference_mismatch_is_failure_even_for_successful_process(self):
        self.model_trial_fixture()
        campaign = QualificationCampaign.parse(self.raw)
        with patch("sys.stdin", io.TextIOWrapper(io.BytesIO(b"1"))), redirect_stdout(io.StringIO()) as out, patch(
                "tensormeld.qualification_campaign._run_process", return_value=dict(captured(), stdout=b"wrong output")):
            code = _worker(self.save(), "model", campaign.fingerprint, code_identity()["source_sha256"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out.getvalue().splitlines()[1])["detail"]["reference_stdout_match"])

    def test_model_trial_reference_controls_and_model_drift_fail_before_launch(self):
        self.model_trial_fixture()
        original = copy.deepcopy(self.raw)
        for key, mutation in (("reference", lambda value: value.update(trial_spec_sha256="f" * 64)),
                              ("controls", lambda value: value.update(argv=["arbitrary"])),
                              ("controls", lambda value: value.update(context_tokens=1)),
                              ("controls", lambda value: value.update(predict_tokens=True))):
            self.raw = copy.deepcopy(original)
            value_ref = self.raw["stages"][0]["parameters"][key]
            value_path = Path(value_ref["path"])
            before = value_path.read_bytes()
            value = json.loads(before)
            mutation(value)
            value_path.write_text(json.dumps(value), encoding="utf-8")
            value_ref["sha256"] = _file_sha(value_path)
            try:
                with self.subTest(key=key), self.assertRaises(ValidationError):
                    QualificationCampaign.parse(self.raw)
            finally:
                value_path.write_bytes(before)
        self.raw = original
        gguf = Path(self.raw["stages"][0]["parameters"]["gguf"]["path"])
        gguf.write_bytes(b"changed GGUF")
        with self.assertRaises(ValidationError):
            QualificationCampaign.parse(self.raw)


class BoundedProcessTests(unittest.TestCase):
    def assert_stopped(self, pid):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x00100000, False, pid)
            if handle:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
                finally:
                    kernel.CloseHandle(handle)
        else:
            if sys.platform == "linux" and hasattr(os, "pidfd_open"):
                import select
                try:
                    descriptor = os.pidfd_open(pid)
                except ProcessLookupError:
                    return
                try:
                    self.assertTrue(select.select([descriptor], [], [], 5)[0])
                finally:
                    os.close(descriptor)
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists():
                self.assertEqual(stat.read_text().split()[2], "Z")

    def test_actual_timeout_stops_gated_child_and_inherited_descendant(self):
        # Fixtures only: no configurable campaign command executes Python -c.
        child = "import sys,time;sys.stdout.write('descendant\\n');sys.stdout.flush();time.sleep(60)"
        script = ("import subprocess,sys,time;sys.stdin.buffer.read(1);"
                  f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);"
                  "print(p.pid,flush=True);time.sleep(60)")
        outcome = _run_process([sys.executable, "-c", script], timeout_s=1, output_limit=4096, gated=True)
        self.assertTrue(outcome["timed_out"])
        self.assertTrue(outcome["child_exit_confirmed"])
        self.assertTrue(outcome["pipes_closed"])
        pid = int(outcome["stdout"].splitlines()[0])
        self.assert_stopped(pid)

    def test_ordinary_stage_exit_stops_live_descendant_and_closes_inherited_pipes(self):
        child = "import time;time.sleep(60)"
        script = ("import subprocess,sys;sys.stdin.buffer.read(1);"
                  f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);print(p.pid,flush=True)")
        outcome = _run_process([sys.executable, "-c", script], timeout_s=10, output_limit=4096, gated=True)
        self.assertEqual(outcome["returncode"], 0)
        self.assertFalse(outcome["timed_out"])
        self.assertTrue(outcome["pipes_closed"])
        self.assert_stopped(int(outcome["stdout"].splitlines()[0]))

    def test_actual_output_overflow_is_bounded_and_cleanup_is_confirmed(self):
        script = "import sys,os;sys.stdin.buffer.read(1);os.write(1,b'x'*2000000)"
        outcome = _run_process([sys.executable, "-c", script], timeout_s=10, output_limit=1024, gated=True)
        self.assertTrue(outcome["output_limit_exceeded"])
        self.assertEqual(outcome["failure_reason"], "OUTPUT_LIMIT")
        self.assertLessEqual(len(outcome["stdout"]) + len(outcome["stderr"]), 1024)
        self.assertGreater(outcome["output_bytes_observed"], 1024)
        self.assertTrue(outcome["child_exit_confirmed"])
        self.assertTrue(outcome["pipes_closed"])

    def test_containment_failure_prevents_gate_and_closes_child(self):
        if os.name != "nt":
            self.skipTest("Windows job-object failure gate")
        with patch("tensormeld.qualification_campaign._WindowsJob", side_effect=ValidationError("job unavailable")), self.assertRaises(ValidationError):
            _run_process([sys.executable, "-c", "import sys;sys.stdin.buffer.read(1);raise RuntimeError('gate leaked')"], timeout_s=10, output_limit=4096, gated=True)

    def test_campaign_parent_death_stops_active_worker_and_descendant(self):
        # A loopback notification makes parent death event-controlled: the parent
        # is terminated only after the guarded worker has spawned its descendant.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(10)
            address = listener.getsockname()
            child = "import time;time.sleep(60)"
            worker = ("import os,sys,subprocess,socket,time,json;"
                      "from tensormeld.qualification_campaign import _install_parent_death_guard;"
                      "_install_parent_death_guard(os.getppid());sys.stdin.buffer.read(1);"
                      f"p=subprocess.Popen([sys.executable,'-c',{child!r}]);"
                      f"s=socket.create_connection({address!r});"
                      "s.sendall(json.dumps([os.getpid(),p.pid]).encode()+b'\\n');s.close();time.sleep(60)")
            helper = ("import sys;from tensormeld.qualification_campaign import _run_process;"
                      f"_run_process([sys.executable,'-c',{worker!r}],timeout_s=120,output_limit=4096,gated=True)")
            env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
            process = subprocess.Popen([sys.executable, "-c", helper], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
            try:
                with listener.accept()[0] as connection:
                    connection.settimeout(10)
                    payload = connection.recv(1024)
                worker_pid, child_pid = json.loads(payload)
                process.kill()
                process.wait(timeout=10)
                self.assert_stopped(worker_pid)
                self.assert_stopped(child_pid)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
