from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from tensormeld.llamacpp_probe import (
    LLAMACPP_PINNED_COMMIT,
    MAX_OUTPUT_BYTES,
    probe_llamacpp,
)
from tensormeld.schema import ValidationError


VERSION = (
    b"version: b9999 (build 9999, commit 552f18f9)\n"
    b"built with clang 20.1 for x86_64-pc-linux-gnu\n"
)
DEVICES = (
    b"Available devices:\n"
    b"  CUDA0: NVIDIA RTX Example (12288 MiB, 10000 MiB free)\n"
    b"  ROCm0: AMD Example (65536 MiB, 60000 MiB free)\n"
)


class FixtureRunner:
    def __init__(self, version=VERSION, devices=DEVICES, rc=0):
        self.version = version
        self.devices = devices
        self.rc = rc
        self.calls = []

    def __call__(self, argv, timeout):
        self.calls.append((tuple(argv), timeout))
        if argv[-1] == "--version":
            return self.rc, b"", self.version
        return self.rc, self.devices, b""


class LlamaCppProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.binary = Path(self.tmp.name) / "llama-cli"
        self.binary.write_bytes(b"fixture-native-binary")
        self.sha = hashlib.sha256(self.binary.read_bytes()).hexdigest()

    def probe(self, runner=None, **kwargs):
        return probe_llamacpp(
            self.binary,
            trusted_local_binary=True,
            runner=runner or FixtureRunner(),
            **kwargs,
        )

    def test_probe_hashes_artifact_and_parses_bounded_observations(self):
        runner = FixtureRunner()
        result = self.probe(runner)
        self.assertEqual(result["artifact_sha256"], self.sha)
        self.assertEqual(result["build"]["commit"], "552f18f9")
        self.assertEqual(result["build"]["target"], "x86_64-pc-linux-gnu")
        self.assertEqual(len(result["devices"]), 2)
        self.assertEqual(result["devices"][0]["total_bytes"], 12288 * 1024**2)
        self.assertEqual(
            [call[0][-1] for call in runner.calls], ["--version", "--list-devices"]
        )
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["listener_started"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["executable"])

    def test_explicit_trust_required(self):
        with self.assertRaises(ValidationError):
            probe_llamacpp(self.binary, runner=FixtureRunner())

    def test_expected_artifact_hash_is_enforced(self):
        result = self.probe(expected_artifact_sha256=self.sha)
        self.assertEqual(result["artifact_sha256"], self.sha)
        with self.assertRaises(ValidationError):
            self.probe(expected_artifact_sha256="0" * 64)

    def test_wrong_source_revision_rejected(self):
        runner = FixtureRunner(
            version=(
                b"version: b1 (build 1, commit deadbee)\n"
                b"built with cc for target\n"
            )
        )
        with self.assertRaises(ValidationError):
            self.probe(runner)

    def test_full_pinned_commit_is_accepted(self):
        runner = FixtureRunner(
            version=(
                f"version: b1 (build 1, commit {LLAMACPP_PINNED_COMMIT})\n"
                "built with cc for target\n"
            ).encode()
        )
        self.assertEqual(self.probe(runner)["build"]["commit"], LLAMACPP_PINNED_COMMIT)

    def test_unrecognized_device_output_is_rejected(self):
        runner = FixtureRunner(devices=b"Available devices:\n  surprise\n")
        with self.assertRaises(ValidationError):
            self.probe(runner)

    def test_no_device_build_is_valid_observation(self):
        runner = FixtureRunner(devices=b"Available devices:\n  (none)\n")
        self.assertEqual(self.probe(runner)["devices"], [])

    def test_duplicate_engine_device_names_rejected(self):
        line = b"  CUDA0: Example (100 MiB, 50 MiB free)\n"
        runner = FixtureRunner(devices=b"Available devices:\n" + line + line)
        with self.assertRaises(ValidationError):
            self.probe(runner)

    def test_impossible_memory_values_rejected(self):
        runner = FixtureRunner(
            devices=b"Available devices:\n  CUDA0: Example (100 MiB, 101 MiB free)\n"
        )
        with self.assertRaises(ValidationError):
            self.probe(runner)

    def test_nonzero_exit_rejected(self):
        with self.assertRaises(ValidationError):
            self.probe(FixtureRunner(rc=2))

    def test_output_bound_enforced(self):
        runner = FixtureRunner(version=b"x" * (MAX_OUTPUT_BYTES + 1))
        with self.assertRaises(ValidationError):
            self.probe(runner)

    def test_timeout_bound_is_enforced_before_execution(self):
        with self.assertRaises(ValidationError):
            self.probe(timeout_s=100)

    def test_engine_names_are_not_promoted_to_tensormeld_ids(self):
        result = self.probe()
        self.assertEqual(result["device_identity_mapping"], "unresolved")
        self.assertNotIn("backend", result["devices"][0])
