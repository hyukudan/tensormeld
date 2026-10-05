from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest

from tensormeld.llamacpp_managed_server import (
    ManagedLlamaCppServer,
    build_llamacpp_server_launch_spec,
)
from tensormeld.llamacpp_native_trial import approve_single_file_gguf
from tensormeld.llamacpp_placement import LlamaCppPlacementTranslation
from tensormeld.native_worker import approved_worker_artifact
from tensormeld.schema import ValidationError
from test_llamacpp_native_trial import make_model, sha256
from test_llamacpp_placement import adapter as placement_adapter, bundle as placement_bundle

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"
FIXTURE = Path(__file__).parent / "fixtures" / "llamacpp_server_fixture.py"
FAIL_FIXTURE = Path(__file__).parent / "fixtures" / "llamacpp_server_fail_fixture.py"
STUBBORN_FIXTURE = Path(__file__).parent / "fixtures" / "llamacpp_server_stubborn_fixture.py"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ManagedLlamaCppServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.gguf_path = root / "fixture.gguf"
        self.gguf_path.write_bytes(b"GGUF" + b"x" * 32)
        self.model = make_model(self.gguf_path)
        self.gguf = approve_single_file_gguf(self.model, self.gguf_path)
        self.adapter = placement_adapter()
        self.bundle = replace(
            placement_bundle(self.adapter),
            model_manifest_sha256=self.model.manifest_sha256,
        )
        self.placement = LlamaCppPlacementTranslation(
            source_revision=PIN,
            accepted_bundle_sha256=self.bundle.bundle_sha256,
            placement_binding_sha256="1" * 64,
            block_owners=((0, "g0", "CPU"),),
            device_buffer_types=(("g0", "CPU"),),
            override_tensor_value=r"^blk\.0\..*=CPU",
            argv_fragment=(
                "--fit",
                "off",
                "--device",
                "CPU",
                "--override-tensor",
                r"^blk\.0\..*=CPU",
            ),
            fingerprint="2" * 64,
        )
        self.launcher = approved_worker_artifact(
            sys.executable,
            expected_sha256=sha256(Path(sys.executable)),
            where="python launcher",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def artifact(self, path: Path):
        return approved_worker_artifact(
            path,
            expected_sha256=sha256(path),
            where="fixture server program",
        )

    def spec(self, path=FIXTURE):
        return build_llamacpp_server_launch_spec(
            bundle=self.bundle,
            placement=self.placement,
            model=self.model,
            llama_server=self.artifact(path),
            launcher=self.launcher,
            gguf=self.gguf,
            context_tokens=128,
            port=free_port(),
        )

    def test_launch_spec_is_closed_loopback_and_identity_bound(self):
        spec = self.spec()
        self.assertEqual(spec.source_revision, PIN)
        self.assertEqual(spec.accepted_bundle_sha256, self.bundle.bundle_sha256)
        self.assertEqual(spec.model_manifest_sha256, self.model.manifest_sha256)
        self.assertEqual(spec.gguf_sha256, self.gguf.sha256)
        self.assertEqual(spec.host, "127.0.0.1")
        self.assertIn("--parallel", spec.argv)
        self.assertEqual(spec.argv[spec.argv.index("--parallel") + 1], "1")
        self.assertNotIn("--rpc", spec.argv)
        self.assertNotIn("0.0.0.0", spec.argv)
        self.assertEqual(len(spec.spec_sha256), 64)

    def test_real_fixture_server_reaches_health_and_stops_cleanly(self):
        spec = self.spec()
        old = os.environ.get("LLAMA_ARG_HOST")
        os.environ["LLAMA_ARG_HOST"] = "0.0.0.0"
        server = ManagedLlamaCppServer(spec, log_lines=8)
        try:
            started = server.start()
            self.assertEqual(started["status"], "STARTED")
            ready = server.wait_ready(timeout_s=5, poll_s=0.05)
            self.assertEqual(ready["status"], "READY")
            self.assertTrue(ready["ready"])
            self.assertIsNotNone(server.pid)
            self.assertTrue(
                any("fixture llama-server starting" in line for line in server.log_tail())
            )
            stopped = server.stop(terminate_timeout_s=2, kill_timeout_s=2)
            self.assertEqual(stopped["status"], "STOPPED")
            self.assertTrue(stopped["terminated"])
            self.assertFalse(stopped["real_model_inference"])
            self.assertEqual(server.state, "stopped")
            self.assertEqual(server.stop()["status"], "ALREADY_STOPPED")
        finally:
            try:
                server.stop()
            except Exception:
                pass
            if old is None:
                os.environ.pop("LLAMA_ARG_HOST", None)
            else:
                os.environ["LLAMA_ARG_HOST"] = old

    def test_early_exit_reports_bounded_log_tail(self):
        server = ManagedLlamaCppServer(self.spec(FAIL_FIXTURE), log_lines=4)
        server.start()
        with self.assertRaises(ValidationError) as cm:
            server.wait_ready(timeout_s=3, poll_s=0.05)
        self.assertIn("code 23", str(cm.exception))
        self.assertTrue(
            any("failed to initialize backend" in line for line in server.log_tail())
        )
        self.assertLessEqual(len(server.log_tail()), 4)
        server.stop()

    @unittest.skipIf(os.name == "nt", "TerminateProcess cannot be ignored on Windows")
    def test_stubborn_process_falls_back_to_kill(self):
        server = ManagedLlamaCppServer(self.spec(STUBBORN_FIXTURE))
        server.start()
        server.wait_ready(timeout_s=5, poll_s=0.05)
        stopped = server.stop(terminate_timeout_s=0.2, kill_timeout_s=2)
        self.assertTrue(stopped["terminated"])
        self.assertTrue(stopped["killed"])
        self.assertEqual(server.state, "stopped")

    def test_wrong_model_manifest_is_rejected_before_launch(self):
        other_path = Path(self.tmp.name) / "other.gguf"
        other_path.write_bytes(b"GGUF" + b"y" * 32)
        other_model = make_model(other_path)
        with self.assertRaises(ValidationError):
            build_llamacpp_server_launch_spec(
                bundle=self.bundle,
                placement=self.placement,
                model=other_model,
                llama_server=self.artifact(FIXTURE),
                launcher=self.launcher,
                gguf=approve_single_file_gguf(other_model, other_path),
                context_tokens=128,
                port=free_port(),
            )

    def test_placement_from_another_bundle_is_rejected(self):
        bad = replace(
            self.placement,
            accepted_bundle_sha256="0" * 64,
        )
        with self.assertRaises(ValidationError):
            build_llamacpp_server_launch_spec(
                bundle=self.bundle,
                placement=bad,
                model=self.model,
                llama_server=self.artifact(FIXTURE),
                launcher=self.launcher,
                gguf=self.gguf,
                context_tokens=128,
                port=free_port(),
            )


if __name__ == "__main__":
    unittest.main()
