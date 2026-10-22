from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import socket
import sys
import tempfile
import unittest

from tensormeld.admission import LocalAdmissionController
from tensormeld.llamacpp_admitted_server import (
    AdmittedManagedLlamaCppServer,
    bind_admitted_llamacpp_server,
)
from tensormeld.llamacpp_managed_server import build_llamacpp_server_launch_spec
from tensormeld.llamacpp_native_trial import approve_single_file_gguf
from tensormeld.llamacpp_package import build_llamacpp_package_identity
from tensormeld.llamacpp_placement import LlamaCppPlacementTranslation
from tensormeld.native_worker import approved_worker_artifact
from tensormeld.schema import ValidationError
from tensormeld.target_host_admission import orchestrate_target_host_admission
from test_llamacpp_managed_server import FAIL_FIXTURE, FIXTURE
from test_llamacpp_native_trial import sha256
from test_target_host_admission import TargetHostAdmissionOrchestratorTests

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def probe(artifact_sha: str, name: str, *, build=777, compiler="fixture-cc"):
    return {
        "probe_schema": "tensormeld/llamacpp-probe-v1",
        "engine": "llama.cpp",
        "pinned_source_revision": PIN,
        "artifact_sha256": artifact_sha,
        "binary_name": name,
        "build": {
            "version": "fixture-version",
            "build": build,
            "commit": PIN[:8],
            "compiler": compiler,
            "target": "fixture-target",
        },
        "devices": [],
        "device_identity_mapping": "unresolved",
        "model_loaded": False,
        "listener_started": False,
        "qualified": False,
        "executable": False,
        "warnings": [],
    }


class LlamaCppPackageIdentityTests(unittest.TestCase):
    def test_exact_sibling_builds_form_deterministic_package(self):
        package = build_llamacpp_package_identity(
            cli_probe=probe("1" * 64, "llama-cli"),
            server_probe=probe("2" * 64, "llama-server"),
            backend_libraries=[
                {"name": "ggml-hip", "sha256": "3" * 64},
                {"name": "ggml-base", "sha256": "4" * 64},
            ],
        )
        self.assertEqual(package.llama_cli_sha256, "1" * 64)
        self.assertEqual(package.llama_server_sha256, "2" * 64)
        self.assertEqual(
            package.backend_libraries,
            (("ggml-base", "4" * 64), ("ggml-hip", "3" * 64)),
        )
        self.assertEqual(
            package,
            build_llamacpp_package_identity(
                cli_probe=probe("1" * 64, "llama-cli"),
                server_probe=probe("2" * 64, "llama-server"),
                backend_libraries=[
                    {"name": "ggml-hip", "sha256": "3" * 64},
                    {"name": "ggml-base", "sha256": "4" * 64},
                ],
            ),
        )

    def test_mismatched_build_metadata_is_rejected(self):
        with self.assertRaises(ValidationError):
            build_llamacpp_package_identity(
                cli_probe=probe("1" * 64, "llama-cli", build=777),
                server_probe=probe("2" * 64, "llama-server", build=778),
            )

    def test_cli_and_server_remain_distinct_artifacts(self):
        with self.assertRaises(ValidationError):
            build_llamacpp_package_identity(
                cli_probe=probe("1" * 64, "llama-cli"),
                server_probe=probe("1" * 64, "llama-server"),
            )

    def test_duplicate_backend_library_identity_is_rejected(self):
        with self.assertRaises(ValidationError):
            build_llamacpp_package_identity(
                cli_probe=probe("1" * 64, "llama-cli"),
                server_probe=probe("2" * 64, "llama-server"),
                backend_libraries=[
                    {"name": "ggml-a", "sha256": "3" * 64},
                    {"name": "ggml-b", "sha256": "3" * 64},
                ],
            )


class AdmittedManagedLlamaCppServerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TargetHostAdmissionOrchestratorTests(
            methodName="test_reserve_fresh_recheck_builds_accepted_bundle_without_starting_inference"
        )
        self.fixture.setUp()
        self.controller = LocalAdmissionController()
        reserve, launch = self.fixture.snapshots()
        self.admission = orchestrate_target_host_admission(
            controller=self.controller,
            lease_id="server-package-lease",
            config=self.fixture.config,
            planning=self.fixture.planning,
            candidate=self.fixture.candidate,
            adapter=self.fixture.adapter,
            model=self.fixture.model,
            handoff=self.fixture.handoff,
            collection=self.fixture.collection,
            reservation_snapshot=reserve,
            launch_snapshot=launch,
        )
        self.tmp = tempfile.TemporaryDirectory()
        gguf_path = Path(self.tmp.name) / "fixture.gguf"
        gguf_path.write_bytes(b"GGUF" + b"x" * 32)
        self.gguf = approve_single_file_gguf(self.fixture.model, gguf_path)
        self.launcher = approved_worker_artifact(
            sys.executable,
            expected_sha256=sha256(Path(sys.executable)),
            where="python launcher",
        )

    def tearDown(self):
        for lease in self.controller.active_leases():
            self.controller.release(lease.lease_id)
        self.tmp.cleanup()
        self.fixture.tearDown()

    def server_artifact(self, path=FIXTURE):
        return approved_worker_artifact(
            path,
            expected_sha256=sha256(path),
            where="fixture llama-server",
        )

    def package(self, path=FIXTURE):
        server = self.server_artifact(path)
        return build_llamacpp_package_identity(
            cli_probe=probe(
                self.admission.bundle.worker_artifact_sha256,
                "llama-cli",
            ),
            server_probe=probe(server.sha256, "llama-server"),
            backend_libraries=[
                {"name": "ggml-fixture", "sha256": "3" * 64}
            ],
        )

    def placement(self):
        bundle = self.admission.bundle
        device = bundle.compute_devices[0]
        owners = tuple(
            (index, device, "CPU")
            for index, _ in enumerate(bundle.unit_ids)
        )
        overrides = ",".join(
            rf"^blk\.{index}\..*=CPU"
            for index, _ in enumerate(bundle.unit_ids)
        )
        return LlamaCppPlacementTranslation(
            source_revision=PIN,
            accepted_bundle_sha256=bundle.bundle_sha256,
            placement_binding_sha256="4" * 64,
            block_owners=owners,
            device_buffer_types=((device, "CPU"),),
            override_tensor_value=overrides,
            argv_fragment=(
                "--fit", "off",
                "--device", "CPU",
                "--override-tensor", overrides,
            ),
            fingerprint="5" * 64,
        )

    def spec(self, path=FIXTURE):
        return build_llamacpp_server_launch_spec(
            bundle=self.admission.bundle,
            placement=self.placement(),
            model=self.fixture.model,
            llama_server=self.server_artifact(path),
            launcher=self.launcher,
            gguf=self.gguf,
            context_tokens=128,
            port=free_port(),
        )

    def test_same_build_package_bridges_qualified_cli_and_server_artifacts(self):
        package = self.package()
        binding = bind_admitted_llamacpp_server(
            controller=self.controller,
            admission=self.admission,
            package=package,
            server_spec=self.spec(),
        )
        self.assertEqual(
            binding.llama_cli_sha256,
            self.admission.bundle.worker_artifact_sha256,
        )
        self.assertEqual(
            binding.llama_server_sha256,
            package.llama_server_sha256,
        )
        record = binding.as_record()
        self.assertTrue(record["qualified_cli_artifact"])
        self.assertTrue(record["server_same_build_package"])
        self.assertFalse(record["server_semantic_equivalence_qualified"])
        self.assertFalse(record["inference_request_authorized"])

    def test_real_server_keeps_lease_until_confirmed_stop(self):
        server = AdmittedManagedLlamaCppServer(
            controller=self.controller,
            admission=self.admission,
            package=self.package(),
            server_spec=self.spec(),
        )
        server.start()
        server.wait_ready(timeout_s=5, poll_s=0.05)
        self.assertEqual(server.state, "ready")
        self.assertEqual(len(self.controller.active_leases()), 1)

        stopped = server.stop(terminate_timeout_s=2, kill_timeout_s=2)
        self.assertEqual(stopped["status"], "STOPPED")
        self.assertEqual(stopped["lease_release_status"], "RELEASED")
        self.assertFalse(stopped["inference_request_authorized"])
        self.assertFalse(stopped["real_model_inference"])
        self.assertEqual(self.controller.active_leases(), ())

    def test_server_artifact_outside_package_is_rejected(self):
        wrong = replace(
            self.package(),
            llama_server_sha256="0" * 64,
        )
        with self.assertRaises(ValidationError):
            bind_admitted_llamacpp_server(
                controller=self.controller,
                admission=self.admission,
                package=wrong,
                server_spec=self.spec(),
            )
        self.assertEqual(len(self.controller.active_leases()), 1)

    def test_package_cli_must_be_exact_qualified_bundle_worker(self):
        package = build_llamacpp_package_identity(
            cli_probe=probe("0" * 64, "llama-cli"),
            server_probe=probe(self.server_artifact().sha256, "llama-server"),
        )
        with self.assertRaises(ValidationError):
            bind_admitted_llamacpp_server(
                controller=self.controller,
                admission=self.admission,
                package=package,
                server_spec=self.spec(),
            )

    def test_early_server_exit_releases_lease_after_process_cleanup(self):
        server = AdmittedManagedLlamaCppServer(
            controller=self.controller,
            admission=self.admission,
            package=self.package(FAIL_FIXTURE),
            server_spec=self.spec(FAIL_FIXTURE),
        )
        server.start()
        with self.assertRaises(ValidationError):
            server.wait_ready(timeout_s=3, poll_s=0.05)
        self.assertEqual(server.state, "failed")
        self.assertEqual(self.controller.active_leases(), ())


if __name__ == "__main__":
    unittest.main()
