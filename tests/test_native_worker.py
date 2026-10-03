from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

from tensormeld.native_worker import (
    NativeSubprocessWholeBlockBackend,
    approved_worker_artifact,
)
from tensormeld.schema import ValidationError
from tensormeld.whole_block_execution import (
    AcceptedExecutionBundle,
    ReferenceWholeBlockSession,
)

FIXTURE = Path(__file__).parent / "fixtures" / "native_worker_fixture.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def bundle(worker_sha: str) -> AcceptedExecutionBundle:
    return AcceptedExecutionBundle(
        config_sha256="1" * 64,
        profile="interactive",
        planning_input_sha256="2" * 64,
        plan_sha256="3" * 64,
        representability_sha256="4" * 64,
        adapter_id="fixture-native",
        adapter_capabilities_sha256="5" * 64,
        model_manifest_sha256="6" * 64,
        qualification_evidence_sha256="7" * 64,
        runtime_manifest_sha256="8" * 64,
        worker_artifact_sha256=worker_sha,
        backend_readiness=(("g0", "9" * 64, "a" * 64),),
        launch_leases=(("n0", "lease-n0", "b" * 64),),
        segments=(("g0", 0, 2),),
        unit_ids=("block.0", "block.1"),
        compute_devices=("g0",),
        compute_nodes=("n0",),
        bundle_sha256="c" * 64,
    )


class NativeSubprocessWholeBlockTests(unittest.TestCase):
    def setUp(self):
        self.worker_sha = sha256(FIXTURE)
        self.executable = approved_worker_artifact(
            sys.executable,
            expected_sha256=sha256(Path(sys.executable)),
            where="python launcher",
        )
        self.program = approved_worker_artifact(
            FIXTURE,
            expected_sha256=self.worker_sha,
            where="fixture worker program",
        )

    def backend(self, **changes):
        kwargs = {
            "bundle": bundle(self.worker_sha),
            "adapter_id": "fixture-native",
            "engine_revision": "fixture-revision",
            "worker_artifact_sha256": self.worker_sha,
            "executable": self.executable,
            "program": self.program,
            "timeout_s": 5.0,
        }
        kwargs.update(changes)
        return NativeSubprocessWholeBlockBackend(**kwargs)

    def test_real_subprocess_executes_exact_segment_protocol(self):
        backend = self.backend()
        session = ReferenceWholeBlockSession(bundle(self.worker_sha), backend)
        result = session.run(b"seed")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["output"], b"seed|g0:block.0,block.1")
        self.assertFalse(result["real_model_inference"])
        self.assertEqual(session.release()["status"], "RELEASED")

    def test_launched_worker_must_match_bundle_worker_identity(self):
        wrong_bundle = bundle("d" * 64)
        with self.assertRaises(ValidationError):
            NativeSubprocessWholeBlockBackend(
                bundle=wrong_bundle,
                adapter_id="fixture-native",
                engine_revision="fixture-revision",
                worker_artifact_sha256="d" * 64,
                executable=self.executable,
                program=self.program,
            )

    def test_artifact_hash_mismatch_is_rejected_before_launch(self):
        with self.assertRaises(ValidationError):
            approved_worker_artifact(
                FIXTURE,
                expected_sha256="0" * 64,
                where="fixture worker program",
            )

    def test_nonzero_worker_exit_is_rejected(self):
        def runner(argv, stdin, timeout):
            return 7, b"", b"failure"

        backend = self.backend(runner=runner)
        with self.assertRaises(ValidationError):
            backend.execute_segment(
                device_id="g0",
                unit_ids=("block.0",),
                payload=b"x",
            )

    def test_response_identity_tampering_is_rejected(self):
        def runner(argv, stdin, timeout):
            import base64, json
            request = json.loads(stdin)
            response = {
                "worker_protocol": request["worker_protocol"],
                "adapter_id": request["adapter_id"],
                "engine_revision": request["engine_revision"],
                "worker_artifact_sha256": request["worker_artifact_sha256"],
                "bundle_sha256": "0" * 64,
                "request_sha256": "0" * 64,
                "segment_sha256": request["segment_sha256"],
                "status": "ok",
                "output_b64": base64.b64encode(b"x").decode(),
                "real_model_inference": False,
            }
            return 0, json.dumps(response).encode(), b""

        backend = self.backend(runner=runner)
        with self.assertRaises(ValidationError):
            backend.execute_segment(
                device_id="g0",
                unit_ids=("block.0",),
                payload=b"x",
            )

    def test_worker_cannot_self_claim_real_model_inference(self):
        def runner(argv, stdin, timeout):
            import base64, hashlib, json
            request = json.loads(stdin)
            request_sha = hashlib.sha256(
                json.dumps(
                    request,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            response = {
                "worker_protocol": request["worker_protocol"],
                "adapter_id": request["adapter_id"],
                "engine_revision": request["engine_revision"],
                "worker_artifact_sha256": request["worker_artifact_sha256"],
                "bundle_sha256": request["bundle_sha256"],
                "request_sha256": request_sha,
                "segment_sha256": request["segment_sha256"],
                "status": "ok",
                "output_b64": base64.b64encode(b"x").decode(),
                "real_model_inference": True,
            }
            return 0, json.dumps(response).encode(), b""

        backend = self.backend(runner=runner)
        with self.assertRaises(ValidationError):
            backend.execute_segment(
                device_id="g0",
                unit_ids=("block.0",),
                payload=b"x",
            )


if __name__ == "__main__":
    unittest.main()
