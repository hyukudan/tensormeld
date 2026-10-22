from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.runtime_identity import RuntimeIdentity, load_runtime_identity
from tensormeld.schema import ValidationError


def identity_data(**changes):
    data = {
        "runtime_identity_schema": "tensormeld/runtime-identity-v1",
        "node_id": "node-a",
        "worker_artifact_sha256": "1" * 64,
        "worker_build_id": "worker-build-1",
        "os_name": "linux",
        "os_version": "fixture-os",
        "driver_id": "fixture-driver",
        "driver_version": "1",
        "runtime_id": "fixture-runtime",
        "runtime_version": "2",
        "tensormeld_device_id": "gpu0",
        "physical_device_id": "pci:0000:01:00.0",
        "topology_sha256": "2" * 64,
    }
    data.update(changes)
    return data


class RuntimeIdentityTests(unittest.TestCase):
    def test_identity_is_deterministic_and_exact(self):
        a = RuntimeIdentity.parse(identity_data())
        b = RuntimeIdentity.parse(identity_data())
        self.assertEqual(a.identity_sha256, b.identity_sha256)
        changed = RuntimeIdentity.parse(identity_data(driver_version="2"))
        self.assertNotEqual(a.identity_sha256, changed.identity_sha256)

    def test_loader_accepts_fingerprint_and_rejects_tampering(self):
        identity = RuntimeIdentity.parse(identity_data())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "identity.json"
            p.write_text(json.dumps(identity.as_record()), encoding="utf-8")
            self.assertEqual(
                load_runtime_identity(p).identity_sha256,
                identity.identity_sha256,
            )
            tampered = identity.as_record()
            tampered["driver_version"] = "different"
            p.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_runtime_identity(p)

    def test_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "identity.json"
            p.write_text('{"runtime_identity_schema":"x","runtime_identity_schema":"y"}')
            with self.assertRaises(ValidationError):
                load_runtime_identity(p)


if __name__ == "__main__":
    unittest.main()
