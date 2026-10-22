from __future__ import annotations

import hashlib
import unittest

from tensormeld.agent import Enrollment, HostAgent
from tensormeld.private_transport import bind_enrollment_to_peer_certificate
from tensormeld.remote_control import (
    LocalObjectRegistry,
    RemoteAgentDispatcher,
    private_endpoint,
    validate_endpoint_binding,
)
from tensormeld.schema import ValidationError
from test_agent import setup
from test_admission import snapshot


class RemoteControlTests(unittest.TestCase):
    def test_private_endpoint_rejects_public_hostname_and_link_local(self):
        sha = "a" * 64
        ep = private_endpoint(
            node_id="n", enrollment_id="e", address="10.1.2.3", port=7443,
            certificate_sha256=sha,
        )
        self.assertEqual(ep.address, "10.1.2.3")
        for address in ("8.8.8.8", "example.com", "169.254.1.2"):
            with self.subTest(address=address), self.assertRaises(ValidationError):
                private_endpoint(
                    node_id="n", enrollment_id="e", address=address, port=7443,
                    certificate_sha256=sha,
                )

    def test_endpoint_must_match_enrollment_and_certificate_binding(self):
        enrollment = Enrollment("e", "n", "k")
        binding = bind_enrollment_to_peer_certificate(
            enrollment, certificate_sha256="a" * 64
        )
        endpoint = private_endpoint(
            node_id="n", enrollment_id="e", address="127.0.0.1", port=7443,
            certificate_sha256="a" * 64,
        )
        validate_endpoint_binding(endpoint, enrollment=enrollment, binding=binding)
        wrong = private_endpoint(
            node_id="n", enrollment_id="e", address="127.0.0.1", port=7443,
            certificate_sha256="b" * 64,
        )
        with self.assertRaises(ValidationError):
            validate_endpoint_binding(wrong, enrollment=enrollment, binding=binding)

    def test_dispatch_uses_only_locally_registered_manifest_and_snapshot(self):
        cfg, manifest, enrollment, secret = setup()
        agent = HostAgent(config=cfg, enrollment=enrollment, shared_secret=secret)
        registry = LocalObjectRegistry()
        registry.register_manifest(manifest)
        snap = snapshot(cfg, "obs-1")
        registry.register_snapshot("obs-1", snap)
        dispatcher = RemoteAgentDispatcher(agent, registry)
        result = dispatcher.dispatch("reserve", {
            "request_schema": "tensormeld/remote-agent-request-v1",
            "request_id": "r1",
            "args": {
                "lease_id": "lease",
                "runtime_manifest_sha256": manifest.fingerprint,
                "observation_id": "obs-1",
            },
        })
        self.assertEqual(result["result"]["status"], "RESERVED")
        self.assertTrue(result["result"]["reservation_created"])

        with self.assertRaises(ValidationError):
            dispatcher.dispatch("reserve", {
                "request_schema": "tensormeld/remote-agent-request-v1",
                "request_id": "r2",
                "args": {
                    "lease_id": "other",
                    "runtime_manifest_sha256": "f" * 64,
                    "observation_id": "obs-1",
                },
            })

    def test_dispatch_rejects_peer_supplied_extra_fields(self):
        cfg, _, enrollment, secret = setup()
        agent = HostAgent(config=cfg, enrollment=enrollment, shared_secret=secret)
        dispatcher = RemoteAgentDispatcher(agent, LocalObjectRegistry())
        with self.assertRaises(ValidationError):
            dispatcher.dispatch("release", {
                "request_schema": "tensormeld/remote-agent-request-v1",
                "request_id": "r1",
                "args": {"lease_id": "x", "command": "rm -rf /"},
            })

    def test_duplicate_request_id_is_rejected(self):
        cfg, _, enrollment, secret = setup()
        agent = HostAgent(config=cfg, enrollment=enrollment, shared_secret=secret)
        dispatcher = RemoteAgentDispatcher(agent, LocalObjectRegistry())
        body = {
            "request_schema": "tensormeld/remote-agent-request-v1",
            "request_id": "same",
            "args": {},
        }
        dispatcher.dispatch("health", body)
        with self.assertRaises(ValidationError):
            dispatcher.dispatch("health", body)

    def test_lifecycle_methods_are_bounded(self):
        cfg, _, enrollment, secret = setup()
        agent = HostAgent(config=cfg, enrollment=enrollment, shared_secret=secret)
        dispatcher = RemoteAgentDispatcher(agent, LocalObjectRegistry())
        health = dispatcher.dispatch("health", {
            "request_schema": "tensormeld/remote-agent-request-v1",
            "request_id": "h",
            "args": {},
        })
        self.assertEqual(health["result"]["lifecycle"], "enabled")
        drain = dispatcher.dispatch("drain", {
            "request_schema": "tensormeld/remote-agent-request-v1",
            "request_id": "d",
            "args": {},
        })
        self.assertEqual(drain["result"]["status"], "DRAINING")


if __name__ == "__main__":
    unittest.main()
