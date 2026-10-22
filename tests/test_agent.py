from __future__ import annotations

import copy
import unittest

from tensormeld.agent import (
    HostAgent,
    ReplayGuard,
    enrollment_from_runtime,
    sign_capability_envelope,
    verify_capability_envelope,
)
from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.config_v2 import Config
from tensormeld.model_manifest import ModelManifest
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.schema import ValidationError
from test_config_v2 import data
from test_admission import snapshot


def setup():
    model = ModelManifest.parse({
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/model",
        "revision": "r1",
        "format": "gguf",
        "architecture": "fixture",
        "tokenizer_ref": "tok",
        "chat_template_ref": None,
        "files": [{"name": "m.gguf", "size_bytes": 1, "sha256": "a" * 64}],
        "tensor_index_sha256": "b" * 64,
        "tensor_count": 1,
        "tensor_payload_bytes": 1,
    })
    raw = data()
    raw["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
    cfg = Config.parse(raw)
    adapter = AdapterCapabilities.parse({
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "fixture",
        "engine": "fixture",
        "engine_revision": "r1",
        "placement": {
            "strategies": ["whole_blocks"],
            "exact_owner_binding": True,
            "explicit_unit_ranges": True,
            "mixed_backends": True,
            "remote_compute": True,
            "coordinator_outside_compute": True,
            "max_compute_devices": 128,
            "max_compute_nodes": 64,
            "max_segments": 256,
        },
        "route_modes": ["direct"],
        "coordinator_nodes": [n.id for n in cfg.nodes],
        "devices": [{"id": d.id, "node": d.node, "backend": d.backend} for d in cfg.devices],
    })
    p = cfg.profile_map["interactive"]
    manifest = parse_runtime_model_manifest({
        "runtime_manifest_schema": "tensormeld/runtime-model-manifest-v1",
        "provenance": "fixture",
        "config_sha256": cfg.fingerprint,
        "profile": "interactive",
        "model_manifest_sha256": model.manifest_sha256,
        "adapter_id": adapter.adapter_id,
        "adapter_capabilities_sha256": adapter.fingerprint,
        "engine_revision": adapter.engine_revision,
        "worker_artifact_sha256": "c" * 64,
        "workload": {
            "task": p.workload.task,
            "context_tokens": p.workload.context_tokens,
            "max_output_tokens": p.workload.max_output_tokens,
            "concurrency": p.workload.max_active_requests,
        },
        "required_operators": ["ADD"],
        "devices": [{"id": d.id, "node": d.node, "backend": d.backend, "operators": ["ADD"]} for d in cfg.devices],
        "physical_pool_memory": [{
            "pool_ref": pool.id,
            "node": pool.node,
            "resident_bytes": 0,
            "state_bytes": 0,
            "workspace_peak_bytes": 0,
            "preparation_peak_bytes": 1024,
        } for pool in cfg.pools],
        "reservation_created": False,
        "qualified": False,
        "executable": False,
    }, config=cfg, model=model, adapter=adapter)
    enrollment, secret = enrollment_from_runtime(
        enrollment_id="fixture-enrollment",
        node_id=cfg.nodes[0].id,
        key_id="fixture-key",
        shared_secret=b"k" * 32,
    )
    return cfg, manifest, enrollment, secret


class AgentEnrollmentTests(unittest.TestCase):
    def test_authenticated_capability_envelope_and_replay_rejection(self):
        cfg, _, enrollment, secret = setup()
        envelope = sign_capability_envelope(
            enrollment=enrollment,
            shared_secret=secret,
            config=cfg,
            instance_id="instance",
            sequence=1,
            lifecycle="enabled",
            operations=["describe", "health"],
        )
        guard = ReplayGuard()
        payload = verify_capability_envelope(
            envelope,
            enrollment=enrollment,
            shared_secret=secret,
            config=cfg,
            replay_guard=guard,
        )
        self.assertEqual(payload["node_id"], enrollment.node_id)
        with self.assertRaises(ValidationError):
            verify_capability_envelope(
                envelope,
                enrollment=enrollment,
                shared_secret=secret,
                config=cfg,
                replay_guard=guard,
            )

    def test_tampering_or_wrong_secret_fails(self):
        cfg, _, enrollment, secret = setup()
        envelope = sign_capability_envelope(
            enrollment=enrollment, shared_secret=secret, config=cfg,
            instance_id="instance", sequence=1, lifecycle="enabled",
            operations=["describe"],
        )
        tampered = copy.deepcopy(envelope)
        tampered["node_id"] = "other"
        with self.assertRaises(ValidationError):
            verify_capability_envelope(
                tampered, enrollment=enrollment, shared_secret=secret, config=cfg
            )
        with self.assertRaises(ValidationError):
            verify_capability_envelope(
                envelope, enrollment=enrollment, shared_secret=b"x" * 32, config=cfg
            )

    def test_secret_is_not_serialized_in_envelope(self):
        cfg, _, enrollment, secret = setup()
        envelope = sign_capability_envelope(
            enrollment=enrollment, shared_secret=secret, config=cfg,
            instance_id="instance", sequence=1, lifecycle="enabled",
            operations=["describe"],
        )
        self.assertNotIn(secret.hex(), str(envelope))
        self.assertNotIn("shared_secret", envelope)

    def test_agent_reserves_only_its_owned_node_pools(self):
        cfg, manifest, enrollment, secret = setup()
        agent = HostAgent(
            config=cfg, enrollment=enrollment, shared_secret=secret, instance_id="instance"
        )
        result = agent.reserve(
            lease_id="local",
            manifest=manifest,
            snapshot=snapshot(cfg, "o1"),
        )
        owned_pools = {p.id for p in cfg.pools if p.node == enrollment.node_id}
        self.assertEqual(set(result["pool_bytes"]), owned_pools)

    def test_drain_blocks_new_reservations_but_allows_release(self):
        cfg, manifest, enrollment, secret = setup()
        agent = HostAgent(config=cfg, enrollment=enrollment, shared_secret=secret)
        agent.reserve(lease_id="a", manifest=manifest, snapshot=snapshot(cfg, "o1"))
        self.assertEqual(agent.drain()["status"], "DRAINING")
        with self.assertRaises(ValidationError):
            agent.reserve(lease_id="b", manifest=manifest, snapshot=snapshot(cfg, "o2"))
        self.assertEqual(agent.release("a")["status"], "RELEASED")
        self.assertEqual(agent.disable()["status"], "DISABLED")

    def test_no_arbitrary_peer_execution_surface(self):
        cfg, _, enrollment, secret = setup()
        agent = HostAgent(config=cfg, enrollment=enrollment, shared_secret=secret)
        with self.assertRaises(ValidationError):
            agent.execute_peer_request("rm -rf /")

    def test_describe_is_signed_and_monotonic(self):
        cfg, _, enrollment, secret = setup()
        agent = HostAgent(
            config=cfg, enrollment=enrollment, shared_secret=secret, instance_id="instance"
        )
        a = agent.describe()
        b = agent.describe()
        self.assertEqual(a["sequence"] + 1, b["sequence"])
        guard = ReplayGuard()
        verify_capability_envelope(
            a, enrollment=enrollment, shared_secret=secret, config=cfg, replay_guard=guard
        )
        verify_capability_envelope(
            b, enrollment=enrollment, shared_secret=secret, config=cfg, replay_guard=guard
        )


if __name__ == "__main__":
    unittest.main()
