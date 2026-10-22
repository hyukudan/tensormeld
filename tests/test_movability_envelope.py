from __future__ import annotations

import copy
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.config_v2 import Config
from tensormeld.llamacpp_package import build_llamacpp_package_identity
from tensormeld.model_manifest import ModelManifest
from tensormeld.movability_envelope import (
    derive_movability_envelope,
    movability_envelope_summary,
)
from tensormeld.predictive_memory import parse_predictive_memory_profile
from tensormeld.runtime_model_manifest import parse_runtime_model_manifest
from tensormeld.tensor_movability import parse_tensor_movability_profile
from test_config_v2 import data
from test_llamacpp_package import probe
from test_predictive_memory import predictive_raw
from test_runtime_model_manifest import manifest
from test_target_host_qualification import runtime_identity
from test_tensor_movability import index_fixture, raw_profile, setup_env


def full_env():
    config, model, index, adapter, package, identities = setup_env()
    runtime = parse_runtime_model_manifest(
        manifest(config, model, adapter),
        config=config,
        model=model,
        adapter=adapter,
    )
    predictive = parse_predictive_memory_profile(
        predictive_raw(runtime),
        config=config,
        runtime_manifest=runtime,
    )
    movability = parse_tensor_movability_profile(
        raw_profile(config, model, index, adapter, package, identities),
        config=config,
        model=model,
        tensor_index=index,
        adapter=adapter,
        package=package,
        runtime_identities=identities,
    )
    return config, runtime, predictive, movability


def shared_pool_env():
    index = index_fixture()
    model = ModelManifest.parse({
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/model",
        "revision": "fixture-rev",
        "format": "gguf",
        "architecture": "fixture-arch",
        "tokenizer_ref": "fixture-tokenizer",
        "chat_template_ref": None,
        "files": [{"name": "model.gguf", "size_bytes": 1000, "sha256": "a" * 64}],
        "tensor_index_sha256": index["index_sha256"],
        "tensor_count": 2,
        "tensor_payload_bytes": 250,
    })
    raw_cfg = data()
    raw_cfg["devices"][1]["node"] = raw_cfg["devices"][0]["node"]
    raw_cfg["devices"][1]["pool_ref"] = raw_cfg["devices"][0]["pool_ref"]
    removed_pool = "helper-a-ram"
    raw_cfg["resource_policies"] = [
        p for p in raw_cfg["resource_policies"] if p["pool_ref"] != removed_pool
    ]
    raw_cfg["resource_pools"] = [
        p for p in raw_cfg["resource_pools"] if p["id"] != removed_pool
    ]
    raw_cfg["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
    config = Config.parse(raw_cfg)
    adapter = AdapterCapabilities.parse({
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "fixture-adapter",
        "engine": "llama.cpp",
        "engine_revision": "fixture-engine-rev",
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
        "route_modes": ["direct", "via_coordinator"],
        "coordinator_nodes": [n.id for n in config.nodes],
        "devices": [
            {"id": d.id, "node": d.node, "backend": d.backend}
            for d in config.devices
        ],
    })
    package = build_llamacpp_package_identity(
        cli_probe=probe("1" * 64, "llama-cli"),
        server_probe=probe("2" * 64, "llama-server"),
        backend_libraries=[{"name": "ggml-fixture", "sha256": "3" * 64}],
    )
    identities = tuple(
        runtime_identity(d.node, d.id, package.llama_server_sha256)
        for d in config.devices
    )
    raw_move = raw_profile(config, model, index, adapter, package, identities)
    raw_move["tensors"][1]["movement_class"] = "owner_local"
    movability = parse_tensor_movability_profile(
        raw_move,
        config=config,
        model=model,
        tensor_index=index,
        adapter=adapter,
        package=package,
        runtime_identities=identities,
    )
    runtime = parse_runtime_model_manifest(
        manifest(config, model, adapter),
        config=config,
        model=model,
        adapter=adapter,
    )
    predictive = parse_predictive_memory_profile(
        predictive_raw(runtime),
        config=config,
        runtime_manifest=runtime,
    )
    return config, runtime, predictive, movability


class MovabilityEnvelopeTests(unittest.TestCase):
    def test_device_envelopes_overlap_for_movable_tensor(self):
        config, runtime, predictive, movability = full_env()
        envelope = derive_movability_envelope(
            config=config,
            runtime_manifest=runtime,
            predictive_memory=predictive,
            movability=movability,
        )
        by_device = {row.device: row for row in envelope.devices}
        first, second = [d.id for d in config.devices]
        self.assertEqual(by_device[first].hard_resident_eligible_bytes, 100)
        self.assertEqual(by_device[first].reclaimable_file_backed_eligible_bytes, 150)
        self.assertEqual(by_device[first].pinned_bytes, 100)
        self.assertEqual(by_device[second].hard_resident_eligible_bytes, 0)
        self.assertEqual(by_device[second].reclaimable_file_backed_eligible_bytes, 150)

    def test_shared_pool_union_does_not_double_count_tensor(self):
        config, runtime, predictive, movability = shared_pool_env()
        envelope = derive_movability_envelope(
            config=config,
            runtime_manifest=runtime,
            predictive_memory=predictive,
            movability=movability,
        )
        shared_pool = config.devices[0].pool
        row = next(item for item in envelope.pools if item.pool == shared_pool)
        self.assertEqual(row.hard_resident_union_bytes, 100)
        self.assertEqual(row.reclaimable_file_backed_union_bytes, 150)
        self.assertEqual(row.pinned_bytes, 100)
        self.assertEqual(row.tensor_count, 2)
        device_reclaimable = sum(
            item.reclaimable_file_backed_eligible_bytes
            for item in envelope.devices
            if item.pool == shared_pool
        )
        self.assertEqual(device_reclaimable, 300)
        self.assertEqual(row.reclaimable_file_backed_union_bytes, 150)

    def test_envelope_is_not_memory_measurement_or_ownership(self):
        config, runtime, predictive, movability = full_env()
        summary = movability_envelope_summary(derive_movability_envelope(
            config=config,
            runtime_manifest=runtime,
            predictive_memory=predictive,
            movability=movability,
        ))
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])
        warning = " ".join(summary["warnings"])
        self.assertIn("not current ownership", warning)
        self.assertIn("not a runtime-memory measurement", warning)

    def test_runtime_model_identity_mismatch_is_rejected(self):
        config, runtime, predictive, movability = full_env()
        bad = copy.copy(movability)
        object.__setattr__(bad, "model_manifest_sha256", "0" * 64)
        with self.assertRaisesRegex(Exception, "model identity mismatch"):
            derive_movability_envelope(
                config=config,
                runtime_manifest=runtime,
                predictive_memory=predictive,
                movability=bad,
            )

    def test_predictive_runtime_identity_mismatch_is_rejected(self):
        config, runtime, predictive, movability = full_env()
        bad = copy.copy(predictive)
        object.__setattr__(bad, "runtime_manifest_sha256", "0" * 64)
        with self.assertRaisesRegex(Exception, "predictive memory/runtime"):
            derive_movability_envelope(
                config=config,
                runtime_manifest=runtime,
                predictive_memory=bad,
                movability=movability,
            )


if __name__ == "__main__":
    unittest.main()
