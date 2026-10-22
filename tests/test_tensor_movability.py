from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.cli import main
from tensormeld.config_v2 import Config
from tensormeld.llamacpp_package import (
    build_llamacpp_package_identity,
    load_llamacpp_package_identity,
)
from tensormeld.model_manifest import ModelManifest
from tensormeld.runtime_identity import RuntimeIdentity
from tensormeld.schema import ValidationError
from tensormeld.tensor_movability import (
    load_gguf_tensor_index,
    parse_tensor_movability_profile,
    tensor_movability_summary,
)
from test_config_v2 import data
from test_llamacpp_package import probe
from test_target_host_qualification import runtime_identity


def index_fixture():
    index = {
        "index_schema": "tensormeld/gguf-index-v1",
        "reader": "gguf==0.19.0",
        "architecture": "fixture-arch",
        "complete_shard_set": True,
        "tensor_count": 2,
        "files": [{
            "file_name": "model.gguf",
            "file_size_bytes": 1000,
            "architecture": "fixture-arch",
            "split_count": 1,
            "split_index": 0,
            "declared_total_tensors": 2,
            "tensors": [
                {
                    "name": "tensor.a",
                    "gguf_shape": [10, 10],
                    "encoding": "F32",
                    "n_elements": 100,
                    "n_bytes": 100,
                    "data_offset": 64,
                },
                {
                    "name": "tensor.b",
                    "gguf_shape": [15, 10],
                    "encoding": "F32",
                    "n_elements": 150,
                    "n_bytes": 150,
                    "data_offset": 164,
                },
            ],
        }],
        "tensor_payload_bytes": 250,
        "checkpoint_sha256": None,
        "qualified": False,
        "executable": False,
        "warnings": [
            "Directory inspection only: payload bytes are NOT runtime RAM/VRAM requirements.",
            "No tensor values were inspected, weights executed, or full checkpoint hashes computed.",
            "Matching split metadata is not proof of matching checkpoint provenance.",
            "Index hash covers metadata only, not weights; state/workspace/aliases require an adapter.",
            "Explicit trusted local files only; upstream parsing is not sandboxed.",
        ],
    }
    index["index_sha256"] = hashlib.sha256(
        json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return index


def setup_env():
    index = index_fixture()
    model = ModelManifest.parse({
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/model",
        "revision": "fixture-rev",
        "format": "gguf",
        "architecture": "fixture-arch",
        "tokenizer_ref": "fixture-tokenizer",
        "chat_template_ref": None,
        "files": [{
            "name": "model.gguf",
            "size_bytes": 1000,
            "sha256": "a" * 64,
        }],
        "tensor_index_sha256": index["index_sha256"],
        "tensor_count": 2,
        "tensor_payload_bytes": 250,
    })
    raw = data()
    raw["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
    config = Config.parse(raw)
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
    return config, model, index, adapter, package, identities


def raw_profile(config, model, index, adapter, package, identities):
    ids = [d.id for d in config.devices]
    by_device = {i.tensormeld_device_id: i for i in identities}
    runtime = [by_device[d].identity_sha256 for d in sorted(by_device)]
    return {
        "tensor_movability_schema": "tensormeld/tensor-movability-v1",
        "provenance": "fixture",
        "model_manifest_sha256": model.manifest_sha256,
        "tensor_index_sha256": index["index_sha256"],
        "adapter_capabilities_sha256": adapter.fingerprint,
        "package_sha256": package.fingerprint,
        "runtime_identity_sha256": runtime,
        "tensors": [
            {
                "name": "tensor.a",
                "n_bytes": 100,
                "storage_class": "hard_resident",
                "movement_class": "pinned",
                "allowed_devices": [ids[0]],
                "alias_group": None,
            },
            {
                "name": "tensor.b",
                "n_bytes": 150,
                "storage_class": "reclaimable_file_backed",
                "movement_class": "placement_movable",
                "allowed_devices": ids,
                "alias_group": None,
            },
        ],
        "qualified": False,
        "executable": False,
    }


class TensorMovabilityTests(unittest.TestCase):
    def setUp(self):
        (
            self.config,
            self.model,
            self.index,
            self.adapter,
            self.package,
            self.identities,
        ) = setup_env()

    def parse(self, raw=None, *, index=None, package=None, identities=None):
        return parse_tensor_movability_profile(
            raw or raw_profile(
                self.config,
                self.model,
                self.index,
                self.adapter,
                self.package,
                self.identities,
            ),
            config=self.config,
            model=self.model,
            tensor_index=index or self.index,
            adapter=self.adapter,
            package=package or self.package,
            runtime_identities=identities or self.identities,
        )

    def test_complete_explicit_classification_is_accepted(self):
        profile = self.parse()
        self.assertEqual(len(profile.tensors), 2)
        summary = tensor_movability_summary(profile)
        self.assertEqual(summary["movement_counts"]["pinned"], 1)
        self.assertEqual(summary["movement_counts"]["placement_movable"], 1)
        self.assertEqual(
            summary["bytes_by_storage_class"]["reclaimable_file_backed"], 150
        )
        self.assertFalse(summary["qualified"])
        self.assertFalse(summary["executable"])

    def test_missing_tensor_is_rejected(self):
        raw = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        raw["tensors"].pop()
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_tensor_size_must_match_exact_index(self):
        raw = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        raw["tensors"][0]["n_bytes"] += 1
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_pinned_requires_exactly_one_device(self):
        raw = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        raw["tensors"][0]["allowed_devices"] = [d.id for d in self.config.devices]
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_owner_local_cannot_span_nodes(self):
        raw = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        raw["tensors"][1]["movement_class"] = "owner_local"
        with self.assertRaises(ValidationError):
            self.parse(raw)

    def test_owner_local_same_node_is_allowed(self):
        raw_cfg = data()
        raw_cfg["devices"][1]["node"] = raw_cfg["devices"][0]["node"]
        raw_cfg["devices"][1]["pool_ref"] = raw_cfg["devices"][0]["pool_ref"]
        raw_cfg["resource_policies"] = [
            p for p in raw_cfg["resource_policies"]
            if p["pool_ref"] != "helper-a-ram"
        ]
        raw_cfg["resource_pools"] = [
            p for p in raw_cfg["resource_pools"]
            if p["id"] != "helper-a-ram"
        ]
        index = index_fixture()
        model = ModelManifest.parse({
            "model_manifest_schema": "tensormeld/model-manifest-v1",
            "model_id": "fixture/model",
            "revision": "fixture-rev",
            "format": "gguf",
            "architecture": "fixture-arch",
            "tokenizer_ref": "fixture-tokenizer",
            "chat_template_ref": None,
            "files": [{"name": "model.gguf", "size_bytes": 1000, "sha256": "a"*64}],
            "tensor_index_sha256": index["index_sha256"],
            "tensor_count": 2,
            "tensor_payload_bytes": 250,
        })
        raw_cfg["profiles"]["interactive"]["model_manifest_ref"] = model.manifest_sha256
        config = Config.parse(raw_cfg)
        adapter = AdapterCapabilities.parse({
            "adapter_schema":"tensormeld/adapter-capabilities-v1",
            "adapter_id":"fixture-adapter","engine":"llama.cpp",
            "engine_revision":"fixture-engine-rev",
            "placement":{"strategies":["whole_blocks"],"exact_owner_binding":True,
            "explicit_unit_ranges":True,"mixed_backends":True,"remote_compute":True,
            "coordinator_outside_compute":True,"max_compute_devices":128,
            "max_compute_nodes":64,"max_segments":256},
            "route_modes":["direct","via_coordinator"],
            "coordinator_nodes":[n.id for n in config.nodes],
            "devices":[{"id":d.id,"node":d.node,"backend":d.backend} for d in config.devices],
        })
        package = self.package
        identities = tuple(
            runtime_identity(d.node,d.id,package.llama_server_sha256)
            for d in config.devices
        )
        raw = raw_profile(config,model,index,adapter,package,identities)
        raw["tensors"][1]["movement_class"] = "owner_local"
        profile = parse_tensor_movability_profile(
            raw, config=config, model=model, tensor_index=index, adapter=adapter,
            package=package, runtime_identities=identities
        )
        self.assertEqual(profile.tensors[1].movement_class, "owner_local")

    def test_runtime_identity_change_invalidates_profile(self):
        raw = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        changed = list(self.identities)
        rec = changed[0].as_record()
        rec.pop("identity_sha256")
        rec["driver_version"] = "changed"
        changed[0] = RuntimeIdentity.parse(rec)
        with self.assertRaises(ValidationError):
            self.parse(raw, identities=tuple(changed))

    def test_package_change_invalidates_profile(self):
        raw = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        changed = build_llamacpp_package_identity(
            cli_probe=probe("4" * 64, "llama-cli"),
            server_probe=probe("5" * 64, "llama-server"),
            backend_libraries=[{"name": "ggml-fixture", "sha256": "6" * 64}],
        )
        with self.assertRaises(ValidationError):
            self.parse(raw, package=changed)

    def test_index_content_tampering_with_old_hash_is_rejected(self):
        tampered = copy.deepcopy(self.index)
        tampered["files"][0]["tensors"][0]["n_bytes"] += 1
        with self.assertRaises(ValidationError):
            self.parse(index=tampered)

    def test_alias_group_requires_consistent_policy(self):
        raw = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        raw["tensors"][0]["alias_group"] = "tied"
        raw["tensors"][1]["alias_group"] = "tied"
        with self.assertRaises(ValidationError):
            self.parse(raw)

        raw["tensors"][1]["storage_class"] = "hard_resident"
        raw["tensors"][1]["movement_class"] = "pinned"
        raw["tensors"][1]["allowed_devices"] = raw["tensors"][0]["allowed_devices"]
        profile = self.parse(raw)
        self.assertEqual(
            {t.alias_group for t in profile.tensors}, {"tied"}
        )

    def test_profile_never_self_promotes(self):
        for field in ("qualified", "executable"):
            raw = raw_profile(
                self.config, self.model, self.index, self.adapter,
                self.package, self.identities
            )
            raw[field] = True
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_order_does_not_change_fingerprint(self):
        a = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        b = copy.deepcopy(a)
        b["tensors"].reverse()
        self.assertEqual(self.parse(a).fingerprint, self.parse(b).fingerprint)

    def test_persisted_package_loader_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "package.json"
            path.write_text(json.dumps(self.package.as_record()), encoding="utf-8")
            loaded = load_llamacpp_package_identity(path)
            self.assertEqual(loaded.fingerprint, self.package.fingerprint)
            raw = self.package.as_record()
            raw["artifacts"]["llama_server_sha256"] = "f" * 64
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_llamacpp_package_identity(path)

    def test_tensor_index_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "index.json"
            path.write_text('{"index_schema":"a","index_schema":"b"}', encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_gguf_tensor_index(path)

    def test_cli_roundtrip_and_output_safety(self):
        raw_cfg = data()
        raw_cfg["profiles"]["interactive"]["model_manifest_ref"] = self.model.manifest_sha256
        raw_model = {
            "model_manifest_schema": "tensormeld/model-manifest-v1",
            "model_id": self.model.model_id,
            "revision": self.model.revision,
            "format": self.model.format,
            "architecture": self.model.architecture,
            "tokenizer_ref": self.model.tokenizer_ref,
            "chat_template_ref": self.model.chat_template_ref,
            "files": [
                {"name": item.name, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in self.model.files
            ],
            "tensor_index_sha256": self.model.tensor_index_sha256,
            "tensor_count": self.model.tensor_count,
            "tensor_payload_bytes": self.model.tensor_payload_bytes,
        }
        raw_adapter = {
            "adapter_schema": "tensormeld/adapter-capabilities-v1",
            "adapter_id": self.adapter.adapter_id,
            "engine": self.adapter.engine,
            "engine_revision": self.adapter.engine_revision,
            "placement": {
                "strategies": sorted(self.adapter.strategies),
                "exact_owner_binding": self.adapter.exact_owner_binding,
                "explicit_unit_ranges": self.adapter.explicit_unit_ranges,
                "mixed_backends": self.adapter.mixed_backends,
                "remote_compute": self.adapter.remote_compute,
                "coordinator_outside_compute": self.adapter.coordinator_outside_compute,
                "max_compute_devices": self.adapter.max_compute_devices,
                "max_compute_nodes": self.adapter.max_compute_nodes,
                "max_segments": self.adapter.max_segments,
            },
            "route_modes": sorted(self.adapter.route_modes),
            "coordinator_nodes": [
                n.id for n in self.config.nodes
                if n.id in self.adapter.coordinator_nodes
            ],
            "devices": [
                {"id": item.id, "node": item.node, "backend": item.backend}
                for item in self.adapter.devices
            ],
        }
        raw_move = raw_profile(
            self.config, self.model, self.index, self.adapter,
            self.package, self.identities
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = {
                "config": root / "config.json",
                "model": root / "model.json",
                "index": root / "index.json",
                "adapter": root / "adapter.json",
                "package": root / "package.json",
                "move": root / "move.json",
                "out": root / "out.json",
            }
            paths["config"].write_text(json.dumps(raw_cfg), encoding="utf-8")
            paths["model"].write_text(json.dumps(raw_model), encoding="utf-8")
            paths["index"].write_text(json.dumps(self.index), encoding="utf-8")
            paths["adapter"].write_text(json.dumps(raw_adapter), encoding="utf-8")
            paths["package"].write_text(json.dumps(self.package.as_record()), encoding="utf-8")
            paths["move"].write_text(json.dumps(raw_move), encoding="utf-8")
            identity_paths = []
            for i, identity in enumerate(self.identities):
                path = root / f"identity-{i}.json"
                path.write_text(json.dumps(identity.as_record()), encoding="utf-8")
                identity_paths.append(path)
            argv = [
                "validate-tensor-movability",
                str(paths["config"]),
                str(paths["model"]),
                str(paths["index"]),
                str(paths["adapter"]),
                str(paths["package"]),
                str(paths["move"]),
            ]
            for path in identity_paths:
                argv += ["--runtime-identity", str(path)]
            argv += ["--out", str(paths["out"])]
            self.assertEqual(main(argv), 0)
            result = json.loads(paths["out"].read_text(encoding="utf-8"))
            self.assertEqual(result["tensor_count"], 2)
            self.assertFalse(result["qualified"])
            before = paths["move"].read_bytes()
            overwrite = argv[:-2] + ["--out", str(paths["move"])]
            self.assertEqual(main(overwrite), 1)
            self.assertEqual(paths["move"].read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
