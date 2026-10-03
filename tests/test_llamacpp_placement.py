from __future__ import annotations

from dataclasses import replace
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.llamacpp_placement import LlamaCppPlacementBinding, translate_whole_blocks_to_llamacpp
from tensormeld.model_manifest import ModelManifest
from tensormeld.schema import ValidationError
from tensormeld.whole_block_execution import AcceptedExecutionBundle

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"
INDEX_SHA = "1" * 64


def gguf_index(blocks=(0, 1, 2, 3)):
    tensors = []
    for block in blocks:
        tensors.extend([
            {"name": f"blk.{block}.attn_q.weight"},
            {"name": f"blk.{block}.ffn_up.weight"},
        ])
    return {
        "index_schema": "tensormeld/gguf-index-v1",
        "architecture": "llama",
        "complete_shard_set": True,
        "tensor_count": len(tensors),
        "index_sha256": INDEX_SHA,
        "files": [{"tensors": tensors}],
    }


def model(index=None):
    index = index or gguf_index()
    return ModelManifest.parse({
        "model_manifest_schema": "tensormeld/model-manifest-v1",
        "model_id": "fixture/llama",
        "revision": "fixture-rev",
        "format": "gguf",
        "architecture": index["architecture"],
        "tokenizer_ref": "fixture-tokenizer",
        "chat_template_ref": None,
        "files": [{"name": "fixture.gguf", "size_bytes": 1, "sha256": "2" * 64}],
        "tensor_index_sha256": index["index_sha256"],
        "tensor_count": index["tensor_count"],
        "tensor_payload_bytes": 1,
    })


def adapter():
    return AdapterCapabilities.parse({
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "llamacpp-native",
        "engine": "llama.cpp",
        "engine_revision": PIN,
        "placement": {"strategies": ["whole_blocks"], "exact_owner_binding": True, "explicit_unit_ranges": True, "mixed_backends": True, "remote_compute": True, "coordinator_outside_compute": True, "max_compute_devices": 8, "max_compute_nodes": 8, "max_segments": 16},
        "route_modes": ["direct", "via_coordinator"],
        "coordinator_nodes": ["n0"],
        "devices": [{"id": "g0", "node": "n0", "backend": "hip"}, {"id": "g1", "node": "n0", "backend": "hip"}],
    })


def binding(a=None):
    a = a or adapter()
    return LlamaCppPlacementBinding.parse({
        "placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
        "adapter_id": a.adapter_id,
        "adapter_capabilities_sha256": a.fingerprint,
        "source_revision": PIN,
        "native_binding_sha256": "0" * 64,
        "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "ROCm0"}, {"device_id": "g1", "engine_device_name": "ROCm1", "buffer_type": "ROCm1"}],
    }, adapter=a)


def bound_result(a=None):
    a = a or adapter()
    return {
        "result_schema": "tensormeld/llamacpp-device-binding-result-v1",
        "config_sha256": "1" * 64,
        "binding_sha256": "0" * 64,
        "resolved_mappings": [
            {
                "tensormeld_device_id": "g0",
                "engine_device_name": "ROCm0",
                "backend_from_config": "hip",
            },
            {
                "tensormeld_device_id": "g1",
                "engine_device_name": "ROCm1",
                "backend_from_config": "hip",
            },
        ],
    }


def bundle(a=None, m=None):
    a = a or adapter()
    m = m or model()
    return AcceptedExecutionBundle(
        config_sha256="1" * 64,
        profile="interactive",
        planning_input_sha256="2" * 64,
        plan_sha256="3" * 64,
        representability_sha256="4" * 64,
        adapter_id=a.adapter_id,
        engine_revision=a.engine_revision,
        adapter_capabilities_sha256=a.fingerprint,
        model_manifest_sha256=m.manifest_sha256,
        qualification_evidence_sha256="6" * 64,
        runtime_manifest_sha256="7" * 64,
        worker_artifact_sha256="8" * 64,
        backend_readiness=(("g0", "9" * 64, "a" * 64), ("g1", "b" * 64, "c" * 64)),
        launch_leases=(("n0", "l0", "d" * 64),),
        segments=(("g0", 0, 2), ("g1", 2, 4)),
        unit_ids=("blk.0", "blk.1", "blk.2", "blk.3"),
        compute_devices=("g0", "g1"),
        compute_nodes=("n0",),
        bundle_sha256="f" * 64,
    )


class LlamaCppPlacementTests(unittest.TestCase):
    def test_exact_blocks_translate_to_anchored_tensor_overrides(self):
        a = adapter()
        result = translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=binding(a), bound_result=bound_result(a), model=model(), gguf_index=gguf_index())
        self.assertEqual(result.block_owners, ((0, "g0", "ROCm0"), (1, "g0", "ROCm0"), (2, "g1", "ROCm1"), (3, "g1", "ROCm1")))
        self.assertEqual(result.argv_fragment[:4], ("--fit", "off", "--device", "ROCm0,RPC0[10.0.0.2:50052]"))
        self.assertIn(r"^blk\.0\..*=ROCm0", result.override_tensor_value)
        self.assertIn(r"^blk\.3\..*=ROCm1", result.override_tensor_value)
        self.assertFalse(result.as_record()["real_model_inference"])

    def test_non_block_unit_is_rejected(self):
        a = adapter()
        bad = replace(bundle(a), unit_ids=("blk.0", "embedding", "blk.2", "blk.3"))
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(bad, adapter=a, binding=binding(a), bound_result=bound_result(a), model=model(), gguf_index=gguf_index())

    def test_gap_or_reordered_blocks_is_rejected(self):
        a = adapter()
        for ids in (("blk.0", "blk.2", "blk.3", "blk.4"), ("blk.1", "blk.0", "blk.2", "blk.3")):
            with self.subTest(ids=ids):
                with self.assertRaises(ValidationError):
                    translate_whole_blocks_to_llamacpp(replace(bundle(a), unit_ids=ids), adapter=a, binding=binding(a), bound_result=bound_result(a), model=model(), gguf_index=gguf_index())

    def test_overlap_or_missing_segment_coverage_is_rejected(self):
        a = adapter()
        for segments in (("overlap", (("g0", 0, 3), ("g1", 2, 4))), ("missing", (("g0", 0, 2), ("g1", 3, 4)))):
            with self.subTest(kind=segments[0]), self.assertRaises(ValidationError):
                translate_whole_blocks_to_llamacpp(replace(bundle(a), segments=segments[1]), adapter=a, binding=binding(a), bound_result=bound_result(a), model=model(), gguf_index=gguf_index())

    def test_binding_must_cover_exact_compute_devices(self):
        a = adapter()
        one = LlamaCppPlacementBinding.parse({"placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1", "adapter_id": a.adapter_id, "adapter_capabilities_sha256": a.fingerprint, "source_revision": PIN, "native_binding_sha256": "0" * 64, "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "ROCm0"}]}, adapter=a)
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=one, bound_result=bound_result(a), model=model(), gguf_index=gguf_index())

    def test_user_like_regex_or_delimiter_is_not_allowed_as_buffer_type(self):
        a = adapter()
        for buft in ("ROCm0,CPU", "x=y", ".*"):
            raw = {"placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1", "adapter_id": a.adapter_id, "adapter_capabilities_sha256": a.fingerprint, "source_revision": PIN, "native_binding_sha256": "0" * 64, "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": buft}, {"device_id": "g1", "engine_device_name": "ROCm1", "buffer_type": "ROCm1"}]}
            with self.subTest(buft=buft), self.assertRaises(ValidationError):
                LlamaCppPlacementBinding.parse(raw, adapter=a)

    def test_non_primary_device_buffer_is_rejected(self):
        a = adapter()
        raw = {
            "placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
            "adapter_id": a.adapter_id,
            "adapter_capabilities_sha256": a.fingerprint,
            "source_revision": PIN,
            "native_binding_sha256": "0" * 64,
            "devices": [
                {"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "CPU"},
                {"device_id": "g1", "engine_device_name": "ROCm1", "buffer_type": "ROCm1"},
            ],
        }
        with self.assertRaises(ValidationError):
            LlamaCppPlacementBinding.parse(raw, adapter=a)

    def test_wrong_revision_or_adapter_identity_is_rejected(self):
        a = adapter()
        raw = {"placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1", "adapter_id": a.adapter_id, "adapter_capabilities_sha256": a.fingerprint, "source_revision": "0" * 40, "native_binding_sha256": "0" * 64, "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "ROCm0"}, {"device_id": "g1", "engine_device_name": "ROCm1", "buffer_type": "ROCm1"}]}
        with self.assertRaises(ValidationError):
            LlamaCppPlacementBinding.parse(raw, adapter=a)

    def test_remote_rpc_binding_is_rejected(self):
        a = adapter()
        raw = {
            "placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
            "adapter_id": a.adapter_id,
            "adapter_capabilities_sha256": a.fingerprint,
            "source_revision": PIN,
            "native_binding_sha256": "0" * 64,
            "devices": [
                {"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "ROCm0"},
                {"device_id": "g1", "engine_device_name": "RPC0", "buffer_type": "RPC0[10.0.0.2:50052]"},
            ],
        }
        with self.assertRaises(ValidationError):
            LlamaCppPlacementBinding.parse(raw, adapter=a)

    def test_multi_node_bundle_is_rejected_until_secure_remote_worker_exists(self):
        a = adapter()
        bad = replace(bundle(a), compute_nodes=("n0", "n1"))
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(bad, adapter=a, binding=binding(a), bound_result=bound_result(a), model=model(), gguf_index=gguf_index())

    def test_native_mapping_identity_must_match(self):
        a = adapter()
        current = bound_result(a)
        current["resolved_mappings"][1]["engine_device_name"] = "ROCm9"
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(
                bundle(a), adapter=a, binding=binding(a), bound_result=current, model=model(), gguf_index=gguf_index()
            )

    def test_native_binding_fingerprint_must_match(self):
        a = adapter()
        current = bound_result(a)
        current["binding_sha256"] = "1" * 64
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(
                bundle(a), adapter=a, binding=binding(a), bound_result=current, model=model(), gguf_index=gguf_index()
            )

    def test_bundle_must_cover_every_real_gguf_block(self):
        a = adapter()
        index = gguf_index(blocks=(0, 1, 2, 3, 4))
        m = model(index)
        bad = replace(
            bundle(a, m),
            unit_ids=("blk.0", "blk.1", "blk.2", "blk.3"),
            segments=(("g0", 0, 2), ("g1", 2, 4)),
        )
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(
                bad,
                adapter=a,
                binding=binding(a),
                bound_result=bound_result(a),
                model=m,
                gguf_index=index,
            )

    def test_gguf_index_identity_must_match_model_manifest(self):
        a = adapter()
        index = gguf_index()
        m = model(index)
        changed = dict(index)
        changed["index_sha256"] = "9" * 64
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(
                bundle(a, m),
                adapter=a,
                binding=binding(a),
                bound_result=bound_result(a),
                model=m,
                gguf_index=changed,
            )

    def test_unsupported_blk_tensor_namespace_is_rejected(self):
        a = adapter()
        index = gguf_index()
        index["files"][0]["tensors"].append({"name": "blk.bad.weight"})
        index["tensor_count"] += 1
        m = model(index)
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(
                bundle(a, m),
                adapter=a,
                binding=binding(a),
                bound_result=bound_result(a),
                model=m,
                gguf_index=index,
            )

    def test_translation_is_deterministic(self):
        a = adapter()
        x = translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=binding(a), bound_result=bound_result(a), model=model(), gguf_index=gguf_index())
        y = translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=binding(a), bound_result=bound_result(a), model=model(), gguf_index=gguf_index())
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
