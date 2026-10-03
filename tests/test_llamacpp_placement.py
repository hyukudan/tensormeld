from __future__ import annotations

from dataclasses import replace
import unittest

from tensormeld.adapter_contract import AdapterCapabilities
from tensormeld.llamacpp_placement import LlamaCppPlacementBinding, translate_whole_blocks_to_llamacpp
from tensormeld.schema import ValidationError
from tensormeld.whole_block_execution import AcceptedExecutionBundle

PIN = "552f18f912a32ea86edf82e2b76431cb7131538d"


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
        "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "ROCm0"}, {"device_id": "g1", "engine_device_name": "ROCm1", "buffer_type": "ROCm1"}],
    }, adapter=a)


def bundle(a=None):
    a = a or adapter()
    return AcceptedExecutionBundle(
        config_sha256="1" * 64,
        profile="interactive",
        planning_input_sha256="2" * 64,
        plan_sha256="3" * 64,
        representability_sha256="4" * 64,
        adapter_id=a.adapter_id,
        engine_revision=a.engine_revision,
        adapter_capabilities_sha256=a.fingerprint,
        model_manifest_sha256="5" * 64,
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
        result = translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=binding(a))
        self.assertEqual(result.block_owners, ((0, "g0", "ROCm0"), (1, "g0", "ROCm0"), (2, "g1", "ROCm1"), (3, "g1", "ROCm1")))
        self.assertEqual(result.argv_fragment[:4], ("--fit", "off", "--device", "ROCm0,RPC0[10.0.0.2:50052]"))
        self.assertIn(r"^blk\.0\..*=ROCm0", result.override_tensor_value)
        self.assertIn(r"^blk\.3\..*=ROCm1", result.override_tensor_value)
        self.assertFalse(result.as_record()["real_model_inference"])

    def test_non_block_unit_is_rejected(self):
        a = adapter()
        bad = replace(bundle(a), unit_ids=("blk.0", "embedding", "blk.2", "blk.3"))
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(bad, adapter=a, binding=binding(a))

    def test_gap_or_reordered_blocks_is_rejected(self):
        a = adapter()
        for ids in (("blk.0", "blk.2", "blk.3", "blk.4"), ("blk.1", "blk.0", "blk.2", "blk.3")):
            with self.subTest(ids=ids):
                with self.assertRaises(ValidationError):
                    translate_whole_blocks_to_llamacpp(replace(bundle(a), unit_ids=ids), adapter=a, binding=binding(a))

    def test_overlap_or_missing_segment_coverage_is_rejected(self):
        a = adapter()
        for segments in (("overlap", (("g0", 0, 3), ("g1", 2, 4))), ("missing", (("g0", 0, 2), ("g1", 3, 4)))):
            with self.subTest(kind=segments[0]), self.assertRaises(ValidationError):
                translate_whole_blocks_to_llamacpp(replace(bundle(a), segments=segments[1]), adapter=a, binding=binding(a))

    def test_binding_must_cover_exact_compute_devices(self):
        a = adapter()
        one = LlamaCppPlacementBinding.parse({"placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1", "adapter_id": a.adapter_id, "adapter_capabilities_sha256": a.fingerprint, "source_revision": PIN, "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "ROCm0"}]}, adapter=a)
        with self.assertRaises(ValidationError):
            translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=one)

    def test_user_like_regex_or_delimiter_is_not_allowed_as_buffer_type(self):
        a = adapter()
        for buft in ("ROCm0,CPU", "x=y", ".*"):
            raw = {"placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1", "adapter_id": a.adapter_id, "adapter_capabilities_sha256": a.fingerprint, "source_revision": PIN, "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": buft}, {"device_id": "g1", "engine_device_name": "ROCm1", "buffer_type": "ROCm1"}]}
            with self.subTest(buft=buft), self.assertRaises(ValidationError):
                LlamaCppPlacementBinding.parse(raw, adapter=a)

    def test_wrong_revision_or_adapter_identity_is_rejected(self):
        a = adapter()
        raw = {"placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1", "adapter_id": a.adapter_id, "adapter_capabilities_sha256": a.fingerprint, "source_revision": "0" * 40, "devices": [{"device_id": "g0", "engine_device_name": "ROCm0", "buffer_type": "ROCm0"}, {"device_id": "g1", "engine_device_name": "ROCm1", "buffer_type": "ROCm1"}]}
        with self.assertRaises(ValidationError):
            LlamaCppPlacementBinding.parse(raw, adapter=a)

    def test_remote_rpc_binding_is_rejected(self):
        a = adapter()
        raw = {
            "placement_binding_schema": "tensormeld/llamacpp-placement-binding-v1",
            "adapter_id": a.adapter_id,
            "adapter_capabilities_sha256": a.fingerprint,
            "source_revision": PIN,
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
            translate_whole_blocks_to_llamacpp(bad, adapter=a, binding=binding(a))

    def test_translation_is_deterministic(self):
        a = adapter()
        x = translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=binding(a))
        y = translate_whole_blocks_to_llamacpp(bundle(a), adapter=a, binding=binding(a))
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
