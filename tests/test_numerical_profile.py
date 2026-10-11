from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.numerical_profile import (
    BOUNDARY_CAPABILITY_SCHEMA, NUMERICAL_CALIBRATION_SCHEMA,
    NUMERICAL_CAPABILITY_SCHEMA, NUMERICAL_PROFILE_SCHEMA,
    BoundaryCapability, NumericalCalibration, NumericalCapability,
    NumericalProfile, NumericalScope, StateHistoryBinding,
    load_numerical_profile, validate_boundary_applicability,
    validate_numerical_applicability,
)
from tensormeld.schema import MAX_INPUT_BYTES, ValidationError


def phase_data(phase="decode", activation="float16", decode="local_float_conversion"):
    integer = activation in {"symmetric_a8", "symmetric_a4"}
    return {
        "phase": phase,
        "activation": {
            "encoding": activation, "group_elements": 32 if integer else 1,
            "rounding": "nearest_even" if integer else "none",
            "clipping": {"symmetric_a8": "symmetric_127", "symmetric_a4": "symmetric_7"}.get(activation, "none"),
            "finite_input": "reject_nonfinite", "zero_group": "unit_scale_zero_codes" if integer else "not_applicable",
            "denormals": "preserve", "division": "ieee_divide" if integer else "none",
            "quantization_sha256": "3" * 64 if integer else None,
        },
        "operands": {
            "weight_decode": decode, "weight_dtype": "int8" if integer else "float16",
            "activation_dtype": {"symmetric_a8": "int8", "symmetric_a4": "int4"}.get(activation, activation),
            "accumulator_dtype": "int32" if integer else "float32",
            "saturation": "disabled_checked" if integer else "not_applicable",
            "scale_order": "ordered_groups", "reduction_sha256": "4" * 64,
        },
        "output_dtype": "float32", "row_invariant": True, "input_layout_sha256": "5" * 64,
        "outlier_sha256": None, "rotation_sha256": None, "activation_calibration_sha256": None,
    }


def profile_data(activation="float16", nonlinear=False):
    decode = "codebook_lookup_int8" if nonlinear and activation.startswith("symmetric") else "raw_integer" if activation.startswith("symmetric") else "local_float_conversion"
    return {
        "numerical_profile_schema": NUMERICAL_PROFILE_SCHEMA,
        "model_manifest_sha256": "1" * 64, "target_semantics_sha256": "2" * 64,
        "mode": "declared_target",
        "weights": {
            "encoding": "fixture-iq4" if nonlinear else "fixture-affine4",
            "storage_bits": 4, "code_dtype": "uint4",
            "interpretation": "nonlinear_codebook" if nonlinear else "affine",
            "block_elements": 256 if nonlinear else 32, "group_elements": 32,
            "scale_rule": "scale_times_codebook" if nonlinear else "scale_times_code_minus_zero",
            "codebook_sha256": "6" * 64 if nonlinear else None, "layout_sha256": "7" * 64,
        },
        "phases": [phase_data(p, activation, decode) for p in ("prefill", "decode", "verify")],
        "state": {"kv_dtype": "float16", "recurrent_dtype": "float32", "layout_sha256": "8" * 64, "update_semantics_sha256": "9" * 64},
        "wire": {"dtype": "float16", "quantization_sha256": None, "layout_sha256": "a" * 64, "abi_sha256": "b" * 64},
        "qualified": False, "executable": False,
    }


def scope(profile, phase="decode", rows=1, k=2560):
    return NumericalScope.parse({
        "profile_sha256": profile.fingerprint, "phase": phase, "actual_rows": rows,
        "input_features": k, "output_features": 640, "backend": "fixture-backend",
        "architecture": "fixture-architecture", "runtime_identity_sha256": "c" * 64,
        "implementation_sha256": "d" * 64,
    })


def capability(request):
    return NumericalCapability.parse({"numerical_capability_schema": NUMERICAL_CAPABILITY_SCHEMA,
                                      "scope": asdict(request), "qualified": False, "executable": False})


def calibration(request, provenance="native-adapter"):
    return NumericalCalibration.parse({
        "numerical_calibration_schema": NUMERICAL_CALIBRATION_SCHEMA, "scope": asdict(request), "provenance": provenance,
        "costs_us": {"preparation_us": 3, "layout_lookup_us": 2, "grouping_scatter_us": 0, "compute_us": 7, "scale_output_us": 1, "snapshot_us": 0},
        "transient_peak_bytes": 4096, "qualified": False, "executable": False,
    })


def apply(profile, request=None, cap=None, **kwargs):
    request = request or scope(profile)
    return validate_numerical_applicability(profile, request, cap or capability(request),
        approved_profile_sha256s=kwargs.pop("approved_profile_sha256s", frozenset({profile.fingerprint})), **kwargs)


def codes(report):
    return {reason["code"] for reason in report["reasons"]}


class NumericalProfileTests(unittest.TestCase):
    def test_roundtrip_canonical_order_and_detached_input(self):
        data = profile_data()
        p = NumericalProfile.parse(data)
        data["phases"].reverse()
        self.assertEqual(p.fingerprint, NumericalProfile.parse(data).fingerprint)
        self.assertEqual(p, NumericalProfile.parse(p.as_record()))
        data["state"]["kv_dtype"] = "float32"
        self.assertEqual(p.state.kv_dtype, "float16")
        returned = p.as_record()
        returned["phases"][0]["activation"]["encoding"] = "symmetric_a4"
        self.assertEqual(p.phase("decode").activation.encoding, "float16")

    def test_semantic_fields_affect_identity_and_state_history(self):
        original = NumericalProfile.parse(profile_data("symmetric_a8"))
        changes = [
            ("weights", "layout_sha256", "f" * 64),
            ("state", "layout_sha256", "e" * 64),
            ("state", "update_semantics_sha256", "e" * 64),
            ("wire", "abi_sha256", "e" * 64),
        ]
        for section, field, value in changes:
            data = profile_data("symmetric_a8")
            data[section][field] = value
            changed = NumericalProfile.parse(data)
            self.assertNotEqual(original.fingerprint, changed.fingerprint)
            self.assertNotEqual(original.state_history_sha256, changed.state_history_sha256)
        for field in ("rotation_sha256", "outlier_sha256", "activation_calibration_sha256", "input_layout_sha256"):
            data = profile_data("symmetric_a8")
            data["phases"][0][field] = "f" * 64
            self.assertNotEqual(original.fingerprint, NumericalProfile.parse(data).fingerprint)
        for field, value in (("rounding", "toward_zero"), ("denormals", "flush_to_zero"), ("division", "approximate_reciprocal")):
            data = profile_data("symmetric_a8")
            data["phases"][0]["activation"][field] = value
            self.assertNotEqual(original.fingerprint, NumericalProfile.parse(data).fingerprint)

    def test_unknown_missing_duplicate_phase_and_self_promotion_rejected(self):
        variations = []
        data = profile_data(); data["auto_relax_precision"] = True; variations.append(data)
        data = profile_data(); del data["wire"]; variations.append(data)
        data = profile_data(); data["phases"].append(deepcopy(data["phases"][0])); variations.append(data)
        data = profile_data(); data["phases"][0]["phase"] = "draft"; variations.append(data)
        data = profile_data(); data["qualified"] = True; variations.append(data)
        data = profile_data(); data["executable"] = True; variations.append(data)
        data = profile_data(); data["weights"]["block_elements"] = True; variations.append(data)
        data = profile_data(); data["weights"]["layout_sha256"] = "ABC" * 20; variations.append(data)
        for data in variations:
            with self.subTest(data=data):
                with self.assertRaises(ValidationError): NumericalProfile.parse(data)

    def test_nonlinear_codes_require_lookup_or_float_conversion(self):
        p = NumericalProfile.parse(profile_data("symmetric_a8", nonlinear=True))
        self.assertEqual(p.phase("decode").operands.weight_dtype, "int8")
        self.assertEqual(apply(p)["status"], "REPRESENTABLE")
        for dtype in ("int4", "uint4", "int8"):
            data = profile_data("symmetric_a8", nonlinear=True)
            data["phases"][1]["operands"].update(weight_decode="raw_integer", weight_dtype=dtype)
            with self.assertRaisesRegex(ValidationError, "nonlinear"): NumericalProfile.parse(data)

    def test_signedness_and_activation_width_are_not_relaxed(self):
        data = profile_data("symmetric_a4")
        data["phases"][0]["operands"]["weight_dtype"] = "int4"
        with self.assertRaisesRegex(ValidationError, "signedness"): NumericalProfile.parse(data)
        data = profile_data("symmetric_a8")
        data["phases"][0]["operands"]["activation_dtype"] = "int4"
        with self.assertRaisesRegex(ValidationError, "activation preparation"): NumericalProfile.parse(data)
        data = profile_data("symmetric_a4")
        data["phases"][0]["operands"]["activation_dtype"] = "int8"
        self.assertEqual(NumericalProfile.parse(data).phase("prefill").activation.encoding, "symmetric_a4")

    def test_scaled_fp4_fp8_are_unsupported_not_affine_integer_modes(self):
        for encoding, dtype, bits in (("scaled_fp4_e2m1", "fp4_e2m1", 4), ("scaled_fp8", "fp8_e4m3", 8), ("scaled_fp8", "fp8_e5m2", 8)):
            data = profile_data()
            data["weights"].update(encoding=encoding, code_dtype=dtype, storage_bits=bits, interpretation="floating")
            with self.assertRaisesRegex(ValidationError, "unsupported"): NumericalProfile.parse(data)
        data = profile_data()
        data["weights"]["scale_hierarchy_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValidationError, "unknown"): NumericalProfile.parse(data)

    def test_loader_duplicate_nonfinite_oversize_and_tampering(self):
        p = NumericalProfile.parse(profile_data())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(p.as_record()), encoding="utf-8")
            self.assertEqual(load_numerical_profile(path), p)
            tampered = p.as_record(); tampered["state"]["kv_dtype"] = "float32"
            for payload in (json.dumps(tampered), '{"x":1,"x":2}', '{"x":NaN}', '[NaN]', '"' + "x" * MAX_INPUT_BYTES + '"'):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValidationError): load_numerical_profile(path)

    def test_exact_approval_and_scope_not_vendor_or_phase_inferred(self):
        p = NumericalProfile.parse(profile_data("symmetric_a8"))
        request = scope(p, "verify", 8)
        result = apply(p, request)
        self.assertEqual(result["status"], "REPRESENTABLE")
        self.assertIsNone(result["complete_phase_cost_us"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["executable"])
        self.assertIn("NUMERICAL_MODE_NOT_APPROVED", codes(apply(p, request, approved_profile_sha256s=frozenset())))
        for changed in (replace(request, actual_rows=16), replace(request, phase="prefill"), replace(request, backend="other-backend"), replace(request, architecture="other-arch"), replace(request, runtime_identity_sha256="e" * 64), replace(request, implementation_sha256="e" * 64)):
            self.assertIn("CAPABILITY_SCOPE_MISMATCH", codes(apply(p, changed, capability(request))))
        different = NumericalProfile.parse(profile_data("symmetric_a4"))
        self.assertIn("REQUEST_PROFILE_MISMATCH", codes(apply(different, request)))

    def test_k640_is_not_a_256_storage_block_tail(self):
        p = NumericalProfile.parse(profile_data("symmetric_a8", nonlinear=True))
        self.assertIn("STORAGE_BLOCK_INCOMPATIBLE", codes(apply(p, scope(p, k=640))))
        affine = NumericalProfile.parse(profile_data("symmetric_a8"))
        self.assertEqual(apply(affine, scope(affine, k=640))["status"], "REPRESENTABLE")
        self.assertIn("ACTIVATION_GROUP_INCOMPATIBLE", codes(apply(affine, scope(affine, k=641))))

    def test_profile_cannot_borrow_an_undeclared_phase(self):
        data = profile_data(); data["phases"] = [data["phases"][1]]
        p = NumericalProfile.parse(data)
        self.assertIn("PHASE_NOT_DECLARED", codes(apply(p, scope(p, "verify", 8))))
        self.assertIn("DECODE_ROW_SCOPE_MISMATCH", codes(apply(p, scope(p, "decode", 2))))

    def test_retained_state_invalidates_after_any_phase_history_change(self):
        p = NumericalProfile.parse(profile_data())
        state = StateHistoryBinding(p.fingerprint, p.state_history_sha256)
        self.assertEqual(apply(p, retained_state=state)["status"], "REPRESENTABLE")
        data = profile_data(); data["phases"][0]["operands"]["reduction_sha256"] = "e" * 64
        changed = NumericalProfile.parse(data)
        self.assertEqual(changed.state, p.state)
        self.assertIn("STATE_HISTORY_MISMATCH", codes(apply(changed, retained_state=state)))
        self.assertIn("STATE_HISTORY_MISMATCH", codes(apply(p, retained_state=replace(state, state_history_sha256="e" * 64))))

    def test_a4_verifier_cannot_commit_a16_target_even_with_same_target_label(self):
        target = NumericalProfile.parse(profile_data())
        verifier = NumericalProfile.parse(profile_data("symmetric_a4"))
        self.assertEqual(target.target_semantics_sha256, verifier.target_semantics_sha256)
        result = apply(verifier, scope(verifier, "verify", 8), authoritative_target=target, exact_commit=True,
                       approved_profile_sha256s=frozenset({target.fingerprint, verifier.fingerprint}))
        self.assertIn("AUTHORITATIVE_NUMERICAL_MISMATCH", codes(result))
        self.assertIn("AUTHORITATIVE_TARGET_REQUIRED", codes(apply(verifier, scope(verifier, "verify", 8), exact_commit=True)))
        result = apply(target, scope(target, "verify", 8), authoritative_target=target, exact_commit=True)
        self.assertEqual(result["status"], "REPRESENTABLE")
        self.assertFalse(result["qualified"])
        data = profile_data(); data["phases"][2]["row_invariant"] = False
        changed = NumericalProfile.parse(data)
        self.assertIn("AUTHORITATIVE_NUMERICAL_MISMATCH", codes(apply(changed, scope(changed, "verify", 8), authoritative_target=changed, exact_commit=True)))

    def test_approximate_and_proposal_modes_do_not_self_authorize_exact_commit(self):
        for mode in ("proposal_only", "approximate_target"):
            data = profile_data(); data["mode"] = mode
            p = NumericalProfile.parse(data)
            self.assertEqual(apply(p)["status"], "REPRESENTABLE")
            self.assertIn("AUTHORITATIVE_NUMERICAL_MISMATCH", codes(apply(p, scope(p, "verify", 8), authoritative_target=p, exact_commit=True)))

    def test_calibration_requires_complete_positive_exact_scope(self):
        p = NumericalProfile.parse(profile_data("symmetric_a8"))
        request = scope(p, "verify", 8)
        missing = apply(p, request, require_calibration=True)
        self.assertIn("CALIBRATION_MISSING", codes(missing)); self.assertIsNone(missing["complete_phase_cost_us"])
        cal = calibration(request)
        self.assertEqual(apply(p, request, calibration=cal, require_calibration=True)["complete_phase_cost_us"], 13)
        self.assertIn("CALIBRATION_NOT_NATIVE", codes(apply(p, request, calibration=calibration(request, "fixture"), require_calibration=True)))
        self.assertIn("CALIBRATION_SCOPE_MISMATCH", codes(apply(p, request, calibration=calibration(scope(p)))))
        bad = cal.as_record(); bad["costs_us"] = {key: 0 for key in bad["costs_us"]}; del bad["calibration_sha256"]
        with self.assertRaisesRegex(ValidationError, "positive"): NumericalCalibration.parse(bad)
        bad = cal.as_record(); del bad["costs_us"]["preparation_us"]
        with self.assertRaises(ValidationError): NumericalCalibration.parse(bad)
        for value in (True, -1, float("nan"), 2**63):
            bad = cal.as_record(); bad["costs_us"]["compute_us"] = value
            with self.assertRaises(ValidationError): NumericalCalibration.parse(bad)

    def test_exact_commit_cannot_borrow_authority_across_prefill_decode_or_wire_history(self):
        target = NumericalProfile.parse(profile_data())
        for changed_section in ("prefill", "decode", "wire", "input_layout"):
            data = profile_data()
            if changed_section == "wire":
                data["wire"].update(dtype="int4", quantization_sha256="f" * 64)
            elif changed_section == "input_layout":
                data["phases"][0]["input_layout_sha256"] = "e" * 64
            else:
                index = 0 if changed_section == "prefill" else 1
                data["phases"][index] = phase_data(changed_section, "symmetric_a4", "raw_integer")
            verifier = NumericalProfile.parse(data)
            self.assertEqual(verifier.phase("verify").arithmetic_sha256, target.phase("decode").arithmetic_sha256)
            result = apply(verifier, scope(verifier, "verify", 8), authoritative_target=target, exact_commit=True,
                retained_state=StateHistoryBinding(verifier.fingerprint, verifier.state_history_sha256),
                approved_profile_sha256s=frozenset({target.fingerprint, verifier.fingerprint}))
            with self.subTest(changed_section=changed_section):
                self.assertIn("AUTHORITATIVE_NUMERICAL_MISMATCH", codes(result))

    def test_public_frozen_records_cannot_bypass_integrity_via_replace(self):
        p = NumericalProfile.parse(profile_data())
        with self.assertRaisesRegex(ValidationError, "fingerprint"):
            apply(replace(p, mode="approximate_target"))
        request = scope(p)
        forged = replace(capability(request), scope=replace(request, architecture="different"))
        with self.assertRaisesRegex(ValidationError, "fingerprint"): apply(p, request, forged)
        forged_cal = replace(calibration(request), transient_peak_bytes=0)
        with self.assertRaisesRegex(ValidationError, "fingerprint"): apply(p, calibration=forged_cal)

    def test_wire_pair_exact_direction_runtime_dtype_and_mechanism(self):
        p = NumericalProfile.parse(profile_data("symmetric_a4"))
        pair = BoundaryCapability.parse({
            "boundary_capability_schema": BOUNDARY_CAPABILITY_SCHEMA,
            "source_profile_sha256": p.fingerprint, "target_profile_sha256": p.fingerprint,
            "source_runtime_sha256": "c" * 64, "target_runtime_sha256": "d" * 64,
            "wire_sha256": p.wire.fingerprint, "mechanism": "network_stream", "qualified": False, "executable": False,
        })
        arguments = dict(source_runtime_sha256="c" * 64, target_runtime_sha256="d" * 64, mechanism="network_stream")
        self.assertEqual(validate_boundary_applicability(p, p, pair, **arguments)["status"], "REPRESENTABLE")
        self.assertIn("PAIR_CAPABILITY_MISSING", codes(validate_boundary_applicability(p, p, None, **arguments)))
        arguments["mechanism"] = "gpu_dma"
        self.assertIn("PAIR_CAPABILITY_MISMATCH", codes(validate_boundary_applicability(p, p, pair, **arguments)))
        arguments["mechanism"] = "network_stream"; arguments["source_runtime_sha256"] = "d" * 64; arguments["target_runtime_sha256"] = "c" * 64
        self.assertIn("PAIR_CAPABILITY_MISMATCH", codes(validate_boundary_applicability(p, p, pair, **arguments)))
        data = profile_data("symmetric_a4"); data["wire"].update(dtype="int4", quantization_sha256="f" * 64)
        different = NumericalProfile.parse(data)
        self.assertIn("WIRE_CONTRACT_MISMATCH", codes(validate_boundary_applicability(p, different, pair, **arguments)))
        bad = replace(pair, mechanism="gpu_dma")
        with self.assertRaisesRegex(ValidationError, "fingerprint"): validate_boundary_applicability(p, p, bad, **arguments)


if __name__ == "__main__":
    unittest.main()
