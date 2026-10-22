from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.request_envelope import (
    DECLARATION_PURPOSES, ENVELOPE_SCHEMA, FEATURES, IDENTITY_FIELDS, REQUEST_SCHEMA,
    ServingRequestEnvelope, load_serving_request_envelope, preflight_serving_request,
)
from tensormeld.schema import MAX_INPUT_BYTES, ValidationError


def identity_record():
    return {
        **{name: f"{index + 1:064x}" for index, name in enumerate(sorted(IDENTITY_FIELDS))},
        "runtime_identity_sha256": ["a" * 64, "b" * 64],
    }


def envelope_record():
    return {
        "envelope_schema": ENVELOPE_SCHEMA, "identity": identity_record(),
        "limits": {"context_windows": [128, 512], "max_input_tokens": 480,
                   "max_output_tokens": 64, "max_input_bytes": 2048, "concurrency": 1},
        "features": dict(FEATURES), "cache_modes": ["disabled"],
        "declaration_references": [
            {"kind": "request_family_declaration", "purpose": purpose,
             "provenance": "fixture", "record_sha256": f"{index + 100:064x}"}
            for index, purpose in enumerate(sorted(DECLARATION_PURPOSES))
        ],
        "native_qualified": False, "executable": False,
    }


def request_record():
    return {
        "request_schema": REQUEST_SCHEMA, "identity": identity_record(),
        "rendered_input_sha256": "c" * 64, "input_bytes": 1024, "input_tokens": 64,
        "context_window_tokens": 128, "max_output_tokens": 64, "concurrency": 1,
        "task": "text_completion", "sampler": "greedy", "stream": False, "cache_mode": "disabled",
    }


class ServingRequestEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.envelope = ServingRequestEnvelope.parse(envelope_record())

    def evaluate(self, request=None, current=None, envelope=None):
        return preflight_serving_request(
            envelope or self.envelope,
            request=request if request is not None else request_record(),
            current_identity=current if current is not None else identity_record(),
        )

    def test_content_variability_does_not_change_durable_envelope_or_authorize(self):
        first = self.evaluate()
        request = request_record()
        request["rendered_input_sha256"] = "d" * 64
        request["input_tokens"] = 63
        other = self.evaluate(request=request)
        self.assertTrue(first["applies"])
        self.assertTrue(other["applies"])
        self.assertEqual(first["envelope_sha256"], other["envelope_sha256"])
        self.assertNotEqual(first["request_sha256"], other["request_sha256"])
        self.assertNotEqual(first["result_sha256"], other["result_sha256"])
        for result in (first, other):
            self.assertTrue(result["declaration_only"])
            for field in ("native_qualified", "inference_request_authorized", "executable",
                          "resource_admission_performed", "launch_canary_performed"):
                self.assertFalse(result[field])

    def test_every_current_or_requested_identity_dimension_is_exact(self):
        for field in sorted(IDENTITY_FIELDS | {"runtime_identity_sha256"}):
            changed = identity_record()
            changed[field] = ["f" * 64] if field == "runtime_identity_sha256" else "f" * 64
            for scope in ("CURRENT", "REQUEST"):
                request = request_record()
                request["identity"] = changed if scope == "REQUEST" else identity_record()
                with self.subTest(field=field, scope=scope):
                    result = self.evaluate(request=request, current=changed if scope == "CURRENT" else None)
                    self.assertFalse(result["applies"])
                    self.assertIn(f"{scope}_{field.removesuffix('_sha256').upper()}_MISMATCH", result["reasons"])

    def test_full_context_boundary_accepts_and_one_extra_token_rejects(self):
        request = request_record()
        request.update(context_window_tokens=512, input_tokens=448)
        self.assertTrue(self.evaluate(request=request)["applies"])
        request["input_tokens"] = 449
        result = self.evaluate(request=request)
        self.assertIn("INPUT_PLUS_OUTPUT_EXCEEDS_CONTEXT", result["reasons"])
        self.assertFalse(result["applies"])

    def test_limits_are_checked_independently_and_shapes_are_not_interpolated(self):
        cases = (
            ("input_tokens", 481, "INPUT_TOKEN_LIMIT_EXCEEDED"),
            ("input_bytes", 2049, "INPUT_BYTE_LIMIT_EXCEEDED"),
            ("max_output_tokens", 65, "OUTPUT_TOKEN_LIMIT_EXCEEDED"),
            ("context_window_tokens", 256, "CONTEXT_WINDOW_NOT_DECLARED"),
            ("concurrency", 2, "CONCURRENCY_NOT_DECLARED"),
        )
        for field, value, reason in cases:
            request = request_record()
            request[field] = value
            with self.subTest(field=field):
                result = self.evaluate(request=request)
                self.assertFalse(result["applies"])
                self.assertIn(reason, result["reasons"])

    def test_feature_and_cache_changes_fail_closed(self):
        for field, value, reason in (
            ("task", "chat_completion", "UNSUPPORTED_TASK"),
            ("sampler", "categorical", "UNSUPPORTED_SAMPLER"),
            ("stream", True, "STREAMING_NOT_DECLARED"),
            ("cache_mode", "prefix_reuse", "CACHE_MODE_NOT_DECLARED"),
        ):
            request = request_record()
            request[field] = value
            with self.subTest(field=field):
                self.assertIn(reason, self.evaluate(request=request)["reasons"])

    def test_request_flags_paths_prompt_tools_and_unknown_fields_are_rejected(self):
        for field in ("prompt", "messages", "tools", "backend_flags", "model_path", "peer_address",
                      "shell_command", "temperature", "qualified", "lease_id"):
            request = request_record()
            request[field] = "fixture-unknown-field"
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "unknown="):
                self.evaluate(request=request)

    def test_descriptors_require_exact_counts_digest_and_boolean_stream(self):
        for field in ("input_bytes", "input_tokens", "context_window_tokens", "max_output_tokens", "concurrency"):
            for value in (True, 0, -1, 1.5, 1 << 63):
                request = request_record()
                request[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    self.evaluate(request=request)
        for field, value in (("rendered_input_sha256", "invalid"), ("stream", 0)):
            request = request_record()
            request[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.evaluate(request=request)

    def test_imported_flags_and_evidence_hashes_cannot_self_promote(self):
        for field in ("native_qualified", "executable"):
            raw = envelope_record()
            raw[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "self-promote"):
                ServingRequestEnvelope.parse(raw)
        for field, value in (("kind", "serving_equivalence"), ("provenance", "native")):
            raw = envelope_record()
            raw["declaration_references"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "not native qualification"):
                ServingRequestEnvelope.parse(raw)

    def test_declaration_categories_and_unique_references_are_required(self):
        for change in ("missing_category", "duplicate_hash"):
            raw = envelope_record()
            if change == "missing_category":
                raw["declaration_references"][0]["purpose"] = raw["declaration_references"][1]["purpose"]
            else:
                raw["declaration_references"][0]["record_sha256"] = raw["declaration_references"][1]["record_sha256"]
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ServingRequestEnvelope.parse(raw)

    def test_durable_identity_rejects_lease_and_replay_output_fields(self):
        for field in ("lease_id", "reservation_observation_id", "expected_output_sha256", "e4_spec_sha256"):
            raw = envelope_record()
            raw[field] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ServingRequestEnvelope.parse(raw)

    def test_reordered_sets_round_trip_and_deterministic_result(self):
        raw = envelope_record()
        raw["identity"]["runtime_identity_sha256"].reverse()
        raw["limits"]["context_windows"].reverse()
        raw["declaration_references"].reverse()
        other = ServingRequestEnvelope.parse(raw)
        self.assertEqual(self.envelope, other)
        self.assertEqual(self.envelope, ServingRequestEnvelope.parse(self.envelope.as_record()))
        self.assertEqual(self.evaluate(), self.evaluate())

    def test_mutated_envelope_cannot_reuse_its_fingerprint(self):
        bad = replace(self.envelope, max_output_tokens=63)
        with self.assertRaisesRegex(ValidationError, "fingerprint mismatch"):
            self.evaluate(envelope=bad)

    def test_envelope_does_not_infer_new_supported_features(self):
        for field, value in (("sampling", True), ("sse", True), ("tools", True), ("text", 1)):
            raw = envelope_record()
            raw["features"][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ServingRequestEnvelope.parse(raw)
        raw = envelope_record()
        raw["cache_modes"] = ["disabled", "prefix_reuse"]
        with self.assertRaises(ValidationError):
            ServingRequestEnvelope.parse(raw)

    def test_loader_round_trip_duplicate_keys_and_size_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "envelope.json"
            path.write_text(json.dumps(self.envelope.as_record()), encoding="utf-8")
            self.assertEqual(load_serving_request_envelope(path), self.envelope)
            path.write_text('{"envelope_schema":"a","envelope_schema":"b"}', encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_serving_request_envelope(path)
            path.write_bytes(b" " * (MAX_INPUT_BYTES + 1))
            with self.assertRaisesRegex(ValidationError, "exceeds 2 MiB"):
                load_serving_request_envelope(path)


if __name__ == "__main__":
    unittest.main()
