from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld import transport_commissioning as campaign
from tensormeld.schema import ValidationError


class TransportCommissioningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        artifacts = {}
        for name in ("harness_binary", "source_record", "kernel_build_record"):
            path = self.root / name
            path.write_bytes(b"synthetic provenance: " + name.encode())
            artifacts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        kernel_path = Path(artifacts["kernel_build_record"]["path"])
        kernel_path.write_text(json.dumps({"schema": "strix-private-full-kernel-build-v1", "kernel_release": "synthetic-kernel",
                                          "modules": {"thunderbolt_stream": {"gnu_build_id": "abcd", "sha256": "a" * 64}}}))
        artifacts["kernel_build_record"]["sha256"] = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
        self.plan = {
            "schema": campaign.PLAN_SCHEMA, "campaign_id": "synthetic-campaign",
            "production_qualified": False, "launch_authorized": False,
            "nodes": [{"id": role, "artifacts": deepcopy(artifacts), "expected": {
                "local_endpoint_identity": char * 32, "gpu_uuid": char * 32,
                "gpu_architecture": "gfx1151", "gpu_runtime_version": 70240000,
                "gpu_driver_version": 70240000, "kernel_identity": "synthetic-kernel",
                "driver_identity": "thunderbolt_stream:gnu-build-id:abcd", "driver_build_id": "abcd",
                "abi_identity": "synthetic-abi"}} for role, char in (("a", "1"), ("b", "2"))],
            "runs": [{"run_id": "forward", "sender": "a", "receiver": "b", "physical_link_id": "synthetic-link",
                      "connection_epoch": 20, "sequence": 1, "payload_bytes": 4097, "pattern_seed": 31,
                      "model_plan_id": 40, "request_id": 50, "generation": 60,
                      "boundary_id": 70, "shape_fingerprint": 80, "dtype_id": 1}]}
        run = self.plan["runs"][0]
        run["shape_fingerprint"] = campaign.expected_shape_fingerprint(self.plan["campaign_id"], run)
        self.sender, self.receiver = [self.receipt(role) for role in ("sender", "receiver")]

    def receipt(self, role):
        run = self.plan["runs"][0]
        node = self.plan["nodes"][0 if role == "sender" else 1]
        peer = self.plan["nodes"][1 if role == "sender" else 0]
        expected = campaign.expected_payload_sha256(run["payload_bytes"], run["pattern_seed"])
        applicable = ({"local_tx_complete", "transport_ack", "consumer_credit"} if role == "sender" else
                      {"full_payload_verified", "gpu_acquire_complete", "consumer_complete", "receive_reposted", "consumer_credit"})
        return {**{field: run[field] for field in campaign.RUN_FIELDS - {"sender", "receiver"}},
                **node["expected"], **{flag: flag in applicable or flag == "controller_frames_complete" for flag in campaign.FLAGS},
                "schema": campaign.RECEIPT_SCHEMA, "campaign_id": self.plan["campaign_id"],
                "role": role, "executed": True, "production_qualified": False,
                "iterations": 1, "completed_count": 1, "pattern_version": 1,
                "peer_endpoint_identity": peer["expected"]["local_endpoint_identity"],
                "device_identity": "synthetic-open-" + role, "device_instance_generation": 1,
                "pool_binding_identity": 2, "import_generation": 3,
                "expected_payload_sha256": expected, "observed_payload_sha256": expected if role == "receiver" else None,
                "cleanup": "released", "error": ""}

    def evaluate(self):
        return campaign.evaluate_pair(self.plan, "forward", self.sender, self.receiver)

    def test_paired_observations_never_authorize(self):
        result = self.evaluate()
        self.assertEqual(result["outcome"], "OBSERVATIONS_MATCH")
        for field in ("production_qualified", "launch_authorized", "external_dma_visibility_qualified"):
            self.assertIs(result[field], False)
        self.assertTrue(result["requires_independent_native_review"])

    def test_exact_descriptor_peer_runtime_and_role_binding(self):
        for field, value in (("connection_epoch", 99), ("payload_bytes", 4096), ("run_id", "other"),
                             ("shape_fingerprint", 1), ("peer_endpoint_identity", "3" * 32),
                             ("driver_build_id", "dead"), ("gpu_runtime_version", 1),
                             ("gpu_architecture", "other"), ("role", "sender")):
            with self.subTest(field=field):
                old = self.receiver[field]
                self.receiver[field] = value
                self.assertEqual(self.evaluate()["outcome"], "FAILED")
                self.receiver[field] = old

    def test_every_required_lifecycle_gate_is_independent(self):
        for receipt, flags in ((self.sender, ("executed", "controller_frames_complete", "local_tx_complete", "transport_ack", "consumer_credit")),
                               (self.receiver, ("executed", "controller_frames_complete", "gpu_acquire_complete", "consumer_complete", "receive_reposted", "full_payload_verified", "consumer_credit"))):
            for flag in flags:
                with self.subTest(flag=flag):
                    receipt[flag] = False
                    self.assertEqual(self.evaluate()["outcome"], "FAILED")
                    receipt[flag] = True

    def test_same_forged_hash_on_both_hosts_fails_independent_oracle(self):
        self.sender["expected_payload_sha256"] = "a" * 64
        self.receiver["expected_payload_sha256"] = "a" * 64
        self.receiver["observed_payload_sha256"] = "a" * 64
        self.assertEqual(self.evaluate()["outcome"], "FAILED")

    def test_failure_cleanup_and_partial_transfer_do_not_pass(self):
        for field, value in (("cleanup", "retained"), ("completed_count", 0), ("error", "late copy failed"),
                             ("import_generation", 0), ("observed_payload_sha256", None)):
            with self.subTest(field=field):
                old = self.receiver[field]
                self.receiver[field] = value
                self.assertEqual(self.evaluate()["outcome"], "FAILED")
                self.receiver[field] = old

    def test_unknown_fields_boolean_numbers_and_self_promotion_rejected(self):
        for field, value in (("unknown", 1), ("payload_bytes", True), ("executed", 1),
                             ("production_qualified", True), ("pattern_version", 2)):
            modified = deepcopy(self.receiver)
            modified[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                campaign.evaluate_pair(self.plan, "forward", self.sender, modified)

    def test_artifact_drift_blocks_even_matching_receipts(self):
        Path(self.plan["nodes"][0]["artifacts"]["harness_binary"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValidationError, "identity mismatch"):
            self.evaluate()

    def test_kernel_build_record_must_bind_expected_loaded_identity(self):
        for field, value in (("kernel_identity", "other"), ("driver_build_id", "beef")):
            plan = deepcopy(self.plan)
            plan["nodes"][0]["expected"][field] = value
            if field == "driver_build_id":
                plan["nodes"][0]["expected"]["driver_identity"] = "thunderbolt_stream:gnu-build-id:" + value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                campaign.validate_plan(plan)

    def config_inputs(self):
        refs = {}
        for role, receipt in (("sender", self.sender), ("receiver", self.receiver)):
            config = {field: receipt[field] for field in campaign.CONFIG_FIELDS}
            config.update(schema="strix-direct-gpu-commissioning-config-v1", executed=False)
            path = self.root / (role + "-config.json")
            path.write_text(json.dumps(config))
            refs[role + "_config"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return {"schema": "tensormeld/direct-gpu-commissioning-config-inputs-v1", "campaign_id": self.plan["campaign_id"],
                "nodes": [{"id": node["id"], "artifacts": node["artifacts"]} for node in self.plan["nodes"]],
                "runs": [{"run_id": "forward", "sender": "a", "receiver": "b", **refs}]}

    def test_host_only_configuration_assembly_cannot_claim_execution(self):
        inputs = self.config_inputs()
        self.assertEqual(campaign.assemble_plan(inputs), self.plan)
        self.assertIs(campaign.assemble_plan(inputs)["launch_authorized"], False)

    def test_config_assembly_rejects_wrong_peer_descriptor_and_pattern(self):
        for field, value in (("executed", True), ("peer_endpoint_identity", "3" * 32),
                             ("payload_bytes", 1), ("expected_payload_sha256", "0" * 64)):
            inputs = self.config_inputs()
            ref = inputs["runs"][0]["receiver_config"]
            path = Path(ref["path"])
            value_json = json.loads(path.read_text())
            value_json[field] = value
            path.write_text(json.dumps(value_json))
            ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.subTest(field=field), self.assertRaises(ValidationError):
                campaign.assemble_plan(inputs)

    def test_campaign_requires_exact_coverage_and_keeps_failures(self):
        pair = {"run_id": "forward", "sender": self.sender, "receiver": self.receiver}
        self.assertEqual(campaign.evaluate_campaign(self.plan, [pair])["outcome"], "OBSERVATIONS_MATCH")
        for pairs in ([], [pair, pair], [{**pair, "run_id": "unknown"}]):
            with self.assertRaises(ValidationError):
                campaign.evaluate_campaign(self.plan, pairs)
        self.receiver["cleanup"] = "retained"
        self.assertEqual(campaign.evaluate_campaign(self.plan, [pair])["outcome"], "FAILED")

    def test_fresh_epoch_and_exact_nodes_required(self):
        other = deepcopy(self.plan["runs"][0])
        other.update(run_id="reverse", sender="b", receiver="a")
        other["shape_fingerprint"] = campaign.expected_shape_fingerprint(self.plan["campaign_id"], other)
        self.plan["runs"].append(other)
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            campaign.validate_plan(self.plan)
        other["connection_epoch"] = 21
        campaign.validate_plan(self.plan)
        other["receiver"] = "unknown"
        with self.assertRaises(ValidationError):
            campaign.validate_plan(self.plan)

    def test_preparation_is_private_exclusive_and_nonexecuting(self):
        out = self.root / "prepared"
        result = campaign.prepare_campaign(self.plan, out)
        self.assertIs(result["executed"], False)
        self.assertIs(result["launch_authorized"], False)
        self.assertEqual(json.loads((out / "plan.json").read_text()), self.plan)
        with self.assertRaises(FileExistsError):
            campaign.prepare_campaign(self.plan, out)
        checkout = self.root / "checkout"
        checkout.mkdir()
        (checkout / ".git").write_text("gitdir: elsewhere")
        with self.assertRaisesRegex(ValidationError, "outside Git"):
            campaign.prepare_campaign(self.plan, checkout / "private")

    def test_template_preserves_unknown_hardware_and_covers_both_directions(self):
        out = self.root / "template"
        result = campaign.write_template(out)
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertEqual(result["run_count"], 28)
        plan = json.loads((out / "plan.template.json").read_text())
        self.assertEqual({run["sender"] for run in plan["runs"]}, {"node-a", "node-b"})
        self.assertTrue(all(run["connection_epoch"] is None for run in plan["runs"]))
        with self.assertRaises(ValidationError):
            campaign.validate_plan(plan)
        with self.assertRaises(FileExistsError):
            campaign.write_template(out)

    def test_bounded_duplicate_and_nonfinite_json(self):
        path = self.root / "input.json"
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '[' * 2000, ' ' * (campaign.MAX_INPUT_BYTES + 1)):
            path.write_text(raw)
            with self.assertRaises(ValidationError):
                campaign._json(path)

    def test_native_pattern_wrap_and_boundaries(self):
        # Independently fixed known bytes: empty slices/truncation are not accepted.
        self.assertNotEqual(campaign.expected_payload_sha256(4096, 0), campaign.expected_payload_sha256(4097, 0))
        self.assertNotEqual(campaign.expected_payload_sha256(4097, 0), campaign.expected_payload_sha256(4097, (1 << 64) - 1))
        self.assertNotEqual(campaign.expected_payload_sha256(4097, 0), campaign.expected_payload_sha256(4097, 0, 2))
        with self.assertRaises(ValidationError):
            campaign.expected_payload_sha256(0, 0)


if __name__ == "__main__":
    unittest.main()
