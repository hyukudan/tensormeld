from __future__ import annotations

import copy
import unittest

from tensormeld.adapter_contract import AdapterCapabilities, validate_candidate_representability
from tensormeld.config_v2 import Config
from tensormeld.planning_contract import PlanningInput
from tensormeld.planner_v2 import plan_v2
from tensormeld.schema import ValidationError
from test_planner_v2 import fixture


def capability(config_dict, *, mixed=True, remote=True, external_coord=True):
    return {
        "adapter_schema": "tensormeld/adapter-capabilities-v1",
        "adapter_id": "test-native",
        "engine": "test-engine",
        "engine_revision": "deadbeef",
        "placement": {
            "strategies": ["whole_blocks"],
            "exact_owner_binding": True,
            "explicit_unit_ranges": True,
            "mixed_backends": mixed,
            "remote_compute": remote,
            "coordinator_outside_compute": external_coord,
            "max_compute_devices": 16,
            "max_compute_nodes": 16,
            "max_segments": 32,
        },
        "route_modes": ["direct", "via_coordinator"],
        "coordinator_nodes": [n["id"] for n in config_dict["nodes"]],
        "devices": [
            {"id": d["id"], "node": d["node"], "backend": d["backend"]}
            for d in config_dict["devices"]
        ],
    }


def planned(count=2, distributed=False):
    c, p = fixture(count=count, units=3)
    if distributed:
        c["profiles"]["interactive"]["execution_mode"] = "distributed"
    cfg = Config.parse(c)
    planning = PlanningInput.parse(p, cfg)
    result = plan_v2(cfg, planning)
    return c, p, cfg, planning, result["best"]


class AdapterContractTests(unittest.TestCase):
    def test_exact_candidate_is_representable_but_not_executable(self):
        c,_,cfg,p,best = planned()
        report = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(capability(c))
        )
        self.assertEqual(report["status"], "REPRESENTABLE")
        self.assertTrue(report["exact"])
        self.assertFalse(report["qualified"])
        self.assertFalse(report["executable"])
        self.assertEqual(report["reasons"], [])

    def test_adapter_cannot_self_promote_execution(self):
        c,_,cfg,p,best = planned()
        best = copy.deepcopy(best)
        best["executable"] = True
        with self.assertRaises(ValidationError):
            validate_candidate_representability(
                cfg, p, best, AdapterCapabilities.parse(capability(c))
            )

    def test_exact_binding_required(self):
        c,_,cfg,p,best = planned()
        cap = capability(c)
        cap["placement"]["exact_owner_binding"] = False
        r = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(cap)
        )
        self.assertEqual(r["status"], "REJECTED")
        self.assertIn(
            "EXACT_OWNER_BINDING_UNSUPPORTED", {x["code"] for x in r["reasons"]}
        )

    def test_explicit_ranges_required(self):
        c,_,cfg,p,best = planned()
        cap = capability(c)
        cap["placement"]["explicit_unit_ranges"] = False
        r = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(cap)
        )
        self.assertIn(
            "EXPLICIT_UNIT_RANGES_UNSUPPORTED", {x["code"] for x in r["reasons"]}
        )

    def test_remote_compute_can_be_rejected(self):
        c,_,cfg,p,best = planned(distributed=True)
        r = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(capability(c, remote=False))
        )
        self.assertIn("REMOTE_COMPUTE_UNSUPPORTED", {x["code"] for x in r["reasons"]})

    def test_missing_device_rejected(self):
        c,_,cfg,p,best = planned(distributed=True)
        cap = capability(c)
        cap["devices"] = cap["devices"][:1]
        r = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(cap)
        )
        self.assertIn("DEVICE_NOT_EXPOSED", {x["code"] for x in r["reasons"]})

    def test_backend_mismatch_rejected(self):
        c,_,cfg,p,best = planned()
        cap = capability(c)
        cap["devices"][0]["backend"] = "other"
        r = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(cap)
        )
        self.assertIn(
            "DEVICE_BACKEND_MISMATCH", {x["code"] for x in r["reasons"]}
        )

    def test_mixed_backends_require_explicit_support(self):
        c,praw = fixture(count=2, units=3)
        c["profiles"]["interactive"]["execution_mode"] = "distributed"
        c["devices"][1]["backend"] = "hip"
        cfg = Config.parse(c)
        planning = PlanningInput.parse(praw, cfg)
        best = plan_v2(cfg, planning)["best"]
        r = validate_candidate_representability(
            cfg,
            planning,
            best,
            AdapterCapabilities.parse(capability(c, mixed=False)),
        )
        self.assertIn(
            "MIXED_BACKENDS_UNSUPPORTED", {x["code"] for x in r["reasons"]}
        )

    def test_route_mode_must_be_supported(self):
        c,_,cfg,p,best = planned()
        cap = capability(c)
        cap["route_modes"] = ["via_coordinator"]
        r = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(cap)
        )
        self.assertIn("ROUTE_MODE_UNSUPPORTED", {x["code"] for x in r["reasons"]})

    def test_adapter_limits_are_enforced(self):
        c,_,cfg,p,best = planned(distributed=True)
        cap = capability(c)
        cap["placement"]["max_compute_devices"] = 1
        r = validate_candidate_representability(
            cfg, p, best, AdapterCapabilities.parse(cap)
        )
        self.assertIn("ADAPTER_DEVICE_LIMIT", {x["code"] for x in r["reasons"]})

    def test_external_coordinator_is_explicit_capability(self):
        c,praw = fixture(1,2)
        c["nodes"].append({
            "id": "control",
            "enrollment_ref": "test:control",
            "enabled": True,
            "allowed_roles": ["coordinator"],
        })
        c["resource_pools"].append({
            "id": "control-ram",
            "node": "control",
            "kind": "system_ram",
            "reported_capacity_bytes": 1024,
        })
        c["resource_policies"].append({
            "id": "control-policy",
            "owner_node": "control",
            "pool_ref": "control-ram",
            "allocation_cap_bytes": 1024,
            "safety_headroom_bytes": 0,
        })
        c["selection"]["coordinator"] = {
            "mode": "fixed",
            "allowed_nodes": ["control"],
        }
        praw["coordinator_resident"] = {"control": {"control-ram": 1}}
        cfg = Config.parse(c)
        planning = PlanningInput.parse(praw, cfg)
        best = plan_v2(cfg, planning)["best"]
        r = validate_candidate_representability(
            cfg,
            planning,
            best,
            AdapterCapabilities.parse(capability(c, external_coord=False)),
        )
        self.assertIn(
            "EXTERNAL_COORDINATOR_UNSUPPORTED", {x["code"] for x in r["reasons"]}
        )

    def test_candidate_segment_tampering_is_invalid_not_adapter_rejection(self):
        c,_,cfg,p,best = planned()
        best = copy.deepcopy(best)
        best["segments"][0]["last_unit_exclusive"] -= 1
        with self.assertRaises(ValidationError):
            validate_candidate_representability(
                cfg, p, best, AdapterCapabilities.parse(capability(c))
            )

    def test_candidate_units_must_match_planning_input(self):
        c,_,cfg,p,best = planned()
        best = copy.deepcopy(best)
        best["unit_ids"][0] = "other"
        with self.assertRaises(ValidationError):
            validate_candidate_representability(
                cfg, p, best, AdapterCapabilities.parse(capability(c))
            )

    def test_capability_schema_rejects_unknown_route_modes(self):
        c,_,_,_,_ = planned()
        cap = capability(c)
        cap["route_modes"] = ["magic"]
        with self.assertRaises(ValidationError):
            AdapterCapabilities.parse(cap)

    def test_report_hash_is_deterministic(self):
        c,_,cfg,p,best = planned()
        cap = AdapterCapabilities.parse(capability(c))
        self.assertEqual(
            validate_candidate_representability(cfg, p, best, cap),
            validate_candidate_representability(cfg, p, best, cap),
        )
