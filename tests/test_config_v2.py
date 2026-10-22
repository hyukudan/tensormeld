from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.cli import main
from tensormeld.config_v2 import Config, load_config
from tensormeld.schema import ValidationError
from tensormeld.selection import resolve_candidates

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "examples" / "12gb-pc-one-companion.config.json"
FOUR = ROOT / "docs" / "examples" / "four-nodes-multi-gpu.config.json"
COMPANION = ROOT / "docs" / "examples" / "companion-only.config.json"


def data(path=EXAMPLE):
    return json.loads(path.read_text())


class ConfigV2Tests(unittest.TestCase):
    def test_example_parses(self):
        c = load_config(EXAMPLE)
        self.assertEqual(c.installation.entrypoint_node, "pc")
        self.assertEqual(len(c.devices), 2)

    def test_multi_node_multi_gpu_parses(self):
        c = load_config(FOUR)
        self.assertEqual(len(c.nodes), 4)
        self.assertEqual(len(c.devices), 5)

    def test_unknown_entrypoint_rejected(self):
        d = data(); d["installation"]["entrypoint_node"] = "ghost"
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_device_pool_must_be_local(self):
        d = data(); d["devices"][0]["pool_ref"] = "helper-a-shared-ram"
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_duplicate_device_rejected(self):
        d = data(); d["devices"].append(copy.deepcopy(d["devices"][0]))
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_required_and_excluded_conflict(self):
        d = data(); d["selection"]["required_nodes"]=["pc"]; d["selection"]["excluded_nodes"]=["pc"]
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_fixed_coordinator_is_single_node(self):
        d=data(); d["selection"]["coordinator"]["mode"]="fixed"
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_every_pool_has_policy(self):
        d=data(); d["resource_policies"] = d["resource_policies"][:-1]
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_cap_cannot_exceed_reported_capacity(self):
        d=data(); d["resource_policies"][0]["allocation_cap_bytes"] = 999999999999
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_default_profile_exists(self):
        d=data(); d["installation"]["default_profile"]="missing"
        with self.assertRaises(ValidationError): Config.parse(d)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"c.json"; p.write_text('{"config_schema":"tensormeld/v2","config_schema":"bad"}')
            with self.assertRaises(ValidationError): load_config(p)


class SelectionTests(unittest.TestCase):
    def test_default_auto_returns_both_devices(self):
        r=resolve_candidates(load_config(EXAMPLE))
        self.assertEqual({d["id"] for d in r["eligible_compute_devices"]}, {"pc-gpu","helper-a-igpu"})
        self.assertFalse(r["qualified"])

    def test_companion_only_excludes_local_gpu(self):
        r=resolve_candidates(load_config(COMPANION))
        self.assertEqual([d["id"] for d in r["eligible_compute_devices"]], ["helper-a-igpu"])

    def test_local_only_uses_entrypoint(self):
        d=data(); d["profiles"]["interactive"]["execution_mode"]="local_only"
        r=resolve_candidates(Config.parse(d))
        self.assertEqual([x["id"] for x in r["eligible_compute_devices"]], ["pc-gpu"])

    def test_required_device_is_reported(self):
        d=data(); d["selection"]["required_devices"]=["helper-a-igpu"]
        r=resolve_candidates(Config.parse(d))
        self.assertEqual(r["required_devices"], ["helper-a-igpu"])

    def test_required_disabled_device_rejected(self):
        d=data(); d["selection"]["required_devices"]=["helper-a-igpu"]; d["devices"][1]["enabled"]=False
        with self.assertRaises(ValidationError): resolve_candidates(Config.parse(d))

    def test_required_node_without_device_rejected(self):
        d=data(); d["selection"]["required_nodes"]=["helper-a"]; d["devices"][1]["enabled"]=False
        with self.assertRaises(ValidationError): resolve_candidates(Config.parse(d))

    def test_pool_budget_uses_explicit_cap(self):
        r=resolve_candidates(load_config(EXAMPLE))
        gpu=next(x for x in r["eligible_compute_devices"] if x["id"]=="pc-gpu")
        self.assertEqual(gpu["pool_budget_bytes"], 9663676416)

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValidationError): resolve_candidates(load_config(EXAMPLE), "ghost")

    def test_four_node_maxima_are_preserved(self):
        r=resolve_candidates(load_config(FOUR))
        self.assertEqual(r["max_compute_nodes"], 3)
        self.assertEqual(r["max_compute_devices"], 4)
        self.assertEqual(len(r["eligible_compute_devices"]), 5)


class CLIV2Tests(unittest.TestCase):
    def test_validate_config_cli(self):
        self.assertEqual(main(["validate-config", str(EXAMPLE)]), 0)

    def test_select_cli(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/"selection.json"
            self.assertEqual(main(["select", str(EXAMPLE), "--out", str(out)]), 0)
            self.assertEqual(json.loads(out.read_text())["config_schema"], "tensormeld/v2")


if __name__ == "__main__":
    unittest.main()

class ScaleContractTests(unittest.TestCase):
    def _make(self, count: int) -> dict:
        d = data()
        d["installation"]["id"] = f"scale-{count}"
        d["nodes"] = []
        d["resource_pools"] = []
        d["devices"] = []
        d["resource_policies"] = []
        allowed = []
        for i in range(count):
            node = f"n{i}"
            dev = f"g{i}"
            pool = f"p{i}"
            allowed.append(node)
            roles = ["coordinator", "compute"]
            if i == 0:
                roles += ["front_door", "control"]
            d["nodes"].append({"id": node, "enrollment_ref": f"test:{node}", "enabled": True,
                               "allowed_roles": roles})
            d["resource_pools"].append({"id": pool, "node": node, "kind": "vram",
                                        "reported_capacity_bytes": 16 * 1024**3})
            d["devices"].append({"id": dev, "node": node, "kind": "discrete_gpu", "backend": "test",
                                 "pool_ref": pool, "enabled": True})
            d["resource_policies"].append({"id": f"rp{i}", "owner_node": node, "pool_ref": pool,
                                           "allocation_cap_bytes": 12 * 1024**3,
                                           "safety_headroom_bytes": 2 * 1024**3})
        d["installation"]["entrypoint_node"] = "n0"
        d["selection"]["allowed_nodes"] = allowed
        d["selection"]["max_compute_nodes"] = "auto"
        d["selection"]["max_compute_devices"] = "auto"
        d["selection"]["coordinator"]["allowed_nodes"] = allowed
        return d

    def test_supported_simulated_node_counts(self):
        for count in (1, 2, 3, 4, 8, 16):
            with self.subTest(count=count):
                c = Config.parse(self._make(count))
                r = resolve_candidates(c)
                self.assertEqual(len(r["eligible_compute_nodes"]), count)
                self.assertEqual(len(r["eligible_compute_devices"]), count)

    def test_parser_bound_is_explicit(self):
        d = self._make(16)
        # Expand beyond the documented parser bound without constructing a huge file by duplication.
        for i in range(16, 65):
            node=f"n{i}"; pool=f"p{i}"; dev=f"g{i}"
            d["nodes"].append({"id":node,"enrollment_ref":f"test:{node}","enabled":True,
                               "allowed_roles":["coordinator","compute"]})
            d["resource_pools"].append({"id":pool,"node":node,"kind":"vram","reported_capacity_bytes":16*1024**3})
            d["devices"].append({"id":dev,"node":node,"kind":"discrete_gpu","backend":"test","pool_ref":pool,"enabled":True})
            d["resource_policies"].append({"id":f"rp{i}","owner_node":node,"pool_ref":pool,
                                           "allocation_cap_bytes":12*1024**3,"safety_headroom_bytes":2*1024**3})
            d["selection"]["allowed_nodes"].append(node)
            d["selection"]["coordinator"]["allowed_nodes"].append(node)
        with self.assertRaises(ValidationError):
            Config.parse(d)
