"""Pipeline deployment: plan layout, manifests, launch, health, route, failure, switchover.

Engines are the fake in tests/fixtures/fake_pipeline_engine.py, all on loopback; the plan is
the example 40-layer descriptor on the example lab hardware.
"""
from __future__ import annotations

import copy
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensormeld.cli import main as cli_main  # noqa: E402
from tensormeld.hetero_hardware import load_hardware  # noqa: E402
from tensormeld.hetero_model import load_model_parts  # noqa: E402
from tensormeld.hetero_planner import Request, plan_hetero  # noqa: E402
from tensormeld.pipeline_controller import (  # noqa: E402
    PipelineController, engine_nodes, launchers_for, resolve_describes)
from tensormeld.pipeline_deploy import (  # noqa: E402
    build_deployment, canonical_sha256, parse_inventory, validate_manifest)
from tensormeld.process_supervisor import (  # noqa: E402
    LaunchSpec, ManagedProcess, SshLauncher, check_env, parse_event)
from tensormeld.schema import ValidationError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
FAKE = FIXTURES / "fake_pipeline_engine.py"
MODEL = ROOT / "examples" / "hetero" / "deepseek-v4.1-flash.parts.json"
HARDWARE = ROOT / "examples" / "hetero" / "lab-hardware.json"
FAKE_SHA = hashlib.sha256(FAKE.read_bytes()).hexdigest()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def next_token(prev: int, pos: int, seed: int) -> int:
    return (prev * 1103515245 + 12345 + pos * 7 + seed) % 50021


def expected_tokens(prompts: list[tuple[str, int]], start=(0, 1)) -> list[list[int]]:
    pos, prev = start
    out = []
    for prompt, n in prompts:
        seed = sum(prompt.encode()) % 997
        toks = []
        for _ in range(n):
            prev = next_token(prev, pos, seed)
            pos += 1
            toks.append(prev)
        out.append(toks)
    return out


def post(port: int, path: str, value: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = b"" if value is None else json.dumps(value).encode()
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json",
                                                       "Content-Length": str(len(body))})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def get(port: int, path: str) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read() or b"{}")
    finally:
        conn.close()


def wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class PipelineFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model_parts(MODEL)
        cls.hw = load_hardware(HARDWARE)
        cls.plan = json.loads(json.dumps(plan_hetero(cls.model, cls.hw, Request(
            prompt_tokens=8192, context_tokens=8192, output_tokens=256, decode_rows=1, slots=1), top_k=5)))
        cls.plan.pop("_best_assigns", None)
        # The cut depends on the measured hardware numbers; the tests take whatever the planner picks.
        cls.pipeline = cls.plan["best_by_mode"]["hetero-pipeline"]["name"]
        assert cls.plan["best"]["name"] == cls.pipeline, "tests expect the pipeline to be the best plan"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tm-pipe-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def inventory(self, *, start="cold", router=True, describe=None, **changes) -> dict:
        engine = {"binary": str(FAKE), "sha256": FAKE_SHA, "model_path": "model.gguf",
                  "launch_prefix": [sys.executable], "describe": describe or {"probe": True}}
        inv = {
            "schema": "tensormeld/pipeline-inventory-v1",
            "nodes": [{"id": n, "address": "127.0.0.1", "work_dir": str(self.tmp / n), "launcher": {"kind": "local"}}
                      for n in ("pc", "node1", "node2")],
            "engines": [
                {**engine, "id": "front-eng", "backends": ["cuda", "hip", "host"], "argv_form": "pipeline-front-v1",
                 "env": {"CUDA_VISIBLE_DEVICES": "0"}},
                {**engine, "id": "tail-eng", "backends": ["hip"], "argv_form": "pipeline-tail-v1"},
                {**engine, "id": "serve-eng", "backends": ["hip"], "argv_form": "group-serve-v1"},
            ],
            "front": {"set": "rtx", "engine": "front-eng", "http_port": free_port(), "backend": "host"},
            "tail": {"set": "amd-pair", "engine": "tail-eng", "group_mode": "ep2-tp", "transport": "usb4auto",
                     "stage_port": free_port(), "ranks": ["node1-gpu", "node2-gpu"]},
            "fallback": {"set": "amd-pair", "engine": "serve-eng", "group_mode": "ep2-tp", "transport": "usb4auto",
                         "http_port": free_port(), "start": start, "ranks": ["node1-gpu", "node2-gpu"]},
            "timeouts": {"describe_s": 30, "tail_ready_s": 30, "front_ready_s": 30, "fallback_ready_s": 30,
                         "health_interval_s": 0.2, "health_timeout_s": 2, "failures_to_down": 2, "stop_s": 5,
                         "drain_s": 10, "export_s": 10, "import_s": 10},
        }
        if router:
            inv["router"] = {"listen": {"address": "127.0.0.1", "port": free_port()}, "queue_timeout_s": 60}
        for key, value in changes.items():
            if value is None:
                inv.pop(key, None)
            else:
                inv[key] = value
        return inv

    def deployment(self, inv_data: dict, candidate=None, plan=None, hw=None):
        inv = parse_inventory(inv_data)
        launchers = launchers_for(inv)
        describes = resolve_describes(inv, launchers, probe=True, engine_nodes=engine_nodes(inv, hw or self.hw))
        return build_deployment(plan or self.plan, self.model, hw or self.hw, inv, candidate=candidate or self.pipeline,
                                describes=describes, deployment_id="t1"), launchers


class PlanLayoutTest(PipelineFixture):
    def test_pipeline_candidate_has_stages_and_sets(self):
        best = self.plan["best_by_mode"]["hetero-pipeline"]
        layout = best["layout"]
        front, tail = layout["stages"]
        self.assertEqual(best["sets"]["amd-pair"], ["node1-gpu", "node2-gpu"])
        self.assertEqual(front["layers"][0], 0)
        self.assertEqual(front["layers"][1], tail["layers"][0])
        self.assertEqual(tail["layers"][1], self.model.layers)
        self.assertEqual(layout["cuts_after_layer"], [front["layers"][1] - 1])
        self.assertTrue(front["has_embedding"] and tail["has_head"])
        # Disk-backed lookups stay where their table is and are served to the FRONT.
        self.assertEqual(front["remote_parts"], tail["serves_parts"])
        every = set(front["parts"]) | set(tail["parts"]) | set(tail["serves_parts"])
        self.assertEqual(every, {p.id for p in self.model.parts if not p.optional})
        self.assertIsNone(self.plan["best_by_mode"]["engine-group"]["layout"])


class ManifestTest(PipelineFixture):
    def test_manifests_follow_the_contract(self):
        dep, _ = self.deployment(self.inventory())
        summary = dep.summary()
        self.assertEqual(summary["start_order"], ["tail-r0", "tail-r1", "front"])
        tail0, tail1 = dep.tail
        front, = dep.front
        k = front["layers"][1]
        self.assertEqual(tail0["argv"][1:], [
            "model.gguf", "--rank", "0", "--transport", "usb4auto", "--role", "pipeline-stage", "--first-layer",
            str(k), "--layers", "40", "--stage-listen", tail0["endpoints"]["stage_listen"],
            "--stage-token-file", tail0["secrets"]["token_file"]["file"]])
        self.assertEqual(tail1["argv"][3], "1")
        self.assertEqual(front["argv"][1:8], ["model.gguf", "--backend", "host", "--layers", str(k),
                                              "--stage-connect", tail0["endpoints"]["stage_listen"]])
        self.assertEqual(front["engine"]["group_mode"], None)
        self.assertEqual(tail0["engine"]["group_mode"], "ep2-tp")
        self.assertEqual(tail0["node"]["id"], "node1")
        self.assertEqual(tail1["node"]["id"], "node2")
        self.assertEqual(front["env"], {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertIn("a test substitute", " ".join(dep.warnings))
        self.assertTrue(dep.state_handoff)
        self.assertEqual([m["process"]["id"] for m in dep.fallback], ["fallback-r0", "fallback-r1"])
        self.assertEqual(dep.fallback[0]["layers"], [0, 40])
        for m in dep.manifests:
            validate_manifest(m)
            self.assertEqual(m["plan"]["candidate"], self.pipeline)
            self.assertEqual(m["engine"]["sha256"], FAKE_SHA)
            self.assertEqual(m["engine"]["describe"]["source"], "probe")
            for secret in m["secrets"].values():  # where the secret goes, never the secret
                self.assertEqual(set(secret), {"file", "bytes", "encoding", "mode", "generated"})

    def test_manifest_tampering_is_rejected(self):
        dep, _ = self.deployment(self.inventory())
        m = copy.deepcopy(dep.tail[0])
        m["argv"].append("--extra")
        with self.assertRaisesRegex(ValidationError, "manifest_sha256"):
            validate_manifest(m)
        body = {k: v for k, v in m.items() if k != "manifest_sha256"}
        m["manifest_sha256"] = canonical_sha256(body)
        with self.assertRaisesRegex(ValidationError, "argv does not match"):
            validate_manifest(m)
        m = copy.deepcopy(dep.front[0])
        m["extra"] = 1
        with self.assertRaisesRegex(ValidationError, "unknown"):
            validate_manifest(m)

    def test_binding_rejections(self):
        inv = self.inventory()
        cases = [
            ({**inv, "tail": {**inv["tail"], "set": "rtx"}}, "not the plan's tail set"),
            ({**inv, "tail": {**inv["tail"], "group_mode": "ep4"}}, "no group mode"),
            ({**inv, "tail": {**inv["tail"], "transport": "tcp"}}, "does not list transport"),
            ({**inv, "tail": {**inv["tail"], "ranks": ["node1-gpu"]}}, "do not cover"),
            ({**inv, "front": {**inv["front"], "engine": "tail-eng"}}, "not usable as front"),
            ({**inv, "front": {**inv["front"], "backend": "rocm"}}, "FRONT backend"),
            ({**inv, "front": {**inv["front"], "http_address": "10.0.0.5"}}, "loopback"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValidationError, message):
                self.deployment(data)
        nodes = [dict(n) for n in inv["nodes"]]
        nodes[0]["launcher"] = {"kind": "ssh", "target": "user@pc"}
        with self.assertRaisesRegex(ValidationError, "local launcher"):
            build_deployment(self.plan, self.model, self.hw, parse_inventory({**inv, "nodes": nodes}),
                             candidate=self.pipeline, describes={})
        with self.assertRaisesRegex(ValidationError, "not a hetero-pipeline"):
            self.deployment(inv, candidate="whole:amd-pair")
        other = self.hw.with_overrides(links={"eth-pc-c1": (1_150_000_000, 20.0)})
        with self.assertRaisesRegex(ValidationError, "another hardware"):
            self.deployment(inv, hw=other)
        bad_env = copy.deepcopy(inv)
        bad_env["engines"][0]["env"] = {"LD_PRELOAD": "x.so"}
        with self.assertRaisesRegex(ValidationError, "not allowed"):
            parse_inventory(bad_env)
        with self.assertRaisesRegex(ValidationError, "standby"):
            parse_inventory({**inv, "front": {**inv["front"], "start": "standby"}})

    def test_engine_lacking_a_capability_is_rejected(self):
        out = io.StringIO()
        with redirect_stdout(out):
            sys.argv_backup, sys.argv = sys.argv, [str(FAKE), "--describe-json"]
            try:
                import runpy
                runpy.run_path(str(FAKE), run_name="__main__")
            finally:
                sys.argv = sys.argv_backup
        desc = json.loads(out.getvalue())
        desc["capabilities"] = [c for c in desc["capabilities"] if c != "ds41.moe-routed"]
        desc["group_modes"][0]["capabilities"] = desc["capabilities"]
        path = self.tmp / "describe.json"
        path.write_text(json.dumps(desc), encoding="utf-8")
        inv = self.inventory(describe={"file": str(path)})
        with self.assertRaisesRegex(ValidationError, r"lacks capabilities \['ds41.moe-routed'\]"):
            self.deployment(inv)
        # Another binary's description never vouches for this one.
        desc["engine"]["executable_sha256"] = "0" * 64
        path.write_text(json.dumps(desc), encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "approved artifact"):
            self.deployment(inv)

    def test_cli_dry_run_prints_manifests(self):
        plan_path = self.tmp / "plan.json"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        inv_path = self.tmp / "inventory.json"
        inv_path.write_text(json.dumps(self.inventory(describe={"unchecked": "engine has no describe yet"})),
                            encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cli_main(["deploy-pipeline", str(plan_path), str(MODEL), str(HARDWARE), str(inv_path),
                             "--candidate", self.pipeline, "--dry-run"])
        self.assertEqual(code, 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(len(result["manifests"]), 5)
        self.assertEqual(result["manifests"][0]["engine"]["describe"],
                         {"source": "unchecked", "reason": "engine has no describe yet"})
        self.assertFalse(result["qualified"])


class LaunchTest(PipelineFixture):
    def start(self, inv: dict, **kwargs) -> tuple[PipelineController, dict]:
        dep, launchers = self.deployment(inv)
        controller = PipelineController(dep, launchers, **kwargs)
        self.addCleanup(controller.stop)
        controller.start()
        return controller, inv

    def test_launch_route_request_and_stop(self):
        controller, inv = self.start(self.inventory())
        self.assertEqual(controller.state, "serving")
        port = inv["router"]["listen"]["port"]
        status, body = post(port, "/v1/completions", {"prompt": "hello", "max_tokens": 5})
        self.assertEqual(status, 200)
        status2, body2 = post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}],
                                                             "max_tokens": 3})
        self.assertEqual([body["tokens"], body2["tokens"]], expected_tokens([("hello", 5), ("x", 3)]))
        status, health = get(port, "/health")
        self.assertEqual((status, health["replicas_up"]), (200, 1))
        status, st = get(port, "/api/status")
        self.assertEqual(st["deployment"]["state"], "serving")
        token_files = [Path(m["secrets"]["token_file"]["file"]) for m in (*controller.d.tail, *controller.d.front)]
        self.assertTrue(all(p.exists() for p in token_files))
        self.assertEqual(len({p.read_text() for p in token_files}), 1)
        self.assertEqual(len(token_files[0].read_text()), 64)
        final = controller.stop()
        self.assertEqual(final["final_state"], "serving")
        self.assertFalse(any(p.exists() for p in token_files))
        self.assertEqual(len(final["processes"]), 3)
        self.assertFalse(any(p.alive for p in controller.procs.values()))

    def test_wrong_token_never_connects(self):
        dep, launchers = self.deployment(self.inventory(router=None))
        controller = PipelineController(dep, launchers)
        self.addCleanup(controller.stop)
        original = controller._put_secret

        def corrupt(m, path, value):
            original(m, path, "f" * 64 if m["process"]["position"] == "front" else value)

        controller._put_secret = corrupt
        with self.assertRaisesRegex(ValidationError, "front: exited"):
            controller.start()
        self.assertEqual(controller.state, "failed")
        self.assertFalse(any(p.alive for p in controller.procs.values()))

    def test_tail_failure_fails_clearly(self):
        controller, inv = self.start(self.inventory())
        port = inv["router"]["listen"]["port"]
        self.assertEqual(post(port, "/v1/completions", {"prompt": "a", "max_tokens": 2})[0], 200)
        controller.procs["tail-r0"]._child.kill()
        self.assertTrue(wait_for(lambda: controller.state == "failed"), controller.status())
        self.assertIn("tail-r0", controller.reason)
        self.assertEqual(get(port, "/health")[0], 503)
        self.assertEqual(post(port, "/v1/completions", {"prompt": "a", "max_tokens": 2})[0], 503)
        self.assertTrue(wait_for(lambda: not any(p.alive for p in controller.procs.values())))

    def test_retire_moves_state_to_the_fallback(self):
        controller, inv = self.start(self.inventory(start="cold"))
        port = inv["router"]["listen"]["port"]
        first = post(port, "/v1/completions", {"prompt": "before", "max_tokens": 6})[1]["tokens"]
        # A request that arrives during the switchover waits in the router queue and lands on the fallback.
        late: dict = {}
        t = threading.Thread(target=lambda: late.update(r=post(port, "/v1/completions",
                                                                 {"prompt": "during", "max_tokens": 2})))
        record_holder: dict = {}
        switch = threading.Thread(target=lambda: record_holder.update(r=controller.retire("rtx needed")))
        switch.start()
        self.assertTrue(wait_for(lambda: controller.state == "switching"))
        t.start()
        switch.join(60)
        t.join(60)
        rec = record_holder["r"]
        self.assertEqual(rec["status"], "OK", rec)
        self.assertTrue(rec["state_handoff"])
        self.assertFalse(rec["state_lost"])
        self.assertEqual(controller.state, "fallback")
        self.assertEqual([s["step"] for s in rec["steps"]],
                         ["router-drain", "front-drain", "export", "stages-stopped", "fallback-ready", "import",
                          "routed"])
        after = post(port, "/v1/completions", {"prompt": "after", "max_tokens": 4})
        self.assertEqual(after[0], 200)
        self.assertEqual(late["r"][0], 200)
        expected = expected_tokens([("before", 6), ("during", 2), ("after", 4)])
        self.assertEqual([first, late["r"][1]["tokens"], after[1]["tokens"]], expected)
        self.assertFalse(controller.procs["front"].alive)
        self.assertFalse(controller.procs["tail-r0"].alive)
        status, st = get(port, "/api/status")
        self.assertEqual(st["deployment"]["switchover"]["status"], "OK")

    def test_admin_retire_route_with_standby_fallback(self):
        controller, inv = self.start(self.inventory(start="standby"))
        port = inv["router"]["listen"]["port"]
        self.assertTrue(controller.procs["fallback-r0"].alive)
        first = post(port, "/v1/completions", {"prompt": "p", "max_tokens": 3})[1]["tokens"]
        status, body = post(port, "/admin/pipeline/retire")
        self.assertEqual(status, 202)
        self.assertEqual(post(port, "/admin/pipeline/retire")[0], 409)
        self.assertTrue(wait_for(lambda: controller.state == "fallback", 30), controller.status())
        second = post(port, "/v1/completions", {"prompt": "q", "max_tokens": 3})[1]["tokens"]
        self.assertEqual([first, second], expected_tokens([("p", 3), ("q", 3)]))
        self.assertEqual(controller.switchover["steps"][-3]["start"], "standby")

    def test_front_loss_falls_back_without_state(self):
        retire_file = self.tmp / "unused"
        controller, inv = self.start(self.inventory(start="cold"), on_failure="fallback", retire_file=retire_file)
        port = inv["router"]["listen"]["port"]
        post(port, "/v1/completions", {"prompt": "lost", "max_tokens": 3})
        controller.procs["front"]._child.kill()
        self.assertTrue(wait_for(lambda: controller.state == "fallback", 30), controller.status())
        rec = controller.switchover
        self.assertTrue(rec["state_lost"])
        self.assertNotIn("export", [s["step"] for s in rec["steps"]])
        fresh = post(port, "/v1/completions", {"prompt": "new", "max_tokens": 2})[1]["tokens"]
        self.assertEqual(fresh, expected_tokens([("new", 2)])[0])

    def test_retire_file_triggers_switchover_without_router(self):
        retire_file = self.tmp / "retire-now"
        controller, inv = self.start(self.inventory(router=None, start="cold"), retire_file=retire_file)
        front_port = inv["front"]["http_port"]
        self.assertEqual(controller.url(), f"http://127.0.0.1:{front_port}")
        first = post(front_port, "/v1/completions", {"prompt": "f", "max_tokens": 2})[1]["tokens"]
        retire_file.write_text("go")
        self.assertTrue(wait_for(lambda: controller.state == "fallback", 30), controller.status())
        self.assertEqual(controller.url(), f"http://127.0.0.1:{inv['fallback']['http_port']}")
        second = post(inv["fallback"]["http_port"], "/v1/completions", {"prompt": "g", "max_tokens": 2})[1]["tokens"]
        self.assertEqual([first, second], expected_tokens([("f", 2), ("g", 2)]))

    def test_cli_real_mode_runs_and_stops(self):
        plan_path = self.tmp / "plan.json"
        plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        inv = self.inventory()
        inv_path = self.tmp / "inventory.json"
        inv_path.write_text(json.dumps(inv), encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cli_main(["deploy-pipeline", str(plan_path), str(MODEL), str(HARDWARE), str(inv_path),
                             "--candidate", self.pipeline, "--run-seconds", "0.5"])
        self.assertEqual(code, 0, out.getvalue())
        result = json.loads(out.getvalue())
        self.assertEqual(result["final_state"], "serving")
        self.assertEqual(len(result["processes"]), 3)


class SupervisorTest(unittest.TestCase):
    def test_event_lines(self):
        self.assertEqual(parse_event('READY {"role": "x"}'), ("READY", {"role": "x"}))
        self.assertEqual(parse_event('{"event": "READY", "rank": 0}')[0], "READY")
        self.assertEqual(parse_event('{"event":"POISONED","reason":"peer lost"}')[1]["reason"], "peer lost")
        self.assertEqual(parse_event("READY [1]")[0], "POISONED")
        self.assertIsNone(parse_event("connected"))
        self.assertEqual(check_env({"STRIX_X": "1"}, "env"), {"STRIX_X": "1"})
        for bad in ({"PATH": "/x"}, {"lower": "1"}, {"A": "x\ny"}):
            with self.assertRaises(ValidationError):
                check_env(bad, "env")

    @unittest.skipUnless(shutil.which("sh") and shutil.which("sha256sum"), "needs a POSIX sh and sha256sum")
    def test_ssh_wrapper_dead_man_switch_and_secrets(self):
        tmp = Path(tempfile.mkdtemp(prefix="tm-ssh-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        posix = lambda p: str(p).replace("\\", "/")  # noqa: E731
        launcher = SshLauncher("n1", "user@node1", ssh_command=(sys.executable, str(FIXTURES / "fake_ssh.py")))
        script = tmp / "engine.sh"
        script.write_text('echo "READY {\\"role\\": \\"pipeline-stage\\", \\"layers\\": [3, 40]}"\n'
                          'echo "secret=$TM_SECRET_NAME"\nwhile :; do sleep 1; done\n', encoding="utf-8")
        secret = posix(tmp / "dep" / "token")
        launcher.put_secret(secret, b"abc123")
        self.assertEqual(Path(secret).read_text(), "abc123")
        with self.assertRaises(ValidationError):  # set -C: never overwrite
            launcher.put_secret(secret, b"other")
        self.assertEqual(launcher.sha256(posix(script)), hashlib.sha256(script.read_bytes()).hexdigest())
        proc = ManagedProcess(LaunchSpec("tail-r0", "n1", (posix(script),), {"TM_SECRET_NAME": "file"},
                                         posix(tmp / "dep"), ("sh",)), launcher)
        proc.start()
        ready = proc.wait_ready(20)
        self.assertEqual(ready["layers"], [3, 40])
        self.assertIsNotNone(proc.remote_pid)
        self.assertTrue(wait_for(lambda: "secret=file" in proc.snapshot()["stdout_tail"], 5))
        result = proc.stop(10)
        self.assertFalse(result["killed"])
        self.assertFalse(proc.alive)
        self.assertIsNotNone(proc.remote_exit)
        launcher.remove(secret)
        self.assertFalse(Path(secret).exists())

    def test_ssh_wrapper_keeps_the_program_while_the_channel_is_open(self):
        # The dead-man switch must wait for the channel's stdin, not fire on the /dev/null a
        # non-interactive sh gives a background job by default.
        tmp = Path(tempfile.mkdtemp(prefix="tm-ssh-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        posix = lambda p: str(p).replace("\\", "/")  # noqa: E731
        launcher = SshLauncher("n1", "user@node1", ssh_command=(sys.executable, str(FIXTURES / "fake_ssh.py")))
        script = tmp / "engine.sh"
        script.write_text('sleep 1\necho "READY {\\"role\\": \\"pipeline-stage\\", \\"layers\\": [0, 1]}"\n'
                          'while :; do sleep 1; done\n', encoding="utf-8")
        proc = ManagedProcess(LaunchSpec("tail-r1", "n1", (posix(script),), {}, posix(tmp / "dep"), ("sh",)),
                              launcher)
        proc.start()
        self.assertEqual(proc.wait_ready(20)["layers"], [0, 1])
        time.sleep(1)
        self.assertTrue(proc.alive)
        result = proc.stop(10)
        self.assertFalse(result["killed"])
        self.assertFalse(proc.alive)


if __name__ == "__main__":
    unittest.main()
