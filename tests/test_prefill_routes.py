"""Policy routes: the orchestrator's prefill decision table (parse, selection, state freshness, end to end)."""
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from tensormeld.prefill_routes import Health, NoRoute, RouteTable, StateFile, select
from tensormeld.replica_router import NoReplicaAvailable, RouterConfig
from tensormeld.schema import ValidationError
from tensormeld.serving_policy import JointReservation

from test_replica_router import FakePrefillService, RouterCase, call, chat, router_data


def table(state_file="state.json", **changes):
    data = {
        "schema": "tensormeld/prefill-route-table-v1", "recipe": "mm-example", "recipe_sha256": "a" * 64,
        "effective_sha256": "b" * 64, "floor_gib": 8.0, "commit_floor_gib": 0.0,
        "binding": {"rtx": [{"id": "GPU-1", "host": "pc"}],
                    "pair": [{"id": "n1", "host": "n1"}, {"id": "n2", "host": "n2"}]},
        "roles": {"rtx": "service", "pair": "decode"},
        "routes": [
            {"id": "prefill_long", "place": "offload", "when": {"prompt_tokens_min": 4096, "prompt_tokens_max": None},
             "handoff_when": {"prompt_tokens_min": 0, "prompt_tokens_max": None}, "max_context_tokens": 32768,
             "fallback": "prefill_nodes", "sets": ["rtx", "pair"], "links": ["pc-nodes"],
             "hosts": ["pc", "n1", "n2"], "disk_members": [], "charges": {"pc": 53.0}, "commit_charges": {"pc": 53.0},
             "running_while": "service", "enabled": True},
            {"id": "prefill_nodes", "place": "local", "when": {"prompt_tokens_min": 0, "prompt_tokens_max": None},
             "handoff_when": {"prompt_tokens_min": 0, "prompt_tokens_max": None}, "max_context_tokens": 33024,
             "fallback": None, "sets": ["pair"], "links": [], "hosts": ["n1", "n2"], "disk_members": [],
             "charges": {"n1": 111.0, "n2": 111.0}, "commit_charges": {"n1": 111.0, "n2": 111.0},
             "running_while": "decode", "enabled": True},
        ],
        "state": {"file": state_file, "max_age_s": 10}, "offload_max_wait_s": 30, "default_generation_tokens": 1024,
    }
    data.update(changes)
    return data


def state(**changes):
    data = {"schema": "tensormeld/prefill-route-state-v1", "at": time.time(),
            "devices": {"GPU-1": {"available": True}, "n1": {"available": True}, "n2": {"available": True}},
            "links": {"pc-nodes": {"up": True}},
            "hosts": {"pc": {"free_ram_gib": 70.0}, "n1": {"free_ram_gib": 20.0}, "n2": {"free_ram_gib": 20.0}}}
    data.update(changes)
    return data


UP = Health(service_up=True, replica_up=True)


class TableParseTests(unittest.TestCase):
    def test_valid_table(self):
        t = RouteTable.parse(table())
        self.assertEqual([r.id for r in t.routes], ["prefill_long", "prefill_nodes"])
        self.assertEqual(t.default_generation_tokens, 1024)

    def test_refusals(self):
        def routes(fn):
            data = table()
            fn(data["routes"])
            return data
        bad = [
            table(schema="other"),
            routes(lambda r: r[0].update(fallback="prefill_long")),                       # self fallback
            routes(lambda r: (r[0].update(fallback="prefill_nodes"), r[1].update(fallback="prefill_long"))),  # cycle
            routes(lambda r: r[1]["when"].update(prompt_tokens_min=1)),                    # no catch-all last
            routes(lambda r: r[0].update(place="gpu")),
            routes(lambda r: r[0].update(sets=["nope"])),
            routes(lambda r: r[0]["when"].update(prompt_tokens_max=10)),                   # max < min
            routes(lambda r: r[0].update(running_while="always")),
            routes(lambda r: r[0].update(max_context_tokens=0)),
            routes(lambda r: r[0]["handoff_when"].update(prompt_tokens_max=-1)),
            routes(lambda r: r[0].update(sets=["pair"])),                                  # offload without service
            routes(lambda r: r.append(dict(r[1]))),                                        # duplicate id
            table(roles={"rtx": "service"}),
            table(state={"file": "s.json", "max_age_s": 0}),
            dict(table(), extra=1),
            routes(lambda r: r[0].update(sets=[["rtx"], "pair"])),                          # unhashable ids
            routes(lambda r: r[0].update(hosts=[["pc"]])),
            routes(lambda r: r[0].update(disk_members=[{"id": "GPU-1"}])),
            routes(lambda r: r[0].update(links=[1])),
            routes(lambda r: r[0].update(charges={"n9": 1.0})),                            # charge off the route
            routes(lambda r: r[1].update(commit_charges={"pc": 1.0})),
        ]
        for data in bad:
            with self.assertRaises(ValidationError, msg=json.dumps(data)[:200]):
                RouteTable.parse(data)


class SelectTests(unittest.TestCase):
    def setUp(self):
        self.t = RouteTable.parse(table())

    def pick(self, tokens, st=None, health=UP, failed=frozenset(), why="", context=None):
        route, trace = select(self.t, tokens, tokens + 1024 if context is None else context,
                              state() if st is None else st, why, health, failed)
        return route.id, trace

    def test_context_bound_and_handoff_when(self):
        with self.assertRaises(NoRoute) as caught:
            self.pick(8192, context=40000)
        self.assertIn("context 40000 > max_context_tokens 32768", str(caught.exception))
        self.assertEqual(self.pick(8192, context=33000)[0], "prefill_nodes")      # over the PRO's 32768, nodes' 33024
        data = table()
        data["routes"][0]["handoff_when"] = {"prompt_tokens_min": 0, "prompt_tokens_max": 6000}
        t = RouteTable.parse(data)
        route, trace = select(t, 8192, 9000, state(), "", UP)
        self.assertEqual((route.id, trace[0]["result"]), ("prefill_nodes", "handoff route: when does not hold"))

    def test_threshold_inclusive(self):
        self.assertEqual(self.pick(4095)[0], "prefill_nodes")
        self.assertEqual(self.pick(4096)[0], "prefill_long")

    def test_service_down_device_busy_link_down_take_the_fallback(self):
        cases = [
            (state(), Health(service_up=False, replica_up=True), "no prefill service up"),
            (state(devices={"GPU-1": {"available": False, "why": "PRO busy"}, "n1": {"available": True},
                            "n2": {"available": True}}), UP, "PRO busy"),
            (state(links={"pc-nodes": {"up": False, "why": "n1 unreachable"}}), UP, "n1 unreachable"),
        ]
        for st, health, why in cases:
            route, trace = self.pick(8192, st, health)
            self.assertEqual(route, "prefill_nodes")
            self.assertIn(why, trace[0]["result"])

    def test_running_routes_charge_nothing_but_keep_the_floor(self):
        nodes = {"n1": {"free_ram_gib": 20.0}, "n2": {"free_ram_gib": 20.0}}
        self.assertEqual(self.pick(8192, state(hosts=dict(nodes, pc={"free_ram_gib": 9.0})))[0], "prefill_long")
        route, trace = self.pick(8192, state(hosts=dict(nodes, pc={"free_ram_gib": 7.0})))
        self.assertEqual(route, "prefill_nodes")                                   # PC below the floor even credited
        self.assertIn("free RAM 7.0 GiB - charge 0.0 GiB < floor 8", trace[0]["result"])
        with self.assertRaises(NoRoute) as caught:
            self.pick(10, state(hosts={"pc": {"free_ram_gib": 70.0}, "n1": {"free_ram_gib": 20.0}}))
        self.assertIn("host n2: free RAM unknown", str(caught.exception))
        data = table()
        # A route whose process is not up is charged on every host it uses (here: tied to a service that is down).
        data["routes"][1]["running_while"] = "service"
        t = RouteTable.parse(data)
        down = Health(service_up=False, replica_up=True)
        for hosts, expect in (({}, "free RAM unknown"), ({"n1": {"free_ram_gib": 100.0}}, "< floor 8 GiB")):
            with self.assertRaises(NoRoute) as caught:
                select(t, 10, 1034, state(hosts=hosts), "", down)
            self.assertIn(expect, str(caught.exception))
        hosts = {"n1": {"free_ram_gib": 200.0}, "n2": {"free_ram_gib": 200.0}}
        self.assertEqual(select(t, 10, 1034, state(hosts=hosts), "", down)[0].id, "prefill_nodes")

    def test_disk_members_need_disk_available(self):
        data = table()
        data["routes"][0]["disk_members"] = ["GPU-1"]
        t = RouteTable.parse(data)
        route, trace = select(t, 8192, 9216, state(), "", UP)                      # running, but its disk is unknown
        self.assertEqual((route.id, trace[0]["result"]), ("prefill_nodes", "member GPU-1: disk unavailable"))
        ready = state(devices={"GPU-1": {"available": True, "disk_available": True}, "n1": {"available": True},
                               "n2": {"available": True}})
        self.assertEqual(select(t, 8192, 9216, ready, "", UP)[0].id, "prefill_long")

    def test_disabled_routes_and_commit_floor(self):
        data = table()
        data["routes"][0]["enabled"] = False
        route, trace = select(RouteTable.parse(data), 8192, 9216, state(), "", UP)
        self.assertEqual((route.id, trace[0]), ("prefill_nodes", {"route": "prefill_long", "result": "disabled"}))
        data = table(commit_floor_gib=4.0)
        t = RouteTable.parse(data)
        with self.assertRaises(NoRoute) as caught:
            select(t, 8192, 9216, state(), "", UP)
        self.assertIn("free commit unknown", str(caught.exception))
        hosts = {"pc": {"free_ram_gib": 70.0, "free_commit_gib": 3.0}, "n1": {"free_ram_gib": 20.0, "free_commit_gib": 90.0},
                 "n2": {"free_ram_gib": 20.0, "free_commit_gib": 90.0}}
        route, trace = select(t, 8192, 9216, state(hosts=hosts), "", UP)
        self.assertEqual(route.id, "prefill_nodes")
        self.assertIn("free commit 3.0 GiB", trace[0]["result"])
        with self.assertRaises(NoRoute) as caught:
            select(self.t, 10, 1034, state(), "", Health(service_up=True, replica_up=False))
        self.assertIn("decode replica down", str(caught.exception))

    def test_failed_handoff_resumes_at_the_fallback(self):
        route, trace = self.pick(8192, failed=frozenset({"prefill_long"}))
        self.assertEqual(route, "prefill_nodes")
        self.assertEqual(trace[0]["result"], "hand-off failed for this request")

    def test_stale_or_missing_state_fails_closed(self):
        with self.assertRaises(NoRoute) as caught:
            select(self.t, 10, 1034, None, "route state stale (30.0 s old, max 10 s)", UP)
        self.assertIn("stale", str(caught.exception))

    def test_fallback_target_must_match(self):
        data = table()
        data["routes"].insert(1, {"id": "mid", "place": "local", "when": {"prompt_tokens_min": 0,
                                                                          "prompt_tokens_max": 1000},
                                  "handoff_when": None, "max_context_tokens": 4096, "hosts": ["n1", "n2"],
                                  "disk_members": [],
                                  "commit_charges": {}, "enabled": True,
                                  "fallback": None, "sets": ["pair"], "links": [], "charges": {},
                                  "running_while": "decode"})
        data["routes"][0]["fallback"] = "mid"
        t = RouteTable.parse(data)
        route, trace = select(t, 8192, 9216, state(), "", Health(service_up=False, replica_up=True))
        self.assertEqual(route.id, "prefill_nodes")
        self.assertEqual(trace[1], {"route": "mid", "result": "no match (fallback)"})


class StateFileTests(unittest.TestCase):
    def test_fresh_stale_unreadable(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            now = [1000.0]
            sf = StateFile(path, 10, clock=lambda: now[0])
            self.assertIsNone(sf.read()[0])                      # missing
            with open(path, "w") as f:
                json.dump(state(at=995.0), f)
            self.assertIsNotNone(sf.read()[0])
            now[0] = 1006.0
            data, why = sf.read()
            self.assertIsNone(data)
            self.assertIn("stale", why)
            with open(path, "w") as f:
                f.write("{")
            self.assertIn("unreadable", sf.read()[1])

    def test_non_finite_and_malformed_state_fail_closed(self):
        # NaN must never make an old state fresh or pass a RAM floor; a bad shape is unreadable, never a crash.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            sf = StateFile(path, 10)
            good = state()
            bad = [
                json.dumps(good).replace(str(good["at"]), "NaN"),
                json.dumps(state(hosts={"pc": {"free_ram_gib": 70.0}, "n1": {"free_ram_gib": 20.0},
                                        "n2": {"free_ram_gib": 20.0}})).replace("70.0", "NaN"),
                json.dumps(good).replace(str(good["at"]), "Infinity"),
                json.dumps(state(devices=[])),
                json.dumps(state(links={"pc-nodes": ["up"]})),
                json.dumps(state(hosts={"pc": 70.0})),
                json.dumps(state(devices={"GPU-1": {"available": "yes"}})),
                json.dumps(state(hosts={"pc": {"free_ram_gib": "70"}})),
                "[" * 100000,
                json.dumps(good).replace(str(good["at"]), "1" + "0" * 400),         # isfinite overflows
                json.dumps(state(hosts={"pc": {"free_ram_gib": 10 ** 400}})),
            ]
            for text in bad:
                with open(path, "w") as f:
                    f.write(text)
                data, why = sf.read()
                self.assertIsNone(data, text[:120])
                self.assertIn("unreadable", why)


class RoutesEndToEndTests(RouterCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.state_path = os.path.join(self.dir.name, "state.json")
        self.write_state(state())

    def write_state(self, data):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.state_path)

    def router(self, sv, engines, **routes_changes):
        prefill = {"policy": "routes", "wait_s": 5,
                   "services": [{"id": "rtx", "node": "pc", "address": "127.0.0.1", "port": sv.port,
                                 "max_inflight": 1}],
                   "routes": table(self.state_path, **routes_changes)}
        router, port = self.start(engines, prefill=prefill)
        return router, port

    def service(self, **kw):
        sv = FakePrefillService(**kw)
        self.addCleanup(sv.stop)
        return sv

    def test_long_prompts_offload_short_ones_stay_local(self):
        engines = self.engines(1)
        sv = self.service()
        router, port = self.router(sv, engines)
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "route:prefill_long;offload:rtx"))
        self.assertEqual(len(engines[0].imports), 1)
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(100))})
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "route:prefill_nodes;local"))
        self.assertEqual(len(sv.calls), 1)
        routes = json.loads(call(port, "GET", "/api/status")[2])["prefill_routes"]
        self.assertEqual(routes["admitted"], {"prefill_long": 1, "prefill_nodes": 1})

    def test_admission_is_checked_again_after_the_service_wait(self):
        engines = self.engines(1)
        sv = self.service()
        router, port = self.router(sv, engines)
        original = router.server.services.start

        def start(sid, token):
            # The PC drops below its floor while this request waits for the service.
            self.write_state(state(hosts={"pc": {"free_ram_gib": 1.0}, "n1": {"free_ram_gib": 20.0},
                                          "n2": {"free_ram_gib": 20.0}}))
            return original(sid, token)
        router.server.services.start = start
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "route:prefill_nodes;local"))
        self.assertEqual(sv.calls, [])                                         # nothing was sent to the service
        routes = json.loads(call(port, "GET", "/api/status")[2])["prefill_routes"]
        self.assertEqual((routes["handoff_failed"], routes["not_admitted_after_wait"]), (0, 1))
        record = routes["recent"][0]
        self.assertEqual((record["outcome"], record["status"]), ("served", status))
        self.assertEqual(record["attempts"][0]["outcome"], "not_admitted_after_wait")
        self.assertIn("not admitted after the wait", record["attempts"][0]["why"])

    def test_service_gone_after_selection_falls_back(self):
        engines = self.engines(1)
        sv = self.service()
        router, port = self.router(sv, engines)
        services = router.server.services
        original, calls = services.up_ids, []

        def up_ids():
            # Up at selection, gone at the hand-off (the second call in the same walk).
            calls.append(1)
            return original() if len(calls) % 2 else []
        services.up_ids = up_ids
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "route:prefill_nodes;local"))
        self.assertEqual(sv.calls, [])
        routes = json.loads(call(port, "GET", "/api/status")[2])["prefill_routes"]
        self.assertEqual((routes["handoff_failed"], routes["not_admitted_after_wait"]), (0, 1))
        record = routes["recent"][0]
        self.assertEqual((record["outcome"], record["status"]), ("served", status))
        self.assertEqual([a["outcome"] for a in record["attempts"]], ["not_admitted_after_wait", "admitted"])
        self.assertEqual(record["attempts"][0]["why"], "service gone after selection")

    def test_malformed_state_is_a_503_not_a_dropped_connection(self):
        engines = self.engines(1)
        sv = self.service()
        self.write_state(state(devices=[], hosts=[]))
        router, port = self.router(sv, engines)
        status, _, body = call(port, "POST", "/v1/completions", {"prompt": list(range(10))})
        self.assertEqual(status, 503)
        self.assertIn("unreadable", json.loads(body)["error"]["message"])

    def test_failed_handoff_falls_back_to_the_nodes(self):
        engines = self.engines(1)
        sv = self.service(status=500)
        router, port = self.router(sv, engines)
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "route:prefill_nodes;local"))
        routes = json.loads(call(port, "GET", "/api/status")[2])["prefill_routes"]
        self.assertEqual((routes["handoff_failed"], routes["not_admitted_after_wait"]), (1, 0))

    def test_unavailable_pro_in_the_state_keeps_prompts_on_the_nodes(self):
        engines = self.engines(1)
        sv = self.service()
        self.write_state(state(devices={"GPU-1": {"available": False, "why": "RTX PRO busy"},
                                        "n1": {"available": True}, "n2": {"available": True}}))
        router, port = self.router(sv, engines)
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        self.assertEqual(headers["X-TensorMeld-Prefill"], "route:prefill_nodes;local")
        self.assertEqual(sv.calls, [])

    def test_stale_state_refuses_with_the_trace(self):
        engines = self.engines(1)
        sv = self.service()
        self.write_state(state(at=time.time() - 60))
        router, port = self.router(sv, engines)
        status, _, body = call(port, "POST", "/v1/completions", {"prompt": list(range(10))})
        self.assertEqual(status, 503)
        self.assertIn("stale", json.loads(body)["error"]["message"])
        self.assertEqual(engines[0].requests, [])

    def test_unknown_prompt_length_is_refused(self):
        engines = self.engines(1)
        sv = self.service()
        router, port = self.router(sv, engines)
        status, _, body = call(port, "POST", "/v1/completions", {"prompt": ["a", "b"]})
        self.assertEqual(status, 400)
        self.assertIn("one prompt per request", json.loads(body)["error"]["message"])
        status, headers, _ = chat(port, "x" * 50)                    # tokenized by the replica
        self.assertEqual((status, headers["X-TensorMeld-Prefill"]), (200, "route:prefill_nodes;local"))
        self.assertEqual(engines[0].tokenized, 1)


class RouteRecordTests(RoutesEndToEndTests):
    """prefill_routes.recent: one record per recipe-routed request (route, attempts, outcome, status, timings), its id
    returned to the client and forwarded to the engine; instance/started_at identify the counters' lifetime."""

    def recent(self, port):
        return json.loads(call(port, "GET", "/api/status")[2])["prefill_routes"]

    def test_tokenization_unavailable_records_the_client_503(self):
        engines = self.engines(1)
        router, port = self.router(self.service(), engines)
        cases = [
            (patch("tensormeld.replica_router._RouterHandler._tokenize", return_value=None), "tokenization failed"),
            (patch.object(router.server.pool, "pick_any", side_effect=NoReplicaAvailable("unavailable")),
             "no replica to tokenize (unavailable)"),
        ]
        for unavailable, why in cases:
            with self.subTest(why=why), unavailable:
                status, headers, body = call(port, "POST", "/v1/completions", {"prompt": "text"})
                routes = self.recent(port)
                record = routes["recent"][-1]
                self.assertEqual(status, 503)
                self.assertEqual(json.loads(body)["error"]["type"], "overloaded_error")
                self.assertEqual((record["outcome"], record["status"], record["why"]), ("refused", status, why))
                self.assertIsNone(record["prompt_tokens"])
                self.assertEqual(headers["X-TensorMeld-Route-Id"], record["id"])
        self.assertEqual(routes["refused"], 2)
        self.assertEqual(engines[0].requests, [])

    def test_service_wait_refusal_records_no_send(self):
        engines = self.engines(1)
        sv = self.service()
        router, port = self.router(sv, engines)
        with patch.object(router.server.services, "start", return_value=False):
            status, _, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        routes = self.recent(port)
        record = routes["recent"][0]
        self.assertEqual((status, record["status"], record["outcome"]), (200, 200, "served"))
        self.assertEqual((routes["handoff_failed"], routes["not_admitted_after_wait"]), (0, 1))
        self.assertEqual([a["outcome"] for a in record["attempts"]], ["not_admitted_after_wait", "admitted"])
        self.assertEqual(record["attempts"][0]["why"], "wait")
        self.assertEqual(sv.calls, [])

    def test_accelerator_refusal_records_no_send(self):
        engines = self.engines(1)
        sv = self.service()
        router, port = self.router(sv, engines)
        resource = JointReservation(100, enabled=True)
        lease = resource.acquire("front", 90, safe_point="between_requests")
        router.server.joint_reservation = resource
        status, _, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        routes = self.recent(port)
        record = routes["recent"][0]
        self.assertEqual((status, record["status"], record["outcome"]), (200, 200, "served"))
        self.assertEqual((routes["handoff_failed"], routes["not_admitted_after_wait"]), (0, 1))
        self.assertEqual([a["outcome"] for a in record["attempts"]], ["not_admitted_after_wait", "admitted"])
        self.assertEqual(record["attempts"][0]["why"], "accelerator_reserved")
        self.assertEqual(sv.calls, [])
        self.assertIs(resource.current, lease)

    def test_records_offload_local_and_ids(self):
        engines = self.engines(1)
        sv = self.service()
        router, port = self.router(sv, engines)
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(5000)), "max_tokens": 8})
        _, headers_short, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(100))})
        routes = self.recent(port)
        self.assertTrue(routes["instance"] and routes["started_at"] <= time.time())
        long, short = routes["recent"]
        self.assertEqual((long["route"], long["place"], long["outcome"], long["status"]),
                         ("prefill_long", "offload", "served", 200))
        self.assertEqual((long["prompt_tokens"], long["uncached_tokens"], long["context_tokens"]), (5000, 5000, 5008))
        self.assertEqual([a["outcome"] for a in long["attempts"]], ["admitted"])
        self.assertEqual((short["route"], short["place"]), ("prefill_nodes", "local"))
        self.assertEqual(headers["X-TensorMeld-Route-Id"], long["id"])
        self.assertEqual(headers_short["X-TensorMeld-Route-Id"], short["id"])
        self.assertEqual(engines[0].requests[-1]["headers"]["X-Request-Id"], short["id"])
        self.assertIsInstance(long["ttfb_ms"], float)
        self.assertGreaterEqual(long["total_ms"], long["ttfb_ms"])

    def test_fallback_is_an_attempt_list(self):
        engines = self.engines(1)
        sv = self.service(status=500)
        router, port = self.router(sv, engines)
        call(port, "POST", "/v1/completions", {"prompt": list(range(5000))})
        record = self.recent(port)["recent"][0]
        self.assertEqual([(a["route"], a["outcome"]) for a in record["attempts"]],
                         [("prefill_long", "handoff_failed"), ("prefill_nodes", "admitted")])
        self.assertEqual((record["route"], record["outcome"]), ("prefill_nodes", "served"))

    def test_refusals_are_recorded(self):
        engines = self.engines(1)
        sv = self.service()
        self.write_state(state(at=time.time() - 60))
        router, port = self.router(sv, engines)
        call(port, "POST", "/v1/completions", {"prompt": list(range(10))})
        call(port, "POST", "/v1/completions", {"prompt": ["a", "b"]})
        stale, unknown = self.recent(port)["recent"]
        self.assertEqual((stale["outcome"], stale["status"], stale["attempts"][0]["outcome"]), ("refused", 503, "refused"))
        self.assertIn("stale", stale["attempts"][0]["why"])
        self.assertEqual((unknown["outcome"], unknown["status"], unknown["prompt_tokens"]), ("refused", 400, None))

    def test_complete_non_2xx_is_an_upstream_error_with_its_status(self):
        engines = self.engines(1)
        base = engines[0].server.RequestHandlerClass

        class Failing(base):
            def do_POST(self):
                if self.path in ("/tokenize", "/admin/import-state"):
                    return base.do_POST(self)
                self.rfile.read(int(self.headers["Content-Length"]))
                self._json(500, {"error": "engine failed"})
        engines[0].server.RequestHandlerClass = Failing
        sv = self.service()
        router, port = self.router(sv, engines)
        status, _, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(100))})
        self.assertEqual(status, 500)
        record = self.recent(port)["recent"][0]
        self.assertEqual((record["outcome"], record["status"]), ("upstream_error", 500))

    def test_truncated_fixed_length_response_is_an_upstream_error(self):
        engines = self.engines(1)
        base = engines[0].server.RequestHandlerClass

        class Short(base):
            def do_POST(self):
                if self.path in ("/tokenize", "/admin/import-state"):
                    return base.do_POST(self)
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "10")
                self.end_headers()
                self.wfile.write(b"abc")
                self.wfile.flush()
                self.close_connection = True
        engines[0].server.RequestHandlerClass = Short
        router, port = self.router(self.service(), engines)
        try:
            call(port, "POST", "/v1/completions", {"prompt": list(range(100))})
        except Exception:
            pass                                     # the client sees the response end early
        record = self.recent(port)["recent"][0]
        self.assertEqual((record["outcome"], record["status"]), ("upstream_error", 200))

    def test_local_refusals_carry_the_route_id(self):
        engines = self.engines(1)
        router, port = self.router(self.service(), engines)
        status, headers, _ = call(port, "POST", "/v1/completions", {"prompt": ["a", "b"]})
        record = self.recent(port)["recent"][0]
        self.assertEqual((status, headers["X-TensorMeld-Route-Id"]), (400, record["id"]))
        self.assertIsInstance(record["total_ms"], float)
        status, headers, _ = call(port, "GET", "/api/status")
        self.assertNotIn("X-TensorMeld-Route-Id", headers)        # the record lives for one request

    def test_connect_failure_retries_another_replica(self):
        engines = self.engines(2)
        sv = self.service()
        service = dict(router_data([sv.port])["replicas"][0], id="rtx")
        prefill = {"policy": "routes", "wait_s": 5, "services": [service],
                   "routes": table(self.state_path)}
        router, port = self.start(engines, prefill=prefill,
                                  health={"interval_s": 30, "timeout_s": 1, "failures_to_down": 1})
        call(port, "POST", "/v1/completions", {"prompt": list(range(100))})
        first = self.recent(port)["recent"][0]["replica"]                  # the replica an idle pool picks
        dead, alive = (0, "r1") if first == "r0" else (1, "r0")
        engines[dead].stop()                    # still "up" for the pool until a connection fails
        status, _, _ = call(port, "POST", "/v1/completions", {"prompt": list(range(100))})
        self.assertEqual(status, 200)
        record = self.recent(port)["recent"][1]
        outcomes = [(a["replica"], a["outcome"]) for a in record["attempts"]]
        self.assertIn((first, "connect_failed"), outcomes)
        self.assertEqual(outcomes[-1], (alive, "admitted"))
        self.assertEqual(record["outcome"], "served")

class ConfigTests(unittest.TestCase):
    def test_policy_routes_needs_its_table_and_services(self):
        with self.assertRaises(ValidationError):
            RouterConfig.parse(router_data([18081], prefill={"policy": "routes"}))
        with self.assertRaises(ValidationError):
            RouterConfig.parse(router_data([18081], prefill={"policy": "routes", "routes": table()}))
        with self.assertRaises(ValidationError):
            RouterConfig.parse(router_data([18081], prefill={"policy": "nodes", "routes": table()}))
        two = [{"id": f"rtx{i}", "node": "pc", "address": "127.0.0.1", "port": 18090 + i,
                "max_inflight": 1} for i in range(2)]
        with self.assertRaisesRegex(ValidationError, "at most one prefill service"):
            RouterConfig.parse(router_data([18081], prefill={"policy": "routes", "services": two, "routes": table()}))


if __name__ == "__main__":
    unittest.main()
