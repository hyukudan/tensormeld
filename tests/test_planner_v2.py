from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import unittest

from tensormeld.config_v2 import Config
from tensormeld.planning_contract import PlanningInput
from tensormeld.planner_v2 import plan_v2
from tensormeld.schema import ValidationError
from tensormeld.selection import resolve_candidates
import test_config_v2 as config_fixtures

G = 1024**3


def fixture(count=2, units=3):
    c = config_fixtures.ScaleContractTests()._make(count)
    c['planning_policy']['search_budget'] = {'candidate_limit': 100_000, 'deadline_ms': 60_000}
    plan = {
        'planning_schema': 'tensormeld/planning-v1', 'provenance': 'synthetic',
        'manifest_ref': c['profiles']['interactive']['model_manifest_ref'],
        'workload': {'phase': 'decode', 'context_tokens': 8192, 'max_output_tokens': 1024, 'concurrency': 1},
        'worker_resident': {f'g{i}': {f'p{i}': 1} for i in range(count)},
        'coordinator_resident': {f'n{i}': {f'p{i}': 1} for i in range(count)},
        'units': [{'id': f'block.{u}', 'output_bytes': 100,
                   'placements': {f'g{i}': {'decode_us': 100*(i+1), 'resident_bytes': {f'p{i}': 100},
                                           'workspace_bytes': {f'p{i}': 10}} for i in range(count)}} for u in range(units)],
        'links': [{'id': f'l{i}-{j}', 'source': f'device:g{i}', 'target': f'device:g{j}',
                   'payload_bytes_per_s': 1_000_000, 'latency_us': 10, 'physical_group': 'shared-wire'}
                  for i in range(count) for j in range(count) if i != j],
        'route_mode': 'direct', 'feedback_bytes': 10,
    }
    return c, plan


def run(c, p, **kwargs):
    config = Config.parse(c)
    return plan_v2(config, PlanningInput.parse(p, config), **kwargs)


class PlannerV2Tests(unittest.TestCase):
    def test_single_owner_is_a_real_alternative(self):
        c,p=fixture(); r=run(c,p)
        self.assertEqual(r['best']['owners'],['g0']*3)
        self.assertFalse(r['executable']); self.assertFalse(r['best']['qualified'])
        self.assertEqual(r['status'],'CANDIDATES_FOUND')

    def test_distributed_requires_two_devices(self):
        c,p=fixture(); c['profiles']['interactive']['execution_mode']='distributed'
        r=run(c,p); self.assertEqual(r['best']['compute_device_count'],2)

    def test_same_node_two_devices_is_distributed(self):
        c,p=fixture(); c['profiles']['interactive']['execution_mode']='distributed'
        c['devices'][1]['node']='n0'; c['resource_pools'][1]['node']='n0'
        c['resource_policies'][1]['owner_node']='n0'
        c['selection']['coordinator']['allowed_nodes']=['n0']; p['coordinator_resident']={'n0':{'p0':1}}
        r=run(c,p); self.assertEqual(r['best']['compute_device_count'],2)
        self.assertEqual(r['best']['compute_node_count'],1)

    def test_memory_forces_split(self):
        c,p=fixture()
        for policy in c['resource_policies']: policy['allocation_cap_bytes']=212
        r=run(c,p); self.assertEqual(r['best']['compute_device_count'],2)
        self.assertTrue(all(v<=212 for v in r['best']['pool_usage_bytes'].values()))

    def test_shared_pool_not_counted_twice(self):
        c,p=fixture()
        c['devices'][1].update(node='n0',pool_ref='p0')
        for u in p['units']:
            u['placements']['g1']['resident_bytes']={'p0':100}
            u['placements']['g1']['workspace_bytes']={'p0':10}
        p['worker_resident']['g1']={'p0':1}
        c['resource_policies'][0]['allocation_cap_bytes']=212
        r=run(c,p); self.assertIsNone(r['best'])
        self.assertEqual(r['status'],'NO_CANDIDATE_IN_SEARCH_SPACE')

    def test_workspace_is_peak_per_worker_not_per_unit(self):
        c,p=fixture(1); r=run(c,p)
        self.assertEqual(r['best']['pool_usage_bytes']['p0'],312)

    def test_multiple_pools_for_one_worker(self):
        c,p=fixture(1)
        c['resource_pools'].append({'id':'host','node':'n0','kind':'system_ram','reported_capacity_bytes':10})
        c['resource_policies'].append({'id':'host-policy','owner_node':'n0','pool_ref':'host','allocation_cap_bytes':10,'safety_headroom_bytes':0})
        p['worker_resident']['g0']['host']=11
        self.assertIsNone(run(c,p)['best'])

    def test_coordinator_host_memory_is_charged(self):
        c,p=fixture(1); p['coordinator_resident']['n0']['p0']=12*G
        self.assertIsNone(run(c,p)['best'])

    def test_unknown_memory_is_not_infinite(self):
        c,p=fixture(1)
        c['resource_pools'][0]['reported_capacity_bytes']=None
        c['resource_policies'][0]['allocation_cap_bytes']=None
        self.assertIsNone(run(c,p)['best'])

    def test_required_owner_and_node_limits(self):
        c,p=fixture(3)
        c['selection'].update(required_devices=['g2'],max_compute_nodes=1,max_compute_devices=1)
        self.assertEqual(run(c,p)['best']['owners'],['g2']*3)

    def test_missing_feedback_route_rejected(self):
        c,p=fixture(); c['profiles']['interactive']['execution_mode']='distributed'
        p['links']=p['links'][:1]
        r=run(c,p); self.assertIsNone(r['best'])
        self.assertGreater(r['search']['rejections']['missing_feedback_route'],0)

    def test_feedback_included_in_cost(self):
        c,p=fixture(units=2); c['profiles']['interactive']['execution_mode']='distributed'
        r=run(c,p); b=r['best']
        self.assertEqual(b['compute_us'],300)
        self.assertEqual(b['transfer_us'],130)
        self.assertEqual(b['estimated_decode_us'],430)
        self.assertEqual([x['role'] for x in b['routes']],['boundary','token_feedback'])

    def test_parallel_links_not_added(self):
        c,p=fixture(units=2); c['profiles']['interactive']['execution_mode']='distributed'
        a=run(c,p)['best']['estimated_decode_us']
        p['links'] += [dict(l,id=l['id']+'-copy') for l in p['links']]
        self.assertEqual(a,run(c,p)['best']['estimated_decode_us'])

    def test_noncontiguous_reentry_is_not_supported(self):
        c,p=fixture()
        p['units'][0]['placements'].pop('g1'); p['units'][1]['placements'].pop('g0'); p['units'][2]['placements'].pop('g1')
        self.assertIsNone(run(c,p)['best'])

    def test_budget_exhaustion_not_infeasibility(self):
        c,p=fixture(); c['planning_policy']['search_budget']['candidate_limit']=1
        r=run(c,p); self.assertEqual(r['status'],'SEARCH_INCOMPLETE')
        self.assertEqual(r['search']['work_units'],1)
        self.assertIsNone(r['best'])

    def test_budget_preserves_best_so_far(self):
        c,p=fixture(); c['planning_policy']['search_budget']['candidate_limit']=5
        r=run(c,p); self.assertEqual(r['status'],'SEARCH_INCOMPLETE')
        self.assertIsNotNone(r['best'])

    def test_deadline_is_observed(self):
        c,p=fixture(); c['planning_policy']['search_budget']['deadline_ms']=1
        times=iter([0.0,0.01,0.02,0.03])
        r=run(c,p,clock=lambda:next(times))
        self.assertEqual(r['status'],'SEARCH_INCOMPLETE'); self.assertEqual(r['search']['reason'],'deadline')

    def test_deterministic_plan_hash(self):
        c,p=fixture()
        self.assertEqual(run(c,p)['candidates'],run(c,p)['candidates'])

    def test_top_k_bounded(self):
        c,p=fixture(3)
        self.assertEqual(len(run(c,p,top_k=1)['candidates']),1)
        for invalid in (0,21,True):
            with self.assertRaises(ValidationError): run(c,p,top_k=invalid)

    def test_capacity_prefers_fewer_owners(self):
        c,p=fixture(); c['profiles']['interactive']['objective']='capacity'
        self.assertEqual(run(c,p)['best']['compute_device_count'],1)

    def test_unsupported_objective_not_silently_reinterpreted(self):
        c,p=fixture(); c['profiles']['interactive']['objective']='throughput'
        with self.assertRaises(ValidationError):run(c,p)

    def test_unsupported_partition_not_silently_reinterpreted(self):
        c,p=fixture(); c['profiles']['interactive']['placement']['strategy']='expert'
        with self.assertRaises(ValidationError):run(c,p)

    def test_profile_contract_must_match(self):
        for field in ('context_tokens','max_output_tokens'):
            c,p=fixture(); p['workload'][field] //= 2
            with self.subTest(field=field),self.assertRaises(ValidationError):run(c,p)
        c,p=fixture();p['manifest_ref']='other'
        with self.assertRaises(ValidationError):run(c,p)

    def test_no_input_can_self_qualify(self):
        c,p=fixture();p['provenance']='measured'
        with self.assertRaises(ValidationError):run(c,p)

    def test_cross_node_demand_invalid(self):
        c,p=fixture();p['units'][0]['placements']['g0']['resident_bytes']={'p1':1}
        with self.assertRaises(ValidationError):run(c,p)

    def test_unknown_links_rejected(self):
        c,p=fixture();p['links'][0]['target']='device:ghost'
        with self.assertRaises(ValidationError):run(c,p)

    def test_small_search_matches_independent_exhaustive_cost(self):
        c,p=fixture(3,4);c['profiles']['interactive']['execution_mode']='distributed'
        for u,unit in enumerate(p['units']):
            for d in range(3):unit['placements'][f'g{d}']['decode_us']=(u+1)*(d+1)*37
        costs=[]
        for owners in itertools.product(range(3),repeat=4):
            if len(set(owners))<2:continue
            segments=[owners[0]]+[owners[i] for i in range(1,4) if owners[i]!=owners[i-1]]
            if len(segments)!=len(set(segments)):continue
            cost=sum((u+1)*(d+1)*37 for u,d in enumerate(owners))
            cost+=110*(len(segments)-1)+20
            costs.append(cost)
        self.assertEqual(run(c,p)['best']['estimated_decode_us'],min(costs))

    def test_simulated_scale_with_explicit_budgets(self):
        for n in (1,2,3,4,8,16):
            c,p=fixture(n,3);c['planning_policy']['search_budget']['candidate_limit']=2000
            with self.subTest(nodes=n):
                r=run(c,p);self.assertIsNotNone(r['best'])
                self.assertLessEqual(r['search']['work_units'],2000)
                self.assertLessEqual(len(r['candidates']),3)
                self.assertFalse(r['qualified'])

    def test_coordinator_independent_of_gpu(self):
        c,p=fixture(1)
        c['nodes'].append({'id':'control','enrollment_ref':'test:control','enabled':True,'allowed_roles':['coordinator']})
        c['resource_pools'].append({'id':'control-ram','node':'control','kind':'system_ram','reported_capacity_bytes':1024})
        c['resource_policies'].append({'id':'control-policy','owner_node':'control','pool_ref':'control-ram','allocation_cap_bytes':1024,'safety_headroom_bytes':0})
        c['selection']['coordinator']={'mode':'fixed','allowed_nodes':['control']}
        p['coordinator_resident']={'control':{'control-ram':100}}
        b=run(c,p)['best'];self.assertEqual(b['coordinator_node'],'control')
        self.assertEqual(b['pool_usage_bytes']['control-ram'],100)
        self.assertEqual(b['compute_node_count'],1)

    def test_via_coordinator_uses_explicit_staging_links(self):
        c,p=fixture(units=2);c['profiles']['interactive']['execution_mode']='distributed'
        c['selection']['coordinator']={'mode':'fixed','allowed_nodes':['n0']}
        p['coordinator_resident']={'n0':{'p0':1}};p['route_mode']='via_coordinator';p['links']=[]
        for d in ('g0','g1'):
            for a,b in [(f'device:{d}','node:n0'),('node:n0',f'device:{d}')]:
                p['links'].append({'id':a+b,'source':a,'target':b,'payload_bytes_per_s':1_000_000,'latency_us':10,'physical_group':'wire'})
        r=run(c,p);self.assertEqual(r['best']['transfer_us'],260)
        self.assertTrue(all(len(x['link_ids'])==2 for x in r['best']['routes']))
