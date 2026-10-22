from __future__ import annotations
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tensormeld.cli import main
from tensormeld.config_v2 import Config
from tensormeld.migration import migrate_config
from tensormeld.selection import resolve_candidates
from tensormeld.schema import ValidationError
from test_config_v2 import data


class PolicyRevisionTests(unittest.TestCase):
    def test_empty_allowlist_means_none(self):
        c=data();c['selection']['allowed_nodes']=[]
        with self.assertRaises(ValidationError):resolve_candidates(Config.parse(c))

    def test_absent_allowlist_means_eligible_registry(self):
        c=data();del c['selection']['allowed_nodes']
        self.assertEqual(len(resolve_candidates(Config.parse(c))['eligible_compute_nodes']),2)

    def test_manual_means_exact_compute_nodes(self):
        c=data();c['selection'].update(mode='manual',selected_nodes=['helper-a'])
        r=resolve_candidates(Config.parse(c))
        self.assertEqual(r['eligible_compute_nodes'],['helper-a'])
        self.assertEqual(r['required_nodes'],['helper-a'])

    def test_manual_without_selection_is_rejected(self):
        c=data();c['selection']['mode']='manual'
        with self.assertRaises(ValidationError):Config.parse(c)

    def test_auto_cannot_ignore_selected_nodes(self):
        c=data();c['selection']['selected_nodes']=['helper-a']
        with self.assertRaises(ValidationError):Config.parse(c)

    def test_required_devices_imply_node_requirement(self):
        c=data();c['selection']['required_devices']=['pc-gpu','helper-a-igpu']
        c['selection']['max_compute_nodes']=1
        with self.assertRaises(ValidationError):resolve_candidates(Config.parse(c))

    def test_disabled_cpu_offload_excludes_cpu(self):
        c=data();c['devices'].append({'id':'cpu','node':'pc','kind':'cpu','backend':'cpu','pool_ref':'pc-ram','enabled':True})
        c['selection']['allowed_device_kinds'].append('cpu')
        r=resolve_candidates(Config.parse(c))
        self.assertNotIn('cpu',[x['id'] for x in r['eligible_compute_devices']])
        c['profiles']['interactive']['placement']['cpu_offload']='allowed'
        r=resolve_candidates(Config.parse(c));self.assertIn('cpu',[x['id'] for x in r['eligible_compute_devices']])

    def test_local_gpu_required_is_any_not_all(self):
        c=data();c['devices'].append(dict(c['devices'][0],id='pc-gpu2'))
        c['selection']['participation']['local_gpu']='required';c['selection']['max_compute_devices']=1
        r=resolve_candidates(Config.parse(c))
        self.assertEqual(r['required_any_local_gpu'],['pc-gpu','pc-gpu2'])
        self.assertEqual(r['required_devices'],[])

    def test_explicit_cap_does_not_bypass_headroom(self):
        c=data();c['resource_policies'][0]['allocation_cap_bytes']=12*1024**3
        r=resolve_candidates(Config.parse(c))
        self.assertEqual(r['physical_pool_budgets']['pc-vram'],10*1024**3)

    def test_local_requirement_conflicts_with_companion_only(self):
        c=data();c['profiles']['interactive']['execution_mode']='companion_only'
        c['selection']['participation']['local_gpu']='required'
        with self.assertRaises(ValidationError):resolve_candidates(Config.parse(c))

    def test_request_cannot_make_search_unbounded(self):
        for field,value in [('candidate_limit',100001),('deadline_ms',60001)]:
            c=data();c['planning_policy']['search_budget'][field]=value
            with self.subTest(field=field),self.assertRaises(ValidationError):Config.parse(c)

    def test_output_cannot_exceed_total_context(self):
        c=data();c['profiles']['interactive']['workload']['max_output_tokens']=99999
        with self.assertRaises(ValidationError):Config.parse(c)


class MigrationTests(unittest.TestCase):
    def test_explicit_migration_preserves_source(self):
        c=data();c['config_schema']='inference-companion/v2';c['selection']['allowed_nodes']=[]
        with tempfile.TemporaryDirectory() as d:
            src,dst=Path(d)/'old.json',Path(d)/'new.json';src.write_text(json.dumps(c))
            old=src.read_bytes();r=migrate_config(src,dst)
            self.assertTrue(r['source_unchanged']);self.assertEqual(src.read_bytes(),old)
            n=json.loads(dst.read_text());self.assertEqual(n['config_schema'],'tensormeld/v2')
            self.assertNotIn('allowed_nodes',n['selection'])
            Config.parse(n)

    def test_legacy_requires_explicit_migration(self):
        c=data();c['config_schema']='inference-companion/v2'
        with self.assertRaises(ValidationError):Config.parse(c)

    def test_no_overwrite_or_inplace_migration(self):
        c=data();c['config_schema']='inference-companion/v2'
        with tempfile.TemporaryDirectory() as d:
            p,q=Path(d)/'a.json',Path(d)/'b.json';p.write_text(json.dumps(c));q.write_text('keep')
            with self.assertRaises(ValidationError):migrate_config(p,p)
            with self.assertRaises(FileExistsError):migrate_config(p,q)
            self.assertEqual(q.read_text(),'keep')

    def test_cli_output_cannot_overwrite_config(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';p.write_text(json.dumps(data()));old=p.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):code=main(['validate-config',str(p),'--out',str(p)])
            self.assertEqual(code,1);self.assertEqual(p.read_bytes(),old)

    def test_migration_preserves_old_all_local_gpu_requirement(self):
        c=data();c['config_schema']='inference-companion/v2'
        c['devices'].append(dict(c['devices'][0],id='pc-second'))
        c['selection']['participation']['local_gpu']='required'
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'old',Path(d)/'new';a.write_text(json.dumps(c));migrate_config(a,b)
            n=json.loads(b.read_text());self.assertEqual(n['selection']['required_devices'],['pc-gpu','pc-second'])


class OutputSafetyTests(unittest.TestCase):
    def test_report_cannot_overwrite_input_via_hardlink(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            source,alias=Path(d)/'config.json',Path(d)/'alias.json'
            source.write_text(json.dumps(data()));before=source.read_bytes()
            try:os.link(source,alias)
            except OSError:self.skipTest('hard links unavailable on this filesystem')
            with contextlib.redirect_stderr(io.StringIO()):
                code=main(['validate-config',str(source),'--out',str(alias)])
            self.assertEqual(code,1);self.assertEqual(source.read_bytes(),before)
