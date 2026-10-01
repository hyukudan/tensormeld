from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensormeld.gguf_catalog import inspect_gguf
from tensormeld.schema import ValidationError
from tensormeld import probe


class Field:
    def __init__(self,value):self.value=value
    def contents(self):return self.value


def reader(name='blk.0.weight',split=1,index=0,total=1,offset=32):
    return SimpleNamespace(fields={
        'general.architecture':Field('test'), 'split.count':Field(split),'split.no':Field(index),
        'split.tensors.count':Field(total)}, tensors=[SimpleNamespace(
            name=name,shape=[2,2],n_bytes=16,data_offset=offset,n_elements=4,
            tensor_type=SimpleNamespace(name='F32'))])


class GGUFAdapterContractTests(unittest.TestCase):
    """Contract tests with injected readers; not upstream-parser integration evidence."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'test.gguf';self.path.write_bytes(b'GGUF'+b'\0'*60)

    def inspect(self, value=None):
        calls=[]
        def factory(path,mode):
            calls.append((path,mode));return value or reader()
        result=inspect_gguf([self.path],trusted_local_file=True,reader_factory=factory)
        self.assertEqual(calls[0][1],'r');return result

    def test_read_only_index_no_runtime_memory_claim(self):
        before=self.path.read_bytes();r=self.inspect()
        self.assertEqual(r['tensor_payload_bytes'],16);self.assertEqual(r['tensor_count'],1)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertIsNone(r['checkpoint_sha256']);self.assertFalse(r['executable'])
        self.assertTrue(r['complete_shard_set'])

    def test_explicit_trusted_local_input_required(self):
        with self.assertRaises(ValidationError):inspect_gguf([self.path],reader_factory=lambda *a,**k:reader())

    def test_partial_shards_are_identified(self):
        r=self.inspect(reader(split=2,total=2));self.assertFalse(r['complete_shard_set'])

    def test_duplicate_paths_rejected(self):
        with self.assertRaises(ValidationError):inspect_gguf([self.path,self.path],trusted_local_file=True,reader_factory=lambda *a,**k:reader())

    def test_out_of_file_tensor_range_rejected(self):
        with self.assertRaises(ValidationError):self.inspect(reader(offset=999))

    def test_complete_shard_count_must_match(self):
        with self.assertRaises(ValidationError):self.inspect(reader(total=2))

    def test_non_gguf_magic_rejected(self):
        self.path.write_bytes(b'bad!'+b'\0'*60)
        with self.assertRaises(ValidationError):self.inspect()

    def test_duplicate_cross_shard_tensors_rejected(self):
        other=self.path.with_name('other.gguf');other.write_bytes(self.path.read_bytes())
        def factory(path,mode):return reader(split=2,index=0 if Path(path).samefile(self.path) else 1,total=2)
        with self.assertRaises(ValidationError):inspect_gguf([self.path,other],trusted_local_file=True,reader_factory=factory)

    def test_consistent_complete_shards(self):
        other=self.path.with_name('other.gguf');other.write_bytes(self.path.read_bytes())
        def factory(path,mode):
            i=0 if Path(path).samefile(self.path) else 1
            return reader(name=f'blk.{i}.weight',split=2,index=i,total=2)
        r=inspect_gguf([other,self.path],trusted_local_file=True,reader_factory=factory)
        self.assertTrue(r['complete_shard_set']);self.assertEqual(r['tensor_count'],2)


    def test_complete_shards_with_noncanonical_input_path(self):
        nested = self.path.parent / 'nested'
        nested.mkdir()
        self.path = nested / '..' / self.path.name
        other = self.path.with_name('other.gguf')
        other.write_bytes(self.path.read_bytes())
        def factory(path, mode):
            index = 0 if Path(path).samefile(self.path) else 1
            return reader(name=f'blk.{index}.weight', split=2, index=index, total=2)
        result = inspect_gguf([other, self.path], trusted_local_file=True, reader_factory=factory)
        self.assertTrue(result['complete_shard_set'])
        self.assertEqual(result['tensor_count'], 2)
        self.assertEqual([part['split_index'] for part in result['files']], [0, 1])


class SystemReuseTests(unittest.TestCase):
    def test_memory_uses_psutil_when_present(self):
        fake=SimpleNamespace(__version__='7.2.2',virtual_memory=lambda:SimpleNamespace(total=100,available=40))
        with patch.dict('sys.modules',{'psutil':fake}):r=probe.memory()
        self.assertEqual(r['source'],'psutil.virtual_memory');self.assertEqual(r['available_bytes'],40)

    def test_network_unknown_speed_is_not_zero_bandwidth(self):
        fake=SimpleNamespace(__version__='7.2.2',net_if_stats=lambda:{'test':SimpleNamespace(isup=True,speed=0,duplex=0,mtu=1500)})
        with patch.dict('sys.modules',{'psutil':fake}):r=probe.interfaces()
        self.assertIsNone(r[0]['negotiated_mbps'])

    def test_optional_dependency_absence_preserves_fallback(self):
        with patch.dict('sys.modules',{'psutil':None}),patch.object(probe.platform,'system',return_value='Other'):
            self.assertEqual(probe.memory()['source'],'unavailable');self.assertEqual(probe.interfaces(),[])

    @unittest.skipUnless(importlib.util.find_spec('psutil'), 'optional psutil is not installed')
    def test_real_psutil_inventory_in_this_environment(self):
        r=probe.memory();self.assertEqual(r['source'],'psutil.virtual_memory')
        self.assertGreater(r['total_bytes'],0)
        self.assertTrue(all(x.get('source')=='psutil.net_if_stats' for x in probe.interfaces()))


@unittest.skipUnless(importlib.util.find_spec('gguf'), 'upstream gguf is unavailable: integration not executed')
class GGUFUpstreamIntegrationTests(unittest.TestCase):
    def test_upstream_writer_reader_roundtrip(self):
        import gguf
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tiny.gguf'
            writer=gguf.GGUFWriter(str(path),'llama')
            writer.add_tensor('blk.0.test.weight',np.ones((2,2),dtype=np.float32))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            result=inspect_gguf([path],trusted_local_file=True)
            self.assertEqual(result['tensor_payload_bytes'],16)
            self.assertEqual(result['architecture'],'llama')
