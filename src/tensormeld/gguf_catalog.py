"""Optional GGUF directory inspection using upstream gguf, never a replacement parser.

Only explicitly trusted, local files are accepted. The upstream reader maps the file;
this is not an adversarial-file sandbox. We do not read tensor values or hash all weights.
The output is an index, NOT a complete memory/execution manifest.
"""
from __future__ import annotations

import gc
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
from typing import Callable, Any

from .schema import ValidationError

GGUF_VERSION_PIN = '0.19.0'
MAX_SHARDS = 64
MAX_TENSORS = 100_000


def _upstream_reader() -> Callable:
    try:
        installed = version('gguf')
    except PackageNotFoundError as exc:
        raise ValidationError('GGUF support is optional; install TensorMeld with the gguf extra: pip install ".[gguf]"') from exc
    if installed != GGUF_VERSION_PIN:
        raise ValidationError(f'GGUF adapter requires gguf=={GGUF_VERSION_PIN}; found {installed}')
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise ValidationError('The optional gguf runtime or a dependency is missing; reinstall the gguf extra') from exc
    return GGUFReader


def _extract(reader: Any, size: int) -> dict:
    def field(name: str, default=None):
        value = reader.fields.get(name)
        return default if value is None else value.contents()
    architecture = field('general.architecture')
    if architecture is not None and (not isinstance(architecture, str) or not 1 <= len(architecture) <= 128):
        raise ValidationError('invalid architecture metadata')
    if len(reader.tensors) > MAX_TENSORS:
        raise ValidationError('GGUF tensor count exceeds inspector limit')
    tensors = []
    names = set()
    for tensor in reader.tensors:
        name = tensor.name
        if not isinstance(name, str) or not 1 <= len(name) <= 512 or name in names:
            raise ValidationError('invalid or duplicate GGUF tensor name')
        names.add(name)
        shape = [int(x) for x in tensor.shape]
        nbytes, offset, elements = int(tensor.n_bytes), int(tensor.data_offset), int(tensor.n_elements)
        if not 1 <= len(shape) <= 8 or any(x <= 0 for x in shape) or nbytes < 0 or elements < 1:
            raise ValidationError('invalid tensor dimensions or size')
        if offset < 0 or offset + nbytes > size:
            raise ValidationError('tensor range is outside the local file')
        tensors.append({'name': name, 'gguf_shape': shape, 'encoding': tensor.tensor_type.name,
                        'n_elements': elements, 'n_bytes': nbytes, 'data_offset': offset})
    count, part, total = field('split.count', 1), field('split.no', 0), field('split.tensors.count', len(tensors))
    if any(type(x) is not int for x in (count, part, total)) or not 1 <= count <= MAX_SHARDS or not 0 <= part < count or total < len(tensors):
        raise ValidationError('invalid split metadata')
    return {'architecture': architecture, 'split_count': count, 'split_index': part,
            'declared_total_tensors': total, 'tensors': tensors}


def inspect_gguf(paths: list[Path], *, trusted_local_file: bool = False,
                 reader_factory: Callable | None = None) -> dict:
    if not trusted_local_file:
        raise ValidationError('GGUF parsing requires --trusted-local-file; remote/untrusted files are not sandboxed')
    if not 1 <= len(paths) <= MAX_SHARDS:
        raise ValidationError(f'supply 1..{MAX_SHARDS} explicit shard paths')
    paths = [p.resolve(strict=True) for p in paths]
    if len(set(paths)) != len(paths) or any(not p.is_file() for p in paths):
        raise ValidationError('GGUF paths must be unique regular files')
    factory = reader_factory or _upstream_reader()
    records = []
    seen_names: set[str] = set()
    for path in paths:
        before = path.stat()
        if before.st_size < 24:
            raise ValidationError('file is too small for a GGUF header')
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValidationError('file has no GGUF magic')
        try:
            reader = factory(str(path), mode='r')
            try:
                info = _extract(reader, before.st_size)
            finally:
                del reader
                gc.collect()
        except ValidationError:
            raise
        except (ValueError, IndexError, KeyError, OverflowError, TypeError) as exc:
            raise ValidationError(f'GGUF inspection failed: {exc}') from exc
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValidationError('file changed during inspection')
        names = {t['name'] for t in info['tensors']}
        if names & seen_names:
            raise ValidationError('duplicate tensor names across shards')
        seen_names |= names
        if len(seen_names) > MAX_TENSORS:
            raise ValidationError('total tensor count exceeds inspector limit')
        records.append({'file_name': path.name, 'file_size_bytes': before.st_size, **info})
    first = records[0]
    for rec in records:
        if any(rec[k] != first[k] for k in ('architecture', 'split_count', 'declared_total_tensors')):
            raise ValidationError('inconsistent shard metadata')
    indexes = [r['split_index'] for r in records]
    if len(set(indexes)) != len(indexes):
        raise ValidationError('duplicate split index')
    complete = len(records) == first['split_count']
    if complete and len(seen_names) != first['declared_total_tensors']:
        raise ValidationError('complete shard set has an inconsistent total tensor count')
    result = {'index_schema': 'tensormeld/gguf-index-v1', 'reader': f'gguf=={GGUF_VERSION_PIN}',
              'architecture': first['architecture'], 'complete_shard_set': complete,
              'tensor_count': len(seen_names), 'files': sorted(records, key=lambda r: r['split_index']),
              'tensor_payload_bytes': sum(t['n_bytes'] for r in records for t in r['tensors']),
              'checkpoint_sha256': None, 'qualified': False, 'executable': False,
              'warnings': [
                  'Directory inspection only: payload bytes are NOT runtime RAM/VRAM requirements.',
                  'No tensor values were inspected, weights executed, or full checkpoint hashes computed.',
                  'Matching split metadata is not proof of matching checkpoint provenance.',
                  'Index hash covers metadata only, not weights; state/workspace/aliases require an adapter.',
                  'Explicit trusted local files only; upstream parsing is not sandboxed.',
              ]}
    result['index_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result
