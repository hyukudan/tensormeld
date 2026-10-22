"""Explicit, non-overwriting migration from the previous working title."""
from __future__ import annotations
import json
from pathlib import Path
from .config_v2 import Config
from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates


def migrate_config(source: Path, destination: Path) -> dict:
    if source.resolve() == destination.resolve():
        raise ValidationError('migration requires a different destination; source is never overwritten')
    with source.open('rb') as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError('config exceeds 2 MiB')
    data = json.loads(raw, object_pairs_hook=_no_duplicates)
    if not isinstance(data, dict) or data.get('config_schema') != 'inference-companion/v2':
        raise ValidationError('expected an inference-companion/v2 installation config')
    data['config_schema'] = 'tensormeld/v2'
    selection = data.get('selection', {})
    if not isinstance(selection, dict):
        raise ValidationError('selection must be an object')
    if selection.get('mode') == 'manual' and not selection.get('selected_nodes'):
        raise ValidationError('legacy manual selection is ambiguous; define selected_nodes before migrating')
    if selection.get('allowed_nodes') == []:
        selection.pop('allowed_nodes')
    if selection.get('participation', {}).get('local_gpu') == 'required':
        local = data.get('installation', {}).get('entrypoint_node')
        ids = {d['id'] for d in data.get('devices', []) if d.get('node') == local and d.get('kind') != 'cpu'}
        selection['required_devices'] = sorted(set(selection.get('required_devices', [])) | ids)
    config = Config.parse(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return {'migrated': True, 'config_schema': 'tensormeld/v2', 'config_sha256': config.fingerprint,
            'source_unchanged': True, 'destination': str(destination)}
