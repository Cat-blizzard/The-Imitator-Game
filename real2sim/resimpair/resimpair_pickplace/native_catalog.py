"""Separate qualified native geometry from legacy proposal-cache eligibility."""
from pathlib import Path
from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_grasp.grasps import GRIPPER


def geometry_entries(conversion, output):
    conversion, output = Path(conversion), Path(output)
    if (output / 'cache_index.json').exists(): return read(output / 'cache_index.json')['entries']
    parent = read(conversion / 'cache_index.json')
    entries = list(parent['entries']); recovered = []
    for terminal in sorted(conversion.glob('materialized/*/terminal.json')):
        row = read(terminal)
        if row.get('reason') != 'ValueError: legacy_controller_height_gate_has_no_candidate': continue
        catalog_path = terminal.parent / 'refined/catalog.json'
        catalog = read(catalog_path)
        for asset in catalog['assets']:
            d = read(resolve(asset))
            if d['qualification']['geometry'] != 'pass' or not d['roles']['rigid_manipulation_candidate']:
                raise ValueError('Cannot recover failed native geometry')
            # An empty legacy cache lets the asset participate in independent
            # learned/optimized proposal generation. It is never an executable
            # grasp and never substitutes for actual method outputs.
            signature = {'descriptor': asset, 'gripper': GRIPPER,
                         'native_geometry_evidence': binding(catalog_path)}
            path = output / d['id'] / 'empty_legacy_cache.json'
            write(path, {'schema': 'resimpair_mesh_grasp_cache_v1', 'signature': signature,
                'key': digest(signature), 'candidates': [], 'physical_qualification': 'unknown'})
            entries.append({'asset': asset, 'cache': binding(path)})
            recovered.append({'source_id': d['source_id'], 'original_failure': binding(terminal),
                              'geometry_evidence': binding(catalog_path)})
    write(output / 'cache_index.json', {'schema': 'resimpair_grasp_cache_index_v1',
        'entries': entries, 'recovered_geometry_only': recovered,
        'parent': binding(conversion / 'cache_index.json'), 'physical_executions': 0,
        'old_acceptance_changed': False, 'empty_legacy_cache_not_a_success': True})
    return entries
