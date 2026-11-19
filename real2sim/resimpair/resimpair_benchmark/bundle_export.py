"""Publish generated data independently of optional optimizer repositories."""
import argparse
from pathlib import Path
import shutil

from resimpair_grasp.common import ROOT, binding, read, resolve, write, sha
from resimpair_pickplace.pose_bundle import export_bundle
from .campaign import BASE, METHODS, proposal_folder


def native_bundle(output):
    entries = read(BASE / 'native_geometry_index/cache_index.json')['entries']
    sources = {read(resolve(e['asset']))['source_id']: e for e in entries}
    indices = {}; container_bindings = {}
    for method in METHODS:
        method_entries = []
        for entry in read(BASE / 'selection.json')['entries']:
            if not entry['physical_subset']: continue
            original = entry.get('native_entry', sources.get(entry['source_id']))
            q = BASE / 'qualified_native' / method / entry['id'] / 'decisions.json'
            if original is not None and q.exists():
                method_entries.append({'asset': original['asset'], 'cache': read(q)['cache']})
        index = BASE / 'export_indices' / (method + '.json')
        write(index, {'entries': method_entries, 'failures': [], 'scope': 'all native candidates including empty caches'})
        indices[method] = index
    for p in BASE.glob('physics_*/scenes/*.json'):
        scene = read(p)
        for support in scene.get('supports', []):
            if 'container_descriptor' in support:
                b = support['container_descriptor']; container_bindings[b['sha256']] = b
    cat = BASE / 'export_indices/containers.json'; write(cat, {'assets': list(container_bindings.values())})
    return export_bundle(indices, cat, output)


def offline_archive(output):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing archive')
    rows = []
    novelty = {r['source_id']: r for r in read(BASE / 'source_novelty_audit.json')['rows']}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(source), target)
        return {'path': str(target.relative_to(output)), 'sha256': sha(target)}
    for entry in read(BASE / 'selection.json')['entries']:
        folder = output / entry['id']; d = read(resolve(entry['asset']))
        visual = copy(d['visual'], folder / 'visual.glb')
        provenance = novelty[entry['source_id']]
        write(folder / 'asset.json', {**d, 'visual': visual, 'original_asset_binding': entry['asset'],
            'partition': provenance['effective_partition'], 'original_partition': d['partition'],
            'source_novelty': provenance['novelty'],
            'scope': 'offline visual mesh/support/pose; native collision model not included here'})
        row = {'id': entry['id'], 'source_id': entry['source_id'], 'split': provenance['effective_partition'],
            'frozen_experiment_split': entry['split'], 'source_novelty': provenance['novelty'],
            'asset': {'path': str((folder / 'asset.json').relative_to(output)), 'sha256': sha(folder / 'asset.json')},
            'methods': {}}
        for method in METHODS:
            src = proposal_folder(method, entry['id'])
            if not (src / 'result.json').exists():
                row['methods'][method] = {'generation_completed': False}; continue
            result = read(src / 'result.json')
            raw = copy(result['raw'], folder / (method + '_raw.npz'))
            q = read(BASE / 'qualified_visual' / method / entry['id'] / 'decisions.json')
            cache = copy(q['cache'], folder / (method + '_visual_candidates.json'))
            row['methods'][method] = {'raw': raw, 'visual_candidates': cache,
                                     'retained': q['retained'], 'physical_success': 'unmeasured_by_archive'}
        rows.append(row)
    write(output / 'manifest.json', {'schema': 'precomputed_grasp_research_archive_v1',
        'rows': rows, 'objects': len(rows), 'requires_optimizer_to_read': False,
        'native_trajectory_bundle': 'assets/precomputed_grasps_v1',
        'this_archive_is_not_a_native_dynamic_asset_catalog': True})
    return {'sources': len(rows)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=['native', 'offline'])
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    print(native_bundle(a.output) if a.kind == 'native' else offline_archive(a.output))
