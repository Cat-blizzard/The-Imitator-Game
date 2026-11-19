"""Relocatable, solver-free pose/mesh bundles with immutable fingerprints.

Only metadata is derived during import. Meshes and six-dimensional grasp poses
are copied unchanged. Provenance links are informational; runtime dependencies
are explicitly enumerated and verified, including the native robot geometry.
"""
import argparse
import json
from pathlib import Path
import shutil

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, sha, write
from resimpair_grasp.grasps import GRIPPER, load


def checked_file(root, value):
    root = Path(root).resolve()
    path = (root / value['path']).resolve()
    if not path.is_relative_to(root): raise ValueError('Bundle reference escapes its root')
    if sha(path) != value['sha256']: raise ValueError('Bundle file changed: ' + str(path))
    return path


def export_bundle(indices, containers, output):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Preserve existing pose bundle')
    output.mkdir(parents=True)
    objects = {}; rows = []; files = []
    def local(path):
        value = {'path': str(path.relative_to(output)), 'sha256': sha(path)}
        if value not in files: files.append(value)
        return value
    def copy_file(source, relative):
        target = output / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(source), target)
        return local(target)
    def metadata(value, relative):
        path = output / relative; write(path, value); return local(path)
    for method, index in indices.items():
        for entry in read(index)['entries']:
            d = read(resolve(entry['asset'])); cache = load(entry['cache'], entry['asset'])
            identity = digest({'source_id': d['source_id'], 'descriptor_sha256': entry['asset']['sha256'], 'visual': d['visual']['sha256'],
                               'collision': [c['sha256'] for c in d['collisions']]})[:20]
            if identity not in objects:
                folder = 'objects/' + identity
                visual = copy_file(d['visual'], folder + '/visual.glb')
                colliders = [copy_file(c, folder + f'/collision_{i}.obj') for i, c in enumerate(d['collisions'])]
                descriptor = metadata({**d, 'visual': visual, 'collisions': colliders,
                    'portable_parent': entry['asset']}, folder + '/descriptor.json')
                objects[identity] = {'id': identity, 'source_id': d['source_id'],
                                     'descriptor': descriptor, 'methods': {}}
            obj = objects[identity]
            value = {**cache, 'signature': {**cache['signature'], 'descriptor': obj['descriptor'],
                      'portable_parent_cache': entry['cache']}}
            value['key'] = digest(value['signature'])
            obj['methods'][method] = metadata(value, 'objects/' + identity + '/' + method + '.json')
            rows.append({'source_id': d['source_id'], 'method': method, 'poses': len(cache['candidates'])})
    container_entries = []
    for b in read(containers)['assets']:
        d = read(resolve(b)); folder = 'containers/' + d['id']
        visual = copy_file(d['visual'], folder + '/visual.glb')
        collision = copy_file(d['collision'], folder + '/collision.obj')
        container_entries.append(metadata({**d, 'visual': visual, 'collision': collision,
            'portable_parent': b}, folder + '/descriptor.json'))
    manifest = {'schema': 'resimpair_pose_bundle_v1', 'objects': list(objects.values()),
                'containers': container_entries, 'files': files, 'rows': rows,
                'gripper': GRIPPER, 'methods': sorted(indices),
                'runtime_requires_optimizer': False, 'runtime_requires_weights': False,
                'qualification': 'native geometry candidates; dynamic success is task/layout specific',
                'source_status': 'synthetic_only', 'provenance_paths_not_runtime_dependencies': True}
    write(output / 'manifest.json', manifest)
    return {'objects': len(objects), 'method_entries': len(rows), 'files': len(files)}


def inspect(root):
    root = Path(root).resolve(); manifest = read(root / 'manifest.json')
    if manifest.get('schema') != 'resimpair_pose_bundle_v1': raise ValueError('Unknown pose bundle version')
    if manifest['gripper'] != GRIPPER: raise ValueError('Bundle gripper differs from the installed native gripper')
    for geometry in GRIPPER['geometry_bindings']: resolve(geometry)
    for item in manifest['files']: checked_file(root, item)
    for obj in manifest['objects']:
        d = read(checked_file(root, obj['descriptor']))
        checked_file(root, d['visual'])
        for c in d['collisions']: checked_file(root, c)
        for method, b in obj['methods'].items():
            cache = read(checked_file(root, b))
            if cache['signature']['descriptor'] != obj['descriptor']:
                raise ValueError('Pose/asset binding mismatch')
            if cache['signature']['gripper'] != GRIPPER or cache['key'] != digest(cache['signature']):
                raise ValueError('Cache signature changed')
    for value in manifest['containers']:
        d = read(checked_file(root, value)); checked_file(root, d['visual']); checked_file(root, d['collision'])
    return manifest


def materialize(root, output):
    """Bind a bundle at any filesystem location to this checkout; no solver."""
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = inspect(root); indices = {method: [] for method in manifest['methods']}
    stamp = {'bundle': binding(root / 'manifest.json'), 'location': str(root), 'gripper': GRIPPER}
    if (output / 'import.json').exists():
        if read(output / 'import.json') != stamp: raise ValueError('Imported library location/content changed')
        return {method: output / (method + '_index.json') for method in indices}
    for obj in manifest['objects']:
        d = read(checked_file(root, obj['descriptor'])); folder = output / obj['id']
        derived = {**d, 'visual': binding(checked_file(root, d['visual'])),
            'collisions': [binding(checked_file(root, c)) for c in d['collisions']],
            'portable_bundle_descriptor': obj['descriptor']}
        write(folder / 'descriptor.json', derived); asset = binding(folder / 'descriptor.json')
        for method, b in obj['methods'].items():
            cache = read(checked_file(root, b))
            value = {**cache, 'signature': {**cache['signature'], 'descriptor': asset,
                     'portable_bundle_cache': b}}
            value['key'] = digest(value['signature'])
            path = folder / (method + '.json'); write(path, value)
            entry = {'asset': asset, 'cache': binding(path)}
            load(entry['cache'], entry['asset'])
            indices[method].append(entry)
    containers = []
    for b in manifest['containers']:
        d = read(checked_file(root, b))
        derived = {**d, 'visual': binding(checked_file(root, d['visual'])),
                   'collision': binding(checked_file(root, d['collision']))}
        path = output / 'containers' / (d['id'] + '.json'); write(path, derived); containers.append(binding(path))
    for method, entries in indices.items():
        write(output / (method + '_index.json'), {'schema': 'resimpair_grasp_cache_index_v1',
              'entries': entries, 'failures': [], 'bundle': stamp})
    write(output / 'container_catalog.json', {'schema': 'stationary_container_catalog_v1', 'assets': containers, 'rows': []})
    write(output / 'import.json', stamp)
    return {method: output / (method + '_index.json') for method in indices}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['inspect', 'import'])
    p.add_argument('--library', type=Path, required=True); p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.mode == 'inspect':
        m = inspect(a.library); print(json.dumps({'objects': len(m['objects']), 'methods': m['methods'],
            'files_verified': len(m['files']), 'requires_cudex': False}))
    else:
        if a.output is None: p.error('--output required')
        print({k: str(v) for k, v in materialize(a.library, a.output).items()})
