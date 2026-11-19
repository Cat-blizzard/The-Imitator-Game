"""Agent-free candidate factory using frozen expanded grasp/scene catalogs.

One seed/config produces compatible source/container choices, broad layouts,
clutter and five goal programs per scene. No optimizer, physics, VLM or paid
API is invoked here. Geometry rejection remains in the requested denominator.
Native cache scale/support pose and the original source partitions are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import SCHEMA, Rejected, compile_scene, request_index
from resimpair_pickplace.source_role_catalog_v1 import SCHEMA as ROLE_SCHEMA
from .spatial_factory_v2 import sample_request
from .task_factory_v1 import load_pool

VERSION = 'expanded_compatible_asset_sampling_v3'
MARGIN = .008


def snapshot(source, destination):
    """Read a live atomic index once; future producer updates cannot change it."""
    content = Path(source).read_bytes()
    value = json.loads(content)
    write(destination, value)
    return {'source_path_for_provenance': str(Path(source).resolve()),
            'source_bytes_sha256': hashlib.sha256(content).hexdigest(),
            'snapshot': binding(destination), 'captured_unix': time.time()}


def make_role_catalog(native_index, pose_index, containers, directory):
    native, poses, container_catalog = read(native_index), read(pose_index), read(containers)
    directory = Path(directory)
    reserved = {r['source_id'] for r in poses['entries']}
    reserved.update(read(resolve(b))['source_id'] for b in container_catalog['assets'])
    groups = [set([r['source_id'], *r.get('source_aliases', [])]) for r in native['entries']]
    for group in groups:
        if group & reserved:
            reserved.update(group)
    sources = {}; catalog = []; seen = set()

    def add(source, descriptor, bound, role):
        row = sources.setdefault(source, {'partitions': set(), 'roles': set(), 'descriptors': []})
        row['partitions'].add(descriptor['partition']); row['roles'].add(role)
        if bound not in row['descriptors']:
            row['descriptors'].append(bound)

    for entry in native['entries']:
        bound = entry['asset']; descriptor = read(resolve(bound)); source = descriptor['source_id']
        if source != entry['source_id']:
            raise ValueError('native_index_source_descriptor_mismatch')
        aliases = set([source, *entry.get('source_aliases', [])])
        for alias in aliases:
            add(alias, descriptor, bound, 'native_static_candidate')
        if source not in seen and descriptor['roles'].get('static_distractor'):
            catalog.append(bound); seen.add(source)
    for entry in poses['entries']:
        descriptor = read(resolve(entry['asset']))
        for source in [descriptor['source_id'], *entry.get('source_aliases', [])]:
            add(source, descriptor, entry['asset'], 'cached_manipulation')
    for bound in container_catalog['assets']:
        descriptor = read(resolve(bound)); add(descriptor['source_id'], descriptor, bound, 'container')
    conflicts = {s for s, r in sources.items() if len(r['partitions']) != 1}
    for group in groups:
        if group & conflicts:
            conflicts.update(group)
    catalog_path = directory / 'expanded_static_catalog.json'
    write(catalog_path, {'schema': 'resimpair_mesh_catalog_v1', 'assets': catalog,
        'source_snapshot': binding(native_index), 'scope': 'native geometry; dynamic success unmeasured'})
    for source, row in sources.items():
        row['partitions'] = sorted(row['partitions']); row['roles'] = sorted(row['roles'])
        row['allowed_as_distractor'] = source not in reserved | conflicts
        row['reserved_active_or_container'] = source in reserved
    role_path = directory / 'source_role_index.json'
    write(role_path, {'schema': ROLE_SCHEMA, 'catalog': binding(catalog_path),
        'sources': sources, 'partition_conflicts': sorted(conflicts),
        'pose_index': binding(pose_index), 'containers': binding(containers),
        'partition_labels_rewritten': False, 'inline_exclusion_count': 0})
    return binding(catalog_path), binding(role_path)


def compatibility(asset, container, mode='supported_inside'):
    if mode not in {'supported_inside', 'fully_inside'}:
        raise ValueError('unsupported_containment_mode')
    cavity = container['descriptor']['cavity']
    radial = asset['floor_radial_extent_m']
    floor_remaining = float(cavity['flat_floor_radius_m']) - radial
    wall_remaining = float(cavity['minimum_wall_radius_m']) - radial
    height = float(np.ptp(asset['vertices'][:, 2]))
    depth = float(cavity['rim_z_m'] - cavity['floor_z_m'])
    height_pass = mode != 'fully_inside' or depth - height >= .002
    return {'passed': floor_remaining >= MARGIN and wall_remaining >= MARGIN and height_pass,
            'relation': mode, 'floor_margin_m': floor_remaining, 'sampled_wall_margin_m': wall_remaining,
            'payload_height_m': height, 'cavity_depth_m': depth, 'height_pass': height_pass,
            'payload_margin_required_m': MARGIN, 'gripper_entry_release_clearance': 'planner_required',
            'scope': 'payload geometry; sampled cavity envelope; no dynamic grasp claim'}


def pack_pair(first, second, container, angle):
    """Conservative side-by-side disk packing, matching existing goal separation."""
    r0, r1 = first['radius'], second['radius']
    cavity = container['descriptor']['cavity']
    available = min(cavity['flat_floor_radius_m'], cavity['minimum_wall_radius_m']) - MARGIN
    # The existing compiler requires strictly more than radii + 15 mm.
    separation = .016
    if r0 + r1 + separation / 2 > available:
        return None
    axis = np.array([np.cos(angle), np.sin(angle)])
    return [np.round(-(r1 + separation / 2) * axis, 6).tolist(),
            np.round((r0 + separation / 2) * axis, 6).tolist()]


def partition_schedule(number, partitions, heldout_fraction, seed):
    if set(partitions) - {'development', 'heldout_asset'}:
        raise ValueError('explicit_partition_weights_required_for_new_partition_labels')
    n_heldout = round(number * heldout_fraction) if 'heldout_asset' in partitions else 0
    schedule = ['heldout_asset'] * n_heldout + ['development'] * (number - n_heldout)
    if any(part not in partitions for part in schedule):
        raise ValueError('requested_partition_has_no_compatible_assets')
    np.random.default_rng(seed).shuffle(schedule)
    return schedule


def choose_pair(pool, containers, partition, asset_use, container_use, seed, number, mode):
    assets = [a for a in pool if a['partition'] == partition]
    assets.sort(key=lambda a: (asset_use[a['source_id']], digest([seed, number, 'asset', a['source_id']])))
    candidates = [c for c in containers if c['partition'] == partition]
    candidates.sort(key=lambda c: (container_use[c['source_id']], digest([seed, number, 'container', c['source_id']])))
    for anchor in assets:
        for container in candidates:
            first_check = compatibility(anchor, container, mode)
            if not first_check['passed']:
                continue
            companions = [a for a in assets if a['source_id'] != anchor['source_id']
                          and compatibility(a, container, mode)['passed']]
            if not companions:
                continue
            # Before layout/physics, prefer a companion allowing the collect
            # program on every third scene. Otherwise retain split goals.
            if number % 3 == 0:
                companions.sort(key=lambda a: (pack_pair(anchor, a, container, 0.) is None,
                    asset_use[a['source_id']], digest([seed, number, a['source_id']])))
            companion = companions[0]
            return [anchor, companion], container, [first_check, compatibility(companion, container, mode)]
    raise Rejected('no_same_partition_compatible_asset_pair', partition=partition, containment_mode=mode)


def make_request(group, container, number, seed, catalog, role_index):
    sid = f'expanded_scene_{number:04d}_' + digest([seed, [a['source_id'] for a in group]])[:8]
    objects = [{'id': f'object_{i}', 'source_id': a['source_id'], 'xy': [-.3, -.3 + i * .6], 'yaw': 0.}
               for i, a in enumerate(group)]
    table = lambda oid, y: {'object': oid, 'target': {'kind': 'table', 'xy': [.1, y]}}
    contain = lambda oid: {'object': oid, 'target': {'kind': 'container', 'container': 'container_0', 'offset_xy_m': [0., 0.]}}
    tasks = [
        {'id': 'relocate_selected', 'placements': [table('object_0', -.3)]},
        {'id': 'orient_selected', 'placements': [{'object': 'object_0', 'target': {'kind': 'table', 'xy': [.1, -.3], 'yaw': np.pi / 2}}]},
        {'id': 'contain_selected', 'placements': [contain('object_0')]},
        {'id': 'arrange_group', 'placements': [table('object_0', -.3), table('object_1', .3)]},
    ]
    packing = pack_pair(*group, container, float(np.random.default_rng(seed).uniform(-np.pi, np.pi)))
    collect = number % 3 == 0 and packing is not None
    tasks.append({'id': 'collect_pair' if collect else 'split_goals',
                  'placements': [contain('object_0'), contain('object_1')] if collect
                  else [table('object_0', -.3), contain('object_1')]})
    spec = {'id': sid, 'objects': objects,
        'containers': [{'id': 'container_0', 'source_id': container['source_id'], 'xy': [.1, 0.], 'yaw': 0.}],
        'distractors': {'count': 3 + number % 6, 'max_dimension_m': [.10, .13, .16][number % 3],
                        'catalog': catalog, 'source_role_index': role_index}, 'tasks': tasks}
    request, trace = sample_request(spec, {a['source_id']: a['descriptor'] for a in group},
                                    {container['source_id']: container['descriptor']}, seed)
    if collect:
        for i, placement in enumerate(request['tasks'][-1]['placements']):
            placement['target']['offset_xy_m'] = packing[i]
    trace['same_container_packing'] = {'requested': collect, 'offsets_local_xy_m': packing if collect else None,
        'minimum_between_payload_footprints_m': .016, 'height_or_stacking': 'no stacking; disjoint floor footprints'}
    trace['roles'] = {'objects': [a['source_id'] for a in group], 'container': container['source_id']}
    return request, trace


def build(index, containers, static_index, output, *, scenes=100, seed=20261029,
          heldout_fraction=.2, containment_mode='supported_inside'):
    output = Path(output)
    if output.exists():
        raise ValueError('existing_factory_must_not_be_overwritten')
    if type(scenes) is not int or not 1 <= scenes <= 10000 or type(seed) is not int or seed < 0:
        raise ValueError('finite_scene_count_and_seed_required')
    if not 0 <= heldout_fraction <= 1:
        raise ValueError('invalid_heldout_fraction')
    started = time.monotonic(); output.mkdir(parents=True)
    snapshots = {
        'index': snapshot(index, output / 'snapshots/pose_index.json'),
        'containers': snapshot(containers, output / 'snapshots/containers.json'),
        'static_index': snapshot(static_index, output / 'snapshots/native_ready_index.json')}
    index, containers, static_index = (resolve(snapshots[k]['snapshot']) for k in ('index', 'containers', 'static_index'))
    catalog, role_index = make_role_catalog(static_index, index, containers, output / 'catalogs')
    pool, container_pool, rejects, container_rejects = load_pool(index, containers)
    # A same-source alias in different partitions cannot enter any role.
    roles = read(resolve(role_index))
    conflicts = set(roles['partition_conflicts'])
    pool = [a for a in pool if a['source_id'] not in conflicts]
    container_pool = [c for c in container_pool if c['source_id'] not in conflicts]
    schedule = partition_schedule(scenes, {a['partition'] for a in pool}, heldout_fraction, seed)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__), 'snapshots': snapshots,
        'index': binding(index), 'containers': binding(containers), 'catalog': catalog, 'source_role_index': role_index,
        'compiler': binding(ROOT / 'resimpair_pickplace/requested_tasks.py'),
        'role_loader': binding(ROOT / 'resimpair_pickplace/source_role_catalog_v1.py'),
        'layout_sampler': binding(ROOT / 'resimpair_tasks/spatial_factory_v2.py'),
        'scenes': scenes, 'tasks_per_scene': 5, 'seed': seed, 'partition_schedule': schedule,
        'containment_mode': containment_mode, 'selection': 'least-used source/container then seeded tie break; geometry only',
        'physical_attempts': 0, 'api_calls': 0, 'optimizer_calls': 0, 'default_promoted': False,
        'size_policy': 'fixed native cache geometry and support orientation; XY/yaw only',
        'geometry_failed_scenes_not_replaced': True, 'split_labels_rewritten': False})
    write(output / 'inventory.json', {'operable_cached_assets': len(pool), 'containers': len(container_pool),
        'static_catalog_entries': len(read(resolve(catalog))['assets']),
        'distractor_eligible_sources': sum(r['allowed_as_distractor'] and 'native_static_candidate' in r['roles'] for r in roles['sources'].values()),
        'partitions': dict(Counter(a['partition'] for a in pool)), 'rejected_assets': rejects,
        'rejected_containers': container_rejects, 'source_partition_conflicts': sorted(conflicts),
        'physical_status': 'unmeasured_for_new_assets'})
    entries = read(index)['entries']; container_entries = read(containers)['assets']
    rows = []; asset_use = Counter(); container_use = Counter(); clutter_use = Counter(); chunks = []
    for number, partition in enumerate(schedule):
        scene_seed = seed + number; folder = output / 'scenes' / f'proposal_{number:04d}'
        request = None; failure = None; accepted = False; group = []
        try:
            group, container, compatibility_checks = choose_pair(pool, container_pool, partition,
                asset_use, container_use, seed, number, containment_mode)
            for asset in group: asset_use[asset['source_id']] += 1
            container_use[container['source_id']] += 1
            request, trace = make_request(group, container, number, scene_seed, catalog, role_index)
            folder = output / 'scenes' / request['id']
            trace['containment_compatibility'] = compatibility_checks
            write(folder / 'sampling.json', trace); write(folder / 'request.json', request)
            scene, tasks, checks = compile_scene(request, entries, container_entries, scene_seed)
            write(folder / 'compiled_scene.json', scene); write(folder / 'compiled_tasks.json', {'tasks': tasks})
            write(folder / 'geometry_checks.json', checks)
            clutter_use.update(scene['randomization']['distractor_source_ids']); accepted = True
        except (Rejected, ValueError) as error:
            failure = {'reason': error.code, 'details': error.details} if isinstance(error, Rejected) else {
                'reason': type(error).__name__, 'details': {'message': str(error)}}
            write(folder / 'rejection.json', failure)
        row = {'scene_id': folder.name, 'seed': scene_seed, 'partition': partition,
            'source_ids': [a['source_id'] for a in group], 'requested_tasks': 5,
            'accepted_geometry': accepted, 'failure': failure,
            'request': binding(folder / 'request.json') if (folder / 'request.json').exists() else None,
            'physics_status': 'not_evaluated', 'human_video_status': 'not_evaluated'}
        rows.append(row)
        if row['request']:
            # Register this single scene at this exact seed to preserve clutter.
            chunk_path = output / 'requests' / f'{number:04d}.json'
            write(chunk_path, request_index({'schema': SCHEMA, 'scenes': [request]}))
            chunks.append({'requests': binding(chunk_path), 'seed': scene_seed, 'scene_id': folder.name})
        if number % 10 == 0:
            print({'scene': number + 1, 'accepted_geometry': accepted}, flush=True)
    write(output / 'scenes.json', {'scenes': rows})
    write(output / 'registration.json', {'chunks': chunks,
        'instruction': 'pass each requests binding and exact seed to existing cached_pose register; no live index reads'})
    counts = {'schema': VERSION, 'requested_scenes': scenes, 'requested_tasks': scenes * 5,
        'geometry_passed_scenes': sum(r['accepted_geometry'] for r in rows),
        'geometry_rejected_scenes': sum(not r['accepted_geometry'] for r in rows),
        'unique_operated_sources': len(asset_use), 'unique_container_sources': len(container_use),
        'unique_distractor_sources': len(clutter_use), 'partition_scenes': dict(Counter(schedule)),
        'failure_types': dict(Counter(r['failure']['reason'] for r in rows if r['failure'])),
        'seconds': time.monotonic() - started, 'physical_executions': 0, 'api_calls': 0,
        'optimizer_calls': 0, 'default_promoted': False}
    write(output / 'coverage.json', {'asset_uses': dict(asset_use), 'container_uses': dict(container_use),
        'distractor_uses': dict(clutter_use), 'split_group_rule': 'original source partition across all roles'})
    write(output / 'status.json', counts)
    return counts


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'containers', 'static-index', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scenes', type=int, default=100); p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--heldout-fraction', type=float, default=.2)
    p.add_argument('--containment-mode', choices=('supported_inside', 'fully_inside'), default='supported_inside')
    a = p.parse_args()
    print(json.dumps(build(a.index, a.containers, a.static_index, a.output, scenes=a.scenes,
        seed=a.seed, heldout_fraction=a.heldout_fraction, containment_mode=a.containment_mode), ensure_ascii=False))
