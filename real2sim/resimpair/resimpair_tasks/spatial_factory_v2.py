"""Opt-in broad layouts for the retained multi-object pick/place factory.

Consumes a frozen factory's sources, roles and goal programs. Samples positions
before IK or physics, writes standard requests, and uses the unchanged compiler.
Every original scene has one bounded proposal; a failed scene is not replaced.
The same local cuDex caches are reused without changing scale or support pose.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, read, resolve, write
from resimpair_pickplace.requested_tasks import (
    SCHEMA, Rejected, TABLE_XY, check_separation, compile_scene, request_index, table_fit,
)

VERSION = 'broad_multiobject_spatial_factory_v2'
# Same support workspace as the existing single-object scenes.sample entry.
SOURCE_XY = np.array([[-.34, .16], [-.54, .54]])
DESTINATION_XY = np.array([[-.30, .18], [-.54, .54]])
MAX_POINT_PROPOSALS = 100


def sample_request(spec, assets, containers, seed):
    """Return a standard request plus every geometric rejection, never run IK.

    Input descriptors preserve real metric radii; different task programs in one
    scene share one initial layout and a consistent goal per object. Container
    offsets stay inside the measured cavity with the compiler's 8 mm margin.
    """
    rng = np.random.default_rng(seed)
    result = copy.deepcopy(spec)
    result['id'] = spec['id'] + '_spatialv2'
    occupied, draws = [], []

    def point(identity, radius, workspace):
        bounds = np.array(workspace, dtype=float)
        bounds[:, 0] = np.maximum(bounds[:, 0], TABLE_XY[:, 0] + radius)
        bounds[:, 1] = np.minimum(bounds[:, 1], TABLE_XY[:, 1] - radius)
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise Rejected('asset_does_not_fit_sampling_workspace', object=identity, radius_m=radius)
        for attempt in range(MAX_POINT_PROPOSALS):
            xy = np.round(rng.uniform(bounds[:, 0], bounds[:, 1]), 6)
            row = {'id': identity, 'attempt': attempt, 'xy': xy.tolist()}
            try:
                table_fit(xy, radius)
                check_separation(xy, radius, occupied, margin=.035)
            except Rejected as error:
                draws.append({**row, 'accepted': False, 'reason': error.code, 'details': error.details})
                continue
            draws.append({**row, 'accepted': True})
            occupied.append((identity, xy, radius))
            return xy.tolist()
        raise Rejected('bounded_spatial_sampling_exhausted', object=identity, attempts=draws,
                       partial_request=result, maximum_point_proposals=MAX_POINT_PROPOSALS)

    radii = {o['id']: float(np.linalg.norm(assets[o['source_id']]['half_size'][:2]))
             for o in result['objects']}
    # Place larger footprints first; all positions are continuous random draws.
    items = [(radii[o['id']], o, SOURCE_XY) for o in result['objects']]
    items.extend((float(np.linalg.norm(np.asarray(containers[c['source_id']]['dimensions_m'])[:2] / 2)),
                  c, DESTINATION_XY) for c in result.get('containers', []))
    for radius, obj, bounds in sorted(items, key=lambda row: (-row[0], row[1]['id'])):
        obj['xy'] = point(obj['id'], radius, bounds)
        obj['yaw'] = round(float(rng.uniform(-math.pi, math.pi)), 6)

    old_objects = {o['id']: o for o in spec['objects']}
    objects = {o['id']: o for o in result['objects']}
    container_by_id = {c['id']: containers[c['source_id']] for c in result.get('containers', [])}
    # Keep intentionally shared goals shared across the five programs.
    table_goals = {}
    table_ids = {p['object'] for t in result['tasks'] for p in t['placements'] if p['target']['kind'] == 'table'}
    for oid in sorted(table_ids, key=lambda key: (-radii[key], key)):
        table_goals[oid] = point('goal_' + oid, radii[oid], DESTINATION_XY)
    offsets = {}
    for task in result['tasks']:
        for placement in task['placements']:
            oid, target = placement['object'], placement['target']
            if 'yaw' in target:
                delta = target['yaw'] - old_objects[oid].get('yaw', 0.)
                target['yaw'] = round(float((objects[oid]['yaw'] + delta + math.pi) % (2 * math.pi) - math.pi), 6)
            if target['kind'] == 'table':
                target['xy'] = table_goals[oid]
            else:
                key = (oid, target['container'])
                if key not in offsets:
                    # Bounding disk is conservative for every object yaw. Do
                    # not scale assets to fit and do not alter the cavity rule.
                    slack = container_by_id[target['container']]['cavity']['flat_floor_radius_m'] - radii[oid] - .008
                    extent = min(.03, max(0., slack - 1e-5))
                    distance = math.sqrt(float(rng.uniform())) * extent
                    angle = float(rng.uniform(-math.pi, math.pi))
                    offsets[key] = np.round([distance * math.cos(angle), distance * math.sin(angle)], 6).tolist()
                target['offset_xy_m'] = offsets[key]
    request_index({'schema': SCHEMA, 'scenes': [result]})
    return result, {'schema': VERSION, 'seed': seed, 'point_draws': draws,
                    'source_xy_bounds_m': SOURCE_XY.tolist(), 'destination_xy_bounds_m': DESTINATION_XY.tolist(),
                    'yaw_rad': [-math.pi, math.pi], 'scale_randomized': False,
                    'support_orientation_randomized': False, 'outcome_based_resampling': False,
                    'maximum_proposals_per_point': MAX_POINT_PROPOSALS}


def build(source_factory, output, seed):
    source_factory, output = Path(source_factory), Path(output)
    if output.exists():
        raise ValueError('existing_spatial_campaign_must_not_be_overwritten')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed_must_be_nonnegative_integer')
    parent = read(source_factory / 'protocol.json')
    index, container_catalog = resolve(parent['index']), resolve(parent['containers'])
    entries, container_entries = read(index)['entries'], read(container_catalog)['assets']
    assets = {read(resolve(e['asset']))['source_id']: read(resolve(e['asset'])) for e in entries}
    containers = {read(resolve(b))['source_id']: read(resolve(b)) for b in container_entries}
    records = read(source_factory / 'scenes.json')['scenes']
    output.mkdir(parents=True)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'parent_factory': binding(source_factory / 'protocol.json'), 'parent_scenes': binding(source_factory / 'scenes.json'),
        'index': binding(index), 'containers': binding(container_catalog), 'seed': seed,
        'compiler': binding(Path(__file__).parents[1] / 'resimpair_pickplace/requested_tasks.py'),
        'source_xy_bounds_m': SOURCE_XY.tolist(), 'destination_xy_bounds_m': DESTINATION_XY.tolist(),
        'maximum_proposals_per_point': MAX_POINT_PROPOSALS, 'scene_replacement_after_failure': False,
        'frozen_roles_assets_sizes_and_goal_programs': True, 'physics_attempts': 0, 'api_calls': 0,
        'new_contact_mechanisms': 0, 'status': 'candidate_not_default'})
    rows, chunks, chunk = [], [], []
    started = time.monotonic()
    for i, record in enumerate(records):
        original = read(resolve(record['request']))
        directory = output / 'scenes' / (original['id'] + '_spatialv2')
        request = None; accepted = False; failure = None
        try:
            request, trace = sample_request(original, assets, containers, seed + i)
            write(directory / 'sampling.json', trace)
            write(directory / 'request.json', request)
            scene, tasks, checks = compile_scene(request, entries, container_entries, seed + i)
            write(directory / 'compiled_scene.json', scene)
            write(directory / 'compiled_tasks.json', {'tasks': tasks})
            write(directory / 'geometry_checks.json', checks)
            accepted = True
        except Rejected as error:
            failure = {'reason': error.code, 'details': error.details}
            write(directory / 'rejection.json', failure)
        row = {'scene_id': directory.name, 'parent_request': record['request'], 'seed': seed + i,
               'partition': record['partition'], 'source_ids': [o['source_id'] for o in original['objects']],
               'requested_tasks': len(original['tasks']), 'accepted_geometry': accepted,
               'request': binding(directory / 'request.json') if request else None, 'failure': failure,
               'physics_status': 'not_evaluated', 'human_video_status': 'not_evaluated'}
        rows.append(row)
        # Keep seed alignment explicit even when geometry rejects a scene.
        if request is not None:
            chunk.append({'request': binding(directory / 'request.json'), 'seed': seed + i})
        if len(chunk) == 8 or i == len(records) - 1:
            chunks.append(chunk); chunk = []
    write(output / 'scenes.json', {'scenes': rows})
    write(output / 'registration.json', {'chunks': chunks,
        'instruction': 'register each scene with its explicit seed; skipped geometry failures do not shift later seeds'})
    status = {'schema': VERSION, 'requested_scenes': len(records), 'requested_tasks': sum(r['requested_tasks'] for r in rows),
              'geometry_passed_scenes': sum(r['accepted_geometry'] for r in rows),
              'geometry_rejected_scenes': sum(not r['accepted_geometry'] for r in rows),
              'seconds': time.monotonic() - started, 'physical_executions': 0, 'api_calls': 0,
              'optimizer_calls': 0, 'default_promoted': False, 'failures_retained': True}
    write(output / 'status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-factory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    print(build(args.source_factory, args.output, args.seed))
