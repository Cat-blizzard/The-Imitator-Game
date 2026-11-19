"""Bounded layout adapter connecting source-first tasks to the retained arm selector.

Sources, exact size/cache variants, receiving assets, local goal slots and task
programs are fixed. Only initial XY/yaw and container XY change. The actual controller workspace
is read before sampling and applied to all four required endpoint poses. This is a CPU
proposal/IK stage, never a physical success label or a replacement for native
whole-path validation. Failed proposals remain in the output directory.
"""
import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected, table_fit, check_separation
from resimpair_pickplace.cartesian_reachability import pose
from resimpair_pickplace.above_rim_release_v1 import ReleaseSelector, release_geometry
from . import metric_task_factory_v3 as factory

VERSION = 'source_first_reachable_layout_v3'


@lru_cache(maxsize=1)
def runtime_workspace():
    # Read the frozen controller settings, not table size or arm reach radius.
    from resimpair_grasp import motion
    tree = ast.parse(Path(motion.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MeshMotion')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'check')
    values = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Dict):
            continue
        selected = {k.value: ast.literal_eval(v) for k, v in zip(node.keys, node.values)
                    if isinstance(k, ast.Constant) and k.value in {'workspace_min', 'workspace_max'}}
        if len(selected) == 2:
            values.append(selected)
    if len(values) != 1:
        raise ValueError('one_explicit_retained_controller_workspace_required')
    lower, upper = np.asarray(values[0]['workspace_min']), np.asarray(values[0]['workspace_max'])
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or not (lower < upper).all():
        raise ValueError('finite_retained_controller_workspace_required')
    return values[0]


def within_workspace(T):
    p = np.asarray(T, float)[:3, 3]
    bounds = runtime_workspace()
    return bool(np.isfinite(p).all() and (p >= bounds['workspace_min']).all()
                and (p <= bounds['workspace_max']).all())


class WorkspaceReleaseSelector(ReleaseSelector):
    def ik(self, arm, T, seed_qpos):
        if not within_workspace(T):
            return None, 'outside_retained_complete_path_workspace'
        return super().ik(arm, T, seed_qpos)



def propose(original, assets, receivers, seed):
    """Preserve task-relative slots; sample source positions near their goals.

    The target lane is moved towards the fixed robot bases. The same sources
    are never replaced to make packing or reaching easier. This proposal is
    only valid for the existing two fixed bases on the negative-X table side.
    """
    limits = runtime_workspace()
    rng = np.random.default_rng(seed)
    result = deepcopy(original)
    occupied, draws = [], []
    # Preserve Y packing but give all receivers one reachable X lane. Keeping
    # radius-dependent X offsets had moved the largest plates into the only
    # usable source lane, making the fixed six-object layout impossible.
    targets = result.get('containers', [])
    if not targets or any(p['target']['kind'] != 'container'
                          for t in result['tasks'] for p in t['placements']):
        raise Rejected('this_layout_adapter_requires_explicit_container_goals')
    lane_x = float(rng.uniform(.10, .15))
    target_displacements = []
    for c in targets:
        target_displacements.append({'id': c['id'], 'original_xy': list(c['xy']),
                                     'revised_xy': [round(lane_x, 7), c['xy'][1]]})
        c['xy'][0] = round(lane_x, 7)
        radius = float(np.linalg.norm(np.asarray(receivers[c['source_id']]['dimensions_m'][:2]) / 2))
        table_fit(c['xy'], radius); check_separation(c['xy'], radius, occupied)
        occupied.append((c['id'], np.asarray(c['xy']), radius))
    by_id = {c['id']: c for c in targets}
    destinations = {}
    for t in result['tasks']:
        for p in t['placements']:
            destinations.setdefault(p['object'], set()).add(p['target']['container'])
    # Large payloads first; this changes no object IDs or task ordering.
    radii = {o['id']: float(np.linalg.norm(assets[o['source_id']]['half_size'][:2]))
             for o in result['objects']}
    for o in sorted(result['objects'], key=lambda o: (-radii[o['id']], o['id'])):
        radius = radii[o['id']]
        ys = [by_id[c]['xy'][1] for c in sorted(destinations[o['id']])]
        y0 = float(np.mean(ys))
        bounds = [[limits['workspace_min'][0] + radius + .012, -.08 - radius],
                  [max(limits['workspace_min'][1] + radius + .04, y0 - .30),
                   min(limits['workspace_max'][1] - radius - .04, y0 + .30)]]
        if any(a >= b for a, b in bounds):
            raise Rejected('fixed_asset_has_no_source_lane', object=o['id'])
        for attempt in range(128):
            xy = np.round([rng.uniform(*b) for b in bounds], 7)
            try:
                table_fit(xy, radius); check_separation(xy, radius, occupied)
            except Rejected as error:
                draws.append({'object': o['id'], 'attempt': attempt, 'xy': xy.tolist(),
                              'accepted': False, 'reason': error.code})
                continue
            o['xy'] = xy.tolist(); o['yaw'] = float(rng.uniform(-np.pi, np.pi))
            occupied.append((o['id'], xy, radius))
            draws.append({'object': o['id'], 'attempt': attempt, 'xy': xy.tolist(), 'accepted': True})
            break
        else:
            raise Rejected('fixed_sources_bounded_layout_exhausted', object=o['id'], draws=draws)
    # Coordinates may change, but identities, exact variants and local goals cannot.
    for key in ('tasks', 'variant_choices', 'container_variant_choices'):
        if result[key] != original[key]:
            raise ValueError('layout_changed_source_size_cache_or_task')
    return result, {'schema': VERSION, 'seed': seed, 'target_common_lane_x_m': lane_x, 'target_displacements': target_displacements,
                    'point_proposals': draws, 'retained_controller_workspace': limits, 'asset_reselection': False, 'runtime_rescaling': False}


def unique_operations(tasks):
    seen = set()
    for task in tasks:
        for operation in task['operations']:
            key = digest([operation['object'], operation['target'], operation.get('release')])
            if key not in seen:
                seen.add(key)
                yield task['requested_task_id'], operation


def workspace(scene, tasks, radius=.90):
    if not 0 < radius <= .95:
        raise ValueError('original_runtime_radius_must_not_be_relaxed')
    roots = np.asarray(scene['robot_roots'], float)
    rows = []
    for task_id, operation in unique_operations(tasks):
        obj = next(o for o in scene['objects'] if o['id'] == operation['object'])
        source = pose(obj['position'], obj['yaw'])
        cache = read(resolve(scene['grasp_caches'][obj['id']]))
        trials = []
        for candidate in cache['candidates'][:12]:
            local = np.asarray(candidate['tcp_in_asset'])
            try:
                desired, proof = release_geometry(scene, operation, local)
            except ValueError as error:
                trials.append({'candidate': candidate['id'], 'passed': False, 'reason': str(error)})
                continue
            grasp, release = source @ local, desired @ local
            pre, above = grasp.copy(), release.copy()
            pre[2, 3] += .17; above[2, 3] += .18
            for arm, root in enumerate(roots):
                distances = [float(np.linalg.norm(T[:3, 3] - root)) for T in (pre, grasp, above, release)]
                within = [within_workspace(T) for T in (pre, grasp, above, release)]
                trials.append({'candidate': candidate['id'], 'arm': arm,
                               'passed': max(distances) <= radius and all(within),
                               'endpoint_workspace_passed': within,
                               'world_tcp_positions_m': [T[:3, 3].tolist() for T in (pre, grasp, above, release)],
                               'maximum_endpoint_distance_m': max(distances), 'release_geometry': proof})
        rows.append({'task': task_id, 'object': obj['id'], 'passed': any(r['passed'] for r in trials),
                     'trials': trials})
    return {'passed': bool(rows) and all(r['passed'] for r in rows), 'operations': rows,
            'sampling_radius_m': radius, 'controller_workspace': runtime_workspace(),
            'scope': 'all four endpoints inside retained controller workspace and same arm radius; not IK or physics'}


def qualify(scene, tasks, directory):
    selector = WorkspaceReleaseSelector(scene['robot_roots'], scene['seed'])
    rows = []; operations = list(unique_operations(tasks)); began = time.monotonic()
    for number, (tid, operation) in enumerate(operations):
        support = next(s for s in scene['supports'] if s['id'] == operation['target']['support'])
        proof = selector.select(scene, operation, support)
        path = Path(directory) / f'{number:02d}.json'; write(path, proof)
        rows.append({'task': tid, 'object': operation['object'], 'passed': proof['passed'], 'proof': binding(path)})
        if not proof['passed']:
            break
    result = {'passed': len(rows) == len(operations) and all(r['passed'] for r in rows),
              'checked': rows, 'unique_operations': len(operations), 'not_checked': len(operations) - len(rows),
              'seconds': time.monotonic() - began, 'physical_success': 'unmeasured',
              'controller_workspace': runtime_workspace(),
              'scope': 'all required same-arm endpoint IK and controller workspace; complete path and measured execution still required'}
    write(Path(directory) / 'RESULT.json', result)
    return result


def build(bundle, source_cohorts, output, *, seed, maximum_proposals=16):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_attempts')
    if not 1 <= len(source_cohorts) <= 12 or type(maximum_proposals) is not int or not 1 <= maximum_proposals <= 16:
        raise ValueError('at_most_twelve_scenes_sixteen_layout_proposals')
    protocol = {'schema': VERSION, 'implementation': binding(__file__), 'bundle': binding(bundle),
        'source_cohorts': [binding(p) for p in source_cohorts], 'seed': seed,
        'maximum_proposals_per_scene': maximum_proposals,
        'selector': binding(Path(__file__).parents[1] / 'resimpair_pickplace/above_rim_release_v1.py'),
        'exact_factory': binding(factory.__file__),
        'controller_workspace': runtime_workspace(),
        'retained_motion': binding(Path(__file__).parents[1] / 'resimpair_grasp/motion.py'), 'API_calls': 0, 'physics_attempts': 0, 'GPU_calls': 0,
        'fixed_sources_sizes_caches_receivers_tasks': True, 'default_promoted': False}
    write(output / 'PROTOCOL.json', protocol)
    b = read(bundle); all_entries = read(resolve(b['native_cache_index']))['entries']
    rows = []; began = time.monotonic()
    for index, cp in enumerate(source_cohorts):
        cases = read(cp)['cases']; first = cases[0]; original = read(resolve(first['source_request']))
        previous_scene = read(resolve(first['scene']))
        if previous_scene['robot_roots'] != [[-.6, -.35, 0.], [-.6, .35, 0.]]:
            raise ValueError('retained_layout_robot_roots_required')
        entries = factory.select_variants(all_entries, [o['source_id'] for o in original['objects']], seed,
                                          original['variant_choices'])
        assets = {e['source_id']: read(resolve(e['asset'])) for e in entries}
        receivers = {sid: read(resolve(ref)) for sid, ref in original['container_variant_choices'].items()}
        folder = output / 'scenes' / f'{index:02d}'; attempts = []; accepted = None
        for number in range(maximum_proposals):
            current = folder / 'proposals' / f'{number:02d}'; current_seed = seed + index * 101 + number
            request = None; stage = 'layout'; attempt_started = time.monotonic()
            try:
                request, trace = propose(original, assets, receivers, current_seed)
                request['id'] = original['id'] + '_reachablev3'
                write(current / 'requests.json', {'schema': factory.VERSION, 'scenes': [request]})
                write(current / 'sampling.json', trace)
                stage = 'compile'
                summary = factory.compile_requests(bundle, current / 'requests.json', current / 'factory', previous_scene['seed'])
                if summary['compiled_scenes'] != 1:
                    raise Rejected('retained_factory_rejected', report=binding(current / 'factory/scenes.json'))
                fresh = read(current / 'factory/cohort.json')['cases']
                scene = read(resolve(fresh[0]['scene'])); tasks = [read(resolve(c['task'])) for c in fresh]
                if scene['metric_variants'] != previous_scene['metric_variants']:
                    raise ValueError('same_source_size_cache_native_properties_required')
                stage = 'workspace'; geometry = workspace(scene, tasks)
                write(current / 'workspace.json', geometry)
                if not geometry['passed']:
                    raise Rejected('no_common_arm_inside_retained_controller_workspace')
                stage = 'endpoint_IK'; proof = qualify(scene, tasks, current / 'endpoint_IK')
                if not proof['passed']:
                    raise Rejected('no_common_arm_required_endpoint_IK')
                accepted = {'proposal': number, 'cohort': binding(current / 'factory/cohort.json'),
                            'qualification': binding(current / 'endpoint_IK/RESULT.json')}
                attempts.append({'passed': True, **accepted, 'seconds': time.monotonic() - attempt_started})
                break
            except (Rejected, ValueError) as error:
                record = {'proposal': number, 'stage': stage, 'passed': False,
                          'reason': getattr(error, 'code', str(error)), 'details': getattr(error, 'details', {}),
                          'seconds': time.monotonic() - attempt_started}
                write(current / 'REJECTED.json', record); attempts.append(record)
        row = {'parent_scene': first['scene'], 'partition': first['partition'],
               'source_ids': [o['source_id'] for o in original['objects']], 'N': len(original['objects']),
               'accepted': accepted, 'attempts': attempts, 'physical_success': 'unmeasured'}
        write(folder / 'RESULT.json', row); rows.append(row)
        print({'scene': index, 'endpoint_eligible': accepted is not None, 'proposals': len(attempts)}, flush=True)
    result = {'schema': VERSION, 'rows': rows, 'scenes': len(rows),
              'endpoint_eligible_scenes': sum(r['accepted'] is not None for r in rows),
              'seconds': time.monotonic() - began, 'physics_attempts': 0, 'API_calls': 0, 'GPU_calls': 0,
              'source_size_cache_native_properties_preserved': True, 'default_promoted': False}
    write(output / 'RESULTS.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--source-cohort', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--maximum-proposals', type=int, default=16)
    args = parser.parse_args()
    print(build(args.bundle, args.source_cohort, args.output,
                seed=args.seed, maximum_proposals=args.maximum_proposals))
