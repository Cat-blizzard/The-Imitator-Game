"""Finite new goal-position requests for an existing development scene.

Rotate the retained size-only slots and translate the whole group toward the
robot bases, bounded by the original circular packing and actual floor fit.
Initial poses, object/container descriptors, caches and task membership stay
fixed. These are different target-position tasks, not same-goal control fixes.
Only the retained static compiler and endpoint selector run; no simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import inspect
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace import requested_tasks as retained
from . import metric_population_sampler_v2 as packing
from . import source_first_reachable_layout_v3 as endpoint

VERSION = 'source_first_slot_layout_v1'
ANGLES = (0., np.pi / 2, np.pi, -np.pi / 2)
MAX_SHIFT_M = .04
SHIFT_FRACTION = .9


def disk_interval(points, radii, direction):
    """Exact ray interval inside every disk; no numerical margin relaxation."""
    points = np.asarray(points, float).reshape((-1, 2))
    radii = np.broadcast_to(np.asarray(radii, float), (len(points),))
    direction = np.asarray(direction, float)
    if (not np.isfinite([*points.ravel(), *radii, *direction]).all()
            or np.any(radii < 0) or not np.isclose(np.linalg.norm(direction), 1.)):
        raise ValueError('finite_points_nonnegative_disk_radii_and_unit_direction_required')
    projected = points @ direction
    discriminant = projected ** 2 + radii ** 2 - np.sum(points ** 2, axis=1)
    if np.any(discriminant < 0):
        return None
    root = np.sqrt(discriminant)
    lower, upper = max(0., float(np.max(-projected - root))), float(np.min(-projected + root))
    return None if lower > upper else [lower, upper]


def load_groups(request, scene):
    import trimesh
    objects = {o['id']: o for o in request['objects']}
    supports = {s['id']: s for s in scene['supports']}
    groups = {}
    for task in request['tasks']:
        for placement in task['placements']:
            target = placement['target']
            if target['kind'] != 'container':
                raise ValueError('explicit_container_goals_required')
            cid, oid = target['container'], placement['object']
            members = groups.setdefault(cid, {})
            data = {'offset': target.get('offset_xy_m', [0., 0.]),
                    'yaw': target.get('yaw', objects[oid].get('yaw', 0.))}
            if oid in members and members[oid] != data:
                raise ValueError('one_consistent_slot_per_object_and_container_required')
            members[oid] = data
    result = []
    for cid, members in groups.items():
        support = supports[cid]
        descriptor = read(resolve(support['container_descriptor']))
        ids, radii, vertices, yaws, offsets = [], [], [], [], []
        for oid, goal in members.items():
            asset = read(resolve(scene['metric_variants'][oid]['asset']))
            mesh = trimesh.load(resolve(asset['visual']), force='scene', process=False).to_geometry()
            ids.append(oid); radii.append(float(np.linalg.norm(asset['half_size'][:2])))
            vertices.append(np.asarray(mesh.vertices)); yaws.append(goal['yaw']); offsets.append(goal['offset'])
        model = packing.slots(radii, descriptor['cavity']['flat_floor_radius_m'])
        if not model['passed']:
            raise ValueError('original_size_only_slots_do_not_fit_fixed_container')
        if not np.allclose(offsets, model['offsets'], atol=1e-12, rtol=0):
            raise ValueError('original_request_must_bind_retained_canonical_slots')
        center = (np.asarray(support['position'][:2]) + retained.rotation(support['yaw'])
                  @ np.asarray(descriptor['cavity']['center_xy_m']))
        toward = np.mean(np.asarray(scene['robot_roots'])[:, :2], axis=0) - center
        if np.linalg.norm(toward) <= 1e-12:
            raise ValueError('robot_side_direction_is_undefined')
        direction = retained.rotation(support['yaw']).T @ (toward / np.linalg.norm(toward))
        result.append({'container': cid, 'support': support, 'descriptor': descriptor,
                       'objects': ids, 'vertices': vertices, 'yaws': yaws, 'packing': model,
                       'offsets': np.asarray(offsets, float), 'direction': direction})
    return result


def group_proposal(group, angle, shifted):
    model = group['packing']; cavity = group['descriptor']['cavity']
    offsets = group['offsets'] @ retained.rotation(angle).T
    baseline = [retained.floor_fit(v, cavity, yaw, group['support']['yaw'], offset)
                for v, yaw, offset in zip(group['vertices'], group['yaws'], offsets)]
    circle_limits = (cavity['flat_floor_radius_m'] - np.asarray(model['radii_m'])
                     - model['required_boundary_margin_m'])
    circle_interval = disk_interval(offsets, circle_limits, group['direction'])
    floor_intervals = []
    for vertices, yaw, offset, fit in zip(group['vertices'], group['yaws'], offsets, baseline):
        points = vertices[:, :2] @ retained.rotation(yaw - group['support']['yaw']).T + offset
        floor_intervals.append(disk_interval(points, cavity['flat_floor_radius_m']
                                             - fit['required_margin_m'], group['direction']))
    intervals = [circle_interval, *floor_intervals]
    feasible = (all(i is not None and i[0] == 0. for i in intervals)
                and all(f['passed'] for f in baseline))
    maximum = min(i[1] for i in intervals) if feasible else 0.
    amount = min(MAX_SHIFT_M, SHIFT_FRACTION * maximum) if shifted else 0.
    delta = group['direction'] * amount
    final = offsets + delta
    fits = [retained.floor_fit(v, cavity, yaw, group['support']['yaw'], offset)
            for v, yaw, offset in zip(group['vertices'], group['yaws'], final)]
    margins = cavity['flat_floor_radius_m'] - np.linalg.norm(final, axis=1) - model['radii_m']
    original_distances = np.linalg.norm(group['offsets'][:, None] - group['offsets'][None, :], axis=2)
    new_distances = np.linalg.norm(final[:, None] - final[None, :], axis=2)
    spacing_preserved = bool(np.allclose(original_distances, new_distances, atol=1e-12, rtol=0))
    # Retained slots already qualifies its exact construction, then rounds the
    # returned coordinates to 1e-7 m. Rigid motion preserves that same spacing.
    # Do not change its gap threshold to accommodate an unrelated new packing.
    passed = (feasible and spacing_preserved and all(f['passed'] for f in fits)
              and bool(np.all(margins >= model['required_boundary_margin_m'] - 1e-9)))
    return {'container': group['container'], 'objects': group['objects'], 'passed': passed,
            'retained_slots': model, 'rotation_radians': float(angle), 'shift_requested': shifted,
            'shift_direction_container_xy': group['direction'].tolist(), 'shift_distance_m': float(amount),
            'maximum_geometry_feasible_shift_m': float(maximum), 'shift_xy_m': delta.tolist(),
            'circle_shift_interval_m': circle_interval, 'actual_floor_shift_intervals_m': floor_intervals,
            'unshifted_rotated_floor_fit': baseline, 'final_floor_fit': fits,
            'minimum_circle_boundary_margin_m': float(margins.min()),
            'pair_spacing_preserved_from_original_request': spacing_preserved,
            'original_offsets_xy_m': group['offsets'].tolist(), 'new_offsets_xy_m': final.tolist(),
            'original_floor_fit_calls': 2 * len(fits)}


def propose(request, groups, angle, shifted, number):
    traces = [group_proposal(g, angle, shifted) for g in groups]
    mapping = {(g['container'], oid): offset for g in traces
               for oid, offset in zip(g['objects'], g['new_offsets_xy_m'])}
    candidate = deepcopy(request); changes = []
    candidate['id'] = request['id'] + '_slotsv1_' + f'{number:02d}'
    for task in candidate['tasks']:
        for index, placement in enumerate(task['placements']):
            target = placement['target']; old = deepcopy(target.get('offset_xy_m', [0., 0.]))
            new = mapping[(target['container'], placement['object'])]
            target['offset_xy_m'] = new
            if old != new:
                changes.append({'path': f"tasks/{task['id']}/placements/{index}/target/offset_xy_m",
                                'object': placement['object'], 'container': target['container'],
                                'old': old, 'new': new})
    restored = deepcopy(candidate); restored['id'] = request['id']
    for actual, original in zip(restored['tasks'], request['tasks']):
        for a, b in zip(actual['placements'], original['placements']):
            if 'offset_xy_m' in b['target']:
                a['target']['offset_xy_m'] = deepcopy(b['target']['offset_xy_m'])
            else:
                a['target'].pop('offset_xy_m', None)
    if restored != request:
        raise ValueError('slot_adapter_changed_fields_other_than_goal_offset_and_request_id')
    return candidate, {'proposal': number, 'passed': all(g['passed'] for g in traces),
                       'groups': traces, 'changes': changes, 'initial_poses_and_fixed_fields_preserved': True,
                       'task_variant': 'new_target_positions' if changes else 'unchanged_target_reference',
                       'same_goal_control_fix': False}


def qualify_all(scene, tasks, directory):
    """Same retained selector and thresholds; inspect all six operations."""
    selector = endpoint.WorkspaceReleaseSelector(scene['robot_roots'], scene['seed'])
    workspace = endpoint.workspace(scene, tasks)
    write(directory / 'WORKSPACE.json', workspace)
    rows, by_key, failure_counts = [], {}, Counter()
    decisions = ik_calls = 0
    for number, (tid, operation) in enumerate(endpoint.unique_operations(tasks)):
        support = next(s for s in scene['supports'] if s['id'] == operation['target']['support'])
        proof = selector.select(scene, operation, support)
        write(directory / f'{number:02d}.json', proof)
        row = {'task': tid, 'object': operation['object'], 'passed': bool(proof['passed']),
               'workspace_passed': workspace['operations'][number]['passed'],
               'proof': binding(directory / f'{number:02d}.json')}
        rows.append(row); decisions += len(proof['decisions'])
        for decision in proof['decisions']:
            ik_calls += len(decision['checks'])
            if not decision['accepted']:
                if decision['checks']:
                    last = decision['checks'][-1]
                    failure_counts[last['stage'] + ':' + str(last['status'])] += 1
                else:
                    failure_counts[decision.get('reason', 'container_hand_clearance')] += 1
        key = digest([operation['object'], operation['target'], operation.get('release')])
        by_key[key] = row['passed'] and row['workspace_passed']
    task_rows = [{'task': t['requested_task_id'], 'passed': all(by_key[digest(
        [op['object'], op['target'], op.get('release')])] for op in t['operations'])} for t in tasks]
    result = {'passed': bool(task_rows) and all(t['passed'] for t in task_rows),
              'checked': rows, 'tasks': task_rows, 'unique_operations': len(rows),
              'selector_candidate_arm_decisions': decisions, 'endpoint_ik_calls': ik_calls,
              'failed_candidate_stage_counts': dict(failure_counts), 'initial_yaw_samples': 1,
              'physical_success': 'unmeasured', 'physics_steps': 0,
              'scope': 'retained workspace and same-arm endpoint IK only; no full paths or physical execution'}
    write(directory / 'RESULT.json', result)
    return result


def revise(bundle, source_cohort, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_slot_layout_outputs')
    cases = read(source_cohort)['cases']
    if (not cases or len({digest(c['scene']) for c in cases}) != 1
            or len({digest(c['source_request']) for c in cases}) != 1
            or any(c['partition'] != 'development' for c in cases)):
        raise ValueError('one_fixed_development_scene_required')
    request = read(resolve(cases[0]['source_request'])); scene = read(resolve(cases[0]['scene']))
    original_tasks = [read(resolve(c['task'])) for c in cases]
    if (not 3 <= len(request['objects']) <= 8
            or {t['id'] for t in request['tasks']} != {t['requested_task_id'] for t in original_tasks}):
        raise ValueError('complete_three_to_eight_object_cohort_required')
    frozen = [binding(p) for p in (retained.__file__, packing.__file__, endpoint.__file__,
        endpoint.factory.__file__, inspect.getfile(endpoint.ReleaseSelector),
        Path(__file__).with_name('source_first_yaw_qualification_v1.py'),
        Path(__file__).parents[1] / 'resimpair_pickplace/cartesian_reachability.py',
        Path(__file__).parents[1] / 'resimpair_grasp/motion.py')]
    specifications = [{'proposal': i, 'rotation_radians': float(angle), 'shifted': shifted,
                       'role': 'unchanged_target_reference' if i == 0 else 'new_target_position_variant'}
                      for i, (angle, shifted) in enumerate((a, s) for a in ANGLES for s in (False, True))]
    write(output / 'PROTOCOL.json', {'schema': VERSION, 'implementation': binding(__file__),
        'bundle': binding(bundle), 'source_cohort': binding(source_cohort),
        'original_request': cases[0]['source_request'], 'original_scene': cases[0]['scene'],
        'frozen_implementations': frozen, 'proposals': specifications,
        'maximum_slot_layout_proposals': 8, 'maximum_unique_operation_selections': 8 * len(list(endpoint.unique_operations(original_tasks))),
        'initial_yaw_samples': 1, 'shift_upper_limit_m': MAX_SHIFT_M,
        'shift_fraction_of_geometry_limit': SHIFT_FRACTION,
        'shift_direction': 'cavity center toward mean fixed robot base, transformed to container XY',
        'goal_contract': 'explicit new target-position tasks; original request and failures retained',
        'retained_floor_fit_margin_m': inspect.signature(retained.floor_fit).parameters['margin'].default,
        'retained_slots_margin_m': inspect.signature(packing.slots).parameters['margin'].default,
        'retained_slots_gap_m': inspect.signature(packing.slots).parameters['gap'].default,
        'controller_workspace': endpoint.runtime_workspace(),
        'fixed_initial_objects_containers_caches_N': True, 'new_source_or_XY_draws': 0,
        'new_grasps': 0, 'runtime_rescaling': False, 'physics_steps': 0, 'physics_attempts': 0,
        'reserved_physical_scopes_consumed': 0, 'GPU_jobs': 0, 'API_calls': 0,
        'default_promoted': False, 'physical_success': 'unmeasured'})
    write(output / 'original_request.json', request)
    began = time.monotonic(); groups = load_groups(request, scene)
    plans = []
    # Freeze all geometry requests before the first endpoint IK call.
    for specification in specifications:
        number = specification['proposal']; directory = output / 'proposals' / f'{number:02d}'
        candidate, trace = propose(request, groups, specification['rotation_radians'], specification['shifted'], number)
        write(directory / 'requests.json', {'schema': endpoint.factory.VERSION, 'scenes': [candidate]})
        write(directory / 'GEOMETRY.json', trace)
        plans.append({**specification, 'requests': binding(directory / 'requests.json'),
                      'geometry': binding(directory / 'GEOMETRY.json'), 'geometry_passed': trace['passed']})
    write(output / 'PROPOSALS.json', {'all_frozen_before_endpoint_ik': True, 'proposals': plans})
    rows = []
    for plan in plans:
        directory = output / 'proposals' / f"{plan['proposal']:02d}"
        row = {**plan, 'factory_compiled': False, 'endpoint_qualified': False}
        stage = 'original_floor_and_slots'
        try:
            if not plan['geometry_passed']:
                raise ValueError('insufficient_actual_floor_or_original_slots_margin')
            stage = 'retained_factory'
            summary = endpoint.factory.compile_requests(bundle, resolve(plan['requests']), directory / 'factory', scene['seed'])
            row['factory_summary'] = summary
            if summary['compiled_scenes'] != 1:
                row['factory_rejections'] = binding(directory / 'factory/scenes.json')
                raise ValueError('retained_factory_rejected')
            fresh = read(directory / 'factory/cohort.json')['cases']
            actual_scene = read(resolve(fresh[0]['scene'])); tasks = [read(resolve(c['task'])) for c in fresh]
            stage = 'fixed_scene_invariants'
            for key in ('objects', 'supports', 'grasp_caches', 'metric_variants', 'robot_roots',
                        'table_top_z', 'cameras', 'appearance'):
                if actual_scene[key] != scene[key]:
                    raise ValueError('final_compile_changed_fixed_scene_field:' + key)
            for actual, original in zip(tasks, original_tasks):
                check = deepcopy(actual); check['id'] = original['id']
                for op, old in zip(check['operations'], original['operations']):
                    op['target']['offset_xy'] = deepcopy(old['target']['offset_xy'])
                if check != original:
                    raise ValueError('final_compile_changed_fields_other_than_goal_offsets_and_task_ids')
            row.update(factory_compiled=True, cohort=binding(directory / 'factory/cohort.json'),
                       fixed_initial_scene_verified=True)
            geometry_path = resolve(fresh[0]['scene']).with_name('geometry_checks.json')
            row['retained_factory_geometry'] = binding(geometry_path)
            row['retained_factory_floor_fit_calls'] = sum(len(c['payload_checks'])
                for c in read(geometry_path)['containers'])
            stage = 'endpoint_ik'
            proof = qualify_all(actual_scene, tasks, directory / 'endpoint_IK')
            row.update(endpoint_qualified=proof['passed'], endpoint_result=binding(directory / 'endpoint_IK/RESULT.json'),
                       passed_unique_operations=sum(r['passed'] and r['workspace_passed'] for r in proof['checked']),
                       task_results=proof['tasks'])
        except (ValueError, RuntimeError, KeyError) as error:
            row.update(failed_stage=stage, reason=getattr(error, 'code', str(error)),
                       details=getattr(error, 'details', None))
        write(directory / 'RESULT.json', row); rows.append(row)
    for ref in frozen:
        resolve(ref)
    proofs = [read(resolve(r['endpoint_result'])) for r in rows if r.get('endpoint_result')]
    geometries = [read(resolve(r['geometry'])) for r in rows]
    selected = next((r['cohort'] for r in rows if r['endpoint_qualified']), None)
    result = {'schema': VERSION, 'proposals': rows, 'requested_objects': len(request['objects']),
        'requested_tasks': len(request['tasks']), 'slot_layout_proposals': len(rows),
        'geometry_passed': sum(r['geometry_passed'] for r in rows),
        'factory_compiled': sum(r['factory_compiled'] for r in rows),
        'all_tasks_endpoint_qualified_proposals': sum(r['endpoint_qualified'] for r in rows),
        'selected_new_goal_cohort': selected, 'frozen_implementations_unchanged': True,
        'explicit_adapter_floor_fit_calls': sum(g['original_floor_fit_calls'] for t in geometries for g in t['groups']),
        'successful_factory_floor_fit_calls': sum(r.get('retained_factory_floor_fit_calls', 0) for r in rows),
        'new_target_position_variants': sum(bool(t['changes']) for t in geometries),
        'unique_operation_selections': sum(p['unique_operations'] for p in proofs),
        'selector_candidate_arm_decisions': sum(p['selector_candidate_arm_decisions'] for p in proofs),
        'endpoint_ik_calls': sum(p['endpoint_ik_calls'] for p in proofs),
        'new_source_or_XY_draws': 0, 'new_grasps': 0, 'physics_steps': 0, 'physics_attempts': 0,
        'reserved_physical_scopes_consumed': 0, 'GPU_jobs': 0, 'API_calls': 0,
        'initial_yaw_samples': 1, 'physical_success': 'unmeasured', 'default_promoted': False,
        'scope': 'one related development scene; new goal positions, retained static and endpoint checks only',
        'seconds': time.monotonic() - began}
    write(output / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'source-cohort', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = revise(args.bundle, args.source_cohort, args.output)
    print({k: result[k] for k in ('slot_layout_proposals', 'factory_compiled',
        'all_tasks_endpoint_qualified_proposals', 'endpoint_ik_calls', 'seconds')})
