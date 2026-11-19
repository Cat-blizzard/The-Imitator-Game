"""Independent request-to-measurement checks, in addition to retained rules.

The generator does not import this module. Goals are reconstructed from the
original user request, not copied from generated target coordinates.
"""
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import binding, digest, read, resolve, write

VERSION = 'requested_metric_goal_v1'
POSITION_TOLERANCE_M = .015
YAW_TOLERANCE_RAD = .15


def measured_goal(states, position, requested_yaw=None):
    states = np.asarray(states)
    errors = np.linalg.norm(states[:, :3] - position, axis=1)
    yaw_error = None
    if requested_yaw is not None:
        matrices = Rotation.from_quat(states[:, 3:7], scalar_first=True).as_matrix()
        yaw = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0])
        delta = yaw - requested_yaw
        yaw_error = float(np.abs(np.arctan2(np.sin(delta), np.cos(delta))).max())
    return {'passed': bool(len(states) == 20 and np.isfinite(states).all()
                          and errors.max() < POSITION_TOLERANCE_M
                          and (yaw_error is None or yaw_error < YAW_TOLERANCE_RAD)),
            'maximum_position_error_m': float(errors.max()), 'maximum_yaw_error_rad': yaw_error,
            'position_tolerance_m': POSITION_TOLERANCE_M, 'yaw_tolerance_rad': YAW_TOLERANCE_RAD,
            'settled_samples': len(states)}


def evaluate(directory, case):
    from .evaluator import evaluate as native_evaluate
    directory = Path(directory); output = directory / 'requested_goal_audit.json'
    if output.exists(): return read(output)
    native = native_evaluate(directory)
    result = {'version': VERSION, 'rule': binding(__file__), 'native_audit': binding(directory / 'container_audit.json'),
              'source_request': case['source_request'], 'passed': False, 'goals': [], 'errors': []}
    try:
        request = read(resolve(case['source_request']))
        requested_task = next(t for t in request['tasks'] if t['id'] == case['requested_task_id'])
        bundle = read(directory / 'execution_bundle.json'); scene = bundle['scene']
        ids = [o['id'] for o in scene['objects']]; objects = {o['id']: o for o in scene['objects']}
        requested_objects = {o['id']: o for o in request['objects']}
        supports = {s['id']: s for s in scene['supports']}
        requested_containers = {c['id']: c for c in request.get('containers', [])}
        with np.load(resolve(bundle['measurements']), allow_pickle=False) as data:
            states = data['object_states']
        for placement in requested_task['placements']:
            oid = placement['object']; obj = objects[oid]; wanted_object = requested_objects[oid]
            descriptor = read(resolve(obj['asset_descriptor']))
            if descriptor['source_id'] != wanted_object['source_id']: raise ValueError('requested_object_identity_changed')
            if not np.allclose(obj['position'][:2], wanted_object['xy'], atol=1e-9): raise ValueError('requested_source_xy_changed')
            goal = placement['target']; height = descriptor['half_size'][2]
            if goal['kind'] == 'table': expected = [*goal['xy'], scene['table_top_z'] + height]
            else:
                c = supports[goal['container']]; original = requested_containers[c['id']]
                d = read(resolve(c['container_descriptor']))
                if original.get('source_id', d['source_id']) != d['source_id']: raise ValueError('requested_container_changed')
                if not np.allclose(c['position'][:2], original['xy'], atol=1e-9): raise ValueError('requested_container_xy_changed')
                a = original.get('yaw', 0.); center = d['cavity']['center_xy_m']
                offset = goal.get('offset_xy_m', [0., 0.]); u, v = np.asarray(center) + offset
                expected = [original['xy'][0] + math.cos(a)*u - math.sin(a)*v,
                            original['xy'][1] + math.sin(a)*u + math.cos(a)*v,
                            d['cavity']['floor_z_m'] + height]
            result['goals'].append({'object': oid, 'expected_world_position_m': expected,
                **measured_goal(states[-20:, ids.index(oid)], expected, goal.get('yaw'))})
        result['passed'] = bool(native['passed'] and result['goals'] and all(x['passed'] for x in result['goals']))
        if not native['passed']: result['errors'].append('retained_native_acceptance_failed')
        if any(not x['passed'] for x in result['goals']): result['errors'].append('requested_metric_goal_not_met')
    except Exception as exc: result['errors'].append(repr(exc))
    write(output, result); return result


def audit(args):
    from .runner import verify
    root = args.root.resolve(); verify(root); rows = []
    for case in read(root / 'cohort.json')['cases']:
        directory = root / 'executions' / case['id']
        if not (directory / 'terminal.json').exists(): continue
        terminal = read(directory / 'terminal.json')
        result = evaluate(directory, case) if terminal['physical_attempts'] else {'passed': False, 'errors': [terminal['reason']]}
        rows.append({'case': case['id'], 'passed': result['passed'], 'errors': result['errors'],
                     'physical_attempts': terminal['physical_attempts'], 'elapsed_seconds': terminal['elapsed_seconds'],
                     'audit': binding(directory / 'requested_goal_audit.json') if terminal['physical_attempts'] else None})
    report = {'version': VERSION, 'registered': len(read(root / 'cohort.json')['cases']), 'rows': rows,
              'physical_executions': sum(x['physical_attempts'] for x in rows),
              'passed': sum(x['passed'] for x in rows), 'api_calls': 0}
    output = root / 'audits' / ('requested_' + digest(report)[:16] + '.json'); write(output, report)
    return {'report': str(output), **report}
