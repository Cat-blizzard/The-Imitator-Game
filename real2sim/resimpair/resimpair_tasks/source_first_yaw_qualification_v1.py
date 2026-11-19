"""Choose bounded initial yaw candidates for fixed sources, receivers and XY.

This is a planning-only layout step. It preserves explicit task goal yaw, exact
asset/cache bindings and N. Implicit goal yaw follows the sampled source yaw,
as in the existing task compiler. Failed trials stay in the evidence directory.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from . import source_first_reachable_layout_v3 as layout

VERSION = 'source_first_yaw_qualification_v1'


def sampled_state(scene, tasks, request, oid, yaw):
    scene, tasks = deepcopy(scene), deepcopy(tasks)
    next(o for o in scene['objects'] if o['id'] == oid)['yaw'] = yaw
    for task in tasks:
        intent = next(t for t in request['tasks'] if t['id'] == task['requested_task_id'])
        for op in task['operations']:
            if op['object'] != oid:
                continue
            matches = [p for p in intent['placements'] if p['object'] == oid
                       and p['target'].get('container') == op['target']['support']]
            if len(matches) != 1:
                raise ValueError('unambiguous_original_goal_orientation_required')
            if 'yaw' not in matches[0]['target']:
                op['target']['yaw'] = yaw
    return scene, tasks


def revise(bundle, source_cohort, output, *, yaw_samples=8):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_yaw_attempts')
    if type(yaw_samples) is not int or not 1 <= yaw_samples <= 8:
        raise ValueError('one_to_eight_finite_yaw_proposals_required')
    cases = read(source_cohort)['cases']
    if not cases or len({digest(c['scene']) for c in cases}) != 1:
        raise ValueError('one_existing_scene_required')
    first = cases[0]; original = read(resolve(first['source_request']))
    scene = read(resolve(first['scene'])); tasks = [read(resolve(c['task'])) for c in cases]
    if any(p['target']['kind'] != 'container' for t in original['tasks'] for p in t['placements']):
        raise ValueError('explicit_container_goal_contract_required')
    write(output / 'PROTOCOL.json', {'schema': VERSION, 'implementation': binding(__file__),
        'bundle': binding(bundle), 'source_cohort': binding(source_cohort),
        'maximum_yaw_proposals_per_object': yaw_samples, 'maximum_objects': 8,
        'retained_endpoint_selector': binding(layout.__file__), 'physics_steps': 0, 'API_calls': 0,
        'new_grasps': 0, 'rescaling': False, 'new_source_or_XY_draws': 0,
        'explicit_target_yaw_preserved': True, 'scope': 'related development layout; not blind heldout or physics'})
    if not 3 <= len(original['objects']) <= 8:
        raise ValueError('three_to_eight_original_objects_required')
    request = deepcopy(original); rows = []; began = time.monotonic()
    selector = layout.WorkspaceReleaseSelector(scene['robot_roots'], scene['seed'])
    for obj in request['objects']:
        old = obj['yaw']; attempts = []; chosen = None
        for k in range(yaw_samples):
            yaw = float((old + 2 * np.pi * k / yaw_samples + np.pi) % (2 * np.pi) - np.pi)
            candidate_scene, candidate_tasks = sampled_state(scene, tasks, request, obj['id'], yaw)
            reports = []
            for tid, op in layout.unique_operations(candidate_tasks):
                if op['object'] != obj['id']:
                    continue
                support = next(s for s in candidate_scene['supports'] if s['id'] == op['target']['support'])
                proof = selector.select(candidate_scene, op, support)
                reports.append({'requested_task_id': tid, 'operation': op, 'proof': proof})
                if not proof['passed']:
                    break
            passed = bool(reports) and all(r['proof']['passed'] for r in reports)
            path = output / 'yaw_trials' / obj['id'] / f'{k:02d}.json'
            write(path, {'object': obj['id'], 'yaw': yaw, 'passed': passed, 'operations': reports})
            attempts.append({'yaw': yaw, 'passed': passed, 'evidence': binding(path)})
            if passed:
                chosen = yaw; scene, tasks = candidate_scene, candidate_tasks; obj['yaw'] = yaw
                break
        rows.append({'object': obj['id'], 'original_yaw': old, 'chosen_yaw': chosen, 'attempts': attempts})
        if chosen is None:
            break
    ready = len(rows) == len(request['objects']) and all(r['chosen_yaw'] is not None for r in rows)
    final = {'schema': VERSION, 'rows': rows, 'all_objects_endpoint_qualified': ready,
             'objects_requested': len(request['objects']), 'physics_steps': 0, 'API_calls': 0,
             'physical_success': 'unmeasured', 'runtime_rescaling': False, 'new_source_or_XY_draws': 0}
    if ready:
        for key in ('tasks', 'variant_choices', 'container_variant_choices', 'containers'):
            if request[key] != original[key]:
                raise ValueError('yaw_selection_changed_fixed_task_or_geometry')
        request['id'] = original['id'] + '_yaw_' + digest(rows)[:16]
        write(output / 'requests.json', {'schema': layout.factory.VERSION, 'scenes': [request]})
        result = layout.factory.compile_requests(bundle, output / 'requests.json', output / 'factory', scene['seed'])
        final['factory_summary'] = result
        if result['compiled_scenes'] == 1:
            fresh = read(output / 'factory/cohort.json')['cases']
            actual_scene = read(resolve(fresh[0]['scene'])); actual_tasks = [read(resolve(c['task'])) for c in fresh]
            for key in ('objects', 'supports', 'grasp_caches', 'metric_variants',
                        'robot_roots', 'table_top_z', 'cameras', 'appearance'):
                if actual_scene[key] != scene[key]:
                    raise ValueError('final_yaw_compile_changed_other_scene_conditions:' + key)
            proof = layout.qualify(actual_scene, actual_tasks, output / 'final_endpoint_IK')
            final.update(cohort=binding(output / 'factory/cohort.json'),
                         final_endpoint_qualified=proof['passed'],
                         final_endpoint_evidence=binding(output / 'final_endpoint_IK/RESULT.json'))
        else:
            final['final_endpoint_qualified'] = False
    else:
        final['final_endpoint_qualified'] = False
    final['seconds'] = time.monotonic() - began
    write(output / 'RESULT.json', final)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'source-cohort', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    value = revise(args.bundle, args.source_cohort, args.output)
    print({key: value[key] for key in ('objects_requested', 'all_objects_endpoint_qualified',
                                     'final_endpoint_qualified', 'seconds')})
