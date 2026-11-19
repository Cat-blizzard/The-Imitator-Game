"""Create a controller-release variant without changing scene or final goals."""
from copy import deepcopy
from pathlib import Path

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.above_rim_release_v1 import policy
from resimpair_pickplace.drop_event_protocol_v1 import validate_contract


def apply_release_variant(case, operations, output, *, mode='drop_from_above'):
    parent = read(case) if isinstance(case, (str, Path)) else deepcopy(case)
    if not isinstance(parent, dict) or mode not in ('drop_from_above', 'supported_place'):
        raise ValueError('bound_case_and_explicit_release_mode_required')
    if not isinstance(operations, list) or not 1 <= len(operations) <= 8 or len(set(operations)) != len(operations):
        raise ValueError('one_to_eight_unique_operation_ids_required')
    task = deepcopy(read(resolve(parent['task'])))
    source = deepcopy(read(resolve(parent['source_request'])))
    resolve(parent['scene'])
    if mode == 'drop_from_above' and validate_contract(task) is None:
        raise ValueError('original_contract_does_not_allow_low_drop; new_protocol_required')
    by_id = {o['id']: o for o in task['operations']}
    if not set(operations) <= set(by_id):
        raise ValueError('unknown_requested_operation')
    changes = []
    for oid in operations:
        operation = by_id[oid]
        before = deepcopy(operation.get('release', {}))
        after = {**before, 'mode': mode, 'allow_drop': mode == 'drop_from_above'}
        operation['release'] = after
        policy(operation)  # Includes explicit non-fragile assumption and fall-height bounds.
        changes.append({'operation': oid, 'object': operation['object'], 'before': before, 'after': after})
    if not any(c['before'] != c['after'] for c in changes):
        raise ValueError('release_variant_requires_an_actual_controller_change')
    request = {'parent': parent, 'operations': operations, 'mode': mode,
               'implementation': binding(__file__)}
    output = Path(output).resolve()
    if output.exists():
        if not (output / 'revision.json').exists() or read(output / 'request.json') != request:
            raise ValueError('existing_release_revision_conflict_or_incomplete')
        result = read(output / 'revision.json')
        for ref in [result['case']['task'], result['case']['scene'], result['case']['source_request']]: resolve(ref)
        return result
    new_id = 'release_' + digest(request)[:12] + '_' + parent['requested_task_id']
    task['id'] = new_id
    source['controller_release_revision'] = {
        'parent_source_request': parent['source_request'], 'requested_task_id': parent['requested_task_id'],
        'changes': changes, 'same_final_goals_and_original_low_drop_contract': True,
        'real_world_material_confirmation': False}
    write(output / 'request.json', request)
    write(output / 'task.json', task)
    write(output / 'source_request.json', source)
    revised = {**deepcopy(parent), 'id': new_id, 'execution_id': new_id,
        'task': binding(output / 'task.json'), 'source_request': binding(output / 'source_request.json'),
        'physical_status': 'not_executed', 'revision_scope': 'same_final_goals_controller_release_variant',
        'revision_parent': {'case_id': parent['id'], 'scene': parent['scene'],
                            'task': parent['task'], 'source_request': parent['source_request']}}
    result = {'schema': 'immutable_controller_release_variant_v1', 'case': revised,
        'changes': changes, 'same_scene_objects_sizes_goals_and_contract': True,
        'physical_success': 'unmeasured', 'required_execution_profile': 'task_release',
        'new_physical_attempts': 0, 'API_calls': 0, 'GPU_jobs': 0}
    write(output / 'case.json', revised); write(output / 'revision.json', result)
    return result
