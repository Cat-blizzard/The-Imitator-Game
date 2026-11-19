import copy

from resimpair_grasp.common import read, resolve
from resimpair_tasks.multitask_scene_factory_v3 import endpoint_qualification


def test_same_layout_endpoint_cache_preserves_failure_and_goal_identity(tmp_path):
    calls = []

    class Selector:
        def __init__(self, roots, seed): pass

        def select(self, scene, operation, container):
            calls.append((operation['object'], copy.deepcopy(operation['target']), container))
            passed = operation['target']['offset_xy'][0] < .2
            return {'passed': passed, 'reason': None if passed else 'no_single_arm_source_and_destination_grasp'}

    source = {'robot_roots': [[-.6, -.35, 0], [-.6, .35, 0]], 'seed': 1, 'supports': []}
    op = {'id': 'transfer_0', 'object': 'a', 'target': {'support': 'table', 'offset_xy': [.0, .0], 'yaw': .0}}
    bad = copy.deepcopy(op); bad['id'] = 'transfer_1'; bad['target']['offset_xy'][0] = .3
    tasks = [{'requested_task_id': 'one', 'operations': [op]},
             {'requested_task_id': 'two', 'operations': [copy.deepcopy(op), bad]},
             {'requested_task_id': 'three', 'operations': [copy.deepcopy(bad)]}]
    before = copy.deepcopy(tasks)
    result = endpoint_qualification(source, tasks, tmp_path, Selector)
    assert not result['passed']
    assert len(calls) == result['unique_target_checks'] == 2
    assert result['operation_requests'] == 4 and result['cache_reuses'] == 2
    assert result['failed_unique_targets'] == 1
    assert [r['passed'] for r in result['operations']] == [True, True, False, False]
    assert tasks == before
    assert read(resolve(result['operations'][2]['report']))['report']['reason'] == 'no_single_arm_source_and_destination_grasp'


def test_container_sweep_is_requested_without_treating_vacancy_as_initial_collision(tmp_path):
    received = []

    class Selector:
        def __init__(self, roots, seed): pass

        def select(self, scene, operation, container):
            received.append(container)
            return {'passed': container is None, 'reason': 'container_sweep_rejected' if container else None}

    support = {'id': 'bin', 'container_descriptor': {'path': 'unused_by_test'}}
    scene = {'robot_roots': [[-.6, -.35, 0], [-.6, .35, 0]], 'seed': 1, 'supports': [support],
             'objects': [{'id': 'blocker', 'position': [0., 0., .05]}]}
    tasks = [{'requested_task_id': 'clear_then_place', 'operations': [
        {'id': 'a', 'object': 'a', 'target': {'support': 'table', 'offset_xy': [0., 0.], 'yaw': 0.}}]},
        {'requested_task_id': 'container', 'operations': [
        {'id': 'a', 'object': 'a', 'target': {'support': 'bin', 'offset_xy': [0., 0.], 'yaw': 0.}}]}]
    result = endpoint_qualification(scene, tasks, tmp_path, Selector)
    assert received == [None, support]
    assert result['operations'][0]['passed'] and not result['operations'][1]['passed']
    assert not result['passed']
    assert result['physical_executions'] == result['GPU_jobs'] == result['API_calls'] == 0
