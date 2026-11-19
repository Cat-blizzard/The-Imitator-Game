import copy
from types import SimpleNamespace

import numpy as np
import pytest

from resimpair_pickplace.rolling_pickplace_v1 import PlanningView, ready_operations, select_operation


def task(edges=()):
    return {'operations': [{'id': 'clear', 'object': 'B'}, {'id': 'place', 'object': 'A'}],
            'required_operation_precedence': list(edges)}


def test_required_dependency_prevents_premature_placement():
    value = task([['clear', 'place']])
    assert [o['id'] for o in ready_operations(value, set())] == ['clear']
    assert [o['id'] for o in ready_operations(value, {'clear'})] == ['place']
    assert ready_operations(value, {'clear', 'place'}) == []


def test_unrequired_order_keeps_alternatives_without_regrasp():
    assert len(ready_operations(task(), set())) == 2
    value = task(); value['operations'][1]['object'] = 'B'
    with pytest.raises(ValueError, match='regrasp'):
        ready_operations(value, set())
    with pytest.raises(ValueError, match='cyclic'):
        ready_operations(task([['clear', 'place'], ['place', 'clear']]), set())


def fake_rec(tmp_path):
    state = np.arange(4, dtype=np.float32)
    transform = np.eye(4); transform[:3, 3] = [1., 2., 3.]
    return SimpleNamespace(directory=tmp_path, scene_spec={'objects': [{'id': 'A', 'position': [0, 0, 0], 'yaw': 0}],
        'grasp_caches': {'A': {'path': 'original'}}}, object_ids=['A'],
        objects=[SimpleNamespace(pose=SimpleNamespace(to_transformation_matrix=lambda: transform))],
        rows=[], unwrapped=SimpleNamespace(get_state=lambda: state)), state


def test_planning_view_refreshes_live_pose_without_mutating_source(tmp_path):
    rec, _ = fake_rec(tmp_path); original = copy.deepcopy(rec.scene_spec)
    view = PlanningView(rec, tmp_path / 'preview', {'A': {'path': 'full_cache'}})
    assert view.scene_spec['objects'][0]['position'] == [1., 2., 3.]
    view.scene_spec['grasp_caches']['A'] = {'path': 'selected'}
    assert rec.scene_spec == original


def test_preview_state_mutation_fails_closed_even_if_planner_errors(tmp_path, monkeypatch):
    from resimpair_pickplace import rolling_pickplace_v1 as module
    rec, state = fake_rec(tmp_path)
    def broken(view, one):
        state[0] += 1
        raise ValueError('planner_failed')
    monkeypatch.setattr(module.retained, 'prepare', broken)
    with pytest.raises(RuntimeError, match='changed_physical_state'):
        select_operation(rec, {'operations': []}, {'id': 'move', 'object': 'A'}, tmp_path / 'preview', rec.scene_spec['grasp_caches'])


def test_two_step_execution_preserves_append_only_evidence(tmp_path, monkeypatch):
    from resimpair_pickplace import rolling_pickplace_v1 as module, stability_motion
    from resimpair_grasp.common import binding, read, write
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import planning_protocols
    rec, _ = fake_rec(tmp_path)
    rec.scene_spec.update(supports=[], grasp_caches={'A': {}, 'B': {}})
    rec.object_ids = ['A', 'B']; rec.stage = 'initial'; calls = []
    rec.hold = lambda count: calls.append(count)
    value = task(); value['object_ids'] = ['A', 'B']
    value['requested_task_id'] = 'collect_group'
    for op in value['operations']: op['arm'] = 0
    def choose(rec, task, op, directory, caches):
        directory.mkdir(parents=True)
        write(directory / 'selection.json', {'selected': op['id']})
        return copy.deepcopy(op), caches[op['object']], {op['id']: [.2, .18]}, binding(directory / 'selection.json')
    def solve(rec, one, *args, **kwargs):
        rec.rows.append(one['operations'][0]['id'])
        rec.stage = 'final_settle'; rec.hold(40)
    monkeypatch.setattr(module, 'select_operation', choose)
    monkeypatch.setattr(module, 'targets', lambda *args: {})
    monkeypatch.setattr(stability_motion, 'solve', solve)
    monkeypatch.setattr(planning_protocols, 'pick_place', lambda scene, one, *args: {'operation': one['operations'][0]})
    module.execute(rec, value, release_policy='legacy_v1', motion_policy='transport_fallback_v2')
    assert calls == [40]
    assert len(list((tmp_path / 'operation_commitments').glob('[0-9][0-9].json'))) == 2
    assert [r['completed'] for r in read(tmp_path / 'executed_operations.json')['operations']] == [True, True]
    assert read(tmp_path / 'rolling_execution_summary.json')['physical_retries'] == 0
