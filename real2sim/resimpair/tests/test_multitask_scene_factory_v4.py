import copy

import pytest

from resimpair_grasp.common import ROOT, read, resolve
from resimpair_tasks import multitask_scene_factory_v4 as factory


def test_selection_uses_eligible_count_then_original_order_and_keeps_minimum():
    proposals = [{'proposal': 3, 'qualified_task_count': 7},
                 {'proposal': 8, 'qualified_task_count': 7},
                 {'proposal': 1, 'qualified_task_count': 6}]
    assert factory.select_existing_layout(proposals)['proposal'] == 3
    assert not factory.select_existing_layout([{'proposal': 0, 'qualified_task_count': 2}])['scene_admitted']
    assert factory.select_existing_layout([{'proposal': 0, 'qualified_task_count': 3}])['scene_admitted']
    with pytest.raises(ValueError, match='unique'):
        factory.select_existing_layout(proposals + [proposals[0]])


def test_missing_operation_or_mismatched_object_cannot_be_admitted():
    request = {'tasks': [{'id': 'pair', 'placements': [{'object': 'a'}, {'object': 'b'}]}]}
    rows = [{'task': 'pair', 'operation': 'move_a', 'object': 'a', 'passed': True, 'report': {}},
            {'task': 'pair', 'operation': 'move_b', 'object': 'b', 'passed': False, 'report': {}}]
    report = factory.summarize_proof(request, {'operations': rows})
    assert report['pair']['endpoint_passed'] is False
    assert report['pair']['failed_operations'] == ['move_b']
    with pytest.raises(ValueError, match='coverage'):
        factory.summarize_proof(request, {'operations': rows[:1]})
    wrong = copy.deepcopy(rows); wrong[1]['object'] = 'a'
    with pytest.raises(ValueError, match='identity'):
        factory.summarize_proof(request, {'operations': wrong})


def test_real_frozen_v3_admission_build_preserves80_requests_and_runs_no_IK(tmp_path, monkeypatch):
    source = ROOT / 'runs/multitask_factory_20261030_v3'
    if not (source / 'status.json').exists(): pytest.skip('frozen actual endpoint fixture unavailable')
    from resimpair_pickplace.cartesian_reachability import ArmSelector

    def forbidden(*args, **kwargs): raise AssertionError('new IK call forbidden in evidence-only admission')

    monkeypatch.setattr(ArmSelector, '__init__', forbidden)
    monkeypatch.setattr(factory.retained, 'endpoint_qualification', forbidden)
    result = factory.build(source, tmp_path / 'factory')
    assert result['requested_scenes'] == 10 and result['requested_tasks'] == 80
    assert result['admitted_scenes'] == 10 and result['eligible_tasks'] == 75
    assert result['ineligible_tasks'] == 5
    assert result['new_endpoint_IK_calls'] == result['new_layout_proposals'] == result['physical_executions'] == 0
    for row in read(tmp_path / 'factory/scenes.json')['scenes']:
        requested = read(resolve(row['request']))['tasks']
        admission = read(resolve(row['task_admission']))
        assert len(requested) == len(admission['tasks']) == 8
        assert 3 <= admission['eligible_task_count'] <= 8
        for decision in admission['tasks'].values():
            resolve(decision['endpoint_qualification'])
            assert decision['eligible'] or decision['failed_operations']
    with pytest.raises(ValueError, match='preserve_existing'):
        factory.build(source, tmp_path / 'factory')
