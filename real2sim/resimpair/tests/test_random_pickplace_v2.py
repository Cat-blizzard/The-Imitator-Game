"""Zero-authority adapter boundaries, no simulations or model calls."""
import copy
from pathlib import Path

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import random_pickplace_v2 as adapter


def test_assigned_arm_does_not_change_task_but_other_goal_changes_do():
    intent = {'operations': [{'arm': None, 'target': [1, 2], 'object': 'box'}]}
    task = copy.deepcopy(intent); task['operations'][0]['arm'] = 1
    assert adapter.same_intent(task, intent)
    task['operations'][0]['target'] = [3, 4]
    assert not adapter.same_intent(task, intent)


def test_budget_inherits_26_without_granting_any_new_attempt(tmp_path):
    refs = {}
    for name, count in [('v1_comparison', 14), ('expanded_factory_integration', 6)]:
        path = tmp_path / name / 'budget/limits.json'
        write(path, {'physical_attempt_limit': count, 'api_limit': 0, 'gpu_jobs': 0}); refs[name] = binding(path)
    latest = tmp_path / 'v2_regression/budget/limits.json'
    write(latest, {'physical_attempt_limit': 6, 'api_limit': 0, 'gpu_jobs': 0,
        'per_attempt_timeout_seconds': 240, 'parent_comparison_budget': refs['v1_comparison'],
        'allocation_this_turn': {'v1_comparison': 14, 'expanded_factory_integration': 6, 'v2_regression': 6},
        'this_turn_cumulative_scope_limit': 26})
    parent = tmp_path / 'controller'
    write(parent / 'experiment.json', {'budget': str(tmp_path / 'expanded_factory_integration/budget')})
    budget = adapter.zero_budget(latest, parent)
    assert budget['physical_attempt_limit'] == 0 and budget['launch_enabled'] is False
    assert budget['scope_cumulative_attempt_limit'] == 26
    assert sum(budget['existing_scope_allocation'].values()) == 26
    assert budget['unspent_old_allowance_not_reassigned'] is True


def test_run_blocks_before_old_supervisor_or_any_reservation(tmp_path, monkeypatch):
    from resimpair_pickplace import runner
    root = tmp_path / 'campaign'
    write(root / 'budget/limits.json', {'physical_attempt_limit': 0, 'launch_enabled': False})
    write(root / 'candidate/frozen_inputs.json', {'budget': binding(root / 'budget/limits.json')})
    monkeypatch.setattr(runner, 'verify', lambda _: {})
    monkeypatch.setattr(adapter.execution, 'run', lambda *_: pytest.fail('must not dispatch'))
    with pytest.raises(ValueError, match='zero_physical_authorization'):
        adapter.run(root, 1)
    assert not (root / 'budget/reservations').exists()


def test_v3_cannot_be_relabelled_as_v5(tmp_path):
    write(tmp_path / 'protocol.json', {'schema': 'expanded_compatible_asset_sampling_v3'})
    with pytest.raises(ValueError, match='actual_v5'):
        adapter.factory_chain(tmp_path)
