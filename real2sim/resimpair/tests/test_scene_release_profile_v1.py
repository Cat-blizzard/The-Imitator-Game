"""Prevent the observed supported-only/low-drop composition failure before execution."""
from copy import deepcopy
from pathlib import Path

import pytest
from resimpair_grasp.common import ROOT, binding, read, write
from resimpair_scene.execution import validate_release_profile
from resimpair_scene.release_variants import apply_release_variant

SOURCE = ROOT / 'runs/scene_tools_20261031_v1/baseline_cases.json'


def candidate(tmp_path):
    cases = read(SOURCE)
    if isinstance(cases, dict): cases = cases['cases']
    case = deepcopy(cases[0]); task = read(ROOT / case['task']['path'])
    for op in task['operations']:
        op['release'].update(mode='drop_from_above', allow_drop=True, fragile=False,
            bottom_above_rim_m=.03, hand_above_rim_m=.01, maximum_fall_m=.15)
    path = tmp_path / 'task.json'; write(path, task); case['task'] = binding(path)
    return case, task


@pytest.mark.skipif(not SOURCE.exists(), reason='retained task absent')
@pytest.mark.parametrize('profile', ['checked_withdrawal', 'contact_latch', 'conditional_descent'])
def test_supported_only_profile_rejects_actual_drop_task_before_submission(tmp_path, profile):
    case, _ = candidate(tmp_path)
    with pytest.raises(ValueError, match='explicit_drop_requires_task_release_profile'):
        validate_release_profile([case], profile)


@pytest.mark.skipif(not SOURCE.exists(), reason='retained task absent')
def test_retained_drop_profile_keeps_original_contract_and_rejects_implicit_permission(tmp_path):
    case, task = candidate(tmp_path)
    validate_release_profile([case], 'task_release')
    assert task['placement_contract']['xy_tolerance_m'] == .015
    task.pop('placement_contract')
    path = tmp_path / 'without_contract.json'; write(path, task); case['task'] = binding(path)
    with pytest.raises(ValueError, match='frozen_supported_or_low_drop_contract'):
        validate_release_profile([case], 'task_release')


@pytest.mark.skipif(not SOURCE.exists(), reason='retained task absent')
def test_revision_changes_only_release_policy_and_preserves_all_final_goals(tmp_path):
    cases = read(SOURCE)
    if isinstance(cases, dict): cases = cases['cases']
    parent = cases[0]; original = read(ROOT / parent['task']['path'])
    result = apply_release_variant(parent, ['transfer_6'], tmp_path / 'revision')
    assert result == apply_release_variant(parent, ['transfer_6'], tmp_path / 'revision')
    revised = read(Path(result['case']['task']['path']))
    assert result['case']['scene'] == parent['scene']
    revised['id'] = original['id']
    revised['operations'][-1]['release']['mode'] = original['operations'][-1]['release']['mode']
    revised['operations'][-1]['release']['allow_drop'] = original['operations'][-1]['release']['allow_drop']
    assert revised == original
    assert result['physical_success'] == 'unmeasured'
