"""Bounded submission and provenance regressions; no physical execution."""
from pathlib import Path

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import multitask_campaign_v1 as campaign


def test_new_allocation_preserves_historical_unused_allowance():
    prior = campaign.prior_lineage()
    assert prior['prior_physical_limit'] == 34
    assert prior['prior_observed_attempts'] == 32
    assert prior['unspent_old_allowance_not_reassigned']


def test_80_candidate_plus_8_baseline_is_a_shared_ceiling():
    campaign.validate_counts(80, 8, 10)
    for values in [(81, 8, 10), (80, 9, 10), (80, 8, 11), (0, 8, 10)]:
        with pytest.raises(ValueError): campaign.validate_counts(*values)
    record = {'allocation_id': campaign.ALLOCATION_ID, 'allocation_limit': 88,
              'physical_attempt_limit': 88, 'root': '/different/root'}
    assert campaign.remaining_allocation([record]) == 0
    with pytest.raises(ValueError, match='must_not_reset'):
        campaign.remaining_allocation([{**record, 'allocation_limit': 89}])


def test_registration_cannot_restart_budget_with_new_directory(tmp_path, monkeypatch):
    history = {'prior_physical_limit': 34, 'prior_observed_attempts': 32}
    monkeypatch.setattr(campaign, 'prior_lineage', lambda _: history)
    registry = tmp_path / 'registry'
    receipt, budget = campaign.register_budget(tmp_path / 'first', tmp_path, 80, 8, 10, registry)
    assert read(Path(receipt['path']))['budget']['sha256'] == binding(tmp_path / 'first/budget/limits.json')['sha256']
    assert budget['cumulative_scope_limit'] == 122
    with pytest.raises(ValueError, match='allocation_reserved'):
        campaign.register_budget(tmp_path / 'second', tmp_path, 80, 8, 10, registry)
    assert not (tmp_path / 'second/budget/limits.json').exists()


def test_failed_factory_scenes_keep_all_requested_task_rows(tmp_path):
    task_ids = ['single', 'collect_group', 'clear_then_place']
    row = {'scene_id': 'fixed_rejected', 'partition': 'development', 'source_ids': ['fixed_source'],
        'requested_tasks': 3, 'task_ids': task_ids, 'accepted_geometry': False,
        'request': None, 'compiled_scene': None, 'compiled_tasks': None, 'task_metadata': None,
        'failure': {'reason': 'no_compatible_layout'}}
    write(tmp_path / 'scenes.json', {'scenes': [row]})
    scenes, cases = campaign.factory_inputs(tmp_path)
    assert len(scenes) == 1 and len(cases) == 3
    assert [c['short'] for c in cases] == task_ids
    assert all(c['row']['failure'] == row['failure'] for c in cases)


def test_provisional_failure_export_arm_does_not_change_target_or_input():
    scene = {'robot_roots': [[-.6, -.35, 0.], [-.6, .35, 0.]],
             'objects': [{'id': 'a', 'position': [-.3, .3, .03]}]}
    task = {'operations': [{'id': 'transfer_0', 'object': 'a', 'arm': None,
                           'target': {'support': 'table', 'offset_xy': [.1, -.1]}}]}
    assigned = campaign.provisional_task(task, scene)
    assert assigned['operations'][0]['arm'] == 1
    assert task['operations'][0]['arm'] is None
    assert assigned['operations'][0]['target'] == task['operations'][0]['target']


def test_unknown_reservation_stops_before_subprocess(tmp_path, monkeypatch):
    budget = {'candidate_attempt_limit': 80, 'baseline_attempt_limit': 8}
    monkeypatch.setattr(campaign, 'verify', lambda root: budget)
    monkeypatch.setattr(campaign, 'status', lambda root: {'unknown_reserved': 1})
    monkeypatch.setattr(campaign, 'REGISTRY', tmp_path / 'registry')
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **k: pytest.fail('No duplicate submission'))
    with pytest.raises(ValueError, match='recover_without_repeat'):
        campaign.run(tmp_path, 'candidate', 1)


def test_independent_protocol_requires_structured_fresh_evidence(tmp_path):
    evidence = tmp_path / 'cohort.json'; write(evidence, {'cases': []})
    ref = binding(evidence)
    campaign.verify_independent_bindings({'cohort': ref}, [evidence])
    with pytest.raises(ValueError, match='must_bind'):
        campaign.verify_independent_bindings({'comment': ref['sha256']}, [evidence])
    evidence.write_text('{"cases": ["changed"]}')
    with pytest.raises(ValueError, match='hash changed'):
        campaign.verify_independent_bindings({'cohort': ref}, [evidence])
