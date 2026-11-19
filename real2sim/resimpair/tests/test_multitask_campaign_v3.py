"""Bounded CPU admission and caught-engineering-error regressions; no physics."""
import os
from pathlib import Path

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import multitask_campaign_v3 as campaign


def test_new_allocation_preserves_both_stopped_versions_and_unused_allowances():
    prior = campaign.prior_lineage()
    assert prior['prior_physical_limit'] == 210
    assert prior['prior_observed_attempts'] == 50
    assert prior['unspent_old_allowance_not_reassigned']
    assert prior['earlier_v1_actual_attempts'] == 2
    assert prior['stopped_v2_actual_attempts'] == 16
    assert prior['stopped_v2_unused_allowance_retained'] == 72
    assert prior['current_repair_round_actual_attempt_ceiling_including_v1_v2'] == 106


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
    history = {'prior_physical_limit': 210, 'prior_observed_attempts': 50}
    monkeypatch.setattr(campaign, 'prior_lineage', lambda _: history)
    registry = tmp_path / 'registry'
    receipt, budget = campaign.register_budget(tmp_path / 'first', tmp_path, 80, 8, 10, registry)
    assert read(Path(receipt['path']))['budget']['sha256'] == binding(tmp_path / 'first/budget/limits.json')['sha256']
    assert budget['cumulative_scope_limit'] == 298
    with pytest.raises(ValueError, match='allocation_reserved'):
        campaign.register_budget(tmp_path / 'second', tmp_path, 80, 8, 10, registry)
    assert not (tmp_path / 'second/budget/limits.json').exists()


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


def factory_with_task_admission(tmp_path, eligible):
    ids = ['relocate_selected', 'collect_group', 'clear_then_place', 'arrange_group']
    decisions = {k: {'eligible': ok, 'reason': None if ok else 'no_endpoint_ik'}
                 for k, ok in zip(ids, eligible)}
    write(tmp_path / 'request.json', {'tasks': [{'id': k, 'placements': [{'object': 'a'}]} for k in ids]})
    write(tmp_path / 'compiled_tasks.json', {'tasks': [{'requested_task_id': k} for k in ids]})
    write(tmp_path / 'task_admission.json', {'tasks': decisions, 'minimum_eligible_tasks_per_scene': 3,
        'scene_admitted': any(eligible)})
    row = {'scene_id': 'scene', 'partition': 'development', 'source_ids': ['source'],
        'requested_tasks': len(ids), 'task_ids': ids, 'accepted_geometry': True,
        'request': binding(tmp_path / 'request.json'), 'compiled_tasks': binding(tmp_path / 'compiled_tasks.json'),
        'task_admission': binding(tmp_path / 'task_admission.json'), 'task_metadata': None}
    write(tmp_path / 'scenes.json', {'scenes': [row]})
    return ids, decisions


def test_per_task_admission_preserves_rejected_request_without_dropping_valid_scene(tmp_path):
    ids, decisions = factory_with_task_admission(tmp_path, [True, True, False, True])
    scenes, cases = campaign.factory_inputs(tmp_path)
    assert len(scenes) == 1 and len(cases) == 4
    assert [c['short'] for c in cases] == ids
    assert [c['eligible'] for c in cases] == [True, True, False, True]
    assert cases[2]['preflight_reason'] == 'factory_task_endpoint_rejected:no_endpoint_ik'
    assert cases[2]['task_admission_decision'] == decisions['clear_then_place']
    assert cases[2]['task'] == {'requested_task_id': 'clear_then_place'}


def test_per_task_admission_is_not_permission_to_ignore_shared_geometry_failure(tmp_path):
    factory_with_task_admission(tmp_path, [True, True, False, True])
    p = tmp_path / 'scenes.json'; data = read(p)
    data['scenes'][0].update(accepted_geometry=False, failure='scene_collision')
    import json
    p.write_text(json.dumps(data))
    _, cases = campaign.factory_inputs(tmp_path)
    assert not any(c['eligible'] for c in cases)
    assert all(c['preflight_reason'] == 'factory_geometry_rejected:scene_collision' for c in cases)


def test_deferred_scene_keeps_all_requests_with_explicit_task_reasons(tmp_path):
    ids, _ = factory_with_task_admission(tmp_path, [False] * 4)
    _, cases = campaign.factory_inputs(tmp_path)
    assert len(cases) == len(ids)
    assert not any(c['eligible'] for c in cases)
    assert all(c['preflight_reason'] for c in cases)


def test_partial_scene_requires_at_least_three_explicitly_eligible_tasks(tmp_path):
    factory_with_task_admission(tmp_path, [True, True, False, False])
    with pytest.raises(ValueError, match='at_least_three'):
        campaign.factory_inputs(tmp_path)


@pytest.mark.parametrize('failure', [
    *({'type': kind, 'message': 'caught'} for kind in campaign.ENGINEERING_FAILURE_TYPES),
    *({'type': 'ValueError', 'message': marker} for marker in campaign.ENGINEERING_FAILURE_MARKERS),
    {'type': 'ValueError', 'message': 'Refusing overwrite: frozen/result.json'},
    {'type': 'RuntimeError', 'message': 'planning_preview_changed_physical_state: op0'},
])
def test_caught_engineering_failures_stop_even_with_zero_exit(tmp_path, failure):
    write(tmp_path / 'worker_result.json', {'failure': failure, 'bundle': None})
    result = campaign.admission_check(tmp_path, 0)
    assert result['stop_admission']
    assert result['reasons'] == ['caught_engineering_failure']


@pytest.mark.parametrize('message', [
    'no_ready_operation_has_complete_path',
    'no_complete_pick_place_proposal:unreachable',
    'lost_grasp_after_lift',
])
def test_expected_planning_failures_remain_failures_without_stopping_campaign(tmp_path, message):
    write(tmp_path / 'worker_result.json', {'failure': {'type': 'RuntimeError', 'message': message}, 'bundle': None})
    result = campaign.admission_check(tmp_path, 0)
    assert result['stop_admission'] is False
    assert result['reported_failure']['message'] == message


@pytest.mark.parametrize('filename', [
    'rolling_selection/00_transfer_0/whole_path_selection/transfer_0/selection.json',
    'rolling_decisions.json',
])
@pytest.mark.parametrize('message', [
    'target_has_no_declared_goal', 'preview_changed_physical_state',
    'geometry_preview_changed_physical_state', 'Refusing to overwrite evidence: existing.json',
])
def test_nested_engineering_reason_is_not_hidden_by_generic_planning_failure(tmp_path, filename, message):
    write(tmp_path / 'worker_result.json', {'failure': {'type': 'RuntimeError', 'message': 'no_ready_operation_has_complete_path'}})
    write(tmp_path / filename, {'decisions': [{'passed': False, 'reason': message}]})
    result = campaign.admission_check(tmp_path, 0)
    assert result['stop_admission']
    assert result['reasons'] == ['caught_nested_engineering_failure']
    assert result['nested_planning_evidence']['matches'][0]['evidence'] == binding(tmp_path / filename)


def test_realistic_nested_collision_ik_and_no_complete_failures_do_not_stop(tmp_path):
    current = tmp_path / 'current'
    write(current / 'worker_result.json', {'failure': {'type': 'RuntimeError', 'message': 'no_ready_operation_has_complete_path'}})
    write(current / 'rolling_decisions.json', {'rows': [{'passed': False, 'reason': 'no_complete_pick_place_proposal:transfer_0'}]})
    write(current / 'whole_path_selection/transfer_0/selection.json', {'decisions': [
        {'passed': False, 'reason': 'IK_failed'}, {'passed': False, 'reason': 'payload_collision'}]})
    write(tmp_path / 'other_attempt/rolling_decisions.json', {'rows': [{'reason': 'target_has_no_declared_goal'}]})
    result = campaign.admission_check(current, 0)
    assert result['stop_admission'] is False
    assert len(result['nested_planning_evidence']['checked_files']) == 2


def test_corrupt_nested_selection_cannot_be_silently_ignored(tmp_path):
    write(tmp_path / 'worker_result.json', {'failure': None})
    (tmp_path / 'selection.json').write_text('{broken')
    result = campaign.admission_check(tmp_path, 0)
    assert result['stop_admission']
    assert result['reasons'] == ['nested_planning_evidence_invalid']


@pytest.mark.parametrize('payload', [None, '{broken', '{}', '{"failure":"invalid"}'])
def test_missing_or_malformed_worker_result_cannot_pass_as_ordinary_failure(tmp_path, payload):
    if payload is not None: (tmp_path / 'worker_result.json').write_text(payload)
    assert campaign.admission_check(tmp_path, 0)['stop_admission']


def test_zero_exit_engineering_error_terminates_admission_and_preserves_receipt(tmp_path, monkeypatch):
    budget = {'candidate_attempt_limit': 2, 'baseline_attempt_limit': 1, 'physical_attempt_limit': 3}
    monkeypatch.setattr(campaign, 'verify', lambda root: budget)
    monkeypatch.setattr(campaign, 'status', lambda root: {'unknown_reserved': 0})
    monkeypatch.setattr(campaign, 'REGISTRY', tmp_path / 'registry')
    cases = [{'id': name, 'scene_id': 'scene', 'eligible': True} for name in ('bad', 'pending')]
    write(tmp_path / 'candidate/cohort.json', {'cases': cases})
    write(tmp_path / 'baseline/cohort.json', {'cases': [{'id': 'baseline', 'scene_id': 'scene', 'eligible': True}]})
    write(tmp_path / 'candidate/protocol.json', {'schema': 'test'})
    write(tmp_path / 'independent_protocol.json', {'schema': 'test'})
    calls = []

    class CaughtFailureWorker:
        pid = os.getpid()

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            assert argv[-1] == 'bad'
            assert campaign.WORKER_MODULE in argv

        def wait(self, **kwargs):
            write(tmp_path / 'candidate/executions/bad/worker_result.json', {
                'failure': {'type': 'ValueError', 'message': 'target_has_no_declared_goal'}, 'bundle': None})
            return 0

    monkeypatch.setattr(campaign.subprocess, 'Popen', CaughtFailureWorker)
    result = campaign.run(tmp_path, 'candidate', 2)
    assert len(calls) == result['new_attempts'] == result['resolved_requests'] == 1
    receipt = read(tmp_path / 'admission_stop.json')
    assert receipt['actual_attempts'] == 1
    assert receipt['unused_allocation_not_reassigned'] == 2
    assert receipt['remaining_candidate_requests_not_submitted'] == 1
    assert receipt['remaining_baseline_requests_not_submitted'] == 1
    assert receipt['repeat_failed_requests_in_this_root'] is False
    assert receipt['classification']['stop_admission'] is True
    terminal = read(tmp_path / 'candidate/executions/bad/terminal.json')
    assert terminal['returncode'] == 0
    assert terminal['physical_attempts'] == 1
    for arm in ('candidate', 'baseline'):
        with pytest.raises(ValueError, match='campaign_admission_stopped'):
            campaign.run(tmp_path, arm, 1)
    assert len(calls) == 1


@pytest.mark.parametrize('returncode', [1, 'timeout'])
def test_process_failure_stops_even_when_result_reports_no_exception(tmp_path, returncode):
    write(tmp_path / 'worker_result.json', {'failure': None, 'bundle': None})
    assert campaign.admission_check(tmp_path, returncode)['stop_admission']
