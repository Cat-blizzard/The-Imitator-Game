"""Activation boundaries; never start a physical worker."""
import copy

import pytest

from resimpair_grasp.common import binding, digest, read, write
from resimpair_tasks import prepared_batch_execution_v1 as activation


def fixture(tmp_path):
    prepared, root = tmp_path / 'prepared', tmp_path / 'execute'
    write(prepared / 'candidate/cohort.json', {'cases': []})
    refs = {}
    for name, count in [('v1_comparison', 14), ('expanded_factory_integration', 6), ('v2_regression', 6)]:
        path = tmp_path / name / 'budget/limits.json'
        write(path, {'physical_attempt_limit': count, 'api_limit': 0, 'gpu_jobs': 0})
        refs[name] = binding(path)
    budget = {'existing_scope_budget_references': refs,
        'existing_scope_allocation': {'v1_comparison': 14, 'expanded_factory_integration': 6, 'v2_regression': 6},
        'scope_cumulative_attempt_limit': 26}
    protocol = {'new_physical_attempt_limit': 2, 'new_request_limit': 2, 'expected_output_root': str(root),
        'api_limit': 0, 'gpu_jobs': 0, 'parallel_workers': 1, 'render_backend': 'none',
        'per_attempt_timeout_seconds': 240, 'prior_round_limit_unchanged': 26,
        'total_current_round_physical_limit_after_explicit_increment': 28}
    return prepared, root, protocol, budget, [{'id': 'relocate'}, {'id': 'contain'}]


def test_explicit_two_attempt_increment_preserves_26_ceiling(tmp_path):
    prepared, root, protocol, old_budget, cases = fixture(tmp_path)
    before = copy.deepcopy(old_budget)
    budget = activation.authorize(root, prepared, protocol, old_budget, cases, 2, [])
    assert budget['physical_attempt_limit'] == 2
    assert budget['prior_round_limit_unchanged'] == 26
    assert budget['scope_cumulative_attempt_limit'] == 28
    assert budget['existing_scope_reservations_observed'] == dict.fromkeys(old_budget['existing_scope_allocation'], 0)
    assert old_budget == before and budget['retries_per_request'] == 0


@pytest.mark.parametrize('limit', [0, -1, 3, 97, True])
def test_no_zero_unbounded_or_mismatched_authority(tmp_path, limit):
    prepared, root, protocol, budget, cases = fixture(tmp_path)
    with pytest.raises(ValueError, match='finite_one_attempt'):
        activation.authorize(root, prepared, protocol, budget, cases, limit, [])


def test_different_directory_does_not_reactivate_same_prepared_request(tmp_path):
    prepared, root, protocol, budget, cases = fixture(tmp_path)
    registered = [{'scope_key': 'old', 'source_prepared_cohort': binding(prepared / 'candidate/cohort.json'),
        'case_ids': ['relocate']}]
    with pytest.raises(ValueError, match='already_activated'):
        activation.authorize(root, prepared, protocol, budget, cases, 2, registered)


def test_previous_increment_cannot_be_ignored(tmp_path):
    prepared, root, protocol, budget, cases = fixture(tmp_path)
    registered = [{'scope_key': digest({'references': budget['existing_scope_budget_references'],
        'allocation': budget['existing_scope_allocation']}), 'source_prepared_cohort': {'path': 'different', 'sha256': 'different'},
        'case_ids': ['other'], 'physical_attempt_limit': 2, '_binding': {'path': 'receipt', 'sha256': 'receipt'}}]
    with pytest.raises(ValueError, match='all_prior_registered'):
        activation.authorize(root, prepared, protocol, budget, cases, 2, registered)
    protocol.update(prior_round_limit_unchanged=28, total_current_round_physical_limit_after_explicit_increment=30)
    result = activation.authorize(root, prepared, protocol, budget, cases, 2, registered)
    assert result['scope_cumulative_attempt_limit'] == 30


def test_same_case_scene_task_and_partition_are_preserved(tmp_path):
    prepared = tmp_path / 'prepared'
    write(tmp_path / 'request.json', {'source': 'original', 'target': [0.1, 0.2]})
    source = {'id': 'scene_relocate', 'source_request': binding(tmp_path / 'request.json'),
        'requested_task_id': 'relocate', 'source_ids': ['asset:a'], 'split_group': ['asset:a'],
        'partition': 'development', 'launch_enabled': False, 'experiment_role': 'prepared',
        'scene': binding(tmp_path / 'request.json'), 'task': binding(tmp_path / 'request.json')}
    cohort = {'cases': [source]}
    write(prepared / 'candidate/cohort.json', cohort)
    cohort_ref = binding(prepared / 'candidate/cohort.json')
    row = {'case_id': source['id'], 'source_prepared_case_index': 0, 'source_prepared_case': source,
        **{k: source[k] for k in ('source_request', 'requested_task_id', 'source_ids', 'split_group')},
        'dataset_partition': 'development', 'experiment_role': 'integration'}
    write(tmp_path / 'selection.json', {'source_cohort': cohort_ref, 'selected_before_execution': True, 'cases': [row]})
    protocol = {'source_prepared_cohort': cohort_ref, 'cases': binding(tmp_path / 'selection.json')}
    result = activation.select_cases(prepared, cohort, [source['id']], protocol)
    expected = copy.deepcopy(source); expected.update(launch_enabled=True, experiment_role='integration')
    assert result == [expected] and cohort['cases'][0]['launch_enabled'] is False
    changed = copy.deepcopy(cohort); changed['cases'][0]['split_group'] = ['asset:b']
    with pytest.raises(ValueError, match='payload_mismatch'):
        activation.select_cases(prepared, changed, [source['id']], protocol)
    with pytest.raises(ValueError, match='independently_frozen_selection'):
        activation.select_cases(prepared, cohort, ['different'], protocol)


def test_run_reuses_original_worker_supervisor_and_does_not_launch_on_invalid_limit(tmp_path, monkeypatch):
    write(tmp_path / 'budget/limits.json', {'physical_attempt_limit': 2})
    monkeypatch.setattr(activation, 'verify', lambda _: {})
    calls = []
    monkeypatch.setattr(activation.execution, 'run', lambda *args: calls.append(args) or {'new_attempts': 0})
    with pytest.raises(ValueError, match='bounded_launch_count'):
        activation.run(tmp_path, 3)
    assert not calls
    assert activation.run(tmp_path, 2) == {'new_attempts': 0}
    assert calls == [(tmp_path.resolve(), 'candidate', 2)]
    assert not (tmp_path / 'budget/reservations').exists()
