"""Selection and resource boundaries only; no physical executions or API calls."""
import subprocess
import sys

import pytest

from resimpair_grasp.common import ROOT, binding, write
from resimpair_tasks import random_pickplace_v1 as batch


@pytest.fixture
def factory(tmp_path):
    rows = []
    for i, partition in enumerate(['development', 'heldout_asset', 'development']):
        goals = ['relocate_selected', 'contain_selected'] + (['collect_pair'] if i == 2 else [])
        folder = tmp_path / 'scenes' / f'scene{i}'
        write(folder / 'request.json', {'tasks': [{'id': goal} for goal in goals]})
        write(folder / 'compiled_scene.json', {'fixture': i})
        write(folder / 'compiled_tasks.json', {'fixture': i})
        rows.append({'scene_id': f'scene{i}', 'partition': partition, 'seed': 100+i,
            'source_ids': [f'source{i}'], 'request': binding(folder / 'request.json'),
            'accepted_geometry': True, 'failure': None})
    write(tmp_path / 'scenes.json', {'scenes': rows})
    return tmp_path


def test_pilot_selects_exact_six_before_IK_or_physics(factory):
    cases = batch.select_cases(factory, selection='coverage-pilot')
    assert [c['case_id'] for c in cases] == [
        'scene0_relocate_selected', 'scene0_contain_selected',
        'scene1_relocate_selected', 'scene1_contain_selected',
        'scene2_contain_selected', 'scene2_collect_pair']
    assert cases[2]['dataset_partition'] == 'heldout_asset'
    assert cases[2]['experiment_role'] == 'new_asset_reserved'
    assert cases[-1]['source_ids'] == ['source2']


def test_requested_subset_retains_order_and_source_binding(factory):
    cases = batch.select_cases(factory, scenes=2, goals=['contain_selected'])
    assert [c['case_id'] for c in cases] == ['scene0_contain_selected', 'scene1_contain_selected']
    assert all(c['source_request']['sha256'] for c in cases)


@pytest.mark.parametrize('kwargs', [
    {'scenes': 4}, {'scenes': 0}, {'goals': []},
    {'goals': ['contain_selected', 'contain_selected']}, {'goals': ['nonexistent']},
])
def test_invalid_selection_does_not_create_a_campaign(factory, kwargs):
    with pytest.raises(ValueError):
        batch.select_cases(factory, **kwargs)
    assert not (factory / 'budget').exists()


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / 'parent'
    write(root / 'budget/limits.json', {'physical_attempt_limit': 14, 'api_limit': 0,
        'gpu_jobs': 0, 'per_attempt_timeout_seconds': 240})
    write(root / 'budget/reservations/0000.json', {'known_or_unknown': 'both count'})
    return root


def test_new_six_inherits_parent_fourteen_without_reset(parent):
    limits = batch.budget_spec(parent, 6, 6)
    assert limits['physical_attempt_limit'] == 6
    assert limits['round_cumulative_attempt_limit_including_parent'] == 20
    assert limits['parent_reservations_observed_at_prepare'] == 1
    assert limits['parent_budget'] == binding(parent / 'budget/limits.json')
    assert limits['api_limit'] == limits['gpu_jobs'] == 0
    assert not (parent / 'candidate').exists()


@pytest.mark.parametrize('limit,count', [(0, 0), (97, 6), (5, 6), (6, 0)])
def test_invalid_budget_fails_before_reservation(parent, limit, count):
    with pytest.raises(ValueError, match='finite_budget'):
        batch.budget_spec(parent, limit, count)
    assert len(list((parent / 'budget/reservations').glob('*.json'))) == 1


def test_run_delegates_original_supervisor_without_new_controller(parent, tmp_path, monkeypatch):
    from resimpair_pickplace import runner, whole_path_motion_v1
    from resimpair_tasks import whole_path_campaign_v1
    root = tmp_path / 'new'; limits = batch.budget_spec(parent, 6, 6)
    write(root / 'budget/limits.json', limits); write(root / 'rules.json', {'fixture': True})
    write(root / 'candidate/protocol.json', {'batch_driver': binding(batch.__file__),
        'original_rules': binding(root / 'rules.json'), 'shared_budget': binding(root / 'budget/limits.json')})
    monkeypatch.setattr(runner, 'verify', lambda _: {'schema': batch.VERSION,
        'candidate_motion_override': whole_path_motion_v1.VERSION})
    seen = []
    monkeypatch.setattr(whole_path_campaign_v1, 'run', lambda *args: seen.append(args) or {'fixture': 'delegated'})
    assert batch.run(root, 6) == {'fixture': 'delegated'}
    assert seen == [(root.resolve(), 'candidate', 6)]
    with pytest.raises(ValueError, match='bounded_launch_count'):
        batch.run(root, 7)
    assert len(seen) == 1


def test_repository_CLI_exposes_existing_prepare_and_run_without_work():
    result = subprocess.run([sys.executable, 'resimpair.py', 'random-pickplace', 'prepare', '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    for expected in ['--factory', '--physical-limit', '--goals', '--scenes']:
        assert expected in result.stdout
