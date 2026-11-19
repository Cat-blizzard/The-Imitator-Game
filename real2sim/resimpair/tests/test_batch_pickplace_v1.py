"""Orchestration invariants; these tests do not count as physical evidence."""
from pathlib import Path

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_tasks import batch_pickplace_v1 as batch
from resimpair_tasks import spatial_reachability_v6 as layout


def test_profile_cannot_hide_more_requests_than_its_allocation():
    profile = read(batch.DEFAULT_PROFILE)
    batch.validate_profile(profile, 3, ['contain_selected', 'relocate_selected'], 6)
    for scenes, goals, budget in [(4, ['contain_selected', 'relocate_selected'], 8),
                                 (3, ['contain_selected', 'relocate_selected'], 5),
                                 (3, ['contain_selected', 'contain_selected'], 6)]:
        with pytest.raises(ValueError): batch.validate_profile(profile, scenes, goals, budget)


def test_changing_output_directory_does_not_reset_scope_allowance():
    profile = read(batch.DEFAULT_PROFILE)
    used = [{'allocation_id': profile['allocation_id'], 'allocation_limit': 6,
             'physical_attempt_limit': 6, 'root': '/old/root'}]
    assert batch.remaining_allocation(profile, used) == 0
    altered = dict(profile, allocation_physical_limit=12)
    with pytest.raises(ValueError, match='must_not_reset'):
        batch.remaining_allocation(altered, used)


def test_layout_count_expands_without_changing_per_scene_sampling_limit():
    layout.configuration(500, 500, ['contain_selected'], 16, 20261029, .9)
    for values in [(501, 500, ['contain_selected'], 16, 1, .9),
                   (500, 500, ['contain_selected'], 17, 1, .9),
                   (500, 500, ['contain_selected'], 16, 1, 1.0)]:
        with pytest.raises(ValueError): layout.configuration(*values)


def test_geometry_failed_sources_are_kept_without_replacement_or_IK(tmp_path, monkeypatch):
    source = tmp_path / 'factory'; output = tmp_path / 'layouts'
    write(source / 'index.json', {'entries': []}); write(source / 'containers.json', {'assets': []})
    write(source / 'protocol.json', {'index': binding(source / 'index.json'), 'containers': binding(source / 'containers.json')})
    rows = [{'scene_id': f'rejected_{i}', 'partition': 'development', 'source_ids': ['fixed_source_' + str(i)],
             'request': None, 'accepted_geometry': False, 'failure': {'reason': 'real_geometry_not_available'}}
            for i in range(8)]
    write(source / 'scenes.json', {'scenes': rows})
    monkeypatch.setattr(layout.retained, 'qualify', lambda *args: pytest.fail('Rejected source must not call IK'))
    report = layout.build(source, output, scenes=8, goals=['contain_selected', 'relocate_selected'], seed=1)
    saved = read(output / 'scenes.json')['scenes']
    assert len(saved) == 8
    assert [r['parent_scene_id'] for r in saved] == [r['scene_id'] for r in rows]
    assert report['exhausted_scenes'] == 8 and report['endpoint_eligible_goal_programs'] == 0
    assert report['requested_goal_programs'] == 16 and report['total_proposals'] == 0
    assert report['physical_attempts'] == report['api_calls'] == report['gpu_jobs'] == 0


def test_prior_actual_attempts_and_unused_allowance_are_separate():
    history = batch.historical_budget(batch.path(read(batch.DEFAULT_PROFILE)['parent_execution']))
    assert history['prior_physical_limit'] == 28
    assert history['prior_observed_attempts'] == 26
