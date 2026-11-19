"""Real saved-source invariants plus explicit optional-resource failure cases."""
from copy import deepcopy
from pathlib import Path

import pytest

from resimpair_grasp.common import ROOT, digest, read, write
from resimpair_tasks import multitask_appearance_v5 as appearance
from resimpair_tasks import scene_domain_v2 as domain

SOURCES = (
    'multitask_scene_000_79ef2169_clear_then_place',
    'multitask_scene_007_381e05b9_collect_group',
)
EXECUTIONS = ROOT / 'runs/multitask_10x8_20261030_v3/candidate/executions'


@pytest.fixture(scope='module')
def source():
    directory = EXECUTIONS / SOURCES[0]
    scene = read(directory / 'effective_scene.json')
    envelope = domain.build_motion_envelope(scene, directory / 'measurements.npz')
    return scene, envelope


def test_actual_multitask_geometry_actions_and_camera_are_immutable(source):
    scene, envelope = source; before = deepcopy(scene)
    candidate, receipt = appearance.apply_overlay(scene, 2026103001, motion_envelope=envelope)
    assert scene == before
    assert receipt['appearance_applied']
    assert receipt['new_physical_executions'] == receipt['API_calls'] == 0
    assert receipt['R2H_ready'] is False
    assert domain.physical_scene(candidate) == domain.physical_scene(scene)
    assert candidate['cameras'] == scene['cameras']
    assert candidate['objects'] == scene['objects'] and candidate['supports'] == scene['supports']
    assert receipt['physical_scene_digest'] == digest(domain.physical_scene(candidate))
    assert receipt['furniture_ids'] == ['wooden_display_shelves_01']
    assert candidate['scene_domain']['geometry']['furniture'][0]['scale'] == [1., 1., 1.]
    assert all(e is None or e['id'].startswith('polyhaven:')
               for e in candidate['scene_domain']['resources'].values())


@pytest.mark.parametrize('mode', ['catalog_only', 'workspace', 'metric_room'])
def test_optional_modes_are_deterministic_without_physics(source, mode):
    scene, envelope = source
    kw = dict(mode=mode, motion_envelope=envelope)
    a, ar = appearance.apply_overlay(scene, 2026103001, **kw)
    b, br = appearance.apply_overlay(scene, 2026103001, **kw)
    assert a == b and ar == br and ar['appearance_applied']
    if mode == 'catalog_only': assert a['scene_domain']['geometry'] is None


def test_missing_background_preserves_source_and_never_claims_quality(source, tmp_path):
    scene, envelope = source
    result, receipt = appearance.apply_overlay(scene, 1, motion_envelope=envelope,
                                               catalog=tmp_path / 'missing.json')
    assert result == scene and result is not scene
    assert receipt['status'] == 'visual_degraded_original_retained'
    assert receipt['appearance_applied'] is receipt['R2H_ready'] is False
    assert receipt['physical_success_label_modified'] is False
    assert receipt['error']['type'] == 'FileNotFoundError'


def test_missing_furniture_does_not_leave_a_partial_overlay(source, tmp_path):
    scene, envelope = source
    result, receipt = appearance.apply_overlay(scene, 2026103001, motion_envelope=envelope,
                                      furniture_catalog=tmp_path / 'missing_furniture.json')
    assert result == scene and 'scene_domain' not in result
    assert receipt['status'] == 'visual_degraded_original_retained'


def test_disabled_layer_needs_no_catalog_or_motion_envelope(source, tmp_path):
    scene, _ = source
    result, receipt = appearance.apply_overlay(scene, 1, mode='disabled',
                                               catalog=tmp_path / 'nonexistent.json')
    assert result == scene and receipt['status'] == 'disabled'
    assert not receipt['appearance_applied']


def test_strict_visual_resource_mode_reports_failure(source, tmp_path):
    scene, envelope = source
    with pytest.raises(FileNotFoundError):
        appearance.apply_overlay(scene, 1, motion_envelope=envelope,
                                 catalog=tmp_path / 'missing.json', allow_visual_fallback=False)


def test_physical_or_camera_mutation_is_not_a_fallback(source, monkeypatch):
    scene, envelope = source
    original = appearance.quality.apply_furniture
    def corrupted(prepared, **kwargs):
        candidate = original(prepared, **kwargs)
        candidate['objects'][0]['position'][0] += .1
        return candidate
    monkeypatch.setattr(appearance.quality, 'apply_furniture', corrupted)
    with pytest.raises(RuntimeError, match='changed_physical_scene'):
        appearance.apply_overlay(scene, 2026103001, motion_envelope=envelope)
    changed = deepcopy(scene); changed['cameras']['robot_side']['eye'][0] += .1
    with pytest.raises(RuntimeError, match='changed_source_cameras'):
        appearance.assert_invariants(scene, changed)


def test_lack_of_motion_evidence_is_explicit_not_silently_ignored(source):
    scene, _ = source
    with pytest.raises(ValueError, match='actual_motion_envelope'):
        appearance.apply_overlay(scene, 1)


def test_budget_extends_lineage_and_does_not_reset_by_changing_output(tmp_path):
    prior = tmp_path / 'prior/budget/limits.json'
    write(prior, {'historical_actual_render_jobs': 74,
                  'maximum_lineage_render_jobs_including_history': 77})
    cases = [{'id': 'baseline', 'stage': 'screen'}, {'id': 'candidate', 'stage': 'screen'}]
    _, limits = appearance._register_budget(tmp_path / 'output_a', prior,
                                             'fixed_scope', cases, registry=tmp_path / 'registry')
    assert limits['historical_actual_render_jobs'] == 74
    assert limits['maximum_lineage_render_jobs_including_history'] == 79
    assert limits['maximum_actual_jobs_including_history'] == 76
    assert limits['maximum_attempts'] == limits['allocation']['screen'] == 2
    with pytest.raises(ValueError, match='already_registered'):
        appearance._register_budget(tmp_path / 'output_b', prior,
                                    'fixed_scope', cases, registry=tmp_path / 'registry')


def test_real_two_case_renderer_protocol_is_prepared_without_launch(source, tmp_path):
    prior = tmp_path / 'prior/budget/limits.json'
    write(prior, {'historical_actual_render_jobs': 74,
                  'maximum_lineage_render_jobs_including_history': 77})
    output = tmp_path / 'prepared'
    receipt = appearance.prepare(EXECUTIONS / SOURCES[0], output, seed=2026103001,
        prior_budget=prior, allocation_id='pytest_only', registry=tmp_path / 'registry')
    assert receipt['prepared_cases'] == 2 and receipt['render_jobs_started'] == 0
    protocol, limits = appearance.render.verify(output)
    assert [c['variant'] for c in protocol['cases']] == ['baseline', 'environment_3d']
    assert len(protocol['sources']) == 1
    assert all(len(c['state_indices']) == 3 for c in protocol['cases'])
    assert not list((output / 'budget/reservations').glob('*.json'))
