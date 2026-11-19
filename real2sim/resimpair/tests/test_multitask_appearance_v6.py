from copy import deepcopy
import pytest
from resimpair_grasp.common import ROOT, read, digest, write
from resimpair_tasks import multitask_appearance_v5 as v5
from resimpair_tasks import multitask_appearance_v6 as v6

SOURCE = ROOT / ('runs/multitask_10x8_20261030_v3/candidate/executions/'
                 'multitask_scene_000_79ef2169_clear_then_place')


@pytest.fixture(scope='module')
def source():
    scene = read(SOURCE / 'effective_scene.json')
    return scene, v5.domain.build_motion_envelope(scene, SOURCE / 'measurements.npz')


def test_known_open_workspace_failure_is_local_to_optional_furniture(source):
    scene, envelope = source
    with pytest.raises(StopIteration):
        v5.apply_overlay(scene, 2026103102, preset='workshop_assembly_bay', motion_envelope=envelope)
    candidate, receipt = v6.apply_overlay(scene, 2026103102, preset='workshop_assembly_bay', motion_envelope=envelope)
    assert receipt['appearance_applied'] and receipt['furniture_status'] == 'incompatible_open_workspace'
    assert receipt['furniture_error']['missing'] == ['right_wall']
    assert receipt['effective_mode'] == 'workspace'
    assert candidate['scene_domain']['geometry'] is not None
    assert v5.domain.physical_scene(candidate) == v5.domain.physical_scene(scene)
    assert candidate['cameras'] == scene['cameras']
    assert receipt['R2H_ready'] is False


def test_compatible_measured_office_branch_is_exactly_preserved(source):
    scene, envelope = source
    before, _ = v5.apply_overlay(scene, 2026103001, motion_envelope=envelope)
    after, receipt = v6.apply_overlay(scene, 2026103001, motion_envelope=envelope)
    assert before == after
    assert receipt['effective_mode'] == 'metric_room' and receipt['furniture_status'] == 'prepared'
    assert receipt['result_scene_digest'] == digest(after)


def test_missing_furniture_preserves_successful_surfaces_and_room(source, tmp_path):
    scene, envelope = source
    result, receipt = v6.apply_overlay(scene, 2026103001, motion_envelope=envelope,
                                       furniture_catalog=tmp_path / 'missing_furniture.json')
    assert receipt['appearance_applied'] and receipt['effective_mode'] == 'workspace'
    assert receipt['furniture_status'] == 'unavailable_or_incompatible'
    assert result['scene_domain']['resources']['tabletop']['id'].startswith('polyhaven:')
    assert v5.domain.physical_scene(result) == v5.domain.physical_scene(scene)


def test_corruption_is_never_swallowed_as_furniture_failure(source, monkeypatch):
    scene, envelope = source
    original = v5.quality.apply_furniture
    def broken(prepared, **kwargs):
        result = original(prepared, **kwargs)
        result['objects'][0]['position'][0] += .1
        return result
    monkeypatch.setattr(v5.quality, 'apply_furniture', broken)
    with pytest.raises(RuntimeError, match='changed_physical_scene'):
        v6.apply_overlay(scene, 2026103001, motion_envelope=envelope)


def test_open_workspace_prepares_actual_renderer_instead_of_dead_end(source, tmp_path):
    prior = tmp_path / 'prior/budget/limits.json'
    write(prior, {'historical_actual_render_jobs': 76,
                  'maximum_lineage_render_jobs_including_history': 79})
    output = tmp_path / 'prepared'
    result = v6.prepare(SOURCE, output, seed=2026103102, preset='workshop_assembly_bay',
        prior_budget=prior, allocation_id='pytest_open_workspace', registry=tmp_path / 'registry')
    assert result['prepared_cases'] == 2
    assert result['optional_furniture_status'] == 'incompatible_open_workspace'
    assert result['effective_mode'] == 'workspace'
    protocol, limits = v5.render.verify(output)
    assert len(protocol['cases']) == 2 and limits['maximum_attempts'] == 2
    assert not list((output / 'budget/reservations').glob('*.json'))
