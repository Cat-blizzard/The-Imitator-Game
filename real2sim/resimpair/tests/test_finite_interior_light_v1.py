from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from resimpair_grasp.common import ROOT, read, digest
from resimpair_tasks.finite_interior_light_v1 import box_ray, diagnose, prepare

PARENT = ROOT / 'runs/metric_group_functional_rooms_20261030_v1/cases'


def test_finite_segment_stops_before_opaque_ceiling():
    box = {'id': 'ceiling', 'center_m': [0, 0, 2.09], 'size_m': [4, 4, .08]}
    assert box_ray([0, 0, .12], [0, 0, 1], box)['entry_distance_m'] == pytest.approx(1.93)
    assert box_ray([0, 0, .12], [0, 0, 1], box, 1.87) is None
    assert box_ray([3, 0, .12], [0, 0, 1], box) is None


@pytest.mark.parametrize('folder', [
    'screen_00_metric_group_000_collect_all_office_candidate',
    'screen_01_metric_group_000_distribute_groups_dining_candidate'])
def test_actual_previously_rendered_rooms_only_change_finite_key(folder):
    scene = read(PARENT / folder / 'render_scene.json')
    before = digest(scene)
    result, diagnosis = prepare(scene)
    assert digest(scene) == before
    assert diagnosis['infinite_structure_blocked_count'] == 64
    assert diagnosis['finite_structure_blocked_count'] == 0
    assert diagnosis['samples'][0]['infinite_direction_first_structure']['box'] == 'ceiling'
    finite = result['scene_domain'].pop('finite_interior_key')
    assert result == scene
    assert finite['position_m'] == scene['scene_domain']['lighting']['primary_source']['position_m']
    assert np.allclose(np.asarray(finite['emission_rgb']) / finite['reference_distance_m'] ** 2,
                       scene['scene_domain']['lighting']['key_rgb'], rtol=1e-12)


def test_source_inside_ceiling_rejected_instead_of_moving_fixture():
    scene = read(PARENT / 'screen_00_metric_group_000_collect_all_office_candidate/render_scene.json')
    scene['scene_domain']['lighting']['primary_source']['position_m'][2] = 2.09
    with pytest.raises(ValueError, match='occluded_by_room_structure'):
        prepare(scene)
