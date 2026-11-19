"""Camera-consumption policy regressions over retained measured view evidence."""
from copy import deepcopy

import pytest

from resimpair_grasp.common import ROOT, read, resolve
from resimpair_pickplace.render_lineage_validator_v4 import reference_visibility


FOLDER=ROOT/'validation/pickplace_automation_20261029/independent/scene_domain_v2'


def load_case(path):
    if not path.exists():pytest.skip('requires retained actual render measurements')
    result=read(path);parent=read(resolve(result['source_native_execution']))
    return result,parent['task_intent']


@pytest.fixture
def actual_screen():
    return load_case(FOLDER/'screen_reviews/screen_00_kitchen_window_wall.json')


def test_auxiliary_crop_does_not_change_actual_selected_input():
    old,task=load_case(FOLDER/'video_reviews_v3/video_source_2_baseline.json')
    assert old['pixel_readability']['status']=='failed'
    new=reference_visibility(old['pixel_readability'],task,'baseline')
    assert new['status']=='confirmed'
    assert new['selected_full_visual_bbox_in_frame']
    assert new['auxiliary_view_diagnostics']['robot_side']['frames_with_critical_crop']==395


def test_auxiliary_pixels_cannot_rescue_occluded_selected_object(actual_screen):
    old,task=actual_screen;pixels=deepcopy(old['pixel_readability'])
    row=pixels['per_frame_geometry_and_visible_pixels'][0]
    target=task['object_ids'][0]
    assert row['views']['robot_side']['entities'][target]['visible_pixels']>=16
    row['views']['opposite_high']['entities'][target]['visible_pixels']=0
    new=reference_visibility(pixels,task,'environment_3d_camera')
    assert new['status']=='failed' and not new['no_hard_reference_failure']


def test_selected_bbox_crop_rejected_even_when_auxiliary_is_whole(actual_screen):
    old,task=actual_screen;pixels=deepcopy(old['pixel_readability'])
    row=pixels['per_frame_geometry_and_visible_pixels'][0];target=task['object_ids'][0]
    assert row['views']['robot_side']['entities'][target]['fully_in_frame']
    row['views']['opposite_high']['entities'][target]['fully_in_frame']=False
    assert reference_visibility(pixels,task,'environment_3d_camera')['status']=='failed'


def test_active_hand_must_be_visible_in_selected_reference(actual_screen):
    old,task=actual_screen;pixels=deepcopy(old['pixel_readability'])
    pixels['per_frame_geometry_and_visible_pixels'][1]['views']['opposite_high']['active_hand_pixels']['arm0']=0
    assert reference_visibility(pixels,task,'environment_3d_camera')['status']=='failed'


def test_structures_seen_only_by_auxiliary_do_not_qualify(actual_screen):
    old,task=actual_screen;pixels=deepcopy(old['pixel_readability'])
    pixels['per_frame_geometry_and_visible_pixels'][0]['views']['opposite_high']['structural_pixels']=0
    assert reference_visibility(pixels,task,'environment_3d_camera')['status']=='failed'


def test_hidden_terminal_containment_stays_unknown(actual_screen):
    old,task=actual_screen;pixels=deepcopy(old['pixel_readability']);target=task['object_ids'][0]
    pixels['per_frame_geometry_and_visible_pixels'][-1]['views']['opposite_high']['entities'][target]['visible_pixels']=0
    result=reference_visibility(pixels,task,'environment_3d_camera')
    assert result['status']=='unknown' and result['no_hard_reference_failure']


def test_missing_per_view_measurements_cannot_qualify():
    assert reference_visibility({'status':'confirmed'}, {}, 'baseline')['status']=='unknown'
