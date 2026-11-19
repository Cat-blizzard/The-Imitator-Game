from copy import deepcopy

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, read
from resimpair_tasks import motion_reference_aspect_v2 as media
from resimpair_tasks.camera_resolution_replay_v1 import candidate_scene


@pytest.mark.parametrize('source,expected', [((512,512),(720,720)), ((1280,720),(1280,720)),
    ((1920,1080),(1280,720)), ((720,1280),(720,1280)), ((1024,512),(1280,640))])
def test_exact_aspect_even_dimensions(source,expected):
    width,height=media.dimensions(*source)
    assert (width,height)==expected
    assert width*source[1]==height*source[0]
    assert width%2==height%2==0


def test_frame_mapping_retains_endpoints_and_explicit_duplicates():
    selected,target=media.frame_indices([0,.1,.5],4,fps=1)
    assert selected.tolist()==[0,1,2,2]
    assert np.all(np.diff(selected)>=0)
    assert target[0]==0 and target[-1]==.5


def test_irregular_source_pts_not_assumed_constant_fps():
    selected,_=media.frame_indices([0,.001,.002,.9,1.],4,fps=1)
    assert selected.tolist()==[0,2,3,4]


@pytest.mark.parametrize('times', [[0,0,.1], [0,.2,.1], [0,float('nan'),1]])
def test_invalid_clocks_fail_before_encoding(times):
    with pytest.raises(ValueError):
        media.frame_indices(times,8)


def test_real_scene_resolution_changes_only_declared_fields():
    source=ROOT/'runs/metric_group_functional_rooms_20261030_v1/cases/video_01_metric_group_000_distribute_groups_dining_candidate/render_scene.json'
    baseline=read(source);frozen=deepcopy(baseline)
    candidate=candidate_scene(baseline)
    assert baseline==frozen
    assert candidate['cameras']['robot_side']==baseline['cameras']['robot_side']
    for key in ('eye','target','fov'):
        assert candidate['cameras']['opposite_high'][key]==baseline['cameras']['opposite_high'][key]
    candidate['cameras']=deepcopy(baseline['cameras'])
    candidate['scene_domain']['camera_sampling']=deepcopy(baseline['scene_domain']['camera_sampling'])
    assert candidate==baseline


def test_resolution_candidate_rejects_unregistered_source_dimensions():
    source=ROOT/'runs/metric_group_functional_rooms_20261030_v1/cases/video_01_metric_group_000_distribute_groups_dining_candidate/render_scene.json'
    scene=read(source);scene['cameras']['opposite_high']['width']=640
    with pytest.raises(ValueError,match='512_square'):
        candidate_scene(scene)
