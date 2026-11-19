from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, digest, read, resolve, write
from resimpair_tasks import visual_furniture_bulk_scene_v1 as scene_layer


@pytest.mark.parametrize('title,source_category,expected_category,anchor,height', [
    ('Oval Coffee Table', 'clothes_rack', 'table', 'floor', .4),
    ('Floating Wall Shelf', 'shelf', 'shelf', 'wall', .3),
    ('King Bed', 'storage_box', 'bed', 'floor', 1.),
    ('Bedside Table Lamp', 'lamp', 'lamp', 'furniture_top', .6),
    ('Floor Lamp', 'lamp', 'lamp', 'floor', 1.6),
    ('Ceiling Pendant Lamp', 'lamp', 'lamp', 'ceiling', .6),
    ('Unknown Model', 'cabinet', 'unknown', 'unknown', 1.),
])
def test_title_and_anchor_are_independent_of_source_category(title,source_category,expected_category,anchor,height):
    result = scene_layer.semantic_anchor({'name': title, 'category': source_category,
                                          'size_m': [.5, .5, height], 'placement_support': 'floor'})
    assert result['resolved_category'] == expected_category and result['anchor'] == anchor
    assert result['source_category'] == source_category
    assert result['supported_by_current_installer'] == (anchor == 'floor')


@pytest.fixture(scope='module')
def sources():
    spec = read(ROOT/'reports/visual_furniture_bulk_20261030/source_screens_v2.json')
    values = []
    for row in spec['sources']:
        source = scene_layer.appearance.replay.validate_source(resolve(row['execution']).parent)
        envelope = scene_layer.domain.build_motion_envelope(source['scene'],source['source']/'measurements.npz')
        values.append((row,source['scene'],envelope))
    return values


@pytest.mark.parametrize('index', range(4))
def test_four_real_sources_keep_physics_cameras_scale_and_protection(sources,index):
    spec, scene, envelope = sources[index]; original = deepcopy(scene)
    result, receipt = scene_layer.apply_overlay(scene,spec['seed'],motion_envelope=envelope,
        group=spec['group'],preset=spec['preset'],partition=spec['partition'])
    assert scene == original
    assert scene_layer.domain.physical_scene(result) == scene_layer.domain.physical_scene(scene)
    assert result['cameras'] == scene['cameras']
    protected = np.asarray(result['scene_domain']['geometry']['protected_bounds_m'])
    items = result['scene_domain']['geometry']['furniture']
    assert len(items)>=2
    for item in items:
        assert item['scale'] == [1.,1.,1.] and item['native_metric_scale_preserved']
        assert item['semantics']['anchor'] == 'floor'
        assert item['source_partition'] == spec['partition']
        assert not scene_layer._overlaps(item['world_bounds_m'],protected,gap=.08)
        assert item['world_bounds_m'][0][2] == pytest.approx(-.78)
        assert resolve(item['mesh']).is_file()
    policy=result['scene_domain']['bulk_furniture_policy']
    assert all(c['eye_inside_room'] for c in policy['original_camera_checks'].values())
    assert policy['new_physical_actors']==policy['new_collision_shapes']==0
    assert receipt['R2H_ready'] is False


def test_random_seed_changes_furniture_identity_not_task(sources):
    spec, scene, envelope = sources[0]
    kwargs=dict(motion_envelope=envelope,group=spec['group'],preset=spec['preset'],partition=spec['partition'])
    first,_=scene_layer.apply_overlay(scene,1,**kwargs)
    second,_=scene_layer.apply_overlay(scene,2,**kwargs)
    a=first['scene_domain']['bulk_furniture_policy']['selected'];b=second['scene_domain']['bulk_furniture_policy']['selected']
    assert a!=b
    assert scene_layer.domain.physical_scene(first)==scene_layer.domain.physical_scene(second)


def test_inconsistent_units_and_unsupported_anchors_are_not_selected():
    analysis=scene_layer.inspect_catalog()
    for row in analysis['rows']:
        if row['eligible']:
            assert row['semantics']['anchor']=='floor'
            assert row['asset_partition'] in ('development','heldout_asset')
        if row['semantics']['anchor'] in ('wall','ceiling','furniture_top','unknown'):
            assert not row['eligible']


def test_background_partition_cannot_override_heldout_task(sources):
    spec,scene,envelope=sources[1]
    with pytest.raises(ValueError,match='partition_must_match'):
        scene_layer.apply_overlay(scene,spec['seed'],motion_envelope=envelope,
            group=spec['group'],preset=spec['preset'],partition='development')
