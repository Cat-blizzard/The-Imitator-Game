from copy import deepcopy

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, read, resolve
from resimpair_tasks import visual_furniture_bulk_scene_v2 as layer

PILOT = ROOT/'assets/visual_furniture_bulk_20261030_v2/pilot_catalog.json'


@pytest.mark.parametrize('title,category,anchor', [
    ('Classic L-Shaped Desk with Open Bookshelves, Black','desk','floor'),
    ('Dining Table with Chairs','table','floor'),
    ('Table Lamp with USB charging','lamp','furniture_top'),
    ('Floating Wall Shelf with LED','shelf','wall'),
    ('Ceiling Pendant Lamp','lamp','ceiling'),
    ('Queen Bed Frame with Headboard','bed','floor'),
    ('Upholstered Headboard','unknown','unsupported_accessory'),
    ('Storage Ottoman with Hidden Compartment','ottoman','floor'),
])
def test_head_noun_and_anchor(title,category,anchor):
    result=layer.semantic_anchor({'name':'untrusted translated title','category':'shelf',
        'size_m':[1.,1.,.7], 'source_metric_records':[{'item_name':[{'language_tag':'en_US','value':title}]}]})
    assert result['resolved_category']==category and result['anchor']==anchor
    assert result['title_source']=='publisher_english_item_name'


def test_all_floor_categories_have_sampling_room_and_old_omissions_are_reachable():
    catalog,analysis,_=layer.catalog_data(PILOT)
    eligible=[r for r in analysis['rows'] if r['eligible'] and r['semantics']['anchor']=='floor']
    for kind in ('bed','ottoman','stool'):
        rows=[r for r in eligible if r['semantics']['resolved_category']==kind]
        assert rows,kind
        row=rows[0]
        items=layer.sample(catalog,analysis,seed=1,room=row['room_purposes'][0],
            partition=row['asset_partition'],required_asset=row['id'])
        assert row['id'] in [x['entry']['id'] for x in items]
    assert all(r['room_purposes'] for r in eligible)


@pytest.fixture(scope='module')
def source():
    spec=read(ROOT/'reports/visual_furniture_bulk_20261030/source_screens_v2.json')['sources'][0]
    actual=layer.appearance.replay.validate_source(resolve(spec['execution']).parent)
    baseline,_=layer.appearance.render.baseline_scene(actual)
    envelope=layer.domain.build_motion_envelope(baseline,actual['source']/'measurements.npz')
    return baseline,envelope


@pytest.mark.parametrize('room',list(layer.ROOMS))
def test_real_saved_source_rooms_preserve_physics_camera_native_size_and_corridor(source,room):
    scene,envelope=source; original=deepcopy(scene)
    result,receipt=layer.apply_overlay(scene,20261030,motion_envelope=envelope,catalog_path=PILOT,
        room=room,partition=scene['partition'])
    assert receipt['appearance_applied'] and not receipt['R2H_ready']
    assert scene==original
    assert layer.domain.physical_scene(scene)==layer.domain.physical_scene(result)
    assert scene['cameras']==result['cameras']
    g=result['scene_domain']['geometry']; protected=g['protected_bounds_m']
    floor=[x for x in g['furniture'] if x['anchor']=='floor']
    assert len({x['placement_zone'] for x in floor})>=min(3,len(floor))
    for item in g['furniture']:
        assert item['scale']==[1.,1.,1.]
        assert item['source_partition']==scene['partition']
        assert not layer._overlaps(item['world_bounds_m'],protected,.12)
        assert item['collision_shapes_added']==0
        if item['anchor']=='floor': assert item['world_bounds_m'][0][2]==pytest.approx(-.78)
        else: assert item['support_evidence']['full_footprint_covered_by_single_convex_native_triangle']


def test_missing_visual_catalog_returns_exact_source_not_physical_failure(source,tmp_path):
    scene,envelope=source
    result,receipt=layer.apply_overlay(scene,3,motion_envelope=envelope,catalog_path=tmp_path/'absent.json')
    assert result==scene and not receipt['appearance_applied']
    assert receipt['visual_degraded'] and not receipt['R2H_ready']


def test_triangle_support_does_not_bridge_a_hole():
    triangles=np.array([[[0,0,0],[1,0,0],[0,1,0]],[[1,1,0],[1,0.8,0],[.8,1,0]]])
    assert layer._points_in_triangles([[.1,.1],[.2,.2]],triangles)
    assert not layer._points_in_triangles([[.55,.55]],triangles)


def test_long_support_obstacle_triangle_with_vertices_outside_lamp_is_rejected():
    spanning=np.array([[[-2.,-2.,.95],[2.,-2.,.95],[0.,2.,.95]]])
    assert not layer.support_volume_clear(spanning,[0,0],[.3,.3],.7,.6)
    below=np.array([[[-2.,-2.,.7],[2.,-2.,.7],[0.,2.,.7]]])
    assert layer.support_volume_clear(below,[0,0],[.3,.3],.7,.6)


def test_floor_layout_never_rescales_large_asset_and_fixed_view_changes_positions(source):
    scene,envelope=source
    prepared,_=layer.appearance.apply_overlay(scene,7,mode='workspace',preset='office_bookcase',motion_envelope=envelope)
    catalog,analysis,_=layer.catalog_data(PILOT)
    chosen=layer.sample(catalog,analysis,seed=9,room='living',partition=scene['partition'])
    bounds=prepared['scene_domain']['geometry']['protected_bounds_m']
    a=layer.floor_layout(chosen,bounds,9,view_layout='robot_side')
    b=layer.floor_layout(chosen,bounds,9,view_layout='opposite_high')
    assert [x['source_id'] for x in a]==[x['source_id'] for x in b]
    assert a[0]['placement_zone']!=b[0]['placement_zone']
    assert all(x['scale']==[1.,1.,1.] for x in a+b)


def test_real_native_lamp_base_can_use_desk_without_rescaling_whole_lamp(source):
    scene,envelope=source
    result,_=layer.apply_overlay(scene,2026103022,motion_envelope=envelope,catalog_path=PILOT,
        room='office',partition=scene['partition'])
    supported=[x for x in result['scene_domain']['geometry']['furniture'] if x['anchor']=='furniture_top']
    assert supported
    evidence=supported[0]['support_evidence']
    assert evidence['full_footprint_covered_by_single_convex_native_triangle']
    assert evidence['clearance_triangle_AABB_broadphase']
    assert evidence['base_footprint']['native_bottom_band_vertex_count']>=3
    assert supported[0]['scale']==[1.,1.,1.]


def test_conflicting_alias_dimensions_cannot_be_hidden_by_consistent_representative():
    catalog,_,_=layer.catalog_data(PILOT)
    entry=deepcopy(next(r for r in catalog['entries'] if r['scale_confidence']=='source_metric_record_consistent'))
    entry['geometry_aliases']=[{'scale_confidence':'source_dimensions_conflict'}]
    row=layer.analyze({'entries':[entry]})['rows'][0]
    assert not row['eligible']
    assert 'same_geometry_alias_has_conflicting_dimensions' in row['reasons']


def test_partition_cannot_be_overridden(source):
    scene,envelope=source
    with pytest.raises(ValueError,match='partition_must_match'):
        layer.apply_overlay(scene,1,motion_envelope=envelope,catalog_path=PILOT,
            room='office',partition='heldout_asset')
