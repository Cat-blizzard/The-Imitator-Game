import pytest
from resimpair_grasp.common import ROOT,read,resolve
from resimpair_tasks import visual_furniture_room_v3 as room


def test_real_container_graphic_cupboard_is_not_office_storage():
    catalog,analysis,_=room.base.catalog_data()
    entry=next(e for e in catalog['entries'] if any(a['id']=='abo:B075GWYJRC' for a in e['geometry_aliases']))
    semantics=room.base.semantic_anchor(entry)
    assert not room.room_fit(entry,semantics,'office')['eligible']


def test_living_room_wingback_is_not_a_task_chair():
    e={'size_m':[1.,1.,1.]};s={'title':'Living Room Wingback Accent Chair','resolved_category':'chair'}
    assert not room.room_fit(e,s,'office')['eligible']
    s['title']='Adjustable Ergonomic Office Chair'
    assert room.room_fit(e,s,'office')['eligible']


@pytest.mark.parametrize('material_id,role',[('polyhaven:granite_tile_02','tabletop'),
    ('polyhaven:anti_skid_tiles','wall'),('polyhaven:climbing_wall_02','wall'),('polyhaven:patio_tiles','floor')])
def test_retained_real_surface_role_errors_are_rejected(material_id,role):
    index=room.materials.load_index(room.materials.DEFAULT_INDEX)
    entry=next(x['entry'] for x in index['rows'] if x['id']==material_id)
    assert not room.surface_fit(entry,role)['eligible']


@pytest.fixture(scope='module')
def metric_source():
    path=ROOT/'runs/production_integration_20261030_v1/metric_crayon_qualified/executions/metric_crayon_L2_into_bowl'
    actual=room.appearance.replay.validate_source(path)
    baseline,_=room.appearance.render.baseline_scene(actual)
    envelope=room.domain.build_motion_envelope(baseline,path/'measurements.npz')
    return baseline,envelope


@pytest.mark.parametrize('purpose,seed',[('office',2026103051),('dining',2026103052)])
def test_new_metric_source_room_candidate_retains_original_physics_camera_native_scale(metric_source,purpose,seed):
    baseline,envelope=metric_source
    out,receipt=room.apply_overlay(baseline,seed,motion_envelope=envelope,room=purpose,
        partition=baseline['partition'])
    assert receipt['appearance_applied'] and not receipt['R2H_ready']
    assert room.domain.physical_scene(out)==room.domain.physical_scene(baseline)
    assert out['cameras']==baseline['cameras']
    d=out['scene_domain'];g=d['geometry']
    assert all(x['scale']==[1.,1.,1.] and x['room_function_evidence']['eligible'] for x in g['furniture'])
    assert all(x['source_partition']==baseline['partition'] for x in g['furniture'])
    assert {x['placement_zone'] for x in g['furniture']}=={'rear','front'}
    assert all(not room.base._overlaps(x['world_bounds_m'],g['protected_bounds_m'],.12) for x in g['furniture'])
    assert not d['visible_light_geometry_installed']
    for role in ['tabletop','wall','floor']:
        assert room.surface_fit(d['resources'][role],role)['eligible']
        assert d['resources'][role]['texture_size_source']=='publisher_dimensions_mm'
