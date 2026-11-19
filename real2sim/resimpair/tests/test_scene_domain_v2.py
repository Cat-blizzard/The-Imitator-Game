"""CPU geometry/lineage qualification. Rendering belongs to the independent run."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, read, write, digest
from resimpair_tasks import scene_domain_v2 as domain

SOURCE = ROOT / ('runs/whole_path_comparison_20261029_v1/candidate/executions/'
                 'factory_scene_000_3973d7e9_spatialv2_contain_selected')


@pytest.fixture(scope='module')
def source():
    scene=read(SOURCE/'effective_scene.json')
    envelope=domain.build_motion_envelope(scene,SOURCE/'measurements.npz')
    return scene,envelope


@pytest.fixture
def catalog(tmp_path):
    # Read-only existing byte-bound sources used solely as a schema fixture.
    old=read(ROOT/'assets/appearance_v1/catalog.json')['entries']
    rows=[]
    for asset,roles in [('angli_veneer',['tabletop','cabinet']),
                        ('brick_wall_04',['wall','floor']),
                        ('blue_metal_plate',['metal']),
                        ('poly_haven_studio',['environment'])]:
        row=deepcopy(next(r for r in old if r['id']=='polyhaven:'+asset))
        row.update(kind='hdri' if roles==['environment'] else 'material',roles=roles,
                   categories=['generic_indoor'],texture_size_m=[2.,1.],
                   texture_size_source='test_fixture_only',asset_partition='development')
        rows.append(row)
    path=tmp_path/'catalog.json';write(path,{'schema':domain.CATALOG_VERSION,'entries':rows})
    return path


def test_complete_source_mesh_and_robot_envelope_without_simulation(source):
    scene,e=source
    assert e['frame_count']==398
    assert e['maximum_fk_matrix_error']<3e-6
    assert len([r for r in e['rows'] if r['group']=='robot'])==22
    assert all(r['timeline_frames']==398 for r in e['rows'])
    assert e['object_order']==[r['id'] for r in scene['objects']]
    assert e['occlusion_checked'] is False
    assert e['physical_scene_digest']==digest(domain.physical_scene(scene))


def test_twenty_structurally_distinct_workspaces_protect_full_sweep_and_cameras(source):
    scene,e=source;signatures=set()
    assert len(domain.PRESET_IDS)==20
    for preset in domain.PRESET_IDS:
        room=domain._workspace(preset,e,scene['cameras'])
        signatures.add(room['structural_signature'])
        assert room['maniskill_actors_added']==room['collision_shapes_added']==0
        assert len(room['boxes'])>=10
        protected=np.asarray(room['protected_bounds_m']); margin=room['minimum_protected_margin_m']
        for b in room['boxes']:
            low=np.asarray(b['center_m'])-np.asarray(b['size_m'])/2
            high=np.asarray(b['center_m'])+np.asarray(b['size_m'])/2
            assert not (np.all(high>protected[0]-margin) and np.all(low<protected[1]+margin))
    assert len(signatures)==20


def test_metric_box_uvs_repeat_by_metres_and_normals_are_outward():
    v,f,n,uv=domain.textured_box_mesh([4.,2.,1.],[2.,.5])
    assert v.shape==n.shape==(24,3) and f.shape==(12,3) and uv.shape==(24,2)
    assert np.max(uv[:,0])==2. and np.max(uv[:,1])==4.
    for face in f:
        assert np.dot(np.cross(v[face[1]]-v[face[0]],v[face[2]]-v[face[0]]),n[face[0]])>0


def test_full_bounds_projection_detects_cropped_edge_and_behind_camera():
    camera={'eye':[0.,0.,1.],'target':[0.,1.,1.],'width':512,'height':512,'fov':1.}
    assert domain.camera_projection(camera,[[-.1,1.,.9],[.1,1.1,1.1]])['fully_in_frame']
    edge=domain.camera_projection(camera,[[-2.,1.,.9],[.1,1.1,1.1]])
    assert not edge['fully_in_frame'] and 0<edge['visible_area_fraction']<1
    assert not domain.camera_projection(camera,[[-.1,-1.,.9],[.1,-.9,1.1]])['in_front']


def test_camera_uses_complete_critical_boxes_and_retains_robot_side(source):
    scene,e=source
    cameras,record=domain.sample_camera(scene,np.random.default_rng(20261029001),e)
    assert record['status']=='accepted_geometric_proposal'
    assert cameras['robot_side']==scene['cameras']['robot_side']
    assert set(cameras)=={'robot_side','opposite_high'}
    assert cameras['opposite_high']!=scene['cameras']['opposite_high']
    assert all(v['fully_in_frame'] for v in record['actual_coverage'].values() if v['critical'])
    assert record['baseline_coverage']['container_1']['fully_in_frame'] is False
    assert record['all_robot_geometry_in_frame_claimed'] is False
    assert np.all(np.abs(np.asarray(cameras['opposite_high']['eye'])-scene['cameras']['opposite_high']['eye'])<=domain.CAMERA_LIMITS['eye_delta_m'])


def test_impossible_camera_does_not_shrink_bounds_or_invent_success(source):
    scene,e=source;e=deepcopy(e)
    e['rows'][0]['bounds_m']=[[-100.,-100.,0.],[100.,100.,100.]]
    cameras,record=domain.sample_camera(scene,np.random.default_rng(5),e)
    assert cameras==scene['cameras']
    assert record['status']=='no_feasible_proposal_retained_original'
    assert len(record['attempts'])==domain.CAMERA_LIMITS['max_attempts']
    assert all(r['reasons'] for r in record['attempts'])


def test_sampling_retains_physics_and_is_deterministic(source,catalog):
    scene,e=source;before=deepcopy(scene)
    kwargs=dict(catalog_path=catalog,preset_id='kitchen_window_wall',camera_mode='bounded',motion_envelope=e)
    a=domain.sample_domain(scene,13,**kwargs); b=domain.sample_domain(scene,13,**kwargs)
    assert a==b and scene==before
    assert domain.physical_scene(a)==domain.physical_scene(scene)
    assert a['scene_domain']['physical_execution_count']==0 and not a['scene_domain']['new_trajectory']
    assert a['scene_domain']['resource_fallbacks']['tabletop'].startswith('no_category_specific')
    assert a['cameras']['robot_side']==scene['cameras']['robot_side']
    domain.validate_scene(a)
    a['objects'][0]['position'][0]+=.01
    with pytest.raises(ValueError,match='physics_binding'):domain.validate_scene(a)


def test_fixed_camera_catalog_ablation_has_no_workspace_geometry(source,catalog):
    scene,e=source
    a=domain.sample_domain(scene,5,catalog_path=catalog,mode='catalog_only',camera_mode='fixed')
    assert a['cameras']==scene['cameras']
    assert a['scene_domain']['geometry'] is None
    assert not a['scene_domain']['visible_light_geometry_installed']


def test_ablations_share_resources_lighting_and_workspace_geometry(source,catalog):
    scene,e=source
    kwargs=dict(catalog_path=catalog,preset_id='kitchen_window_wall',motion_envelope=e)
    full=domain.sample_domain(scene,5,**kwargs,mode='workspace',camera_mode='bounded')
    fixed=domain.sample_domain(scene,5,**kwargs,mode='workspace',camera_mode='fixed')
    catalog_only=domain.sample_domain(scene,5,**kwargs,mode='catalog_only',camera_mode='fixed')
    a,b,c=[s['scene_domain'] for s in (full,fixed,catalog_only)]
    assert a['resources']==b['resources']==c['resources']
    assert a['lighting']==b['lighting']==c['lighting']
    assert a['geometry']==b['geometry']
    assert a['preset_lighting_geometry_signature']==c['preset_lighting_geometry_signature']


def test_no_material_category_or_partition_guessing(source,catalog,tmp_path):
    scene,_=source
    with pytest.raises(ValueError,match='no_semantically_compatible_material'):
        domain.sample_domain(scene,1,catalog_path=catalog,asset_partition='heldout_appearance')
    value=read(catalog)
    for row in value['entries']:row['categories']=['workshop']
    altered=tmp_path/'incompatible.json';write(altered,value)
    with pytest.raises(ValueError,match='no_semantically_compatible_material:kitchen'):
        domain.sample_domain(scene,1,catalog_path=altered,preset_id='kitchen_window_wall')


def test_cameras_and_envelopes_are_bound_not_mutable_annotations(source,catalog):
    scene,e=source
    with pytest.raises(ValueError,match='bound_full_motion_envelope'):
        domain.sample_domain(scene,1,catalog_path=catalog,camera_mode='bounded')
    a=domain.sample_domain(scene,1,catalog_path=catalog)
    a['cameras']['robot_side']['eye'][0]+=.01
    with pytest.raises(ValueError,match='camera_binding'):domain.validate_scene(a)
