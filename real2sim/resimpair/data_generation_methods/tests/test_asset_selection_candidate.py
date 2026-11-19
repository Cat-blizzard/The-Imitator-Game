"""Failure-oriented asset selection tests; no API, GPU or production mutations."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from data_generation_methods.asset_library.task_asset_selection import (
    SelectionError, _active_evidence, historical_apple_assets, load_static_catalog,
    proposal_distractor, role_candidates, select_distractors, select_task_assets,
)
from data_generation_methods.human_to_robot_pipeline.household_assets import bind
from data_generation_methods.r2h_pipeline.common import file_sha256


@pytest.fixture
def static_assets(tmp_path):
    result=[]
    for i,category in enumerate(['hammer','flashlight','scissors','calculator','bowl','apple']):
        folder=tmp_path/category;folder.mkdir()
        visual=folder/'visual.glb';collision=folder/'collision.obj'
        visual.write_bytes(f'file binding test visual {i}'.encode())
        collision.write_text(f'file binding test collision {i}')
        asset={'uid':category+'_001','category':category,'semantic_categories':[category],
            'usage':'static_distractor','visual':str(visual),'collision':str(collision),
            'files':{str(visual):file_sha256(visual),str(collision):file_sha256(collision)},
            'qualification':{'static_geometry_preflight':'passed'},
            'extents_m':[.1,.08,.04], 'bounds_rotated_m':[[-.05,-.04,0],[.05,.04,.04]],
            'collision_policy':'whole_object_convex_hull','license':'CC0-1.0',
            'physics_qualified':False,'training_admitted':False}
        spec=folder/'asset.json';spec.write_text(json.dumps(asset));asset['asset_spec']=bind(spec)
        result.append(asset)
    return result


def choose(assets, **overrides):
    options={'excluded_categories':['apple','bowl'],'count':4,'seed':7,
             'permit_preflight_experiment':True}
    options.update(overrides)
    return select_distractors(assets,**options)


def test_target_categories_are_excluded_and_other_categories_are_distinct(static_assets):
    selected=choose(static_assets)
    assert {a['category'] for a in selected}=={'hammer','flashlight','scissors','calculator'}
    assert all(not a['physics_qualified'] and not a['training_admitted'] for a in selected)


def test_qualification_is_not_inferred_from_raw_boolean_or_usage(static_assets):
    raw=deepcopy(static_assets[0]);raw['qualification']={};raw['physics_qualified']=True
    with pytest.raises(SelectionError):choose([raw],count=1)
    raw=deepcopy(static_assets[0]);raw['usage']='active_task_object';raw['physics_qualified']=True
    assert role_candidates({'category':'hammer','required_capabilities':['grasp_lift_release']},[raw],
                           'apple_bowl_transfer')==[]


def test_preflight_static_experiment_is_opt_in(static_assets):
    with pytest.raises(SelectionError):choose(static_assets,permit_preflight_experiment=False)


def test_insufficient_diversity_does_not_silently_reduce_count(static_assets):
    with pytest.raises(SelectionError,match='No silent reduction'):
        choose(static_assets,count=5)


def test_duplicate_geometry_under_two_categories_is_counted_once(static_assets):
    a,b=deepcopy(static_assets[:2]);b['files']=a['files'];b['visual']=a['visual'];b['collision']=a['collision']
    with pytest.raises(SelectionError):choose([a,b],count=2)


def test_catalog_order_does_not_change_seed_selection(static_assets):
    assert [a['uid'] for a in choose(static_assets)]==[a['uid'] for a in choose(static_assets[::-1])]


def test_alias_categories_can_exclude_target_ambiguity(static_assets):
    a=deepcopy(static_assets[0]);a['semantic_categories'].append('Apple')
    with pytest.raises(SelectionError):choose([a],count=1)


@pytest.mark.parametrize('damage',['changed_collision','missing_visual','changed_sidecar'])
def test_bound_asset_tampering_is_detected(static_assets,damage):
    a=static_assets[0]
    if damage=='changed_collision':Path(a['collision']).write_text('changed')
    elif damage=='missing_visual':Path(a['visual']).unlink()
    else:Path(a['asset_spec']['path']).write_text('{}')
    with pytest.raises(SelectionError):choose([a],count=1)


def test_in_memory_category_relabel_does_not_bypass_bound_sidecar(static_assets):
    a=deepcopy(static_assets[4]);a['category']='hammer';a['semantic_categories']=['hammer']
    with pytest.raises(SelectionError,match='differs from bound sidecar'):
        choose([a],count=1)


def test_placement_uses_baked_bottom_and_no_runtime_scaling(static_assets):
    p=proposal_distractor(static_assets[0],instance_id='tool_1',center_xy=[.2,-.4],yaw_rad=.5)
    assert p['center']==[.2,-.4,.02] and p['body_type']=='static'
    assert p['no_runtime_scaling'] and p['asset_files']==static_assets[0]['files']


@pytest.mark.parametrize('position',[[0,float('nan')],[0],[0,0,0]])
def test_invalid_placement_rejected(static_assets,position):
    with pytest.raises(SelectionError):proposal_distractor(static_assets[0],instance_id='x',center_xy=position)


def test_unknown_role_blocks_entire_selection(static_assets):
    result=select_task_assets({'mechanism':'cabinet_hinge_open','roles':{
        'door':{'category':'cabinet','required_capabilities':['hinge_open']}},
        'distractors':{'count':2,'permit_preflight_experiment':True}},static_assets)
    assert result['status']=='blocked' and not result['distractors']


@pytest.fixture
def historical_assets():
    folder=Path('/mnt/sdg/datasets/real_sim_human_pipeline/strict_20261020e/layout2001_v2')
    if not folder.exists():pytest.skip('Local frozen physical qualification fixture absent')
    return historical_apple_assets(folder)


def test_actual_historical_roles_select_same_bowl_model_as_distinct_instances(historical_assets):
    result=select_task_assets({'mechanism':'apple_bowl_transfer','roles':{
        'apple':{'category':'apple','required_capabilities':['grasp_lift_release']},
        'source':{'category':'bowl','required_capabilities':['support_inside_cavity']},
        'target':{'category':'bowl','required_capabilities':['support_inside_cavity']}}},historical_assets)
    assert result['status']=='candidate_ready_for_small_validation'
    assert result['roles']['source']['instance_id'] != result['roles']['target']['instance_id']
    assert result['roles']['source']['asset']['uid']==result['roles']['target']['asset']['uid']
    assert not result['physics_qualified']


def test_bound_apple_episode_cannot_be_relabelled_as_other_mechanism(historical_assets):
    a=deepcopy(historical_assets[0]);a['capability_evidence'][0]['mechanism']='cup_lid_turn'
    assert _active_evidence(a,'cup_lid_turn',['grasp_lift_release']) is None


def test_bowl_cannot_borrow_apple_grasp_capability(historical_assets):
    a=deepcopy(historical_assets[1]);a['capability_evidence'][0]['capabilities']=['grasp_lift_release']
    with pytest.raises(SelectionError,match='does not match'):
        _active_evidence(a,'apple_bowl_transfer',['grasp_lift_release'])


def test_actual_apple_cannot_be_semantically_relabelled(historical_assets):
    a=deepcopy(historical_assets[0]);a['semantic_categories']=['book'];a['category']='book'
    with pytest.raises(SelectionError,match='does not match'):
        role_candidates({'category':'book','required_capabilities':['grasp_lift_release']},[a],
                        'apple_bowl_transfer')


def test_size_constraint_is_applied_and_unknown_size_is_not_pass(historical_assets):
    assert role_candidates({'category':'apple','max_extent_m':.01,
                            'required_capabilities':['grasp_lift_release']},historical_assets,
                           'apple_bowl_transfer')==[]
    assert len(role_candidates({'category':'apple','max_extent_m':.06,
                            'required_capabilities':['grasp_lift_release']},historical_assets,
                           'apple_bowl_transfer'))==1
    assert role_candidates({'category':'bowl','max_extent_m':10,
                            'required_capabilities':['support_inside_cavity']},historical_assets,
                           'apple_bowl_transfer')==[]


def test_multiple_instances_cannot_be_silently_collapsed(historical_assets):
    with pytest.raises(SelectionError):role_candidates({'category':'apple','count':2,
        'required_capabilities':['grasp_lift_release']},historical_assets,'apple_bowl_transfer')


def test_actual_candidate_assets_have_reloadable_geometry_and_bound_preflight():
    root=Path('/mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/assets/static_catalog_v2')
    if not root.exists():pytest.skip('Candidate artifacts absent')
    import numpy as np
    import trimesh
    assets=load_static_catalog(root/'catalog.json')
    assert len(assets)==10
    for a in assets:
        visual=trimesh.load(a['visual'],force='scene',process=False)
        collision=trimesh.load(a['collision'],force='mesh',process=False)
        assert collision.is_convex and collision.is_watertight
        np.testing.assert_allclose(visual.bounds,collision.bounds,atol=1e-6,rtol=0)
        assert not a['physics_qualified'] and not a['training_admitted']


def test_real_gso_material_resolver_preserves_textured_visual_and_originals():
    root=Path('/mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/assets/gso_static')
    if not root.exists():pytest.skip('GSO candidate artifacts absent')
    import trimesh
    a=load_static_catalog(root/'catalog.json')[0]
    for b in a['original_dependency_bindings']:
        assert file_sha256(b['path'])==b['sha256']
    scene=trimesh.load(a['visual'],force='scene',process=False)
    assert any(getattr(getattr(g.visual,'material',None),'baseColorTexture',None) is not None
               for g in scene.geometry.values())
    assert a['cavities_handles_may_be_filled_by_conservative_collision']


def test_downloaded_indoor_backgrounds_are_bound_decodable_and_under_budget():
    root=Path('/mnt/sdg/datasets/real_sim_human_pipeline/optimization_20261021_180115/assets/backgrounds')
    if not root.exists():pytest.skip('Background candidates absent')
    import cv2
    import hashlib
    import numpy as np
    r=json.loads((root/'background_cases.json').read_text())
    assert len(r['new_hdri_records'])==3 and len(r['local_diffuse_surfaces'])==29
    assert r['downloaded_bytes_including_partials'] < r['download_byte_budget']
    assert r['paid_API_calls']==0 and not r['download_failures']
    for a in r['new_hdri_records']:
        assert a['license']=='CC0-1.0' and a['usage']=='visual_only'
        assert file_sha256(a['path'])==a['sha256']
        assert hashlib.md5(Path(a['path']).read_bytes()).hexdigest()==a['publisher_md5']
        assert np.isfinite(cv2.imread(a['path'],cv2.IMREAD_UNCHANGED)).all()
