from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from resimpair_grasp.common import ROOT,binding,read,digest
from resimpair_tasks import scene_material_policy_v3 as policy
from resimpair_tasks.scene_domain_v2 import physical_scene


def entry(tmp_path,rgb,name='synthetic_texture',size=(1.,1.)):
    path=tmp_path/(name+'.png');Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(path)
    return {'id':name,'resources':{'base_color':binding(path)},'texture_size_m':list(size),
            'texture_size_source':'known_test_metres','source_material_family':'wood',
            'roles':list(policy.ROLES),'categories':list(policy.CATEGORIES),
            'source_tags':['veneer'],'asset_partition':'development'}


def test_lab_conversion_has_known_neutral_endpoints():
    out=policy.rgb_to_lab(np.array([[0.,0.,0.],[1.,1.,1.],[.5,.5,.5]]))
    np.testing.assert_allclose(out[:,0],[0.,100.,53.38896],atol=2e-4)
    np.testing.assert_allclose(out[:,1:],0,atol=2e-4)


def test_actual_rgb_distinguishes_uniform_and_equal_mean_high_frequency(tmp_path):
    uniform=np.full((512,512,3),.5)
    y,x=np.indices((512,512));checker=np.repeat((.5+.20*((x//4+y//4)%2*2-1))[:,:,None],3,axis=2)
    quiet=policy.describe_material(entry(tmp_path,uniform,'quiet'))
    noisy=policy.describe_material(entry(tmp_path,checker,'checker'))
    np.testing.assert_allclose(quiet['mean_srgb'],noisy['mean_srgb'],atol=.003)
    assert noisy['metric_highpass_sigma4_rms']>quiet['metric_highpass_sigma4_rms']+.1
    assert quiet['metric_highpass_sigma4_rms']<1e-6
    assert noisy['analysis_metric_input_px']==[512,512]
    assert len(noisy['metric_patches'])==4


def test_metric_texture_is_compared_in_metres_not_source_pixel_units(tmp_path):
    def wave(n):
        y,x=np.indices((n,n));v=.5+.15*np.sin(x/n*2*np.pi*8)
        return np.repeat(v[:,:,None],3,axis=2)
    small=policy.describe_material(entry(tmp_path,wave(256),'source256'))
    large=policy.describe_material(entry(tmp_path,wave(1024),'source1024'))
    assert abs(small['metric_highpass_sigma4_rms']-large['metric_highpass_sigma4_rms'])<.002
    changed_scale=policy.describe_material(entry(tmp_path,wave(1024),'scale4m',size=(4.,4.)))
    assert changed_scale['metric_highpass_sigma4_rms']<large['metric_highpass_sigma4_rms']/4


def test_daily_worktop_requires_finish_not_raw_floorboard_metadata():
    base={'id':'polyhaven:test','name':'Natural wood','source_material_family':'wood',
          'roles':['tabletop','wall','cabinet','floor'],'categories':list(policy.CATEGORIES)}
    veneer={**base,'source_categories':['wood','raw wood'],'source_tags':['veneer','natural grain']}
    assert policy.semantic_reasons(veneer,'tabletop','office')==[]
    planks={**base,'id':'polyhaven:rough_plank_floor','source_tags':['weathered','outdoor','flooring']}
    reasons=policy.semantic_reasons(planks,'tabletop','office')
    assert 'floorboard_or_deck_is_not_worktop' in reasons
    assert 'rough_wood_is_not_finished_worktop' in reasons
    pavers={**base,'source_material_family':'tile','id':'polyhaven:brick_pavement','source_tags':['tileable']}
    assert any(r.startswith('exterior_surface:') for r in policy.semantic_reasons(pavers,'floor','office'))


def test_white_worktop_and_strong_texture_are_explicit_separate_reasons(tmp_path):
    white=policy.describe_material(entry(tmp_path,np.ones((256,256,3))*.95,'white'))
    assert 'median_lightness_outside_slot_range' in policy.rgb_reasons(white,'tabletop')
    artificial=deepcopy(white);artificial['L_percentiles']['p50']=55.
    artificial['metric_highpass_sigma4_rms']=.15
    reasons=policy.rgb_reasons(artificial,'tabletop')
    assert 'median_lightness_outside_slot_range' not in reasons
    assert 'metric_highpass_sigma4_rms_above_slot_limit' in reasons


@pytest.fixture(scope='module')
def index():return policy.load_index(policy.DEFAULT_INDEX)


def test_all_221_source_materials_measured_without_rewriting_rgb(index):
    catalog=read(policy.DEFAULT_CATALOG)
    material_ids={r['id'] for r in catalog['entries'] if r['kind']=='material'}
    assert {r['id'] for r in index['rows']}==material_ids and len(material_ids)==221
    assert index['gpu_initializations']==index['api_calls']==0
    assert not index['resource_bytes_modified']
    for row in index['rows']:
        assert row['descriptor']['source_base_color']==row['entry']['resources']['base_color']
        assert row['descriptor']['analysis_metric_input_px']==row['descriptor']['source_dimensions_px']


def test_sampler_preserves_texture_and_material_family_diversity(index):
    selected=[]
    for seed in range(64):
        asset,evidence=policy.choose(index,'office','tabletop',seed,asset_partition='development')
        selected.append(asset)
        assert evidence['eligibility']['eligible']
        assert 'base_color' in asset['resources']
    assert len({r['id'] for r in selected})>=7
    assert {r['source_material_family'] for r in selected}=={'wood','stone'}
    assert policy.choose(index,'office','tabletop',42,asset_partition='development')==policy.choose(index,'office','tabletop',42,asset_partition='development')


def test_no_invented_heldout_worktop_or_cross_split_fallback(index):
    with pytest.raises(ValueError,match='no_qualified_material_slot:office:tabletop:heldout_appearance'):
        policy.choose(index,'office','tabletop',1,asset_partition='heldout_appearance')
    with pytest.raises(ValueError,match='no_qualified_material_slot:office:metal:development'):
        policy.choose(index,'office','metal',1,asset_partition='development')


@pytest.fixture
def scene():
    return read(ROOT/'runs/scene_domain_20261029_v2/cases/screen_00_kitchen_window_wall/render_scene.json')


def test_policy_changes_only_four_surface_choices_with_explicit_lineage(scene):
    before=deepcopy(scene);after=policy.apply_policy(scene,17)
    a,b=before['scene_domain'],after['scene_domain']
    assert scene==before
    assert physical_scene(scene)==physical_scene(after)
    assert after['cameras']==before['cameras']
    assert all(a[key]==b[key] for key in ('geometry','lighting','camera_sampling'))
    assert all(a['resources'][r]==b['resources'][r] for r in ('metal','environment'))
    assert b['asset_partition']=='mixed_inherited_and_development_materials'
    p=b['material_policy'];assert p['previous_asset_partition']==a['asset_partition']
    assert p['intervention_roles']==list(policy.INTERVENTION_ROLES)
    assert p['reads_task_objects_or_target_colors'] is False
    assert p['resource_RGB_bytes_modified'] is False


def test_selection_is_independent_of_task_objects_or_colors(scene):
    a=policy.apply_policy(scene,6)
    changed=deepcopy(scene)
    for obj in changed['objects']:
        obj['rgba']=[1.,.1,.4,1.];obj['semantic_description']='unrelated task object'
    changed['scene_domain']['physical_scene_digest']=digest(physical_scene(changed))
    b=policy.apply_policy(changed,6)
    assert a['scene_domain']['resources']==b['scene_domain']['resources']


def test_missing_partition_or_repeat_application_is_not_silently_recovered(scene):
    with pytest.raises(ValueError,match='explicit_material_partition_required'):
        policy.apply_policy(scene,1,asset_partition=None)
    candidate=policy.apply_policy(scene,1)
    with pytest.raises(ValueError,match='original_prepared_scene_only'):
        policy.apply_policy(candidate,1)
