"""Use retained real video for schema checks; do not synthesize physics success."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, binding, read, write
from resimpair_pickplace import render_lineage_validator_v3 as validator
from resimpair_tasks.deferred_render_v1 import REQUIRED_EVIDENCE


PRE = ROOT/'validation/pickplace_automation_20261029/independent/scene_domain_v2/protocol.json'
OLD = ROOT/'runs/deferred_render_20261029_v1/original'


@pytest.fixture
def retained_media_fixture(tmp_path):
    if not PRE.exists() or not (OLD/'execution_bundle.json').exists():
        pytest.skip('retained actual media required; no mock physics substitute')
    fixed=read(PRE); selected=deepcopy(next(c for c in fixed['cases'] if c['id']=='video_source_0_baseline'))
    source=next(s for s in fixed['sources'] if s['id']=='source_0')
    source_folder=ROOT/Path(source['source_execution']['path']).parent
    source_row={**source,'artifacts':{name:binding(source_folder/name) for name in REQUIRED_EVIDENCE}}
    selected['scene']=binding(source_folder/'effective_scene.json')
    proto={'independent_protocol':binding(PRE),'cases':[selected],'sources':[source_row],
           'code':{'validator':binding(validator.__file__)},'fixture_only_not_a_new_render':True}
    write(tmp_path/'protocol.json',proto); ref=binding(tmp_path/'protocol.json')
    write(tmp_path/'reservation.json',{'protocol':ref,'case':selected,'fixture_only_not_a_new_render':True})
    shutil.copyfile(OLD/'frame_mapping.json',tmp_path/'frame_mapping.json')
    b=read(OLD/'execution_bundle.json');b['schema']=validator.BUNDLE_VERSION
    b['render_protocol']=ref;b['render_case_id']=selected['id'];b['frame_mapping']=binding(tmp_path/'frame_mapping.json')
    for name,camera in b['cameras'].items():
        camera['source_config']=b['scene']['cameras'][name];camera['config_changed']=False
    write(tmp_path/'execution_bundle.json',b)
    worker=read(OLD/'worker_result.json');worker.update(case_id=selected['id'],stage='video',
        bundle=binding(tmp_path/'execution_bundle.json'),state_indices=list(range(source['frame_count'])),
        native_execution=False,env_step_calls=0,cameras=b['cameras'],render_protocol=ref,
        render_case_id=selected['id'],fixture_only_not_a_new_render=True)
    write(tmp_path/'worker_result.json',worker)
    return tmp_path


def rebind_bundle(folder):
    worker=read(folder/'worker_result.json');worker['bundle']=binding(folder/'execution_bundle.json')
    overwrite_fixture(folder/'worker_result.json',worker)


def overwrite_fixture(path,value):
    assert 'pytest-' in str(path) and not path.is_relative_to(ROOT)
    path.write_text(json.dumps(value,indent=2)+'\n')


def test_real_old_media_wrapping_cannot_invent_new_pixel_visibility(retained_media_fixture):
    result=validator.validate(retained_media_fixture)
    assert result['passed'],result['errors']
    assert result['pixel_readability']['status']=='unknown'
    assert result['generation_ready'] is False and result['screen_gate_passed'] is False
    assert result['new_physical_executions']==0


@pytest.mark.parametrize('mutation,expected',[
    ('intrinsic','camera_intrinsic_not_predeclared'),
    ('partition','video_bundle_lineage_changed'),
    ('mapping','mapped_frame_identity_changed'),
    ('parent_artifact','native_source_artifact_not_preregistered')])
def test_real_media_lineage_corruption_rejected(retained_media_fixture,mutation,expected):
    folder=retained_media_fixture;b=read(folder/'execution_bundle.json')
    if mutation=='intrinsic':b['cameras']['opposite_high']['parameters']['intrinsic_cv'][0][0]*=1.2
    elif mutation=='partition':b['case']['partition']='changed'
    elif mutation=='mapping':
        mapping=read(folder/'frame_mapping.json');mapping['frames'][2]['source_state_index']=3
        overwrite_fixture(folder/'frame_mapping.json',mapping);b['frame_mapping']=binding(folder/'frame_mapping.json')
    else:
        proto=read(folder/'protocol.json')
        proto['sources'][0]['artifacts']['effective_scene.json']=proto['sources'][0]['artifacts']['effective_task.json']
        overwrite_fixture(folder/'protocol.json',proto);ref=binding(folder/'protocol.json');b['render_protocol']=ref
        reservation=read(folder/'reservation.json');reservation['protocol']=ref;overwrite_fixture(folder/'reservation.json',reservation)
    overwrite_fixture(folder/'execution_bundle.json',b);rebind_bundle(folder)
    result=validator.validate(folder)
    assert not result['passed']
    assert any(expected in error['reason'] for error in result['errors']),result['errors']


def test_full_bbox_detects_clipping_despite_center_in_frame():
    camera={'config':{'width':512,'height':512},'parameters':{
        'extrinsic_cv':[[1,0,0,0],[0,1,0,0],[0,0,1,0]],
        'intrinsic_cv':[[100,0,256],[0,100,256],[0,0,1]]}}
    corners=np.array(list(product([-3.,3.],[-.1,.1],[.9,1.1])))
    result=validator.projected_bounds(corners,np.eye(4),camera)
    assert result['in_front'] and not result['fully_in_frame']
    center=np.asarray(camera['parameters']['intrinsic_cv']) @ corners.mean(axis=0)
    center=center[:2]/center[2]
    assert np.all(center>=0) and np.all(center<512)  # center-only test misses crop


def test_bbox_behind_camera_is_not_visible():
    camera={'config':{'width':512,'height':512},'parameters':{
        'extrinsic_cv':[[1,0,0,0],[0,1,0,0],[0,0,1,0]],
        'intrinsic_cv':[[100,0,256],[0,100,256],[0,0,1]]}}
    corners=np.array(list(product([-.1,.1],[-.1,.1],[-1.1,-.9])))
    result=validator.projected_bounds(corners,np.eye(4),camera)
    assert not result['in_front'] and not result['fully_in_frame']
