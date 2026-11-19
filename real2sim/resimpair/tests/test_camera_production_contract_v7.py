"""CPU tests over the two actual registered production preset declarations."""
from copy import deepcopy
import pytest
from resimpair_grasp.common import read,resolve,binding
from resimpair_pickplace import camera_production_contract_v7 as contract
from resimpair_pickplace import render_lineage_validator_v5 as retained
from resimpair_pickplace import render_lineage_validator_v7 as candidate


def inputs():
    protocol=read(resolve(contract.REGISTERED_PROTOCOL));independent=read(resolve(protocol['independent_protocol']))
    case=protocol['cases'][0];source=read(resolve(case['source_execution']))
    declaration=read(resolve(source['planning_declaration']))
    source_scene=read(resolve(declaration['effective_scene']))
    return read(resolve(case['scene'])),source_scene,protocol,independent,case,case['source_execution']


def test_registered_two_sources_and_frozen_visibility_rules():
    scene,source,protocol,independent,case,ref=inputs()
    row=contract.configuration(scene,source,protocol,independent,case,ref)
    assert row['same_source_512_RGB_available'] is False
    assert candidate.RULE==retained.RULE
    assert candidate.FROZEN_CORE_SHA256==binding(candidate.base.__file__)['sha256']
    assert candidate.FROZEN_RESOLUTION_SHA256==binding(contract.__file__)['sha256']


@pytest.mark.parametrize('change',['source','eye','fov','resolution','robot_side','unregistered_protocol'])
def test_rejects_unregistered_change(change):
    scene,source,protocol,independent,case,ref=deepcopy(inputs())
    if change=='source':ref['sha256']='0'*64
    elif change=='eye':scene['cameras']['opposite_high']['eye'][0]+=.001
    elif change=='fov':scene['cameras']['opposite_high']['fov']+=.001
    elif change=='resolution':scene['cameras']['opposite_high']['width']=1279
    elif change=='robot_side':scene['cameras']['robot_side']['width']=720
    else:protocol['unregistered']='yes'
    with pytest.raises(Exception):contract.configuration(scene,source,protocol,independent,case,ref)


def test_actual_screen_calibration_and_per_pixel_rays():
    scene,source,protocol,independent,case,ref=inputs()
    row=contract.configuration(scene,source,protocol,independent,case,ref)
    worker=read(resolve(case['scene']).parent/'worker_result.json')
    proof=contract.actual_calibration(worker['cameras'],row)
    assert proof['passed'] and len(proof['pixel_ray_tests'])==2
    assert {x['pixel_centres'] for x in proof['pixel_ray_tests']}=={512*512,1280*720}
    assert not proof['resolution_improvement_comparison_claimed']


@pytest.mark.parametrize('field',['intrinsic_cv','extrinsic_cv','cam2world_gl'])
def test_rejects_actual_matrix_tampering(field):
    scene,source,protocol,independent,case,ref=inputs()
    row=contract.configuration(scene,source,protocol,independent,case,ref)
    cameras=read(resolve(case['scene']).parent/'worker_result.json')['cameras']
    cameras['opposite_high']['parameters'][field][0][0]+=.01
    with pytest.raises(Exception):contract.actual_calibration(cameras,row)
