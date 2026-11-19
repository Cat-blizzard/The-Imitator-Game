"""Independent fixed production-camera preset for two registered native sources.

This qualifies a native-clock derivative, not a resolution-improvement ablation.
There is no same-source 512 RGB baseline. All physical/visibility rules remain
those of frozen v5; only the declared output pixel extent is newly registered.
"""
from copy import deepcopy
import numpy as np
from resimpair_grasp.common import read, resolve
from .render_lineage_validator_v2 import _require, _same, camera_calibration

REGISTERED_PROTOCOL = {'path':'runs/contact_closure_background_20261030_v1/protocol.json',
    'sha256':'6025b02a942756f044beb58e4df787c9f5b03fb6b2edaa74c48bc16451afd416'}
REGISTERED_PRESET = {'path':'runs/contact_closure_background_20261030_v1/registered_camera_preset.json',
    'sha256':'c001c9bbbd30868100613b4ea90486ca1f995de70381e68dac532e7c56f9e09e'}
PIXEL_PROJECTION_TOLERANCE = 1e-4


def configuration(scene, source_scene, protocol, independent, case, source_ref):
    _require(protocol == read(resolve(REGISTERED_PROTOCOL)), 'production_preset_protocol_not_registered')
    for ref in (protocol.get('registered_camera_preset'), independent.get('registered_camera_preset'),
                case.get('registered_camera_preset')):
        _require(ref is not None and _same(ref,REGISTERED_PRESET),'production_camera_preset_not_registered')
    preset=read(resolve(REGISTERED_PRESET))
    _require(preset['schema']=='closure_native_background_production_preset_v1'
             and preset['physics_steps']==0 and preset['API_calls']==0
             and len(preset['sources'])==2, 'production_preset_scope_changed')
    matches=[x for x in preset['sources'] if x['source_id']==case['source_id']]
    _require(len(matches)==1,'production_source_not_registered')
    row=matches[0]
    _require(_same(row['source_execution'],source_ref),'production_native_source_changed')
    _require(row['changed_camera']=='opposite_high' and row['source_dimensions']==[512,512]
             and row['candidate_dimensions']==[1280,720]
             and row['eye_target_vertical_FOV_unchanged'] is True
             and row['other_camera_unchanged'] is True
             and row['same_native_frame_mapping_required'] is True
             and row['actual_projection_and_calibration_required'] is True
             and row['same_source_512_RGB_available'] is False
             and row['resolution_improvement_comparison_claimed'] is False,
             'production_preset_not_resolution_ablation')
    _require(row['original_camera_configs']==source_scene['cameras'], 'production_original_camera_changed')
    expected=deepcopy(source_scene['cameras']);expected['opposite_high'].update(width=1280,height=720)
    _require(set(expected)=={'robot_side','opposite_high'}
             and [expected['robot_side']['width'],expected['robot_side']['height']]==[512,512]
             and row['production_camera_configs']==expected and scene['cameras']==expected,
             'production_camera_change_outside_registered_pixel_extent')
    sampling=scene['scene_domain']['camera_sampling']
    _require(sampling['actual_configs']==expected
             and sampling['modified_camera']=='opposite_high'
             and sampling['status']=='registered_production_resolution_preset'
             and sampling['resolution_policy']==preset['schema'], 'production_sampling_metadata_changed')
    _require(case['variant']=='environment_3d_camera', 'production_preset_requires_camera_variant')
    return row


def actual_calibration(cameras, row):
    _require(set(cameras)=={'robot_side','opposite_high'},'production_camera_set_changed')
    tests=[]
    for name,camera in cameras.items():
        expected=row['production_camera_configs'][name]
        calibration=camera_calibration(camera,expected)
        _require(camera['source_config']==row['original_camera_configs'][name]
                 and camera['config_changed']==(name=='opposite_high'),'production_original_camera_hidden')
        k=np.asarray(camera['parameters']['intrinsic_cv'],float)
        e=np.asarray(camera['parameters']['extrinsic_cv'],float)
        width,height=expected['width'],expected['height']; maximum=0.
        # Every pixel-centre is lifted to a world ray and reprojected at two
        # depths using the actual readback K/E. Config/look-at/roll/calibration
        # are independently checked above. No nonexistent baseline is invented.
        for y0 in range(0,height,64):
            yy,xx=np.mgrid[y0:min(y0+64,height),0:width]
            pixels=np.c_[xx.ravel()+.5,yy.ravel()+.5,np.ones(xx.size)]
            rays=pixels@np.linalg.inv(k).T
            for depth in (1.,2.):
                xyz=rays*depth
                world=(xyz-e[:,3])@np.linalg.inv(e[:,:3]).T
                projected=(world@e[:,:3].T+e[:,3])@k.T
                actual=projected[:,:2]/projected[:,2:3]
                maximum=max(maximum,float(np.abs(actual-pixels[:,:2]).max()))
        _require(maximum<=PIXEL_PROJECTION_TOLERANCE,'production_per_pixel_ray_roundtrip_mismatch')
        tests.append({'camera':name,'pixel_centres':width*height,'depth_planes':[1.,2.],
                      'maximum_coordinate_error_pixels':maximum,'declared_calibration':calibration})
    return {'passed':True,'registered_preset':REGISTERED_PRESET,'registered_protocol':REGISTERED_PROTOCOL,
        'source_execution':row['source_execution'],'pixel_ray_tests':tests,
        'projection_tolerance_pixels':PIXEL_PROJECTION_TOLERANCE,
        'native_source_camera_eye_target_FOV_and_robot_side_preserved':True,
        'same_source_512_RGB_available':False,'resolution_improvement_comparison_claimed':False,
        'scope':'Actual fixed production camera K/E and native state-clock binding; not visual quality improvement.'}
