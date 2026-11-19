"""Independent scene-domain render evidence checks, without simulation or APIs.

``passed`` is source/state/calibration/media integrity. Pixel visibility is a
separate result; it cannot certify aesthetic quality, task semantics or human
video quality. Original native physical acceptance is retained unchanged.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import binding, read, resolve
from .render_lineage_validator_v2 import (
    READBACK_TOLERANCES, _require, _same, camera_calibration, decode_media)

VERSION = 'independent_scene_domain_lineage_visibility_v3'
BUNDLE_VERSION = 'resimpair_saved_state_scene_domain_bundle_v2'
CAMERAS = {'robot_side', 'opposite_high'}
VISUAL_FIELDS = {'appearance', 'scene_domain', 'cameras'}
CAMERA_LIMITS = {'eye': np.array([.10, .13, .09]), 'target': np.array([.025, .04, .03]),
                 'fov_min': -.025, 'fov_max': .13}


def _refs(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values(): yield from _refs(child)
    elif isinstance(value, list):
        for child in value: yield from _refs(child)


def _without_visual(scene):
    return {k:v for k,v in scene.items() if k not in VISUAL_FIELDS}


def _baseline_values(scene):
    # Digest repair is metadata only; every actual material/light value remains.
    value = dict(scene)
    if isinstance(value.get('appearance'), dict):
        value['appearance'] = {k:v for k,v in value['appearance'].items() if k != 'physical_scene_digest'}
    return value


@lru_cache(maxsize=128)
def _visual_corners(path, sha256):
    import trimesh
    asset = resolve({'path':path, 'sha256':sha256})
    mesh = trimesh.load(asset, force='scene', process=False)
    bounds = np.asarray(mesh.bounds, float)
    _require(bounds.shape == (2, 3) and np.isfinite(bounds).all() and np.all(bounds[1]>bounds[0]),
             'visual_mesh_has_no_finite_extent')
    return np.array(list(product(*zip(bounds[0],bounds[1]))), float)


def projected_bounds(corners, transform, camera):
    """Project all eight bound visual-mesh bbox corners, not just its center."""
    world = corners @ transform[:3,:3].T + transform[:3,3]
    params = camera['parameters']; extrinsic = np.asarray(params['extrinsic_cv'], float)
    xyz = world @ extrinsic[:,:3].T + extrinsic[:,3]
    if not np.isfinite(xyz).all() or np.any(xyz[:,2] <= 1e-8):
        return {'in_front':False,'fully_in_frame':False,'bbox_pixels':None,'area_pixels':0.}
    projected = xyz @ np.asarray(params['intrinsic_cv'],float).T
    uv = projected[:,:2]/projected[:,2:3]; low=uv.min(axis=0); high=uv.max(axis=0)
    config=camera['config']
    inside = bool(np.all(low>=0) and high[0]<=config['width']-1 and high[1]<=config['height']-1)
    return {'in_front':True,'fully_in_frame':inside,'bbox_pixels':[low.tolist(),high.tolist()],
            'area_pixels':float(np.prod(np.maximum(high-low,0)))}


def _pose_transform(state):
    pose=np.eye(4); pose[:3,3]=state[:3]
    pose[:3,:3]=Rotation.from_quat(np.r_[state[4:7],state[3]]).as_matrix()
    return pose


def _visibility(context, result):
    """Read actual raster IDs; geometry and visibility stay separate metrics."""
    directory, source, scene, task, arrays, indices, cameras, source_row, case = (
        context[k] for k in ['directory','source','scene','task','arrays','indices','cameras','source_row','case'])
    visibility_ref = context['record'].get('visibility_manifest')
    if visibility_ref is None:
        result['pixel_readability']={'status':'unknown','reason':'missing_actual_segmentation_evidence'}
        return
    manifest=read(resolve(visibility_ref)); segmentation_path=resolve(manifest['segmentation'])
    result['visibility_manifest']=visibility_ref
    mapping=manifest['entity_mapping']
    _require(set(manifest['arrays'])==CAMERAS, 'segmentation_camera_set_changed')
    _require(set(manifest['segmentation_formats'])==CAMERAS,'segmentation_shader_evidence_missing')
    for fmt in manifest['segmentation_formats'].values():
        if fmt['shader_pack']=='minimal':
            _require(fmt['segmentation_channel_index']==3 and fmt['source_texture']=='PositionSegmentation',
                     'minimal_shader_actor_id_channel_not_declared')
        else:
            _require(fmt['shader_pack'] in ('default','rt') and fmt['segmentation_channel_index']==1
                     and fmt['source_texture']=='Segmentation','actor_id_channel_not_declared')
        _require(fmt['stored_channels']==1,'segmentation_must_store_actor_ids_only')
    _require(manifest['state_indices']==indices and manifest['RGB_and_segmentation_same_get_obs'] is True
             and manifest['all_frames_included']==(case['stage']=='video')
             and manifest['physical_execution_count']==0,'segmentation_state_or_capture_mismatch')
    _require(_same(manifest['source_execution'],source_row['source_execution'])
             and _same(manifest['render_scene'],case['scene'])
             and _same(manifest['render_protocol'],context['record']['render_protocol'])
             and manifest['render_case_id']==case['id'],'segmentation_not_bound_to_this_render')
    object_map=mapping['objects']; support_map=mapping['supports']; robot_map=mapping['robot_links']
    operated=set(task['object_ids']); operations=task['operations']
    targets={op['target']['support'] for op in operations if op['target']['support']!='table'}
    critical=operated|targets
    _require(operated<=set(object_map) and targets<=set(support_map),'segmentation_missing_task_entity_mapping')
    entity_names={int(row['entity_id']):row['name'] for row in mapping['all_entities']}
    for name,ids in {**object_map,**support_map}.items():
        _require(ids and all(type(i) is int and i>0 and i in entity_names
                            and entity_names[i].endswith(name) for i in ids),
                 'segmentation_logical_entity_mapping_not_actual_scene:'+name)
    entities={}
    for obj in source['scene']['objects']:
        descriptor=read(resolve(obj['asset_descriptor'])); visual=descriptor['visual']
        entities[obj['id']]={'corners':_visual_corners(visual['path'],visual['sha256']),
            'object_index':source['object_ids'].index(obj['id']), 'visual':visual,
            'seg_ids':object_map.get(obj['id'],[])}
    native_containers=read(resolve(source_row['artifacts']['native_containers.json']))['containers']
    for item in native_containers:
        descriptor=read(resolve(item['descriptor'])); visual=descriptor['visual']
        entities[item['id']]={'corners':_visual_corners(visual['path'],visual['sha256']),
            'fixed_transform':np.asarray(item['world_pose'],float),'visual':visual,
            'seg_ids':support_map.get(item['id'],[])}
    _require(critical<=set(entities),'missing_bound_visual_geometry_for_task_entity')
    structural_ids=[]
    for name,ids in mapping.get('scene_domain',{}).items():
        if name.split('/')[-1] not in ('table_surface','catalog_backdrop'):structural_ids.extend(ids)
    active_hand_ids={}
    for op in operations:
        arm='arm'+str(op['arm']); links=robot_map.get(arm,{})
        active_hand_ids[arm]=[i for name,ids in links.items() if 'finger' in name or 'hand' in name for i in ids]
        _require(active_hand_ids[arm],'missing_active_hand_segmentation_mapping')
    frame_rows=[]; key_indices=source_row['screen_state_indices']
    with np.load(segmentation_path,allow_pickle=False) as data:
        _require(np.array_equal(data['source_state_indices'],indices),'segmentation_array_state_indices_changed')
        raster={name:data[key] for name,key in manifest['arrays'].items()}
        for name,pixels in raster.items():
            cfg=cameras[name]['config']
            _require(pixels.shape==(len(indices),cfg['height'],cfg['width']) and np.issubdtype(pixels.dtype,np.integer),
                     'segmentation_shape_or_dtype_invalid')
        for local,index in enumerate(indices):
            row={'source_state_index':index,'keyframe':index in key_indices,'views':{}}
            for name,camera in cameras.items():
                pixels=raster[name][local]; projections={}
                for entity,item in entities.items():
                    transform=item.get('fixed_transform')
                    if transform is None:transform=_pose_transform(arrays['object_states'][index,item['object_index']])
                    projection=projected_bounds(item['corners'],transform,camera)
                    visible=int(np.count_nonzero(np.isin(pixels,item['seg_ids'])))
                    projections[entity]={**projection,'critical':entity in critical,'visible_pixels':visible,
                        'visible_pixels_over_bbox_area':visible/max(projection['area_pixels'],1.)}
                row['views'][name]={'entities':projections,
                    'active_hand_pixels':{arm:int(np.count_nonzero(np.isin(pixels,ids))) for arm,ids in active_hand_ids.items()},
                    'structural_pixels':int(np.count_nonzero(np.isin(pixels,structural_ids)))}
            frame_rows.append(row)
    bbox_ok=all(view['entities'][entity]['fully_in_frame'] for row in frame_rows
                for view in row['views'].values() for entity in critical)
    key_rows={row['source_state_index']:row for row in frame_rows if row['keyframe']}
    _require(set(key_rows)==set(key_indices),'segmentation_missing_registered_keyframes')
    findings=[]; hard_failure=not bbox_ok; unknown=False
    for position,index in enumerate(key_indices):
        row=key_rows[index]
        for entity in operated:
            count=max(v['entities'][entity]['visible_pixels'] for v in row['views'].values())
            contained=any(op['object']==entity and op['target']['support']!='table' for op in operations)
            status='confirmed' if count>=16 else ('unknown' if position==2 and contained else 'failed')
            unknown|=status=='unknown';hard_failure|=status=='failed'
            findings.append({'entity':entity,'source_state_index':index,'visible_pixels_best_view':count,
                'minimum_pixels':16,'status':status,
                'reason':'natural_terminal_container_occlusion_possible' if status=='unknown' else None})
        for entity in targets:
            count=max(v['entities'][entity]['visible_pixels'] for v in row['views'].values())
            hard_failure|=count<64
            findings.append({'entity':entity,'source_state_index':index,'visible_pixels_best_view':count,
                             'minimum_pixels':64,'status':'confirmed' if count>=64 else 'failed'})
        if position==1:
            for arm in active_hand_ids:
                count=max(v['active_hand_pixels'][arm] for v in row['views'].values())
                hard_failure|=count<16
                findings.append({'entity':arm+'_hand','source_state_index':index,'visible_pixels_best_view':count,
                                 'minimum_pixels':16,'status':'confirmed' if count>=16 else 'failed'})
    structure_required=case['variant'] in ('environment_3d','environment_3d_camera')
    structure_ok=not structure_required or all(max(v['structural_pixels'] for v in row['views'].values())>=1024
                                             for row in key_rows.values())
    hard_failure|=not structure_ok
    status='failed' if hard_failure else 'unknown' if unknown else 'confirmed'
    result['pixel_readability']={'status':status,'full_visual_bbox_in_frame':bbox_ok,
        'actual_3d_structural_pixels_present':structure_ok,'keyframe_findings':findings,
        'per_frame_geometry_and_visible_pixels':frame_rows,
        'scope':'Minimum observed pixel visibility only; no aesthetic, contact-semantic or human-video quality certification.',
        'geometry_basis':'Bound original visual mesh bbox, transformed by original measured poses; eight corners.'}
    result['screen_gate_passed']=bool(result['passed'] and not hard_failure)
    result['generation_ready']=bool(result['passed'] and status=='confirmed')


def _inspect(directory,result):
    directory=Path(directory).resolve();worker=read(directory/'worker_result.json')
    reservation=read(directory/'reservation.json');protocol=read(resolve(reservation['protocol']))
    independent_ref=protocol['independent_protocol']; independent=read(resolve(independent_ref))
    case=next(c for c in protocol['cases'] if c['id']==worker['case_id'])
    prereg=next(c for c in independent['cases'] if c['id']==case['id'])
    _require(case['source_id']==prereg['source_id'] and case['stage']==prereg['stage']
             and case['variant']==prereg['variant'] and case['seed']==prereg['seed'],'render_case_not_preregistered')
    _require(reservation['case']==case,'reservation_case_changed')
    for ref in protocol['code'].values():resolve(ref)
    for ref in independent['frozen_original_evaluation_dependencies'].values():resolve(ref)
    source_row=next(s for s in protocol['sources'] if s['id']==case['source_id'])
    expected_source=next(s for s in independent['sources'] if s['id']==case['source_id'])
    _require(_same(source_row['source_execution'],expected_source['source_execution']),'source_not_preregistered')
    source_row={**source_row,'screen_state_indices':expected_source['screen_state_indices']}
    source_path=resolve(source_row['source_execution']);source=read(source_path)
    for ref in source_row['artifacts'].values():resolve(ref)
    for name,ref in expected_source['source_artifacts'].items():
        _require(_same(source_row['artifacts'][name],ref),'native_source_artifact_not_preregistered:'+name)
    for name in ('native_contact_policy.json','planned_paths.json'):
        _require(resolve(source_row['artifacts'][name])==source_path.parent/name,
                 'extra_native_artifact_not_in_source_execution:'+name)
    _require(source['native_execution'] is True and source['physical_execution_count']==1
             and source['planning_failure'] is None,'successful_native_parent_required')
    declaration=read(resolve(source['planning_declaration']))
    _require(declaration['scope']=='before_physical_execution' and declaration['case']==source['case']
             and _same(declaration['event_protocol'],source['event_protocol']),'native_declaration_mismatch')
    source_scene=read(resolve(declaration['effective_scene']));task=read(resolve(declaration['effective_task']))
    _require(task==source['task_intent'],'native_task_changed')
    _require(_without_visual(source['scene'])==_without_visual(source_scene)
             and source['object_ids']==[o['id'] for o in source_scene['objects']],
             'native_scene_or_actor_order_differs_from_declaration')
    for label,keys in [('audit.json',['task_events','kinematics','geometry','executed_paths_guarded']),
                       ('container_audit.json',['native_container_geometry','payload_inside_qualified_floor'])]:
        audit=read(resolve(source_row['artifacts'][label]))
        _require(all(audit['checks'][key]['passed'] is True for key in keys),'native_physical_subcheck_failed:'+label)
    goals=read(resolve(source_row['artifacts']['requested_goal_audit.json']))
    _require(goals['goals'] and all(g['passed'] for g in goals['goals']),'original_requested_goal_failed')
    scene=read(resolve(case['scene']))
    _require(_without_visual(scene)==_without_visual(source_scene),'scene_domain_changed_physical_scene')
    _require(set(scene['cameras'])==CAMERAS and scene['cameras']['robot_side']==source_scene['cameras']['robot_side'],
             'original_robot_side_not_preserved')
    if case['variant']=='baseline':
        _require(_baseline_values(scene)==_baseline_values(source_scene),'baseline_visual_values_changed')
        if scene!=source_scene:_require((directory/'metadata_repair.json').exists(),'baseline_metadata_repair_not_recorded')
    if case['variant']!='environment_3d_camera':
        _require(scene['cameras']==source_scene['cameras'],'camera_changed_outside_camera_arm')
    old=source_scene['cameras']['opposite_high'];new=scene['cameras']['opposite_high']
    _require(all(np.all(np.abs(np.asarray(new[k])-old[k])<=CAMERA_LIMITS[k]+1e-9) for k in ['eye','target'])
             and CAMERA_LIMITS['fov_min']-1e-9<=new['fov']-old['fov']<=CAMERA_LIMITS['fov_max']+1e-9
             and all(new[k]==old[k] for k in ['width','height']),'camera_outside_frozen_bounds')
    for ref in _refs(scene):resolve(ref)
    manifest=read(resolve(source_row['artifacts']['deferred_render_manifest.json']))
    _require(_same(manifest['physics_states'],source_row['artifacts']['deferred_render_states.npz'])
             and _same(manifest['original_scene'],declaration['effective_scene'])
             and manifest['planned_cameras']==source_scene['cameras'],'native_deferred_manifest_changed')
    with np.load(resolve(manifest['physics_states']),allow_pickle=False) as data:
        states=data['physics_states'];timestamps=data['timestamps_seconds']
    with np.load(resolve(source['measurements']),allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files}
    _require(np.array_equal(timestamps,arrays['timestamps_seconds']) and len(states)==len(timestamps)
             and len(arrays['native_commands'])==len(states)-1 and np.isfinite(states).all(),
             'native_saved_state_clock_mismatch')
    indices=list(range(len(states))) if case['stage']=='video' else expected_source['screen_state_indices']
    _require(worker['state_indices']==indices and worker['physical_execution_count']==0
             and worker['native_execution'] is False and worker['new_trajectory'] is False
             and worker['env_step_calls']==0,'worker_not_exact_zero_physics_derivative')
    record=read(directory/'execution_bundle.json') if case['stage']=='video' else worker
    if case['stage']=='video':
        _require(record['schema']==BUNDLE_VERSION and record['case']==source['case'] and record['task_intent']==task
                 and record['physical_execution_count']==0 and record['native_execution'] is False
                 and record['new_trajectory'] is False and record['saved_state_rerender'] is True,
                 'video_bundle_lineage_changed')
        _require(record['object_ids']==source['object_ids'] and record['source_native_execution'] is True
                 and record['source_physical_execution_count']==1
                 and record['source_execution_id']==source['case']['execution_id'],
                 'video_native_parent_or_actor_order_changed')
        _require(record['scene']==scene,'bundle_scene_not_frozen_case')
        for key in ['measurements','event_protocol','planning_declaration','planned_paths']:
            _require(_same(record[key],source[key]),'video_native_evidence_changed:'+key)
        _require(_same(worker['bundle'],binding(directory/'execution_bundle.json')),'worker_bundle_mismatch')
        result['render_bundle']=binding(directory/'execution_bundle.json')
    _require(_same(record['source_execution'],source_row['source_execution'])
             and _same(record['render_protocol'],reservation['protocol']) and record['render_case_id']==case['id'],
             'render_protocol_or_source_binding_mismatch')
    mapping=read(directory/'frame_mapping.json')
    if 'frame_mapping' in record:
        _require(record['frame_mapping']['sha256']==binding(directory/'frame_mapping.json')['sha256'],
                 'frame_mapping_not_declared')
        resolve(record['frame_mapping'])
    _require(_same(mapping['source_execution'],source_row['source_execution'])
             and _same(mapping['source_states'],manifest['physics_states'])
             and _same(mapping['source_measurements'],source['measurements']),'mapping_source_changed')
    _require(mapping['env_step_calls']==0 and mapping['contact_signals_recomputed'] is False
             and mapping['all_states_rendered_once']==(case['stage']=='video')
             and len(mapping['frames'])==len(indices),'mapping_incomplete_or_physics_advanced')
    maximum={key:0. for key in READBACK_TOLERANCES}
    for local,(index,row) in enumerate(zip(indices,mapping['frames'])):
        _require(row['frame']==local and row['source_state_index']==index
                 and row['source_timestamp_seconds']==float(timestamps[index])
                 and row['state_sha256']==hashlib.sha256(np.ascontiguousarray(states[index]).tobytes()).hexdigest()
                 and set(row['cameras'])==CAMERAS,'mapped_frame_identity_changed')
        _require(set(row['errors'])==set(READBACK_TOLERANCES),'missing_state_readback_dimension')
        for key,tolerance in READBACK_TOLERANCES.items():
            value=row['errors'][key]
            _require(np.isfinite(value) and 0<=value<=tolerance,'state_readback_tolerance_exceeded:'+key)
            maximum[key]=max(maximum[key],float(value))
    _require(worker['maximum_errors']==maximum,'worker_maximum_errors_not_all_frame_maxima')
    runtime=worker.get('scene_domain_runtime_evidence')
    if case['variant']!='baseline':
        _require(runtime and runtime['collision_shapes_added']==0 and runtime['maniskill_actors_added']==0
                 and runtime['physical_entities_and_collision_geometry_unchanged'] is True
                 and runtime['physical_inventory_before_sha256']==runtime['physical_inventory_after_sha256'],
                 'domain_visual_install_changed_physics')
        _require(all(row['before']==row['after'] for row in runtime['hidden_original_render_components']),
                 'hiding_original_visual_changed_physics')
    cameras=record['cameras'];_require(set(cameras)==CAMERAS,'camera_set_changed')
    from PIL import Image
    camera_results=[]
    for name,camera in cameras.items():
        _require(camera['source_config']==source_scene['cameras'][name]
                 and camera['config_changed']==(camera['config']!=camera['source_config']),
                 'original_camera_calibration_hidden_or_mislabeled')
        calibration=camera_calibration(camera,scene['cameras'][name])
        if case['stage']=='video':media=decode_media(camera,timestamps)
        else:
            for row in mapping['frames']:
                with Image.open(resolve(row['images'][name])) as image:
                    _require(image.size==(camera['config']['width'],camera['config']['height']),
                             'screen_image_dimensions_mismatch')
            _require(camera['frame_count']==len(indices) and camera['synchronous_with_measurements'] is True,
                     'screen_camera_frame_declaration_mismatch')
            media={'passed':True,'bound_images':len(indices),'indices':indices}
        camera_results.append({'name':name,'calibration':calibration,'media':media,
                               'source_config':camera['source_config'],'actual_config':camera['config']})
    result.update(passed=True,stage=case['stage'],render_case_id=case['id'],variant=case['variant'],
        render_protocol=reservation['protocol'],independent_protocol=independent_ref,
        source_native_execution=source_row['source_execution'],source_case=source['case'],
        frame_mapping=binding(directory/'frame_mapping.json'),cameras=camera_results,
        frames_per_camera=len(indices),maximum_readback_errors=maximum,
        checks={'native_source_and_task_unchanged':True,'physical_scene_unchanged':True,
                'all_saved_states_and_clock_bound':True,'actual_camera_calibrations_verified':True,
                'all_media_frames_and_endpoints_verified':True})
    if 'independent_quality_rules' in protocol:
        resolve(protocol['independent_quality_rules'])
        result['independent_quality_rules']=protocol['independent_quality_rules']
    return {'directory':directory,'source':source,'scene':scene,'task':task,'arrays':arrays,'indices':indices,
            'cameras':cameras,'source_row':source_row,'case':case,'record':record}


def validate(directory):
    result={'schema':VERSION,'validator':binding(__file__),'passed':False,'checks':{},'errors':[],
        'pixel_readability':{'status':'unknown','reason':'not_yet_inspected'},
        'generation_ready':False,'screen_gate_passed':False,'new_physical_executions':0,
        'new_independent_trajectories':0,'api_calls':0,'old_physical_and_lineage_validators_unchanged':True}
    try:
        context=_inspect(directory,result)
    except Exception as exc:
        result['errors'].append({'type':type(exc).__name__,'reason':str(exc)})
        return result
    try:
        _visibility(context,result)
    except Exception as exc:
        result['pixel_readability']={'status':'unknown','reason':'visibility_evidence_unverifiable',
                                    'error':{'type':type(exc).__name__,'reason':str(exc)}}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(validate(args.execution),indent=2,ensure_ascii=False))
