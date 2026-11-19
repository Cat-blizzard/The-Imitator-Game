"""Independent, read-only validation of native-state RGB derivatives.

This admits no new physical execution and changes no native physical rule. It
checks the extra evidence a saved-state renderer must supply before conditions
may be compiled. Image checks establish codec/endpoint consistency, not task
success or pixel-level physical truth. No GPU, simulator, API, or output writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from resimpair_grasp.common import binding, read, resolve

VERSION = 'independent_native_render_lineage_v2'
EXPECTED_CAMERAS = {'robot_side', 'opposite_high'}
READBACK_TOLERANCES = {'state_max_abs': 1e-6, 'qpos_max_abs': 1e-6,
                       'object_state_max_abs': 1e-6, 'TCP_max_abs': 1e-5}
PTS_TOLERANCE_SECONDS = 1e-5
ENDPOINT_MIN_PSNR_DB = 30.0
ENDPOINT_MAX_MAE = 5.0


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _same(left, right):
    return resolve(left) == resolve(right)


def camera_calibration(camera, planned):
    """Cross-check recorded sensor matrices against the declared static camera."""
    _require(camera['config'] == planned, 'camera_configuration_not_predeclared')
    config = camera['config']; params = camera['parameters']
    intrinsic = np.asarray(params['intrinsic_cv'], float)
    transform = np.asarray(params['cam2world_gl'], float)
    extrinsic = np.asarray(params['extrinsic_cv'], float)
    _require(intrinsic.shape == (3, 3) and transform.shape == (4, 4)
             and extrinsic.shape == (3, 4), 'camera_matrix_shape_invalid')
    _require(all(np.isfinite(x).all() for x in (intrinsic, transform, extrinsic)),
             'camera_matrix_nonfinite')
    _require(np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0),
             'camera_transform_not_homogeneous')
    rotation = transform[:3, :3]
    _require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
             and abs(np.linalg.det(rotation) - 1) < 1e-6, 'camera_rotation_invalid')
    expected_cv = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(transform)
    _require(np.allclose(extrinsic, expected_cv[:3], atol=1e-5, rtol=0),
             'camera_extrinsic_cv_gl_mismatch')
    eye = np.asarray(config['eye'], float); target = np.asarray(config['target'], float)
    _require(eye.shape == (3,) and target.shape == (3,) and np.isfinite(eye).all()
             and np.isfinite(target).all() and np.linalg.norm(target-eye) > 1e-9,
             'camera_look_at_invalid')
    forward = (target-eye) / np.linalg.norm(target-eye)
    _require(np.allclose(transform[:3, 3], eye, atol=1e-6, rtol=0)
             and np.allclose(-transform[:3, 2], forward, atol=1e-6, rtol=0),
             'camera_pose_not_predeclared')
    right = np.cross(forward, [0., 0., 1.])
    _require(np.linalg.norm(right) > 1e-9, 'camera_vertical_up_ambiguous')
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    _require(np.allclose(rotation[:, 0], right, atol=1e-6, rtol=0)
             and np.allclose(rotation[:, 1], up, atol=1e-6, rtol=0),
             'camera_roll_not_predeclared')
    _require(config['width'] > 0 and config['height'] > 0 and 0 < config['fov'] < math.pi,
             'camera_dimensions_or_fov_invalid')
    focal = config['height'] / (2*math.tan(config['fov']/2))
    expected_k = [[focal, 0, config['width']/2], [0, focal, config['height']/2], [0, 0, 1]]
    _require(np.allclose(intrinsic, expected_k, atol=1e-4, rtol=0),
             'camera_intrinsic_not_predeclared')
    return {'passed': True, 'extrinsic_max_abs_error': float(np.abs(extrinsic-expected_cv[:3]).max()),
            'intrinsic_max_abs_error': float(np.abs(intrinsic-expected_k).max())}


def decode_media(camera, timestamps):
    """Decode every frame; use strict absolute clock tolerance and real pixels."""
    import av
    from PIL import Image
    video = resolve(camera['video']); config = camera['config']
    pts = []; first = last = None
    with av.open(str(video)) as clip:
        _require(len(clip.streams.video) == 1, 'one_video_stream_required')
        stream = clip.streams.video[0]
        _require(stream.width == config['width'] and stream.height == config['height'],
                 'video_resolution_not_predeclared')
        for frame in clip.decode(video=0):
            _require(frame.pts is not None and frame.time_base is not None, 'video_frame_missing_pts')
            pts.append(float(frame.pts*frame.time_base))
            pixels = frame.to_ndarray(format='rgb24')
            if first is None: first = pixels
            last = pixels
    pts = np.asarray(pts)
    _require(pts.shape == timestamps.shape and np.isfinite(pts).all()
             and np.allclose(pts, timestamps, atol=PTS_TOLERANCE_SECONDS, rtol=0),
             'video_pts_not_original_native_clock')
    _require(camera['frame_count'] == len(pts) and camera['synchronous_with_measurements'] is True
             and camera['capture_mode'] == 'saved_state_rerender'
             and camera['source_state_index_equals_video_frame_index'] is True,
             'video_capture_declaration_mismatch')
    expected_fps = 1.0 / float(timestamps[1]-timestamps[0])
    _require(abs(camera['fps'] - expected_fps) < 1e-9, 'video_declared_fps_mismatch')
    endpoints = []
    for label, expected, opposite in [('initial', first, last), ('final', last, first)]:
        with Image.open(resolve(camera[label+'_image'])) as image:
            actual = np.asarray(image.convert('RGB'), dtype=float)
        _require(actual.shape == expected.shape, 'endpoint_image_shape_mismatch:'+label)
        difference = actual-expected.astype(float)
        mse = float(np.square(difference).mean()); mae = float(np.abs(difference).mean())
        opposite_mse = float(np.square(actual-opposite.astype(float)).mean())
        psnr = 10*math.log10(255**2/mse) if mse else 300.
        _require(psnr >= ENDPOINT_MIN_PSNR_DB and mae <= ENDPOINT_MAX_MAE,
                 'endpoint_image_not_codec_consistent:'+label)
        _require(mse <= opposite_mse+1e-9, 'endpoint_image_closer_to_opposite_end:'+label)
        endpoints.append({'endpoint': label, 'passed': True, 'MAE': mae, 'PSNR_dB': psnr,
                          'expected_MSE': mse, 'opposite_MSE': opposite_mse,
                          'visually_distinguishable_endpoints': abs(mse-opposite_mse) > 1e-9})
    return {'passed': True, 'decoded_frames': len(pts),
            'maximum_PTS_error_seconds': float(np.abs(pts-timestamps).max()), 'endpoints': endpoints}


def _inspect(directory, result):
    directory = Path(directory).resolve()
    render_path = directory/'execution_bundle.json'; render = read(render_path)
    result['render_bundle'] = binding(render_path)
    _require(render['schema'] == 'resimpair_saved_state_render_bundle_v1'
             and render['native_execution'] is False and render['physical_execution_count'] == 0
             and render['new_trajectory'] is False and render['saved_state_rerender'] is True,
             'explicit_zero_physics_derivative_required')
    parent_path = resolve(render['source_execution']); parent = read(parent_path)
    result['source_native_execution'] = render['source_execution']
    _require(parent['native_execution'] is True and parent['physical_execution_count'] == 1
             and parent['planning_failure'] is None, 'successful_native_parent_required')
    artifacts = read(directory/'source_artifacts.json')
    for ref in artifacts.values(): resolve(ref)
    _require(_same(artifacts['execution_bundle.json'], render['source_execution']),
             'render_source_artifact_parent_mismatch')
    for name in ['effective_scene.json', 'effective_task.json', 'deferred_render_manifest.json',
                 'deferred_render_states.npz', 'measurements.npz']:
        _require(resolve(artifacts[name]) == parent_path.parent/name, 'source_artifact_not_native_parent:'+name)
    declaration = read(resolve(parent['planning_declaration']))
    _require(declaration['scope'] == 'before_physical_execution' and declaration['case'] == parent['case']
             and _same(declaration['event_protocol'], parent['event_protocol']),
             'before_execution_declaration_mismatch')
    scene_ref = declaration.get('effective_scene', parent['case']['scene'])
    task_ref = declaration.get('effective_task', parent['case']['task'])
    scene = read(resolve(scene_ref)); task = read(resolve(task_ref))
    _require(_same(scene_ref, artifacts['effective_scene.json'])
             and read(resolve(artifacts['effective_task.json'])) == task == parent['task_intent'],
             'source_effective_scene_task_mismatch')
    resolve(parent['case']['source_request'])
    _require(render['case'] == parent['case'] and render['task_intent'] == task
             and render['source_execution_id'] == parent['case']['execution_id']
             and render['source_native_execution'] is True and render['source_physical_execution_count'] == 1,
             'render_changed_native_identity_partition_or_task')
    for key in ('measurements', 'event_protocol', 'planning_declaration', 'planned_paths'):
        _require(_same(render[key], parent[key]), 'render_changed_native_evidence:'+key)
    _require(_same(parent['measurements'], artifacts['measurements.npz']), 'native_measurement_binding_mismatch')
    _require({k:v for k,v in parent['scene'].items() if k != 'cameras'} ==
             {k:v for k,v in scene.items() if k != 'cameras'}, 'native_scene_differs_from_predeclared_scene')
    _require({k:v for k,v in render['scene'].items() if k != 'appearance'} ==
             {k:v for k,v in scene.items() if k != 'appearance'}, 'render_changed_nonappearance_scene')
    _require(render['object_ids'] == parent['object_ids'] == [o['id'] for o in scene['objects']],
             'source_actor_order_changed')
    result['checks']['source_lineage_identity_partition'] = True
    manifest = read(resolve(artifacts['deferred_render_manifest.json']))
    _require(_same(manifest['physics_states'], artifacts['deferred_render_states.npz'])
             and _same(manifest['original_scene'], scene_ref)
             and manifest['planned_cameras'] == scene['cameras'], 'native_saved_state_manifest_mismatch')
    declared_mapping = resolve(render['frame_mapping'])
    # Packaging may make byte-identical copies, but may not change mapped data.
    _require(binding(directory/'frame_mapping.json')['sha256'] == render['frame_mapping']['sha256'],
             'frame_mapping_not_declared_binding')
    mapping = read(declared_mapping)
    _require(_same(mapping['source_execution'], render['source_execution'])
             and _same(mapping['source_measurements'], parent['measurements']), 'mapping_native_source_mismatch')
    _require(_same(mapping['source_states'], manifest['physics_states']), 'mapping_states_not_original_manifest')
    with np.load(resolve(manifest['physics_states']), allow_pickle=False) as data:
        states = data['physics_states']; timestamps = data['timestamps_seconds']
    with np.load(resolve(parent['measurements']), allow_pickle=False) as data:
        measured_time = data['timestamps_seconds']; actions = data['native_commands']
    _require(states.ndim == 3 and states.shape[1] == 1 and len(states) >= 2
             and timestamps.shape == (len(states),) and np.isfinite(states).all()
             and np.isfinite(timestamps).all() and timestamps[0] == 0
             and np.array_equal(timestamps, measured_time) and len(actions) == len(states)-1
             and np.all(np.diff(timestamps)>0), 'saved_states_do_not_match_native_measurement_clock')
    _require(np.allclose(np.diff(timestamps), timestamps[1]-timestamps[0], atol=1e-10, rtol=0),
             'native_control_clock_not_uniform')
    _require(mapping['env_step_calls'] == 0 and mapping['all_states_rendered_once'] is True
             and mapping['contact_signals_recomputed'] is False and len(mapping['frames']) == len(states),
             'mapping_not_complete_nonphysical_render')
    maximum = {k: 0. for k in READBACK_TOLERANCES}
    for index, row in enumerate(mapping['frames']):
        _require(row['frame'] == index and row['source_state_index'] == index
                 and row['state_sha256'] == hashlib.sha256(np.ascontiguousarray(states[index]).tobytes()).hexdigest()
                 and row['source_timestamp_seconds'] == float(timestamps[index])
                 and set(row['cameras']) == EXPECTED_CAMERAS, 'mapping_frame_identity_mismatch')
        _require(set(row['errors']) == set(READBACK_TOLERANCES), 'mapping_missing_readback_error_dimension')
        for key, tolerance in READBACK_TOLERANCES.items():
            value = row['errors'][key]
            _require(np.isfinite(value) and 0 <= value <= tolerance, 'saved_state_readback_exceeds_tolerance:'+key)
            maximum[key] = max(maximum[key], float(value))
    result['checks']['original_state_mapping_and_all_readbacks'] = True
    result['maximum_readback_errors'] = maximum
    result['frame_mapping'] = binding(declared_mapping)
    _require(set(render['cameras']) == set(scene['cameras']) == EXPECTED_CAMERAS,
             'both_predeclared_cameras_required')
    for name, camera in render['cameras'].items():
        calibration = camera_calibration(camera, scene['cameras'][name])
        media = decode_media(camera, timestamps)
        result['cameras'].append({'name': name, 'calibration': calibration, 'media': media})
    result['checks']['all_decoded_pts_calibrations_endpoints'] = True
    result['frames_per_camera'] = len(states)
    result['dataset_partition'] = parent['case']['partition']
    result['split_group'] = parent['case']['split_group']


def validate(directory):
    """Return auditable evidence; fail closed on malformed or unverifiable inputs."""
    result = {'schema': VERSION, 'validator': binding(__file__), 'passed': False,
              'checks': {}, 'errors': [], 'cameras': [], 'new_physical_executions': 0,
              'new_independent_trajectories': 0, 'api_calls': 0,
              'native_physics_acceptance_replaced': False,
              'scope': 'Additional source/state/media lineage only. Original independent physical acceptance remains mandatory. Pixel/state correspondence relies on renderer evidence; endpoint pixel checks are not semantic task or 3D verification.',
              'tolerances': {'state_readback': READBACK_TOLERANCES,
                             'PTS_absolute_seconds': PTS_TOLERANCE_SECONDS,
                             'endpoint_min_PSNR_dB': ENDPOINT_MIN_PSNR_DB,
                             'endpoint_max_MAE': ENDPOINT_MAX_MAE,
                             'endpoint_expected_MSE_must_not_exceed_opposite_MSE': True}}
    try:
        _inspect(directory, result)
        result['passed'] = True
    except Exception as exc:
        result['errors'].append({'type': type(exc).__name__, 'reason': str(exc)})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.execution), indent=2, ensure_ascii=False))
