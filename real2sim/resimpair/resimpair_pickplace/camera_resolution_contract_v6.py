"""Independent, narrowly registered resolution-only replay contract.

This permits one frozen resolution experiment, not arbitrary camera changes.
It verifies actual calibration and all pixel-centre rays; raster values need
not match after a resolution change. Physical and visibility rules are retained.
"""
from copy import deepcopy

import numpy as np

from resimpair_grasp.common import read, resolve, binding
from .render_lineage_validator_v2 import _require, _same, camera_calibration

REGISTERED_PROTOCOL = {
    'path': 'runs/metric_group_resolution_20261030_v1/protocol.json',
    'sha256': 'fd386794c5adbce072844aa5508d6cd729d03f9d3a52b4e0b08ccb7a46f905a3'}
REGISTERED_ABLATION = {
    'path': 'runs/metric_group_resolution_20261030_v1/resolution_ablation.json',
    'sha256': '70824dd7267268c62ad71bfb5e3e21b13e6a9c532d9833be9078fb7498c0aa71'}
PIXEL_PROJECTION_TOLERANCE = 1e-4


def configuration(scene, source_scene, protocol, independent, case, source_ref):
    """Require exact registered source and only the declared camera edit."""
    _require(protocol == read(resolve(REGISTERED_PROTOCOL)), 'resolution_protocol_not_registered')
    for ref in (protocol.get('camera_resolution_ablation'), independent.get('camera_resolution_ablation'),
                case.get('resolution_ablation')):
        _require(ref is not None and _same(ref, REGISTERED_ABLATION), 'resolution_ablation_not_registered')
    ablation = read(resolve(REGISTERED_ABLATION))
    _require(ablation['schema'] == 'registered_native_camera_resolution_replay_v1'
             and ablation['changed_camera'] == 'opposite_high'
             and ablation['source_dimensions'] == [512, 512]
             and ablation['candidate_dimensions'] == [1280, 720]
             and ablation['eye_target_vertical_FOV_unchanged'] is True
             and ablation['other_camera_unchanged'] is True
             and ablation['geometry_materials_lighting_unchanged'] is True,
             'resolution_declaration_outside_authorized_scope')
    _require(_same(source_ref, ablation['source_execution']), 'resolution_source_changed')
    baseline_protocol = read(resolve(ablation['baseline_protocol']))
    baseline_scene = read(resolve(ablation['baseline_scene']))
    baseline_worker = read(resolve(ablation['baseline_worker_result']))
    baseline_bundle = read(resolve(ablation['baseline_bundle']))
    baseline_screen = read(resolve(ablation['baseline_screen_worker_result']))
    baseline_case = next(c for c in baseline_protocol['cases'] if c['id'] == ablation['baseline_video_case_id'])
    screen_case = next(c for c in baseline_protocol['cases'] if c['id'] == ablation['baseline_screen_case_id'])
    _require(_same(baseline_case['scene'], ablation['baseline_scene'])
             and _same(baseline_case['source_execution'], source_ref)
             and read(resolve(screen_case['scene'])) == baseline_scene
             and _same(screen_case['source_execution'], source_ref)
             and baseline_worker['case_id'] == baseline_case['id']
             and baseline_screen['case_id'] == screen_case['id']
             and baseline_bundle['scene'] == baseline_scene
             and _same(baseline_bundle['source_execution'], source_ref),
             'resolution_baseline_not_bound_to_same_scene_and_native_source')
    previous = read(resolve(ablation['baseline_independent_review']))
    rows = [r for r in previous['rows'] if r['case_id'] == baseline_case['id']]
    _require(len(rows) == 1 and rows[0]['render_candidate_qualified'] is True
             and rows[0]['native_state_TCP_dual_camera_encoded_frames_passed'] is True,
             'resolution_baseline_not_independently_qualified')
    resolve(rows[0]['review'])
    _require(baseline_scene['cameras'] == source_scene['cameras'],
             'resolution_baseline_camera_not_original')
    expected = deepcopy(baseline_scene)
    expected['cameras']['opposite_high'].update(width=1280, height=720)
    sampling = expected['scene_domain']['camera_sampling']
    sampling['actual_configs'] = deepcopy(expected['cameras'])
    sampling['modified_camera'] = 'opposite_high'
    sampling['status'] = 'explicit_resolution_only_candidate'
    sampling['resolution_policy'] = 'registered_native_camera_resolution_replay_v1'
    _require(scene == expected, 'resolution_candidate_changed_more_than_registered_camera_dimensions')
    _require(case['variant'] == 'environment_3d_camera', 'resolution_requires_camera_variant')
    return ablation


def calibration_pair(baseline, candidate):
    """Verify fixed pose, actual K/E, and every pixel-centre ray both ways."""
    _require(set(baseline) == set(candidate) == {'robot_side', 'opposite_high'},
             'resolution_camera_set_changed')
    for name in baseline:
        camera_calibration(baseline[name], baseline[name]['config'])
        camera_calibration(candidate[name], candidate[name]['config'])
        for field in ('extrinsic_cv', 'cam2world_gl'):
            _require(np.array_equal(baseline[name]['parameters'][field], candidate[name]['parameters'][field]),
                     'resolution_actual_pose_changed:' + name + ':' + field)
    _require(baseline['robot_side']['config'] == candidate['robot_side']['config']
             and baseline['robot_side']['parameters'] == candidate['robot_side']['parameters'],
             'resolution_auxiliary_camera_changed')
    old = baseline['opposite_high']; new = candidate['opposite_high']
    expected = deepcopy(old['config']); expected.update(width=1280, height=720)
    _require([old['config']['width'], old['config']['height']] == [512, 512]
             and new['config'] == expected, 'resolution_config_outside_fixed_pose_fov_dimensions')
    k0 = np.asarray(old['parameters']['intrinsic_cv'], float)
    k1 = np.asarray(new['parameters']['intrinsic_cv'], float)
    e0 = np.asarray(old['parameters']['extrinsic_cv'], float)
    e1 = np.asarray(new['parameters']['extrinsic_cv'], float)
    scale = 720 / 512
    affine = np.array([[scale, 0., 280.], [0., scale, 0.], [0., 0., 1.]])
    intrinsic_error = float(np.abs(k1 - affine @ k0).max())
    _require(intrinsic_error <= 1e-4, 'resolution_actual_intrinsic_affine_mismatch')
    tests = []
    # Pixel centres, not RGB interpolation. Two depths also exercise translation.
    for label, width, height, first_k, first_e, second_k, second_e, transform in (
        ('512_to_1280x720', 512, 512, k0, e0, k1, e1, affine),
        ('1280x720_to_512', 1280, 720, k1, e1, k0, e0, np.linalg.inv(affine))):
        maximum = 0.
        for y0 in range(0, height, 64):
            yy, xx = np.mgrid[y0:min(y0 + 64, height), 0:width]
            pixels = np.c_[xx.ravel() + .5, yy.ravel() + .5, np.ones(xx.size)]
            rays = pixels @ np.linalg.inv(first_k).T
            wanted_h = pixels @ transform.T; wanted = wanted_h[:, :2] / wanted_h[:, 2:3]
            for depth in (1., 2.):
                xyz = rays * depth
                world = (xyz - first_e[:, 3]) @ np.linalg.inv(first_e[:, :3]).T
                projected = (world @ second_e[:, :3].T + second_e[:, 3]) @ second_k.T
                actual = projected[:, :2] / projected[:, 2:3]
                maximum = max(maximum, float(np.abs(actual - wanted).max()))
        _require(maximum <= PIXEL_PROJECTION_TOLERANCE, 'resolution_per_pixel_ray_projection_mismatch')
        tests.append({'direction': label, 'pixel_centres': width * height, 'depth_planes': [1., 2.],
                      'maximum_coordinate_error_pixels': maximum})
    return {'passed': True, 'actual_intrinsic_affine_max_abs_error': intrinsic_error,
            'actual_extrinsic_and_cam2world_exactly_unchanged': True,
            'robot_side_calibration_exactly_unchanged': True,
            'pixel_coordinate_transform': affine.tolist(), 'pixel_ray_tests': tests,
            'projection_tolerance_pixels': PIXEL_PROJECTION_TOLERANCE,
            'linear_pixel_scale': scale, 'area_scale_if_unoccluded': scale ** 2,
            'horizontal_field_of_view_widened': True,
            'RGB_pixel_identity_expected': False,
            'algorithm_visibility_or_generation_quality_improvement_claimed': False}


def actual_calibration(cameras, ablation):
    baseline = read(resolve(ablation['baseline_worker_result']))
    return {'declaration': REGISTERED_ABLATION, 'registered_protocol': REGISTERED_PROTOCOL,
            'baseline_worker_result': ablation['baseline_worker_result'],
            **calibration_pair(baseline['cameras'], cameras)}
