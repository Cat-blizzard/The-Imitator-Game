"""Restore a saved attached lift using counter-bound measured geometry.

This is a read-only replay adapter, not a new planner or acceptance rule.
Multiple refreshes can occur at one simulation frame. A request uses the last
refresh whose counter is no greater than its own; future refreshes are never
selected. All same-frame transforms must agree with the bound measurement.
"""
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'counter_bound_measured_lift_restore_v2'
# These only check serialization/kinematic provenance, not task acceptance.
QPOS_ATOL_RAD = 1e-7
ROOT_MATRIX_ATOL = 1e-7
ATTACHMENT_MATRIX_ATOL = 1e-5
FK_POSITION_ATOL_M = 5e-5
FK_ROTATION_ATOL_RAD = 5e-5


def _index(value, label):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative_integer_required:' + label)
    return value


def _pose(value, label):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('finite_transform_required:' + label)
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(pose[:3, :3])-1) > 1e-5):
        raise ValueError('rigid_transform_required:' + label)
    return pose


def select_attachment(request_path, *, expected_request=None):
    """Return a JSON-safe binding/measurement audit; no native planner needed.

    Inputs: an existing ``<counter>_lift_*/request.json`` in an execution's
    ``<operation>_motion`` directory and its bound execution measurements.
    Failure: invalid state, missing/ambiguous refresh, a future-only refresh,
    hash mismatch or any same-frame attachment inconsistent with measurement.
    Output: the exact selected refresh and every other same-frame binding.
    """
    from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk

    request_path = Path(request_path).resolve(strict=True)
    if expected_request is not None and resolve(expected_request).resolve() != request_path:
        raise ValueError('unexpected_request_binding')
    match = re.fullmatch(r'(\d+)_lift(?:_.*)?', request_path.parent.name)
    if request_path.name != 'request.json' or match is None:
        raise ValueError('counter_bound_lift_request_directory_required')
    counter = int(match[1])
    request = read(request_path)
    if request.get('stage') != 'lift' or request.get('attached') is not True:
        raise ValueError('attached_lift_request_required')
    frame = _index(request.get('frame'), 'request.frame')
    current = np.asarray(request.get('current'), dtype=float)
    if current.shape != (9,) or not np.isfinite(current).all():
        raise ValueError('finite_native_panda_current_required')
    _pose(request.get('goal'), 'request.goal')
    motion = request_path.parent.parent
    if not motion.name.endswith('_motion'):
        raise ValueError('operation_motion_directory_required')
    execution = motion.parent
    bundle_path = execution/'execution_bundle.json'
    bundle = read(bundle_path)
    operations = [o for o in bundle['task_intent']['operations']
                  if o['id'] == motion.name.removesuffix('_motion')]
    if len(operations) != 1:
        raise ValueError('unique_bound_operation_required')
    operation = operations[0]
    arm = _index(operation['arm'], 'operation.arm')
    if arm not in (0, 1) or bundle['object_ids'].count(operation['object']) != 1:
        raise ValueError('unique_native_arm_object_required')
    object_index = bundle['object_ids'].index(operation['object'])
    measurements = resolve(bundle['measurements'])
    with np.load(measurements, allow_pickle=False) as arrays:
        if frame >= len(arrays['states']):
            raise ValueError('request_frame_outside_measurements')
        measured_current = arrays['states'][frame, arm*9:(arm+1)*9].astype(float)
        root = _pose(arrays['robot_roots_world'][arm], 'measured.root')
        tcp = _pose(arrays['tcp_world'][frame, arm], 'measured.tcp')
        object_state = arrays['object_states'][frame, object_index].astype(float)
        timestamp = float(arrays['timestamps_seconds'][frame])
        measured_grasp = bool(arrays['grasp'][frame, arm, object_index]) if 'grasp' in arrays else None
    if (measured_current.shape != (9,) or not np.isfinite(measured_current).all()
            or object_state.shape != (13,) or not np.isfinite(object_state).all()
            or not np.isfinite(timestamp)):
        raise ValueError('invalid_measured_state')
    if abs(np.linalg.norm(object_state[3:7])-1) > 1e-5:
        raise ValueError('measured_object_quaternion_not_unit')
    object_pose = np.eye(4)
    object_pose[:3, :3] = Rotation.from_quat(object_state[3:7], scalar_first=True).as_matrix()
    object_pose[:3, 3] = object_state[:3]
    object_in_tcp = np.linalg.inv(tcp) @ object_pose
    q_error = float(np.max(abs(current-measured_current)))
    scene_root = np.eye(4); scene_root[:3, 3] = bundle['scene']['robot_roots'][arm]
    root_error = float(np.max(abs(scene_root-root)))
    fk = world_fk(current[None, :7], root)[0]
    fk_position = float(np.linalg.norm(fk[:3, 3]-tcp[:3, 3]))
    fk_rotation = float(Rotation.from_matrix(fk[:3, :3].T @ tcp[:3, :3]).magnitude())
    if q_error > QPOS_ATOL_RAD or root_error > ROOT_MATRIX_ATOL:
        raise ValueError('request_state_does_not_match_measurement')
    if fk_position > FK_POSITION_ATOL_M or fk_rotation > FK_ROTATION_ATOL_RAD:
        raise ValueError('saved_joint_fk_does_not_match_measured_tcp')

    same_frame = []
    for path in sorted(motion.glob('attachment_*_lift.json')):
        attachment = read(path)
        attachment_frame = _index(attachment.get('frame'), path.name + '.frame')
        if attachment_frame != frame:
            continue
        named = re.fullmatch(r'attachment_(\d+)_lift\.json', path.name)
        if named is None or attachment.get('planning_geometry_only') is not True:
            raise ValueError('invalid_same_frame_attachment_metadata:' + path.name)
        transform = _pose(attachment.get('measured_object_in_tcp'), path.name)
        error = float(np.max(abs(transform-object_in_tcp)))
        if error > ATTACHMENT_MATRIX_ATOL:
            raise ValueError('attachment_disagrees_with_measured_frame:' + path.name)
        item_counter = int(named[1])
        same_frame.append({'binding': binding(path), 'counter': item_counter,
                           'eligible_before_or_at_request': item_counter <= counter,
                           'matrix_max_error_vs_measurement': error,
                           'measured_object_in_tcp': transform.tolist()})
    eligible = [a for a in same_frame if a['eligible_before_or_at_request']]
    if not eligible:
        raise ValueError('no_attachment_at_or_before_request_counter')
    latest_counter = max(a['counter'] for a in eligible)
    latest = [a for a in eligible if a['counter'] == latest_counter]
    if len(latest) != 1:
        raise ValueError('ambiguous_attachment_counter')
    selected = latest[0]
    return {
        'schema': VERSION, 'request': binding(request_path), 'request_counter': counter,
        'bundle': binding(bundle_path), 'measurements': bundle['measurements'],
        'operation': operation['id'], 'arm': arm, 'object': operation['object'], 'frame': frame,
        'timestamp_seconds': timestamp, 'selected_attachment': selected,
        'other_same_frame_attachments': [a for a in same_frame if a is not selected],
        'selection_rule': 'unique_max_attachment_counter_not_greater_than_request_counter',
        'measured_root': root.tolist(), 'measured_object_pose': object_pose.tolist(),
        'measurement_alignment': {'qpos_max_error_rad': q_error, 'scene_root_matrix_max_error': root_error,
                                  'fk_position_error_m': fk_position, 'fk_rotation_error_rad': fk_rotation,
                                  'measured_grasp_at_frame': measured_grasp},
        'serialization_tolerances': {'qpos_rad': QPOS_ATOL_RAD, 'scene_root_matrix': ROOT_MATRIX_ATOL,
                                     'attachment_matrix': ATTACHMENT_MATRIX_ATOL,
                                     'fk_position_m': FK_POSITION_ATOL_M, 'fk_rotation_rad': FK_ROTATION_ATOL_RAD},
        'implementation': binding(__file__), 'physical_execution': False,
        'acceptance_changed': False}


def restore_guard(request_path, *, expected_request=None, evidence_output=None):
    """Return ``(unchanged MeshMotion guard, audit)`` with exact measured root.

    Native FCL/planner reconstruction is CPU-only. No scene, renderer, dynamics
    stepping, control, GPU job, API call or physical-object attachment is made.
    An optional new evidence_output records the returned audit without replacing
    historical files. The original frame-only restorer is untouched.
    """
    import trimesh
    from mplib.pymp.collision_detection import fcl
    from .cartesian_transfer import restore_unattached_guard
    from data_generation_methods.human_to_robot_pipeline.strict_motion import matrix_to_pose7, sampled_mesh_points

    audit = select_attachment(request_path, expected_request=expected_request)
    request_path = resolve(audit['request'])
    motion, execution = request_path.parent.parent, request_path.parent.parent.parent
    request = read(request_path)
    # Reuse the original exact-mesh world constructor; attachment and measured
    # base restoration are installed here before any path is checked.
    guard = restore_unattached_guard(execution, motion, {**request, 'attached': False})
    guard.root = np.asarray(audit['measured_root'], dtype=float)
    guard.planner.set_base_pose(matrix_to_pose7(guard.root))
    table = trimesh.load(motion/'complete_table.obj', force='mesh')
    base = trimesh.load(motion/'robot_panda_link0.obj', force='mesh')
    base.apply_transform(guard.root)
    cloud = sampled_mesh_points(table, 18000, 401)
    inside = np.all((cloud[:, :2] >= base.bounds[0, :2]-.005) &
                    (cloud[:, :2] <= base.bounds[1, :2]+.005), axis=1)
    guard.planner.update_point_cloud(cloud[~inside], radius=.001)
    payload_path = motion/'selected_physical.obj'
    payload_geometry = fcl.load_mesh_as_BVH(str(payload_path), np.ones(3))
    guard.payload_collision = fcl.CollisionObject(payload_geometry)
    guard.objects.append((payload_geometry, guard.payload_collision))
    guard.payload_to_tcp = np.asarray(audit['selected_attachment']['measured_object_in_tcp'], dtype=float)
    guard.selected_world_pose = np.asarray(audit['measured_object_pose'], dtype=float)
    guard.use_attach = True
    audit.update(world=binding(motion/'world.json'),
                 source_meshes=[binding(p) for p in sorted(motion.glob('*.obj'))],
                 retained_guard=binding(ROOT/'resimpair_grasp/motion.py'),
                 reused_world_constructor=binding(ROOT/'resimpair_pickplace/cartesian_transfer.py'),
                 root_source='measurements.robot_roots_world.original_values',
                 payload_geometry_only=True)
    if evidence_output is not None:
        write(evidence_output, audit)
    return guard, audit
