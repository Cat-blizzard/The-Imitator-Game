"""Bounded, read-only queries over retained scene and execution evidence.

No simulator, mesh cooker, GPU, API, planner, auditor or artifact writer runs
here. Independent acceptance is reported from bound evidence, never inferred
from a worker exit code. NPZ cannot actually be memory-mapped; small measurement
arrays are cached with size/mtime invalidation. Dense RGB/segmentation is not read.
"""
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

VERSION = 'resimpair_scene_observe_v1'
ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 32 * 2**20
MAX_NPZ_BYTES = 128 * 2**20
MAX_OBJECTS = 64
MAX_OPERATIONS = 64
MAX_DECISIONS = 64
_PRIVATE_PARTS = {'private', 'secrets', 'credentials', '.ssh', '.aws', '.gnupg'}
# Versioned trust anchors, not case IDs or date-directory discovery rules.
_METRIC_VALIDATOR = ('resimpair_validation/metric_cohort_v1.py',
    '3da0248f0e85df6e62bcca196757ae4d291007dbf481b83821be9652d1900bea')
_METRIC_ACCEPTANCE = ('resimpair_validation/frozen_metric_acceptance_v1.json',
    '1f9b4e3ca9aac69d4a107e7741cec1c3faec55131d86e5f7edba276935359dfd')


def _path(value):
    p = Path(value).expanduser().resolve()
    if any(x.lower() in _PRIVATE_PARTS or x.lower().startswith('.env') for x in p.parts):
        raise ValueError('private_configuration_is_not_observation_evidence')
    return p


def _token(path):
    p = _path(path); s = p.stat()
    return str(p), s.st_size, s.st_mtime_ns


@lru_cache(maxsize=24)
def _json_cached(path, size, mtime):
    if size > MAX_JSON_BYTES:
        raise ValueError('observation_JSON_byte_limit_exceeded')
    return json.loads(Path(path).read_text())


def _read(path, optional=False):
    p = _path(path)
    if optional and not p.is_file():
        return None
    return _json_cached(*_token(p))


@lru_cache(maxsize=192)
def _sha(path, size, mtime):
    if size > MAX_NPZ_BYTES:
        raise ValueError('observation_binding_byte_limit_exceeded')
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def _evidence(path):
    p = _path(path)
    return {'path': str(p), 'sha256': _sha(*_token(p))}


def _bound(ref):
    if isinstance(ref, str):
        p = Path(ref); return _path(p if p.is_absolute() else ROOT / p)
    if not isinstance(ref, dict) or 'path' not in ref:
        raise ValueError('bound_evidence_path_required')
    p = Path(ref['path']); p = _path(p if p.is_absolute() else ROOT / p)
    if ref.get('sha256') and _evidence(p)['sha256'] != ref['sha256']:
        raise ValueError('observation_evidence_hash_mismatch')
    return p


@lru_cache(maxsize=2)
def _npz_cached(path, size, mtime):
    if size > MAX_NPZ_BYTES:
        raise ValueError('measurement_archive_byte_limit_exceeded')
    with ZipFile(path) as archive:
        info = archive.infolist()
        if len(info) > 32 or sum(x.file_size for x in info) > MAX_NPZ_BYTES:
            raise ValueError('measurement_uncompressed_byte_limit_exceeded')
    keys = {'states', 'tcp_world', 'object_states', 'grasp', 'support_impulses',
            'robot_object_force_N', 'cross_robot_force_N', 'native_commands',
            'timestamps_seconds', 'robot_roots_world'}
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files if name in keys}
    for value in result.values():
        if value.dtype.hasobject:
            raise ValueError('object_array_measurements_forbidden')
        value.flags.writeable = False
    return result


def _measurements(execution, bundle):
    ref = bundle.get('measurements') if bundle else None
    p = _bound(ref) if ref else execution / 'measurements.npz'
    if not p.is_file():
        return {}, None
    # In particular, never load deferred full physics or dense segmentation.
    if p.name != 'measurements.npz':
        raise ValueError('only_retained_measurements_archive_supported')
    return _npz_cached(*_token(p)), _evidence(p)


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(x) for x in value]
    if isinstance(value, np.ndarray): return _clean(value.tolist())
    if isinstance(value, np.generic): return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return 'unknown'
    if isinstance(value, Path): return str(value)
    return value


def _execution(value):
    p = _path(value)
    if not p.is_dir():
        raise ValueError('execution_directory_required')
    return p


def _base(execution):
    bundle = _read(execution / 'execution_bundle.json', True) or {}
    scene = _read(execution / 'effective_scene.json', True) or bundle.get('scene') or {}
    task = _read(execution / 'effective_task.json', True) or bundle.get('task_intent') or {}
    if len(task.get('operations', [])) > MAX_OPERATIONS:
        raise ValueError('operation_count_limit_exceeded')
    return bundle, scene, task


def _require_binding(ref, expected, sha256=None):
    if not isinstance(ref, dict) or not re.fullmatch(r'[0-9a-f]{64}', ref.get('sha256', '')):
        raise ValueError('independent_SHA256_binding_required')
    if _bound(ref) != _path(expected) or (sha256 is not None and ref['sha256'] != sha256):
        raise ValueError('independent_binding_target_or_version_mismatch')


def _audit_chain(execution, supplied_path, document):
    """Verify the retained validator's protocol/summary/leaf binding chain.

    A caller-created JSON with a plausible flag is not an independent verdict.
    This checks recorded provenance; it neither re-runs nor replaces acceptance.
    """
    is_summary = 'rows' in document
    directory = supplied_path.parent if is_summary else supplied_path.parent.parent
    final_path, protocol_path = directory / 'FINAL_REVIEW.json', directory / 'PROTOCOL.json'
    leaf_path = directory / execution.name / 'independent_result.json'
    if supplied_path != (final_path if is_summary else leaf_path):
        raise ValueError('canonical_independent_summary_or_leaf_required')
    final = document if is_summary else _read(final_path)
    protocol = _read(protocol_path)
    expected_schema = 'independent_metric_cohort_acceptance_v1'
    if final.get('schema') != expected_schema or protocol.get('schema') != expected_schema:
        raise ValueError('unsupported_independent_acceptance_version')
    if final.get('rules_relaxed') is not False or protocol.get('rules_relaxed') is not False:
        raise ValueError('unchanged_independent_rules_required')
    _require_binding(final.get('protocol'), protocol_path)
    _require_binding(protocol.get('validator'), ROOT / _METRIC_VALIDATOR[0], _METRIC_VALIDATOR[1])
    _require_binding(protocol.get('frozen_acceptance'), ROOT / _METRIC_ACCEPTANCE[0], _METRIC_ACCEPTANCE[1])
    frozen = _read(ROOT / _METRIC_ACCEPTANCE[0])
    dependencies = protocol.get('retained_evaluation_dependencies')
    if not isinstance(dependencies, dict) or dependencies != frozen.get('dependencies') or len(dependencies) > 32:
        raise ValueError('independent_retained_dependencies_mismatch')
    for ref in dependencies.values():
        if not isinstance(ref, dict) or not re.fullmatch(r'[0-9a-f]{64}', ref.get('sha256', '')):
            raise ValueError('independent_dependency_hash_required')
        _bound(ref)
    cohort = execution.parent.parent
    if _path(final.get('cohort_directory', '')) != cohort:
        raise ValueError('independent_cohort_directory_mismatch')
    _require_binding(protocol.get('cohort'), cohort / 'cohort.json')
    _require_binding(protocol.get('execution_protocol'), cohort / 'protocol.json')
    rows = final.get('rows', [])
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError('audit_row_limit_exceeded')
    matches = [row for row in rows if row.get('case_id') == execution.name]
    if len(matches) != 1:
        raise ValueError('unique_case_in_audit_required')
    leaf = _read(leaf_path) if is_summary else document
    if leaf != matches[0]:
        raise ValueError('independent_leaf_and_summary_disagree')
    return leaf


def _audit(execution, supplied=None):
    """Only explicit audit paths or existing independent bridge; no tree search."""
    path = None
    try:
        if supplied is not None:
            path = _path(supplied)
            if path.is_dir():
                choices = [path / 'independent_result.json', path / execution.name / 'independent_result.json']
                path = next((p for p in choices if p.is_file()), None)
        else:
            bridge = _read(execution / 'independent_metric_cli_audit_bridge_v1.json', True)
            if bridge:
                path = _bound(bridge['independent_result'])
        if path is None:
            return None, None, 'no_bound_independent_audit'
        if path.is_relative_to(execution):
            return None, None, 'independent_result_must_be_outside_execution'
        value = _audit_chain(execution, path, _read(path))
        if (value.get('case_id') != execution.name
                or value.get('source_files_unchanged_during_review') is not True
                or value.get('source_execution') is None):
            raise ValueError('independent_audit_source_binding_required')
        source_ref = value['source_execution']
        if not isinstance(source_ref, dict) or not re.fullmatch(r'[0-9a-f]{64}', source_ref.get('sha256', '')):
            raise ValueError('independent_source_SHA256_binding_required')
        source = _bound(source_ref)
        if source != (execution / 'execution_bundle.json').resolve():
            raise ValueError('independent_audit_belongs_to_another_execution')
        if value.get('audit_error') is not None:
            return value, path, 'independent_audit_incomplete_or_error'
        if type(value.get('physical_qualified')) is not bool:
            raise ValueError('explicit_independent_physical_result_required')
        return value, path, None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, path, 'invalid_or_unavailable_audit:' + type(exc).__name__ + ':' + str(exc)


def run_status(execution, *, independent_audit=None):
    """Separate process completion, physical acceptance, RGB and human pairing."""
    e = _execution(execution); bundle, scene, task = _base(e)
    terminal = _read(e / 'terminal.json', True)
    reservation = _read(e / 'reservation.json', True)
    worker = _read(e / 'worker_result.json', True) or {}
    failure = worker.get('failure') or (terminal or {}).get('admission', {}).get('reported_failure')
    if terminal is not None and type(terminal.get('returncode')) is int:
        worker_state = 'completed' if terminal.get('returncode') == 0 else 'failed_or_timed_out'
    elif terminal is not None:
        worker_state = 'unknown_terminal_without_returncode'
    elif reservation is not None:
        worker_state = 'unknown_reserved_without_terminal'
    elif worker or bundle:
        worker_state = 'unknown_without_terminal'
    else:
        worker_state = 'not_submitted'
    audit, ap, audit_problem = _audit(e, independent_audit)
    physical = 'unknown' if audit_problem else ('passed' if audit['physical_qualified'] else 'failed')
    if terminal is None:
        physical = 'unknown'
        audit_problem = 'execution_not_terminal'
    rgb_evidence = (audit or {}).get('RGB_synchronization', {})
    media = []
    for name, camera in list(bundle.get('cameras', {}).items())[:8]:
        if camera.get('video'):
            p = _bound(camera['video']); media.append({'camera': name, 'path': str(p), 'exists': p.is_file()})
    rgb_state = ('passed' if rgb_evidence.get('status') == 'confirmed' and not audit_problem else
                 'failed' if rgb_evidence.get('status') == 'failed' and not audit_problem else
                 'present_unreviewed' if any(x['exists'] for x in media) else 'unknown')
    return _clean({'schema': VERSION, 'execution': str(e), 'worker_status': worker_state,
        'worker_completed': worker_state == 'completed', 'terminal': _evidence(e / 'terminal.json') if terminal else None,
        'physical': {'status': physical, 'reason': audit_problem,
                     'basis': 'bound retained independent audit; no new physics or reacceptance',
                     'provenance_chain': 'verified_metric_cohort_v1' if not audit_problem else 'unknown',
                     'audit': _evidence(ap) if ap and ap.is_file() else None,
                     'subchecks': (audit or {}).get('retained_physical_review', {}).get('physical_checks', {})},
        'execution_failure': {'type': failure.get('type'), 'message': failure.get('message')} if failure else None,
        'RGB': {'status': rgb_state, 'media': media, 'independent_evidence': rgb_evidence},
        'human_pair': {'status': 'unknown', 'reason': 'this_native_observer_has_no_bound_human_reference_audit',
                       'reported_full_pair_flag': (audit or {}).get('full_pair_qualified')},
        'native_physics_execution': bundle.get('native_execution', 'unknown'),
        'requested_operation_count': len(task.get('operations', [])),
        'completed_operations': (_read(e / 'rolling_execution_summary.json', True) or {}).get('completed_operation_ids', []),
        'automatic_resubmission_allowed': False, 'read_only': True})


def _pose(position, quaternion=None, yaw=0.):
    T = np.eye(4); T[:3, 3] = np.asarray(position, float)
    if quaternion is None:
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    else:
        q = np.asarray(quaternion, float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
            raise ValueError('finite_nonzero_wxyz_quaternion_required')
        w, x, y, z = q / np.linalg.norm(q)
        T[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    if not np.isfinite(T).all(): raise ValueError('finite_pose_required')
    return T


def _native_vertices(row):
    values = []
    shapes = row.get('shapes', [])
    if len(shapes) > 128: raise ValueError('native_shape_count_limit_exceeded')
    vertex_count = 0
    for shape in shapes:
        if 'vertices' not in shape: return None
        v = np.asarray(shape['vertices'], float)
        vertex_count += len(v)
        if v.ndim != 2 or v.shape[1] != 3 or vertex_count > 250000:
            raise ValueError('native_vertices_shape_or_count_limit')
        T = np.asarray(shape.get('pose', np.eye(4)), float)
        scale = np.asarray(shape.get('scale', [1, 1, 1]), float)
        if T.shape != (4, 4) or scale.shape != (3,) or not np.isfinite(v).all() or not np.isfinite(T).all() or not np.isfinite(scale).all():
            raise ValueError('finite_native_geometry_required')
        values.append((v * scale) @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(values) if values else None


def scene_measure(source, *, frame=None, object_ids=None, max_objects=64):
    """Native collision extents when recorded; declared bounds otherwise.

    Execution defaults to its final measured frame. A standalone scene file is
    only a declared initial scene. Qualified support disks are not full interiors.
    """
    if type(max_objects) is not int or not 1 <= max_objects <= MAX_OBJECTS:
        raise ValueError('bounded_max_objects_required')
    p = _path(source); e = p if p.is_dir() else None
    if e:
        bundle, scene, _ = _base(e); arrays, measurements = _measurements(e, bundle)
    else:
        scene = _read(p); bundle = {}; arrays = {}; measurements = None
    all_items = list(scene.get('objects', [])) + list(scene.get('supports', []))
    if len(all_items) > MAX_OBJECTS: raise ValueError('scene_object_count_limit_exceeded')
    if object_ids is not None:
        if not isinstance(object_ids, (list, tuple)) or len(object_ids) > max_objects:
            raise ValueError('bounded_object_id_list_required')
        if set(object_ids) - {x['id'] for x in all_items}: raise ValueError('unknown_scene_object_id')
        items = [x for x in all_items if x['id'] in object_ids]
    else:
        items = all_items[:max_objects]
    native = _read(e / 'native_geometry.json', True) if e else None
    containers = _read(e / 'native_containers.json', True) if e else None
    mass = _read(e / 'runtime_mass_properties.json', True) if e else None
    native_by = {x['id']: x for x in (native or {}).get('objects', [])}
    native_by.update({x['id']: x for x in (containers or {}).get('containers', [])})
    masses = {x['object']: x for x in (mass or {}).get('rows', [])}
    object_order = bundle.get('object_ids', [x['id'] for x in scene.get('objects', [])])
    count = len(arrays.get('object_states', [])); selected_frame = None
    if count:
        selected_frame = count - 1 if frame is None or frame == -1 else frame
        if type(selected_frame) is not int or not 0 <= selected_frame < count: raise ValueError('frame_out_of_range')
    elif frame not in (None, 0):
        raise ValueError('no_measured_frames_for_requested_frame')
    rows = []
    for item in items:
        identifier = item['id']; ref = item.get('asset_descriptor') or item.get('container_descriptor')
        descriptor_path = _bound(ref) if ref else None
        d = _read(descriptor_path) if descriptor_path else {}
        T = _pose(item.get('position', [0, 0, 0]), yaw=item.get('yaw', 0))
        pose_kind = 'declared_scene_position_and_yaw'
        nr = native_by.get(identifier, {})
        if selected_frame is not None and identifier in object_order:
            state = arrays['object_states'][selected_frame, object_order.index(identifier)]
            T = _pose(state[:3], state[3:7]); pose_kind = 'measured_native_actor_pose'
        elif nr.get('world_pose') is not None:
            T = np.asarray(nr['world_pose'], float); pose_kind = 'saved_native_static_pose'
            if T.shape != (4, 4) or not np.isfinite(T).all(): raise ValueError('finite_native_pose_required')
        vertices = _native_vertices(nr)
        if vertices is not None:
            local_bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
            world = vertices @ T[:3, :3].T + T[:3, 3]
            geometry_kind = 'saved_native_collision_vertices'
        else:
            bounds = d.get('collision_bounds_m', d.get('bounds_m'))
            if bounds is None:
                dims = d.get('dimensions_m')
                if dims is None and 'half_size' in item: dims = np.asarray(item['half_size']) * 2
                bounds = [-np.asarray(dims)/2, np.asarray(dims)/2] if dims is not None else None
            local_bounds = np.asarray(bounds, float) if bounds is not None else None
            if local_bounds is not None:
                if local_bounds.shape != (2, 3) or not np.isfinite(local_bounds).all() or np.any(local_bounds[1] < local_bounds[0]):
                    raise ValueError('descriptor_bounds_shape_or_values')
                corners = np.asarray(list(itertools.product(*zip(local_bounds[0], local_bounds[1]))))
                world = corners @ T[:3, :3].T + T[:3, 3]
            else: world = None
            geometry_kind = 'descriptor_or_declared_box_approximation' if world is not None else 'unknown'
        cavity = d.get('cavity'); support = {'status': 'unknown', 'reason': 'no_qualified_support_region'}
        if cavity and all(k in cavity for k in ('center_xy_m', 'floor_z_m', 'flat_floor_radius_m')):
            center = np.r_[cavity['center_xy_m'], cavity['floor_z_m']]
            support = {'status': 'declared_sampled_geometry', 'kind': 'qualified_floor_disk',
                'local_center_m': center.tolist(), 'world_center_m': (T[:3, :3] @ center + T[:3, 3]).tolist(),
                'normal_world': T[:3, 2].tolist(), 'radius_m': cavity['flat_floor_radius_m'],
                'rim_z_local_m': cavity.get('rim_z_m', 'unknown'), 'depth_m': cavity.get('depth_m', 'unknown'),
                'qualification': cavity.get('qualification', 'unknown'), 'full_inner_cavity_bounds': 'unknown',
                'outer_bounds_are_not_inner_space': True, 'evidence': _evidence(descriptor_path)}
        native_mass = masses.get(identifier, {}).get('native_readback', {})
        rows.append({'id': identifier, 'source_id': d.get('source_id', 'unknown'), 'name': d.get('name', item.get('semantic_description', 'unknown')),
            'partition': d.get('partition', scene.get('partition', 'unknown')),
            'descriptor': _evidence(descriptor_path) if descriptor_path else None,
            'descriptor_dimensions_m': d.get('dimensions_m', 'unknown'),
            'actor_local_collision_dimensions_m': (local_bounds[1]-local_bounds[0]).tolist() if local_bounds is not None else 'unknown',
            'world_bounds_m': [world.min(axis=0).tolist(), world.max(axis=0).tolist()] if world is not None else 'unknown',
            'geometry_basis': geometry_kind, 'geometry_is_visual_or_real_world_truth': False,
            'geometry_evidence': (_evidence(e / ('native_containers.json' if identifier in {x['id'] for x in (containers or {}).get('containers', [])} else 'native_geometry.json'))
                                  if vertices is not None else (_evidence(descriptor_path) if descriptor_path else None)),
            'pose_basis': pose_kind, 'world_transform': T.tolist(), 'support_region': support,
            'mass': {'native_kg': native_mass.get('mass_kg', nr.get('mass', 'unknown')),
                     'descriptor_kg': d.get('mass_kg', item.get('mass_kg', 'unknown')),
                     'source': d.get('mass_provenance', 'unknown'),
                     'native_evidence': (_evidence(e / 'runtime_mass_properties.json') if native_mass else
                                         _evidence(e / 'native_geometry.json') if nr.get('mass') is not None else None),
                     'actual_real_object_mass': 'unknown'},
            'size_evidence': d.get('size_provenance', d.get('metric_size', 'unknown'))})
    return _clean({'schema': VERSION, 'source': str(p), 'scene_id': scene.get('id'),
        'frame': selected_frame, 'measurements': measurements, 'objects': rows,
        'total_scene_objects': len(all_items), 'truncated': len(rows) < len(all_items) and object_ids is None,
        'table': {'top_z_m': scene.get('table_top_z', 'unknown'), 'usable_extent': 'unknown'},
        'read_only': True, 'new_physics': 0, 'GPU_calls': 0, 'API_calls': 0})


def _operation(execution, task, identifier):
    if identifier is not None and (not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier)):
        raise ValueError('plain_operation_id_required')
    records = (_read(execution / 'executed_operations.json', True) or {}).get('operations', [])
    if len(records) > MAX_OPERATIONS: raise ValueError('executed_operation_limit_exceeded')
    if identifier is not None and identifier not in {x['id'] for x in task.get('operations', [])}:
        raise ValueError('unknown_operation')
    return next((x for x in records if x.get('operation_id') == identifier), None), records


def run_observe(execution, *, frame=None, operation=None, max_frames=16, independent_audit=None):
    if type(max_frames) is not int or not 1 <= max_frames <= 32:
        raise ValueError('bounded_max_frames_required')
    if frame is not None and operation is not None:
        raise ValueError('choose_frame_or_operation')
    e = _execution(execution); bundle, scene, task = _base(e)
    arrays, ref = _measurements(e, bundle); count = len(arrays.get('object_states', []))
    record, _ = _operation(e, task, operation)
    indices = []
    if count and operation is None:
        index = count - 1 if frame is None or frame == -1 else frame
        if type(index) is not int or not 0 <= index < count: raise ValueError('frame_out_of_range')
        indices = [index]
    elif count and record:
        lo, hi = int(record['first_frame']), int(record['last_frame'])
        if not 0 <= lo <= hi < count: raise ValueError('operation_frame_range_invalid')
        indices = np.unique(np.linspace(lo, hi, min(max_frames, hi-lo+1), dtype=int)).tolist()
    names = bundle.get('object_ids', [x['id'] for x in scene.get('objects', [])])
    rows = []
    for index in indices:
        objects = []
        for n, name in enumerate(names[:MAX_OBJECTS]):
            state = arrays['object_states'][index, n]
            objects.append({'id': name, 'position_m': state[:3], 'quaternion_wxyz': state[3:7],
                'linear_velocity_m_s': state[7:10], 'angular_velocity_rad_s': state[10:13],
                'native_grasp_by_arm': arrays['grasp'][index, :, n] if 'grasp' in arrays else 'unknown',
                'robot_contact_force_by_arm_N': arrays['robot_object_force_N'][index, :, n] if 'robot_object_force_N' in arrays else 'unknown',
                'support_impulse_by_support': arrays['support_impulses'][index, n] if 'support_impulses' in arrays else 'unknown'})
        commands = arrays.get('native_commands')
        rows.append({'frame': index, 'timestamp_seconds': arrays['timestamps_seconds'][index] if 'timestamps_seconds' in arrays else 'unknown',
            'objects': objects, 'robot_joint_positions': arrays['states'][index] if 'states' in arrays else 'unknown',
            'TCP_world': arrays['tcp_world'][index] if 'tcp_world' in arrays else 'unknown',
            'incoming_native_action': commands[index-1] if commands is not None and 0 < index <= len(commands) else None,
            'action_maps_state_interval': [index-1, index] if index else None})
    return _clean({'schema': VERSION, 'execution': str(e), 'measurements': ref, 'frame_count': count,
        'operation': operation, 'execution_record': record, 'frame_samples': rows,
        'sampling': 'uniform bounded retained-state samples; not all-frame acceptance',
        'support_channel_order': bundle.get('support_ids', []), 'status': run_status(e, independent_audit=independent_audit),
        'reason': None if rows else 'requested_operation_not_executed_or_measurements_missing', 'read_only': True})


@lru_cache(maxsize=2)
def _dense_tcp_cached(path, size, mtime):
    """Only a bounded retained planning trace, never forward dynamics or FK."""
    if size > MAX_NPZ_BYTES: raise ValueError('dense_archive_byte_limit_exceeded')
    with ZipFile(path) as archive:
        info = archive.infolist()
        if len(info) > 8 or sum(x.file_size for x in info) > MAX_NPZ_BYTES:
            raise ValueError('dense_uncompressed_byte_limit_exceeded')
    with np.load(path, allow_pickle=False) as archive:
        poses = archive['tcp']
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not 1 <= len(poses) <= 250000 or not np.isfinite(poses).all():
        raise ValueError('finite_bounded_dense_TCP_transforms_required')
    poses.flags.writeable = False
    return poses


def _corridor_finding(execution, operation, selected, decision_path, decision, bundle):
    """Explain a rejected predicted payload corridor without granting acceptance."""
    corridor = decision.get('payload_corridor')
    if decision.get('accepted') is not False or not isinstance(corridor, dict) or corridor.get('passed') is not False:
        return None
    folder = decision_path.parent
    request_path, dense_path = folder / 'request.json', folder / 'dense.npz'
    request = _read(request_path, True)
    if request is None or not dense_path.is_file(): return None
    local, goal = np.asarray(request.get('payload_to_tcp'), float), np.asarray(request.get('goal'), float)
    limit = corridor.get('planning_limit_m')
    if local.shape != (4, 4) or goal.shape != (4, 4) or not np.isfinite(local).all() or not np.isfinite(goal).all():
        raise ValueError('finite_payload_request_transforms_required')
    if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise ValueError('recorded_positive_planning_limit_required')
    tcp = _dense_tcp_cached(*_token(dense_path))
    predicted = tcp @ local
    target = (goal @ local)[:3, 3]
    errors = np.linalg.norm(predicted[:, :2, 3] - target[:2], axis=1)
    maximum_index = int(errors.argmax())
    if not np.isclose(errors.max(), corridor.get('maximum_predicted_payload_XY_error_m', np.nan), atol=1e-9, rtol=1e-6):
        raise ValueError('saved_corridor_and_dense_evidence_disagree')
    if errors[0] > limit:
        kind, route = 'payload_corridor_start_outside', 'payload_alignment_before_descent'
    elif errors[-1] > limit:
        kind, route = 'payload_corridor_endpoint_outside', 'payload_goal_alignment'
    elif errors.max() > limit:
        kind, route = 'payload_corridor_interior_excursion', 'payload_descent_path_shape'
    else:
        raise ValueError('rejected_corridor_has_no_recorded_limit_violation')
    evidence = [_evidence(p) for p in (request_path, decision_path, dense_path)]
    protocol_path = execution / 'event_protocol.json'
    protocol = _read(protocol_path, True) or {}
    contract = next((x for x in protocol.get('constraints', []) if x.get('id') == operation + '.release_region'), {})
    params = contract.get('parameters', {})
    observations = {
        'dense_samples': len(tcp), 'maximum_error_dense_index': maximum_index,
        'first_violating_dense_index': int(np.flatnonzero(errors > limit)[0]),
        'predicted_start_XY_error_m': float(errors[0]), 'predicted_end_XY_error_m': float(errors[-1]),
        'maximum_predicted_XY_error_m': float(errors.max()), 'planning_limit_m': limit,
        'original_physical_XY_tolerance_m': params.get('xy_tolerance_m', 'unknown'),
        'prediction_target_world_m': target.tolist(),
        'native_path_guard_passed': decision.get('report', {}).get('passed', 'unknown'),
        'native_dense_checks_complete': decision.get('report', {}).get('all_dense_collision_samples_checked', 'unknown'),
        'current_payload_XY_error_to_exact_task_target_m': 'unknown',
        'measured_current_frame': 'unknown', 'measured_native_grasp': 'unknown',
    }
    if protocol_path.is_file(): evidence.append(_evidence(protocol_path))
    # A failed worker can append its final current state at request.frame. Match
    # the saved request qpos explicitly; never silently use the preceding state.
    arrays, measurement_ref = _measurements(execution, bundle)
    frame = request.get('frame'); arm = selected.get('arm'); obj = selected.get('object')
    current = np.asarray(request.get('current', []), float)
    exact_target = params.get('target_position_world')
    if (type(frame) is int and type(arm) is int and obj in bundle.get('object_ids', [])
            and current.shape == (9,) and exact_target is not None and 'states' in arrays):
        candidates = [f for f in (frame, frame-1) if 0 <= f < len(arrays['states'])]
        matched = next((f for f in candidates if np.allclose(arrays['states'][f, 9*arm:9*arm+9], current, rtol=0, atol=1e-7)), None)
        if matched is not None:
            oi = bundle['object_ids'].index(obj)
            measured = arrays['object_states'][matched, oi, :3]
            observations.update(measured_current_frame=matched,
                current_payload_XY_error_to_exact_task_target_m=float(np.linalg.norm(measured[:2]-np.asarray(exact_target)[:2])),
                measured_native_grasp=bool(arrays['grasp'][matched, arm, oi]) if 'grasp' in arrays else 'unknown',
                current_state_binding='saved request qpos equals retained state within 1e-7 rad; no physical threshold changed')
            if measurement_ref: evidence.append(measurement_ref)
    executed = 'unknown'
    path_ref = bundle.get('planned_paths')
    if path_ref:
        path_log = _bound(path_ref); paths = _read(path_log).get('paths', [])
        if len(paths) > MAX_OPERATIONS * 16: raise ValueError('recorded_path_count_limit_exceeded')
        stage = operation + ':' + request.get('stage', '')
        if not any(x.get('stage') == stage for x in paths) and request.get('preview') is False:
            executed = 'not_executed'
        evidence.append(_evidence(path_log))
    trigger = folder / 'trigger.json'
    if trigger.is_file(): evidence.append(_evidence(trigger))
    return {'type': kind, 'operation': operation, 'object': obj, 'stage': request.get('stage'),
        'route': route, 'observations': observations, 'descent_execution': executed,
        'evidence': evidence, 'physical_acceptance_inferred': False,
        'scope': 'plan-only rigid measured-attachment prediction; planning margin is distinct from the original physical event tolerance; no descent success claimed'}


def run_diagnose(execution, *, operation=None, independent_audit=None, max_findings=16):
    """Route observed failures; hypotheses are not causal or acceptance proofs."""
    if type(max_findings) is not int or not 1 <= max_findings <= 32:
        raise ValueError('bounded_max_findings_required')
    requested_operation = operation
    e = _execution(execution); bundle, scene, task = _base(e)
    status = run_status(e, independent_audit=independent_audit)
    record, records = _operation(e, task, operation)
    failure = status['execution_failure'] or (_read(e / 'planning_failure.json', True) or {})
    message = failure.get('message', '')
    ops = {x['id']: x for x in task.get('operations', [])}
    failed_operation = next((key for key in ops if re.search(r'(^|:)' + re.escape(key) + r'(:|$)', message)), None)
    if failed_operation is None:
        failed_operation = next((x['operation_id'] for x in reversed(records) if x.get('completed') is False), None)
    if operation is None:
        operation = failed_operation
    selected = ops.get(operation, {}); findings = []; arrays = {}; array_ref = None
    if operation:
        completed = {x['operation_id'] for x in records if x.get('completed') is True and x['operation_id'] != operation}
        placed = {ops[x].get('object') for x in completed if x in ops}
        directory = e / (operation + '_motion')
        files = list(itertools.islice(directory.glob('*/decision.json'), MAX_DECISIONS+1))
        if len(files) > MAX_DECISIONS: raise ValueError('decision_query_limit_exceeded')
        # Rejected alternatives of a completed operation are not execution failures.
        for path in sorted(files) if operation == failed_operation else []:
            value = _read(path); report = value.get('report', {})
            if value.get('accepted') is not False: continue
            corridor = _corridor_finding(e, operation, selected, path, value, bundle)
            if corridor is not None: findings.append(corridor)
            for collision in report.get('collision_failures', [])[:8]:
                pairs = collision.get('pairs', [])[:16]
                for pair in {tuple(sorted((x.get('link1', ''), x.get('link2', '')))) for x in pairs}:
                    hand = next((x for x in pair if 'hand' in x or 'finger' in x), None)
                    blocker = next((x for x in pair if x != hand), None)
                    if hand is None: kind, route = 'planning_collision', 'motion_or_layout'
                    elif blocker in placed: kind, route = 'hand_vs_placed_object', 'placement_slot_or_order'
                    elif blocker in {x['id'] for x in scene.get('supports', [])}: kind, route = 'hand_vs_container', 'placement_hand_clearance'
                    else: kind, route = 'hand_vs_obstacle', 'motion_or_layout'
                    findings.append({'type': kind, 'operation': operation, 'object': selected.get('object'),
                        'stage': path.parent.name, 'collision_pair': list(pair), 'dense_index': collision.get('dense_index'),
                        'route': route, 'evidence': [_evidence(path)],
                        'scope': 'guard rejected this planned path before execution; not a measured runtime collision'})
        if operation == failed_operation and any(x in message for x in ('attachment_unstable', 'two_finger_grasp', 'persistent_native_grasp')):
            arrays, array_ref = _measurements(e, bundle)
            _, recs = _operation(e, task, operation); oprec = next((x for x in recs if x['operation_id'] == operation), None)
            close_path = e / 'contact_latched_closure' / (operation + '_close.json')
            close = _read(close_path, True) or {}; steps = close.get('steps', [])
            grasp = {'close_end_native_grasp': steps[-1].get('after', {}).get('native_grasp', 'unknown') if steps else 'unknown',
                     'cache_width_m': close.get('geometric_prediction', {}).get('cache_width_m', 'unknown')}
            if arrays and oprec and 'grasp' in arrays and selected.get('object') in bundle.get('object_ids', []):
                oi = bundle['object_ids'].index(selected['object']); arm = int(selected['arm'])
                lo, hi = oprec['first_frame'], oprec['last_frame']
                signals = arrays['grasp'][lo:hi+1, arm, oi]; transitions = np.flatnonzero(signals[:-1] & ~signals[1:]) + lo + 1
                grasp.update(first_grasp_true_frame=int(lo+np.flatnonzero(signals)[0]) if signals.any() else None,
                    grasp_loss_frames=transitions[:16].tolist(), final_native_grasp=bool(signals[-1]))
                if len(transitions):
                    index = int(transitions[0]); a = arrays['object_states']
                    grasp['object_XY_change_at_first_loss_m'] = float(np.linalg.norm(a[index, oi, :2]-a[index-1, oi, :2]))
            evidence = [array_ref] if array_ref else []
            if close_path.is_file(): evidence.append(_evidence(close_path))
            for suffix in ('lift_settle', 'transfer_settle'):
                p = e / (operation + '_' + suffix + '_stability.json')
                if p.is_file(): evidence.append(_evidence(p))
            findings.append({'type': 'grasp_lost_during_lift' if 'lift_settle' in message else 'grasp_contact_or_attachment_failure',
                'operation': operation, 'object': selected.get('object'), 'route': 'grasp_contact_and_attachment_control',
                'observations': grasp, 'evidence': evidence,
                'root_cause': 'unknown; these measurements do not establish mass, friction or pose as the cause'})
    audit, ap, audit_problem = _audit(e, independent_audit)
    if audit is not None and not audit_problem:
        leaf = ap.parent / audit['case_id'] if 'rows' in _read(ap) else ap.parent
        event_path = leaf / 'new_event_audit.json'
        event = _read(event_path, True)
        if event:
            failures = [c for c in event.get('qualification', {}).get('checks', []) if c.get('passed') is False]
            protocol = _read(e / 'event_protocol.json', True) or {}
            for check in failures[:MAX_OPERATIONS]:
                cid = check.get('id', ''); opid = next((x for x in ops if x + '.' in cid), None)
                if operation is not None and opid is not None and opid != operation: continue
                if cid.endswith('.release_region') and opid in ops:
                    contract = next((c for c in protocol.get('constraints', []) if c.get('id') == opid + '.release_region'), {})
                    params = contract.get('parameters', {}); observation = {'first_failure_frame': check.get('first_failure_frame')}
                    if not arrays: arrays, array_ref = _measurements(e, bundle)
                    target = params.get('target_position_world'); obj = ops[opid]['object']
                    lo, hi = check.get('frame_start'), check.get('frame_end_inclusive')
                    if target is not None and arrays and obj in bundle.get('object_ids', []) and type(lo) is int and type(hi) is int:
                        if not 0 <= lo <= hi < len(arrays['object_states']):
                            raise ValueError('independent_event_frame_range_invalid')
                        states = arrays['object_states'][lo:hi+1, bundle['object_ids'].index(obj)]
                        errors = np.linalg.norm(states[:, :2]-np.asarray(target)[:2], axis=1)
                        observation.update(maximum_XY_error_m=float(errors.max()), maximum_error_frame=int(lo+errors.argmax()),
                            original_XY_tolerance_m=params.get('xy_tolerance_m', 'unknown'),
                            original_z_interval_m=[params.get('minimum_z_world_m'), params.get('maximum_z_world_m')])
                    findings.append({'type': 'release_region_violation', 'operation': opid, 'object': obj,
                        'route': 'payload_pose_during_descent', 'observations': observation,
                        'evidence': [_evidence(event_path), _evidence(e / 'event_protocol.json')] + ([array_ref] if array_ref else []),
                        'scope': 'original independently failed transient condition; final placement may still pass'})
                elif opid is None or operation is None or opid == operation:
                    findings.append({'type': 'independent_condition_failure', 'operation': opid,
                        'route': 'inspect_original_condition', 'condition': check,
                        'evidence': [_evidence(event_path)]})
    if not findings:
        relevant_message = message if operation is None or operation == failed_operation else ''
        findings.append({'type': 'execution_failure_unclassified' if relevant_message else 'no_bound_failure_diagnosis',
            'operation': operation, 'route': 'inspect_evidence', 'message': relevant_message or None,
            'reason': audit_problem if not relevant_message else 'no_matching_specific_evidence',
            'evidence': [_evidence(e / 'planning_failure.json')] if relevant_message and (e / 'planning_failure.json').is_file() else []})
    return _clean({'schema': VERSION, 'execution': str(e), 'status': status,
        'findings': findings[:max_findings], 'findings_truncated': len(findings) > max_findings,
        'failed_operation': failed_operation or (findings[0].get('operation') if len(findings) == 1 and findings[0]['type'] != 'no_bound_failure_diagnosis' else None),
        'requested_operation': requested_operation, 'causal_proof_claimed': False,
        'read_only': True, 'new_physics': 0, 'GPU_calls': 0, 'API_calls': 0})
