"""Independent read-only audit of arbitrary native batch cases.

The worker's success flag is not an acceptance input. Event/kinematic tolerances
are inherited from the frozen evaluator. New checks bind media, dynamics, task
separation and dataset grouping; this tool never changes native measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

PACKAGE = ('autonomous_pipeline_v4' if importlib.util.find_spec('autonomous_pipeline_v4')
           else 'corl2026_workshop.experiments.autonomous_pipeline_v4')
core = importlib.import_module(PACKAGE + '.core')
event_module = importlib.import_module(PACKAGE + '.event_protocol')
interface_module = importlib.import_module(PACKAGE + '.interface')
compiler_module = importlib.import_module(PACKAGE + '.compiler')
bind, checked, read, require, write, digest = (getattr(core, x) for x in
    ['bind', 'checked', 'read', 'require', 'write', 'digest'])
trace_path = Path(__file__).with_name('trace_geometry_acceptance_v3.py')
spec = importlib.util.spec_from_file_location('batch_frozen_trace_v3', trace_path)
trace_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace_module)
VERSION = 'batch_native_independent_acceptance_20261025_v1'


def evaluator_files():
    return {Path(p).name: bind(Path(p)) for p in [__file__, trace_path,
        event_module.__file__, interface_module.__file__, compiler_module.__file__]}


def verify_evaluators(manifest):
    declared = read(manifest)
    declared = declared.get('files', declared.get('evaluator_bindings', declared))
    current = evaluator_files()
    for name, actual in current.items():
        require(name in declared, 'missing_frozen_evaluator:' + name)
        checked(declared[name])
        require(actual['sha256'] == declared[name]['sha256'], 'evaluator_changed:' + name)
    return current


def arrays_fingerprint(arrays, *, rounded=False):
    """Exclude render/view/static appearance, count exact and 1e-4 robot motion."""
    h = hashlib.sha256()
    fields = ['timestamps_seconds', 'states', 'native_commands']
    for name in fields:
        value = np.asarray(arrays[name])
        if rounded:
            value = np.rint(value.astype(float) * 1e4).astype('<i8')
        else:
            value = np.ascontiguousarray(value)
        h.update(name.encode()); h.update(str(value.shape).encode()); h.update(value.dtype.str.encode())
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def physical_scene_fingerprint(scene):
    keys = {'id', 'kind', 'shape', 'half_size', 'radius', 'parts', 'center', 'position', 'yaw',
            'mass_kg', 'pivot_world', 'axis_world', 'joint_type', 'limits', 'spring_N_or_Nm',
            'friction', 'damping', 'initial_q', 'rest_q', 'top_z'}
    def physical(value):
        if isinstance(value, dict):
            return {k: physical(v) for k, v in value.items() if k in keys}
        if isinstance(value, list):
            return [physical(v) for v in value]
        return value
    return digest({'robot_roots': scene['robot_roots'], 'table_top_z': scene['table_top_z'],
                   'objects': physical(scene['objects']), 'supports': physical(scene['supports']),
                   'joints': physical(scene.get('joints', []))})


def analytic_dynamics(obj):
    mass = obj['mass_kg']
    if obj['shape'] == 'sphere':
        return np.zeros(3), np.full(3, .4 * mass * obj['radius']**2)
    require(obj['shape'] in {'box', 'compound_boxes'}, 'unsupported_analytic_shape:' + obj['shape'])
    parts = obj.get('parts', [{'center': [0., 0., 0.], 'half_size': obj['half_size']}])
    centers = np.asarray([p['center'] for p in parts], dtype=float)
    half = np.asarray([p['half_size'] for p in parts], dtype=float)
    volumes = np.prod(2. * half, axis=1)
    weights = mass * volumes / volumes.sum()
    center = (weights[:, None] * centers).sum(0) / mass
    inertia = np.zeros((3, 3))
    for c, h, m in zip(centers, half, weights):
        d = c - center
        inertia += np.diag(m / 3. * np.array([h[1]**2+h[2]**2,
            h[0]**2+h[2]**2, h[0]**2+h[1]**2])) + m * ((d@d)*np.eye(3)-np.outer(d, d))
    return center, np.linalg.eigvalsh(inertia)


def dynamics_check(scene, directory):
    path = directory / 'asset_runtime_evidence.json'
    required = [o for o in scene['objects'] if o['kind'] == 'movable']
    if not path.exists():
        return {'passed': not required, 'evidence': None, 'missing_dynamic_asset_count': len(required), 'assets': []}
    native = {x['id']: x for x in read(path)['assets']}
    rows = []
    for obj in required:
        measured = native.get(obj['id'])
        if measured is None:
            rows.append({'id': obj['id'], 'passed': False, 'reason': 'missing_native_dynamics'}); continue
        com, inertia = analytic_dynamics(obj)
        actual = np.asarray(measured['native_inertia_kg_m2'])
        checks = {'mass': bool(np.isclose(measured['native_mass_kg'], obj['mass_kg'], rtol=1e-4)),
                  'positive_inertia': bool(np.all(actual > 0.)),
                  'center_of_mass': bool(np.allclose(measured['native_com_local'], com, rtol=1e-4, atol=1e-7)),
                  'principal_inertia': bool(np.allclose(np.sort(actual), inertia, rtol=1e-4, atol=1e-9)),
                  'shape_materials_present': bool(measured.get('native_shape_materials'))}
        materials = measured.get('native_shape_materials', [])
        expected_count = len(obj.get('parts', [None]))
        checks['native_shape_count'] = len(materials) == expected_count
        checks['declared_density'] = all(np.isclose(x['density_kg_m3'], obj['geometry']['density_kg_m3'], rtol=1e-4)
                                          for x in materials)
        checks['finite_material_parameters'] = all(all(np.isfinite(x[k]) and x[k] >= 0.
            for k in ['density_kg_m3', 'static_friction', 'dynamic_friction', 'restitution']) and x['restitution'] <= 1.
            for x in materials)
        rows.append({'id': obj['id'], 'passed': all(checks.values()), 'checks': checks,
                     'expected_principal_inertia': inertia.tolist(), 'actual_principal_inertia': actual.tolist()})
    return {'passed': all(x['passed'] for x in rows), 'evidence': bind(path), 'assets': rows,
            'scope': 'Declared uniform primitive masses, additive overlap volumes, COM and principal inertias; not measured real-object physics.'}


def event_acceptance(plan, arrays):
    try:
        measured = event_module.compile_events(plan, arrays)
        path = trace_module.trace_geometry(plan, arrays, measured)
        return {'passed': bool(measured['qualification']['passed'] and path['passed']),
                'measured': measured, 'trace_geometry': path}
    except (ValueError, KeyError, IndexError) as exc:
        return {'passed': False, 'error': type(exc).__name__ + ':' + str(exc)}


def plan_semantics(plan):
    return digest({k: plan[k] for k in ['family', 'entities', 'events', 'goals', 'constraints',
                                      'precedence', 'initial_conditions', 'initial_hands', 'terminal_hands']})


def _signals(expr):
    if expr.get('op') == 'signal':
        return {expr['name']}
    return set().union(*[_signals(a) for a in expr.get('args', [])])


def negative_controls(plan, arrays):
    controls = []
    def add(name, mutated):
        result = event_acceptance(plan, mutated)
        controls.append({'mutation': name, 'rejected': not result['passed'],
                         'failure_reason': result.get('error'),
                         'false_positive_checks': result.get('measured', {}).get('qualification') if result['passed'] else None})
    for channel in ['robot_contact_impulses', 'object_contact_impulses', 'grasp']:
        used = any(c['array'] == channel for c in plan['channels'].values())
        positive_needed = channel != 'grasp' or any(e['action'] == 'grasp' for e in plan['events'])
        # Object-contact arrays may be declared only for unchanged objects; only
        # erase them if a task event actually consumes such evidence.
        if channel in {'robot_contact_impulses', 'object_contact_impulses'}:
            positive_needed = any(plan['channels'][s]['array'] == channel
                                  for e in plan['events'] for s in _signals(e['when']))
        if used and positive_needed and channel in arrays:
            changed = {**arrays, channel: np.zeros_like(arrays[channel])}
            add('remove_' + channel, changed)
    # Erasing the whole array cannot show that every distinct required contact
    # matters. Remove each event's contact channel separately as well.
    for event in plan['events']:
        channels = [plan['channels'][name] for name in _signals(event['when'])
                    if plan['channels'][name]['array'] in {'robot_contact_impulses', 'object_contact_impulses'}]
        if not channels:
            continue
        changed = dict(arrays)
        for channel in channels:
            key = channel['array']
            if changed[key] is arrays[key]: changed[key] = arrays[key].copy()
            indices = tuple(slice(*x['slice']) if isinstance(x, dict) else x for x in channel['indices'])
            changed[key][(slice(None), *indices)] = 0.
        add('omit_required_contact_event:' + event['id'], changed)
    changed = dict(arrays)
    for name in ['object_states', 'cabinet_qpos', 'cabinet_qvel', 'tool_tips_world', 'door_world']:
        if name in arrays:
            changed[name] = np.repeat(arrays[name][:1], len(arrays[name]), axis=0)
    add('freeze_all_scene_motion_at_initial', changed)
    if 'cabinet_qpos' in arrays and arrays['cabinet_qpos'].size:
        changed = {**arrays, 'cabinet_qpos': -arrays['cabinet_qpos'], 'cabinet_qvel': -arrays['cabinet_qvel']}
        add('reverse_measured_joint_direction', changed)
    # Explicit forbidden contact must remain rejected even if the task goal is reached.
    for condition in plan['constraints']:
        if condition['relation'] != 'no_contact' or not condition.get('effector'):
            continue
        for name in sorted(_signals(condition['predicate'])):
            channel = plan['channels'][name]
            if channel['kind'] not in {'contact_impulse', 'contact_force'}:
                continue
            key = channel['array']; value = arrays[key].copy()
            indices = tuple(slice(*x['slice']) if isinstance(x, dict) else x for x in channel['indices'])
            value[(slice(None), *indices)] = 100.
            add('inject_forbidden_contact:' + condition['id'], {**arrays, key: value})
    if plan['family'] in {'lever_transmitted_lift', 'support_removal_gravity_release'}:
        # Ordered event search alone can mistake an outcome already present
        # before actuation for an outcome caused by actuation.
        for entity in plan['entities']:
            pose = plan['channels'][entity['pose_channel']]
            if entity['role'] != 'target' or pose['array'] != 'object_states': continue
            value = arrays['object_states'].copy()
            index = pose['indices'][0]
            value[1:, index, :3] = value[-1, index, :3]
            add('payload_outcome_before_trigger:' + entity['id'], {**arrays, 'object_states': value})
    return controls


def video_check(camera, times):
    video = checked(camera['video'])
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time,width,height', '-of', 'json', str(video)]
    output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    require(not output.stderr.strip(), 'video_probe_decode_error')
    info = json.loads(output.stdout); frames = info['frames']; stream = info['streams'][0]
    pts = np.array([float(x['best_effort_timestamp_time']) for x in frames])
    require(len(pts) == len(times) == camera['frame_count'], 'camera_frame_count_mismatch')
    require(np.allclose(pts, times, atol=1e-5, rtol=0), 'camera_pts_mismatch')
    require(camera['synchronous_with_measurements'] is True, 'camera_synchrony_not_declared')
    from fractions import Fraction
    require(abs(float(Fraction(stream['avg_frame_rate'])) - camera['fps']) < 1e-5, 'camera_fps_mismatch')
    for key in ['width', 'height']:
        require(stream[key] == camera['config'][key] and all(f[key] == stream[key] for f in frames), 'camera_shape_mismatch')
    decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2',
        '-i', str(video), '-map', '0:v:0', '-f', 'null', '-'], check=True, capture_output=True, text=True, timeout=180)
    require(not decoded.stderr.strip(), 'video_full_decode_error')
    for field in ['initial_image', 'final_image']:
        checked(camera[field])
    return {'passed': True, 'video': camera['video'], 'full_decoded_frames': len(pts),
            'max_pts_error_seconds': float(np.abs(pts-times).max())}


def audit_one(case, directory):
    row = {'case': case['id'], 'attempt': directory.name, 'case_record': case,
           'status': 'not_terminal', 'independent_qualified': False, 'errors': [], 'negative_controls': []}
    terminal = directory / 'terminal.json'
    if not terminal.exists():
        return row, None, None
    row['terminal'] = bind(terminal); row['worker_terminal'] = read(terminal)
    bundle_path = directory / 'execution_bundle.json'
    if not bundle_path.exists():
        row.update(status='no_native_bundle'); row['errors'].append('missing_execution_bundle'); return row, None, None
    arrays = plan = None
    try:
        bundle = read(bundle_path); scene = read(checked(case['scene'])); task = read(checked(case['task']))
        row['execution'] = bind(bundle_path)
        require(bundle['case'] == case and bundle['scene'] == scene and bundle['task_intent'] == task, 'frozen_input_mismatch')
        for value in bundle['code'].values(): checked(value)
        arrays = dict(np.load(checked(bundle['measurements']), allow_pickle=False))
        interface_module.validate_bundle(bundle, arrays)
        row['frames'] = len(arrays['timestamps_seconds']); row['camera_streams'] = len(bundle['cameras'])
        row['raw_robot_dynamic_sha256'] = arrays_fingerprint(arrays)
        row['quantized_robot_dynamic_sha256'] = arrays_fingerprint(arrays, rounded=True)
        row['measurements'] = bundle['measurements']
        row['physical_scene_sha256'] = physical_scene_fingerprint(scene)
        initial = hashlib.sha256(np.ascontiguousarray(arrays['initial_physics_state']).tobytes()).hexdigest()
        require(initial == bundle['initial_physics_sha256'], 'initial_physics_hash_mismatch')
        require(digest(arrays['object_states'][0].tolist()) == bundle['initial_object_state_sha256'], 'initial_objects_hash_mismatch')
        require(digest(arrays['cabinet_qpos'][0].tolist()) == bundle['initial_articulation_state_sha256'], 'initial_joints_hash_mismatch')
        row['initial'] = {'physics': initial, 'objects': bundle['initial_object_state_sha256'],
                          'joints': bundle['initial_articulation_state_sha256'],
                          'cameras': {k: v['initial_image']['sha256'] for k, v in bundle['cameras'].items()}}
        row['native_kinematics'] = compiler_module.verify_native_kinematics(arrays)
        plan = read(checked(bundle['event_protocol']))
        require(plan['task_id'] == task['id'] and plan['family'] == task['family'], 'task_protocol_identity_mismatch')
        require(set(task['object_ids']) == {x['id'] for x in plan['entities'] if x['role'] == 'target'}, 'task_protocol_target_mismatch')
        row['event_protocol'] = bundle['event_protocol']; row['declared_program_sha256'] = plan_semantics(plan)
        row['events'] = event_acceptance(plan, arrays)
        row['dynamics'] = dynamics_check(scene, directory)
        row['media'] = {name: video_check(camera, arrays['timestamps_seconds']) for name, camera in bundle['cameras'].items()}
        tip_errors = []
        for i, obj in enumerate(scene['objects']):
            if 'tip_local' in obj:
                states = arrays['object_states'][:, i]
                expected = states[:, :3]+Rotation.from_quat(states[:, 3:7], scalar_first=True).apply(obj['tip_local'])
                tip_errors.append(float(np.linalg.norm(expected-arrays['tool_tips_world'][:, i], axis=1).max()))
        row['tip_recompute_max_error_m'] = max(tip_errors, default=0.)
        require(row['tip_recompute_max_error_m'] < 1e-5, 'tool_tip_export_inconsistent')
        if bundle['planning_failure'] is not None: row['errors'].append('native_planning_or_execution_failure')
        if not row['events']['passed']: row['errors'].append('native_event_or_trace_predicates_failed')
        if not row['dynamics']['passed']: row['errors'].append('native_dynamics_mismatch_or_missing')
        row['native_measurements_qualified'] = not row['errors']
        if row['native_measurements_qualified']:
            row['negative_controls'] = negative_controls(plan, arrays)
            if not all(x['rejected'] for x in row['negative_controls']): row['errors'].append('negative_control_false_positive')
        row['independent_qualified'] = not row['errors']; row['status'] = 'qualified' if not row['errors'] else 'rejected'
    except (ValueError, KeyError, IndexError, FileNotFoundError, subprocess.SubprocessError) as exc:
        row['errors'].append(type(exc).__name__+':'+str(exc)); row['status'] = 'rejected'
    return row, plan, arrays


def audit(root, output, evaluator_manifest, case_ids=()):
    evaluators = verify_evaluators(evaluator_manifest)
    cases = [read(p) for p in sorted((root/'registered_cases').glob('*.json'))]
    if case_ids:
        require(set(case_ids) <= {c['id'] for c in cases}, 'requested_case_not_registered')
        cases = [c for c in cases if c['id'] in set(case_ids)]
    require(cases, 'no_registered_cases_selected')
    rows = []; loaded = {}; programs = defaultdict(dict); grouping = defaultdict(list)
    for case in cases:
        attempts = sorted((root/'executions'/case['id']).glob('attempt_*'))
        if not attempts:
            rows.append({'case': case['id'], 'case_record': case, 'status': 'not_started', 'independent_qualified': False,
                         'errors': [], 'negative_controls': []})
        for directory in attempts:
            row, plan, arrays = audit_one(case, directory); rows.append(row)
            if plan is not None and arrays is not None:
                key = (case['id'], directory.name); loaded[key] = (plan, arrays)
                programs[case['scene']['sha256']][plan_semantics(plan)] = (case['id'], plan)
            print(json.dumps({'case': case['id'], 'attempt': directory.name, 'status': row['status'], 'errors': row['errors']}, ensure_ascii=False), flush=True)
    split_keys = defaultdict(set)
    for row in rows:
        case = row['case_record']; partition = case.get('data_split', case.get('partition'))
        require(partition is not None, 'case_partition_missing:' + case['id'])
        group = case.get('split_group', case['scene']['sha256']); grouping[group].append(row)
        for key in [('scene_binding', case['scene']['sha256']), ('split_group', group)]: split_keys[key].add(partition)
        if row.get('physical_scene_sha256'): split_keys['physical_scene', row['physical_scene_sha256']].add(partition)
        if row.get('execution'): split_keys['execution', row['execution']['sha256']].add(partition)
        for session in case.get('source_session_ids', []): split_keys['source_session', session].add(partition)
        if not row.get('native_measurements_qualified'): continue
        plan, arrays = loaded[row['case'], row['attempt']]
        for signature, (other_id, other) in programs[case['scene']['sha256']].items():
            if signature == row['declared_program_sha256']: continue
            rejected = not event_acceptance(other, arrays)['passed']
            row['negative_controls'].append({'mutation': 'wrong_registered_task', 'other_task': other_id, 'rejected': rejected})
        if not all(n['rejected'] for n in row['negative_controls']):
            if 'negative_control_false_positive' not in row['errors']: row['errors'].append('negative_control_false_positive')
            row['independent_qualified'] = False; row['status'] = 'rejected'
    groups = []
    for key, members in grouping.items():
        initial = [r['initial'] for r in members if 'initial' in r]
        same = len({digest(x) for x in initial}) <= 1
        groups.append({'split_group': key, 'executed_members_with_initial_evidence': len(initial),
                       'same_initial_physics_objects_joints_and_views': same,
                       'cases': sorted({r['case'] for r in members})})
        if not same:
            for row in members:
                if row.get('independent_qualified'):
                    row['independent_qualified'] = False; row['status'] = 'rejected'; row['errors'].append('same_scene_initial_state_changed')
    leaks = [{'kind': k[0], 'identity': k[1], 'partitions': sorted(v)} for k, v in split_keys.items() if len(v)>1]
    negatives = [dict(case=r['case'], attempt=r.get('attempt'), **n) for r in rows for n in r['negative_controls']]
    qualified = [r for r in rows if r['independent_qualified']]
    counts = {'registered_cases': len(cases), 'attempt_records': len(rows),
              'physical_reservations': sum('attempt' in r for r in rows),
              'terminal_attempts': sum('terminal' in r for r in rows),
              'native_measurements_qualified': sum(r.get('native_measurements_qualified', False) for r in rows),
              'independent_qualified': len(qualified), 'statuses': dict(Counter(r['status'] for r in rows)),
              'raw_unique_robot_dynamics': len({r['raw_robot_dynamic_sha256'] for r in qualified}),
              'quantized_unique_robot_dynamics': len({r['quantized_robot_dynamic_sha256'] for r in qualified}),
              'qualified_scene_task_programs': len({(r['case_record']['scene']['sha256'], r['declared_program_sha256']) for r in qualified}),
              'physical_scene_layouts': len({r['physical_scene_sha256'] for r in qualified}),
              'frames': sum(r.get('frames', 0) for r in rows), 'camera_streams': sum(r.get('camera_streams', 0) for r in rows),
              'negative_controls': len(negatives), 'negative_controls_rejected': sum(n['rejected'] for n in negatives),
              'split_leakages': len(leaks), 'formal_pairs': 0}
    report = {'schema': VERSION, 'root': str(root), 'auditor': bind(Path(__file__)),
              'evaluator_manifest': bind(evaluator_manifest), 'evaluators': evaluators, 'counts': counts,
              'cases': rows, 'groups': groups, 'negative_controls': negatives, 'split_leakages': leaks,
              'qualified_attempts': [{'case': r['case'], 'attempt': r['attempt']} for r in qualified],
              'batch_ready_for_sampling_within_verified_scope': bool(qualified) and not leaks
                  and all(n['rejected'] for n in negatives)
                  and all(g['same_initial_physics_objects_joints_and_views'] for g in groups),
              'limitations': ['No human-video quality or real-source equivalence claim.',
                  'Sampled native measurements do not prove collision freedom at all continuous times.',
                  'Uniform primitive inertia audit assumes additive volumes, including overlaps.',
                  'Current asset runtime evidence covers movable rigid bodies; passive articulation link inertias are not independently bound to declared geometry.',
                  'Exact and 1e-4 quantized robot-motion fingerprints are reported separately; camera counts are not trajectory counts.',
                  'Cross-task negative controls compare registered semantic programs in the same scene; no inferred task equivalence.']}
    write(output, report)
    return {'counts': counts, 'audit': bind(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluator-manifest', type=Path, required=True)
    parser.add_argument('--case-id', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output, args.evaluator_manifest, args.case_id), ensure_ascii=False))
