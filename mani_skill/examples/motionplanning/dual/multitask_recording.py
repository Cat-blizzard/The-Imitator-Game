"""Shared physical recording for synchronized and official serial task runners.

The shared recorder retains20Hz control and100Hz observations. A measured
physical support exit is recorded separately from geometric support release.
Ordinary tasks retain200 actions; separate declared capability protocols may
select400. Serial execution may supply one phase per target so
an untouched peer may retain its own initial support during the active arm's
release. Actions are canonical dictionaries of two absolute-radian [7] arm
targets and one scalar gripper each; actor/TCP poses are world XYZ/WXYZ [7].
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Callable

import numpy as np

from .trajectory_execution import as_numpy
from .validate_grasp import json_value, recording_summary, save_report
from .validate_sync import applied_action, continue_success_only
from .validate_path_conflict import interarm_contact_rows
from .validate_held_path_conflict import classify_interarm_contacts, physical_contact_points

PHYSICAL_SUPPORT_CLEARANCE_M = .003
RECORDING_VERSION = 'multitask-recording-v06-verified-destination-contents'


def apply_target_contact_policy(rows: list[dict], phase: str, policy: Any) -> list[dict]:
    """Apply an explicit role policy without changing raw force/gap evidence."""
    if policy is None: return rows
    result = copy.deepcopy(rows)
    for row in result:
        chosen = policy(copy.deepcopy(row), phase)
        if chosen not in ('own_finger_allowed','initial_support','destination_support','destination_contents','forbidden','handoff_finger_allowed'):
            raise ValueError('Unknown declared target contact policy class')
        if chosen == 'destination_contents' and not (row.get('destination_contents_proof', {}).get('allowed') is True
                                                     and row['class'] == 'destination_contents'):
            raise ValueError('A custom policy cannot invent an unverified destination contents allowance')
        row['default_class'], row['class'] = row['class'], chosen
    return result


def physical_support_transition(previous: bool, geometry_released: bool, reference_z: float,
                                measured_z: float, source_contact: bool) -> bool:
    """Close initial physical support once, after a measured 3mm upward exit.

    Geometric collision checking retains its original independent release rule.
    Native contact manifolds describe the preceding narrow-phase state; their
    micro-scale startup contacts remain initial support until this explicit
    physical phase transition. No collision depth or force threshold changes.
    """
    if not np.isfinite([reference_z, measured_z]).all():
        raise ValueError('Finite support reference and measured origin heights required')
    return bool(previous or (geometry_released and not source_contact
                            and measured_z-reference_z >= PHYSICAL_SUPPORT_CLEARANCE_M))


def contact_identities(raw: Any, agents: Any, names: Any, targets: Any) -> tuple[dict, dict]:
    identities = {obj.entity: {'kind':'robot','robot_index':i,'link_name':link.name,'actor':link.name}
        for i,agent in enumerate(agents) for link in agent.robot.get_links() for obj in link._objs}
    for name,actor in raw.scene.actors.items():
        for obj in actor._objs:
            if obj not in identities: identities[obj] = {'kind':'actor','actor':name}
    for i,actor in enumerate(targets):
        for obj in actor._objs: identities[obj] = {'kind':'payload','actor':names[i],'target_index':i}
    robots = {entity:(value['robot_index'],value['link_name']) for entity,value in identities.items() if value['kind']=='robot'}
    return identities, robots


def support_name(actor: Any) -> str:
    return str(actor.name)


def checked_geometry(vertices: Any, pose: Any) -> tuple[np.ndarray, np.ndarray]:
    """Finite actor-local vertices[V,3] and world XYZ/WXYZ[7]."""
    points, transform = np.asarray(vertices, dtype=float), np.asarray(pose, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or transform.shape != (7,)
            or not np.isfinite(points).all() or not np.isfinite(transform).all()
            or not np.isclose(np.linalg.norm(transform[3:]), 1., rtol=0, atol=1e-5)):
        raise ValueError('Invalid native local vertices or world pose')
    return points, transform


def save_task_geometry(output: Path, raw: Any, adapter: Any) -> dict[str, Any]:
    """Export native collision vertices for independent placement recomputation."""
    arrays, manifest = {}, {}
    for role, actors in (('target', adapter.targets), ('destination', adapter.destination_actors),
                         ('initial_support', adapter.initial_supports), ('destination_support', adapter.destination_supports),
                         ('contents', getattr(adapter, 'destination_contents', ()))):
        for i, actor in enumerate(actors):
            key = f'{role}{i}'
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            points, pose = checked_geometry(np.empty((0, 3)) if mesh is None else mesh.vertices,
                                           as_numpy(actor.pose.raw_pose)[0])
            if role in ('target', 'initial_support', 'destination_support', 'contents') and len(points) == 0:
                raise ValueError(f'Missing native physical collision geometry for {role}:{actor.name}')
            arrays[key+'_vertices'] = points
            arrays[key+'_pose'] = pose
            state_names = [name for name, value in raw.scene.actors.items() if value is actor]
            if len(state_names) != 1:
                raise ValueError(f'Ambiguous scene actor identity for {actor.name}: {state_names}')
            manifest[key] = {'actor_name': actor.name, 'state_name': state_names[0],
                'local_vertices_array': key+'_vertices', 'world_pose_array': key+'_pose',
                'vertex_count': len(points), 'shape': list(points.shape),
                'local_vertices_sha256': hashlib.sha256(np.ascontiguousarray(points, dtype=np.float64).tobytes(order='C')).hexdigest()}
            if role == 'contents':
                manifest[key].update(canonical_target_index=i, corresponding_target=adapter.target_names[i],
                                     corresponding_destination_actor=adapter.destination_actors[i].name)
    np.savez(output/'task-geometry.npz', **arrays)
    result = {'actors': manifest, 'units': 'metres; actor-local vertices; world XYZ/WXYZ',
              'recording_version': RECORDING_VERSION,
              'geometry_npz_sha256': hashlib.sha256((output/'task-geometry.npz').read_bytes()).hexdigest()}
    save_report(output/'task-geometry.json', result)
    return result


def contact_class(other: dict, index: int, released: bool, phase: str,
                  initial: tuple, destinations: tuple) -> str:
    if other.get('robot_index') == index and other.get('link_name') in ('panda_leftfinger','panda_rightfinger'):
        return 'own_finger_allowed'
    placement = phase in ('release','retreat','verify_placement')
    if placement and other.get('actor') == destinations[index]: return 'destination_support'
    if not placement and not released and other.get('actor') == initial[index]: return 'initial_support'
    return 'forbidden'


def task_object_contact_rows(contacts: Any, identities: dict, released: tuple,
                             phase: str | tuple[str, str], adapter: Any, *,
                             include_destination_contents_proof: bool = True) -> list[dict]:
    rows = []
    if not isinstance(phase, str) and (len(phase) != 2 or any(not isinstance(p, str) for p in phase)):
        raise ValueError('Two canonical target phases are required')
    initial = tuple(support_name(a) for a in adapter.initial_supports)
    destinations = tuple(support_name(a) for a in adapter.destination_supports)
    contents_names = tuple(getattr(adapter, 'destination_contents_state_names', ()))
    for contact in contacts:
        sides = [dict(identities.get(body.entity, {'kind':'unknown','actor':str(getattr(body.entity,'name','unknown'))})) for body in contact.bodies]
        indices = [s['target_index'] for s in sides if s.get('target_index') is not None]
        if len(sides)!=2 or not indices: continue
        which = next(i for i,s in enumerate(sides) if s.get('target_index') is not None)
        index = sides[which]['target_index'];other = sides[1-which]
        points = [{'position_m':np.asarray(p.position).tolist(),'normal':np.asarray(p.normal).tolist(),
            'impulse_Ns':np.asarray(p.impulse).tolist(),'separation_m':float(p.separation)} for p in contact.points]
        actual = physical_contact_points(points)
        chosen = phase if isinstance(phase, str) else phase[index]
        row = {'bodies':sides,'target_indices':indices,'class':contact_class(other,index,released[index],chosen,initial,destinations),
            'support_released_snapshot':list(released),'initial_support_names':list(initial),'destination_support_names':list(destinations),
            'points':points,'physical_contact':actual,'positive_separation_zero_impulse_proximity':bool(points) and not actual}
        if include_destination_contents_proof and other.get('kind') == 'actor' and other.get('actor') in contents_names:
            proof = adapter.destination_contents_contact_proof(index, other['actor'], chosen, points)
            row['destination_contents_proof'] = proof
            if proof['allowed']:
                row['class'] = 'destination_contents'
        if not isinstance(phase, str): row['target_phases'] = list(phase)
        rows.append(row)
    return rows


def make_physics_capture(env: Any, raw: Any, adapter: Any, args: Any, report: dict[str, Any],
                         frames: list, ticks: list, substeps: list, keys: list, *,
                         checker_provider: Callable[[], Any] = lambda: None,
                         support_release_provider: Callable[[], tuple[bool, bool]] | None = None,
                         contact_policy: Callable[[dict, str], str] | None = None,
                         hard_step_budget: int = 200) -> Any:
    """Create the shared observer; caller installs ``capture.record_substep``.

    Existing lists are retained by identity. ``target_phases=None`` preserves
    synchronous rows and classification. Optional callbacks only read caller
    state. They never modify physical poses, collision permissions or physics.
    """
    agents, targets = (raw.left_agent, raw.right_agent), adapter.targets
    if type(hard_step_budget) is not int or hard_step_budget not in (200, 400):
        raise ValueError('Declared official200 or separate engineering400 budget required')
    report['physical_capture_hard_step_budget'] = hard_step_budget
    report['physical_recording_version'] = RECORDING_VERSION
    report['custom_target_contact_policy'] = contact_policy is not None
    contents_config = getattr(adapter, 'destination_contents_policy_configuration', None)
    report['destination_contents_contact_policy'] = contents_config() if contents_config is not None else {'enabled': False, 'pairs': []}
    identities, robot_identities = contact_identities(raw, agents, adapter.target_names, targets)
    original_hook = raw._after_simulation_step
    physical_released = [False, False]
    reference_z = [float(as_numpy(actor.pose.p)[0, 2]) for actor in targets]
    reference_frozen = [False, False]
    initial_names = tuple(support_name(actor) for actor in adapter.initial_supports)
    report['physical_initial_support_policy'] = {
        'version': 'measured-physical-support-exit-v2',
        'minimum_actual_origin_lift_m': PHYSICAL_SUPPORT_CLEARANCE_M,
        'release_requires': 'geometric support already released AND measured lift>=3mm AND no native actual source contact',
        'reference': 'minimum measured origin before first geometric support release; frozen thereafter',
        'permanent': True, 'geometric_collision_thresholds_changed': False,
        'physical_contact_definition_changed': False}

    def measurement() -> dict[str, Any]:
        objects = []
        for name, agent, actor in zip(adapter.target_names, agents, targets):
            forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, actor))[0]
                      for link in (agent.finger1_link, agent.finger2_link)]
            objects.append({'actor': name, 'pose': as_numpy(actor.pose.raw_pose)[0].tolist(),
                'linear_velocity_m_s': as_numpy(actor.linear_velocity)[0].tolist(),
                'angular_velocity_rad_s': as_numpy(actor.angular_velocity)[0].tolist(),
                'is_grasping': bool(as_numpy(agent.is_grasping(actor))[0]),
                'finger_contact_force_norms_N': [float(np.linalg.norm(f)) for f in forces]})
        return {'left_qpos': as_numpy(agents[0].robot.get_qpos())[0].tolist(),
            'right_qpos': as_numpy(agents[1].robot.get_qpos())[0].tolist(),
            'tcp_poses': [as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents], 'objects': objects}

    class Capture:
        unwrapped = raw
        phase = 'preparation'
        target_phases: tuple[str, str] | None = None
        observe_substeps = True
        pending_control_step = 0

        def __getattr__(self, name: str) -> Any:
            # Needed by the unchanged official solver's self.env.agent access.
            return getattr(raw, name)

        def measurement(self) -> dict[str, Any]:
            return measurement()

        def frame(self) -> None:
            frame = as_numpy(env.render())
            if frame.ndim == 4 and frame.shape[0] == 1: frame = frame[0]
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError('Expected HWC uint8 rendering')
            frames.append(frame.copy())

        def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
            actual, inject = applied_action(action, mode=args.mode, phase=self.phase, keys=keys)
            if int(as_numpy(raw.elapsed_steps)[0]) >= hard_step_budget:
                raise RuntimeError(f'Declared{hard_step_budget} hard budget exhausted')
            first_substep = len(substeps)
            self.pending_control_step = int(as_numpy(raw.elapsed_steps)[0]) + 1
            if inject:
                report['fault_events'].append({'control_step': self.pending_control_step, 'type': 'retain_right_gripper_during_release'})
            obs, reward, terminated, truncated, info = env.step(actual)
            done, continued = continue_success_only(terminated, truncated, info)
            row = {'phase': self.phase, 'environment_step': int(as_numpy(raw.elapsed_steps)[0]),
                'requested_actions': json_value(action), 'actions': json_value(actual), 'fault_injected': inject,
                'native_terminated': bool(as_numpy(terminated).reshape(-1)[0]),
                'native_truncated': bool(as_numpy(truncated).reshape(-1)[0]),
                'success_only_termination_continued': continued, 'official_evaluation': json_value(info), **measurement()}
            if self.target_phases is not None: row['target_phases'] = list(self.target_phases)
            ticks.append(row)
            self.frame()
            if any(any(c['physical_contact'] for c in s['interarm_contacts']) or any(
                    c['class'] == 'forbidden' and c['physical_contact'] for c in s['object_contact_rows'])
                    for s in substeps[first_substep:]):
                raise RuntimeError('Forbidden physical contact at100Hz; stop both arms')
            return obs, reward, done, truncated, info

        def record_substep(self) -> None:
            original_hook()
            if self.observe_substeps:
                contacts = raw.scene.px.get_contacts()
                checker = checker_provider()
                if support_release_provider is not None:
                    geometric_released = tuple(support_release_provider())
                else:
                    geometric_released = tuple(s.support_released for s in checker._supports) if checker is not None and checker.attached else (False, False)
                object_rows = task_object_contact_rows(contacts, identities, tuple(physical_released),
                    self.phase if self.target_phases is None else self.target_phases, adapter,
                    include_destination_contents_proof=False)
                source_contact = [False, False]
                for contact in object_rows:
                    if contact['physical_contact']:
                        for index in contact['target_indices']:
                            source_contact[index] |= any(side.get('kind')=='actor' and side.get('actor')==initial_names[index]
                                                         for side in contact['bodies'])
                heights = [float(as_numpy(actor.pose.p)[0, 2]) for actor in targets]
                for index in range(2):
                    if not reference_frozen[index]:
                        if geometric_released[index]: reference_frozen[index] = True
                        else: reference_z[index] = min(reference_z[index], heights[index])
                    physical_released[index] = physical_support_transition(physical_released[index],
                        geometric_released[index], reference_z[index], heights[index], source_contact[index])
                object_rows = task_object_contact_rows(contacts, identities, tuple(physical_released),
                    self.phase if self.target_phases is None else self.target_phases, adapter)
                object_rows = apply_target_contact_policy(object_rows, self.phase, contact_policy)
                contents_snapshot = getattr(adapter, 'destination_contents_pose_snapshot', None)
                contents_poses = contents_snapshot() if contents_snapshot is not None else {
                    'destination_contents_poses': [], 'destination_basket_poses': []}
                row = {'substep': len(substeps) + 1, 'phase': self.phase,
                    'control_step': self.pending_control_step, 'sim_timestep_s': float(raw.sim_timestep),
                    'object_poses': [as_numpy(a.pose.raw_pose)[0].tolist() for a in targets],
                    'tcp_poses': [as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents],
                    'left_qpos': as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                    'right_qpos': as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                    'left_qvel': as_numpy(agents[0].robot.get_qvel())[0].tolist(),
                    'right_qvel': as_numpy(agents[1].robot.get_qvel())[0].tolist(),
                    'object_linear_velocities': [as_numpy(a.linear_velocity)[0].tolist() for a in targets],
                    'object_angular_velocities': [as_numpy(a.angular_velocity)[0].tolist() for a in targets],
                    'objects': measurement()['objects'],
                    'geometric_support_released_snapshot': list(geometric_released),
                    'physical_support_released_snapshot': list(physical_released),
                    'physical_support_reference_origin_z_m': list(reference_z),
                    'physical_support_reference_frozen': list(reference_frozen),
                    'actual_source_contacts': source_contact,
                    **contents_poses,
                    'interarm_contacts': classify_interarm_contacts(interarm_contact_rows(contacts, robot_identities)),
                    'object_contact_rows': object_rows}
                if self.target_phases is not None: row['target_phases'] = list(self.target_phases)
                substeps.append(row)

    captured = Capture()
    captured.original_hook = original_hook
    return captured


def rename_held_audit_targets(audit: dict[str, Any], names: tuple[str, str]) -> dict[str, Any]:
    """Change legacy audit actor labels only; preserve every measured number."""
    if len(names) != 2 or len(audit.get('objects', ())) != 2:
        raise ValueError('Two measured object rows and canonical actor names are required')
    result = copy.deepcopy(audit)
    for row, name in zip(result['objects'], names): row['actor'] = name
    return result


def finalize_physics_capture(env: Any, raw: Any, captured: Any, output: Path, report: dict[str, Any],
                             frames: list, ticks: list, substeps: list, keys: list,
                             original_hook: Any, *, attachment_count: Any = None,
                             robot_contacts: list[dict[str, Any]] | None = None) -> None:
    """Preserve partial recordings after success, safety failure or exceptions."""
    import imageio.v2 as imageio
    import torch
    errors = report.setdefault('artifact_errors', [])
    if raw is not None:
        for name, read in (('total_environment_steps', lambda: int(as_numpy(raw.elapsed_steps)[0])),
                           ('torch_cuda_initialized', torch.cuda.is_initialized),
                           ('official_task_evaluation', lambda: json_value(raw.evaluate()))):
            try: report[name] = read()
            except Exception as exc: errors.append(f'final {name}: {type(exc).__name__}: {exc}')
        if original_hook is not None: raw._after_simulation_step = original_hook
    for filename, rows in (('ticks.jsonl', ticks), ('physics-substeps.jsonl', substeps)):
        try:
            with (output / filename).open('w') as f:
                for row in rows: f.write(json.dumps(json_value(row), allow_nan=False) + '\n')
        except Exception as exc: errors.append(f'{filename}: {type(exc).__name__}: {exc}')
    if env is not None:
        try:
            if getattr(env, '_trajectory_buffer', None) is not None: env.flush_trajectory()
        except Exception as exc: errors.append(f'trajectory: {type(exc).__name__}: {exc}')
        finally:
            try: env.close()
            except Exception as exc: errors.append(f'close: {type(exc).__name__}: {exc}')
        try:
            actions = np.asarray([np.concatenate([t['actions'][k] for k in keys]) for t in ticks]) if ticks else None
            report['recording'] = recording_summary(output / 'trajectory.h5', len(ticks), actions)
            report['recording']['action_key_order'] = keys
        except Exception as exc: errors.append(f'record check: {type(exc).__name__}: {exc}')
    if frames:
        try:
            imageio.mimsave(output / 'execution.mp4', [frames[0]] * 10 + frames + [frames[-1]] * 20, fps=20, macro_block_size=1)
            imageio.imwrite(output / 'preview.png', frames[-1])
            report['video'] = {'path': str(output / 'execution.mp4'), 'fps': 20, 'padding_frames_before': 10, 'padding_frames_after': 20}
        except Exception as exc: errors.append(f'video: {type(exc).__name__}: {exc}')
    report['success_only_termination_continuation_ticks'] = sum(t['success_only_termination_continued'] for t in ticks)
    report['phase_step_counts'] = {phase: sum(t['phase'] == phase for t in ticks) for phase in dict.fromkeys(t['phase'] for t in ticks)}
    report['first_official_success_step'] = next((t['environment_step'] for t in ticks
        if bool(np.asarray(t['official_evaluation'].get('success', False)).all())), None)
    # This is an environment observation, independent of artifact acceptance.
    # No final observation is represented by None rather than a fabricated fail.
    from .run_results import official_completion
    elapsed = report.get('total_environment_steps')
    horizon = report.get('max_episode_steps')
    report['official_200_step_complete'] = official_completion(
        report.get('official_task_evaluation'), elapsed,
        min(horizon, 200) if type(horizon) is int else None)
    forbidden = sum(bool(c.get('physical_contact')) and c.get('class') == 'forbidden'
        for row in substeps for c in row.get('object_contact_rows', []))
    interarm = sum(bool(c.get('physical_contact')) for row in substeps
        for c in row.get('interarm_contacts', []))
    robot_forbidden = (None if robot_contacts is None else sum(
        bool(c.get('physical_contact')) and not c.get('allowed')
        for row in robot_contacts for c in row.get('contacts', [])))
    control_dt, physics_dt = report.get('control_timestep_s'), report.get('physics_timestep_s')
    expected = (None if elapsed is None or not control_dt or not physics_dt else
        int(elapsed * round(control_dt / physics_dt)))
    report['physical_monitoring'] = {'observed_substeps': len(substeps),
        'expected_substeps': expected, 'forbidden_contact_count': int(forbidden),
        'interarm_forbidden_count': int(interarm),
        'robot_world_forbidden_count': robot_forbidden,
        'robot_world_observed_substeps': None if robot_contacts is None else len(robot_contacts),
        'robot_world_coverage_complete': None if robot_contacts is None else (
            expected is not None and expected > 0 and len(robot_contacts) == expected),
        'coverage_complete': expected is not None and expected > 0 and len(substeps) == expected,
        'scope': 'recorded100Hz actual contact observations, not continuous collision proof'}
    calls = report.get('native_planning_calls')
    report['planning_wall_seconds'] = sum(float(x['wall_seconds']) for x in calls) if calls is not None else sum(
        x['native_backend_wall_seconds'] for x in report.get('planning', []))
    report['physical_substeps'] = len(substeps)
    if 'attachment_capture' in report and attachment_count is not None:
        report['attachment_capture']['count_at_end'] = attachment_count
    if errors: report['status'] = 'failed'


def self_test() -> dict[str, Any]:
    """Synthetic array/contact/control boundary checks; no native environment."""
    initial, destination = ('table','source_plate'), ('basket','target_plate')
    raw_contact = [{'class':'forbidden','physical_contact':True,'points':[{'separation_m':-.001}]}]
    changed = apply_target_contact_policy(raw_contact,'handoff',lambda row,phase:'handoff_finger_allowed')
    assert raw_contact[0]['class']=='forbidden' and changed[0]['class']=='handoff_finger_allowed'
    assert changed[0]['points']==raw_contact[0]['points'] and changed[0]['physical_contact']
    try: apply_target_contact_policy(raw_contact,'handoff',lambda row,phase:'anything')
    except ValueError: pass
    else: raise AssertionError('Unknown custom contact class accepted')
    assert not physical_support_transition(False, True, .001, .0011, True)
    assert not physical_support_transition(False, True, .001, .003, False)
    assert not physical_support_transition(False, False, .001, .005, False)
    assert not physical_support_transition(False, True, .001, .005, True)
    assert physical_support_transition(False, True, .001, .005, False)
    assert physical_support_transition(True, False, .001, .000, True)
    try: physical_support_transition(False, True, 0., float('nan'), False)
    except ValueError: pass
    else: raise AssertionError('Nonfinite physical support height accepted')
    assert contact_class({'actor':'table'},0,False,'lift_and_hold',initial,destination)=='initial_support'
    assert contact_class({'actor':'table'},0,True,'carry_lower',initial,destination)=='forbidden'
    assert contact_class({'actor':'target_plate'},1,True,'release',initial,destination)=='destination_support'
    assert contact_class({'actor':'source_plate'},1,False,'preparation',initial,destination)=='initial_support'
    assert contact_class({'robot_index':0,'link_name':'panda_leftfinger'},0,True,'release',initial,destination)=='own_finger_allowed'
    assert contact_class({'robot_index':1,'link_name':'panda_leftfinger'},0,False,'descend',initial,destination)=='forbidden'
    point = {'separation_m': .039, 'impulse_Ns': [0.,0.,0.]}
    assert not physical_contact_points([point])
    assert physical_contact_points([{**point, 'separation_m':0.}])
    assert physical_contact_points([{**point, 'impulse_Ns':[1e-9,0.,0.]}])
    assert checked_geometry([[0.,0.,0.]], [0,0,0,1,0,0,0])[0].shape == (1,3)
    assert checked_geometry(np.empty((0,3)), [0,0,0,1,0,0,0])[0].shape == (0,3)
    for vertices, pose in (([[0,0]], [0,0,0,1,0,0,0]), ([[0,0,float('nan')]], [0,0,0,1,0,0,0]), ([[0,0,0]], [0,0,0,0,0,0,0])):
        try: checked_geometry(vertices,pose)
        except ValueError: pass
        else: raise AssertionError('Invalid geometry accepted')
    entity_a, entity_b, table_entity, source_entity = object(), object(), object(), object()
    adapter = NS(targets=[], target_names=('a','b'), initial_supports=(NS(name='table'),NS(name='source_plate')),
                 destination_supports=(NS(name='basket'),NS(name='target_plate')))
    identities = {entity_b:{'kind':'payload','actor':'b','target_index':1}, source_entity:{'kind':'actor','actor':'source_plate'}}
    contact = NS(bodies=[NS(entity=entity_b),NS(entity=source_entity)], points=[NS(position=np.zeros(3),normal=np.array([0,0,1]),impulse=np.zeros(3),separation=0.)])
    assert task_object_contact_rows([contact],identities,(True,False),('release','preparation'),adapter)[0]['class']=='initial_support'
    assert task_object_contact_rows([contact],identities,(True,False),'release',adapter)[0]['class']=='forbidden'
    audit = {'status':'passed','objects':[{'actor':'apple_out','value':.1},{'actor':'banana_out','value':.2}]}
    renamed = rename_held_audit_targets(audit,('pill_a','pill_b'))
    assert renamed['objects'][0] == {'actor':'pill_a','value':.1} and audit['objects'][0]['actor']=='apple_out'

    # The fake step calls exactly five synthetic hook observations. It tests
    # recording flow and budget accounting without constructing an env/scene.
    targets = []
    agents = []
    for index, entity in enumerate((entity_a,entity_b)):
        pose=NS(raw_pose=np.array([[0.,0,0,1,0,0,0]]),p=np.zeros((1,3)))
        target=NS(_objs=[entity],pose=pose,linear_velocity=np.zeros((1,3)),angular_velocity=np.zeros((1,3)))
        targets.append(target)
        robot=NS(get_links=lambda:[],get_qpos=lambda:np.zeros((1,9)),get_qvel=lambda:np.zeros((1,9)))
        agents.append(NS(robot=robot,tcp=NS(pose=pose),finger1_link=object(),finger2_link=object(),is_grasping=lambda a:np.array([False])))
    adapter.targets=tuple(targets)
    raw=NS(left_agent=agents[0],right_agent=agents[1],elapsed_steps=np.array([0]),sim_timestep=.01,
           scene=NS(actors={'a':targets[0],'b':targets[1]},px=NS(get_contacts=lambda:[]),get_pairwise_contact_forces=lambda a,b:np.zeros((1,3))),
           _after_simulation_step=lambda:None)
    frames,ticks,substeps=[],[],[]
    class Fake:
        def render(self) -> np.ndarray: return np.zeros((2,3,3),dtype=np.uint8)
        def step(self, action: Any) -> tuple[Any,...]:
            for _ in range(5): captured.record_substep()
            raw.elapsed_steps+=1
            return None,0.,np.array([False]),np.array([False]),{'success':np.array([False])}
    fake=Fake()
    captured=make_physics_capture(fake,raw,adapter,NS(mode='positive'),{'fault_events':[]},frames,ticks,substeps,['l','r'])
    captured.step({'l':np.zeros(8),'r':np.zeros(8)})
    assert len(ticks)==1 and len(substeps)==5 and len(frames)==1
    assert substeps[-1]['control_step']==1 and ticks[0]['environment_step']==1
    captured.target_phases=('retreat','preparation')
    captured.step({'l':np.zeros(8),'r':np.zeros(8)})
    assert ticks[-1]['target_phases']==['retreat','preparation']
    raw.elapsed_steps[0]=200
    try: captured.step({'l':np.zeros(8),'r':np.zeros(8)})
    except RuntimeError: pass
    else: raise AssertionError('Hard200 budget bypassed')
    assert len(ticks)==2 and len(substeps)==10
    frames400,ticks400,substeps400=[],[],[]
    raw.elapsed_steps[0]=200
    captured=make_physics_capture(fake,raw,adapter,NS(mode='positive'),{'fault_events':[]},
        frames400,ticks400,substeps400,['l','r'],hard_step_budget=400)
    captured.step({'l':np.zeros(8),'r':np.zeros(8)})
    assert raw.elapsed_steps[0]==201 and len(ticks400)==1 and len(substeps400)==5
    raw.elapsed_steps[0]=400
    try: captured.step({'l':np.zeros(8),'r':np.zeros(8)})
    except RuntimeError: pass
    else: raise AssertionError('Declared engineering400 budget bypassed')
    for invalid_budget in (True, 201, 401):
        try: make_physics_capture(fake,raw,adapter,NS(mode='positive'),{'fault_events':[]},[],[],[],['l','r'],hard_step_budget=invalid_budget)
        except ValueError: pass
        else: raise AssertionError('Unspecified budget accepted')
    # Adapter geometry/phase proofs are unit-tested separately. Here an
    # explicit synthetic provider tests the100Hz recording/classification path,
    # including per-target serial phases and preservation of native raw fields.
    contents_entity = object()
    contents_actor = NS(name='apple_in', _objs=[contents_entity], pose=targets[0].pose)
    raw.scene.actors['apple_in'] = contents_actor
    adapter.destination_contents_state_names = ('apple_in', 'banana_in')
    proof_calls = []
    def prove(index: int, other: str, phase: str, points: list) -> dict:
        proof_calls.append((index, other, phase))
        return {'allowed': index == 0 and other == 'apple_in' and phase in ('release','retreat','verify_placement'),
                'policy_version':'synthetic-recorder-provider', 'phase':phase}
    adapter.destination_contents_contact_proof = prove
    adapter.destination_contents_policy_configuration = lambda: {'enabled':True,'pairs':[{'contents_state_name':'apple_in'}]}
    adapter.destination_contents_pose_snapshot = lambda: {
        'destination_contents_poses': [[0.,0,0,1,0,0,0],[.5,0,0,1,0,0,0]],
        'destination_basket_poses': [[0.,0,0,1,0,0,0],[.5,0,0,1,0,0,0]]}
    content_contact = NS(bodies=[NS(entity=entity_a),NS(entity=contents_entity)],
        points=[NS(position=np.array([0.,0,.05]), normal=np.array([0.,0,1.]),impulse=np.array([0.,0,.01]),separation=-.001)])
    content_ids = {entity_a:{'kind':'payload','actor':'a','target_index':0},
                   contents_entity:{'kind':'actor','actor':'apple_in'}}
    row = task_object_contact_rows([content_contact],content_ids,(True,True),('release','preparation'),adapter)[0]
    assert row['class']=='destination_contents' and row['destination_contents_proof']['phase']=='release'
    assert row['physical_contact'] and row['points']==[{'position_m':[0.,0,.05],'normal':[0.,0,1.],'impulse_Ns':[0.,0,.01],'separation_m':-.001}]
    row_bad = task_object_contact_rows([content_contact],content_ids,(True,True),('carry_lower','release'),adapter)[0]
    assert row_bad['class']=='forbidden' and not row_bad['destination_contents_proof']['allowed']
    before = len(proof_calls)
    skipped = task_object_contact_rows([content_contact],content_ids,(True,True),'release',adapter,
                                      include_destination_contents_proof=False)[0]
    assert skipped['class']=='forbidden' and 'destination_contents_proof' not in skipped and len(proof_calls)==before
    assert apply_target_contact_policy([row],'release',lambda value,phase:value['class'])[0]['class']=='destination_contents'
    try: apply_target_contact_policy([row_bad],'carry_lower',lambda value,phase:'destination_contents')
    except ValueError: pass
    else: raise AssertionError('Custom policy invented contents proof')
    raw.scene.px.get_contacts = lambda:[content_contact]
    raw.elapsed_steps[0] = 0
    content_frames, content_ticks, content_substeps = [], [], []
    captured=make_physics_capture(fake,raw,adapter,NS(mode='positive'),{'fault_events':[]},
        content_frames,content_ticks,content_substeps,['l','r'])
    captured.target_phases=('release','preparation')
    before=len(proof_calls)
    captured.step({'l':np.zeros(8),'r':np.zeros(8)})
    assert len(content_substeps)==5 and len(proof_calls)-before==5
    assert all(s['object_contact_rows'][0]['class']=='destination_contents' for s in content_substeps)
    assert all(s['destination_contents_poses']==adapter.destination_contents_pose_snapshot()['destination_contents_poses'] for s in content_substeps)
    assert all(s['destination_basket_poses']==adapter.destination_contents_pose_snapshot()['destination_basket_poses'] for s in content_substeps)
    # Export exact raw native local geometry + reset pose + state names for
    # contents, independently from the proof provider above.
    import tempfile
    local=np.array([[x,y,z] for x in (-.01,.01) for y in (-.02,.02) for z in (0.,.04)])
    def geom_actor(name: str) -> Any:
        return NS(name=name,pose=NS(raw_pose=np.array([[0.,0,0,1,0,0,0]])),
                  get_first_collision_mesh=lambda to_world_frame=False:NS(vertices=local.copy()))
    actors=[geom_actor(n) for n in ('apple_out','banana_out','apple_in','banana_in','basket_a','basket_b','table')]
    export_adapter=NS(targets=tuple(actors[:2]),target_names=('apple_out','banana_out'),destination_actors=tuple(actors[4:6]),
        initial_supports=(actors[-1],actors[-1]),destination_supports=tuple(actors[4:6]),destination_contents=tuple(actors[2:4]))
    export_raw=NS(scene=NS(actors={a.name:a for a in actors}))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory);manifest=save_task_geometry(path,export_raw,export_adapter)
        arrays=np.load(path/'task-geometry.npz')
        for i,name in enumerate(('apple_in','banana_in')):
            key=f'contents{i}'
            assert manifest['actors'][key]['state_name']==name
            assert np.array_equal(arrays[key+'_vertices'],local)
            assert manifest['actors'][key]['local_vertices_sha256']==hashlib.sha256(np.ascontiguousarray(local,dtype=np.float64).tobytes(order='C')).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        final_report = {'status': 'passed', 'max_episode_steps': 200,
            'total_environment_steps': 171, 'official_task_evaluation': {'success': [True]},
            'project_complete_physical_closure': True, 'artifact_errors': ['video: fixture failure']}
        finalize_physics_capture(None, None, None, Path(directory), final_report,
            [], [], [], [], None)
        assert final_report['status'] == 'failed'
        assert final_report['official_200_step_complete'] is True
        assert final_report['project_complete_physical_closure'] is True
        assert not final_report['physical_monitoring']['coverage_complete']
        unknown_report = {'status': 'failed', 'max_episode_steps': 200}
        finalize_physics_capture(None, None, None, Path(directory), unknown_report,
            [], [], [], [], None)
        assert unknown_report['official_200_step_complete'] is None
        engineering = {'status': 'passed', 'max_episode_steps': 400,
            'total_environment_steps': 259, 'official_task_evaluation': {'success': [True]}}
        finalize_physics_capture(None, None, None, Path(directory), engineering,
            [], [], [], [], None)
        assert engineering['official_200_step_complete'] is False
        contact_report = {'total_environment_steps': 1, 'control_timestep_s': .05,
            'physics_timestep_s': .01}
        robot_rows = [{'contacts': []} for _ in range(5)]
        robot_rows[-1]['contacts'] = [{'physical_contact': True, 'allowed': False}]
        finalize_physics_capture(None, None, None, Path(directory), contact_report,
            [], [], [{} for _ in range(5)], [], None, robot_contacts=robot_rows)
        monitoring = contact_report['physical_monitoring']
        assert monitoring['forbidden_contact_count'] == monitoring['interarm_forbidden_count'] == 0
        assert monitoring['robot_world_forbidden_count'] == 1
        assert monitoring['robot_world_observed_substeps'] == monitoring['expected_substeps'] == 5
        assert monitoring['coverage_complete'] and monitoring['robot_world_coverage_complete']
    return {'status':'passed','official_and_recording_results_independent':True,
            'contact_and_geometry_boundary_cases':17,'physical_support_transition_cases':7,'fake_control_steps':2,
            'fake_substep_observations':10,'engineering400_fake_control_steps':1,'custom_contact_policy_cases':2,
            'recording_version':RECORDING_VERSION,'destination_contents_boundary_and_recording_cases':7,
            'contents_fake_substep_observations':5,'contents_proof_calls_per_substep':1,
            'environment_creations':0,'native_physics_steps':0,
            'shapes':{'action':[16],'single_pose':[7],'single_geometry':[1,3]}}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if not args.self_test: parser.error('use --self-test')
    print(json.dumps(self_test(),indent=2))
