"""Physical cuRobo grasp/payload integration through TIG's existing executors.

One real dual-Panda scene, official 200 control ticks. All native trajectories,
failed phases, measured contacts and standard RecordEpisode states are saved.
Payloads are collision predictions only: physics supplies grasp/contact forces.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .curobo_backend import trajectory_from_result, pose_error
from .curobo_scene import _matrix, _pose
from .curobo_sessions import PROFILE_LIMITS
from .trajectory_execution import as_numpy, bind_arm
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .validate_curobo import check_fk, git_identity, save_json
from .validate_curobo_sessions import verify_finger_spheres
from .validate_multitask import termination_handler
from .validate_sync import held_substep_audit, hold_joint_substep_audit

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
TASK = 'TwoRobotPickAppleBananaToBaskets-v1'


def state_digest(raw: Any) -> str:
    """Hash every actual actor/articulation state and elapsed step, without motion."""
    from .validate_path_conflict import _state_arrays
    values = _state_arrays(raw.get_state_dict())
    digest = hashlib.sha256(as_numpy(raw.elapsed_steps).tobytes())
    for name, value in sorted(values.items()):
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite actual simulator state')
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def contact_allowed(sides: list[dict[str, Any]], phase: str,
                    srdf: dict[int, set[frozenset[str]]]) -> bool:
    """Exact physical robot-contact permissions; no broad target/palm exception."""
    if len(sides) != 2:
        return False
    if any(s.get('link_name') == 'panda_link0' for s in sides) and any(s.get('actor') == 'table' for s in sides):
        return True
    if all(s.get('kind') == 'robot' for s in sides):
        index = sides[0]['robot_index']
        return sides[1]['robot_index'] == index and frozenset(s['link_name'] for s in sides) in srdf[index]
    if phase in ('descend', 'close_and_confirm', 'lift_and_hold', 'release') or phase.startswith('carry_'):
        for i in (0, 1):
            robot, target = sides[i], sides[1-i]
            if (robot.get('kind') == 'robot' and target.get('kind') == 'payload'
                    and robot['robot_index'] == target['target_index']
                    and robot['link_name'] in ('panda_leftfinger', 'panda_rightfinger')):
                return True
    return False


def install_full_robot_contact_capture(raw: Any, capture: Any, adapter: Any,
                                       output: list[dict[str, Any]]) -> None:
    """Supplement existing target/interarm observer with all robot/world contacts."""
    from .multitask_recording import contact_identities
    from .scene_collision import scene_entities
    from .validate_held_path_conflict import physical_contact_points
    agents = tuple(raw.agent.agents)
    identities, _ = contact_identities(raw, agents, adapter.target_names, adapter.targets)
    for name, entity in scene_entities(raw).items():
        identities.setdefault(entity, {'kind': 'actor', 'actor': name})
    srdf = {}
    for i, agent in enumerate(agents):
        tree = ET.parse(Path(agent.urdf_path).with_suffix('.srdf')).getroot()
        srdf[i] = {frozenset((node.attrib['link1'], node.attrib['link2']))
                   for node in tree.findall('disable_collisions')}
    original_record, original_step = capture.record_substep, capture.step

    def record() -> None:
        original_record()
        rows = []
        for contact in raw.scene.px.get_contacts():
            sides = [identities.get(body.entity, {'kind': 'unknown', 'actor': body.entity.name})
                     for body in contact.bodies]
            if not any(side.get('kind') == 'robot' for side in sides):
                continue
            points = [{'position_m': np.asarray(point.position).tolist(),
                       'normal': np.asarray(point.normal).tolist(),
                       'impulse_Ns': np.asarray(point.impulse).tolist(),
                       'separation_m': float(point.separation)} for point in contact.points]
            rows.append({'sides': sides, 'points': points,
                         'physical_contact': physical_contact_points(points),
                         'allowed': contact_allowed(sides, capture.phase, srdf)})
        output.append({'substep': len(output)+1, 'phase': capture.phase,
                       'control_step': capture.pending_control_step, 'contacts': rows})

    def step(action: dict[str, Any]) -> tuple[Any, ...]:
        first = len(output)
        outcome = original_step(action)
        if any(c['physical_contact'] and not c['allowed']
               for row in output[first:] for c in row['contacts']):
            raise RuntimeError('Forbidden actual robot/world contact at100Hz')
        return outcome

    capture.record_substep = record
    capture.step = step


def audit_native_state(updates: list[dict[str, Any]], data: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """Check actual obstacle pose and enabled readback, retaining peer payloads."""
    from scipy.spatial.transform import Rotation
    rows = []
    for update, snapshot in zip(updates, data):
        world = update['native_update']['native_readback']['world']
        state = snapshot['payload_state']
        attachment = state['attachment']
        disabled = set(snapshot['payload_model']['target_obstacle_names']) if attachment else set()
        native_payload = update['native_update']['payload']
        targets = set(snapshot['payload_model']['target_obstacle_names'])
        for model_name, model in native_payload['models'].items():
            expected_finger = [5 if state['contact_allowed'] else 7]
            expected_payload = [3 if state['source_support_allowed'] else 7]
            if (any(value != expected_finger for value in model['finger_collision_masks'].values())
                    or model['payload_collision_masks'] != expected_payload
                    or model['payload_active_sphere_count'] != (len(attachment['spheres']) if attachment else 0)):
                raise RuntimeError(f'Incorrect actual native collision permissions in {model_name}')
        for name, obstacle in native_payload['world'].items():
            expected_layer = 2 if name in targets else (4 if name == 'scene::table::shape_0' else 1)
            if obstacle['layers'] != expected_layer or obstacle['enabled'] != (name not in disabled):
                raise RuntimeError(f'Incorrect native collision layer for {name}')
        maximum, count, peer = 0., 0, 0
        for kind, shapes in snapshot['world'].items():
            if set(world[kind]) != set(shapes):
                raise RuntimeError('Native payload world coverage differs from actual scene')
            for name, shape in shapes.items():
                pose = np.asarray(shape['pose'], dtype=float)
                rotation = Rotation.from_quat(np.r_[pose[4:], pose[3]]).as_matrix()
                expected = np.r_[-rotation.T @ pose[:3], pose[3], -pose[4:]]
                actual = world[kind][name]
                error = pose_error(actual['inverse_pose_xyz_wxyz'], expected)
                if (actual['enabled'] != int(name not in disabled)
                        or error['position_m'] > 1e-5 or error['rotation_rad'] > 1e-4):
                    raise RuntimeError(f'Incorrect native held-target/world state for {name}: {error}')
                maximum = max(maximum, error['position_m'])
                count += 1
                peer += name.startswith('peer::')
        rows.append({'session_id': update['session_id'], 'world_shapes': count,
                     'peer_robot_shapes': peer, 'disabled_own_target_shapes': sorted(disabled),
                     'max_inverse_pose_error_m': maximum})
    return rows


def audit_payload_spheres(fk: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Read actual native base-frame payload spheres and compare real actor geometry."""
    spheres = np.asarray(fk['link_spheres']['tig_payload'], dtype=float)
    model, state = snapshot['payload_model'], snapshot['payload_state']
    if spheres.shape != (1, model['capacity'], 4) or not np.isfinite(spheres).all():
        raise ValueError('Invalid actual native payload sphere tensor')
    active = spheres[0, :, 3] > 0
    if state['attachment'] is None:
        if active.any():
            raise RuntimeError('Unattached model retains active payload collision geometry')
        return {'status': 'passed', 'active_spheres': 0}
    attachment = state['attachment']
    local = np.asarray(attachment['spheres'], dtype=float)
    transform = _matrix(attachment['object_pose_base_wxyz'])
    expected = local[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    actual = spheres[0, active]
    if actual.shape != local.shape:
        raise RuntimeError('Native attachment lost actual target collision spheres')
    error = float(np.max(np.abs(actual[:, :3] - expected)))
    if error > 1e-5 or not np.allclose(actual[:, 3], local[:, 3], rtol=0, atol=1e-7):
        raise RuntimeError(f'Native held geometry differs from actual measured actor: {error}')
    return {'status': 'passed', 'active_spheres': len(actual), 'max_center_coordinate_error_m': error}


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import sapien
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .curobo_payload import CuroboPayloadSessions, export_payload_snapshot
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .dual_arm_execution import (execute_dual_reach, close_dual_and_confirm,
        execute_dual_held_move, release_dual_and_confirm, verify_dual_placement)
    from .combined_collision import validate_schedule
    from .multitask_adapters import get_task_adapter
    from .multitask_recording import make_physics_capture, finalize_physics_capture, save_task_geometry
    from .validate_sync_compact import compact_route_positions

    env, raw, capture, worker, checker, original_hook = None, None, None, None, None, None
    frames, ticks, substeps, contacts, keys = [], [], [], [], []
    attachment_count = 0
    try:
        base = gym.make(TASK, obs_mode='none', reward_mode='none', control_mode='pd_joint_pos',
            sim_backend='cpu', render_backend='gpu', render_mode='rgb_array', max_episode_steps=200)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name='trajectory', save_video=False,
            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
            source_type='motionplanning', source_desc='cuRobo v0.9 real confirmed grasp and payload geometry')
        env.reset(seed=args.seed)
        raw = env.unwrapped
        adapter = get_task_adapter(TASK, raw)
        targets, agents = adapter.targets, tuple(raw.agent.agents)
        keys = list(raw.agent.agents_dict)
        arms = [bind_arm(key, agent) for key, agent in raw.agent.agents_dict.items()]
        report.update(task_configuration=adapter.configuration(), target_names=list(adapter.target_names),
            action_key_order=keys, stages=[], fault_events=[], task_geometry=save_task_geometry(output, raw, adapter))
        from .validate_path_conflict import _state_arrays
        np.savez(output/'initial-state.npz', **_state_arrays(raw.get_state_dict()))
        capture = make_physics_capture(env, raw, adapter, args, report, frames, ticks, substeps, keys,
            checker_provider=lambda: checker, hard_step_budget=200)
        original_hook = capture.original_hook
        install_full_robot_contact_capture(raw, capture, adapter, contacts)
        raw._after_simulation_step = capture.record_substep
        capture.frame()
        hold = {arm.key: np.r_[arm.arm_qpos(), 1.].astype(np.float32) for arm in arms}
        for _ in range(10):
            capture.step(hold)
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=agent.robot.pose, multi_robot_id=i,
            joint_vel_limits=.9, joint_acc_limits=.9, debug=False, vis=False,
            visualize_target_grasp_pose=False, print_env_info=False) for i, agent in enumerate(agents)]
        checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=targets)
        report['external_scene_coverage'] = checker.coverage
        frozen = state_digest(raw)
        initial = tuple(export_payload_snapshot(raw, i, target) for i, target in enumerate(targets))
        for i, data in enumerate(initial):
            save_json(output/f'initial-arm-{i}-snapshot.json', data)
        capacities = [data['payload_model']['capacity'] for data in initial]
        worker = CuroboPayloadSessions(os.environ['CUDA_VISIBLE_DEVICES'], output/'worker.stderr.log')
        report['initialization'] = worker.initialize_pair(initial, seed=0, interpolation_dt=.02, warmup=True)
        report['initial_sessions'] = worker.inspect_sessions()
        if state_digest(raw) != frozen:
            raise RuntimeError('Initial export or planner construction changed physics')
        grasp_evidence: list[dict[str, Any] | None] = [None, None]

        def begin(name: str) -> None:
            report['stage'] = capture.phase = name
            save_json(output/'report.json', report)

        def finish(name: str, result: dict[str, Any]) -> None:
            report['stages'].append({'name': name, **result})
            save_json(output/'report.json', report)
            print(json.dumps({'phase': name, 'status': result['status'],
                              'elapsed_steps': int(as_numpy(raw.elapsed_steps)[0])}), flush=True)
            if result['status'] != 'passed':
                raise RuntimeError(f'{name}: {result["status"]}: {result.get("reason")}')

        def plan(name: str, poses: tuple[Any, Any], *, closed: bool,
                 profile: str, contact: bool = False, source: bool = False) -> Any:
            begin(name)
            frozen = state_digest(raw)
            started = time.monotonic()
            snapshots = tuple(export_payload_snapshot(raw, i, target,
                contact_allowed=contact or closed, grasp_evidence=grasp_evidence[i] if closed else None,
                source_support_allowed=source, capacity=capacities[i]) for i, target in enumerate(targets))
            for i, data in enumerate(snapshots):
                save_json(output/f'{name}-arm-{i}-snapshot.json', data)
            row: dict[str, Any] = {'phase': name, 'profile': profile,
                'before_control_step': int(as_numpy(raw.elapsed_steps)[0]), 'closed': closed,
                'native_source_support_exception': source,
                'source_exact_checker': 'NumericalTableSupportCollisionChecker' if source else None}
            report.setdefault('planning', []).append(row)
            row['updates'] = worker.update_pair(snapshots, profile=profile, snapshot_id=name)
            row['world_readback'] = audit_native_state(row['updates'], snapshots)
            row['fk_checks'], trajectories = [], []
            velocity, acceleration = PROFILE_LIMITS[profile]
            for i, (arm, goal, data) in enumerate(zip(arms, poses, snapshots)):
                q = arm.arm_qpos()
                fk = worker.fk(arm=i, profile=profile, snapshot_id=name, positions=[q.tolist()])
                save_json(output/f'{name}-arm-{i}-fk.json', fk)
                row['fk_checks'].append({'arm': i, 'tcp': check_fk(fk, data, q[None]),
                    'fingers': verify_finger_spheres(fk, data, q[None]),
                    'payload': audit_payload_spheres(fk, data)})
                goal_base = arm.agent.robot.pose.sp.inv() * goal
                call_started = time.perf_counter()
                try:
                    result = worker.plan_pose(arm=i, profile=profile, snapshot_id=name,
                        start=q.tolist(), goal_pose=np.r_[goal_base.p, goal_base.q].tolist(), max_attempts=4)
                finally:
                    report.setdefault('native_planning_calls', []).append({'phase': name, 'arm_index': i,
                        'wall_seconds': time.perf_counter()-call_started})
                save_json(output/f'{name}-arm-{i}-native-result.json', result)
                trajectory, audit = trajectory_from_result(result, action_key=arm.key,
                    joint_names=arm.names, start=q, velocity_limit=velocity, acceleration_limit=acceleration)
                trajectories.append(ArmTrajectory(trajectory.action_key, trajectory.joint_names,
                    trajectory.times_s, trajectory.positions_rad, -1. if closed else 1.))
                row.setdefault('trajectory_audits', []).append(audit)
            if state_digest(raw) != frozen:
                raise RuntimeError('Planning/model updates changed actual simulator state')
            row.update(planning_environment_steps=0, planning_pipeline_wall_s=time.monotonic()-started)
            schedule = build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)
            row['scheduled_steps'] = schedule.num_steps
            if schedule.num_steps + int(as_numpy(raw.elapsed_steps)[0]) + 3 > 200:
                raise RuntimeError('Planned phase cannot finish within official200 budget')
            checker.refresh()
            checker.set_target_contacts_allowed((contact or closed,)*2)
            try:
                row['collision_validation'] = validate_schedule(schedule, *(arm.arm_qpos() for arm in arms), checker)
            finally:
                checker.set_target_contacts_allowed((False, False))
            np.savez(output/f'{name}-schedule.npz', left_times=trajectories[0].times_s,
                right_times=trajectories[1].times_s, left_positions=trajectories[0].positions_rad,
                right_positions=trajectories[1].positions_rad, target_times=schedule.target_times_s,
                left_commands=schedule.left_commands, right_commands=schedule.right_commands)
            save_json(output/'report.json', report)
            if row['collision_validation']['status'] != 'passed':
                raise RuntimeError(f'{name}: paired exact collision validation rejected candidate')
            return schedule

        pairs = adapter.grasp_poses()
        report['grasp_targets'] = [[np.r_[pose.p, pose.q].tolist() for pose in pair] for pair in pairs]
        for index, name, allowed in ((0, 'pregrasp', False), (1, 'descend', True)):
            schedule = plan(name, (pairs[0][index], pairs[1][index]), closed=False,
                            profile='approach', contact=allowed)
            finish(name, execute_dual_reach(capture, schedule, checker, allow_target_contacts=(allowed, allowed)))
        begin('close_and_confirm')
        closed = close_dual_and_confirm(capture, checker, targets)
        finish('close_and_confirm', closed)
        checker.attach()
        attachment_count += 1
        references = np.asarray(checker.attachments).copy()
        grasp_evidence = [{'evidence_id': f'real-close-step-{int(as_numpy(raw.elapsed_steps)[0])}',
            'confirmation': closed, 'tcp_to_object_wxyz': _pose(reference)} for reference in references]
        report['attachment_capture'] = {'count': attachment_count,
            'environment_step': int(as_numpy(raw.elapsed_steps)[0]), 'T_tcp_objects': references.tolist(),
            'physical_weld': False}
        original_q = [as_numpy(agent.tcp.pose.q)[0].copy() for agent in agents]
        carry_q = adapter.carry_quaternions()
        releases = adapter.release_poses(carry_quaternions=carry_q)
        route = compact_route_positions([as_numpy(agent.tcp.pose.p)[0] for agent in agents],
            [np.asarray(pose.p) for pose in releases], route='merge-lift')
        report['route'] = {name: value.tolist() if isinstance(value, np.ndarray) else value
                           for name, value in route.items()}
        lift_poses = tuple(sapien.Pose(xyz, q) for xyz, q in zip(route['lift'], original_q))
        schedule = plan('lift_and_hold', lift_poses, closed=True, profile='transport', source=True)
        lifted = execute_dual_held_move(capture, schedule, checker, targets,
            min_height_gain_m=(.10, .10), hold_steps=5)
        finish('lift_and_hold', lifted)
        if not all(support.support_released for support in checker._supports):
            raise RuntimeError('Exact geometric source support did not permanently release')
        if not substeps or not all(substeps[-1]['physical_support_released_snapshot']):
            raise RuntimeError('Physical source support has not ended')
        hover = [int(lifted['initial_elapsed_steps'][0])+row['step']
                 for row in lifted['trace'] if row['phase']=='hover']
        report['physical_hold_joint_audit'] = hold_joint_substep_audit(substeps, hover)
        if report['physical_hold_joint_audit']['status'] != 'passed':
            raise RuntimeError('Physical hover joint stability failed')
        carry_poses = tuple(sapien.Pose(xyz, q) for xyz, q in zip(route['above_baskets'], carry_q))
        schedule = plan('carry_orient_and_transport', carry_poses, closed=True, profile='transport')
        carried = execute_dual_held_move(capture, schedule, checker, targets, hold_steps=5)
        finish('carry_orient_and_transport', carried)
        report['held_physics_audit'] = held_substep_audit(substeps, references)
        if report['held_physics_audit']['status'] != 'passed':
            raise RuntimeError('100Hz actual payload slip audit failed')
        report['real_grasp_and_payload_status'] = 'passed'
        if args.stop_after_held:
            report['status'] = 'passed'
            return
        lower = tuple(sapien.Pose(xyz, q) for xyz, q in zip(route['release'], carry_q))
        schedule = plan('carry_lower', lower, closed=True, profile='transport')
        finish('carry_lower', execute_dual_held_move(capture, schedule, checker, targets))
        report['held_physics_audit'] = held_substep_audit(substeps, references)
        if report['held_physics_audit']['status'] != 'passed':
            raise RuntimeError('100Hz pre-release payload slip audit failed')
        checker.detach()
        begin('release')
        finish('release', release_dual_and_confirm(capture, checker, targets))
        grasp_evidence = [None, None]
        if args.stop_after_release:
            frozen = state_digest(raw)
            released = tuple(export_payload_snapshot(raw, i, target, capacity=capacities[i])
                             for i, target in enumerate(targets))
            for i, data in enumerate(released):
                save_json(output/f'released-arm-{i}-snapshot.json', data)
            updates = worker.update_pair(released, profile='transport', snapshot_id='released-state')
            report['released_world_readback'] = audit_native_state(updates, released)
            report['released_updates'] = updates
            report['released_payload_geometry'] = []
            for i, data in enumerate(released):
                fk = worker.fk(arm=i, profile='transport', snapshot_id='released-state',
                               positions=[arms[i].arm_qpos().tolist()])
                save_json(output/f'released-arm-{i}-fk.json', fk)
                report['released_payload_geometry'].append(audit_payload_spheres(fk, data))
            if state_digest(raw) != frozen:
                raise RuntimeError('Native payload detach changed simulator state')
            report.update(status='passed', native_payload_detach_verified=True,
                          placement_stability_evaluated=False)
            return
        retreat_start = np.asarray([as_numpy(agent.tcp.pose.p)[0].copy() for agent in agents])
        retreat = tuple(sapien.Pose(position+[0., 0., .081], as_numpy(agent.tcp.pose.q)[0])
                        for position, agent in zip(retreat_start, agents))
        schedule = plan('retreat', retreat, closed=False, profile='transport')
        finish('retreat', execute_dual_reach(capture, schedule, checker))
        retreat_gain = np.asarray([as_numpy(agent.tcp.pose.p)[0] for agent in agents])-retreat_start
        report['actual_retreat_displacement_m'] = retreat_gain.tolist()
        if np.any(retreat_gain[:, 2] < .08):
            raise RuntimeError('Actual retreat gain below8cm')
        begin('verify_placement')
        placed = verify_dual_placement(capture, checker, targets,
            official_predicate_keys=adapter.official_predicate_keys, region_predicates=adapter.region_predicates())
        finish('verify_placement', placed)
        evaluation = raw.evaluate()
        report['official_200_step_complete'] = bool(np.asarray(as_numpy(evaluation['success'])).all()
            and int(as_numpy(raw.elapsed_steps)[0]) < 200)
        report['status'] = 'passed' if report['official_200_step_complete'] else 'failed'
    finally:
        if worker is not None:
            try:
                if not worker._rpc.failed:
                    report['final_sessions'] = worker.inspect_sessions()
                worker.close()
            except Exception as exc:
                report.setdefault('artifact_errors', []).append(f'worker close: {exc}')
            report['worker_requests'] = worker.receipts
        save_json(output/'all-robot-contacts.json', contacts)
        if env is not None:
            finalize_physics_capture(env, raw, capture, output, report, frames, ticks, substeps,
                keys, original_hook, attachment_count=attachment_count)


def self_test() -> dict[str, Any]:
    from .curobo_payload import self_test as model_test
    from .validate_multitask import termination_self_test
    termination_self_test()
    model = model_test()
    robot = {'kind':'robot','robot_index':0,'link_name':'panda_leftfinger'}
    target = {'kind':'payload','target_index':0}
    assert contact_allowed([robot,target], 'descend', {0:set()})
    assert not contact_allowed([robot,target], 'pregrasp', {0:set()})
    assert not contact_allowed([{**robot,'link_name':'panda_hand'},target], 'carry_x', {0:set()})
    assert not contact_allowed([robot,{**target,'target_index':1}], 'descend', {0:set()})
    assert not contact_allowed([robot,{'kind':'unknown'}], 'release', {0:set()})
    data = {'payload_model': {'capacity':1}, 'payload_state': {'attachment': None}}
    assert audit_payload_spheres({'link_spheres': {'tig_payload': [[[0.,0.,0.,-1.]]]}},data)['active_spheres']==0
    data['payload_state']['attachment'] = {'spheres':[[.01,0.,0.,.02]],
                                          'object_pose_base_wxyz':[1.,0.,0.,1.,0.,0.,0.]}
    fk = {'link_spheres': {'tig_payload': [[[1.01,0.,0.,.02]]]}}
    assert audit_payload_spheres(fk,data)['active_spheres']==1
    fk['link_spheres']['tig_payload'][0][0][0] += .001
    try:
        audit_payload_spheres(fk,data)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Stale payload geometry accepted')
    return {'status':'passed','model':model,'contact_policy_and_payload_geometry':'passed'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--stop-after-held', action='store_true')
    parser.add_argument('--stop-after-release', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
        return 0
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    signal.signal(signal.SIGTERM, termination_handler)
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        parser.error('One explicit GPU UUID required')
    if any(os.environ.get(f'MANI_SKILL_L{i}') for i in (1,2,3)):
        parser.error('This fixture requires L0')
    if args.output_dir is None:
        parser.error('An unused output directory is required')
    args.mode = 'positive'
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    sources = list(HERE.glob('*.py'))+[ROOT/f'infrastructure/{name}' for name in
        ('curobo_tig_worker.py','curobo_tig_session_worker.py','curobo_tig_payload_worker.py')]
    archive = output/'source'
    archive.mkdir()
    for source in sources:
        shutil.copy2(source, archive/source.name)
    report: dict[str, Any] = {'status':'running','seed':args.seed,'task':TASK,
        'started_at':datetime.now(timezone.utc).isoformat(),'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
        'max_episode_steps':200,'official_200_step_complete':False,'real_grasp_and_payload_status':'not_completed',
        'scope':'actual confirmed grasps and measured payload collision through existing shared executor',
        'stop_after_held':args.stop_after_held,'stop_after_release':args.stop_after_release,
        'physical_pose_assignment':False,'physical_attachment_created':False,
        'physical_collision_masks_modified':False,'native_timeopt_payload_inertia_applied':False,
        'standard_recording_replay_evaluated':False,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'official_tig':git_identity(ROOT/'The-Imitator-Game'),
        'official_curobo':git_identity(ROOT/'infrastructure/backends/curobo')}
    save_json(output/'report.json',report)
    started = time.monotonic()
    try:
        native_run(args,output,report)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report.update(wall_s=time.monotonic()-started,completed_at=datetime.now(timezone.utc).isoformat())
    save_json(output/'report.json',report)
    print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
