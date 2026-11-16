"""Validate persistent cuRobo sessions using real, unloaded TIG phases.

All three arm motions use the existing shared clock, paired collision checker
and dual reach executor. An intervening empty gripper cycle changes physical
finger positions, without grasping or attaching an object. Planning and model
updates must leave the complete measured simulator state unchanged.
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
import time
import traceback
from typing import Any

import numpy as np

from .curobo_backend import pose_error, trajectory_from_result, urdf_tcp_pose
from .trajectory_execution import as_numpy, bind_arm
from .trajectory_schedule import build_hold_schedule
from .validate_curobo import ReachCapture, check_fk, git_identity, save_json
from .validate_execution import target_offsets

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
PROFILES = {'approach': (.9, .9), 'transport': (1.5, 2.6)}


def state_digest(capture: ReachCapture) -> str:
    state = {'elapsed_steps': as_numpy(capture.unwrapped.elapsed_steps).tolist(),
             **capture.measurement()}
    payload = json.dumps(state, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_frozen(capture: ReachCapture, before: str) -> None:
    if state_digest(capture) != before:
        raise RuntimeError('Planning or model updates changed measured simulator state')


def verify_world_readback(updates: list[dict[str, Any]],
                          snapshots: tuple[dict[str, Any], dict[str, Any]]) -> list[dict[str, Any]]:
    """Check every enabled native obstacle inverse pose against actual exports."""
    from scipy.spatial.transform import Rotation
    rows = []
    for receipt, data in zip(updates, snapshots):
        readback = receipt['native_update']['native_readback']['world']
        count, max_position, max_rotation = 0, 0., 0.
        for kind, shapes in data['world'].items():
            native = readback.get(kind, {})
            if set(native) != set(shapes):
                raise RuntimeError('Native world obstacle coverage differs from exported actual scene')
            for name, shape in shapes.items():
                pose = np.asarray(shape['pose'], dtype=float)
                rotation = Rotation.from_quat(np.r_[pose[4:], pose[3]]).as_matrix()
                expected = np.r_[-rotation.T @ pose[:3], pose[3], -pose[4:]]
                actual = native[name]
                error = pose_error(actual['inverse_pose_xyz_wxyz'], expected)
                if actual['enabled'] != 1 or error['position_m'] > 1e-5 or error['rotation_rad'] > 1e-4:
                    raise RuntimeError(f'Native obstacle {name} has stale/disabled geometry: {error}')
                max_position = max(max_position, error['position_m'])
                max_rotation = max(max_rotation, error['rotation_rad'])
                count += 1
        rows.append({'session_id': receipt['session_id'], 'obstacle_count': count,
                     'max_inverse_position_error_m': max_position,
                     'max_inverse_rotation_error_rad': max_rotation})
    return rows


# Compatibility exports: implementations live in shared runtime modules.
from .physical_observation import (verify_finger_spheres)



def export_pair(raw: Any, output: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from .curobo_scene import export_curobo_snapshot
    pair = tuple(export_curobo_snapshot(raw, i) for i in range(2))
    for i, data in enumerate(pair):
        save_json(output / f'{label}-arm-{i}-snapshot.json', data)
    return pair


def probe_pair(worker: Any, data: tuple[dict[str, Any], dict[str, Any]],
               profile: str, token: str, output: Path) -> list[dict[str, Any]]:
    rows = []
    for i, snapshot_data in enumerate(data):
        q = np.asarray(snapshot_data['q_start'], dtype=float)
        query = np.vstack([q, q + np.array([.03, -.02, .01, .02, -.01, .03, -.02])])
        fk = worker.fk(arm=i, profile=profile, snapshot_id=token, positions=query.tolist())
        save_json(output / f'{token}-arm-{i}-fk.json', fk)
        rows.append({'arm': i, 'tcp': check_fk(fk, snapshot_data, query),
                     'fingers': verify_finger_spheres(fk, snapshot_data, query),
                     'measured_lock_joints': snapshot_data['robot_cfg']['kinematics']['lock_joints']})
    return rows


def empty_gripper_hold(capture: ReachCapture, arms: list[Any], checker: Any,
                       command: float, steps: int) -> dict[str, Any]:
    """Paired stationary-arm ticks for a physical empty-gripper state probe."""
    if not np.isfinite(command) or not -1 <= command <= 1 or steps <= 0:
        raise ValueError('Finite normalized gripper command and positive steps required')
    targets = [arm.arm_qpos() for arm in arms]
    report = {'command': command, 'steps': steps,
              'initial_fingers_m': [arm.full_qpos()[7:].tolist() for arm in arms]}
    for _ in range(steps):
        checker.refresh()
        if checker.check(*(arm.arm_qpos() for arm in arms)):
            raise RuntimeError('Collision before empty gripper tick')
        action = {arm.key: np.r_[target, command].astype(np.float32)
                  for arm, target in zip(arms, targets)}
        _, _, terminated, truncated, _ = capture.step(action)
        checker.refresh()
        if checker.check(*(arm.arm_qpos() for arm in arms)):
            raise RuntimeError('Collision after empty gripper tick')
        if any(np.max(np.abs(arm.arm_qpos() - target)) > .03
               for arm, target in zip(arms, targets)):
            raise RuntimeError('Arm drift during empty gripper probe')
        if np.any(as_numpy(terminated)) or np.any(as_numpy(truncated)):
            raise RuntimeError('Environment ended during empty gripper probe')
    report['final_fingers_m'] = [arm.full_qpos()[7:].tolist() for arm in arms]
    return report


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import sapien
    import mani_skill.envs
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .combined_collision import validate_schedule
    from .curobo_sessions import CuroboSessions
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .dual_arm_execution import execute_dual_reach

    env, capture, worker = None, None, None
    try:
        env = gym.make('TwoRobotPickAppleBananaToBaskets-v1', obs_mode='none', reward_mode='none',
            control_mode='pd_joint_pos', sim_backend='cpu', render_backend='gpu',
            render_mode='rgb_array', max_episode_steps=200)
        env.reset(seed=args.seed)
        raw = env.unwrapped
        capture = ReachCapture(env, report)
        arms = [bind_arm(key, agent) for key, agent in raw.agent.agents_dict.items()]
        hold = {arm.key: np.r_[arm.arm_qpos(), 1.].astype(np.float32) for arm in arms}
        capture.frame()
        for _ in range(10):
            capture.step(hold)
        save_json(output / 'initial-state.json', capture.measurement())
        report.update(initial_state_sha256=hashlib.sha256((output / 'initial-state.json').read_bytes()).hexdigest(),
            control_timestep_s=float(raw.control_timestep), physics_backend=raw.backend.sim_backend)
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=arm.agent.robot.pose, multi_robot_id=i,
            joint_vel_limits=.9, joint_acc_limits=.9, debug=False, vis=False,
            visualize_target_grasp_pose=False, print_env_info=False) for i, arm in enumerate(arms)]
        checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers,
            targets=(raw.apple_out, raw.banana_out))
        report['external_scene_coverage'] = checker.coverage
        frozen = state_digest(capture)
        data = export_pair(raw, output, 'initial')
        worker = CuroboSessions(os.environ['CUDA_VISIBLE_DEVICES'], output / 'worker.stderr.log')
        report['initialization'] = worker.initialize_pair(data, seed=0, interpolation_dt=.02, warmup=True)
        report['sessions_after_init'] = worker.inspect_sessions()
        assert_frozen(capture, frozen)
        report['phases'] = []
        origins = [as_numpy(arm.agent.tcp.pose.p)[0].copy() for arm in arms]
        offsets = target_offsets()
        for phase, profile in enumerate(('approach', 'transport', 'approach')):
            if phase == 2:
                capture.phase = 'empty_gripper_close'
                report['empty_close'] = empty_gripper_hold(capture, arms, checker, -.2, 10)
                initial = np.asarray(report['empty_close']['initial_fingers_m'])
                closed = np.asarray(report['empty_close']['final_fingers_m'])
                if not np.all(initial - closed > .015):
                    raise RuntimeError('Empty gripper probe did not physically change both fingers')
                frozen = state_digest(capture)
                data = export_pair(raw, output, 'closed')
                report['closed_update'] = worker.update_pair(data, profile='approach', snapshot_id='closed')
                report['closed_world_readback'] = verify_world_readback(report['closed_update'], data)
                report['closed_fk'] = probe_pair(worker, data, 'approach', 'closed', output)
                assert_frozen(capture, frozen)
                capture.phase = 'empty_gripper_open'
                report['empty_open'] = empty_gripper_hold(capture, arms, checker, 1., 10)
                if any(np.max(np.abs(arm.full_qpos()[7:] - .04)) > .002 for arm in arms):
                    raise RuntimeError('Hands did not reopen before shared executor phase')
            token = f'phase-{phase}-{profile}'
            frozen = state_digest(capture)
            started = time.monotonic()
            data = export_pair(raw, output, token)
            row: dict[str, Any] = {'index': phase, 'profile': profile, 'snapshot_id': token,
                                   'control_step_before_plan': int(as_numpy(raw.elapsed_steps)[0])}
            report['phases'].append(row)
            row['update'] = worker.update_pair(data, profile=profile, snapshot_id=token)
            row['world_readback'] = verify_world_readback(row['update'], data)
            row['fk'] = probe_pair(worker, data, profile, token, output)
            if phase == 2:
                try:
                    worker.fk(arm=0, profile=profile, snapshot_id='closed', positions=[data[0]['q_start']])
                except (ValueError, RuntimeError) as exc:
                    row['stale_snapshot_rejected'] = {'status': 'passed', 'error': str(exc)}
                else:
                    raise RuntimeError('Client accepted a stale simulator snapshot')
                try:
                    worker._rpc.request('fk', session_id='left:approach', snapshot_id='closed',
                                        positions=[data[0]['q_start']])
                except RuntimeError as exc:
                    row['native_stale_snapshot_rejected'] = {'status': 'passed', 'error': str(exc)}
                else:
                    raise RuntimeError('Native worker accepted a stale simulator snapshot')
            z = (0., .025, .012)[phase]
            goals = [sapien.Pose(origin + offset + np.array([0., 0., z]),
                                  as_numpy(arm.agent.tcp.pose.q)[0])
                     for origin, offset, arm in zip(origins, offsets, arms)]
            row['goals_world_xyz_wxyz'] = [np.r_[goal.p, goal.q].tolist() for goal in goals]
            trajectories = []
            row['arms'] = []
            for i, (arm, goal) in enumerate(zip(arms, goals)):
                goal_base = arm.agent.robot.pose.sp.inv() * goal
                result = worker.plan_pose(arm=i, profile=profile, snapshot_id=token,
                    start=arm.arm_qpos().tolist(), goal_pose=np.r_[goal_base.p, goal_base.q].tolist(), max_attempts=4)
                save_json(output / f'{token}-arm-{i}-raw-result.json', result)
                velocity, acceleration = PROFILES[profile]
                trajectory, audit = trajectory_from_result(result, action_key=arm.key,
                    joint_names=arm.names, start=arm.arm_qpos(),
                    velocity_limit=velocity, acceleration_limit=acceleration)
                trajectories.append(trajectory)
                row['arms'].append({'arm': i, 'trajectory_audit': audit, 'duration_s': trajectory.duration_s,
                                    'native_status': result.get('status')})
            assert_frozen(capture, frozen)
            row.update(planning_environment_steps=0, planning_and_export_wall_s=time.monotonic() - started)
            schedule = build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)
            row['scheduled_steps'] = schedule.num_steps
            if schedule.num_steps + int(as_numpy(raw.elapsed_steps)[0]) + 15 > 200:
                raise RuntimeError('Phase exceeds remaining official 200-step budget')
            np.savez(output / f'{token}-schedule.npz',
                left_times=trajectories[0].times_s, right_times=trajectories[1].times_s,
                left_positions=trajectories[0].positions_rad, right_positions=trajectories[1].positions_rad,
                target_times=schedule.target_times_s, left_commands=schedule.left_commands,
                right_commands=schedule.right_commands)
            checker.refresh()
            row['collision_validation'] = validate_schedule(schedule, *(arm.arm_qpos() for arm in arms), checker)
            save_json(output / 'report.json', report)
            if row['collision_validation']['status'] != 'passed':
                raise RuntimeError('Paired external collision validation rejected phase')
            capture.phase = token
            row['execution'] = execute_dual_reach(capture, schedule, checker)
            row['tcp_goal_errors'] = [pose_error(as_numpy(arm.agent.tcp.pose.raw_pose)[0], np.r_[goal.p, goal.q])
                                      for arm, goal in zip(arms, goals)]
            if row['execution']['status'] != 'passed':
                raise RuntimeError(f'Existing shared executor rejected phase: {row["execution"]["status"]}')
            print(json.dumps({'phase': token, 'status': 'passed',
                              'elapsed_steps': int(as_numpy(raw.elapsed_steps)[0])}), flush=True)
            save_json(output / 'report.json', report)
        report['sessions_after_execution'] = worker.inspect_sessions()
        before_sessions = report['sessions_after_init']['sessions']
        after_sessions = report['sessions_after_execution']['sessions']
        if len(after_sessions) != 4 or len({s['planner_id'] for s in after_sessions.values()}) != 4:
            raise RuntimeError('Expected four distinct persistent native planners')
        for name, current in after_sessions.items():
            if (current['planner_id'] != before_sessions[name]['planner_id']
                    or current['init_count'] != 1 or current['warmup_count'] != 1
                    or current['rebuild_count'] != 0):
                raise RuntimeError('Planner was rebuilt or warmed again between phases')
            expected_updates = 3 if name.endswith(':approach') else 1
            expected_plans = 2 if name.endswith(':approach') else 1
            if current['update_count'] != expected_updates or current['plan_count'] != expected_plans:
                raise RuntimeError('Persistent session update/plan count does not match execution')
        report['persistent_session_verification'] = 'passed'
        report['planning_environment_steps'] = 0
        report['status'] = 'passed'
    finally:
        errors = []
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                errors.append(f'worker close: {exc}')
            report['worker_requests'] = worker.receipts
        if capture is not None:
            try:
                capture.save(output)
            except Exception as exc:
                errors.append(f'capture save: {exc}')
            finally:
                capture.unwrapped._after_simulation_step = capture.original_hook
        if env is not None:
            report['total_environment_steps'] = int(as_numpy(env.unwrapped.elapsed_steps)[0])
            try:
                env.close()
            except Exception as exc:
                errors.append(f'environment close: {exc}')
        if errors:
            report.update(status='failed', artifact_errors=errors)


def self_test() -> dict[str, Any]:
    from types import SimpleNamespace as NS
    from .curobo_sessions import self_test as session_self_test
    client = session_self_test()
    fake = NS(unwrapped=NS(elapsed_steps=np.array([1])), measurement=lambda: {'qpos': [[0.] * 9] * 2})
    digest = state_digest(fake)
    assert_frozen(fake, digest)
    fake.unwrapped.elapsed_steps += 1
    try:
        assert_frozen(fake, digest)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed simulator step was not rejected')
    # Synthetic nine-joint FK checks both physical fingers independently.
    urdf = ROOT / 'infrastructure/development/tig-dual-arm/mani_skill/assets/robots/panda/panda_v3.urdf'
    names = [f'panda_joint{i}' for i in range(1, 8)]
    locks = {'panda_finger_joint1': .01, 'panda_finger_joint2': .025}
    spheres = {name: [{'center': [0., 0., 0.], 'radius': .01}]
               for name in ('panda_leftfinger', 'panda_rightfinger')}
    data = {'joint_names': names, 'robot_cfg': {'kinematics':
            {'urdf_path': str(urdf), 'lock_joints': locks, 'collision_spheres': spheres}}}
    query = np.zeros((1, 7))
    result = {'link_spheres': {name: [[[*urdf_tcp_pose(urdf, dict(zip(names, query[0])) | locks,
                    tcp_link=name)[:3], .01]]] for name in spheres}}
    assert len(verify_finger_spheres(result, data, query)) == 2
    result['link_spheres']['panda_leftfinger'][0][0][0] += .001
    try:
        verify_finger_spheres(result, data, query)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Stale finger geometry was not rejected')
    return {'status': 'passed', 'client': client, 'frozen_state_and_finger_geometry': 'passed'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
        return 0
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        parser.error('Select one explicit GPU UUID')
    if any(os.environ.get(f'MANI_SKILL_L{i}') for i in (1, 2, 3)):
        parser.error('This fixture requires TIG L0')
    output = args.output_dir or ROOT / 'infrastructure/validation/dual-arm/curobo-sessions-20261116' / datetime.now(timezone.utc).strftime('dev-%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    source_files = list(HERE.glob('*.py')) + [ROOT / f'infrastructure/{name}' for name in
                    ('curobo_tig_worker.py', 'curobo_tig_session_worker.py')]
    archive = output / 'source'
    archive.mkdir()
    for source in source_files:
        shutil.copy2(source, archive / source.name)
    report: dict[str, Any] = {'status': 'running', 'seed': args.seed, 'gpu_uuid': gpu,
        'scope': 'persistent planner state; three unloaded phases and empty gripper cycle',
        'task': 'TwoRobotPickAppleBananaToBaskets-v1', 'official_task_completion_evaluated': False,
        'payload_supported': False, 'continuous_collision_detection': False, 'max_control_steps': 200,
        'profiles': {k: {'velocity_rad_s': v[0], 'acceleration_rad_s2': v[1]} for k, v in PROFILES.items()},
        'started_at': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'official_tig': git_identity(ROOT / 'The-Imitator-Game'),
        'official_curobo': git_identity(ROOT / 'infrastructure/backends/curobo')}
    save_json(output / 'report.json', report)
    started = time.monotonic()
    try:
        native_run(args, output, report)
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    report.update(wall_s=time.monotonic() - started, completed_at=datetime.now(timezone.utc).isoformat())
    save_json(output / 'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
