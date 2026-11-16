"""Execute an unloaded cuRobo/TIG paired reach through the existing executor.

This is an integration validation in the apple/banana scene, not a complete
pick/place task or a formal backend benchmark. Official 200-step limit stays
in force. Every attempt, including planning or physical failures, is retained.
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
import subprocess
import time
import traceback
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .curobo_backend import CuroboProcess, pose_error, trajectory_from_result, urdf_tcp_pose
from .trajectory_execution import as_numpy, bind_arm, plan_pair, snapshot
from .trajectory_schedule import build_hold_schedule
from .validate_execution import target_offsets

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent


# Compatibility exports: implementations live in shared runtime modules.
from .runtime_io import (save_json, git_identity)
from .physical_observation import (check_fk)







class ReachCapture:
    """Read-only 100Hz observer and 20Hz recorder around unchanged env.step."""
    def __init__(self, env: Any, report: dict[str, Any]) -> None:
        self.env, self.unwrapped, self.report = env, env.unwrapped, report
        self.ticks: list[dict[str, Any]] = []
        self.substeps: list[dict[str, Any]] = []
        self.frames: list[np.ndarray] = []
        self.phase = 'preparation'
        self.keys = list(self.unwrapped.agent.agents_dict)
        self.original_hook = self.unwrapped._after_simulation_step
        self.identities = {}
        self.allowed_self_pairs: dict[int, set[frozenset[str]]] = {}
        for i, agent in enumerate(self.unwrapped.agent.agents):
            srdf = ET.parse(Path(agent.urdf_path).with_suffix('.srdf')).getroot()
            self.allowed_self_pairs[i] = {frozenset((node.attrib['link1'], node.attrib['link2']))
                                         for node in srdf.findall('disable_collisions')}
            for link in agent.robot.get_links():
                for obj in link._objs:
                    self.identities[obj.entity] = {'kind': 'robot', 'robot': i, 'link': link.name}
        from .scene_collision import scene_entities
        for name, entity in scene_entities(self.unwrapped).items():
            self.identities[entity] = {'kind': 'actor', 'name': name}
        self.unwrapped._after_simulation_step = self.after_physics

    def measurement(self) -> dict[str, Any]:
        raw = self.unwrapped
        return {'qpos': [as_numpy(a.robot.get_qpos())[0].tolist() for a in raw.agent.agents],
                'qvel': [as_numpy(a.robot.get_qvel())[0].tolist() for a in raw.agent.agents],
                'tcp_poses': [as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in raw.agent.agents],
                'actor_poses': {name: as_numpy(actor.pose.raw_pose)[0].tolist()
                                for name, actor in raw.scene.actors.items()}}

    def after_physics(self) -> None:
        from .validate_held_path_conflict import physical_contact_points
        self.original_hook()
        contacts = []
        for contact in self.unwrapped.scene.px.get_contacts():
            sides = [self.identities.get(body.entity, {'kind': 'unknown', 'name': body.entity.name})
                     for body in contact.bodies]
            if not any(side['kind'] == 'robot' for side in sides):
                continue
            points = [{'position_m': np.asarray(p.position).tolist(),
                       'normal': np.asarray(p.normal).tolist(),
                       'impulse_Ns': np.asarray(p.impulse).tolist(),
                       'separation_m': float(p.separation)} for p in contact.points]
            mount = (len(sides) == 2 and any(s.get('link') == 'panda_link0' for s in sides)
                     and any(s.get('name') == 'table' for s in sides))
            same_robot = (len(sides) == 2 and all(s['kind'] == 'robot' for s in sides)
                          and sides[0]['robot'] == sides[1]['robot'])
            srdf_allowed = (same_robot and frozenset((sides[0]['link'], sides[1]['link']))
                            in self.allowed_self_pairs[sides[0]['robot']])
            contacts.append({'bodies': sides, 'points': points, 'physical': physical_contact_points(points),
                             'fixed_base_mount': mount, 'srdf_allowed_self_pair': srdf_allowed})
        self.substeps.append({'substep': len(self.substeps) + 1, 'phase': self.phase,
            'pending_control_step': int(as_numpy(self.unwrapped.elapsed_steps)[0]) + 1,
            'contacts': contacts, **self.measurement()})

    def frame(self) -> None:
        frame = as_numpy(self.env.render())
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected HWC RGB uint8 frame')
        self.frames.append(frame.copy())

    def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
        if int(as_numpy(self.unwrapped.elapsed_steps)[0]) >= 200:
            raise RuntimeError('Official 200 control-step budget exhausted')
        first = len(self.substeps)
        pending: dict[str, Any] = {'phase': self.phase,
            'elapsed_before': int(as_numpy(self.unwrapped.elapsed_steps)[0]),
            'actions': {key: np.asarray(action[key]).tolist() for key in self.keys}}
        try:
            outcome = self.env.step(action)
        except Exception as exc:
            pending.update(error=f'{type(exc).__name__}: {exc}',
                           elapsed_after=int(as_numpy(self.unwrapped.elapsed_steps)[0]), **self.measurement())
            self.ticks.append(pending)
            raise
        pending.update(environment_step=int(as_numpy(self.unwrapped.elapsed_steps)[0]),
                       **self.measurement())
        self.ticks.append(pending)
        self.frame()
        forbidden = [contact for row in self.substeps[first:] for contact in row['contacts']
                     if contact['physical'] and not contact['fixed_base_mount']
                     and not contact['srdf_allowed_self_pair']]
        if forbidden:
            raise RuntimeError('Actual robot/world or interarm contact detected at physics frequency')
        return outcome

    def save(self, output: Path) -> None:
        import h5py
        self.unwrapped._after_simulation_step = self.original_hook
        save_json(output / 'control-trace.json', self.ticks)
        save_json(output / 'physics-trace.json', self.substeps)
        with h5py.File(output / 'trajectory.h5', 'x') as handle:
            handle.attrs['scope'] = 'unloaded paired reach; not TIG task completion'
            handle.attrs['action_keys'] = json.dumps(self.keys)
            for label, rows in [('control', self.ticks), ('physics', self.substeps)]:
                group = handle.create_group(label)
                for field in ('qpos', 'qvel', 'tcp_poses'):
                    group.create_dataset(field, data=np.asarray([row[field] for row in rows]), compression='gzip')
            handle.create_dataset('actions', data=np.asarray([[row['actions'][key] for key in self.keys]
                                                              for row in self.ticks]), compression='gzip')
        self.report.update(control_ticks=len(self.ticks), physical_substeps=len(self.substeps),
                           physical_timestep_s=float(self.unwrapped.sim_timestep))
        if self.frames:
            import imageio.v2 as imageio
            imageio.mimsave(output / 'execution.mp4', [self.frames[0]] * 10 + self.frames + [self.frames[-1]] * 20,
                           fps=20, macro_block_size=1)
            imageio.imwrite(output / 'preview.png', self.frames[-1])


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import sapien
    import mani_skill.envs
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .combined_collision import validate_schedule
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .dual_arm_execution import execute_dual_reach
    from .curobo_scene import export_curobo_snapshot

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
        initial = capture.measurement()
        save_json(output / 'initial-state.json', initial)
        report['initial_state_sha256'] = hashlib.sha256((output / 'initial-state.json').read_bytes()).hexdigest()
        report.update(control_timestep_s=float(raw.control_timestep),
                      physics_backend=raw.backend.sim_backend, render_backend=raw.backend.render_backend)
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=arm.agent.robot.pose, multi_robot_id=i,
            joint_vel_limits=.9, joint_acc_limits=.9, debug=False, vis=False,
            visualize_target_grasp_pose=False, print_env_info=False) for i, arm in enumerate(arms)]
        goals = [sapien.Pose(as_numpy(arm.agent.tcp.pose.p)[0] + offset,
                             as_numpy(arm.agent.tcp.pose.q)[0]) for arm, offset in zip(arms, target_offsets())]
        report['goals_world_xyz_wxyz'] = [np.r_[goal.p, goal.q].tolist() for goal in goals]
        before = snapshot(raw, tuple(arms))
        started = time.monotonic()
        if args.backend == 'mplib':
            schedule = plan_pair(*solvers, *goals)
        else:
            worker = CuroboProcess(os.environ['CUDA_VISIBLE_DEVICES'], output / 'worker.stderr.log')
            trajectories = []
            report['arm_planning'] = []
            for i, (arm, goal) in enumerate(zip(arms, goals)):
                data = export_curobo_snapshot(raw, i)
                save_json(output / f'arm-{i}-snapshot.json', data)
                row: dict[str, Any] = {'arm': i, 'action_key': arm.key, 'coverage': data['coverage']}
                report['arm_planning'].append(row)
                save_json(output / 'report.json', report)
                row['configure'] = worker.request('configure', robot_cfg=data['robot_cfg'],
                    world=data['world'], seed=0, interpolation_dt=.02, warmup=True)
                if tuple(row['configure']['joint_names']) != arm.names:
                    raise ValueError('Configured cuRobo joint order differs from measured FK/start order')
                q = arm.arm_qpos()
                query = np.vstack([q, q + np.array([.03, -.02, .01, .02, -.01, .03, -.02])])
                fk = worker.request('fk', positions=query.tolist())
                row['fk_checks'] = check_fk(fk, data, query)
                goal_base = arm.agent.robot.pose.sp.inv() * goal
                result = worker.request('plan_pose', start=q.tolist(),
                    goal_pose=np.r_[goal_base.p, goal_base.q].tolist(), max_attempts=4)
                save_json(output / f'arm-{i}-raw-result.json', result)
                trajectory, row['trajectory_audit'] = trajectory_from_result(result,
                    action_key=arm.key, joint_names=arm.names, start=q)
                trajectories.append(trajectory)
                row['duration_s'] = trajectory.duration_s
                save_json(output / 'report.json', report)
            schedule = build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)
        report['planning_wall_s'] = time.monotonic() - started
        if not all(np.array_equal(a, b) for a, b in zip(before, snapshot(raw, tuple(arms)))):
            raise RuntimeError('Planning changed simulator state')
        report.update(planning_environment_steps=0, scheduled_steps=schedule.num_steps,
                      source_durations_s=[t.duration_s for t in schedule.trajectories])
        if schedule.num_steps + 10 + 15 > 200:
            raise RuntimeError('Planned phase exceeds remaining official budget')
        np.savez(output / 'scheduled-trajectory.npz',
            left_times=schedule.trajectories[0].times_s, right_times=schedule.trajectories[1].times_s,
            left_positions=schedule.trajectories[0].positions_rad,
            right_positions=schedule.trajectories[1].positions_rad,
            target_times=schedule.target_times_s, left_commands=schedule.left_commands,
            right_commands=schedule.right_commands)
        checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers,
            targets=(raw.apple_out, raw.banana_out))
        report['external_scene_coverage'] = checker.coverage
        report['collision_validation'] = validate_schedule(schedule, *(arm.arm_qpos() for arm in arms), checker)
        save_json(output / 'report.json', report)
        if report['collision_validation']['status'] != 'passed':
            report['status'] = 'rejected'
            return
        capture.phase = 'paired_reach'
        report['execution'] = execute_dual_reach(capture, schedule, checker)
        report['tcp_goal_errors'] = [pose_error(as_numpy(arm.agent.tcp.pose.raw_pose)[0], np.r_[goal.p, goal.q])
                                     for arm, goal in zip(arms, goals)]
        report['status'] = 'passed' if report['execution']['status'] == 'passed' else report['execution']['status']
    finally:
        artifact_errors = []
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                artifact_errors.append(f'worker close: {type(exc).__name__}: {exc}')
            report['worker_requests'] = worker.receipts
        if capture is not None:
            try:
                capture.save(output)
            except Exception as exc:
                artifact_errors.append(f'capture save: {type(exc).__name__}: {exc}')
            finally:
                capture.unwrapped._after_simulation_step = capture.original_hook
        if env is not None:
            report['total_environment_steps'] = int(as_numpy(env.unwrapped.elapsed_steps)[0])
            try:
                env.close()
            except Exception as exc:
                artifact_errors.append(f'environment close: {type(exc).__name__}: {exc}')
        if artifact_errors:
            report.update(status='failed', artifact_errors=artifact_errors)


def self_test() -> dict[str, Any]:
    from .curobo_backend import self_test as backend_self_test
    checked = backend_self_test()
    assert checked['status'] == 'passed'
    assert len(target_offsets()) == 2
    assert all(np.isfinite(offset).all() for offset in target_offsets())
    assert (ROOT / 'infrastructure').is_dir()
    assert pose_error([0, 0, 0, 1, 0, 0, 0], [.1034, 0, 0, 1, 0, 0, 0])['position_m'] > .1
    return {'status': 'passed', 'backend_contract': checked,
            'scope': 'CPU contracts only; no physical task completion'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--backend', choices=('curobo', 'mplib'), default='curobo')
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.startswith('GPU-') or ',' in gpu:
        parser.error('Select one explicit GPU UUID via CUDA_VISIBLE_DEVICES')
    if any(os.environ.get(f'MANI_SKILL_L{i}') for i in (1, 2, 3)):
        parser.error('This integration fixture requires L0')
    output = args.output_dir or ROOT / 'infrastructure/validation/dual-arm/curobo-integration-20261116' / datetime.now(timezone.utc).strftime('reach-%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_files = list(HERE.glob('*.py')) + [ROOT / 'infrastructure/curobo_tig_worker.py']
    archive = output / 'source'
    archive.mkdir()
    for source in source_files:
        if source.exists():
            shutil.copy2(source, archive / source.name)
    report: dict[str, Any] = {'status': 'running', 'backend': args.backend, 'seed': args.seed,
        'task': 'TwoRobotPickAppleBananaToBaskets-v1', 'scope': 'unloaded paired reach integration',
        'official_task_completion_evaluated': False, 'payload_supported': False,
        'joint_dual_arm_optimization': False, 'continuous_collision_detection': False,
        'gpu_uuid': gpu, 'started_at': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.exists()},
        'curobo_source': git_identity(ROOT / 'infrastructure/backends/curobo'),
        'official_tig': git_identity(ROOT / 'The-Imitator-Game')}
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
