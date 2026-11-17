"""Optional MPLib synchronous solution using the original caller-owned env."""
from __future__ import annotations

import time

import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from .dual_arm_execution import (plan_dual_pose, execute_dual_reach, close_dual_and_confirm,
    execute_dual_held_move, release_dual_and_confirm, verify_dual_placement)
from .trajectory_execution import as_numpy, bindings
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .phase_kinematics import (SOLVER_LIMIT_FRACTION, attach_topp_capture, compact_phase_limits,
    compact_set_phase_limits, compact_retime_pair, compact_route_positions)
from .pill_support_collision import PillDualPayloadCollisionChecker
from .runtime_policy import DEFAULT_POLICY, RuntimePolicy, load_policy as load_policy
from .pill_task import TASK_ID, PillTask
from .observation import ObservedEnvironment
from .physical_observation import measured_source_departure
from .task_outcome import observe_dual_task_outcome

LIMITS = {"joint_vel_limit_rad_s": 1.5, "joint_acc_limit_rad_s2": 2.6,
    "approach_joint_vel_limit_rad_s": 1., "approach_joint_acc_limit_rad_s2": 1.3,
    "solver_limit_fraction": SOLVER_LIMIT_FRACTION}


class StageFailure(RuntimeError):
    def __init__(self, category, reason):
        self.category, self.reason = category, reason
        super().__init__(reason)


def phase_sequence(policy):
    phases = ['pregrasp', 'descend', 'close_and_confirm', 'lift_and_hold', 'carry_lower', 'release']
    if policy.require_retreat:
        phases.append('retreat')
    if policy.require_final_stability:
        phases.append('verify_placement')
    return phases


def future_minimum(policy, phase):
    phases = phase_sequence(policy)
    minimum = {'close_and_confirm': policy.grasp_stable_steps,
        'release': policy.release_stable_steps, 'verify_placement': policy.final_stable_steps}
    return sum(minimum.get(name, policy.reach_stable_steps)
        for name in phases[phases.index(phase)+1:])


def movement_budget(policy, before, scheduled, phase):
    """Actual motion plus selected future windows; no fixed spare-tick veto."""
    available = policy.last_completion_step - before - scheduled - future_minimum(policy, phase)
    required = max(0, policy.reach_stable_steps - scheduled)
    if available < required:
        raise StageFailure('budget', f'{phase}: scheduled motion and selected future windows exceed budget')
    return min(policy.max_settle_steps, available)


def _compact_stage(name, result):
    fields = ('status', 'reason', 'environment_steps', 'settling_steps', 'stable_steps_observed',
        'stable_steps_required', 'wall_seconds', 'object_index', 'error')
    row = {key: result[key] for key in fields if key in result}
    row['name'] = name
    validation = result.get('validation')
    if validation is not None:
        row['collision_validation'] = {key: validation[key] for key in ('status', 'reason', 'collision_count')
            if key in validation}
    if result.get('trace') and result['status'] != 'passed':
        last = result['trace'][-1]
        row['last_event'] = {key: last[key] for key in ('error', 'collisions', 'attachment_geometry') if key in last}
    return row


def solve(env, seed=None, debug=False, vis=False, *, policy=DEFAULT_POLICY):
    """Return original two-arm step tuples; write independent result facts.

    The caller creates/closes the environment and owns RecordEpisode. Planning
    never advances it; the common executor submits both arm actions per tick.
    This is coordinated execution of separate plans, not joint 14-DoF search.
    """
    if not isinstance(policy, RuntimePolicy):
        raise ValueError('Expected RuntimePolicy')
    raw = env.unwrapped
    if (raw.spec.id != TASK_ID or raw.num_envs != 1
            or any(agent.control_mode != 'pd_joint_pos' for agent in raw.agent.agents)):
        raise ValueError(f'Sync supports one {TASK_ID} with pd_joint_pos')
    if raw.backend.sim_backend not in ('cpu', 'physx_cpu'):
        raise ValueError('This sync implementation currently requires CPU physics')
    if not np.isclose(raw.control_timestep, .05) or not np.isclose(raw.sim_timestep, .01):
        raise ValueError('Expected 20Hz control and 100Hz physics')
    if policy.max_episode_steps != 200 or policy.last_completion_step != 199:
        raise ValueError('The official200-step horizon is fixed')
    env.reset(seed=seed)
    report = {'schema': 'dual-panda-sync-v1', 'seed': seed, 'backend': 'mplib',
        'control_freq': 20, 'physics_freq': 100, 'policy': policy.to_dict(),
        'motion_limits': dict(LIMITS), 'stages': [], 'planning': [],
        'execution': {'status': 'running'}, 'quality': {'status': 'not_evaluated'},
        'recording': {'status': 'owned_by_RecordEpisode'}}
    raw.dual_sync_result = report
    task = PillTask(raw)
    checker = None
    capture = ObservedEnvironment(env, task, lambda: checker, policy)
    started = time.monotonic()

    def elapsed():
        return int(as_numpy(raw.elapsed_steps).item())

    def finish(name, result):
        report['stages'].append(_compact_stage(name, result))
        if debug:
            print(f'sync {name}: {result["status"]} at step{elapsed()}', flush=True)
        if result['status'] != 'passed':
            physical = capture.events or result.get('reason') in (
                'grasp_lost', 'object_dropped', 'position_slip', 'rotation_slip', 'measured_collision')
            quality_timeout = ((name == 'retreat' and result['status'] == 'reach_timeout')
                or (name == 'verify_placement' and result['status'] == 'placement_timeout'))
            category = ('physics' if physical else 'planning' if result['status'] == 'rejected'
                else 'quality' if quality_timeout else 'execution')
            raise StageFailure(category, f'{name}: {result.get("reason", result["status"])}')

    try:
        capture.install()
        arms = bindings(raw, tuple(raw.agent.agents_dict))
        opened = {arm.key: np.r_[arm.arm_qpos(), 1.] for arm in arms}
        for _ in range(10):
            capture.step(opened)
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=agent.robot.pose,
            multi_robot_id=i, joint_vel_limits=.9, joint_acc_limits=.9,
            debug=debug, vis=vis, visualize_target_grasp_pose=False, print_env_info=False)
            for i, agent in enumerate(raw.agent.agents)]
        for solver in solvers:
            attach_topp_capture(solver)
        checker = PillDualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=task.targets)
        checker.set_measured_attachment_prediction(True)

        def plan(name, poses, closed):
            capture.phase = name
            before = elapsed()
            limits = compact_phase_limits(LIMITS, name)
            compact_set_phase_limits(solvers, limits)
            call = time.monotonic()
            try:
                source = plan_dual_pose(*solvers, poses, gripper_commands=(-1, -1) if closed else (1, 1), policy=policy)
                results, _ = compact_retime_pair(tuple(s.last_native_result for s in solvers),
                    tuple(s.last_native_parameterization for s in solvers),
                    velocity_limit=limits['nominal_velocity_limit_rad_s'],
                    acceleration_limit=limits['nominal_acceleration_limit_rad_s2'])
                trajectories = tuple(ArmTrajectory(t.action_key, t.joint_names, r['time'], r['position'], -1. if closed else 1.)
                    for t, r in zip(source.trajectories, results))
                schedule = build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)
                settle = movement_budget(policy, before, schedule.num_steps, name)
            except StageFailure:
                raise
            except Exception as exc:
                raise StageFailure('planning', f'{name}: {type(exc).__name__}: {exc}') from exc
            finally:
                report['planning'].append({'name': name, 'wall_seconds': time.monotonic()-call,
                    'endpoint_checks': [{key: audit[key] for key in (
                        'status', 'corrections', 'final_error', 'position_tolerance_m',
                        'rotation_tolerance_rad', 'error') if key in audit}
                        for audit in (getattr(s, 'last_endpoint_audit', {}) for s in solvers)]})
            if elapsed() != before:
                raise RuntimeError('Planning advanced the environment')
            return schedule, settle

        pairs = task.grasp_poses()
        for index, name, contact in ((0, 'pregrasp', False), (1, 'descend', True)):
            schedule, settle = plan(name, (pairs[0][index], pairs[1][index]), False)
            finish(name, execute_dual_reach(capture, schedule, checker,
                allow_target_contacts=(contact, contact), settle_max_steps=settle, policy=policy))
        capture.phase = 'close_and_confirm'
        available = policy.last_completion_step - elapsed() - future_minimum(policy, capture.phase)
        if available < policy.grasp_stable_steps:
            raise StageFailure('budget', 'Insufficient budget for selected grasp window')
        finish(capture.phase, close_dual_and_confirm(capture, checker, task.targets,
            max_steps=available, policy=policy))
        checker.attach()
        capture.references = np.asarray(checker.attachments).copy()
        release = task.release_poses()
        route = compact_route_positions([as_numpy(a.tcp.pose.p)[0] for a in raw.agent.agents],
            [p.p for p in release], route='merge-lift')
        lift = tuple(sapien.Pose(p=xyz, q=as_numpy(a.tcp.pose.q)[0]) for xyz, a in zip(route['lift'], raw.agent.agents))
        schedule, settle = plan('lift_and_hold', lift, True)
        finish('lift_and_hold', execute_dual_held_move(capture, schedule, checker, task.targets,
            min_height_gain_m=((policy.min_lift_m,)*2 if policy.min_lift_m > 0 else (None, None)), hold_steps=0, settle_max_steps=settle, policy=policy))
        if not all(measured_source_departure(checker)) or not all(capture.physical_released):
            raise StageFailure('physics', 'Actual source support departure not confirmed')
        schedule, settle = plan('carry_lower', release, True)
        finish('carry_lower', execute_dual_held_move(capture, schedule, checker, task.targets,
            hold_steps=0, settle_max_steps=settle, policy=policy))
        if (np.any(capture.max_slip_m > policy.position_slip_m)
                or np.any(capture.max_slip_rad > policy.rotation_slip_rad)):
            raise StageFailure('physics', 'Sampled held object slip exceeded selected settings')
        checker.detach()
        capture.phase = 'release'
        available = policy.last_completion_step - elapsed() - future_minimum(policy, capture.phase)
        if available < policy.release_stable_steps:
            raise StageFailure('budget', 'Insufficient budget for selected release window')
        finish('release', release_dual_and_confirm(capture, checker, task.targets, max_steps=available, policy=policy))
        if policy.require_retreat:
            retreat_start = np.asarray([as_numpy(a.tcp.pose.p)[0].copy() for a in raw.agent.agents])
            poses = tuple(sapien.Pose(p=p+[0., 0., policy.retreat_command_m], q=as_numpy(a.tcp.pose.q)[0])
                for p, a in zip(retreat_start, raw.agent.agents))
            schedule, settle = plan('retreat', poses, False)
            finish('retreat', execute_dual_reach(capture, schedule, checker, settle_max_steps=settle, policy=policy))
            gain = np.asarray([as_numpy(a.tcp.pose.p)[0] for a in raw.agent.agents])-retreat_start
            report['quality']['retreat_displacement_m'] = gain.tolist()
            if np.any(gain[:, 2] < policy.minimum_retreat_m):
                raise StageFailure('quality', 'Measured retreat below selected distance')
        if policy.require_final_stability:
            capture.phase = 'verify_placement'
            remaining = policy.last_completion_step - elapsed()
            if remaining < policy.final_stable_steps:
                raise StageFailure('budget', 'Selected final stability window does not fit')
            finish(capture.phase, verify_dual_placement(capture, checker, task.targets,
                official_predicate_keys=task.official_predicate_keys, region_predicates=task.region_predicates(),
                max_steps=remaining, stable_steps=policy.final_stable_steps, policy=policy))
        else:
            capture.phase = 'observe_task_outcome'
            observed = observe_dual_task_outcome(capture, checker, task.targets,
                official_predicate_keys=task.official_predicate_keys, region_predicates=task.region_predicates(),
                max_steps=max(0, policy.last_completion_step-elapsed()), policy=policy)
            finish(capture.phase, observed)
        report['execution'] = {'status': 'passed'}
        report['quality'].update(status='passed' if policy.require_retreat or policy.require_final_stability else 'not_requested',
            retreat_evaluated=policy.require_retreat, final_stability_evaluated=policy.require_final_stability)
    except StageFailure as exc:
        report[ 'quality' if exc.category == 'quality' else 'execution'].update(status='failed',
            category=exc.category, reason=exc.reason)
    except Exception as exc:
        report['execution'] = {'status': 'failed', 'category': 'runtime',
            'reason': f'{type(exc).__name__}: {exc}'}
    finally:
        capture.restore()
        success = bool(as_numpy(raw.evaluate()['success']).item())
        report['official'] = {'success': success, 'elapsed_steps': elapsed(),
            'within_horizon': elapsed() < policy.max_episode_steps,
            'terminated': bool(as_numpy(capture.last_step[2]).item()) if capture.last_step else False,
            'truncated': bool(as_numpy(capture.last_step[3]).item()) if capture.last_step else False}
        report['physics'] = capture.summary()
        report['wall_seconds'] = time.monotonic()-started
        if report['execution']['status'] == 'running':
            report['execution']['status'] = 'passed' if report['quality']['status'] == 'failed' else 'incomplete'
    if capture.last_step is None or report['execution']['status'] != 'passed':
        return -1, -1
    return capture.last_step, capture.last_step
