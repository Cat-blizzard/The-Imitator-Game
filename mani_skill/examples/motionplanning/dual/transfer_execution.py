"""Small role-aware handoff state machine over the existing paired scheduler.

One unique payload, physical left/right order, absolute-radian actions[2,8],
world transforms[4,4], force vectors[2,2,3],20Hz control. Observation calls the
original Panda.is_grasping default predicate: both fingers>=0.5N and angle<=85deg.
No environment construction, reset, state assignment, physical weld or masks.
The native caller supplies its recorder/100Hz hook and frozen episode budget.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .combined_collision import validate_schedule
from .dual_arm_execution import _canonical, _bool
from .grasp_execution import _actions_valid
from .lift_execution import _rotation_distance
from .trajectory_execution import as_numpy
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .physical_observation import continue_success_only

GIVER_HOLD = 'giver_hold'
RECEIVER_CLOSE = 'receiver_close'
RECEIVER_CONFIRMED = 'receiver_confirmed'
GIVER_RELEASE = 'giver_release'
GIVER_RETREAT = 'giver_retreat'
RECEIVER_HOLD = 'receiver_hold'
RECEIVER_FAILED = 'receiver_failed'
SAFE_HOLD = 'safe_giver_hold'
SAFE_COMPLETE = 'safe_refusal_complete'
COMPLETE = 'complete'
ABORTED = 'aborted'
TERMINAL = frozenset((COMPLETE, SAFE_COMPLETE, ABORTED))


def _require(value: Any, message: str) -> None:
    if not bool(value): raise ValueError(message)


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    _require(result.shape == shape and np.isfinite(result).all(), 'Invalid '+label)
    return result


def _matrix(value: Any) -> np.ndarray:
    result = _finite(value, (4, 4), 'rigid transform')
    _require(np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
             and np.allclose(result[:3, :3].T@result[:3, :3], np.eye(3), rtol=0, atol=2e-5)
             and abs(np.linalg.det(result[:3, :3])-1) <= 2e-5, 'Invalid homogeneous SO3 transform')
    return result


@dataclass(frozen=True)
class TransferObservation:
    """One measured frame. qpos is canonical7arm+2finger for each physical arm."""
    environment_step: int
    qpos_rad: np.ndarray
    qvel_rad_s: np.ndarray
    tcp_T_world: np.ndarray
    object_T_world: np.ndarray
    grasp_flags: tuple[bool, bool]
    finger_forces_N: np.ndarray
    finger_open_upper_m: np.ndarray

    def __post_init__(self) -> None:
        _require(type(self.environment_step) is int and self.environment_step >= 0, 'Invalid measured control step')
        _require(len(self.grasp_flags) == 2 and all(type(v) is bool for v in self.grasp_flags), 'Native grasp flags must be two bools')
        for name, shape in (('qpos_rad', (2, 9)), ('qvel_rad_s', (2, 9)), ('tcp_T_world', (2, 4, 4)),
                            ('finger_forces_N', (2, 2, 3)), ('finger_open_upper_m', (2, 2))):
            value = _finite(getattr(self, name), shape, name); value.setflags(write=False)
            object.__setattr__(self, name, value)
        for transform in self.tcp_T_world: _matrix(transform)
        value = _matrix(self.object_T_world); value.setflags(write=False)
        object.__setattr__(self, 'object_T_world', value)

    @property
    def fingers_open(self) -> tuple[bool, bool]:
        return tuple(bool(np.all(np.abs(self.qpos_rad[i, 7:]-self.finger_open_upper_m[i]) <= .002)) for i in range(2))

    def json(self) -> dict[str, Any]:
        return {'environment_step': self.environment_step, 'qpos_rad': self.qpos_rad.tolist(),
                'qvel_rad_s': self.qvel_rad_s.tolist(), 'tcp_T_world': self.tcp_T_world.tolist(),
                'object_T_world': self.object_T_world.tolist(), 'grasp_flags': list(self.grasp_flags),
                'finger_forces_N': self.finger_forces_N.tolist(), 'fingers_open': list(self.fingers_open)}


def measure_transfer(env: Any, target_actor: Any) -> TransferObservation:
    """Read actual Panda predicates/force vectors; never infer grasp from grip command."""
    raw = env.unwrapped; arms = _canonical(env)
    _require(abs(float(raw.control_timestep)-.05) < 1e-12 and abs(float(raw.sim_timestep)-.01) < 1e-12,
             'Handoff retains20Hz control and100Hz physics')
    qpos, qvel, tcps, forces, upper, grasp = [], [], [], [], [], []
    for arm in arms:
        fingers = [i for i in range(9) if i not in arm.indices]
        order = [*arm.indices, *fingers]
        qpos.append(as_numpy(arm.agent.robot.get_qpos())[0, order])
        qvel.append(as_numpy(arm.agent.robot.get_qvel())[0, order])
        tcps.append(as_numpy(arm.agent.tcp.pose.to_transformation_matrix())[0])
        forces.append([as_numpy(raw.scene.get_pairwise_contact_forces(link, target_actor))[0]
                       for link in (arm.agent.finger1_link, arm.agent.finger2_link)])
        upper.append(as_numpy(arm.agent.robot.get_qlimits())[0, fingers, 1])
        grasp.append(_bool(arm.agent.is_grasping(target_actor)))
    return TransferObservation(int(as_numpy(raw.elapsed_steps)[0]), np.asarray(qpos), np.asarray(qvel),
        np.asarray(tcps), as_numpy(target_actor.pose.to_transformation_matrix())[0], tuple(grasp), np.asarray(forces), np.asarray(upper))


class TransferStateMachine:
    """Confirmation before release is enforced at action generation and pre-step.

    The receiver reference is captured at its fifth confirmed tick, before any
    giver-open action. Failure before release retains the giver CLOSED. Failure
    after release freezes the last issued gripper commands; it never fabricates
    a regrasp. A receiver-timeout negative may explicitly request20safe hold ticks.
    """

    def __init__(self, giver_index: int, giver_reference: Any, *, receiver_max_steps: int = 20,
                 release_max_steps: int = 30, minimum_object_z_m: float | None = None) -> None:
        _require(type(giver_index) is int and giver_index in (0, 1), 'Invalid giver physical index')
        _require(type(receiver_max_steps) is int and receiver_max_steps >= 5 and type(release_max_steps) is int
                 and release_max_steps >= 3, 'Confirmation budgets too short')
        if minimum_object_z_m is not None: _require(np.isfinite(minimum_object_z_m), 'Invalid minimum height')
        self.giver_index, self.receiver_index = giver_index, 1-giver_index
        self.references = {giver_index: _matrix(giver_reference)}
        self.receiver_max_steps, self.release_max_steps = receiver_max_steps, release_max_steps
        self.minimum_object_z_m = minimum_object_z_m
        self.state, self.reason = GIVER_HOLD, None
        self.trace: list[dict[str, Any]] = []; self.events: list[dict[str, Any]] = []
        self.confirmation: list[dict[str, Any]] = []; self.release_confirmation: list[dict[str, Any]] = []
        self.previous: TransferObservation | None = None
        self.phase_ticks, self.hold_ticks = 0, 0
        self._last_grips = self._grips(-1, 1)
        self._hover_reference: TransferObservation | None = None
        self._retreat_start: TransferObservation | None = None

    def _grips(self, giver: int, receiver: int) -> tuple[int, int]:
        values = [0, 0]; values[self.giver_index] = giver; values[self.receiver_index] = receiver
        return tuple(values)

    def initialize(self, observation: TransferObservation) -> None:
        _require(self.previous is None and observation.grasp_flags[self.giver_index]
                 and not observation.grasp_flags[self.receiver_index], 'Initialize from one real giver grasp only')
        self.previous = observation; self._guard_object(observation, self.giver_index)
        _require(self.state != ABORTED, 'Initial giver attachment invalid')

    def command_grips(self) -> tuple[int, int]:
        if self.state in (ABORTED, COMPLETE, SAFE_COMPLETE): return self._last_grips
        if self.state == GIVER_HOLD: result = self._grips(-1, 1)
        elif self.state in (RECEIVER_CLOSE, RECEIVER_CONFIRMED, RECEIVER_FAILED, SAFE_HOLD): result = self._grips(-1, -1)
        else:
            _require(self.receiver_index in self.references, 'Giver OPEN forbidden before actual receiver reference capture')
            result = self._grips(1, -1)
        self._last_grips = result
        return result

    def abort(self, reason: str) -> None:
        if self.state == ABORTED: return
        self.events.append({'event': 'aborted', 'reason': reason,
                            'environment_step': None if self.previous is None else self.previous.environment_step})
        self.state, self.reason = ABORTED, reason

    def begin_receiver_close(self) -> None:
        _require(self.state == GIVER_HOLD and self.previous is not None, 'Invalid receiver-close entry')
        _require(self.previous.grasp_flags[self.giver_index], 'Giver lost actual grasp')
        self.state = RECEIVER_CLOSE; self.phase_ticks = 0; self.confirmation = []

    def record_receiver_reference(self, reference: Any) -> None:
        _require(self.state == RECEIVER_CONFIRMED and self.receiver_index not in self.references
                 and len(self.confirmation) == 5 and self.previous is not None, 'Receiver reference capture outside confirmed grasp')
        reference = _matrix(reference)
        expected = np.linalg.solve(self.previous.tcp_T_world[self.receiver_index], self.previous.object_T_world)
        _require(np.allclose(reference, expected, rtol=0, atol=3e-6), 'Receiver relation is not from confirmation-time measurement')
        self.references[self.receiver_index] = reference

    def begin_giver_release(self) -> None:
        _require(self.state == RECEIVER_CONFIRMED and self.receiver_index in self.references and self.previous is not None
                 and all(self.previous.grasp_flags), 'Giver OPEN forbidden before fresh five-tick receiver confirmation')
        self.state = GIVER_RELEASE; self.phase_ticks = 0; self.release_confirmation = []
        self.events.append({'event': 'giver_release_authorized', 'environment_step': self.previous.environment_step})

    def begin_safe_giver_hold(self) -> None:
        _require(self.state == RECEIVER_FAILED and self.previous is not None and self.previous.grasp_flags[self.giver_index],
                 'Safe refusal requires the actual giver still holding')
        self.state = SAFE_HOLD; self.hold_ticks = 0; self._hover_reference = self.previous

    def mark_retreat_complete(self) -> None:
        _require(self.state == GIVER_RETREAT and self.previous is not None and self._retreat_start is not None,
                 'Retreat completion outside released-giver phase')
        gain = self.previous.tcp_T_world[self.giver_index, 2, 3]-self._retreat_start.tcp_T_world[self.giver_index, 2, 3]
        _require(gain >= .08 and self.previous.fingers_open[self.giver_index]
                 and not self.previous.grasp_flags[self.giver_index], 'Giver must really release and retreat at least8cm upward')
        self.state = RECEIVER_HOLD; self.hold_ticks = 0; self._hover_reference = self.previous

    def _guard_object(self, observation: TransferObservation, side: int) -> dict[str, float]:
        reference = self.references[side]
        measured = np.linalg.solve(observation.tcp_T_world[side], observation.object_T_world)
        slip = float(np.linalg.norm(measured[:3, 3]-reference[:3, 3]))
        rotation = _rotation_distance(reference[:3, :3], measured[:3, :3])
        drop = max(0., float((observation.tcp_T_world[side]@reference)[2, 3]-observation.object_T_world[2, 3]))
        if slip > .005: self.abort('position_slip')
        elif rotation > .15: self.abort('rotation_slip')
        elif drop > .01: self.abort('object_dropped')
        elif self.minimum_object_z_m is not None and observation.object_T_world[2, 3] < self.minimum_object_z_m:
            self.abort('object_below_minimum_height')
        return {'position_slip_m': slip, 'rotation_slip_rad': rotation, 'drop_m': drop}

    def pre_step(self, observation: TransferObservation) -> None:
        _require(self.previous is not None and observation.environment_step == self.previous.environment_step,
                 'External physical step detected before handoff command')
        _require(self.state not in TERMINAL and self.state != RECEIVER_FAILED, 'Transfer phase cannot advance after terminal/refusal event')
        _require(np.array_equal(observation.qpos_rad, self.previous.qpos_rad)
                 and np.array_equal(observation.qvel_rad_s, self.previous.qvel_rad_s)
                 and np.array_equal(observation.tcp_T_world, self.previous.tcp_T_world)
                 and np.array_equal(observation.object_T_world, self.previous.object_T_world),
                 'Physical state changed without an observed paired control step')
        side = self.receiver_index if self.state in (GIVER_RELEASE, GIVER_RETREAT, RECEIVER_HOLD) else self.giver_index
        if not observation.grasp_flags[side]: self.abort('fresh_owner_grasp_lost')
        if self.state == RECEIVER_CONFIRMED and not observation.grasp_flags[self.receiver_index]: self.abort('receiver_confirmation_expired')
        self._guard_object(observation, side)
        _require(self.state != ABORTED, 'Handoff pre-step guard failed: '+str(self.reason))

    def observe(self, observation: TransferObservation) -> dict[str, Any]:
        _require(self.previous is not None and observation.environment_step == self.previous.environment_step+1,
                 'Each handoff observation must follow exactly one actual paired step')
        _require(self.state not in TERMINAL and self.state != RECEIVER_FAILED, 'Terminal controller cannot consume another physical step')
        prior = self.previous; entered = self.state; self.phase_ticks += 1
        finger_change = np.max(np.abs(observation.qpos_rad[:, 7:]-prior.qpos_rad[:, 7:]), axis=1)
        row = {'environment_step': observation.environment_step, 'phase': entered, 'grasp_flags': list(observation.grasp_flags),
               'finger_change_m': finger_change.tolist(), 'fingers_open': list(observation.fingers_open),
               'gripper_commands': list(self._last_grips), 'finger_forces_N': observation.finger_forces_N.tolist()}
        side = self.receiver_index if entered in (GIVER_RELEASE, GIVER_RETREAT, RECEIVER_HOLD) else self.giver_index
        row.update(self._guard_object(observation, side))
        if not observation.grasp_flags[side]: self.abort('actual_owner_grasp_lost')
        if self.state != ABORTED:
            if entered == RECEIVER_CLOSE:
                if observation.grasp_flags[self.receiver_index] and finger_change[self.receiver_index] <= .0002:
                    self.confirmation.append(row.copy()); self.confirmation = self.confirmation[-5:]
                else: self.confirmation = []
                row['receiver_stable_ticks'] = len(self.confirmation)
                if len(self.confirmation) == 5:
                    self.state = RECEIVER_CONFIRMED
                    self.events.append({'event': 'receiver_five_ticks_confirmed', 'environment_step': observation.environment_step})
                elif self.phase_ticks >= self.receiver_max_steps:
                    self.state, self.reason = RECEIVER_FAILED, 'receiver_grasp_timeout'
            elif entered == RECEIVER_CONFIRMED and not observation.grasp_flags[self.receiver_index]:
                self.abort('receiver_confirmation_expired')
            elif entered == GIVER_RELEASE:
                valid = observation.fingers_open[self.giver_index] and not observation.grasp_flags[self.giver_index]
                self.release_confirmation = self.release_confirmation+[row.copy()] if valid else []
                self.release_confirmation = self.release_confirmation[-3:]
                if len(self.release_confirmation) == 3:
                    self.state = GIVER_RETREAT; self._retreat_start = observation
                    self.events.append({'event': 'giver_release_confirmed', 'environment_step': observation.environment_step})
                elif self.phase_ticks >= self.release_max_steps: self.abort('giver_release_timeout')
            elif entered in (RECEIVER_HOLD, SAFE_HOLD):
                reference = self._hover_reference
                _require(reference is not None, 'Stable hold reference missing')
                dp = float(np.linalg.norm(observation.object_T_world[:3, 3]-reference.object_T_world[:3, 3]))
                dr = _rotation_distance(reference.object_T_world[:3, :3], observation.object_T_world[:3, :3])
                dq = float(np.max(np.abs(observation.qpos_rad[:, :7]-reference.qpos_rad[:, :7])))
                arm_speed = float(np.max(np.abs(observation.qpos_rad[:, :7]-prior.qpos_rad[:, :7]))/.05)
                speed = float(np.linalg.norm(observation.object_T_world[:3, 3]-prior.object_T_world[:3, 3])/.05)
                angular = _rotation_distance(prior.object_T_world[:3, :3], observation.object_T_world[:3, :3])/.05
                row.update(hold_position_drift_m=dp, hold_rotation_drift_rad=dr, hold_joint_drift_rad=dq,
                           control_frame_arm_speed_rad_s=arm_speed, control_frame_object_speed_m_s=speed,
                           control_frame_object_angular_speed_rad_s=angular)
                if dp > .003 or dr > .03 or dq > .003 or arm_speed > .05 or speed > .03 or angular > .2:
                    self.abort('hold_not_stable')
                else:
                    self.hold_ticks += 1
                    if self.hold_ticks == 20: self.state = COMPLETE if entered == RECEIVER_HOLD else SAFE_COMPLETE
        self.previous = observation; row['state_after'] = self.state
        self.trace.append(row)
        return row


def paired_transfer_action(arm_targets_rad: Any, machine: TransferStateMachine) -> dict[str, np.ndarray]:
    joints = _finite(arm_targets_rad, (2, 7), 'canonical paired arm targets')
    return {f'panda_wristcam-{side}': np.r_[joints[side], machine.command_grips()[side]].astype(np.float32) for side in range(2)}


def step_transfer(env: Any, checker: Any, machine: TransferStateMachine, arm_targets_rad: Any, *,
                  episode_budget: int) -> dict[str, Any]:
    """One guarded paired step; native recording stays in the caller's wrapper."""
    _require(type(episode_budget) is int and episode_budget > 0, 'Explicit frozen episode budget required')
    before = measure_transfer(env, checker.target_actor); machine.pre_step(before)
    _require(before.environment_step < episode_budget, 'Frozen handoff budget exhausted')
    if machine.state == RECEIVER_CLOSE: checker.set_phase('receiver_contact')
    elif machine.state == RECEIVER_CONFIRMED: checker.set_phase('receiver_confirmed')
    elif machine.state == GIVER_RELEASE: checker.set_phase('giver_release')
    elif machine.state == RECEIVER_HOLD: checker.set_phase('receiver_hold')
    elif machine.state == SAFE_HOLD: checker.set_phase('safe_giver_hold')
    action = paired_transfer_action(arm_targets_rad, machine); arms = _canonical(env)
    _actions_valid(env, arms, action); checker.refresh()
    segments = tuple(ArmTrajectory(arm.key, arm.names, [0., .05],
        [before.qpos_rad[side, :7], action[arm.key][:7]], int(action[arm.key][7])) for side, arm in enumerate(arms))
    segment = build_hold_schedule(*segments, control_timestep_s=.05)
    validation = validate_schedule(segment, before.qpos_rad[0, :7], before.qpos_rad[1, :7], checker, max_joint_step_rad=.02)
    if validation['status'] != 'passed':
        machine.abort('candidate_collision')
        return {'status': 'rejected', 'reason': 'candidate_collision', 'environment_steps': 0, 'validation': validation}
    try:
        _, _, terminated, truncated, info = env.step(action)
        after = measure_transfer(env, checker.target_actor)
        _require(after.environment_step == before.environment_step+1, 'One action did not advance exactly one tick')
        checker.refresh(); collisions = checker.measured_check()
        row = machine.observe(after)
        tracking = float(np.max(np.abs(after.qpos_rad[:, :7]-np.asarray(arm_targets_rad))))
        if collisions: machine.abort('measured_collision')
        elif tracking > .2: machine.abort('tracking_error')
        done, continued = continue_success_only(terminated, truncated, info)
        if done or _bool(truncated): machine.abort('native_failure_or_timeout')
        if machine.state == RECEIVER_CONFIRMED and machine.receiver_index not in machine.references:
            machine.record_receiver_reference(checker.capture_receiver(machine.confirmation))
        if machine.state == GIVER_RETREAT and checker.owner_index == machine.giver_index:
            checker.commit_receiver_owner(machine.release_confirmation)
        return {'status': machine.state, 'environment_steps': 1, 'action': {k: v.tolist() for k, v in action.items()},
                'native_terminated': _bool(terminated), 'native_truncated': _bool(truncated),
                'success_only_termination_continued': continued, 'tracking_error_rad': tracking,
                'collisions': collisions, 'actual_segment_validation': validation, 'observation': after.json(), 'controller': row}
    except Exception as exc:
        delta = int(as_numpy(env.unwrapped.elapsed_steps)[0])-before.environment_step
        machine.abort('step_or_measurement_failed')
        return {'status': ABORTED, 'reason': machine.reason, 'environment_steps': delta,
                'error_type': type(exc).__name__, 'error': str(exc), 'action': {k: v.tolist() for k, v in action.items()}}


def validate_transfer_schedule(env: Any, schedule: Any, checker: Any, machine: TransferStateMachine) -> dict[str, Any]:
    """Validate the full existing scheduler polygon with the unique held owner."""
    observed = measure_transfer(env, checker.target_actor); machine.pre_step(observed)
    arms = _canonical(env); grips = machine.command_grips()
    _require(schedule.control_timestep_s == .05, 'Handoff requires existing20Hz control clock')
    for side, (arm, source) in enumerate(zip(arms, schedule.trajectories)):
        _require(source.action_key == arm.key and source.joint_names == arm.names and source.gripper_command == grips[side],
                 'Schedule changed physical arm, joint order or guarded grip state')
    checker.refresh()
    return validate_schedule(schedule, observed.qpos_rad[0, :7], observed.qpos_rad[1, :7], checker, max_joint_step_rad=.02)


def execute_transfer_schedule(env: Any, schedule: Any, checker: Any, machine: TransferStateMachine, *,
                              episode_budget: int, settle_max_steps: int = 15) -> dict[str, Any]:
    """Thin mixed-grip route execution; all interpolation stays in public schedule."""
    _require(machine.state in (GIVER_HOLD, GIVER_RETREAT), 'Movement is separate from close/release confirmation')
    _require(type(settle_max_steps) is int and settle_max_steps >= 0, 'Invalid settle budget')
    validation = validate_transfer_schedule(env, schedule, checker, machine)
    result = {'status': 'running', 'trace': [], 'validation': validation, 'environment_steps': 0}
    if validation['status'] != 'passed':
        machine.abort('candidate_collision'); result.update(status='rejected', reason='candidate_collision'); return result
    commands = list(schedule.iter_actions()); _require(bool(commands), 'Nonempty handoff motion required')
    final = np.stack([source.positions_rad[-1] for source in schedule.trajectories]); stable = 0
    for index in range(len(commands)+settle_max_steps):
        target = np.stack([commands[index][f'panda_wristcam-{side}'][:7] for side in range(2)]) if index < len(commands) else final
        previous = machine.previous
        item = step_transfer(env, checker, machine, target, episode_budget=episode_budget)
        result['trace'].append(item); result['environment_steps'] += item['environment_steps']
        if machine.state == ABORTED: break
        observed = machine.previous; _require(observed is not None, 'Missing measured handoff frame')
        error = float(np.max(np.abs(observed.qpos_rad[:, :7]-final)))
        _require(previous is not None, 'Missing prior actual joint frame')
        quiet = float(np.max(np.abs(observed.qpos_rad[:, :7]-previous.qpos_rad[:, :7]))/.05) <= .05
        stable = stable+1 if error <= .03 and quiet else 0
        if index+1 >= len(commands) and stable >= 3: result['status'] = 'passed'; break
    if result['status'] == 'running': result.update(status='aborted' if machine.state == ABORTED else 'timeout', reason=machine.reason or 'endpoint_not_stable')
    return result


def _hold_arm_targets(measured_arm_qpos_rad: Any | None, arm_targets_rad: Any | None) -> np.ndarray:
    """Own a finite canonical[2,7] target; an explicit command outranks measured deflection."""
    selected = measured_arm_qpos_rad if arm_targets_rad is None else arm_targets_rad
    return _finite(selected, (2, 7), 'hold arm targets')


def execute_transfer_hold(env: Any, checker: Any, machine: TransferStateMachine, *, episode_budget: int,
                          max_steps: int, arm_targets_rad: Any | None = None) -> dict[str, Any]:
    """Advance one already-entered confirmation or true stable-hold phase."""
    _require(machine.state in (RECEIVER_CLOSE, GIVER_RELEASE, RECEIVER_HOLD, SAFE_HOLD), 'Enter a handoff hold phase explicitly')
    _require(type(max_steps) is int and max_steps > 0, 'Finite hold budget required')
    entered = machine.state
    measured = measure_transfer(env, checker.target_actor).qpos_rad[:, :7] if arm_targets_rad is None else None
    targets = _hold_arm_targets(measured, arm_targets_rad)
    result = {'status': 'running', 'initial_phase': entered, 'environment_steps': 0, 'trace': [],
              'arm_targets_rad': targets.tolist(),
              'arm_target_policy': 'measured_entry' if arm_targets_rad is None else 'explicit_inherited_command'}
    for _ in range(max_steps):
        item = step_transfer(env, checker, machine, targets, episode_budget=episode_budget)
        result['trace'].append(item); result['environment_steps'] += item['environment_steps']
        if machine.state != entered: break
    if machine.state == entered: machine.abort('phase_budget_exhausted')
    result.update(status=machine.state, reason=machine.reason)
    return result


def self_test() -> dict[str, Any]:
    from types import SimpleNamespace as NS
    import torch
    from mani_skill.agents.robots.panda.panda import Panda
    def obs(step: int, grasp: tuple[bool, bool] = (True, True), *, open_giver: bool = False,
            receiver_open: bool = False, giver_z: float = 0., object_x: float = 0.) -> TransferObservation:
        q = np.zeros((2, 9)); q[:, 7:] = .02; q[1, 7:] = .04 if open_giver else .02
        if receiver_open: q[0, 7:] = .04
        tcp = np.tile(np.eye(4), (2, 1, 1)); tcp[1, 2, 3] = giver_z
        obj = np.eye(4); obj[0, 3] = object_x
        return TransferObservation(step, q, np.zeros((2, 9)), tcp, obj, grasp,
                                   np.ones((2, 2, 3)), np.full((2, 2), .04))
    tests = []; machine = TransferStateMachine(1, np.eye(4)); machine.initialize(obs(0, (False, True), receiver_open=True))
    try: machine.begin_giver_release()
    except ValueError: pass
    else: raise AssertionError('Giver released before receiver confirmed')
    machine.begin_receiver_close()
    for step in range(1, 7):
        assert machine.command_grips()[1] == -1
        machine.observe(obs(step))
    assert machine.state == RECEIVER_CONFIRMED and len(machine.confirmation) == 5
    machine.record_receiver_reference(np.eye(4)); machine.begin_giver_release()
    assert machine.command_grips() == (-1, 1)
    for step in range(7, 10): machine.observe(obs(step, (True, False), open_giver=True))
    assert machine.state == GIVER_RETREAT
    machine.observe(obs(10, (True, False), open_giver=True, giver_z=.081)); machine.mark_retreat_complete()
    for step in range(11, 31): machine.observe(obs(step, (True, False), open_giver=True, giver_z=.081))
    assert machine.state == COMPLETE and machine.hold_ticks == 20
    tests.append('five consecutive real-grasp flags precede giver-open,3release,8cm retreat,20true hold')
    negative = TransferStateMachine(1, np.eye(4), receiver_max_steps=6)
    negative.initialize(obs(0, (False, True), receiver_open=True)); negative.begin_receiver_close()
    for step in range(1, 7): negative.command_grips(); negative.observe(obs(step, (False, True)))
    assert negative.state == RECEIVER_FAILED and negative.command_grips()[1] == -1
    negative.begin_safe_giver_hold()
    for step in range(7, 27): negative.observe(obs(step, (False, True)))
    assert negative.state == SAFE_COMPLETE and negative.command_grips()[1] == -1
    tests.append('receiver timeout retains actual giver grip through20tick safe-refusal hold')
    interrupted = TransferStateMachine(1, np.eye(4)); interrupted.initialize(obs(0, (False, True)))
    interrupted.begin_receiver_close()
    for step in range(1, 4): interrupted.command_grips(); interrupted.observe(obs(step))
    interrupted.observe(obs(4, (False, True))); assert len(interrupted.confirmation) == 0
    for step in range(5, 10): interrupted.observe(obs(step))
    assert interrupted.state == RECEIVER_CONFIRMED
    interrupted.record_receiver_reference(np.eye(4))
    try: interrupted.pre_step(obs(9, (False, True)))
    except ValueError: pass
    else: raise AssertionError('Stale receiver confirmation accepted')
    assert interrupted.state == ABORTED and interrupted.command_grips()[1] == -1
    tests.append('one false grasp resets counter; fresh loss invalidates confirmation before release')
    dropped = TransferStateMachine(1, np.eye(4)); dropped.initialize(obs(0, (False, True)))
    dropped.observe(obs(1, (False, True), object_x=.006)); assert dropped.state == ABORTED and dropped.reason == 'position_slip'
    tests.append('original5mm slip envelope and immutable owner references')
    mirror = TransferStateMachine(0, np.eye(4)); mirror.initialize(obs(0, (True, False)))
    mirror.begin_receiver_close()
    for step in range(1, 6): mirror.command_grips(); mirror.observe(obs(step))
    mirror.record_receiver_reference(np.eye(4)); mirror.begin_giver_release()
    assert mirror.command_grips() == (1, -1)
    tests.append('physical left/right identity retained when donor and receiver roles reverse')
    finger1 = NS(pose=NS(to_transformation_matrix=lambda: torch.eye(4)[None]))
    finger2 = NS(pose=NS(to_transformation_matrix=lambda: torch.eye(4)[None]))
    forces = [torch.tensor([[0., .5, 0.]]), torch.tensor([[0., -.5, 0.]])]
    fake = NS(finger1_link=finger1, finger2_link=finger2,
              scene=NS(get_pairwise_contact_forces=lambda link, target: forces[0 if link is finger1 else 1]))
    assert bool(Panda.is_grasping(fake, object())[0])
    forces[0] = torch.tensor([[0., .49, 0.]])
    assert not bool(Panda.is_grasping(fake, object())[0])
    angle = np.deg2rad(86.)
    forces[0] = torch.tensor([[float(np.sin(angle)), float(np.cos(angle)), 0.]])
    assert not bool(Panda.is_grasping(fake, object())[0])
    tests.append('unmodified Panda.is_grasping on fake force tensors:0.5N accepted,0.49N and86degrees rejected')
    paired = paired_transfer_action(np.zeros((2, 7)), negative)
    assert tuple(paired) == ('panda_wristcam-0', 'panda_wristcam-1') and all(v.shape == (8,) and v.dtype == np.float32 for v in paired.values())
    issued = np.arange(14, dtype=np.float64).reshape(2, 7)/100
    measured = issued+.005
    inherited = _hold_arm_targets(measured, issued)
    assert inherited.shape == (2, 7) and np.array_equal(inherited, issued)
    assert not np.shares_memory(inherited, issued) and not np.shares_memory(inherited, measured)
    issued_before = issued.copy(); issued[:] = -1.; measured[:] = -2.
    assert np.array_equal(inherited, issued_before)
    assert np.array_equal(_hold_arm_targets(None, inherited), inherited)
    fallback = _hold_arm_targets(measured, None)
    assert np.array_equal(fallback, measured) and not np.shares_memory(fallback, measured)
    tests.append('explicit[2,7]arm command survives5mrad measured deflection; explicit/fallback targets own nonaliased copies')
    angular_guard = TransferStateMachine(1, np.eye(4), receiver_max_steps=6)
    angular_guard.initialize(obs(0, (False, True))); angular_guard.begin_receiver_close()
    for step in range(1, 7): angular_guard.command_grips(); angular_guard.observe(obs(step, (False, True)))
    angular_guard.begin_safe_giver_hold()
    rotated = obs(7, (False, True)); transform = np.eye(4); angle = .0105
    transform[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    rotated = TransferObservation(rotated.environment_step, rotated.qpos_rad, rotated.qvel_rad_s,
        rotated.tcp_T_world, transform, rotated.grasp_flags, rotated.finger_forces_N, rotated.finger_open_upper_m)
    angular_row = angular_guard.observe(rotated)
    assert angular_guard.state == ABORTED and angular_guard.reason == 'hold_not_stable' and angular_guard.hold_ticks == 0
    assert np.isclose(angular_row['control_frame_object_angular_speed_rad_s'], .21, rtol=0, atol=1e-10)
    tests.append('original.2rad_sangular hold guard aborts.21rad_s and records the actual finite difference')
    bad = 0
    for callback in (lambda: TransferStateMachine(2, np.eye(4)),
                     lambda: TransferStateMachine(1, np.eye(4), receiver_max_steps=4),
                     lambda: paired_transfer_action(np.zeros((7,)), negative),
                     lambda: machine.observe(obs(31)),
                     lambda: TransferObservation(0, np.zeros((2, 8)), np.zeros((2, 9)), np.tile(np.eye(4), (2, 1, 1)), np.eye(4), (True, False), np.zeros((2, 2, 3)), np.full((2, 2), .04)),
                     lambda: _hold_arm_targets(None, np.zeros((7,))),
                     lambda: _hold_arm_targets(None, np.full((2, 7), np.nan)),
                     lambda: _hold_arm_targets(None, np.full((2, 7), np.inf)),
                     lambda: _hold_arm_targets(np.zeros((2, 8)), None)):
        try: callback()
        except ValueError: bad += 1
        else: raise AssertionError('Invalid boundary accepted')
    assert bad == 9
    return {'status': 'passed', 'checks': tests, 'invalid_boundaries_rejected': bad,
            'shapes': {'actions': [2, 8], 'force_vectors': [2, 2, 3], 'unique_object_transform': [4, 4]},
            'environment_steps': 0, 'physics_steps': 0,
            'test_scope': 'CPU synthetic role/state/command contracts; native trajectory/collision/transfer NOT_RUN'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-test', action='store_true')
    parsed = parser.parse_args()
    if not parsed.self_test: parser.error('Library only; use --self-test for CPU synthetic checks')
    print(json.dumps(self_test(), indent=2))
