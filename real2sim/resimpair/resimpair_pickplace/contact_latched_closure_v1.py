"""Candidate bounded native-feedback closure; no pose/gain/friction changes.

The cache is a geometric contact prediction, not a target motor position or a
physical grasp certificate. This module changes actions only. It preserves
native Panda is_grasping and all downstream stability/final/path checks.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import numpy as np

from resimpair_grasp.common import binding, read, resolve, write
from .stability import aperture_command

VERSION = 'cached_width_contact_latched_closure_v1'
POLICY = {
    'schema': VERSION, 'maximum_close_steps': 24, 'preclose_steps': 6,
    'precontact_clearance_m': .002, 'fine_full_aperture_step_m': .001,
    'maximum_total_squeeze_below_native_contact_m': .004,
    'contact_latch_preload_full_aperture_m': .001,
    'required_last_consecutive_native_grasp_samples': 3,
    'maximum_physical_opening_m': .08,
    'native_is_grasping_unchanged': True,
    'gains_friction_pose_and_final_acceptance_unchanged': True,
}


def validate_policy(policy):
    if policy != POLICY:
        raise ValueError('candidate_v1_requires_its_frozen_single_policy')


def contact_geometry(candidate):
    width = float(candidate['width_m'])
    sweep = candidate['closing_sweep']
    values = np.asarray([f['first_contact_width_m'] for f in sweep['fingers']], float)
    if (not np.isfinite(width) or not .012 <= width <= .077 or values.shape != (2,)
            or not np.isfinite(values).all() or np.any(values <= 0) or np.any(values > .08)
            or sweep['contact_pair_reachable'] is not True
            or np.any(np.abs(values - width) > float(sweep['width_tolerance_m']) + 1e-12)):
        raise ValueError('finite_native_qualified_contact_pair_required')
    return {'cache_width_m': width, 'finger_first_contact_widths_m': values.tolist(),
            'first_contact_asymmetry_m': float(np.ptp(values)),
            'precontact_width_m': min(.08, float(max(values.max(), width)) + POLICY['precontact_clearance_m']),
            'minimum_command_width_m': max(0., float(min(values.min(), width)) - POLICY['maximum_total_squeeze_below_native_contact_m'])}


class ClosureSchedule:
    """Pure stateful command policy, also reusable for read-only trace replay."""
    def __init__(self, candidate, initial_width, lower, upper, policy=POLICY):
        validate_policy(policy)
        self.geometry = contact_geometry(candidate)
        self.initial_width = float(initial_width)
        self.lower, self.upper = float(lower), float(upper)
        if (not np.isfinite(self.initial_width) or not 0 <= self.initial_width <= .0801
                or not self.lower <= 0 or not self.upper >= .04):
            raise ValueError('unsupported_Panda_aperture_or_controller_bounds')
        # Bound commands to the physical aperture; retain numerical readback.
        self.initial_command_width = min(self.initial_width, .08)
        if self.initial_command_width < self.geometry['precontact_width_m'] - .001:
            raise ValueError('preclose_requires_original_open_hand_not_an_existing_squeeze')
        self.step_index = 0
        self.previous_target = self.initial_command_width
        self.latched_width = None
        self.latch_step = None
        self.grasp_streak = 0
        self.observations = []

    def observe(self, measured_width, native_grasp):
        measured_width = float(measured_width)
        if not np.isfinite(measured_width) or not -.0001 <= measured_width <= .0801:
            raise ValueError('nonfinite_or_out_of_range_measured_aperture')
        if type(native_grasp) is not bool:
            raise ValueError('native_grasp_boolean_required')
        self.grasp_streak = self.grasp_streak + 1 if native_grasp else 0
        if native_grasp and self.latched_width is None:
            # Preserve existing compression, or at most 0.5mm/finger nominal
            # preload if contact was transient while the drive target was wider.
            desired = min(self.previous_target,
                          measured_width - POLICY['contact_latch_preload_full_aperture_m'])
            if desired < self.geometry['minimum_command_width_m'] - 1e-12:
                raise RuntimeError('native_contact_below_bounded_cache_squeeze_range')
            self.latched_width = float(desired)
            self.latch_step = self.step_index
        self.observations.append({'step': self.step_index, 'measured_aperture_m': measured_width,
                                  'native_grasp': native_grasp, 'grasp_streak': self.grasp_streak})

    def next_command(self, measured_width, native_grasp):
        if self.step_index >= POLICY['maximum_close_steps']:
            raise RuntimeError('bounded_close_step_budget_exhausted')
        self.observe(measured_width, native_grasp)
        if self.latched_width is not None:
            width = self.latched_width
            phase = 'hold_fixed_contact_target'
        elif self.step_index < POLICY['preclose_steps']:
            alpha = (self.step_index + 1) / POLICY['preclose_steps']
            width = self.initial_command_width + alpha * (self.geometry['precontact_width_m'] - self.initial_command_width)
            phase = 'preclose_toward_native_contact'
        else:
            width = max(self.geometry['minimum_command_width_m'],
                        self.previous_target - POLICY['fine_full_aperture_step_m'])
            phase = 'bounded_fine_close'
        command = aperture_command(width, self.lower, self.upper)
        result = {'step': self.step_index, 'phase': phase, 'full_aperture_target_m': float(width),
                  'per_finger_target_m': float(width / 2), 'normalized_command': command,
                  'latched': self.latched_width is not None}
        self.previous_target = float(width); self.step_index += 1
        return result

    def finish(self, measured_width, native_grasp):
        self.observe(measured_width, native_grasp)
        if self.step_index != POLICY['maximum_close_steps']:
            raise RuntimeError('closure_did_not_execute_the_fixed_step_budget')
        if (self.latched_width is None or self.grasp_streak < POLICY['required_last_consecutive_native_grasp_samples']):
            raise RuntimeError('no_persistent_native_grasp_after_bounded_contact_closure')
        return aperture_command(self.latched_width, self.lower, self.upper)


def _array(value):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    return array(value)


def _native_feedback(agent, actor):
    q = _array(agent.robot.qpos).reshape(9)
    pair = []
    for link in (agent.finger1_link, agent.finger2_link):
        pair.append(_array(agent.scene.get_pairwise_contact_forces(link, actor)).reshape(3).tolist())
    return {'aperture_m': float(q[7:].sum()), 'qpos': q.tolist(),
            'native_grasp': bool(_array(agent.is_grasping(actor)).item()),
            'finger_object_force_world_N': pair}


@contextmanager
def install(rec, task, policy=POLICY):
    """One isolated worker; wrap the official close and this recorder's actions.

    Open/retreat, original planning, grasp detector, stability tests and final
    validators are untouched. Explicit legacy -1 holds are mapped to the same
    latched target until the operation's original :open stage.
    """
    validate_policy(policy)
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver as Solver
    operations = {op['id']: op for op in task['operations']}
    if len(operations) != len(task['operations']): raise ValueError('unique_operation_ids_required')
    old_close, old_step = Solver.close_gripper, rec.step
    instance_had_step = 'step' in rec.__dict__
    instance_step = rec.__dict__.get('step')
    latch = None; completed_traces = []

    def step(action):
        nonlocal latch
        value = action
        if latch is not None:
            if rec.stage == latch['operation'] + ':open':
                latch['released_at_stage'] = rec.stage
                latch = None
            else:
                prefix = latch['operation'] + ':'
                if not rec.stage.startswith(prefix):
                    raise RuntimeError('latched_grasp_crossed_operation_without_original_open')
                key = f'panda_wristcam-{latch["arm"]}'
                incoming = _array(action[key]).reshape(8)
                if np.isclose(incoming[-1], -1., rtol=0, atol=1e-7):
                    value = dict(action); changed = incoming.copy(); changed[-1] = latch['command']; value[key] = changed
                    latch['legacy_full_close_commands_replaced'] += 1
                    latch['replaced_stage_counts'][rec.stage] = latch['replaced_stage_counts'].get(rec.stage, 0) + 1
                elif not np.isclose(incoming[-1], latch['command'], rtol=0, atol=1e-7):
                    raise RuntimeError('unexpected_gripper_command_before_original_release')
        return old_step(value)

    def close(motion, other_gripper_state=None, t=6):
        nonlocal latch
        if motion.env is not rec or motion.control_mode != 'pd_joint_pos' or motion.multi_robot_id not in (0, 1):
            raise ValueError('candidate_requires_bound_dual_Panda_absolute_position_recorder')
        if latch is not None: raise RuntimeError('second_closure_before_original_release_forbidden')
        opid, suffix = rec.stage.rsplit(':', 1)
        if suffix != 'close' or opid not in operations: raise ValueError('explicit_operation_close_stage_required')
        op = operations[opid]; arm = motion.multi_robot_id; index = rec.object_ids.index(op['object'])
        agent, actor = rec.agent.agents[arm], rec.objects[index]
        selected = read(rec.directory / (opid + '_grasp_selection.json'))
        cache_ref = rec.scene_spec['grasp_caches'][op['object']]; cache = read(resolve(cache_ref))
        candidate = next(c for c in cache['candidates'] if c['id'] == selected['selected'])
        cfg = agent.controller.controllers['gripper'].config
        if not cfg.normalize_action or cfg.use_delta: raise ValueError('absolute_normalized_gripper_required')
        before = _native_feedback(agent, actor)
        schedule = ClosureSchedule(candidate, before['aperture_m'], float(cfg.lower), float(cfg.upper), policy)
        trace = {'schema': VERSION, 'implementation': binding(__file__), 'policy': deepcopy(policy),
            'operation': opid, 'object': op['object'], 'arm': arm, 'candidate': deepcopy(candidate),
            'cache': cache_ref, 'geometric_prediction': schedule.geometry,
            'baseline_requested_steps': t, 'actual_fixed_close_steps': POLICY['maximum_close_steps'],
            'controller_config_readback': {k: getattr(cfg, k) for k in ['lower', 'upper', 'stiffness', 'damping', 'force_limit', 'friction', 'normalize_action', 'use_delta', 'drive_mode']},
            'native_is_grasping_parameters_unchanged': True, 'steps': [],
            'no_new_physical_retry': True, 'failure': None}
        arm_q = np.asarray(before['qpos'][:7]); result = None
        try:
            for _ in range(POLICY['maximum_close_steps']):
                feedback = _native_feedback(agent, actor)
                command = schedule.next_command(feedback['aperture_m'], feedback['native_grasp'])
                actions = {f'panda_wristcam-{arm}': np.r_[arm_q, command['normalized_command']],
                    f'panda_wristcam-{1-arm}': np.r_[_array(rec.agent.agents[1-arm].robot.qpos).reshape(9)[:7], other_gripper_state]}
                result = rec.step(actions); motion.elapsed_steps += 1
                trace['steps'].append({'frame_after_step': len(rec.rows), 'before': feedback, 'command': command,
                                       'after': _native_feedback(agent, actor)})
            feedback = _native_feedback(agent, actor)
            held_command = schedule.finish(feedback['aperture_m'], feedback['native_grasp'])
            motion.gripper_state = held_command
            latch = {'operation': opid, 'arm': arm, 'command': held_command,
                'target_full_aperture_m': schedule.latched_width, 'first_latch_step': schedule.latch_step,
                'legacy_full_close_commands_replaced': 0, 'replaced_stage_counts': {}, 'released_at_stage': None}
            trace['hold_latch'] = latch; completed_traces.append(trace)
            return result
        except Exception as exc:
            trace['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            # Save close evidence now; later carry command substitutions are
            # written separately at context exit, including exceptions.
            immediate = {k: deepcopy(v) for k, v in trace.items() if k != 'hold_latch'}
            write(rec.directory / 'contact_latched_closure' / (opid + '_close.json'), immediate)

    Solver.close_gripper = close; rec.step = step
    try:
        yield
    finally:
        Solver.close_gripper = old_close
        if instance_had_step: rec.step = instance_step
        else: del rec.__dict__['step']
        for trace in completed_traces:
            write(rec.directory / 'contact_latched_closure' / (trace['operation'] + '_hold.json'),
                  {'schema': VERSION, 'implementation': binding(__file__), **trace['hold_latch']})
