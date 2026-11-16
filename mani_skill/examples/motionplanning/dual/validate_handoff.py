"""Single real banana handoff; separate400-tick engineering capability.

This entry owns setup and phase orchestration only. Native MPLib plans feed the
public paired scheduler; transfer_execution enforces mixed grips and physical
confirmation; transfer_collision keeps one payload and one prediction owner.
The apple/banana adapter records two distinct scene actors. Both original Panda
predicates/force vectors on the SAME banana are additionally recorded at100Hz.
Development seeds0..4 and previously exposed310..314/330..334/340..341 permit a fixed recipe.
Independent round3 acceptance seeds350..354 require a fixed recipe without search.
This entry checks scope only; the root parent must prevalidate the separately
frozen capability protocol, source identities and recipe before allocating.
CPU --self-test constructs no simulator. Apply this version and schedule native
runs only after120 paired M1 allocations and their immutable formal summary.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback
from typing import Any

import numpy as np

from .multitask_recording import make_physics_capture, finalize_physics_capture, save_task_geometry
from .trajectory_execution import as_numpy
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .transfer_collision import TransferCollisionChecker
from .transfer_execution import (TransferStateMachine, measure_transfer, execute_transfer_schedule,
    execute_transfer_hold, RECEIVER_CONFIRMED, RECEIVER_FAILED, GIVER_RETREAT, COMPLETE, SAFE_COMPLETE)
from .validate_grasp import json_value, save_report
from .validate_path_conflict import _state_arrays
from .validate_multitask import termination_handler
from .validate_sync_compact import (compact_phase_limits, compact_set_phase_limits, compact_retime_pair,
    compact_topp_reconstruction, compact_kinematic_audit, compact_urdf_velocity_limits)

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
EVIDENCE = ROOT/'infrastructure/validation/dual-arm/m1-multitask-20261116'
TASK = 'TwoRobotPickAppleBananaToBaskets-v1'
SCHEMA = 'm1-single-banana-handoff-world-recipe-v1'
CENTRES = ((-.16, 0., .22), (-.12, 0., .26), (-.08, 0., .30))
FRACTIONS = (.20, .25, .30)
RECEIVER_FRACTIONS = (0., .15, .20, .25, .30)
RECEIVER_TILTS_DEG = (30., 45., 60.)
RECEIVER_HEIGHT_OFFSETS_M = (-.003, 0., .003)
YAWS = (0., float(np.pi/2))
BUDGET = 400
SOURCE_VERSION = 'handoff-confirmed-physical-source-departure-round3-v14'
GRASP_POLICY = 'opposite_PCA_slice_anchors_giver_topdown25mm_receiver_configured_tilt_OBBheight_v4'
RECEIVER_PREGRASP_POLICY = 'negative_actual_grasp_approach_80mm_preserve_wxyz_v2'
RECEIVER_PREGRASP_DISTANCE_M = .08
NEGATIVE_RECEIVER_MISS_M = .12
DEVELOPMENT_SEEDS = tuple(range(5)) + tuple(range(310, 315)) + tuple(range(330, 335)) + (340, 341)
ACCEPTANCE_SEEDS = tuple(range(350, 355))


class QueryMutationError(RuntimeError):
    """A changed real state is fatal; it is never another rejected candidate."""


def require(condition: Any, message: str) -> None:
    if not bool(condition): raise ValueError(message)


def recipe(value: Any) -> dict[str, Any]:
    """A fixed member of the finite development grid, independent of seed."""
    require(isinstance(value, dict) and value.get('schema') == SCHEMA, 'Explicit handoff world recipe required')
    fraction, yaw = float(value['anchor_fraction']), float(value['object_long_axis_yaw_rad'])
    receiver_fraction = float(value.get('receiver_anchor_fraction', fraction))
    tilt, height_offset = float(value['receiver_tilt_deg']), float(value['receiver_height_offset_m'])
    centre = np.asarray(value['object_centre_world_m'], dtype=float)
    require(fraction in FRACTIONS and receiver_fraction in RECEIVER_FRACTIONS and tilt in RECEIVER_TILTS_DEG
            and height_offset in RECEIVER_HEIGHT_OFFSETS_M and yaw in YAWS and centre.shape == (3,) and np.isfinite(centre).all()
            and tuple(centre) in CENTRES, 'Recipe is outside the declared finite geometry grid')
    require(value.get('grasp_policy') == GRASP_POLICY
            and value.get('negative_receiver_miss_m') == NEGATIVE_RECEIVER_MISS_M, 'Frozen grasp/miss policy required')
    return {'schema': SCHEMA, 'anchor_fraction': fraction, 'receiver_anchor_fraction': receiver_fraction, 'object_centre_world_m': centre.tolist(),
            'receiver_tilt_deg': tilt, 'receiver_height_offset_m': height_offset,
            'object_long_axis_yaw_rad': yaw, 'grasp_policy': value['grasp_policy'],
            'negative_receiver_miss_m': NEGATIVE_RECEIVER_MISS_M}


def candidate_recipes(fixed: dict | None = None) -> list[dict]:
    if fixed is not None: return [recipe(fixed)]
    return [recipe({'schema': SCHEMA, 'anchor_fraction': fraction, 'receiver_anchor_fraction': receiver_fraction,
                    'receiver_tilt_deg': tilt, 'receiver_height_offset_m': height_offset,
                    'object_centre_world_m': centre, 'object_long_axis_yaw_rad': yaw,
                    'grasp_policy': GRASP_POLICY,
                    'negative_receiver_miss_m': NEGATIVE_RECEIVER_MISS_M})
            for fraction, receiver_fraction, centre, yaw, tilt, height_offset in itertools.product(
                FRACTIONS, RECEIVER_FRACTIONS, CENTRES, YAWS, RECEIVER_TILTS_DEG, RECEIVER_HEIGHT_OFFSETS_M)]


def validate_scope(seed: int, develop: bool, fixed: dict | None) -> None:
    require(type(seed) is int and seed in DEVELOPMENT_SEEDS + ACCEPTANCE_SEEDS,
            'Handoff scope is development0..4/310..314/330..334/340..341 or separately allocated independent350..354 only')
    require(type(develop) is bool and develop != (fixed is not None), 'Select exactly one of --develop or a fixed --recipe')
    if seed in ACCEPTANCE_SEEDS:
        require(not develop and fixed is not None, 'Independent handoff acceptance requires one fixed --recipe; --develop is forbidden')


def anchor_geometry(vertices: Any, fraction: float, receiver_fraction: float | None = None) -> dict[str, Any]:
    """Actor-local cooked mesh[N,3] -> separate opposite longitudinal slices.

    Slice means follow the curved banana; using a straight PCA axis point alone
    could put a grasp centre outside the object. This is only a pose proposal;
    original native force predicates and all collision checks remain required.
    """
    points = np.asarray(vertices, dtype=float)
    if receiver_fraction is None: receiver_fraction = fraction
    require(points.ndim == 2 and points.shape[1] == 3 and len(points) >= 8
            and np.isfinite(points).all() and fraction in FRACTIONS and receiver_fraction in RECEIVER_FRACTIONS,
            'Finite cooked mesh[N>=8,3] and declared role anchor fractions required')
    mean = points.mean(axis=0); _, _, basis = np.linalg.svd(points-mean, full_matrices=False)
    axis = basis[0].copy()
    if axis[np.argmax(np.abs(axis))] < 0: axis *= -1
    s = (points-mean)@axis; low, high = float(s.min()), float(s.max()); extent = high-low
    require(extent >= .08, 'Payload is not sufficiently elongated for separated grasp proposals')
    anchors, slices = [], []
    for sign, role_fraction in ((1., fraction), (-1., receiver_fraction)):
        location = (low+high)/2+sign*role_fraction*extent
        order = np.argsort(np.abs(s-location), kind='stable')[:max(8, int(np.ceil(.10*len(points))))]
        selected = points[order]; centre = selected.mean(axis=0)
        anchors.append(centre.tolist())
        slices.append({'axis_location_m': location, 'vertices': len(selected),
                       'local_bounds_m': [selected.min(axis=0).tolist(), selected.max(axis=0).tolist()]})
    separation = float(np.linalg.norm(np.asarray(anchors[0])-anchors[1]))
    require(separation >= .035, 'Opposite actual mesh slices do not provide separated grasp anchors')
    return {'long_axis_local': axis.tolist(), 'long_extent_m': extent,
            'giver_anchor_fraction': fraction, 'receiver_anchor_fraction': receiver_fraction,
            'giver_receiver_anchor_local_m': anchors, 'slice_evidence': slices,
            'anchor_separation_m': separation, 'physical_feasibility': 'UNVERIFIED proposal; native force/collision gates required'}


def world_object_goal(object_T_world: Any, local_axis: Any, selected: dict) -> np.ndarray:
    transform = np.asarray(object_T_world, dtype=float).copy()
    require(transform.shape == (4, 4) and np.isfinite(transform).all(), 'Measured object world transform required')
    axis = transform[:3, :3]@np.asarray(local_axis, dtype=float)
    require(np.linalg.norm(axis[:2]) >= .5, 'Banana long axis is not close enough to horizontal for this recipe')
    delta = selected['object_long_axis_yaw_rad']-np.arctan2(axis[1], axis[0])
    c, s = np.cos(delta), np.sin(delta)
    transform[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])@transform[:3, :3]
    transform[:3, 3] = selected['object_centre_world_m']
    return transform


def receiver_pregrasp_world_pose(grasp_xyz_wxyz: Any) -> np.ndarray:
    """Approach along the grasp frame, preserving its world orientation.

    A tilted open finger must not sweep longitudinally through the payload.
    Predicted screening and measured positive/negative targets share this same
    transform. Full scene and actual contact/slip guards remain mandatory.
    """
    pose = np.asarray(grasp_xyz_wxyz, dtype=float)
    require(pose.shape == (7,) and np.isfinite(pose).all(), 'Finite world grasp pose[xyz,wxyz] required')
    require(abs(float(np.linalg.norm(pose[3:]))-1.) <= 1e-6, 'World grasp quaternion must be unit WXYZ')
    result = pose.copy()
    w, x, y, z = pose[3:]
    approaching = np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
    result[:3] -= RECEIVER_PREGRASP_DISTANCE_M*approaching
    return result


def receiver_approach_stages(case: str, pre: Any, grasp: Any) -> tuple[tuple[str, Any, bool], ...]:
    """Declare actual approach targets before moving or closing the receiver.

    The negative closes physically at the checked pregrasp to produce a real
    grasp timeout. It neither executes nor claims the unneeded grasp descent.
    Positive transfer retains its original pregrasp and descent targets.
    """
    require(case in ('positive', 'receiver-failure'), 'Explicit handoff case required')
    stages = (('receiver_pregrasp', pre, False),)
    return stages + (('receiver_descend', grasp, True),) if case == 'positive' else stages


def receiver_grasp_axes(long_axis_world: Any, tilt_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Incline the receiver wrist by a declared angle, retaining its pinch axis.

    This changes a candidate grasp geometry, never its collision exemptions.
    The miss stays on the canonical cross-table closing axis. All proposed
    targets and paired paths still require native IK and full scene checks.
    """
    axis = np.asarray(long_axis_world, dtype=float)
    require(tilt_deg in RECEIVER_TILTS_DEG, 'Declared receiver wrist tilt required')
    require(axis.shape == (3,) and np.isfinite(axis).all(), 'Finite world long axis[3] required')
    horizontal = np.r_[axis[:2], 0.]
    length = float(np.linalg.norm(horizontal))
    require(length >= .5, 'World long axis must remain close to horizontal')
    horizontal /= length
    canonical = np.cross([0., 0., -1.], horizontal)
    angle = np.deg2rad(tilt_deg)
    approaching = np.sin(angle)*horizontal + np.cos(angle)*np.array([0., 0., -1.])
    return canonical, canonical.copy(), approaching


def candidate_joint_limits_pass(joints: Any, limits: Any) -> bool:
    """Reject each invalid IK branch independently, with the checker tolerance."""
    q, bounds = np.asarray(joints, dtype=float), np.asarray(limits, dtype=float)
    require(q.shape == (7,) and bounds.shape == (7, 2) and np.isfinite(q).all()
            and np.isfinite(bounds).all() and np.all(bounds[:, 0] <= bounds[:, 1]),
            'Finite canonical arm candidate[7] and joint bounds[7,2] required')
    return bool(np.all(q >= bounds[:, 0]-1e-6) and np.all(q <= bounds[:, 1]+1e-6))


def handoff_contact_policy(row: dict, phase: str, checker: Any) -> str:
    """Allow role-authorized banana fingers; the unused apple is an obstacle."""
    bodies = row.get('bodies', ())
    bananas = [b for b in bodies if b.get('kind') == 'payload' and b.get('target_index') == 1]
    robots = [b for b in bodies if b.get('kind') == 'robot']
    if robots and any(b.get('kind') == 'payload' and b.get('target_index') == 0 for b in bodies):
        return 'forbidden'
    if not bananas or not robots: return row['class']
    if len(bodies) != 2 or len(bananas) != 1 or len(robots) != 1: return 'forbidden'
    other = robots[0]; index, link = other.get('robot_index'), other.get('link_name')
    if type(index) is not int or index not in (0, 1): return 'forbidden'
    if isinstance(checker, TransferCollisionChecker):
        allowed = checker.allows_target_contact(index, link)
    else:
        # Initial original single-arm reach/close only; no transfer has begun.
        allowed = bool(checker is not None and getattr(checker, 'target_contact_allowed', False)
                       and getattr(checker, 'active_robot', None) == ('left', 'right')[index]
                       and link in ('panda_leftfinger', 'panda_rightfinger'))
    return 'handoff_finger_allowed' if allowed else 'forbidden'


def configuration() -> dict[str, Any]:
    return {'protocol': 'single-banana-handoff-repair-round3-v14', 'max_episode_steps': BUDGET,
            'source_version': SOURCE_VERSION, 'receiver_pregrasp_policy': RECEIVER_PREGRASP_POLICY,
            'source_departure_policy': 'measured_dz_gt_3mm_and_all_geometry_clear_and_no_native_source_contact_v1',
            'receiver_pregrasp_distance_m': RECEIVER_PREGRASP_DISTANCE_M,
            'negative_receiver_miss_m': NEGATIVE_RECEIVER_MISS_M,
            'negative_receiver_closing_pose': 'actual_checked_pregrasp_no_descent_v3',
            'joint_vel_limit_rad_s': 1.5, 'joint_acc_limit_rad_s2': 2.6,
            'approach_joint_vel_limit_rad_s': .9, 'approach_joint_acc_limit_rad_s2': .9,
            'solver_limit_fraction': .99, 'preparation_steps': 10,
            'giver_lift_time_scale': 4.,
            'hold_target_policy': 'inherit_last_issued_controller_command',
            'stationary_path_policy': 'measured_start_to_inherited_command_one_control_interval',
            'receiver_grasp_frame_policy': 'recipe_outward_wrist_long_axis_tilt_v4',
            'receiver_grasp_height_policy': 'PCA_slice_XY_topdown_OBB25mm_Z_not_inclined_axis_depth',
            'negative_miss_direction_policy': 'canonical_unflipped_closing_axis',
            'min_actual_object_origin_lift_m': .10, 'commanded_lift_m': .12,
            'giver_grasp_stable_ticks': 5, 'receiver_grasp_stable_ticks': 5,
            'receiver_close_max_ticks': 20, 'giver_release_stable_ticks': 3,
            'giver_release_max_ticks': 30, 'giver_retreat_minimum_m': .08,
            'giver_retreat_command_m': .085, 'final_actual_hover_ticks': 20,
            'control_hz': 20, 'physics_hz': 100, 'continuous_collision_detection': False,
            'success_scope': 'controlled single-object handoff or declared safe-refusal negative; no basket-placement claim'}


def handoff_phase_limits(name: str) -> dict[str, Any]:
    """Slow initial liftoff through native TOPP limits; leave other phases alone."""
    cfg = configuration()
    approach = name.startswith('initial-anchor-') or name in ('pregrasp', 'descend', 'receiver_pregrasp', 'receiver_descend')
    if name == 'giver_lift':
        scale = cfg['giver_lift_time_scale']
        cfg['joint_vel_limit_rad_s'] /= scale
        cfg['joint_acc_limit_rad_s2'] /= scale**2
    limits = compact_phase_limits(cfg, 'pregrasp' if approach else name)
    if name == 'giver_lift':
        limits.update(limit_group='gentle_liftoff', requested_time_scale=scale)
    return limits


def issued_arm_targets(ticks: list[dict], keys: list[str]) -> np.ndarray:
    """Copy the actual last submitted command, never resample measured qpos."""
    require(bool(ticks) and len(keys) == 2 and len(set(keys)) == 2, 'Recorded command and two canonical keys required')
    actions = np.asarray([ticks[-1]['actions'][key] for key in keys], dtype=float)
    require(actions.shape == (2, 8) and np.isfinite(actions).all(), 'Finite recorded canonical actions[2,8] required')
    return actions[:, :7].copy()


def inherited_stationary_path(source: ArmTrajectory, target: Any) -> ArmTrajectory:
    """Retain measured collision start and emit the previous target at t=50ms.

    The arm can still have tracking error. Check the actual-to-target segment
    with the original full-scene checker instead of pretending it has settled.
    """
    command = np.asarray(target, dtype=float)
    require(command.shape == (7,) and np.isfinite(command).all(), 'Finite inherited arm command[7] required')
    return ArmTrajectory(source.action_key, source.joint_names, [0., .05],
                         np.stack((source.positions_rad[0], command)), source.gripper_command)


def held_substep_audit(substeps: list[dict]) -> dict[str, Any]:
    """Audit each recorded actual holding role against its original capture."""
    from .lift_execution import _rotation_distance
    per_role: dict[int, list] = {0: [], 1: []}
    missing = []
    for row in substeps:
        value = row.get('transfer')
        if not value or value.get('guard_role_index') is None: continue
        role = value['guard_role_index']; reference = value['captured_relations'].get(str(role))
        if reference is None: missing.append(row['substep']); continue
        relative = np.linalg.solve(np.asarray(value['tcp_T_world'])[role], value['object_T_world'])
        ref = np.asarray(reference)
        per_role[role].append((float(np.linalg.norm(relative[:3, 3]-ref[:3, 3])),
                               _rotation_distance(ref[:3, :3], relative[:3, :3]), bool(value['grasp_flags'][role])))
    results = [{'robot_index': role, 'physical_substeps': len(items),
                'max_position_slip_m': max((i[0] for i in items), default=None),
                'max_rotation_slip_rad': max((i[1] for i in items), default=None),
                'native_grasp_false_substeps': sum(not i[2] for i in items)} for role, items in per_role.items()]
    passed = not missing and any(per_role.values()) and all(i[0] <= .005 and i[1] <= .15 for items in per_role.values() for i in items)
    return {'status': 'passed' if passed else 'failed', 'roles': results,
            'missing_reference_substeps': missing, 'position_limit_m': .005, 'rotation_limit_rad': .15,
            'native_grasp_100Hz': 'unaltered observations retained;5/3/20 control-tick predicates enforced by state machine',
            'scope': 'actual sampled held relative poses; no continuous collision certification'}


def integration_audit(ticks: list[dict], substeps: list[dict], *, capture_step: int,
                      receiver_confirmed_step: int | None, negative: bool, giver_key: str) -> dict:
    """Check the recorder/controller join without relying on own-object labels."""
    clock = len(substeps) == 5*len(ticks) and all(
        row['substep'] == index+1 and row['control_step'] == ticks[index//5]['environment_step']
        and abs(row['sim_timestep_s']-.01) <= 1e-12 and 'transfer' in row
        for index, row in enumerate(substeps))
    requested_actual_equal = all(set(t['requested_actions']) == set(t['actions']) and all(
        np.array_equal(t['requested_actions'][key], t['actions'][key]) for key in t['actions']) for t in ticks)
    after_capture = [t for t in ticks if t['environment_step'] > capture_step]
    opens = [t['environment_step'] for t in after_capture if t['actions'][giver_key][-1] == 1]
    gate = bool(after_capture) and (not opens if negative else bool(opens) and receiver_confirmed_step is not None
            and opens[0] > receiver_confirmed_step)
    return {'status': 'passed' if clock and requested_actual_equal and gate else 'failed',
            'all_control_ticks_have_five100Hz_transfer_observations': clock,
            'requested_actual_actions_identical': requested_actual_equal,
            'giver_open_steps_after_initial_capture': opens, 'receiver_confirmed_step': receiver_confirmed_step,
            'giver_open_authorization': gate, 'physical_substeps': len(substeps), 'control_ticks': len(ticks)}


def native_run(args: Any, output: Path, report: dict) -> None:
    import gymnasium as gym
    import sapien
    from transforms3d.quaternions import mat2quat
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
    from .multitask_adapters import get_task_adapter
    from .dual_arm_execution import _canonical, plan_dual_pose
    from .combined_collision import validate_schedule
    from .grasp_collision import LiveGraspCollisionChecker
    from .single_arm_execution import execute_reach, close_and_confirm

    env = raw = capture = checker = original_hook = machine = None
    frames: list = []; ticks: list = []; substeps: list = []; keys: list = []
    support_latched = False; call_number = 0
    report['fault_events'] = []
    fixed = None if args.recipe is None else recipe(json.loads(args.recipe.read_text()))
    candidates = candidate_recipes(fixed)
    try:
        base = gym.make(TASK, obs_mode='none', reward_mode='none', control_mode='pd_joint_pos',
                        sim_backend='cpu', render_backend='gpu', render_mode='rgb_array', max_episode_steps=BUDGET)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name='trajectory', save_video=False,
            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
            source_type='motionplanning', source_desc=f'one-real-banana handoff;engineering400;case={args.case}')
        env.reset(seed=args.seed); raw = env.unwrapped
        agents = (raw.left_agent, raw.right_agent); banana = raw.banana_out
        adapter = get_task_adapter(TASK, raw)
        require(adapter.targets[0] is not banana and adapter.targets[1] is banana, 'Recorder must contain two distinct actors, never two banana aliases')
        report.update(task_configuration=adapter.configuration(), target_names=list(adapter.target_names),
                      task_geometry=save_task_geometry(output, raw, adapter))
        report['target_state_names'] = [report['task_geometry']['actors'][f'target{i}']['state_name'] for i in range(2)]
        initial = _state_arrays(raw.get_state_dict()); np.savez(output/'initial-state.npz', **initial)
        report['initial_state_sha256'] = hashlib.sha256((output/'initial-state.npz').read_bytes()).hexdigest()
        keys = list(raw.agent.agents_dict)
        require(len(keys) == 2 and all(raw.agent.agents_dict[k] is a for k, a in zip(keys, agents)), 'Physical action order changed')
        report.update(action_key_order=keys, backend={'physics': raw.backend.sim_backend, 'render': raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), physics_timestep_s=float(raw.sim_timestep))

        def support() -> tuple[bool, bool]:
            nonlocal support_latched
            if isinstance(checker, TransferCollisionChecker):
                support_latched |= bool(checker.active.support_released)
            return False, support_latched

        capture = make_physics_capture(env, raw, adapter, args, report, frames, ticks, substeps, keys,
            checker_provider=lambda: checker, support_release_provider=support,
            contact_policy=lambda row, phase: handoff_contact_policy(row, phase, checker), hard_step_budget=BUDGET)
        capture.target_phases = ('preparation', 'preparation'); original_hook = capture.original_hook

        def record_transfer_substep() -> None:
            capture.record_substep()
            observed = measure_transfer(capture, banana)
            extra = observed.json()
            extra.update(actor='banana_out', state_machine_phase=None if machine is None else machine.state,
                prediction_owner_index=None if not isinstance(checker, TransferCollisionChecker) else checker.owner_index,
                checker_phase=None if not isinstance(checker, TransferCollisionChecker) else checker.phase,
                captured_relations={} if machine is None else {str(k): v.tolist() for k, v in machine.references.items()},
                guard_role_index=None if machine is None else (machine.receiver_index if machine.state in
                    ('giver_release', 'giver_retreat', 'receiver_hold', COMPLETE) else machine.giver_index))
            substeps[-1]['transfer'] = extra

        raw._after_simulation_step = record_transfer_substep
        capture.frame(); arms = _canonical(capture)
        require(abs(float(raw.control_timestep)-.05) < 1e-12 and abs(float(raw.sim_timestep)-.01) < 1e-12,
                'Original20Hz/100Hz clock required')
        initial_hold = {a.key: np.r_[a.arm_qpos(), 1].astype(np.float32) for a in arms}
        for _ in range(10):
            _, _, done, truncated, _ = capture.step(initial_hold)
            require(not bool(as_numpy(done).any()) and not bool(as_numpy(truncated).any()), 'Native end during preparation')

        class AuditedSolver(PandaArmMotionPlanningSolver):
            def setup_planner(self) -> Any:
                planner = super().setup_planner(); original = planner.TOPP
                def traced(path: Any, step: float = .1, verbose: bool = False) -> Any:
                    values = original(path, step, verbose)
                    payload = {'native_path': np.asarray(path), 'solver_velocity_limits': np.asarray(planner.joint_vel_limits),
                               'solver_acceleration_limits': np.asarray(planner.joint_acc_limits)}
                    payload.update({'raw_'+k: np.asarray(v) for k, v in zip(('time', 'position', 'velocity', 'acceleration', 'duration'), values) if v is not None})
                    np.savez(output/f'query-{call_number:03d}-arm{self.multi_robot_id}-native-topp.npz', **payload)
                    self.parameterization = compact_topp_reconstruction(path, planner.joint_vel_limits,
                        planner.joint_acc_limits, step, values)
                    return values
                planner.TOPP = traced; return planner

            def move_to_pose_with_screw(self, pose: Any, dry_run: bool = False, refine_steps: int = 0,
                                        other_gripper_state: Any = None) -> Any:
                require(dry_run, 'Only original MPLib dry-run planning is allowed')
                self.parameterization = None; self.native_result = None
                started = time.perf_counter(); item = {'query': call_number, 'phase': capture.phase, 'arm_index': self.multi_robot_id}
                try:
                    self.native_result = super().move_to_pose_with_screw(pose, dry_run=True, refine_steps=refine_steps,
                                                                      other_gripper_state=other_gripper_state)
                    item.update(status='returned', result_status=self.native_result.get('status') if isinstance(self.native_result, dict) else str(self.native_result))
                    return self.native_result
                except Exception as exc:
                    item.update(status='raised', error_type=type(exc).__name__, error=str(exc)); raise
                finally:
                    item['wall_seconds'] = time.perf_counter()-started
                    report.setdefault('native_planning_calls', []).append(item)

        solvers = [AuditedSolver(capture, base_pose=a.robot.pose, multi_robot_id=i, debug=False, vis=False,
                    visualize_target_grasp_pose=False, print_env_info=False, joint_vel_limits=1.485, joint_acc_limits=2.574)
                   for i, a in enumerate(agents)]
        report['urdf_joint_velocity_checks'] = [compact_urdf_velocity_limits(a.urdf_path, arm.names, 1.5) for a, arm in zip(agents, arms)]
        checker = LiveGraspCollisionChecker.from_solvers(raw, *solvers, target_actor=banana, active_robot='right')

        def begin(name: str) -> None:
            capture.phase = report['stage'] = name; capture.target_phases = ('preparation', name)
            save_report(output/'report.json', report)

        def finish(name: str, result: dict, *, expected: str = 'passed') -> bool:
            report.setdefault('stages', []).append({'name': name, **result})
            okay = result.get('status') == expected
            if not okay: report.update(status='failed', reason=f'{name}:{result.get("status")}')
            save_report(output/'report.json', report); return okay

        @contextmanager
        def query_guard(label: str):
            before = _state_arrays(raw.get_state_dict())
            before['elapsed_steps'] = as_numpy(raw.elapsed_steps).copy()
            counts = (len(ticks), len(substeps), len(frames)); error = None
            try: yield
            except Exception as exc: error = f'{type(exc).__name__}: {exc}'; raise
            finally:
                after = _state_arrays(raw.get_state_dict()); after['elapsed_steps'] = as_numpy(raw.elapsed_steps).copy()
                unchanged = set(before) == set(after) and all(np.array_equal(before[k], after[k]) for k in before) and counts == (len(ticks), len(substeps), len(frames))
                proof = {'label': label, 'status': 'passed' if unchanged else 'failed', 'physical_state_unchanged': unchanged,
                         'environment_steps': 0 if unchanged else None, 'physics_steps': 0 if unchanged else None, 'query_exception': error,
                         'before_array_sha256': {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in before.items()},
                         'after_array_sha256': {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in after.items()}}
                report.setdefault('query_proofs', []).append(proof)
                save_report(output/'query-proofs.json', {'queries': report['query_proofs']})
                if not unchanged: raise QueryMutationError('Planning/query advanced physics or changed actual state')

        def ik_probe(index: int, pose: Any, label: str) -> bool:
            solver = solvers[index]; planner = solver.planner
            active_names = tuple(j.get_name() for j in solver.robot.get_active_joints())
            full = as_numpy(solver.robot.get_qpos())[0, [active_names.index(n) for n in planner.user_joint_names]].copy()
            mask = [i not in planner.move_group_joint_indices for i in range(len(full))]
            target = np.r_[pose.p, pose.q]
            item = {'label': label, 'arm_index': index, 'world_xyz_wxyz': target.tolist(), 'mask_non_arm': mask}
            started = time.perf_counter()
            try:
                with query_guard(label):
                    status, solutions = planner.IK(planner.transform_goal_to_wrt_base(target), full.copy(), mask=mask, n_init_qpos=4, threshold=1e-4)
                item.update(status=status, solutions=[np.asarray(q).tolist() for q in solutions])
                return status == 'Success' and bool(solutions)
            except QueryMutationError: raise
            except Exception as exc:
                item.update(status='raised', error_type=type(exc).__name__, error=str(exc)); return False
            finally:
                item['wall_seconds'] = time.perf_counter()-started; report.setdefault('ik_queries', []).append(item)
                save_report(output/'ik-queries.json', {'queries': report['ik_queries']})

        def plan(name: str, poses: tuple, grips: tuple[int, int]) -> Any:
            nonlocal call_number
            call_number += 1; label = f'query-{call_number:03d}-{name}'
            limits = handoff_phase_limits(name)
            compact_set_phase_limits(solvers, limits)
            with query_guard(label):
                source = plan_dual_pose(*solvers, poses, gripper_commands=grips)
                results = tuple(s.native_result if p is not None else None for s, p in zip(solvers, poses))
                traces = tuple(s.parameterization if p is not None else None for s, p in zip(solvers, poses))
                retimed, timing = compact_retime_pair(results, traces, velocity_limit=limits['nominal_velocity_limit_rad_s'],
                                                       acceleration_limit=limits['nominal_acceleration_limit_rad_s2'])
                inherited = issued_arm_targets(ticks, keys) if machine is not None else None
                trajectories = tuple((t if inherited is None else inherited_stationary_path(t, inherited[i]))
                                     if r is None else ArmTrajectory(t.action_key, t.joint_names, r['time'], r['position'], t.gripper_command)
                                     for i, (t, r) in enumerate(zip(source.trajectories, retimed)))
                schedule = build_hold_schedule(*trajectories, control_timestep_s=.05)
                arrays, audits = {}, []
                for i, result in enumerate(retimed):
                    if result is None:
                        audits.append({'status': 'passed', 'stationary_hold': True,
                            'target_policy': 'measured_entry' if inherited is None else 'inherit_last_issued_controller_command',
                            'collision_path_starts_at_actual_qpos': True,
                            'command_boundary_delta_rad': None if inherited is None else float(np.max(np.abs(
                                trajectories[i].positions_rad[-1]-inherited[i]))),
                            'actual_to_command_max_delta_rad': float(np.max(np.abs(
                                trajectories[i].positions_rad[-1]-trajectories[i].positions_rad[0])))})
                        continue
                    audit = compact_kinematic_audit(result, velocity_limit=limits['nominal_velocity_limit_rad_s'],
                                                    acceleration_limit=limits['nominal_acceleration_limit_rad_s2'])
                    audits.append(audit)
                    for field in ('time', 'position', 'velocity', 'acceleration', 'duration'):
                        arrays[f'arm{i}_raw_{field}'] = np.asarray(results[i][field])
                        arrays[f'arm{i}_retimed_{field}'] = np.asarray(result[field])
                np.savez(output/f'{label}-plan.npz', **arrays, left_time=trajectories[0].times_s,
                    right_time=trajectories[1].times_s, left_position=trajectories[0].positions_rad,
                    right_position=trajectories[1].positions_rad, target_times=schedule.target_times_s,
                    left_commands=schedule.left_commands, right_commands=schedule.right_commands)
                report.setdefault('planning', []).append({'label': label, 'phase': name, 'environment_steps': 0,
                    'scheduled_steps': schedule.num_steps, 'timing': timing, 'derivative_audits': audits,
                    'phase_limits': limits, 'world_tcp_targets': [None if p is None else np.r_[p.p, p.q].tolist() for p in poses]})
                require(all(a['status'] == 'passed' for a in audits), 'Native retimed derivative gate failed')
                return schedule

        def checked_candidate(name: str, poses: tuple, grips: tuple[int, int]) -> tuple[Any, dict]:
            row: dict = {'label': name}
            try:
                proposed = plan(name, poses, grips)
                with query_guard(name+'-combined-check'):
                    checker.refresh(); actual = measure_transfer(capture, banana)
                    row['validation'] = validate_schedule(proposed, actual.qpos_rad[0, :7], actual.qpos_rad[1, :7], checker, max_joint_step_rad=.02)
                row['status'] = row['validation']['status']
                return proposed if row['status'] == 'passed' else None, row
            except QueryMutationError: raise
            except Exception as exc:
                row.update(status='raised', error_type=type(exc).__name__, error=str(exc)); return None, row
            finally:
                report.setdefault('geometry_candidates', []).append(row)
                save_report(output/'geometry-candidates.json', {'candidates': report['geometry_candidates']})

        mesh = banana.get_first_collision_mesh(to_world_frame=False)
        require(mesh is not None, 'Cooked actor-local banana collision mesh required')
        selected_fraction = None; initial_grasp = None; setup_schedule = None; geometry = None
        for fraction in dict.fromkeys(c['anchor_fraction'] for c in candidates):
            receiver_fraction = fixed['receiver_anchor_fraction'] if fixed is not None else fraction
            geometry = anchor_geometry(mesh.vertices, fraction, receiver_fraction)
            object_matrix = as_numpy(banana.pose.to_transformation_matrix())[0]
            anchor = np.asarray(geometry['giver_receiver_anchor_local_m'][0])
            center = object_matrix[:3, :3]@anchor+object_matrix[:3, 3]
            info = compute_grasp_info_by_obb(get_actor_obb(banana), approaching=np.array([0., 0., -1.]),
                target_closing=as_numpy(agents[1].tcp.pose.to_transformation_matrix())[0, :3, 1], depth=.025)
            center[2] = info['center'][2]  # retain official25mm top-down grasp depth
            initial_grasp = agents[1].build_grasp_pose(np.array([0., 0., -1.]), info['closing'], center)
            pre = initial_grasp*sapien.Pose([0., 0., -.08]); label = f'initial-anchor-{fraction:.2f}'
            save_report(output/f'{label}-geometry.json', {'geometry': geometry,
                'initial_grasp_world_xyz_wxyz': np.r_[initial_grasp.p, initial_grasp.q].tolist(),
                'initial_pregrasp_world_xyz_wxyz': np.r_[pre.p, pre.q].tolist()})
            begin('pregrasp')
            if not ik_probe(1, initial_grasp, label+'-grasp-IK') or not ik_probe(1, pre, label+'-pregrasp-IK'): continue
            setup_schedule, _ = checked_candidate(label, (None, pre), (1, 1))
            if setup_schedule is not None: selected_fraction = fraction; break
        if selected_fraction is None:
            report.update(status='failed', reason='no_initial_anchor_pregrasp_candidate'); return
        report['anchor_geometry'] = geometry
        if not finish('pregrasp', execute_reach(capture, setup_schedule, checker, active_key=keys[1])): return
        begin('descend'); schedule = plan('descend', (None, initial_grasp), (1, 1))
        if not finish('descend', execute_reach(capture, schedule, checker, active_key=keys[1], allow_target_contact=True)): return
        begin('giver_close_and_confirm')
        if not finish('giver_close_and_confirm', close_and_confirm(capture, checker, active_key=keys[1], target_actor=banana, stable_steps=5)): return

        # Capture immediately after the original five physical grasp ticks, once.
        checker = TransferCollisionChecker.from_solvers(raw, *solvers, target_actor=banana, giver_index=1)
        machine = TransferStateMachine(1, checker.reference(1), receiver_max_steps=20, release_max_steps=30)
        machine.initialize(measure_transfer(capture, banana))
        require(machine.previous.fingers_open[0], 'Receiver must remain physically OPEN during donor pickup')
        initial_z = float(machine.previous.object_T_world[2, 3])
        report['transfer_capture'] = {'giver_capture_control_step': machine.previous.environment_step,
            'initial_object_origin_z_m': initial_z, 'giver_T_tcp_object': checker.reference(1).tolist(),
            'physical_attachment': False, 'unique_payload_actor': checker.target_actor_name, 'unique_payload_actor_count': 1}
        begin('giver_lift')
        lift = sapien.Pose(as_numpy(agents[1].tcp.pose.p)[0]+[0., 0., .12], as_numpy(agents[1].tcp.pose.q)[0])
        if not finish('giver_lift', execute_transfer_schedule(capture, plan('giver_lift', (None, lift), machine.command_grips()), checker, machine, episode_budget=BUDGET)): return
        gain = float(machine.previous.object_T_world[2, 3]-initial_z)
        report['actual_object_origin_lift_m'] = gain
        require(gain >= .10, 'Actual banana origin lift below10cm')
        support_latched |= bool(checker.active.support_released)
        require(support_latched, 'Actual lifted checker has not permanently left source support')
        machine.minimum_object_z_m = initial_z+.10

        def receiver_pose(object_matrix: np.ndarray, choice: dict) -> Any:
            """Use measured mesh/OBB depth for current or world-Z-yawed poses."""
            local = np.asarray(geometry['giver_receiver_anchor_local_m'][1])
            point = object_matrix[:3, :3]@local+object_matrix[:3, 3]
            axis = object_matrix[:3, :3]@np.asarray(geometry['long_axis_local'])
            canonical_closing, closing, approaching = receiver_grasp_axes(axis, choice['receiver_tilt_deg'])
            info = compute_grasp_info_by_obb(get_actor_obb(banana), approaching=np.array([0., 0., -1.]), target_closing=closing, depth=.025)
            actual_z = float(as_numpy(banana.pose.p)[0, 2])
            point[2] = info['center'][2]+object_matrix[2, 3]-actual_z+choice['receiver_height_offset_m']
            if args.case == 'receiver-failure': point += canonical_closing*NEGATIVE_RECEIVER_MISS_M
            return agents[0].build_grasp_pose(approaching, closing, point)

        def qualify_receiver(goal: np.ndarray, giver_schedule: Any, label: str, choice: dict) -> dict:
            """Screen both endpoint IKs before one physical transport attempt.

            This proves endpoint feasibility only. Actual receiver approach is
            freshly planned and checked after the measured physical transport.
            """
            grasp = receiver_pose(goal, choice)
            pre_pose = receiver_pregrasp_world_pose(np.r_[grasp.p, grasp.q])
            pre = sapien.Pose(pre_pose[:3], pre_pose[3:])
            result: dict = {'status': 'failed', 'endpoints': [], 'scope': 'private native IK and full paired endpoint geometry, not a receiver path certificate'}
            try:
                for phase, target, contacts in receiver_approach_stages(args.case, pre, grasp):
                    stage = 'grasp' if contacts else 'pregrasp'
                    endpoint = {'stage': stage, 'world_xyz_wxyz': np.r_[target.p, target.q].tolist(), 'checked_native_solutions': []}
                    result['endpoints'].append(endpoint)
                    if not ik_probe(0, target, label+'-receiver-'+stage+'-IK'):
                        result['reason'] = 'receiver_'+stage+'_IK_failed'; return result
                    indices = [solvers[0].planner.user_joint_names.index(n) for n in arms[0].names]
                    checker.set_phase('receiver_contact' if contacts else 'giver_hold')
                    with query_guard(label+'-receiver-'+stage+'-endpoint-check'):
                        for solution in report['ik_queries'][-1]['solutions']:
                            q = np.asarray(solution)[indices]
                            if not candidate_joint_limits_pass(q, arms[0].limits):
                                endpoint['checked_native_solutions'].append({'controller_qpos_rad': q.tolist(),
                                    'status': 'rejected_joint_limits', 'collisions': None})
                                continue
                            hits = checker.check(q, giver_schedule.trajectories[1].positions_rad[-1])
                            endpoint['checked_native_solutions'].append({'controller_qpos_rad': q.tolist(),
                                'status': 'passed' if not hits else 'rejected_collision', 'collisions': hits})
                    if not any(row['status'] == 'passed' for row in endpoint['checked_native_solutions']):
                        result['reason'] = 'no_valid_collision_free_receiver_'+stage+'_endpoint'; return result
                result['status'] = 'passed'; return result
            finally: checker.set_phase('giver_hold')

        selected = None; transport = None
        begin('giver_transport_candidate_search')
        search = report['transport_recipe_search'] = []
        for index, choice in enumerate(c for c in candidates if c['anchor_fraction'] == selected_fraction
                                      and c['receiver_anchor_fraction'] == geometry['receiver_anchor_fraction']):
            attempt = {'index': index, 'recipe': choice, 'status': 'rejected'}; search.append(attempt)
            goal = world_object_goal(machine.previous.object_T_world, geometry['long_axis_local'], choice)
            tcp_goal = goal@np.linalg.inv(checker.reference(1)); pose = sapien.Pose(tcp_goal[:3, 3], mat2quat(tcp_goal[:3, :3]))
            if goal[2, 3] < machine.minimum_object_z_m:
                attempt['reason'] = 'goal_below_frozen_actual_minimum_height'
                continue
            if not ik_probe(1, pose, f'transport-{index:02d}-giver-IK'):
                attempt['reason'] = 'giver_IK_failed'; continue
            transport, item = checked_candidate(f'transport-{index:02d}', (None, pose), machine.command_grips())
            item['recipe'] = choice; attempt['giver_route'] = item
            if transport is None: attempt['reason'] = 'giver_route_failed'; continue
            attempt['receiver_endpoint_screen'] = qualify_receiver(goal, transport, f'transport-{index:02d}', choice)
            if attempt['receiver_endpoint_screen']['status'] != 'passed':
                attempt['reason'] = 'receiver_endpoint_screen_failed'; continue
            attempt['status'] = 'selected'; selected = choice; break
        save_report(output/'transport-recipe-search.json', {'candidates': search,
            'selected_anchor_fraction': selected_fraction, 'declared_total_geometry_grid': len(candidates),
            'physical_search_attempts': 0, 'frozen_recipe': fixed is not None})
        if selected is None: report.update(status='failed', reason='no_readonly_qualified_transport_recipe'); return
        save_report(output/'selected-recipe.json', selected); report['selected_recipe'] = selected
        begin('giver_transport')
        if not finish('giver_transport', execute_transfer_schedule(capture, transport, checker, machine, episode_budget=BUDGET)): return

        # Rebuild receiver target from the CURRENT real object, not predicted goal.
        current = measure_transfer(capture, banana)
        receiver_grasp = receiver_pose(current.object_T_world, selected)
        receiver_pre_pose = receiver_pregrasp_world_pose(np.r_[receiver_grasp.p, receiver_grasp.q])
        receiver_pre = sapien.Pose(receiver_pre_pose[:3], receiver_pre_pose[3:])
        report['receiver_world_targets_xyz_wxyz'] = [np.r_[p.p, p.q].tolist() for p in (receiver_pre, receiver_grasp)]
        require(current.fingers_open[0], 'Receiver closed before approach')
        report['receiver_approach_stages'] = [s[0] for s in receiver_approach_stages(args.case, receiver_pre, receiver_grasp)]
        for name, target, contacts in receiver_approach_stages(args.case, receiver_pre, receiver_grasp):
            begin(name)
            checker.set_phase('receiver_contact' if contacts else 'giver_hold')
            if not ik_probe(0, target, name+'-IK'): report.update(status='failed', reason=name+':IK_failed'); return
            schedule = plan(name, (target, None), machine.command_grips())
            if not finish(name, execute_transfer_schedule(capture, schedule, checker, machine, episode_budget=BUDGET)): return
        begin('receiver_close'); machine.begin_receiver_close()
        closed = execute_transfer_hold(capture, checker, machine, episode_budget=BUDGET, max_steps=20,
                                       arm_targets_rad=issued_arm_targets(ticks, keys))
        report['stages'].append({'name': 'receiver_close', **closed})
        if args.case == 'receiver-failure':
            if machine.state != RECEIVER_FAILED:
                report.update(status='failed', reason='negative_fixture_did_not_produce_receiver_timeout',
                              giver_open_issued=False); return
            begin('safe_giver_hold'); machine.begin_safe_giver_hold()
            held = execute_transfer_hold(capture, checker, machine, episode_budget=BUDGET, max_steps=20,
                                         arm_targets_rad=issued_arm_targets(ticks, keys))
            if not finish('safe_giver_hold', held, expected=SAFE_COMPLETE): return
        else:
            if machine.state != RECEIVER_CONFIRMED:
                report.update(status='failed', reason='receiver_not_five_tick_confirmed'); return
            begin('giver_release'); machine.begin_giver_release()
            released = execute_transfer_hold(capture, checker, machine, episode_budget=BUDGET, max_steps=30,
                                             arm_targets_rad=issued_arm_targets(ticks, keys))
            if not finish('giver_release', released, expected=GIVER_RETREAT): return
            begin('giver_retreat')
            retreat_start = measure_transfer(capture, banana).tcp_T_world[1, :3, 3].copy()
            target = sapien.Pose(retreat_start+[0., 0., .085], as_numpy(agents[1].tcp.pose.q)[0])
            schedule = plan('giver_retreat', (None, target), machine.command_grips())
            if not finish('giver_retreat', execute_transfer_schedule(capture, schedule, checker, machine, episode_budget=BUDGET)): return
            report['actual_giver_retreat_displacement_m'] = (machine.previous.tcp_T_world[1, :3, 3]-retreat_start).tolist()
            machine.mark_retreat_complete(); begin('receiver_hold')
            held = execute_transfer_hold(capture, checker, machine, episode_budget=BUDGET, max_steps=20,
                                         arm_targets_rad=issued_arm_targets(ticks, keys))
            if not finish('receiver_hold', held, expected=COMPLETE): return
        report['transfer_held_physics_audit'] = held_substep_audit(substeps)
        require(report['transfer_held_physics_audit']['status'] == 'passed', 'Actual100Hz held slip envelope failed')
        confirmed_step = next((e['environment_step'] for e in machine.events if e['event'] == 'receiver_five_ticks_confirmed'), None)
        report['transfer_integration_audit'] = integration_audit(ticks, substeps,
            capture_step=report['transfer_capture']['giver_capture_control_step'],
            receiver_confirmed_step=confirmed_step, negative=args.case == 'receiver-failure', giver_key=keys[1])
        require(report['transfer_integration_audit']['status'] == 'passed', 'Controller/recorder handoff join failed')
        expected_captures = {1: 1, 0: 0 if args.case == 'receiver-failure' else 1}
        require(checker.capture_counts == expected_captures, 'Unexpected reference capture count')
        require(int(as_numpy(raw.elapsed_steps)[0]) < BUDGET, 'Completion must precede native400 truncation')
        report.update(status='passed', physical_capability_outcome='passed', stage='finished',
                      official_200_step_complete=False, basket_placement_success_claimed=False)
    finally:
        if machine is not None:
            report['transfer_state_machine'] = {'state': machine.state, 'reason': machine.reason, 'events': machine.events,
                'trace': machine.trace, 'hold_ticks': machine.hold_ticks, 'confirmation': machine.confirmation,
                'release_confirmation': machine.release_confirmation,
                'references': {str(k): v.tolist() for k, v in machine.references.items()}}
            save_report(output/'transfer-controller.json', report['transfer_state_machine'])
        if isinstance(checker, TransferCollisionChecker):
            report['transfer_collision'] = {'capture_counts': checker.capture_counts, 'owner_index': checker.owner_index,
                'events': checker.events, 'coverage': json_value(checker.coverage)}
        if substeps and machine is not None and 'transfer_held_physics_audit' not in report:
            report['transfer_held_physics_audit'] = held_substep_audit(substeps)
        finalize_physics_capture(env, raw, capture, output, report, frames, ticks, substeps, keys, original_hook)


def self_test() -> dict[str, Any]:
    from types import SimpleNamespace as NS
    pre_marker, grasp_marker = object(), object()
    assert receiver_approach_stages('positive', pre_marker, grasp_marker) == (
        ('receiver_pregrasp', pre_marker, False), ('receiver_descend', grasp_marker, True))
    assert receiver_approach_stages('receiver-failure', pre_marker, grasp_marker) == (
        ('receiver_pregrasp', pre_marker, False),)
    try: receiver_approach_stages('unknown', pre_marker, grasp_marker)
    except ValueError: pass
    else: raise AssertionError('Unknown receiver fixture accepted')
    # Pregrasp follows each quaternion's approach axis without input aliasing.
    quaternions = ([1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                   [np.sqrt(.5), 0., 0., np.sqrt(.5)], [.5, .5, .5, .5])
    offset_checks = 0
    for quaternion in quaternions:
        grasp = np.r_[[.12, -.04, .22], quaternion]; untouched = grasp.copy()
        pre = receiver_pregrasp_world_pose(grasp)
        w, x, y, z = quaternion
        expected = -.08*np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
        np.testing.assert_allclose(pre[:3]-grasp[:3], expected, rtol=0., atol=1e-15)
        assert abs(np.linalg.norm(pre[:3]-grasp[:3])-.08) < 1e-15
        assert np.array_equal(pre[3:], grasp[3:]) and np.array_equal(grasp, untouched)
        assert not np.shares_memory(pre, grasp)
        offset_checks += 1
    # Both case targets use this helper after their grasp/miss is constructed.
    for mode, miss in (('positive', [0., 0., 0.]), ('receiver-failure', [NEGATIVE_RECEIVER_MISS_M, 0., 0.])):
        grasp = np.r_[[.12, -.04, .22], quaternions[-1]]; grasp[:3] += miss
        np.testing.assert_allclose(receiver_pregrasp_world_pose(grasp),
                                   receiver_pregrasp_world_pose(grasp-np.r_[miss, np.zeros(4)])+np.r_[miss, np.zeros(4)],
                                   rtol=0., atol=1e-15)
        offset_checks += 1
    invalid_offsets = 0
    for bad in (np.zeros(6), np.zeros(8), np.zeros((1, 7)), np.zeros(7),
                np.array([0., 0., 0., 2., 0., 0., 0.]),
                np.array([np.nan, 0., 0., 1., 0., 0., 0.]),
                np.array([0., 0., 0., 1., 0., np.inf, 0.])):
        try: receiver_pregrasp_world_pose(bad)
        except ValueError: invalid_offsets += 1
        else: raise AssertionError('Invalid world pregrasp pose accepted')
    cfg = configuration()
    bounds = np.tile([-1., 1.], (7, 1))
    assert candidate_joint_limits_pass(np.zeros(7), bounds)
    assert candidate_joint_limits_pass(np.ones(7)+.5e-6, bounds)
    assert not candidate_joint_limits_pass(np.ones(7)+2e-6, bounds)
    assert not candidate_joint_limits_pass(-np.ones(7)-2e-6, bounds)
    for q, lim in ((np.zeros(6), bounds), (np.full(7, np.nan), bounds), (np.zeros(7), bounds[:6]),
                   (np.zeros(7), -bounds)):
        try: candidate_joint_limits_pass(q, lim)
        except ValueError: pass
        else: raise AssertionError('Invalid IK candidate/limits accepted')
    assert cfg['source_version'] == SOURCE_VERSION
    assert cfg['source_departure_policy'] == 'measured_dz_gt_3mm_and_all_geometry_clear_and_no_native_source_contact_v1'
    assert cfg['receiver_pregrasp_policy'] == RECEIVER_PREGRASP_POLICY
    assert cfg['receiver_pregrasp_distance_m'] == .08
    assert cfg['negative_receiver_miss_m'] == NEGATIVE_RECEIVER_MISS_M == .12
    assert cfg['negative_receiver_closing_pose'] == 'actual_checked_pregrasp_no_descent_v3'
    for axis, tilt in itertools.product(([1., 0., 0.], [0., 1., 0.], [.6, .8, .1]), RECEIVER_TILTS_DEG):
        canonical, closing, approaching = receiver_grasp_axes(axis, tilt)
        down = np.array([0., 0., -1.])
        frame = np.stack((np.cross(closing, approaching), closing, approaching), axis=1)
        np.testing.assert_allclose(frame.T@frame, np.eye(3), rtol=0, atol=1e-15)
        assert abs(np.linalg.det(frame)-1.) < 1e-15 and np.dot(approaching, down) == np.cos(np.deg2rad(tilt))
        assert np.array_equal(closing, canonical) and not np.shares_memory(closing, canonical)
        np.testing.assert_allclose(canonical*.12, np.cross(down, axis)/np.linalg.norm(np.cross(down, axis))*.12)
    for bad in ([0., 0., 1.], [np.nan, 0., 0.], [1., 0.], np.ones((1, 3))):
        try: receiver_grasp_axes(bad, 30.)
        except ValueError: pass
        else: raise AssertionError('Invalid receiver frame axis accepted')
    negative_shape_checks = 0
    for closing in ([1., 0., 0.], [0., -1., 0.], [.6, .8, 0.]):
        closing = np.asarray(closing)
        miss = closing*NEGATIVE_RECEIVER_MISS_M
        assert miss.shape == (3,) and abs(np.linalg.norm(miss)-.12) < 1e-15
        point = np.array([.12, -.04, .22]); shifted = point+miss
        assert shifted.shape == point.shape and np.isfinite(shifted).all()
        negative_shape_checks += 1
    assert cfg['max_episode_steps'] == 400 and cfg['receiver_grasp_stable_ticks'] == 5
    lift_limits, transport_limits = handoff_phase_limits('giver_lift'), handoff_phase_limits('giver_transport')
    assert lift_limits['nominal_velocity_limit_rad_s'] == 1.5/4
    assert lift_limits['nominal_acceleration_limit_rad_s2'] == 2.6/16
    assert transport_limits['nominal_velocity_limit_rad_s'] == 1.5
    assert transport_limits['nominal_acceleration_limit_rad_s2'] == 2.6
    assert handoff_phase_limits('receiver_descend')['nominal_velocity_limit_rad_s'] == .9
    names = tuple(f'q{i}' for i in range(7)); locked = np.arange(14).reshape(2, 7)*.01
    actions = {key: np.r_[row, -1.].tolist() for key, row in zip(('left', 'right'), locked)}
    previous = [{'actions': actions}]; targets = issued_arm_targets(previous, ['left', 'right'])
    targets[0, 0] += .1
    assert actions['left'][0] == 0.  # no alias into raw evidence
    paths = tuple(inherited_stationary_path(ArmTrajectory(key, names, [0.], [row+.005], -1.), row)
                  for key, row in zip(('left', 'right'), locked))
    held_schedule = build_hold_schedule(*paths, control_timestep_s=.05)
    assert np.array_equal(paths[0].positions_rad[0], locked[0]+.005)
    assert np.array_equal(held_schedule.left_commands[0, :7], locked[0])
    assert np.array_equal(held_schedule.right_commands[0, :7], locked[1])
    for bad in (np.zeros(6), np.full(7, np.nan), np.zeros((1, 7))):
        try: inherited_stationary_path(paths[0], bad)
        except ValueError: pass
        else: raise AssertionError('Invalid inherited target accepted')
    entries = candidate_recipes(); assert len(entries) == 810 and len(candidate_recipes(entries[0])) == 1
    for seed in DEVELOPMENT_SEEDS:
        validate_scope(seed, True, None); validate_scope(seed, False, entries[0])
    for seed in ACCEPTANCE_SEEDS: validate_scope(seed, False, entries[0])
    rejected = 0
    for callback in (lambda: validate_scope(350, True, None), lambda: validate_scope(354, True, None),
                     lambda: validate_scope(309, False, entries[0]), lambda: validate_scope(315, False, entries[0]),
                     lambda: validate_scope(5, False, entries[0]), lambda: validate_scope(True, False, entries[0]),
                     lambda: validate_scope(0, False, None), lambda: validate_scope(310, False, None),
                     lambda: validate_scope(0, True, entries[0]),
                     lambda: recipe({**entries[0], 'anchor_fraction': .4}),
                     lambda: recipe({**entries[0], 'receiver_anchor_fraction': .4}),
                     lambda: recipe({**entries[0], 'negative_receiver_miss_m': .08}),
                     lambda: anchor_geometry(np.zeros((8, 3)), .25),
                     lambda: anchor_geometry(np.full((8, 3), np.nan), .25)):
        try: callback()
        except ValueError: rejected += 1
        else: raise AssertionError('Invalid geometry/scope boundary accepted')
    x = np.linspace(-.10, .10, 64)
    vertices = np.array([[s, .5*s*s+side*.012, height*.01] for s in x for side in (-1, 1) for height in (-1, 1)])
    geometry = anchor_geometry(vertices, .25)
    assert geometry['anchor_separation_m'] > .08
    asymmetric = anchor_geometry(vertices, .20, .30)
    assert asymmetric['receiver_anchor_fraction'] == .30 and asymmetric['giver_anchor_fraction'] == .20
    assert asymmetric['anchor_separation_m'] > anchor_geometry(vertices, .20)['anchor_separation_m']
    transform = world_object_goal(np.eye(4), geometry['long_axis_local'], entries[1])
    assert np.allclose(transform[:3, 3], CENTRES[0]) and np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3))
    banana = {'kind': 'payload', 'target_index': 1}; robot = {'kind': 'robot', 'robot_index': 1, 'link_name': 'panda_leftfinger'}
    row = {'bodies': [banana, robot], 'class': 'own_finger_allowed', 'points': [{'separation_m': -.001}], 'physical_contact': True}
    live = NS(active_robot='right', target_contact_allowed=False)
    assert handoff_contact_policy(row, 'pregrasp', live) == 'forbidden'
    live.target_contact_allowed = True
    assert handoff_contact_policy(row, 'descend', live) == 'handoff_finger_allowed'
    assert handoff_contact_policy({**row, 'bodies': [banana, {**robot, 'link_name': 'panda_hand'}]}, 'descend', live) == 'forbidden'
    transfer = TransferCollisionChecker.__new__(TransferCollisionChecker)
    transfer._owned = {0: NS(_transfer_finger_robots=frozenset(('left',)))}; transfer.owner_index = 0
    assert handoff_contact_policy(row, 'receiver_hold', transfer) == 'forbidden'  # original own-right exemption is revoked
    assert handoff_contact_policy({**row, 'bodies': [banana, {**robot, 'robot_index': 0}]}, 'receiver_hold', transfer) == 'handoff_finger_allowed'
    assert all(handoff_contact_policy({**row, 'bodies': [{'kind': 'payload', 'target_index': 0},
        {**robot, 'robot_index': index}]}, 'receiver_hold', transfer) == 'forbidden' for index in (0, 1))
    assert handoff_contact_policy({**row, 'bodies': [banana, {'kind': 'actor', 'actor': 'table'}], 'class': 'forbidden'}, 'receiver_hold', transfer) == 'forbidden'
    observation = {'tcp_T_world': np.tile(np.eye(4), (2, 1, 1)).tolist(), 'object_T_world': np.eye(4).tolist(),
                   'guard_role_index': 1, 'captured_relations': {'1': np.eye(4).tolist()}, 'grasp_flags': [False, True]}
    assert held_substep_audit([{'substep': 1, 'transfer': observation}])['status'] == 'passed'
    corrupt = json.loads(json.dumps(observation)); corrupt['object_T_world'][0][3] = .00501
    assert held_substep_audit([{'substep': 1, 'transfer': corrupt}])['status'] == 'failed'
    assert held_substep_audit([])['status'] == 'failed'
    fake_ticks = [{'environment_step': i, 'actions': {'right': [0.]*7+[1. if i == 3 else -1.]},
                   'requested_actions': {'right': [0.]*7+[1. if i == 3 else -1.]}} for i in (1, 2, 3)]
    fake_substeps = [{'substep': i+1, 'control_step': 1+i//5, 'sim_timestep_s': .01, 'transfer': {}} for i in range(15)]
    assert integration_audit(fake_ticks, fake_substeps, capture_step=0, receiver_confirmed_step=2, negative=False, giver_key='right')['status'] == 'passed'
    assert integration_audit(fake_ticks, fake_substeps, capture_step=0, receiver_confirmed_step=3, negative=False, giver_key='right')['status'] == 'failed'
    assert integration_audit(fake_ticks, fake_substeps, capture_step=0, receiver_confirmed_step=None, negative=True, giver_key='right')['status'] == 'failed'
    assert integration_audit(fake_ticks, fake_substeps[:-1], capture_step=0, receiver_confirmed_step=2, negative=False, giver_key='right')['status'] == 'failed'
    return {'status': 'passed', 'source_version': SOURCE_VERSION,
            'receiver_pregrasp_policy': RECEIVER_PREGRASP_POLICY,
            'receiver_pregrasp_distance_m': RECEIVER_PREGRASP_DISTANCE_M,
            'receiver_world_offset_positive_checks': offset_checks,
            'receiver_world_offset_invalid_poses_rejected': invalid_offsets,
            'negative_receiver_miss_m': NEGATIVE_RECEIVER_MISS_M,
            'negative_receiver_closing_pose': 'actual_checked_pregrasp_no_descent_v3',
            'receiver_stage_boundary_checks': 3,
            'negative_miss_offset_shape_checks': negative_shape_checks,
            'finite_recipe_count': len(entries), 'invalid_boundaries_rejected': rejected,
            'scope_positive_cases': 2*len(DEVELOPMENT_SEEDS)+len(ACCEPTANCE_SEEDS), 'development_seed_pool': list(DEVELOPMENT_SEEDS),
            'gentle_liftoff_limits': lift_limits, 'inherited_command_contract_checks': 8,
            'independent_acceptance_seed_pool': list(ACCEPTANCE_SEEDS),
            'contact_policy_checks': 7, 'held_audit_checks': 3, 'recorder_controller_join_checks': 4, 'unique_payload_actor_count': 1,
            'shapes': {'canonical_action': [2, 8], 'unique_payload_force_vectors': [2, 2, 3], 'object_world_transform': [4, 4]},
            'environment_steps': 0, 'physics_steps': 0, 'native_integration': 'NOT_RUN',
            'scope': 'CPU geometry, contact-role, seed-scope and actual-slip boundary contracts; no physics claim'}


def main() -> int:
    faulthandler.enable(); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true'); parser.add_argument('--seed', type=int, default=0)
    selector = parser.add_mutually_exclusive_group(); selector.add_argument('--develop', action='store_true'); selector.add_argument('--recipe', type=Path)
    parser.add_argument('--case', choices=('positive', 'receiver-failure'), default='positive')
    parser.add_argument('--output-dir', type=Path); args = parser.parse_args()
    if args.self_test: print(json.dumps(self_test(), indent=2)); return 0
    signal.signal(signal.SIGTERM, termination_handler)
    fixed = None if args.recipe is None else recipe(json.loads(args.recipe.read_text()))
    validate_scope(args.seed, args.develop, fixed)
    if args.output_dir is None: parser.error('Unused explicit output directory required')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+', os.environ.get('CUDA_VISIBLE_DEVICES', '')): parser.error('Root-assigned GPU UUID required')
    if any(os.environ.get(k) == '1' for k in ('MANI_SKILL_L1', 'MANI_SKILL_L2', 'MANI_SKILL_L3')): parser.error('L0 only')
    args.mode = 'positive'  # shared recorder does not inject task-release faults
    output = args.output_dir; output.mkdir(parents=True, exist_ok=False)
    guard = json.loads((EVIDENCE/'prechange-guard.json').read_text())
    old = {Path(p): identity for p, identity in guard['baseline_source_sha256'].items()}
    require(all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in old.items()), 'Frozen baseline43 identity differs')
    files = list(old)+[Path(__file__), HERE/'transfer_execution.py', HERE/'transfer_collision.py',
                      HERE/'multitask_recording.py', HERE/'multitask_adapters.py']
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    snapshot = output/'source'; snapshot.mkdir()
    for p in files: (snapshot/p.name).write_bytes(p.read_bytes())
    report = {'status': 'running', 'native_integration': 'STARTED', 'started_at': datetime.now(timezone.utc).isoformat(),
        'task': TASK, 'method': 'single-banana-handoff', 'case': args.case, 'mode': 'positive', 'seed': args.seed,
        'development_only': args.seed in DEVELOPMENT_SEEDS, 'geometry_search': bool(args.develop), 'fixed_recipe': fixed,
        'seed_scope': 'development' if args.seed in DEVELOPMENT_SEEDS else 'independent_acceptance',
        'independent_acceptance_seed': args.seed if args.seed in ACCEPTANCE_SEEDS else None,
        'execution_configuration': configuration(), 'stages': [], 'source_snapshot': str(snapshot),
        'source_sha256_by_absolute_path': sources, 'baseline43_source_sha256': {str(p): h for p, h in old.items()},
        'official_default_max_episode_steps': 200, 'max_episode_steps': BUDGET,
        'engineering_capability_budget_override': BUDGET, 'official_200_step_complete': False,
        'physical_pose_assignment': False, 'physical_attachment_created': False, 'collision_masks_modified': False,
        'reserved_acceptance_seed_pool': {'seeds': list(ACCEPTANCE_SEEDS), 'status': 'EXTERNAL_PARENT_ALLOCATION_REQUIRED'},
        'success_scope': configuration()['success_scope']}
    save_report(output/'report.json', report)
    save_report(output/'repro.json', {'argv': sys.argv, 'python': sys.executable, 'cwd': str(Path.cwd()),
        'GPU_UUID': os.environ.get('CUDA_VISIBLE_DEVICES'), 'source_sha256_by_absolute_path': sources,
        'native_physics': 'cpu', 'native_render': 'gpu', 'control_hz': 20, 'physics_hz': 100,
        'max_episode_steps': BUDGET, 'geometry_candidate_grid': candidate_recipes(fixed)})
    start = time.monotonic()
    try: native_run(args, output, report)
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-start, native_integration='RUN')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in sources.items()):
        report.update(status='failed', source_identity_violation=True)
    save_report(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output/'report.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
