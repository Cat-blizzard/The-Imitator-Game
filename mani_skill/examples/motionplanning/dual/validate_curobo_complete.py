"""Complete paired TIG execution using cuRobo or MPLib through one executor.

Canonical left/right absolute joint actions are [2,8], arm samples [N,7],
world/base poses XYZ/WXYZ in metres, and native timestamps are seconds.
Planning never steps physics. Both backends retain their original native
parameterization and pass the same external collision and measured stability
gates. Standard RecordEpisode retains submitted actions and T+1 states.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback
from typing import Any

import numpy as np

from .curobo_backend import pose_error, trajectory_from_result
from .curobo_scene import _matrix, _pose, _shape_mesh
from .curobo_task_payload import SelectedWorkerProcess
from .initial_scene_manifest import save_initial_scene_manifest
from .trajectory_execution import as_numpy, bind_arm
from .trajectory_schedule import ArmTrajectory, build_hold_schedule
from .runtime_policy import LEGACY_STRICT, RuntimePolicy, policy_from_report, resolved_policy_record
from infrastructure.tig_runtime import resolve_runtime_paths
from .runtime_io import git_identity, save_json, json_value, termination_handler, state_digest
from .physical_observation import (check_fk, verify_finger_spheres, audit_payload_spheres,
    install_full_robot_contact_capture, held_substep_audit, make_checker, measured_source_departure)
from .phase_kinematics import (SOLVER_LIMIT_FRACTION, compact_kinematic_audit,
    compact_phase_limits, compact_retime_pair, compact_route_positions,
    compact_set_phase_limits, compact_topp_reconstruction)

RUNTIME_PATHS = resolve_runtime_paths()
ROOT = RUNTIME_PATHS.workspace_root
CODE_ROOT = RUNTIME_PATHS.code_root
HERE = Path(__file__).resolve().parent
TASKS = ('TwoRobotPickAppleBananaToBaskets-v1', 'TwoRobotPickPillToRegions-v1',
         'TwoRobotPickFruitsToPlate-v1')
LAST_COMPLETION_STEP = LEGACY_STRICT.last_completion_step
FINAL_STABLE_STEPS = LEGACY_STRICT.final_stable_steps
MINIMUM_SETTLE_BUDGET = LEGACY_STRICT.minimum_settle_budget
APPROACH_RECIPES = ('standard-8cm', 'compact-4cm', 'compact-5cm')
PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1', 'fixed-task-v2', 'fixed-task-v3', 'fixed-task-v4')
TRANSPORT_RECIPES = ('standard', 'pill-direct-release-v1', 'direct-release-v2')
# Existing executors require three stable endpoint ticks, five stable grasp
# ticks, and three released ticks. These are lower bounds, never promises that
# a future trajectory will fit. The known final placement window is reserved
# before the retreat schedule is allowed to run.
FUTURE_MINIMUM_TICKS = {'pregrasp': 43, 'descend': 40, 'lift_and_hold': 32,
    'carry_orient_and_transport': 29, 'carry_lower': 26, 'retreat': 20}


def execution_configuration(task_id: str = TASKS[0], *,
                            approach_recipe: str = 'compact-4cm',
                            payload_refinement: str = 'baseline',
                            transport_recipe: str = 'standard',
                            policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    """The same fixed external phase limits and diagnostic policy for both backends."""
    from .multitask_adapters import task_execution_parameters
    parameters = task_execution_parameters(task_id)
    if approach_recipe not in APPROACH_RECIPES:
        raise ValueError('Expected an explicit fixed approach recipe')
    if payload_refinement not in PAYLOAD_REFINEMENTS:
        raise ValueError('Expected an explicit fixed payload representation policy')
    if transport_recipe not in TRANSPORT_RECIPES:
        raise ValueError('Expected an explicit fixed transport recipe')
    if transport_recipe == 'pill-direct-release-v1' and task_id != TASKS[1]:
        raise ValueError('The direct-release recipe is confined to the pill task')
    return {'joint_vel_limit_rad_s': parameters['joint_vel_limit'],
        'joint_acc_limit_rad_s2': parameters['joint_acc_limit'],
        'approach_joint_vel_limit_rad_s': parameters['approach_vel_limit'],
        'approach_joint_acc_limit_rad_s2': parameters['approach_acc_limit'],
        'solver_limit_fraction': SOLVER_LIMIT_FRACTION,
        'preparation_steps': 10, 'max_episode_steps': policy.max_episode_steps,
        'runtime_policy': resolved_policy_record(policy),
        'airborne_hover_steps': 0, 'final_common_placement_steps': policy.final_stable_steps,
        'retreat_command_m': policy.retreat_command_m, 'minimum_actual_retreat_m': policy.minimum_retreat_m,
        'min_actual_height_gain_m': [policy.min_lift_m, policy.min_lift_m],
        'route': 'merge-lift-direct-release' if transport_recipe != 'standard' else 'merge-lift',
        'transport_recipe': transport_recipe,
        'transport_phase_sequence': ['carry_lower'] if transport_recipe != 'standard' else [
            'carry_orient_and_transport','carry_lower'],
        'execution_profile': 'execution',
        'diagnostic_hover_removed': 'two development five-tick windows; continuous100Hz held monitoring remains',
        'approach_recipe': approach_recipe,
        'payload_refinement': payload_refinement,
        'payload_refinement_scope': (
            'cuRobo native payload and both fruit-arm panda_link7 conservative sphere geometry only; physical model, control and acceptance thresholds unchanged'
            if payload_refinement in ('fixed-task-v3','fixed-task-v4') else
            'cuRobo native payload representation only; physical model, control and acceptance thresholds unchanged'),
        'pregrasp_distance_m': {'compact-4cm':.04,'compact-5cm':.05}.get(approach_recipe),
        'pregrasp_contact_permission': False, 'descend_contact_permission': 'corresponding own two fingers only',
        'approach_geometry_policy': 'fixed recipe with measured whole-end-effector positive-Z clearance check; same goals for both backends',
        'task_execution_parameters': parameters,
        'source_timestamps': 'cuRobo unchanged; MPLib original analytic common-phase retiming',
        'completion_boundary': 'final elapsed_steps <200; official registration horizon=200'}


def selected_payload_refinement(args: argparse.Namespace) -> str:
    """Old held-runner callers retain the baseline native representation."""
    policy = getattr(args,'payload_refinement','baseline')
    if policy not in PAYLOAD_REFINEMENTS:
        raise ValueError('Unknown payload refinement policy')
    return policy


def selected_transport_recipe(args: argparse.Namespace) -> str:
    """Old callers retain the original two-stage transport recipe."""
    policy = getattr(args,'transport_recipe','standard')
    if policy not in TRANSPORT_RECIPES:
        raise ValueError('Unknown fixed transport recipe')
    if policy == 'pill-direct-release-v1' and getattr(args,'task',None) != TASKS[1]:
        raise ValueError('The direct-release recipe is confined to the pill task')
    return policy


def configured_payload_snapshot(snapshot: dict[str,Any], policy: str, geometry_manifest: dict[str,Any] | None = None) -> dict[str,Any]:
    """Keep baseline untouched; optionally refine only the complete native snapshot."""
    if policy == 'baseline':
        return snapshot
    if policy == 'fixed-task-v1':
        from .curobo_payload_refinement import refine_task_snapshot
        return refine_task_snapshot(snapshot)
    if policy == 'fixed-task-v2':
        from .curobo_payload_refinement_v2 import refine_task_snapshot_v2
        return refine_task_snapshot_v2(snapshot)
    if policy == 'fixed-task-v3':
        from .curobo_fruit_link7_v3 import refine_task_snapshot_v3
        return refine_task_snapshot_v3(snapshot,geometry_manifest)
    if policy == 'fixed-task-v4':
        from .curobo_fruit_link7_v4 import refine_task_snapshot_v4
        return refine_task_snapshot_v4(snapshot,geometry_manifest)
    raise ValueError('Unknown payload refinement policy')


def future_minimum_ticks(policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, int]:
    """Legacy remaining phase windows derived from the selected resolved policy."""
    reach, grasp, release, final = (policy.reach_stable_steps, policy.grasp_stable_steps,
        policy.release_stable_steps, policy.final_stable_steps)
    return {'pregrasp': grasp + release + 5 * reach + final,
        'descend': grasp + release + 4 * reach + final,
        'lift_and_hold': release + 3 * reach + final,
        'carry_orient_and_transport': release + 2 * reach + final,
        'carry_lower': release + reach + final, 'retreat': final}


def phase_budget(before: int, scheduled: int, phase: str, *,
                 transport_recipe: str = 'standard',
                 policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    """Reserve a measured phase's exact schedule plus future mandatory tick minima."""
    if (type(before) is not int or type(scheduled) is not int or before < 0
            or scheduled < 1 or phase not in FUTURE_MINIMUM_TICKS
            or transport_recipe not in TRANSPORT_RECIPES):
        raise ValueError('Known phase, nonnegative elapsed steps and positive schedule length required')
    reserve = future_minimum_ticks(policy)[phase]
    # A direct route has no intermediate carry_orient_and_transport stop.
    # Reserve only the phases that will actually execute, while retaining the
    # original three stable endpoint ticks for every remaining movement.
    removed = (policy.reach_stable_steps if transport_recipe != 'standard'
               and phase in ('pregrasp','descend','lift_and_hold') else 0)
    reserve -= removed
    available = policy.last_completion_step - before - scheduled - reserve
    receipt = {'elapsed_before': before, 'scheduled_steps': scheduled,
        'transport_recipe': transport_recipe, 'removed_intermediate_minimum_ticks': removed,
        'reserved_future_minimum_steps': reserve,
        'last_permitted_completion_step': policy.last_completion_step,
        'available_current_settle_steps': available,
        'settle_max_steps': min(policy.max_settle_steps, max(0, available))}
    if available < policy.minimum_settle_budget:
        raise RuntimeError('Remaining official200 budget cannot fit the preserved schedule and '
                           f'at least{policy.minimum_settle_budget} endpoint-settle ticks plus mandatory tail: {receipt}')
    return receipt


def approach_pairs(grasp_pairs: Any, recipe: str) -> tuple[tuple[Any,Any],tuple[Any,Any]]:
    """Keep each exact grasp pose and move only the fixed pregrasp waypoint."""
    import sapien
    if recipe not in APPROACH_RECIPES or len(grasp_pairs) != 2 or any(len(p) != 2 for p in grasp_pairs):
        raise ValueError('Expected two pregrasp/grasp pairs and a fixed recipe')
    if recipe == 'standard-8cm':
        return tuple(tuple(pair) for pair in grasp_pairs)
    distance = {'compact-4cm':.04,'compact-5cm':.05}[recipe]
    result = []
    for _,grasp in grasp_pairs:
        transform = _matrix(np.r_[grasp.p,grasp.q])
        if not np.allclose(transform[:3,2],[0.,0.,-1.],rtol=0,atol=1e-6):
            raise ValueError('The compact recipe requires the task adapter downward grasp orientation')
        pre = sapien.Pose(np.asarray(grasp.p,dtype=float)+[0.,0.,distance],grasp.q)
        result.append((pre,grasp))
    return tuple(result)


def approach_endpoint_geometry(raw: Any, adapter: Any, pairs: Any) -> list[dict[str,Any]]:
    """Bound the complete open end effector above each target at the pregrasp endpoint.

    Current actual fingers define TCP-relative link transforms. Every shape on
    the fingers, palm and camera contributes its complete scaled solid vertices
    or conservative primitive proxy. The fixed waypoint still requires native
    planning, external full-scene checks and measured100Hz contact monitoring.
    """
    rows = []
    names = {'panda_leftfinger','panda_rightfinger','panda_hand','camera_link'}
    for index,(agent,target,pair) in enumerate(zip(raw.agent.agents,adapter.targets,pairs)):
        tcp = _matrix(agent.tcp.pose)
        goal = _matrix(np.r_[pair[0].p,pair[0].q])
        minima,found,shapes = [],set(),0
        for link in agent.robot.get_links():
            name = link.get_name()
            if name not in names:
                continue
            found.add(name)
            link_goal = goal @ np.linalg.inv(tcp) @ _matrix(link.pose)
            if len(link._objs) != 1:
                raise ValueError('One actual physical end-effector instance required')
            for shape in link._objs[0].get_collision_shapes():
                vertices,_,_ = _shape_mesh(shape)
                transform = link_goal @ _matrix(shape.get_local_pose())
                world = vertices @ transform[:3,:3].T + transform[:3,3]
                minima.append(float(world[:,2].min()))
                shapes += 1
        if found != names or not minima:
            raise ValueError('Incomplete physical end-effector collision geometry')
        target_world = adapter._vertices(target)
        maximum = float(target_world[:,2].max())
        gap = min(minima)-maximum
        row = {'arm_index':index,'end_effector_collision_shapes':shapes,
            'measured_open_end_effector_minimum_z_m':min(minima),
            'measured_target_maximum_z_m':maximum,'positive_vertical_separation_m':gap,
            'endpoint_only':True,'full_scene_and_path_checks_still_required':True}
        rows.append(row)
        if not math.isfinite(gap) or gap <= 0:
            raise ValueError(f'Selected pregrasp recipe lacks complete open-end-effector Z clearance: {row}')
    return rows




def source_departure_evidence(checker: Any) -> dict[str,Any]:
    """Preserve original release flags and measured receipt alongside adapter semantics."""
    effective = measured_source_departure(checker)
    supports = checker._supports
    return {'effective_measured_departure':list(effective),
        'original_release_flags':[bool(s.support_released) for s in supports],
        'initial_support_exists':[getattr(s,'_initial_support_exists',None) for s in supports],
        'actual_measured_receipts':[json_value(getattr(s,'_support_state_receipt',
            getattr(s,'_state_receipt',None))) for s in supports],
        'original_support_flags_or_collision_permissions_modified':False,
        'adapter_departure_is_latched':False,
        'proof_scope':'original monotone released flag or fresh never-registered full-convex separation',
        'physical_recorder_separate_monotone_gate':'measured lift>=3mm and no native source contact'}


def audit_task_pair(updates: list[dict[str,Any]], snapshots: Any) -> list[dict[str,Any]]:
    """Verify task-specific permissions and every actual world inverse pose."""
    from scipy.spatial.transform import Rotation
    from .curobo_task_payload import audit_task_native_state
    rows = []
    for receipt,data in zip(updates,snapshots):
        row = audit_task_native_state(data,receipt)
        world = receipt['native_update']['native_readback']['world']
        disabled = set(data['payload_model']['target_obstacle_names']) if data['payload_state']['attachment'] else set()
        maximum,count = 0.,0
        for kind,shapes in data['world'].items():
            if set(world[kind]) != set(shapes):
                raise RuntimeError('Native task world coverage differs from actual export')
            for name,shape in shapes.items():
                pose = np.asarray(shape['pose'],dtype=float)
                rotation = Rotation.from_quat(np.r_[pose[4:],pose[3]]).as_matrix()
                expected = np.r_[-rotation.T@pose[:3],pose[3],-pose[4:]]
                actual = world[kind][name]
                error = pose_error(actual['inverse_pose_xyz_wxyz'],expected)
                if (actual['enabled'] != int(name not in disabled)
                        or error['position_m'] > 1e-5 or error['rotation_rad'] > 1e-4):
                    raise RuntimeError(f'Incorrect actual native task obstacle state for {name}: {error}')
                maximum = max(maximum,error['position_m'])
                count += 1
        rows.append({'session_id':receipt['session_id'],'max_inverse_pose_error_m':maximum,
            'world_shapes':count,**row})
    return rows


def audit_session_identity(initial: dict[str,Any], current: dict[str,Any]) -> dict[str,Any]:
    """Require the same four planners and one initialization/warmup per session."""
    before,after = initial['sessions'],current['sessions']
    if len(before) != 4 or set(before) != set(after):
        raise RuntimeError('The four persistent physical-arm/profile sessions changed')
    for name,value in after.items():
        if (value['planner_id'] != before[name]['planner_id'] or value['init_count'] != 1
                or value['warmup_count'] != 1 or value['rebuild_count'] != 0):
            raise RuntimeError(f'Planner {name} was rebuilt, replaced or warmed again')
    return {'status':'passed','persistent_sessions':4,'rebuilds':0,'additional_warmups':0}


def enforce_pair_validation(validation: dict[str,Any], *, require_safe_pair: bool = True) -> None:
    """Explicit diagnostic hooks may inspect rejected candidates; executors still recheck."""
    if type(require_safe_pair) is not bool:
        raise ValueError('Pair-validation policy must be an explicit boolean')
    if validation.get('status') != 'passed' and require_safe_pair:
        raise RuntimeError('Paired exact collision validation rejected candidate')




def make_solvers(env: Any, output: Path, report: dict[str, Any], capture: Any) -> list[Any]:
    """Build original Panda planners; preserve MPLib TOPP input/output before conversion."""
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    class AuditedSolver(PandaArmMotionPlanningSolver):
        def setup_planner(self) -> Any:
            planner = super().setup_planner()
            native_topp = planner.TOPP
            topp_calls = 0

            def traced_topp(path: Any, step: float = .1, verbose: bool = False) -> Any:
                nonlocal topp_calls
                topp_calls += 1
                raw = native_topp(path, step, verbose)
                payload = {'native_path': np.asarray(path).copy(),
                    'solver_velocity_limits': np.asarray(planner.joint_vel_limits).copy(),
                    'solver_acceleration_limits': np.asarray(planner.joint_acc_limits).copy()}
                for field, value in zip(('time','position','velocity','acceleration','duration'), raw):
                    if value is not None:
                        payload['raw_'+field] = np.asarray(value).copy()
                np.savez(output/f'{capture.phase}-arm{self.multi_robot_id}-native-topp-call{topp_calls:03d}-raw.npz', **payload)
                self.last_native_parameterization = compact_topp_reconstruction(path,
                    planner.joint_vel_limits, planner.joint_acc_limits, step, raw)
                return raw

            planner.TOPP = traced_topp
            return planner

        def move_to_pose_with_screw(self, pose: Any, dry_run: bool = False,
                refine_steps: int = 0, other_gripper_state: Any = None) -> Any:
            if not dry_run:
                raise ValueError('Complete runner plans only; common executor advances physics')
            self.last_native_parameterization = None
            started = time.perf_counter()
            receipt = {'phase': capture.phase, 'arm_index': int(self.multi_robot_id)}
            try:
                result = super().move_to_pose_with_screw(pose, dry_run=True,
                    refine_steps=refine_steps, other_gripper_state=other_gripper_state)
                self.last_native_result = result
                save_json(output/f'{capture.phase}-arm-{self.multi_robot_id}-native-result.json', json_value(result))
                receipt['status'] = 'returned'
                return result
            except BaseException as exc:
                receipt.update(status='raised', error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                receipt['wall_seconds'] = time.perf_counter()-started
                report.setdefault('native_planning_calls', []).append(receipt)

    return [AuditedSolver(env, base_pose=agent.robot.pose, multi_robot_id=i,
        joint_vel_limits=.9, joint_acc_limits=.9, debug=False, vis=False,
        visualize_target_grasp_pose=False, print_env_info=False)
        for i, agent in enumerate(env.unwrapped.agent.agents)]




def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any], *,
               phase_hook: Any = None, transport_recipe: Any = None) -> None:
    transport_policy = selected_transport_recipe(args)
    if report['execution_configuration'].get('transport_recipe','standard') != transport_policy:
        raise ValueError('Transport selection differs from the common execution configuration')
    if transport_policy == 'pill-direct-release-v1':
        from .transport_recipes import pill_direct_transport_recipe
        if transport_recipe is None:
            transport_recipe = pill_direct_transport_recipe
        elif transport_recipe is not pill_direct_transport_recipe:
            raise ValueError('The fixed pill recipe requires its reviewed route selector')
    elif transport_policy == 'direct-release-v2':
        from .transport_routes import direct_release_stages
        if transport_recipe is None:
            transport_recipe = direct_release_stages
        elif transport_recipe is not direct_release_stages:
            raise ValueError('The direct-release-v2 recipe requires its declared route selector')
    elif transport_recipe is not None:
        raise ValueError('A transport callback requires an explicit fixed recipe')
    policy = policy_from_report(report)
    report['runtime_policy'] = resolved_policy_record(policy)
    report['transport_recipe'] = transport_policy
    payload_refinement = selected_payload_refinement(args)
    if report['execution_configuration'].get('payload_refinement','baseline') != payload_refinement:
        raise ValueError('Native representation selection differs from the common execution configuration')
    report['payload_refinement'] = payload_refinement
    import gymnasium as gym
    import sapien
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from .combined_collision import validate_schedule
    from .curobo_task_payload import CuroboTaskPayloadSessions, export_task_payload_snapshot
    from .dual_arm_execution import (plan_dual_pose, execute_dual_reach,
        close_dual_and_confirm, execute_dual_held_move, release_dual_and_confirm,
        verify_dual_placement)
    from .multitask_adapters import get_task_adapter
    from .multitask_recording import make_physics_capture, finalize_physics_capture, save_task_geometry
    from .runtime_io import _state_arrays

    env = raw = capture = worker = checker = original_hook = None
    frames: list[Any] = []
    ticks: list[dict[str, Any]] = []
    substeps: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    keys: list[str] = []
    attachment_count = 0
    try:
        base = gym.make(args.task, obs_mode='none', reward_mode='none', control_mode='pd_joint_pos',
            sim_backend='cpu', render_backend='gpu', render_mode='rgb_array', max_episode_steps=policy.max_episode_steps)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name='trajectory', save_video=False,
            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
            source_type='motionplanning', source_desc=f'{args.backend}; complete shared dual executor; zero diagnostic hover')
        env.reset(seed=args.seed)
        raw = env.unwrapped
        adapter = get_task_adapter(args.task, raw)
        targets, agents = adapter.targets, tuple(raw.agent.agents)

        def export_snapshot(index: int, **kwargs: Any) -> dict[str,Any]:
            snapshot = export_task_payload_snapshot(raw,index,targets[index],task_id=args.task,
                source_actor=adapter.initial_supports[index],**kwargs)
            geometry_manifest = None
            if payload_refinement in ('fixed-task-v3','fixed-task-v4') and args.task == TASKS[2]:
                from .curobo_fruit_link7_v3 import export_link7_geometry
                geometry_manifest = export_link7_geometry(raw,index)
            return configured_payload_snapshot(snapshot,payload_refinement,geometry_manifest)

        keys = list(raw.agent.agents_dict)
        arms = [bind_arm(key, agent) for key, agent in raw.agent.agents_dict.items()]
        if not np.isclose(raw.control_timestep,.05) or not np.isclose(raw.sim_timestep,.01):
            raise ValueError('Fixed20Hz control and100Hz physics are required')
        initial_arrays = _state_arrays(raw.get_state_dict())
        initial_digest = hashlib.sha256()
        for name, value in sorted(initial_arrays.items()):
            initial_digest.update(name.encode())
            initial_digest.update(str((value.shape,value.dtype.str)).encode())
            initial_digest.update(np.ascontiguousarray(value).tobytes())
        np.savez(output/'initial-state.npz', **initial_arrays)
        report.update(task_configuration=adapter.configuration(), target_names=list(adapter.target_names),
            action_key_order=keys, control_timestep_s=float(raw.control_timestep),
            physics_timestep_s=float(raw.sim_timestep), physics_backend=raw.backend.sim_backend,
            initial_state_content_sha256=initial_digest.hexdigest(),
            initial_state_sha256=hashlib.sha256((output/'initial-state.npz').read_bytes()).hexdigest(),
            stages=[], fault_events=[], task_geometry=save_task_geometry(output, raw, adapter),
            initial_scene_manifest=save_initial_scene_manifest(output,raw))
        capture = make_physics_capture(env, raw, adapter, args, report, frames, ticks, substeps, keys,
            checker_provider=lambda: checker,
            support_release_provider=lambda:measured_source_departure(checker),hard_step_budget=200)
        original_hook = capture.original_hook
        install_full_robot_contact_capture(raw, capture, adapter, contacts)
        raw._after_simulation_step = capture.record_substep
        capture.frame()
        hold = {arm.key: np.r_[arm.arm_qpos(),1.].astype(np.float32) for arm in arms}
        for _ in range(10):
            capture.step(hold)
        solvers = make_solvers(env, output, report, capture)
        checker = make_checker(raw, solvers, adapter)
        report['external_scene_coverage'] = checker.coverage
        report['native_collision_representation'] = {
            'backend': args.backend,
            'curobo': 'conservative actual PhysX spheres and full measured peer/world with declared exact finger/payload permissions',
            'mplib': 'original Panda URDF/SRDF solver; default native solver world; external paired full-scene/payload checker shared by both',
            'external_checker_shared': True, 'internal_representations_identical': False}
        report['native_collision_representation']['payload_refinement'] = payload_refinement
        report['native_collision_representation']['payload_refinement_changes'] = (
            'cuRobo native payload spheres/reserved capacity and both fruit-arm panda_link7 conservative spheres only; no physical state, collision permissions or thresholds changed'
            if payload_refinement in ('fixed-task-v3','fixed-task-v4') else
            'cuRobo native payload spheres and reserved capacity only; no physical state, collision permissions or thresholds changed')
        capacities: list[int] = []
        if args.backend == 'curobo':
            frozen = state_digest(raw)
            initial = tuple(export_snapshot(i) for i in range(2))
            for i, data in enumerate(initial):
                save_json(output/f'initial-arm-{i}-snapshot.json', data)
            capacities = [data['payload_model']['capacity'] for data in initial]
            transport = (SelectedWorkerProcess(os.environ['CUDA_VISIBLE_DEVICES'],
                output/'worker.stderr.log', args.worker_script,task_id=args.task) if args.worker_script else None)
            worker = CuroboTaskPayloadSessions(os.environ['CUDA_VISIBLE_DEVICES'],
                output/'worker.stderr.log',task_id=args.task,transport=transport)
            report['initialization'] = worker.initialize_pair(initial, seed=args.planner_seed,
                interpolation_dt=.02, warmup=True)
            report['initial_sessions'] = worker.inspect_sessions()
            if state_digest(raw) != frozen:
                raise RuntimeError('Planner initialization changed actual physics')
        grasp_evidence: list[dict[str, Any] | None] = [None,None]
        phase_budgets: dict[str, dict[str,int]] = {}

        def begin(name: str) -> None:
            report['stage'] = capture.phase = name
            save_json(output/'report.json', report)

        def finish(name: str, result: dict[str, Any]) -> None:
            report['stages'].append({'name':name, **result})
            # Publish only a result returned by the real executor. An initial
            # physical rejection can happen before the full candidate query.
            for planned in reversed(report.get('planning', [])):
                if planned.get('phase') == name and planned.get('collision_validation_owner') == 'executor':
                    planned['collision_validation'] = result.get('validation', {
                        'status':'not_run', 'reason':result.get('reason',result['status'])})
                    break
            save_json(output/'report.json', report)
            print(json.dumps({'phase':name, 'status':result['status'],
                'elapsed_steps':int(as_numpy(raw.elapsed_steps)[0])}), flush=True)
            if result['status'] != 'passed':
                raise RuntimeError(f'{name}: {result["status"]}: {result.get("reason")}')

        def plan(name: str, poses: tuple[Any,Any], *, closed: bool,
                 profile: str, contact: bool = False, source: bool = False,
                 budget_phase: str | None = None, require_safe_pair: bool = True) -> Any:
            begin(name)
            frozen = state_digest(raw)
            started = time.monotonic()
            limits = compact_phase_limits(report['execution_configuration'],
                'pregrasp' if profile == 'approach' else 'retreat')
            velocity, acceleration = limits['nominal_velocity_limit_rad_s'],limits['nominal_acceleration_limit_rad_s2']
            row: dict[str,Any] = {'phase':name, 'profile':profile,
                'before_control_step':int(as_numpy(raw.elapsed_steps)[0]), 'closed':closed,
                'nominal_velocity_limit_rad_s':velocity,
                'nominal_acceleration_limit_rad_s2':acceleration,
                'native_source_support_exception':source if worker else False}
            report.setdefault('planning',[]).append(row)
            trajectories = []
            if worker is not None:
                snapshots = tuple(export_snapshot(i,
                    contact_allowed=contact or closed,
                    grasp_evidence=grasp_evidence[i] if closed else None,
                    source_support_allowed=source, capacity=capacities[i])
                    for i in range(2))
                for i,data in enumerate(snapshots):
                    save_json(output/f'{name}-arm-{i}-snapshot.json',data)
                row['updates'] = worker.update_pair(snapshots,profile=profile,snapshot_id=name)
                row['world_readback'] = audit_task_pair(row['updates'],snapshots)
                row['fk_checks'] = []
                for i,(arm,goal,data) in enumerate(zip(arms,poses,snapshots)):
                    q = arm.arm_qpos()
                    fk = worker.fk(arm=i,profile=profile,snapshot_id=name,positions=[q.tolist()])
                    save_json(output/f'{name}-arm-{i}-fk.json',fk)
                    row['fk_checks'].append({'arm':i,'tcp':check_fk(fk,data,q[None]),
                        'fingers':verify_finger_spheres(fk,data,q[None]),
                        'payload':audit_payload_spheres(fk,data)})
                    goal_base = arm.agent.robot.pose.sp.inv()*goal
                    call_started = time.perf_counter()
                    try:
                        result = worker.plan_pose(arm=i,profile=profile,snapshot_id=name,
                            start=q.tolist(),goal_pose=np.r_[goal_base.p,goal_base.q].tolist(),max_attempts=4)
                    finally:
                        report.setdefault('native_planning_calls',[]).append({'phase':name,
                            'arm_index':i,'wall_seconds':time.perf_counter()-call_started})
                    save_json(output/f'{name}-arm-{i}-native-result.json',result)
                    trajectory,audit = trajectory_from_result(result,action_key=arm.key,
                        joint_names=arm.names,start=q,velocity_limit=velocity,acceleration_limit=acceleration)
                    trajectories.append(ArmTrajectory(trajectory.action_key,trajectory.joint_names,
                        trajectory.times_s,trajectory.positions_rad,-1. if closed else 1.))
                    row.setdefault('trajectory_audits',[]).append(audit)
            else:
                limits = compact_phase_limits(report['execution_configuration'],name)
                compact_set_phase_limits(solvers,limits)
                for solver in solvers:
                    solver.last_endpoint_audit = None
                try:
                    original_schedule = plan_dual_pose(*solvers,poses,
                        gripper_commands=(-1,-1) if closed else (1,1))
                finally:
                    row['endpoint_audits'] = [s.last_endpoint_audit for s in solvers]
                    for i,audit in enumerate(row['endpoint_audits']):
                        if audit is not None:
                            save_json(output/f'{name}-arm-{i}-endpoint-audit.json',json_value(audit))
                raw_results = tuple(s.last_native_result for s in solvers)
                traces = tuple(s.last_native_parameterization for s in solvers)
                for i,result in enumerate(raw_results):
                    # Retain the initial native return separately. These files
                    # bind the accepted correction to the actual retimed path.
                    save_json(output/f'{name}-arm-{i}-accepted-native-result.json',json_value(result))
                retimed,certificate = compact_retime_pair(raw_results,traces,
                    velocity_limit=velocity,acceleration_limit=acceleration)
                row['phase_limit_configuration'] = limits
                row['analytic_parameterization'] = certificate
                for i,(source_trajectory,result,trace) in enumerate(zip(original_schedule.trajectories,retimed,traces)):
                    audit = compact_kinematic_audit(result,velocity_limit=velocity,acceleration_limit=acceleration)
                    if audit['status'] != 'passed':
                        raise RuntimeError('MPLib derivative audit rejected the preserved phase')
                    row.setdefault('trajectory_audits',[]).append(audit)
                    np.savez(output/f'{name}-arm-{i}-analytic-topp-trace.npz',**trace)
                    trajectories.append(ArmTrajectory(source_trajectory.action_key,source_trajectory.joint_names,
                        result['time'],result['position'],-1. if closed else 1.))
            if state_digest(raw) != frozen:
                raise RuntimeError('Planning/model updates changed actual simulator state')
            schedule = build_hold_schedule(*trajectories,control_timestep_s=raw.control_timestep)
            row.update(planning_environment_steps=0,planning_pipeline_wall_s=time.monotonic()-started,
                scheduled_steps=schedule.num_steps,
                source_durations_s=[t.duration_s for t in trajectories])
            np.savez(output/f'{name}-schedule.npz',left_times=trajectories[0].times_s,
                right_times=trajectories[1].times_s,left_positions=trajectories[0].positions_rad,
                right_positions=trajectories[1].positions_rad,target_times=schedule.target_times_s,
                left_commands=schedule.left_commands,right_commands=schedule.right_commands)
            row['budget'] = phase_budget(int(as_numpy(raw.elapsed_steps)[0]),schedule.num_steps,
                name if budget_phase is None else budget_phase,transport_recipe=transport_policy,policy=policy)
            phase_budgets[name] = row['budget']
            row['require_safe_pair'] = require_safe_pair
            # Diagnostic hooks acquire candidate trajectories without executing
            # them, so retain their original full query and safety policy.
            if phase_hook is not None or not require_safe_pair:
                checker.refresh()
                checker.set_target_contacts_allowed((contact or closed,)*2)
                try:
                    row['collision_validation'] = validate_schedule(schedule,*(arm.arm_qpos() for arm in arms),checker)
                finally:
                    checker.set_target_contacts_allowed((False,False))
                row['collision_validation_owner'] = 'planner_diagnostic'
                enforce_pair_validation(row['collision_validation'],require_safe_pair=require_safe_pair)
            else:
                row['collision_validation_owner'] = 'executor'
                row['collision_validation'] = {'status':'pending_executor_validation'}
            save_json(output/'report.json',report)
            return schedule

        original_pairs = adapter.grasp_poses()
        pairs = approach_pairs(original_pairs,args.approach_recipe)
        report['grasp_targets_original'] = [[np.r_[p.p,p.q].tolist() for p in pair] for pair in original_pairs]
        report['approach_endpoint_geometry'] = approach_endpoint_geometry(raw,adapter,pairs)
        report['approach_recipe'] = args.approach_recipe
        report['grasp_targets'] = [[np.r_[p.p,p.q].tolist() for p in pair] for pair in pairs]
        for index,name,allowed in ((0,'pregrasp',False),(1,'descend',True)):
            schedule = plan(name,(pairs[0][index],pairs[1][index]),closed=False,
                profile='approach',contact=allowed)
            finish(name,execute_dual_reach(capture,schedule,checker,
                allow_target_contacts=(allowed,allowed),settle_max_steps=phase_budgets[name]['settle_max_steps'],policy=policy))
        begin('close_and_confirm')
        close_tail = policy.release_stable_steps + 4 * policy.reach_stable_steps + policy.final_stable_steps
        close_budget = min(40,policy.last_completion_step-int(as_numpy(raw.elapsed_steps)[0])-close_tail)
        if close_budget < policy.grasp_stable_steps:
            raise RuntimeError('Insufficient budget for five stable grasp ticks and mandatory tail')
        closed = close_dual_and_confirm(capture,checker,targets,max_steps=close_budget,policy=policy)
        finish('close_and_confirm',closed)
        checker.attach()
        attachment_count += 1
        report['initial_source_departure_evidence'] = source_departure_evidence(checker)
        references = np.asarray(checker.attachments).copy()
        grasp_evidence = [{'evidence_id':f'real-close-step-{int(as_numpy(raw.elapsed_steps)[0])}',
            'confirmation':closed,'tcp_to_object_wxyz':_pose(reference)} for reference in references]
        report['attachment_capture'] = {'count':attachment_count,
            'environment_step':int(as_numpy(raw.elapsed_steps)[0]),
            'T_tcp_objects':references.tolist(),'physical_weld':False}
        original_q = [as_numpy(a.tcp.pose.q)[0].copy() for a in agents]
        carry_q = adapter.carry_quaternions()
        releases = adapter.release_poses(carry_quaternions=carry_q)
        route = compact_route_positions([as_numpy(a.tcp.pose.p)[0] for a in agents],
            [np.asarray(p.p) for p in releases],route='merge-lift')
        report['route'] = json_value(route)
        report['task_configuration_after_grasp'] = adapter.configuration()
        lift_poses = tuple(sapien.Pose(xyz,q) for xyz,q in zip(route['lift'],original_q))
        schedule = plan('lift_and_hold',lift_poses,closed=True,profile='transport',source=True)
        finish('lift_and_hold',execute_dual_held_move(capture,schedule,checker,targets,
            min_height_gain_m=(policy.min_lift_m,policy.min_lift_m),hold_steps=0,
            settle_max_steps=phase_budgets['lift_and_hold']['settle_max_steps'],policy=policy))
        report['source_departure_after_lift'] = source_departure_evidence(checker)
        if (not all(report['source_departure_after_lift']['effective_measured_departure'])
                or not substeps or not all(substeps[-1]['physical_support_released_snapshot'])):
            raise RuntimeError('Measured geometric or physical source support has not permanently ended')
        if phase_hook is not None:
            continue_task = phase_hook(args=args,output=output,report=report,raw=raw,
                capture=capture,solvers=solvers,checker=checker,adapter=adapter,
                plan=plan,begin=begin,finish=finish,substeps=substeps,ticks=ticks,
                targets=targets,arms=arms,references=references,route=route,
                carry_q=carry_q,original_q=original_q,worker=worker)
            if continue_task is not True:
                return
        transport_stages = (('carry_orient_and_transport',route['above_baskets']),
                            ('carry_lower',route['release']))
        if transport_recipe is not None:
            transport_stages = transport_recipe(args.task,route,report['source_departure_after_lift'])
        report['transport_recipe_selection'] = {'policy':transport_policy,
            'selected_phases':[name for name,_ in transport_stages],
            'goal_positions_world_m':[np.asarray(xyz).tolist() for _,xyz in transport_stages],
            'carry_quaternions':'original task adapter carry_q',
            'acceptance_evidence':'original planning, stages and continuous100Hz held audit'}
        for name,positions in transport_stages:
            poses = tuple(sapien.Pose(xyz,q) for xyz,q in zip(positions,carry_q))
            schedule = plan(name,poses,closed=True,profile='transport')
            finish(name,execute_dual_held_move(capture,schedule,checker,targets,hold_steps=0,
                settle_max_steps=phase_budgets[name]['settle_max_steps'],policy=policy))
        report['held_physics_audit'] = held_substep_audit(substeps,references,policy=policy,target_names=adapter.target_names)
        if report['held_physics_audit']['status'] != 'passed':
            raise RuntimeError('Continuous sampled100Hz held-payload slip audit failed')
        report['real_grasp_and_payload_status'] = 'passed'
        checker.detach()
        begin('release')
        release_budget = min(30,policy.last_completion_step-int(as_numpy(raw.elapsed_steps)[0])
            - policy.reach_stable_steps - policy.final_stable_steps)
        if release_budget < policy.release_stable_steps:
            raise RuntimeError('Insufficient budget for release and mandatory retreat/placement tail')
        finish('release',release_dual_and_confirm(capture,checker,targets,max_steps=release_budget,policy=policy))
        grasp_evidence = [None,None]
        if worker is not None:
            frozen = state_digest(raw)
            released = tuple(export_snapshot(i,capacity=capacities[i]) for i in range(2))
            for i,data in enumerate(released):
                save_json(output/f'released-arm-{i}-snapshot.json',data)
            updates = worker.update_pair(released,profile='transport',snapshot_id='released-state')
            report['released_world_readback'] = audit_task_pair(updates,released)
            report['released_updates'] = updates
            report['released_payload_geometry'] = []
            for i,data in enumerate(released):
                fk = worker.fk(arm=i,profile='transport',snapshot_id='released-state',
                    positions=[arms[i].arm_qpos().tolist()])
                save_json(output/f'released-arm-{i}-fk.json',fk)
                report['released_payload_geometry'].append(audit_payload_spheres(fk,data))
            if state_digest(raw) != frozen:
                raise RuntimeError('Native detach changed measured simulator state')
            report['native_payload_detach_verified'] = True
        else:
            report['native_payload_detach_verified'] = None
            report['native_payload_detach_scope'] = 'MPLib uses shared external measured payload prediction; detached before physical release'
        retreat_start = np.asarray([as_numpy(a.tcp.pose.p)[0].copy() for a in agents])
        retreat = tuple(sapien.Pose(position+[0.,0.,policy.retreat_command_m],as_numpy(a.tcp.pose.q)[0])
            for position,a in zip(retreat_start,agents))
        schedule = plan('retreat',retreat,closed=False,profile='transport')
        finish('retreat',execute_dual_reach(capture,schedule,checker,
            settle_max_steps=phase_budgets['retreat']['settle_max_steps'],policy=policy))
        gain = np.asarray([as_numpy(a.tcp.pose.p)[0] for a in agents])-retreat_start
        report['actual_retreat_displacement_m'] = gain.tolist()
        if np.any(gain[:,2] < policy.minimum_retreat_m):
            raise RuntimeError('Actual retreat gain below8cm')
        begin('verify_placement')
        remaining = policy.last_completion_step-int(as_numpy(raw.elapsed_steps)[0])
        if remaining < policy.final_stable_steps:
            raise RuntimeError('Final continuous20-tick placement window does not fit')
        placed = verify_dual_placement(capture,checker,targets,
            official_predicate_keys=adapter.official_predicate_keys,region_predicates=adapter.region_predicates(),
            max_steps=min(60,remaining),stable_steps=policy.final_stable_steps,policy=policy)
        finish('verify_placement',placed)
        report['placement_stability_evaluated'] = True
        evaluation = raw.evaluate()
        report['official_200_step_complete'] = bool(np.asarray(as_numpy(evaluation['success'])).all()
            and int(as_numpy(raw.elapsed_steps)[0]) < policy.max_episode_steps)
        report['project_complete_physical_closure'] = report['official_200_step_complete']
        report['status'] = 'passed' if report['official_200_step_complete'] else 'failed'
    finally:
        if worker is not None:
            try:
                try:
                    if not worker._rpc.failed:
                        report['final_sessions'] = worker.inspect_sessions()
                        report['persistent_session_audit'] = audit_session_identity(
                            report['initial_sessions'],report['final_sessions'])
                finally:
                    worker.close()
            except BaseException as exc:
                report.setdefault('artifact_errors',[]).append(f'worker close: {type(exc).__name__}: {exc}')
            report['worker_requests'] = worker.receipts
        save_json(output/'all-robot-contacts.json',contacts)
        if env is not None:
            finalize_physics_capture(env,raw,capture,output,report,frames,ticks,substeps,
                keys,original_hook,attachment_count=attachment_count,robot_contacts=contacts)
            recording = report.get('recording',{})
            report['recording_actions_and_Tplus1_verified'] = bool(
                recording.get('submitted_actions_matched') is True
                and recording.get('recorded_steps') == len(ticks)
                and recording.get('state_lengths')
                and all(n == len(ticks)+1 for n in recording['state_lengths'].values()))
            if report.get('status') == 'passed' and not report['recording_actions_and_Tplus1_verified']:
                report['status'] = 'failed'
                report.setdefault('artifact_errors', []).append(
                    'record check: submitted actions or T+1 states mismatch')


def self_test() -> dict[str,Any]:
    """CPU boundary tests for mandatory-tail budgeting and backend-neutral schedule shapes."""
    import torch
    from .initial_scene_manifest import self_test as initial_scene_self_test
    assert not torch.cuda.is_initialized()
    initial_scene_result = initial_scene_self_test()
    assert initial_scene_result['status'] == 'passed' and initial_scene_result['environment_steps'] == 0
    config = execution_configuration()
    assert future_minimum_ticks() == FUTURE_MINIMUM_TICKS
    assert config['airborne_hover_steps'] == 0 and config['final_common_placement_steps'] == 20
    assert config['payload_refinement'] == 'baseline'
    assert selected_transport_recipe(argparse.Namespace()) == 'standard'
    assert selected_transport_recipe(argparse.Namespace(task=TASKS[1],
        transport_recipe='pill-direct-release-v1')) == 'pill-direct-release-v1'
    direct = execution_configuration(TASKS[1],transport_recipe='pill-direct-release-v1')
    standard = execution_configuration(TASKS[1])
    assert direct['route'] == 'merge-lift-direct-release'
    assert direct['transport_phase_sequence'] == ['carry_lower']
    for task in TASKS:
        candidate = execution_configuration(task, transport_recipe='direct-release-v2')
        original = execution_configuration(task)
        assert candidate['transport_phase_sequence'] == ['carry_lower']
        assert {k:v for k,v in candidate.items() if k not in ('route','transport_recipe','transport_phase_sequence')} == {
            k:v for k,v in original.items() if k not in ('route','transport_recipe','transport_phase_sequence')}
    recipe_fields = {'route','transport_recipe','transport_phase_sequence'}
    assert {k:v for k,v in direct.items() if k not in recipe_fields} == {
        k:v for k,v in standard.items() if k not in recipe_fields}
    for task,policy in ((TASKS[0],'pill-direct-release-v1'),
                        (TASKS[2],'pill-direct-release-v1'),(TASKS[1],'unknown')):
        try:
            selected_transport_recipe(argparse.Namespace(task=task,transport_recipe=policy))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid task-specific transport recipe accepted')
    assert selected_payload_refinement(argparse.Namespace()) == 'baseline'
    assert selected_payload_refinement(argparse.Namespace(payload_refinement='fixed-task-v1')) == 'fixed-task-v1'
    for task in TASKS:
        baseline = execution_configuration(task,payload_refinement='baseline')
        refined = execution_configuration(task,payload_refinement='fixed-task-v1')
        assert {k:v for k,v in baseline.items() if k!='payload_refinement'} == {
            k:v for k,v in refined.items() if k!='payload_refinement'}
    for invalid in ('unknown',None,True):
        try:
            selected_payload_refinement(argparse.Namespace(payload_refinement=invalid))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid native payload representation policy accepted')
    apple_path = ROOT/'infrastructure/validation/dual-arm/curobo-complete-20261116/task-source-snapshots-dev02/TwoRobotPickAppleBananaToBaskets-v1-arm-0.json'
    apple = json.loads(apple_path.read_text())
    encoded_apple = json.dumps(apple,sort_keys=True,allow_nan=False)
    assert configured_payload_snapshot(apple,'baseline') is apple
    assert configured_payload_snapshot(apple,'fixed-task-v2') == apple
    assert configured_payload_snapshot(apple,'fixed-task-v3') == apple
    assert configured_payload_snapshot(apple,'fixed-task-v4') == apple
    refined_apple = configured_payload_snapshot(apple,'fixed-task-v1')
    assert refined_apple == apple and refined_apple is not apple
    assert json.dumps(apple,sort_keys=True,allow_nan=False) == encoded_apple
    for task,approach in zip(TASKS,((.9,.9),(1.,1.3),(1.5,2.6))):
        task_config = execution_configuration(task)
        limits = compact_phase_limits(task_config,'pregrasp')
        assert (limits['nominal_velocity_limit_rad_s'],limits['nominal_acceleration_limit_rad_s2']) == approach
        transport = compact_phase_limits(task_config,'retreat')
        assert (transport['nominal_velocity_limit_rad_s'],transport['nominal_acceleration_limit_rad_s2']) == (1.5,2.6)
    feasible = phase_budget(160,10,'retreat')
    assert feasible['settle_max_steps'] == 9 and feasible['reserved_future_minimum_steps'] == 20
    edge = phase_budget(175,1,'retreat')
    assert edge['settle_max_steps'] == 3
    direct_lift = phase_budget(155,10,'lift_and_hold',transport_recipe='direct-release-v2')
    assert direct_lift['reserved_future_minimum_steps'] == 29
    assert direct_lift['settle_max_steps'] == 5
    assert phase_budget(175,1,'retreat',transport_recipe='direct-release-v2')['settle_max_steps'] == 3
    try:
        phase_budget(155,10,'lift_and_hold')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Historical two-stop future budget changed')
    rejected = 0
    for values in ((179,1,'retreat'),(178,1,'retreat'),(177,1,'retreat'),
                   (176,1,'retreat'),(0,0,'retreat'),(True,1,'retreat'),(0,1,'unknown')):
        try:
            phase_budget(*values)
        except (ValueError,RuntimeError):
            rejected += 1
        else:
            raise AssertionError('Invalid or over-budget mandatory tail accepted')
    left = ArmTrajectory('left',tuple(f'j{i}' for i in range(7)),[0.,.1],np.zeros((2,7)),-1.)
    right = ArmTrajectory('right',left.joint_names,[0.,.2],np.ones((2,7)),-1.)
    schedule = build_hold_schedule(left,right,control_timestep_s=.05)
    assert schedule.num_steps == 4 and schedule.left_commands.shape == schedule.right_commands.shape == (4,8)
    assert np.all(schedule.left_commands[:,-1] == -1.) and np.all(schedule.right_commands[:,-1] == -1.)
    assert len(TASKS) == 3 and config['minimum_actual_retreat_m'] == .08
    import sapien
    import sapien.physx as physx
    from types import SimpleNamespace
    grasp = sapien.Pose([0.,0.,.10],[0.,1.,0.,0.])
    pre = sapien.Pose([0.,0.,.18],[0.,1.,0.,0.])
    original_pairs = ((pre,grasp),(pre,grasp))
    compact = approach_pairs(original_pairs,'compact-4cm')
    assert np.allclose(compact[0][0].p,[0.,0.,.14])
    assert compact[0][1] is grasp and np.allclose(original_pairs[0][0].p,[0.,0.,.18])
    compact5 = approach_pairs(original_pairs,'compact-5cm')
    assert np.allclose(compact5[0][0].p,[0.,0.,.15]) and compact5[0][1] is grasp
    assert execution_configuration(TASKS[1],approach_recipe='compact-5cm')['pregrasp_distance_m'] == .05
    assert approach_pairs(original_pairs,'standard-8cm')[0][0] is pre
    try:
        approach_pairs(((pre,sapien.Pose()),(pre,grasp)),'compact-4cm')
    except ValueError:
        rejected += 1
    else:
        raise AssertionError('Non-downward compact grasp accepted')
    shape = physx.PhysxCollisionShapeBox(np.array([.001,.001,.001],dtype=np.float32),
        physx.PhysxMaterial(.5,.5,0.))
    links = [SimpleNamespace(get_name=lambda n=n:n,pose=sapien.Pose(),
        _objs=[SimpleNamespace(get_collision_shapes=lambda:[shape])])
        for n in ('panda_leftfinger','panda_rightfinger','panda_hand','camera_link')]
    agent = SimpleNamespace(tcp=SimpleNamespace(pose=sapien.Pose()),robot=SimpleNamespace(get_links=lambda:links))
    raw = SimpleNamespace(agent=SimpleNamespace(agents=(agent,agent)))
    target_vertices = np.array([[0.,0.,.12],[.001,.001,.11]])
    adapter = SimpleNamespace(targets=(object(),object()),_vertices=lambda target:target_vertices)
    geometry = approach_endpoint_geometry(raw,adapter,compact)
    assert len(geometry) == 2 and all(row['positive_vertical_separation_m'] > .018 for row in geometry)
    target_vertices[:,2] += .1
    try:
        approach_endpoint_geometry(raw,adapter,compact)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError('Pregrasp endpoint penetrating target Z envelope accepted')
    initial = {'sessions':{str(i):{'planner_id':i,'init_count':1,'warmup_count':1,'rebuild_count':0}
        for i in range(4)}}
    assert audit_session_identity(initial,initial)['status'] == 'passed'
    changed = json.loads(json.dumps(initial))
    changed['sessions']['0']['planner_id'] = 99
    try:
        audit_session_identity(initial,changed)
    except RuntimeError:
        rejected += 1
    else:
        raise AssertionError('Replaced persistent planner was accepted')
    unsafe = {'status':'collision','collision_count':1}
    enforce_pair_validation(unsafe,require_safe_pair=False)
    assert unsafe == {'status':'collision','collision_count':1}
    try:
        enforce_pair_validation(unsafe)
    except RuntimeError:
        rejected += 1
    else:
        raise AssertionError('Default complete execution accepted an unsafe pair')
    measured = {'mode':'measured_refresh','all_pieces_separated':True,
        'fcl_touching_pieces':[],'separation_numeric_band_m':1e-6,
        'pieces':[{'target':'actual_piece','bottom_minus_table_top_m':2e-6}]}
    supports = [SimpleNamespace(support_released=False,_initial_support_exists=False,
        _target_names={'actual_piece'},_support_state_receipt=json.loads(json.dumps(measured)),
        _last_support_validation=json.loads(json.dumps(measured))) for _ in range(2)]
    refreshes = []
    source_checker = SimpleNamespace(attached=True,_supports=supports,
        refresh=lambda:refreshes.append(True))
    assert measured_source_departure(source_checker) == (True,True)
    assert len(refreshes) == 1 and not any(s.support_released for s in supports)
    malformed = [None,{},dict(measured,mode='supplied_object_pose'),
        dict(measured,all_pieces_separated=False),dict(measured,fcl_touching_pieces=['actual_piece']),
        dict(measured,pieces=[]),dict(measured,pieces=[{'target':'other','bottom_minus_table_top_m':2e-6}]),
        dict(measured,pieces=[{'target':'actual_piece','bottom_minus_table_top_m':1e-6}]),
        dict(measured,pieces=[{'target':'actual_piece','bottom_minus_table_top_m':float('nan')}]),
        dict(measured,separation_numeric_band_m=0.)]
    for receipt in malformed:
        supports[0]._support_state_receipt = receipt
        assert measured_source_departure(source_checker) == (False,True)
        assert not any(s.support_released for s in supports)
        rejected += 1
    supports[0]._support_state_receipt = measured
    supports[0]._initial_support_exists = True
    assert measured_source_departure(source_checker) == (False,True)
    rejected += 1
    source_checker.attached = False
    assert measured_source_departure(source_checker) == (False,False)
    assert measured_source_departure(None) == (False,False)
    source_checker.attached = True
    for support in supports:
        support.support_released = True
        support._support_state_receipt = None
    before_refreshes = len(refreshes)
    assert measured_source_departure(source_checker) == (True,True)
    assert len(refreshes) == before_refreshes
    return {'status':'passed','rejected_boundary_cases':rejected,
        'paired_action_shape':[2,8],'native_trajectory_shape':[2,7],
        'common_schedule_shape':[4,8], 'environment_steps':0,
        'payload_refinement_wiring':{'default':'baseline','fixed_policy':'fixed-task-v1',
            'apple_snapshot_values_unchanged':True,'input_snapshot_unchanged':True,
            'nonrepresentation_execution_configuration_unchanged':True},
        'initial_scene_manifest_self_test':initial_scene_result,
        'GPU_initialized':bool(torch.cuda.is_initialized())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--backend',choices=('curobo','mplib'),default='curobo')
    parser.add_argument('--task-id','--task',dest='task',choices=TASKS,default=TASKS[0])
    parser.add_argument('--seed',type=int,default=401)
    parser.add_argument('--planner-seed',type=int,default=0)
    parser.add_argument('--approach-recipe',choices=APPROACH_RECIPES,default='compact-4cm')
    parser.add_argument('--payload-refinement',choices=PAYLOAD_REFINEMENTS,default='baseline')
    parser.add_argument('--transport-recipe',choices=TRANSPORT_RECIPES,default='standard')
    parser.add_argument('--worker-script',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),allow_nan=False))
        return 0
    if args.transport_recipe == 'pill-direct-release-v1' and args.task != TASKS[1]:
        parser.error('--transport-recipe pill-direct-release-v1 requires the pill task')
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    signal.signal(signal.SIGTERM,termination_handler)
    gpu_uuid = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu_uuid.startswith('GPU-') or ',' in gpu_uuid:
        parser.error('One explicit GPU UUID required')
    if any(os.environ.get(f'MANI_SKILL_L{i}') for i in (1,2,3)):
        parser.error('This protocol requires L0')
    if args.output_dir is None or args.seed < 0 or args.planner_seed < 0:
        parser.error('Unused output directory and nonnegative seeds required')
    if args.worker_script and (args.backend != 'curobo' or not args.worker_script.is_file()):
        parser.error('--worker-script requires cuRobo and an existing file')
    args.mode = 'positive'
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=False)
    sources = list(HERE.glob('*.py')) + [CODE_ROOT/'infrastructure/tig_runtime.py']
    sources.extend((CODE_ROOT/'infrastructure').glob('curobo_tig*_worker.py'))
    if args.worker_script:
        sources.append(args.worker_script.resolve())
    # An explicit selected worker owns its basename in the archive. Its actual
    # input path and SHA remain in source_sha256, including external overrides.
    sources = list({source.name: source for source in sources}.values())
    archive = output/'source'
    archive.mkdir()
    for source in sources:
        shutil.copy2(source,archive/source.name)
    report: dict[str,Any] = {'status':'running','seed':args.seed,'planner_seed':args.planner_seed,
        'task':args.task,'mode':args.mode,'backend':args.backend,
        'worker_script':str(args.worker_script.resolve()) if args.worker_script else (
            str(CODE_ROOT/'infrastructure/curobo_tig_task_payload_worker.py') if args.backend == 'curobo' else None),
        'approach_recipe':args.approach_recipe,
        'payload_refinement':args.payload_refinement,
        'transport_recipe':args.transport_recipe,
        'started_at':datetime.now(timezone.utc).isoformat(),'gpu_uuid':gpu_uuid,
        'max_episode_steps':LEGACY_STRICT.max_episode_steps,'official_200_step_complete':False,
        'runtime_policy':resolved_policy_record(),
        'project_complete_physical_closure':False,'real_grasp_and_payload_status':'not_completed',
        'placement_stability_evaluated':False,'execution_configuration':execution_configuration(
            args.task,approach_recipe=args.approach_recipe,payload_refinement=args.payload_refinement,
            transport_recipe=args.transport_recipe),
        'scope':'complete measured grasp, held transport, release, actual retreat and continuous20tick placement',
        'physical_pose_assignment':False,'physical_attachment_created':False,
        'physical_collision_masks_modified':False,'native_timeopt_payload_inertia_applied':False,
        'standard_recording_replay_evaluated':False,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'official_tig':git_identity(ROOT/'The-Imitator-Game'),
        'runtime_tig':git_identity(CODE_ROOT),
        'official_curobo':git_identity(RUNTIME_PATHS.curobo_source)}
    save_json(output/'report.json',report)
    started = time.monotonic()
    try:
        native_run(args,output,report)
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report.update(wall_s=time.monotonic()-started,completed_at=datetime.now(timezone.utc).isoformat())
    from .run_results import update_run_results_from_file
    report = json_value(report)
    update_run_results_from_file(report, output/'ticks.jsonl')
    save_json(output/'report.json',report)
    print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
