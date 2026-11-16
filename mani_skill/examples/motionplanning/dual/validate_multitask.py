"""Task-adapted synchronized dual-Panda physical closure under official200.

Contract: canonical left/right actions[N,16], seven absolute radians plus one
normalized gripper per arm; XYZ metres/WXYZ poses;20Hz control/100Hz physics.
One shared orchestrator calls existing executors. Task geometry is supplied by
multitask_adapters. No physical pose assignment, weld or mask changes are used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import time
import traceback
from typing import Any
import numpy as np
from .validate_grasp import json_value, recording_summary, save_report
from .validate_sync import continue_success_only, applied_action, retain_right_valid, held_substep_audit, hold_joint_substep_audit
from .validate_sync_compact import (compact_configuration, compact_phase_limits, compact_set_phase_limits,
    compact_retime_pair, compact_kinematic_audit, compact_route_positions,
    compact_urdf_velocity_limits, compact_topp_reconstruction, SOLVER_LIMIT_FRACTION)
from .validate_path_conflict import _state_arrays, interarm_contact_rows
from .validate_held_path_conflict import classify_interarm_contacts, physical_contact_points
from .multitask_adapters import get_task_adapter, task_execution_parameters, task_phase_limits
from .multitask_recording import (contact_class, checked_geometry, save_task_geometry,
    make_physics_capture, finalize_physics_capture)
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
EVIDENCE = ROOT / 'infrastructure/validation/dual-arm/m1-multitask-20261116'
TASKS = ('TwoRobotPickAppleBananaToBaskets-v1','TwoRobotPickPillToRegions-v1','TwoRobotPickFruitsToPlate-v1')


# Compatibility exports: implementations live in shared runtime modules.
from .runtime_io import (ControlledTermination, termination_handler)





def termination_self_test() -> None:
    previous=signal.signal(signal.SIGTERM,termination_handler);events=[]
    try:
        try: signal.raise_signal(signal.SIGTERM)
        finally: events.append('finalized')
    except ControlledTermination: events.append('caught')
    finally: signal.signal(signal.SIGTERM,previous)
    assert events==['finalized','caught']


def timed_native_call(call: Any, rows: list, *, phase: str, arm_index: int) -> Any:
    """Retain instrumented planner-call time, including failed calls and TOPP recording."""
    started = time.perf_counter()
    receipt = {'phase': phase, 'arm_index': arm_index, 'status': 'running'}
    try:
        result = call()
        receipt['status'] = 'returned'
        return result
    except Exception as exc:
        receipt.update(status='raised', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['wall_seconds'] = time.perf_counter() - started
        rows.append(receipt)














def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any], *, phase_hook: Any = None,
               grasp_pose_hook: Any = None, close_failure_hook: Any = None) -> None:
    import gymnasium as gym
    import sapien
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .dual_arm_execution import (plan_dual_pose, execute_dual_reach, close_dual_and_confirm,
                                    execute_dual_held_move, release_dual_and_confirm, verify_dual_placement)
    from .trajectory_schedule import ArmTrajectory, build_hold_schedule
    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh

    env, raw, original_hook = None, None, None
    checker, adapter, captured = None, None, None
    attachment_count = 0
    frames, ticks, substeps, keys = [], [], [], []
    report["fault_events"] = []
    try:
        base = gym.make(args.task_id, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos", sim_backend="cpu",
                        render_backend="gpu", render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
                            source_type="motionplanning", source_desc=f"task-adapter synchronized execution;task={args.task_id};route={args.route};profile={args.execution_profile}")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        adapter = get_task_adapter(args.task_id, raw)
        agents, targets = (raw.left_agent, raw.right_agent), adapter.targets
        report["task_configuration"] = adapter.configuration()
        report["target_names"] = list(adapter.target_names)
        report["official_predicate_keys"] = list(adapter.official_predicate_keys)
        report['task_geometry'] = save_task_geometry(output, raw, adapter)
        report['target_state_names'] = [report['task_geometry']['actors'][f'target{i}']['state_name'] for i in range(2)]
        initial_arrays = _state_arrays(raw.get_state_dict())
        np.savez(output / "initial-state.npz", **initial_arrays)
        report["initial_state_sha256"] = hashlib.sha256((output / "initial-state.npz").read_bytes()).hexdigest()
        keys = list(raw.agent.agents_dict)
        if len(keys) != 2 or any(raw.agent.agents_dict[k] is not a for k, a in zip(keys, agents)):
            raise ValueError("Physical left/right action identity mismatch")
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), stages=[], action_key_order=keys)

        captured = make_physics_capture(env, raw, adapter, args, report, frames, ticks, substeps, keys,
                                        checker_provider=lambda: checker, hard_step_budget=args.max_episode_steps)
        original_hook = captured.original_hook
        record_substep = captured.record_substep

        from .dual_arm_execution import _canonical
        arms = _canonical(captured)
        report["urdf_joint_velocity_checks"] = [compact_urdf_velocity_limits(a.urdf_path, arm.names,
            max(args.joint_vel_limit, report["execution_configuration"]["approach_joint_vel_limit_rad_s"]))
            for a, arm in zip(agents, arms)]
        if (not np.isclose(raw.control_timestep, .05, rtol=0, atol=1e-12)
                or not np.isclose(raw.sim_timestep, .01, rtol=0, atol=1e-12)):
            raise ValueError("Compact runs require the unchanged 20Hz control/100Hz physics clock")
        report["physics_timestep_s"] = float(raw.sim_timestep)
        raw._after_simulation_step = record_substep
        captured.frame()
        initial_hold = {k: np.r_[as_numpy(a.robot.get_qpos())[0, :7], 1].astype(np.float32) for k, a in zip(keys, agents)}
        for _ in range(args.preparation_steps):
            _, _, done, truncated, _ = captured.step(initial_hold)
            if done or bool(as_numpy(truncated).reshape(-1)[0]): raise RuntimeError("Environment ended during preparation")
        report["preparation_steps"] = args.preparation_steps
        class AuditedSolver(PandaArmMotionPlanningSolver):
            """Retain original native TOPP derivative arrays before conversion."""
            def setup_planner(self) -> Any:
                planner = super().setup_planner()
                native_topp = planner.TOPP

                def traced_topp(path: Any, step: float = .1, verbose: bool = False) -> Any:
                    raw_output = native_topp(path, step, verbose)
                    payload = {"native_path": np.asarray(path).copy(),
                               "solver_velocity_limits": np.asarray(planner.joint_vel_limits).copy(),
                               "solver_acceleration_limits": np.asarray(planner.joint_acc_limits).copy()}
                    for field, value in zip(("time", "position", "velocity", "acceleration", "duration"), raw_output):
                        if value is not None:
                            payload["raw_" + field] = np.asarray(value).copy()
                    # Persist the original output even if reconstruction rejects.
                    np.savez(output / f"{captured.phase}-arm{self.multi_robot_id}-native-topp-raw.npz", **payload)
                    self.last_native_parameterization = compact_topp_reconstruction(
                        path, planner.joint_vel_limits, planner.joint_acc_limits, step, raw_output)
                    return raw_output

                planner.TOPP = traced_topp
                return planner

            def move_to_pose_with_screw(self, pose: Any, dry_run: bool = False,
                                        refine_steps: int = 0, other_gripper_state: Any = None) -> Any:
                if not dry_run:
                    raise ValueError("Compact execution requires plan-only solver calls")
                self.last_native_parameterization = None
                native_call = super().move_to_pose_with_screw
                calls = report.setdefault('native_planning_calls', [])
                result = timed_native_call(lambda: native_call(pose, dry_run=True,
                    refine_steps=refine_steps, other_gripper_state=other_gripper_state),
                    calls, phase=captured.phase, arm_index=int(self.multi_robot_id))
                self.last_native_planning_seconds = calls[-1]['wall_seconds']
                self.last_native_result = result
                return result
        solvers = [AuditedSolver(env, base_pose=a.robot.pose, multi_robot_id=i, debug=False,
                    vis=False, visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=report["execution_configuration"]["solver_joint_vel_limit_rad_s"],
                    joint_acc_limits=report["execution_configuration"]["solver_joint_acc_limit_rad_s2"])
                   for i, a in enumerate(agents)]
        if args.task_id == "TwoRobotPickFruitsToPlate-v1":
            from .support_surface_collision import SourceSupportDualPayloadCollisionChecker
            checker = SourceSupportDualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=targets, source_supports=adapter.initial_supports)
        elif args.task_id == "TwoRobotPickPillToRegions-v1":
            from .pill_support_collision import PillDualPayloadCollisionChecker
            checker = PillDualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=targets)
        else:
            checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=targets)
        report["initial_collision_coverage"] = json_value(checker.coverage)

        def begin(name: str) -> None:
            report["stage"] = captured.phase = name
            save_report(output / "report.json", report)

        def finish(name: str, result: dict[str, Any]) -> bool:
            report["stages"].append({"name": name, **result})
            if result["status"] != "passed": report.update(status="failed", reason=f"{name}:{result['status']}")
            save_report(output / "report.json", report)
            return result["status"] == "passed"

        def plan(name: str, poses: tuple[Any, Any], *, closed: bool, limit_phase: str | None = None) -> Any:
            phase_limits = task_phase_limits(report["execution_configuration"], args.task_id, name if limit_phase is None else limit_phase)
            phase_limits['recorded_phase'] = name
            compact_set_phase_limits(solvers, phase_limits)
            report.setdefault("phase_limit_configurations", []).append(phase_limits)
            before = int(as_numpy(raw.elapsed_steps)[0])
            planning_start = time.perf_counter()
            raw_schedule = plan_dual_pose(*solvers, poses, gripper_commands=(-1, -1) if closed else (1, 1))
            if int(as_numpy(raw.elapsed_steps)[0]) != before: raise RuntimeError("Planning advanced environment")
            raw_results = tuple(solver.last_native_result if target is not None else None for solver, target in zip(solvers, poses))
            traces = tuple(solver.last_native_parameterization if target is not None else None for solver, target in zip(solvers, poses))
            retimed, retiming = compact_retime_pair(raw_results, traces,
                velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"])
            trajectories = tuple(source if result is None else ArmTrajectory(source.action_key, source.joint_names,
                                  result["time"], result["position"], source.gripper_command)
                                 for source, result in zip(raw_schedule.trajectories, retimed))
            schedule = build_hold_schedule(*trajectories, control_timestep_s=raw_schedule.control_timestep_s)
            report.setdefault("phase_time_parameterization", []).append({"phase": name, **retiming,
                "raw_scheduled_steps": raw_schedule.num_steps, "retimed_scheduled_steps": schedule.num_steps})
            derivatives = []
            derivative_arrays = {}
            for index, (solver, target) in enumerate(zip(solvers, poses)):
                if target is None:
                    derivatives.append({"status": "passed", "stationary_hold": True})
                    continue
                result = retimed[index]
                audit = compact_kinematic_audit(result,
                    velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                    acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"])
                audit.update(solver_velocity_limit_rad_s=solver.joint_vel_limits,
                    solver_acceleration_limit_rad_s2=solver.joint_acc_limits,
                    solver_limit_fraction=SOLVER_LIMIT_FRACTION,
                    source_timestamp_scaling_applied=retiming["source_timestamp_scaling_applied"],
                    raw_native_derivative_audit=compact_kinematic_audit(raw_results[index],
                        velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                        acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"]))
                derivatives.append(audit)
                arm_name = "left" if index == 0 else "right"
                derivative_arrays[f"{arm_name}_velocity"] = np.asarray(result["velocity"])
                derivative_arrays[f"{arm_name}_acceleration"] = np.asarray(result["acceleration"])
                for field in ("time", "position", "velocity", "acceleration", "duration"):
                    derivative_arrays[f"{arm_name}_raw_{field}"] = np.asarray(raw_results[index][field])
                    derivative_arrays[f"{arm_name}_retimed_{field}"] = np.asarray(result[field])
                for field in ("native_path", "path_knots_s", "path_coefficients", "grid_s", "grid_speed_squared",
                              "grid_acceleration", "grid_time", "solver_velocity_limits", "solver_acceleration_limits"):
                    derivative_arrays[f"{arm_name}_{field}"] = traces[index][field]
            report.setdefault("kinematic_audits", []).append({"phase": name, "phase_limits": phase_limits, "arms": derivatives})
            np.savez(output / f"{name}-plan.npz", **derivative_arrays, left_time=schedule.trajectories[0].times_s,
                     right_time=schedule.trajectories[1].times_s, left_position=schedule.trajectories[0].positions_rad,
                     right_position=schedule.trajectories[1].positions_rad, target_times=schedule.target_times_s,
                     left_commands=schedule.left_commands, right_commands=schedule.right_commands)
            if any(audit["status"] != "passed" for audit in derivatives):
                raise RuntimeError(f"Native planner velocity/acceleration limit violation in {name}")
            report.setdefault("planning", []).append({"phase": name, "phase_limits": phase_limits,
                "environment_steps": 0, "scheduled_steps": schedule.num_steps,
                "plan_retime_audit_wall_seconds": time.perf_counter() - planning_start,
                "native_backend_wall_seconds": sum(s.last_native_planning_seconds for s, p in zip(solvers, poses) if p is not None),
                "source_durations_s": [t.duration_s for t in schedule.trajectories],
                "target_tcp_poses": [None if p is None else np.r_[p.p, p.q].tolist() for p in poses]})
            return schedule

        def hook_context() -> dict[str, Any]:
            return dict(args=args, output=output, report=report, raw=raw, capture=captured,
                        solvers=solvers, checker=checker, adapter=adapter, plan=plan,
                        begin=begin, finish=finish, substeps=substeps, ticks=ticks)

        grasp_pairs = adapter.grasp_poses()
        if grasp_pose_hook is not None:
            before_hook = _state_arrays(raw.get_state_dict())
            before_elapsed = int(as_numpy(raw.elapsed_steps)[0])
            grasp_pairs = grasp_pose_hook(**hook_context(), grasp_pairs=grasp_pairs)
            after_hook = _state_arrays(raw.get_state_dict())
            if (int(as_numpy(raw.elapsed_steps)[0]) != before_elapsed or set(before_hook) != set(after_hook)
                    or any(not np.array_equal(before_hook[k], after_hook[k]) for k in before_hook)):
                raise RuntimeError('Grasp target adaptation changed physical state')
        report["grasp_tcp_poses"] = [[np.r_[p.p, p.q].tolist() for p in pair] for pair in grasp_pairs]
        for index, name, contact in ((0, "pregrasp", False), (1, "descend", True)):
            begin(name)
            schedule = plan(name, (grasp_pairs[0][index], grasp_pairs[1][index]), closed=False)
            if not finish(name, execute_dual_reach(captured, schedule, checker, allow_target_contacts=(contact, contact))): return
        begin("close_and_confirm")
        closed = close_dual_and_confirm(captured, checker, targets)
        if not finish("close_and_confirm", closed):
            if close_failure_hook is None or close_failure_hook(**hook_context(), close_result=closed) is not True:
                return
            report.setdefault('recovered_phase_failures', []).append({'phase':'close_and_confirm',
                'original_status':closed['status'],'original_reason':closed.get('reason'),
                'recovery_final_environment_step':int(as_numpy(raw.elapsed_steps)[0])})
            report.update(status='running'); report.pop('reason', None)
        begin("attach_collision_prediction")
        checker.attach()
        attachment_count += 1
        assert attachment_count == 1
        references = np.asarray(checker.attachments).copy()
        report["attachment_capture"] = {"count": attachment_count, "environment_step": int(as_numpy(raw.elapsed_steps)[0]), "T_tcp_objects": references.tolist(), "initial_actor_origin_z_m": [float(as_numpy(a.pose.p)[0,2]) for a in targets], "physical_weld": False}
        report["attachment_T_tcp_objects"] = references.tolist()
        report["attachment_initial_coverage"] = json_value(checker.coverage)
        carry_q = adapter.carry_quaternions()
        release_poses = adapter.release_poses(carry_quaternions=carry_q)
        release_xyz = [np.asarray(p.p, dtype=float) for p in release_poses]
        route = compact_route_positions([as_numpy(a.tcp.pose.p)[0] for a in agents], release_xyz, route=args.route)
        original_q = [as_numpy(a.tcp.pose.q)[0].copy() for a in agents]
        report["task_configuration_after_grasp"] = adapter.configuration()
        report["compact_route_world_positions_m"] = json_value(route)
        report["selected_release_tcp_xyz_m"] = [p.tolist() for p in release_xyz]
        lift_poses = tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["lift"], original_q))
        begin("lift_and_hold")
        schedule = plan("lift_and_hold", lift_poses, closed=True)
        captured.observe_substeps = True
        hover = report["execution_configuration"]["airborne_hover_steps"]
        lifted = execute_dual_held_move(captured, schedule, checker, targets,
            min_height_gain_m=(.10, .10), hold_steps=hover)
        if not finish("lift_and_hold", lifted): return
        if hover:
            hover_steps = [int(lifted["initial_elapsed_steps"][0]) + row["step"] for row in lifted["trace"] if row["phase"] == "hover"]
            report["physics_hold_joint_audit"] = hold_joint_substep_audit(substeps, hover_steps)
            if report["physics_hold_joint_audit"]["status"] != "passed":
                report.update(status="failed", reason="physics_hold_joint_stability"); return
        else:
            report["physics_hold_joint_audit"] = {"status": "not_applicable", "observed_hover_steps": 0,
                "reason": "execution profile omits the separate airborne diagnostic hover; old hover regression preserved"}
        if phase_hook is not None:
            continue_task = phase_hook(**hook_context(), references=references, route=route,
                                       carry_q=carry_q, original_q=original_q)
            if continue_task is not True: return
        waypoints = []
        if route["raise"] is not None:
            waypoints.append(("carry_raise", tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["raise"], original_q))))
        waypoints += [
            ("carry_orient_and_transport", tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["above_baskets"], carry_q))),
            ("carry_lower", tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["release"], carry_q)))]
        for name, poses in waypoints:
            begin(name)
            schedule = plan(name, poses, closed=True)
            if not finish(name, execute_dual_held_move(captured, schedule, checker, targets)): return
        report["attachment_final_coverage"] = json_value(checker.coverage)
        report["held_physics_substep_audit"] = held_substep_audit(substeps, references)
        for row, name in zip(report["held_physics_substep_audit"]["objects"], adapter.target_names):
            row['actor'] = name
        if report["held_physics_substep_audit"]["status"] != "passed":
            report.update(status="failed", reason="held_physics_substep_envelope"); return
        checker.detach()
        begin("release")
        released = release_dual_and_confirm(captured, checker, targets)
        finish("release", released)
        if args.mode == "retain-right":
            report["retain_right_control_verified"] = retain_right_valid(released, ticks, keys)
            report["status"] = "passed" if report["retain_right_control_verified"] else "failed"
            report["expected_outcome"] = "left releases; right stays grasped; release_timeout; no retreat/verification"
            return
        if released["status"] != "passed": return
        retreat_initial = np.asarray([as_numpy(a.tcp.pose.p)[0].copy() for a in agents])
        retreat_poses = tuple(sapien.Pose(xyz + [0, 0, .08], as_numpy(a.tcp.pose.q)[0]) for xyz, a in zip(retreat_initial, agents))
        begin("retreat")
        schedule = plan("retreat", retreat_poses, closed=False)
        if not finish("retreat", execute_dual_reach(captured, schedule, checker)): return
        actual_retreat = np.asarray([as_numpy(a.tcp.pose.p)[0] for a in agents]) - retreat_initial
        report["actual_retreat_displacement_m"] = actual_retreat.tolist()
        report["actual_retreat_vertical_gain_m"] = actual_retreat[:, 2].tolist()
        if np.any(actual_retreat[:, 2] < .08):
            report.update(status="failed", reason="actual_retreat_below_8cm"); return

        begin("verify_placement")
        result = verify_dual_placement(captured, checker, targets, official_predicate_keys=adapter.official_predicate_keys, region_predicates=adapter.region_predicates())
        if finish("verify_placement", result):
            evaluation = json_value(raw.evaluate())
            report["physical_task_outcome"] = "passed" if bool(np.asarray(evaluation["success"]).all()) and not report["fault_events"] else "failed"
            report["official_200_step_complete"] = bool(args.max_episode_steps == 200 and report["physical_task_outcome"] == "passed" and int(as_numpy(raw.elapsed_steps)[0]) < 200)
            report["status"] = report["physical_task_outcome"]
        report["stage"] = "finished"
    finally:
        if checker is not None:
            try: report['final_collision_coverage'] = json_value(checker.coverage)
            except Exception as exc:
                report.setdefault('artifact_errors', []).append(f'final collision coverage: {type(exc).__name__}: {exc}')
        finalize_physics_capture(env, raw, captured, output, report, frames, ticks, substeps, keys,
                                 original_hook, attachment_count=attachment_count)

def self_test() -> dict[str, Any]:
    termination_self_test()
    calls = []
    assert timed_native_call(lambda: 'planned', calls, phase='fake', arm_index=0) == 'planned'
    def failed_call(): raise ValueError('synthetic planning failure')
    try: timed_native_call(failed_call, calls, phase='failure', arm_index=1)
    except ValueError: pass
    else: raise AssertionError('Native planning exception was suppressed')
    assert [r['status'] for r in calls] == ['returned', 'raised']
    assert all(r['wall_seconds'] >= 0 for r in calls)
    initial=('table','source_plate');destination=('basket','target_plate')
    assert contact_class({'actor':'table'},0,False,'lift_and_hold',initial,destination)=='initial_support'
    assert contact_class({'actor':'table'},0,True,'carry_lower',initial,destination)=='forbidden'
    assert contact_class({'actor':'source_plate'},1,False,'descend',initial,destination)=='initial_support'
    assert contact_class({'actor':'source_plate'},1,True,'release',initial,destination)=='forbidden'
    assert contact_class({'actor':'target_plate'},1,True,'release',initial,destination)=='destination_support'
    assert contact_class({'actor':'target_plate'},1,False,'carry_lower',initial,destination)=='forbidden'
    assert contact_class({'robot_index':0,'link_name':'panda_leftfinger'},0,True,'release',initial,destination)=='own_finger_allowed'
    assert contact_class({'robot_index':1,'link_name':'panda_leftfinger'},0,False,'descend',initial,destination)=='forbidden'
    point={'separation_m':.039,'impulse_Ns':[0.,0.,0.]};assert not physical_contact_points([point])
    assert physical_contact_points([{**point,'impulse_Ns':[1e-9,0.,0.]}])
    assert physical_contact_points([{**point,'separation_m':0.}])
    arrays=_state_arrays({'actors':{'one':np.zeros((1,13))},'articulations':{'left':np.ones((1,31))}})
    assert sorted(v.shape for v in arrays.values())==[(1,13),(1,31)]
    assert checked_geometry([[0.,0.,0.]], [0.,0.,0.,1.,0.,0.,0.])[0].shape == (1,3)
    assert checked_geometry(np.empty((0,3)), [0.,0.,0.,1.,0.,0.,0.])[0].shape == (0,3)
    for vertices, pose in (([[0,0]], [0,0,0,1,0,0,0]), ([[0,0,float('nan')]], [0,0,0,1,0,0,0]), ([[0,0,0]], [0,0,0,0,0,0,0])):
        try: checked_geometry(vertices, pose)
        except ValueError: pass
        else: raise AssertionError('Invalid geometry accepted')
    return {'status':'passed','contact_boundary_cases':11,'geometry_boundary_cases':5,'shapes':{'single_actor_state':[1,13],'single_robot_state':[1,31],'paired_action':[16],'single_geometry':[1,3],'empty_marker_geometry':[0,3]},'environment_steps':0,'physics_steps':0}


def main() -> int:
    faulthandler.enable();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--task-id',choices=TASKS,default=TASKS[1]);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2));return 0
    signal.signal(signal.SIGTERM,termination_handler)
    if args.output_dir is None: parser.error('An unused explicit output directory is required')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',os.environ.get('CUDA_VISIBLE_DEVICES','')):parser.error('GPU UUID required')
    if any(os.environ.get(k)=='1' for k in ('MANI_SKILL_L1','MANI_SKILL_L2','MANI_SKILL_L3')):parser.error('L0 only')
    args.route='merge-lift';args.execution_profile='execution';args.mode='positive'
    for name, value in task_execution_parameters(args.task_id).items(): setattr(args,name,value)
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((EVIDENCE/'prechange-guard.json').read_text())
    files=[Path(p) for p in baseline['baseline_source_sha256']]+[Path(__file__),HERE/'multitask_adapters.py',HERE/'multitask_recording.py']
    if args.task_id==TASKS[2]: files += [HERE/'support_surface_collision.py']
    if args.task_id==TASKS[1]: files += [HERE/'pill_support_collision.py']
    sources={p.name:p.read_bytes() for p in files};snapshot=output/'source';snapshot.mkdir()
    for name,data in sources.items():(snapshot/name).write_bytes(data)
    configuration=compact_configuration(args)
    for key in ('release_tcp_offsets_basket_m','release_pose_policy','release_pose_evidence_scope'):configuration.pop(key,None)
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'task':args.task_id,'seed':args.seed,
        'method':'sync','mode':'positive','execution_configuration':configuration,'stages':[],
        'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},'source_snapshot':str(snapshot),
        'official_default_max_episode_steps':200,'max_episode_steps':200,'official_200_step_complete':False,
        'physical_pose_assignment':False,'physical_attachment_created':False,'collision_masks_modified':False,
        'physical_contact_definition':'separation<=0 OR any nonzero impulse; positive-gap zero-impulse manifolds retained as proximity'}
    save_report(output/'report.json',report);started=time.monotonic()
    try: native_run(args,output,report)
    except Exception as exc:report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
        planning_timing_scope='Instrumented move_to_pose call including native TOPP raw NPZ export and sync TOPP reconstruction; not pure MPLib backend time')
    if any(hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=h for name,h in report['source_sha256'].items()):
        report.update(status='failed',source_identity_violation=True)
    save_report(output/'report.json',report);print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
