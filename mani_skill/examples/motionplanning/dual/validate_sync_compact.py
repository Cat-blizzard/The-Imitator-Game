"""Evaluate compact paired manipulation under the unchanged official clock.

Contract: physical left/right actions are [7 joint radians, normalized gripper]
per 0.05 s tick; physics remains 0.01 s. Geometry merges lift/raise and turn/carry.
Version analytic-phase-retiming-v1 preserves raw native output, bounds its
piecewise source curve, and slows both arms by one common chain-rule factor.
The default TimeLimit is 200;
its truncation still stops immediately, so completion should precede tick 200.
All original collision, grasp, height, release and final-placement gates remain.
The execution profile omits the separate 20-tick airborne diagnostic hover;
--execution-profile diagnostic-hover retains that independent diagnostic gate.
No physical state/pose assignment, weld or collision-mask mutation is used.
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
import re
import resource
import time
import traceback
from typing import Any

import numpy as np

from .validate_grasp import TASK, grasp_poses, json_value, recording_summary, save_report
from .validate_banana import banana_grasp_poses

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
ACTOR_NAMES = ("apple_out", "banana_out")


# Reuse frozen acceptance logic; only this new entry changes route/configuration.
from .validate_sync import (continue_success_only, applied_action, retain_right_valid,
                            physical_region, held_substep_audit, hold_joint_substep_audit,
                            self_test as original_self_test)


# Compatibility exports: implementations live in shared runtime modules.
from .phase_kinematics import (ROUTES, SOLVER_LIMIT_FRACTION, RETIMING_POLICY, TRANSPORT_CLEARANCE_ABOVE_RELEASE_M, compact_phase_limits, compact_set_phase_limits, compact_route_positions, compact_kinematic_audit, compact_topp_reconstruction, _compact_real_roots, compact_continuous_derivative_bounds, compact_retime_pair, retiming_self_test)
from .task_targets import (RELEASE_TCP_OFFSETS_BASKET_M)

PROFILES = ("execution", "diagnostic-hover")
# Native TOPP interpolation can marginally exceed its own requested bound.
# Solve inside the nominal envelope, while validating unchanged nominal gates.
RELEASE_POLICY = "geometry-informed-v1"
ROUTE_GEOMETRY_POLICY = "compact-clearance-v2"


def compact_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Validate explicit experiment choices before any native environment exists."""
    if args.route not in ROUTES or args.execution_profile not in PROFILES:
        raise ValueError("Unknown compact route or diagnostic profile")
    if (isinstance(args.max_episode_steps, bool) or not isinstance(args.max_episode_steps, int)
            or not 200 <= args.max_episode_steps <= 400):
        raise ValueError("Budget must be an integer in [200,400]; only 200 is official")
    speed, acceleration = float(args.joint_vel_limit), float(args.joint_acc_limit)
    if not np.isfinite([speed, acceleration]).all() or min(speed, acceleration) <= 0:
        raise ValueError("Finite positive joint velocity/acceleration limits are required")
    approach_speed_override = getattr(args, "approach_vel_limit", None)
    approach_acc_override = getattr(args, "approach_acc_limit", None)
    approach_speed = speed if approach_speed_override is None else float(approach_speed_override)
    approach_acceleration = acceleration if approach_acc_override is None else float(approach_acc_override)
    if (not np.isfinite([approach_speed, approach_acceleration]).all()
            or min(approach_speed, approach_acceleration) <= 0):
        raise ValueError("Approach overrides must be finite positive limits")
    release_offsets = np.asarray(RELEASE_TCP_OFFSETS_BASKET_M, dtype=float)
    if release_offsets.shape != (2, 3) or not np.isfinite(release_offsets).all():
        raise ValueError("Release TCP basket offsets must be finite paired XYZ values")
    return {"route": args.route, "execution_profile": args.execution_profile,
            # Previously hardcoded route values are now exposed for the ledger.
            "transport_clearance_above_release_m": TRANSPORT_CLEARANCE_ABOVE_RELEASE_M,
            "minimum_commanded_lift_m": .12,
            "route_geometry_policy": ROUTE_GEOMETRY_POLICY,
            "airborne_hover_steps": 20 if args.execution_profile == "diagnostic-hover" else 0,
            "preparation_steps": 10, "min_actual_height_gain_m": [.10, .10],
            "release_stable_steps": 3, "retreat_command_m": .08,
            "release_tcp_offsets_basket_m": release_offsets.tolist(),
            "release_pose_policy": RELEASE_POLICY,
            "release_pose_evidence_scope": "fixed common recipe selected from cooked geometry endpoint checks; full candidate and measured motion collision gates remain required",
            "final_common_placement_steps": 20,
            "max_episode_steps": args.max_episode_steps,
            "official_budget_candidate": args.max_episode_steps == 200,
            "joint_vel_limit_rad_s": speed, "joint_acc_limit_rad_s2": acceleration,
            "approach_vel_limit_override_rad_s": approach_speed_override,
            "approach_acc_limit_override_rad_s2": approach_acc_override,
            "approach_joint_vel_limit_rad_s": approach_speed,
            "approach_joint_acc_limit_rad_s2": approach_acceleration,
            "approach_limit_phases": ["pregrasp", "descend"],
            "solver_limit_fraction": SOLVER_LIMIT_FRACTION,
            "solver_joint_vel_limit_rad_s": speed * SOLVER_LIMIT_FRACTION,
            "solver_joint_acc_limit_rad_s2": acceleration * SOLVER_LIMIT_FRACTION,
            "solver_margin_reason": "solve 1 percent inside nominal derivative limits; nominal output gates and tolerance remain unchanged",
            "joint_limit_interpretation": "planner outputs checked; velocity bounded by URDF; no hardware acceleration certification",
            "source_timestamp_scaling": True, "time_parameterization_policy": RETIMING_POLICY,
            "native_source_timestamp_scaling": False,
            "retiming_scope": "one common phase factor from numerically evaluated analytic source-curve extrema; not physical actuator or controller continuous-time certification",
            "control_hz": 20, "physics_hz": 100,
            "acceptance_difference_from_original": (
                "separate airborne 20-tick hover omitted; held 100Hz slip, grasp/height, release, retreat and final20 unchanged"
                if args.execution_profile == "execution" else "airborne 20-tick hover and its original audit retained")}








def compact_urdf_velocity_limits(urdf_path: Any, joint_names: Any,
                                 requested_limit: float) -> dict[str, Any]:
    """Require all seven named arm URDF velocity limits; never infer acceleration."""
    import xml.etree.ElementTree as ET
    names = tuple(joint_names)
    value = float(requested_limit)
    if len(names) != 7 or len(set(names)) != 7 or not np.isfinite(value) or value <= 0:
        raise ValueError("Need seven unique controller joints and a positive speed")
    contents = Path(urdf_path).read_bytes()
    nodes = {node.attrib["name"]: node for node in ET.fromstring(contents).findall("joint")}
    limits = []
    for name in names:
        limit = None if name not in nodes else nodes[name].find("limit")
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"URDF joint {name} lacks an explicit velocity limit")
        bound = float(limit.attrib["velocity"])
        if not np.isfinite(bound) or bound <= 0 or value > bound + 1e-12:
            raise ValueError(f"Requested velocity {value} exceeds/invalidates URDF joint {name} limit {bound}")
        limits.append(bound)
    return {"urdf_path": str(urdf_path), "urdf_sha256": hashlib.sha256(contents).hexdigest(),
            "joint_names": list(names), "velocity_limits_rad_s": limits,
            "requested_velocity_limit_rad_s": value,
            "acceleration_limit_source": "explicit planner configuration; URDF has no claimed hardware certification"}












def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import sapien
    import torch
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
    frames, ticks, substeps, keys = [], [], [], []
    report["fault_events"] = []
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos", sim_backend="cpu",
                        render_backend="gpu", render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
                            source_type="motionplanning", source_desc=f"compact synchronized apple and banana; route={args.route}; profile={args.execution_profile}; mode={args.mode}")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        agents, targets = (raw.left_agent, raw.right_agent), (raw.apple_out, raw.banana_out)
        keys = list(raw.agent.agents_dict)
        if len(keys) != 2 or any(raw.agent.agents_dict[k] is not a for k, a in zip(keys, agents)):
            raise ValueError("Physical left/right action identity mismatch")
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), stages=[], action_key_order=keys)

        def measurement() -> dict[str, Any]:
            objects = []
            for name, agent, actor in zip(ACTOR_NAMES, agents, targets):
                forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, actor))[0]
                          for link in (agent.finger1_link, agent.finger2_link)]
                objects.append({"actor": name, "pose": as_numpy(actor.pose.raw_pose)[0].tolist(),
                                "linear_velocity_m_s": as_numpy(actor.linear_velocity)[0].tolist(),
                                "angular_velocity_rad_s": as_numpy(actor.angular_velocity)[0].tolist(),
                                "is_grasping": bool(as_numpy(agent.is_grasping(actor))[0]),
                                "finger_contact_force_norms_N": [float(np.linalg.norm(f)) for f in forces]})
            return {"left_qpos": as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                    "right_qpos": as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                    "tcp_poses": [as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents], "objects": objects}

        class Capture:
            unwrapped = raw
            phase = "preparation"
            observe_substeps = False
            pending_control_step = 0

            def frame(self) -> None:
                frame = as_numpy(env.render())
                if frame.ndim == 4 and frame.shape[0] == 1: frame = frame[0]
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError("Expected HWC uint8 rendering")
                frames.append(frame.copy())

            def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
                actual, inject = applied_action(action, mode=args.mode, phase=self.phase, keys=keys)
                self.pending_control_step = int(as_numpy(raw.elapsed_steps)[0]) + 1
                if inject:
                    report["fault_events"].append({"control_step": self.pending_control_step, "type": "retain_right_gripper_during_release"})
                obs, reward, terminated, truncated, info = env.step(actual)
                done, continued = continue_success_only(terminated, truncated, info)
                ticks.append({"phase": self.phase, "environment_step": int(as_numpy(raw.elapsed_steps)[0]),
                              "requested_actions": json_value(action), "actions": json_value(actual), "fault_injected": inject,
                              "native_terminated": bool(as_numpy(terminated).reshape(-1)[0]),
                              "native_truncated": bool(as_numpy(truncated).reshape(-1)[0]),
                              "success_only_termination_continued": continued,
                              "official_evaluation": json_value(info), **measurement()})
                self.frame()
                return obs, reward, done, truncated, info

        captured = Capture()
        original_hook = raw._after_simulation_step

        def record_substep() -> None:
            original_hook()
            if captured.observe_substeps:
                substeps.append({"substep": len(substeps) + 1, "phase": captured.phase,
                                 "control_step": captured.pending_control_step, "sim_timestep_s": float(raw.sim_timestep),
                                 "object_poses": [as_numpy(a.pose.raw_pose)[0].tolist() for a in targets],
                                 "tcp_poses": [as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents],
                                 "left_qpos": as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                                 "right_qpos": as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                                 "left_qvel": as_numpy(agents[0].robot.get_qvel())[0].tolist(),
                                 "right_qvel": as_numpy(agents[1].robot.get_qvel())[0].tolist(),
                                 "object_linear_velocities": [as_numpy(a.linear_velocity)[0].tolist() for a in targets],
                                 "object_angular_velocities": [as_numpy(a.angular_velocity)[0].tolist() for a in targets]})

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
        for _ in range(10):
            _, _, done, truncated, _ = captured.step(initial_hold)
            if done or bool(as_numpy(truncated).reshape(-1)[0]): raise RuntimeError("Environment ended during preparation")
        report["preparation_steps"] = 10
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
                result = super().move_to_pose_with_screw(pose, dry_run=True,
                    refine_steps=refine_steps, other_gripper_state=other_gripper_state)
                self.last_native_result = result
                return result
        solvers = [AuditedSolver(env, base_pose=a.robot.pose, multi_robot_id=i, debug=False,
                    vis=False, visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=report["execution_configuration"]["solver_joint_vel_limit_rad_s"],
                    joint_acc_limits=report["execution_configuration"]["solver_joint_acc_limit_rad_s2"])
                   for i, a in enumerate(agents)]
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

        def plan(name: str, poses: tuple[Any, Any], *, closed: bool) -> Any:
            phase_limits = compact_phase_limits(report["execution_configuration"], name)
            compact_set_phase_limits(solvers, phase_limits)
            report.setdefault("phase_limit_configurations", []).append(phase_limits)
            before = int(as_numpy(raw.elapsed_steps)[0])
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
                "source_durations_s": [t.duration_s for t in schedule.trajectories],
                "target_tcp_poses": [None if p is None else np.r_[p.p, p.q].tolist() for p in poses]})
            return schedule

        grasp_pairs = (grasp_poses(raw), banana_grasp_poses(raw))
        for index, name, contact in ((0, "pregrasp", False), (1, "descend", True)):
            begin(name)
            schedule = plan(name, (grasp_pairs[0][index], grasp_pairs[1][index]), closed=False)
            if not finish(name, execute_dual_reach(captured, schedule, checker, allow_target_contacts=(contact, contact))): return
        begin("close_and_confirm")
        if not finish("close_and_confirm", close_dual_and_confirm(captured, checker, targets)): return
        begin("attach_collision_prediction")
        checker.attach()
        references = np.asarray(checker.attachments).copy()
        report["attachment_T_tcp_objects"] = references.tolist()
        report["attachment_initial_coverage"] = json_value(checker.coverage)
        baskets = (raw.basket_apple, raw.basket_banana)
        release_offsets = report["execution_configuration"]["release_tcp_offsets_basket_m"]
        release_xyz = [as_numpy(b.pose.p)[0] + offset for b, offset in zip(baskets, release_offsets)]
        route = compact_route_positions([as_numpy(a.tcp.pose.p)[0] for a in agents], release_xyz, route=args.route)
        original_q = [as_numpy(a.tcp.pose.q)[0].copy() for a in agents]
        carry_q = [original_q[0], (sapien.Pose(q=[2**-.5, 0, 0, 2**-.5]) * sapien.Pose(q=original_q[1])).q]
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

        def predicate(index: int) -> Any:
            def measure(actor: Any) -> dict[str, Any]:
                return physical_region(as_numpy(raw._get_actor_geom_center(actor))[0],
                    as_numpy(world_aabb_from_local_mesh(baskets[index], raw.device))[0],
                    as_numpy(world_aabb_from_local_mesh(actor, raw.device))[0])
            return measure
        begin("verify_placement")
        result = verify_dual_placement(captured, checker, targets, region_predicates=(predicate(0), predicate(1)))
        if finish("verify_placement", result):
            evaluation = json_value(raw.evaluate())
            report["physical_task_outcome"] = "passed" if bool(np.asarray(evaluation["success"]).all()) and not report["fault_events"] else "failed"
            report["official_200_step_complete"] = bool(args.max_episode_steps == 200 and report["physical_task_outcome"] == "passed" and int(as_numpy(raw.elapsed_steps)[0]) < 200)
            report["status"] = report["physical_task_outcome"]
        report["stage"] = "finished"
    finally:
        errors = report.setdefault("artifact_errors", [])
        if raw is not None:
            for name, read in (("total_environment_steps", lambda: int(as_numpy(raw.elapsed_steps)[0])),
                               ("torch_cuda_initialized", torch.cuda.is_initialized),
                               ("official_task_evaluation", lambda: json_value(raw.evaluate()))):
                try: report[name] = read()
                except Exception as exc: errors.append(f"final {name}: {type(exc).__name__}: {exc}")
            if original_hook is not None: raw._after_simulation_step = original_hook
        for filename, rows in (("ticks.jsonl", ticks), ("physics-substeps.jsonl", substeps)):
            try:
                with (output / filename).open("w") as f:
                    for row in rows: f.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            except Exception as exc: errors.append(f"{filename}: {type(exc).__name__}: {exc}")
        if env is not None:
            try:
                if getattr(env, "_trajectory_buffer", None) is not None: env.flush_trajectory()
            except Exception as exc: errors.append(f"trajectory: {type(exc).__name__}: {exc}")
            finally:
                try: env.close()
                except Exception as exc: errors.append(f"close: {type(exc).__name__}: {exc}")
            try:
                actions = np.asarray([np.concatenate([t["actions"][k] for k in keys]) for t in ticks]) if ticks else None
                report["recording"] = recording_summary(output / "trajectory.h5", len(ticks), actions)
                report["recording"]["action_key_order"] = keys
            except Exception as exc: errors.append(f"record check: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "execution.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20, fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
                report["video"] = {"path": str(output / "execution.mp4"), "fps": 20, "padding_frames_before": 10, "padding_frames_after": 20}
            except Exception as exc: errors.append(f"video: {type(exc).__name__}: {exc}")
        report["success_only_termination_continuation_ticks"] = sum(t["success_only_termination_continued"] for t in ticks)
        report["phase_step_counts"] = {phase: sum(t["phase"] == phase for t in ticks) for phase in dict.fromkeys(t["phase"] for t in ticks)}
        report["first_official_success_step"] = next((t["environment_step"] for t in ticks
            if bool(np.asarray(t["official_evaluation"].get("success", False)).all())), None)
        report["official_200_step_complete"] = bool(report.get("official_200_step_complete", False))
        if errors: report.update(status="failed", official_200_step_complete=False)




def self_test() -> dict[str, Any]:
    """Synthetic geometry, parser config and native derivative/URDF checks only."""
    import tempfile
    original = original_self_test()
    args = argparse.Namespace(route="merge-lift", execution_profile="execution", max_episode_steps=200,
        joint_vel_limit=.9, joint_acc_limit=.9)
    config = compact_configuration(args)
    assert config["airborne_hover_steps"] == 0 and config["max_episode_steps"] == 200
    assert config["final_common_placement_steps"] == 20 and config["source_timestamp_scaling"]
    assert config["time_parameterization_policy"] == RETIMING_POLICY and not config["native_source_timestamp_scaling"]
    assert np.asarray(config["release_tcp_offsets_basket_m"]).shape == (2, 3)
    assert np.isfinite(config["release_tcp_offsets_basket_m"]).all()
    assert config["release_pose_policy"] == RELEASE_POLICY
    assert config["joint_vel_limit_rad_s"] == config["joint_acc_limit_rad_s2"] == .9
    assert config["solver_limit_fraction"] == .99
    assert config["solver_joint_vel_limit_rad_s"] == config["solver_joint_acc_limit_rad_s2"] == .891
    assert compact_phase_limits(config, "pregrasp")["nominal_velocity_limit_rad_s"] == .9
    args.joint_vel_limit, args.joint_acc_limit = 1.5, 2.4
    args.approach_vel_limit, args.approach_acc_limit = .9, .9
    separated = compact_configuration(args)
    approach = compact_phase_limits(separated, "pregrasp")
    assert compact_phase_limits(separated, "descend")["nominal_acceleration_limit_rad_s2"] == .9
    assert approach["nominal_velocity_limit_rad_s"] == approach["nominal_acceleration_limit_rad_s2"] == .9
    assert approach["solver_velocity_limit_rad_s"] == approach["solver_acceleration_limit_rad_s2"] == .891
    for phase in ("lift_and_hold", "carry_raise", "carry_orient_and_transport", "carry_lower", "retreat"):
        limits = compact_phase_limits(separated, phase)
        assert limits["nominal_velocity_limit_rad_s"] == 1.5
        assert limits["nominal_acceleration_limit_rad_s2"] == 2.4
    from types import SimpleNamespace
    fake_solvers = [SimpleNamespace(planner=SimpleNamespace(move_group_joint_indices=list(range(7)))) for _ in range(2)]
    compact_set_phase_limits(fake_solvers, approach)
    assert all(np.array_equal(s.planner.joint_vel_limits, np.full(7, .891)) for s in fake_solvers)
    compact_set_phase_limits(fake_solvers, compact_phase_limits(separated, "retreat"))
    assert all(np.array_equal(s.planner.joint_acc_limits, np.full(7, 2.4 * .99)) for s in fake_solvers)
    args.joint_vel_limit = args.joint_acc_limit = .9
    args.approach_vel_limit = args.approach_acc_limit = None
    args.execution_profile = "diagnostic-hover"
    assert compact_configuration(args)["airborne_hover_steps"] == 20
    args.max_episode_steps = 400
    assert not compact_configuration(args)["official_budget_candidate"]
    def reject(call: Any) -> None:
        try: call()
        except (ValueError, TypeError): return
        raise AssertionError("Invalid compact configuration or input was accepted")
    args.approach_vel_limit = -1.; reject(lambda: compact_configuration(args))
    args.approach_vel_limit = None
    args.max_episode_steps = 199; reject(lambda: compact_configuration(args))
    args.max_episode_steps = 200; args.joint_vel_limit = float("nan")
    reject(lambda: compact_configuration(args))
    current = [[-.2, -.1, .1], [.2, -.1, .11]]
    release = [[-.3, .2, .17], [.3, .2, .17]]
    merged = compact_route_positions(current, release, route="merge-lift")
    separate = compact_route_positions(current, release, route="merge-transport")
    assert merged["raise"] is None and separate["raise"].shape == (2, 3)
    assert np.array_equal(merged["lift"][:, :2], np.asarray(current)[:, :2])
    assert np.allclose(merged["above_baskets"][:, 2], [.23, .23])
    assert np.allclose(separate["lift"][:, 2], [.22, .23])
    assert np.array_equal(merged["release"], release)
    assert np.all(merged["lift"][:, 2] - np.asarray(current)[:, 2] >= .12 - 1e-12)
    # Isolate the unchanged lift floor from the changed transport clearance.
    mixed_current = np.array([[-.2, -.1, .5], [.2, -.1, .1]])
    mixed_release = np.array([[-.3, .2, .02], [.3, .2, .35]])
    initial_current, initial_release = mixed_current.copy(), mixed_release.copy()
    for selected_route in ROUTES:
        mixed = compact_route_positions(mixed_current, mixed_release, route=selected_route)
        assert np.allclose(mixed["clearance_z"], [.62, .41], rtol=0, atol=1e-12)
        assert mixed["clearance_z"][0] == mixed_current[0, 2] + .12
        assert mixed["clearance_z"][1] == mixed_release[1, 2] + .06
        assert np.array_equal(mixed["release"], initial_release)
        assert np.array_equal(mixed["above_baskets"][:, :2], initial_release[:, :2])
        assert np.array_equal(mixed_current, initial_current) and np.array_equal(mixed_release, initial_release)
        assert np.all(mixed["lift"][:, 2] - mixed_current[:, 2] >= .12 - 1e-12)
    assert config["transport_clearance_above_release_m"] == .06
    assert config["minimum_commanded_lift_m"] == .12
    assert config["route_geometry_policy"] == "compact-clearance-v2"
    reject(lambda: compact_route_positions(current[:1], release, route="merge-lift"))
    reject(lambda: compact_route_positions(current, [[0, 0, np.inf]] * 2, route="merge-lift"))
    native = {"status": "Success", "time": [0., .1, .2], "position": np.zeros((3, 7)),
              "velocity": np.full((3, 7), .9), "acceleration": np.full((3, 7), -.9)}
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "passed"
    native["velocity"][1, 0] = .91
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "failed"
    native["velocity"][1, 0] = .9; native["acceleration"][2, 6] = -.91
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "failed"
    native["acceleration"][2, 6] = .9000788313717154
    original_overshoot = compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)
    assert original_overshoot["status"] == "failed" and original_overshoot["numeric_tolerance"] == 1e-6
    native["acceleration"][2, 6] = .8910788313717154
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "passed"
    native["acceleration"] = np.zeros((2, 7))
    reject(lambda: compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9))
    names = [f"joint{i}" for i in range(7)]
    with tempfile.TemporaryDirectory(prefix="tig-compact-urdf-") as td:
        path = Path(td) / "robot.urdf"
        path.write_text("<robot>" + "".join(f'<joint name="{n}"><limit velocity="2.0"/></joint>' for n in names) + "</robot>")
        audit = compact_urdf_velocity_limits(path, names, 1.5)
        assert audit["velocity_limits_rad_s"] == [2.] * 7
        reject(lambda: compact_urdf_velocity_limits(path, names, 2.01))
        reject(lambda: compact_urdf_velocity_limits(path, names[:-1] + ["unknown"], .9))
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
        "array_shapes": {"paired_world_positions": [2, 3], "native_derivatives": [3, 7]},
        "checks": ["original frozen acceptance helper regression", "official200 and explicit extended diagnostic budget",
            "execution vs retained20 hover profile", "vertical clearance and >=.12 commanded lift geometry",
            "separate-vs-merged lift endpoints", "mixed lift-floor/clearance branch preserves release endpoints and inputs",
            "native derivative limits and malformed array rejection",
            "seven URDF velocity limits and over-limit rejection",
            "explicit99pct solver envelope; nominal .9 gate rejects observed original overshoot",
            "approach overrides apply only to pregrasp/descend; runtime MPLib arrays update for held/retreat"], "original_checks": original["checks"],
        "retiming_suite": retiming_self_test()}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("positive", "retain-right"), default="positive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--route", choices=ROUTES, default="merge-lift")
    parser.add_argument("--execution-profile", choices=PROFILES, default="execution")
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="200 is official; 201..400 are explicitly extended timing diagnostics")
    parser.add_argument("--joint-vel-limit", type=float, default=.9)
    parser.add_argument("--joint-acc-limit", type=float, default=.9)
    parser.add_argument("--approach-vel-limit", type=float,
                        help="Nominal limit only for pregrasp/descend; defaults to global joint velocity limit")
    parser.add_argument("--approach-acc-limit", type=float,
                        help="Nominal limit only for pregrasp/descend; defaults to global joint acceleration limit")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    try: configuration = compact_configuration(args)
    except ValueError as exc: parser.error(str(exc))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu): parser.error("Select one available GPU UUID for rendering")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(f"sync-compact-{args.route}-{args.execution_profile}-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    names = ["trajectory_schedule.py", "trajectory_execution.py", "combined_collision.py", "scene_collision.py", "grasp_collision.py",
             "grasp_execution.py", "validate_grasp.py", "held_collision.py", "lift_execution.py", "validate_lift.py", "place_execution.py",
             "validate_place.py", "single_arm_execution.py", "table_support_collision.py", "validate_banana.py",
             "dual_payload_collision.py", "dual_arm_execution.py", "validate_sync.py", "validate_sync_compact.py"]
    sources = {n: (HERE / n).read_bytes() for n in names}
    snapshot = output / "source"; snapshot.mkdir()
    for name, contents in sources.items(): (snapshot / name).write_bytes(contents)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "task": TASK, "seed": args.seed,
        "mode": args.mode, "gpu_uuid": gpu, "scope": "synchronized left apple and right banana on one physical control timeline",
        "synchronization": "phase barriers; simultaneous start; each arm holds final joint and gripper targets until both ready",
        "max_episode_steps": args.max_episode_steps, "official_default_max_episode_steps": 200,
        "execution_configuration": configuration,
        "source_reference": str(ROOT / "infrastructure/validation/dual-arm/sync-positive-20261115T134421956872Z/report.json"),
        "official_time_limit_policy": "stop at native truncation including tick200; full physical closure should finish by199",
        "termination_policy": "retain native success termination in H5/ticks; continue success-only signals to verify actual release and settling; failure and truncation stop",
        "physical_attachment_created": False, "continuous_collision_detection": False,
        "source_snapshot": str(snapshot), "source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}}
    save_report(output / "report.json", report)
    started = time.monotonic()
    try: native_run(args, output, report)
    except Exception as exc: report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
