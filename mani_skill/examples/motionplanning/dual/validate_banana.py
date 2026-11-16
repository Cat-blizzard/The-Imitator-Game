"""Record a right-banana pick, carry, release and retreat; left arm holds.

The left arm holds throughout. This development harness extends the episode
budget to 400 steps; it does not change the official task's 200-step default.
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
import time
import traceback
from typing import Any

import numpy as np

from .validate_grasp import TASK, json_value, recording_summary, save_report
from .validate_lift import pose_matrix

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]


# Compatibility exports: implementations live in shared runtime modules.
from .task_targets import (banana_grasp_poses)



def applied_action(action: dict[str, np.ndarray], *, mode: str, phase: str,
                   active_key: str) -> tuple[dict[str, np.ndarray], bool]:
    """The retain-grip control overrides only release-phase OPEN requests."""
    if mode not in ("positive", "retain-grip"):
        raise ValueError("Unknown validation mode")
    actual = {key: np.asarray(value, dtype=np.float32).copy() for key, value in action.items()}
    if active_key not in actual or actual[active_key].shape != (8,):
        raise ValueError("Expected an eight-value active-arm action")
    inject = bool(mode == "retain-grip" and phase == "release" and actual[active_key][-1] == 1)
    if inject:
        actual[active_key][-1] = -1
    return actual, inject


def retain_grip_valid(result: dict[str, Any], ticks: list[dict[str, Any]], keys: list[str]) -> bool:
    if (result.get("status") != "release_timeout" or
            any(result.get(key) for key in ("error", "error_type", "traceback")) or
            any(row.get("error") for row in result.get("trace", []))):
        return False
    injected = [row for row in ticks if row.get("fault_injected")]
    if not injected or any(row["phase"] in ("retreat", "verify_placement") for row in ticks):
        return False
    for row in injected:
        request, actual = row["requested_actions"], row["actions"]
        if (row["phase"] != "release" or request[keys[1]][-1] != 1 or actual[keys[1]][-1] != -1 or
                not np.allclose(request[keys[1]][:7], actual[keys[1]][:7], atol=2e-7, rtol=0) or
                any(not np.allclose(request[k], actual[k], atol=2e-7, rtol=0) for k in keys[:1])):
            return False
    return bool(injected[-1]["is_grasping"] and max(injected[-1]["right_qpos"][-2:]) < .038)


def held_pose_envelope(rows: list[dict[str, Any]], reference: np.ndarray) -> dict[str, Any]:
    """Audit sampled attachment slip only while physically grasped, before release."""
    from .lift_execution import _rotation_distance
    held = [row for row in rows if row["phase"] == "lift_and_hold" or row["phase"].startswith("carry_")]
    if not held:
        raise ValueError("Held physics-substep observations are required")
    position, rotation = [], []
    for row in held:
        relative = np.linalg.inv(pose_matrix(row["right_tcp_pose"])) @ pose_matrix(row["banana_pose"])
        position.append(float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])))
        rotation.append(_rotation_distance(reference[:3, :3], relative[:3, :3]))
    return {"status": "passed" if max(position) <= .005 and max(rotation) <= .15 else "failed",
            "physical_substeps": len(held), "max_position_slip_m": max(position),
            "max_rotation_slip_rad": max(rotation), "position_limit_m": .005, "rotation_limit_rad": .15,
            "scope": "sampled held-object relative pose; planned descent is allowed; not CCD"}


def joint_hold_substep_audit(rows: list[dict[str, Any]], control_steps: list[int]) -> dict[str, Any]:
    """Check real 100 Hz joint positions across the full lifted hold window."""
    wanted = sorted(set(control_steps))
    if not wanted or wanted != list(range(wanted[0], wanted[-1] + 1)):
        raise ValueError("A contiguous physical hold window is required")
    selected = [i for i, row in enumerate(rows) if row["control_step"] in wanted]
    if not selected or selected[0] == 0:
        raise ValueError("Hold substeps and their preceding physical sample are required")
    window = rows[selected[0] - 1:selected[-1] + 1]
    dt = np.asarray([row["sim_timestep_s"] for row in window], dtype=float)
    q = np.asarray([row["right_qpos"][:7] for row in window], dtype=float)
    if (q.shape != (len(window), 7) or not np.isfinite(q).all()
            or not np.isfinite(dt).all() or np.any(dt <= 0) or not np.allclose(dt, .01, rtol=0, atol=1e-10)
            or any(sum(row["control_step"] == step for row in window[1:]) != 5 for step in wanted)):
        raise ValueError("Expected five finite 100 Hz observations per hold control tick")
    speed = float(np.max(np.abs(np.diff(q, axis=0) / dt[1:, None])))
    drift = float(np.max(np.abs(q - q[:1])))
    return {"status": "passed" if speed <= .05 and drift <= .003 else "failed",
            "physical_hold_substeps": len(selected), "control_steps": wanted,
            "max_physics_pose_difference_arm_speed_rad_s": speed, "max_hold_joint_drift_rad": drift,
            "speed_limit_rad_s": .05, "joint_drift_limit_rad": .003,
            "scope": "sampled physical joint positions; raw instantaneous velocities remain recorded separately"}


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import sapien
    import torch
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy
    from .single_arm_execution import plan_single_pose, execute_reach, close_and_confirm, execute_held_move, release_and_confirm, verify_placement
    from .grasp_collision import LiveGraspCollisionChecker
    from .table_support_collision import NumericalTableSupportCollisionChecker
    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh

    env, raw = None, None
    frames, ticks, substeps, keys = [], [], [], []
    original_after_simulation_step = None
    report["fault_events"] = []
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos",
                        sim_backend="cpu", render_backend="gpu", render_mode="rgb_array", max_episode_steps=400)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
                            source_type="motionplanning", source_desc=f"right banana placement; mode={args.mode}")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        keys = list(raw.agent.agents_dict)
        if (len(keys) != 2 or raw.agent.agents_dict[keys[0]] is not raw.left_agent
                or raw.agent.agents_dict[keys[1]] is not raw.right_agent):
            raise ValueError("Expected the physical left/right action-key order")
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), stages=[])
        initial_left = as_numpy(raw.left_agent.robot.get_qpos())[0, :7].copy()

        class Capture:
            unwrapped = raw
            phase = "preparation"
            observe_substeps = False
            pending_control_step = 0

            def frame(self) -> None:
                frame = as_numpy(env.render())
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError("Expected HWC uint8 RGB rendering")
                frames.append(frame.copy())

            def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
                if set(action) != set(keys):
                    raise ValueError("Every control tick must submit both arms")
                actual, inject = applied_action(action, mode=args.mode, phase=self.phase, active_key=keys[1])
                self.pending_control_step = int(as_numpy(raw.elapsed_steps)[0]) + 1
                if inject:
                    report["fault_events"].append({"next_environment_step": self.pending_control_step,
                                                   "type": "override_release_with_closed_gripper"})
                result = env.step(actual)
                active = raw.right_agent
                forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, raw.banana_out))[0]
                          for link in (active.finger1_link, active.finger2_link)]
                ticks.append({"phase": self.phase, "environment_step": int(as_numpy(raw.elapsed_steps)[0]),
                              "requested_actions": json_value(action), "actions": json_value(actual),
                              "fault_injected": inject,
                              "left_qpos": as_numpy(raw.left_agent.robot.get_qpos())[0].tolist(),
                              "right_qpos": as_numpy(raw.right_agent.robot.get_qpos())[0].tolist(),
                              "right_tcp_pose": as_numpy(active.tcp.pose.raw_pose)[0].tolist(),
                              "left_tcp_pose": as_numpy(raw.left_agent.tcp.pose.raw_pose)[0].tolist(),
                              "banana_pose": as_numpy(raw.banana_out.pose.raw_pose)[0].tolist(),
                              "banana_linear_velocity_m_s": as_numpy(raw.banana_out.linear_velocity)[0].tolist(),
                              "banana_angular_velocity_rad_s": as_numpy(raw.banana_out.angular_velocity)[0].tolist(),
                              "is_grasping": bool(as_numpy(active.is_grasping(raw.banana_out))[0]),
                              "finger_contact_force_norms_N": [float(np.linalg.norm(f)) for f in forces],
                              "left_initial_error_rad": float(np.max(np.abs(
                                  as_numpy(raw.left_agent.robot.get_qpos())[0, :7] - initial_left)))})
                self.frame()
                return result

        captured = Capture()
        original_after_simulation_step = raw._after_simulation_step

        def record_substep() -> None:
            original_after_simulation_step()
            if captured.observe_substeps:
                substeps.append({"substep": len(substeps) + 1, "phase": captured.phase,
                                 "control_step": captured.pending_control_step, "sim_timestep_s": float(raw.sim_timestep),
                                 "banana_pose": as_numpy(raw.banana_out.pose.raw_pose)[0].tolist(),
                                 "right_tcp_pose": as_numpy(raw.right_agent.tcp.pose.raw_pose)[0].tolist(),
                                 "right_qpos": as_numpy(raw.right_agent.robot.get_qpos())[0].tolist(),
                                 "right_qvel": as_numpy(raw.right_agent.robot.get_qvel())[0].tolist(),
                                 "left_qpos": as_numpy(raw.left_agent.robot.get_qpos())[0].tolist(),
                                 "left_qvel": as_numpy(raw.left_agent.robot.get_qvel())[0].tolist(),
                                 "banana_linear_velocity_m_s": as_numpy(raw.banana_out.linear_velocity)[0].tolist(),
                                 "banana_angular_velocity_rad_s": as_numpy(raw.banana_out.angular_velocity)[0].tolist()})

        raw._after_simulation_step = record_substep
        captured.frame()
        hold = {key: np.r_[as_numpy(agent.robot.get_qpos())[0, :7], 1].astype(np.float32)
                for key, agent in raw.agent.agents_dict.items()}
        for _ in range(10):
            captured.step(hold)
        report["preparation_steps"] = 10
        report["initial_geometry"] = {name: {
            "pose_wxyz": as_numpy(getattr(raw, name).pose.raw_pose)[0].tolist(),
            "world_aabb_m": as_numpy(world_aabb_from_local_mesh(getattr(raw, name), raw.device))[0].tolist()}
            for name in ("banana_out", "banana_in", "basket_banana")}
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=agent.robot.pose, multi_robot_id=i,
                                               debug=False, vis=False, visualize_target_grasp_pose=False,
                                               print_env_info=False)
                   for i, agent in enumerate(raw.agent.agents)]
        live = LiveGraspCollisionChecker.from_solvers(raw, *solvers, target_actor=raw.banana_out, active_robot="right")
        report["scene_coverage"] = live.coverage

        def begin(name: str) -> None:
            report["stage"] = captured.phase = name
            save_report(output / "report.json", report)

        def finish(name: str, result: dict[str, Any]) -> bool:
            report["stages"].append({"name": name, **result})
            if result["status"] != "passed":
                report.update(status="failed", reason=f"{name}:{result['status']}")
            save_report(output / "report.json", report)
            return result["status"] == "passed"

        def plan(name: str, pose: Any, *, closed: bool) -> Any:
            before = int(as_numpy(raw.elapsed_steps)[0])
            schedule = plan_single_pose(*solvers, pose, active_key=keys[1], gripper_command=-1 if closed else 1)
            if int(as_numpy(raw.elapsed_steps)[0]) != before:
                raise RuntimeError("Planning advanced the environment")
            np.savez(output / f"{name}-plan.npz", left_time=schedule.trajectories[0].times_s,
                     left_position=schedule.trajectories[0].positions_rad,
                     right_time=schedule.trajectories[1].times_s,
                     right_position=schedule.trajectories[1].positions_rad, target_times=schedule.target_times_s,
                     left_commands=schedule.left_commands, right_commands=schedule.right_commands)
            report.setdefault("planning", []).append({"phase": name, "environment_steps": 0,
                                                      "scheduled_steps": schedule.num_steps,
                                                      "target_tcp_pose": np.r_[pose.p, pose.q].tolist()})
            return schedule

        for name, pose, contact in zip(("pregrasp", "descend"), banana_grasp_poses(raw), (False, True)):
            begin(name)
            schedule = plan(name, pose, closed=False)
            if not finish(name, execute_reach(captured, schedule, live, active_key=keys[1], allow_target_contact=contact)):
                return
        begin("close_and_confirm")
        if not finish("close_and_confirm", close_and_confirm(captured, live, active_key=keys[1], target_actor=raw.banana_out)):
            return
        begin("attach_collision_prediction")
        attached = NumericalTableSupportCollisionChecker.from_live(live, raw.banana_out, active_robot="right")
        report["attachment_initial_coverage"] = json_value(attached.coverage)
        report["attachment_T_tcp_object"] = attached.attachment_T_tcp_object.tolist()
        current = raw.right_agent.tcp.pose
        lift_target = sapien.Pose(as_numpy(current.p)[0] + [0, 0, .12], as_numpy(current.q)[0])
        begin("lift_and_hold")
        schedule = plan("lift_and_hold", lift_target, closed=True)
        captured.observe_substeps = True
        lifted = execute_held_move(captured, schedule, attached, raw.banana_out,
                                  active_key=keys[1], min_height_gain_m=.10, hold_steps=20)
        if not finish("lift_and_hold", lifted):
            return
        hold_control_steps = [int(lifted["initial_elapsed_steps"][0]) + row["step"]
                              for row in lifted["trace"] if row["phase"] == "hover"]
        report["physics_hold_joint_audit"] = joint_hold_substep_audit(substeps, hold_control_steps)
        if report["physics_hold_joint_audit"]["status"] != "passed":
            report.update(status="failed", reason="physics_hold_joint_stability")
            return
        basket_xyz = as_numpy(raw.basket_banana.pose.p)[0]
        official_release_xyz = basket_xyz + [.03, .03, .12]
        release_xyz = basket_xyz + [args.release_offset_x, args.release_offset_y, .12]
        grasp_q = as_numpy(raw.right_agent.tcp.pose.q)[0].copy()
        yaw = np.deg2rad(args.release_yaw_deg)
        carry_q = (sapien.Pose(q=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]) * sapien.Pose(q=grasp_q)).q
        clearance_z = float(release_xyz[2] + .08)
        current_xyz = as_numpy(raw.right_agent.tcp.pose.p)[0]
        above_current = [current_xyz[0], current_xyz[1], clearance_z]
        waypoints = [("carry_raise", above_current, grasp_q)]
        if abs(args.release_yaw_deg) > 1e-9:
            waypoints.append(("carry_orient", above_current, carry_q))
        waypoints.extend([("carry_above_basket", [release_xyz[0], release_xyz[1], clearance_z], carry_q),
                          ("carry_lower", release_xyz, carry_q)])
        report["official_release_tcp_xyz_m"] = official_release_xyz.tolist()
        report["selected_release_tcp_xyz_m"] = release_xyz.tolist()
        report["release_world_yaw_change_deg"] = args.release_yaw_deg
        for name, xyz, orientation in waypoints:
            begin(name)
            schedule = plan(name, sapien.Pose(xyz, orientation), closed=True)
            if not finish(name, execute_held_move(captured, schedule, attached, raw.banana_out, active_key=keys[1])):
                return
        report["attachment_final_coverage"] = json_value(attached.coverage)
        report["before_release_geometry"] = {name: {
            "pose_wxyz": as_numpy(getattr(raw, name).pose.raw_pose)[0].tolist(),
            "world_aabb_m": as_numpy(world_aabb_from_local_mesh(getattr(raw, name), raw.device))[0].tolist()}
            for name in ("banana_out", "banana_in", "basket_banana")}
        report["held_physics_substep_audit"] = held_pose_envelope(substeps, attached.attachment_T_tcp_object)
        if report["held_physics_substep_audit"]["status"] != "passed":
            report.update(status="failed", reason="held_physics_substep_envelope")
            return
        begin("release")
        result = release_and_confirm(captured, live, raw.banana_out, active_key=keys[1])
        finish("release", result)
        if args.mode == "retain-grip":
            report["retain_grip_control_verified"] = retain_grip_valid(result, ticks, keys)
            report["expected_outcome"] = "release_timeout with actual gripper closed; no retreat or verification"
            report["status"] = "passed" if report["retain_grip_control_verified"] else "failed"
            return
        if result["status"] != "passed":
            return
        solvers[0].gripper_state = solvers[1].gripper_state = 1
        current = raw.right_agent.tcp.pose
        retreat_pose = sapien.Pose(as_numpy(current.p)[0] + [0, 0, .08], as_numpy(current.q)[0])
        begin("retreat")
        schedule = plan("retreat", retreat_pose, closed=False)
        if not finish("retreat", execute_reach(captured, schedule, live, active_key=keys[1], allow_target_contact=False)):
            return

        def region_predicate(actor: Any) -> dict[str, Any]:
            center = as_numpy(raw._get_actor_geom_center(actor))[0]
            bounds = as_numpy(world_aabb_from_local_mesh(raw.basket_banana, raw.device))[0]
            object_bounds = as_numpy(world_aabb_from_local_mesh(actor, raw.device))[0]
            lo, hi = bounds
            xy_contained = bool(np.all(object_bounds[0, :2] >= lo[:2] - .002)
                                and np.all(object_bounds[1, :2] <= hi[:2] + .002))
            inside = bool(xy_contained and np.all(center[:2] >= lo[:2]) and np.all(center[:2] <= hi[:2])
                          and lo[2] - .005 <= center[2] <= hi[2] + .01
                          and object_bounds[0, 2] >= lo[2] - .005)
            return {"inside": inside, "object_geom_center_m": center.tolist(), "basket_aabb_m": bounds.tolist(),
                    "object_aabb_m": object_bounds.tolist(), "object_xy_contained": xy_contained,
                    "object_fully_below_rim": bool(object_bounds[1, 2] <= hi[2]),
                    "center_z_range_m": [float(lo[2] - .005), float(hi[2] + .01)]}

        begin("verify_placement")
        result = verify_placement(captured, live, raw.banana_out, active_key=keys[1],
                                  official_predicate_key="banana_in_basket", stable_steps=20,
                                  region_predicate=region_predicate)
        if finish("verify_placement", result):
            report["status"] = "passed" if not report["fault_events"] else "failed"
        report["official_task_evaluation"] = json_value(raw.evaluate())
        report["stage"] = "finished"
    finally:
        errors = report.setdefault("artifact_errors", [])
        if raw is not None:
            for name, read in (("total_environment_steps", lambda: int(as_numpy(raw.elapsed_steps)[0])),
                               ("torch_cuda_initialized", torch.cuda.is_initialized),
                               ("official_task_evaluation", lambda: json_value(raw.evaluate()))):
                try:
                    report[name] = read()
                except Exception as exc:
                    errors.append(f"final {name}: {type(exc).__name__}: {exc}")
            try:
                if original_after_simulation_step is not None:
                    raw._after_simulation_step = original_after_simulation_step
            except Exception as exc:
                errors.append(f"restore physics observation hook: {type(exc).__name__}: {exc}")
        if ticks:
            report["left_max_error_from_initial_rad"] = max(t["left_initial_error_rad"] for t in ticks)
            if report["left_max_error_from_initial_rad"] > .03:
                report.update(status="failed", reason="left_global_hold_tolerance")
        for filename, rows in (("ticks.jsonl", ticks), ("physics-substeps.jsonl", substeps)):
            try:
                with (output / filename).open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            except Exception as exc:
                errors.append(f"{filename}: {type(exc).__name__}: {exc}")
        if env is not None:
            try:
                if getattr(env, "_trajectory_buffer", None) is not None:
                    env.flush_trajectory()
            except Exception as exc:
                errors.append(f"record flush: {type(exc).__name__}: {exc}")
            finally:
                try:
                    env.close()
                except Exception as exc:
                    errors.append(f"environment close: {type(exc).__name__}: {exc}")
            try:
                expected = np.asarray([np.concatenate([tick["actions"][key] for key in keys]) for tick in ticks]) if ticks else None
                report["recording"] = recording_summary(output / "trajectory.h5", len(ticks), expected)
                report["recording"]["action_key_order"] = keys
            except Exception as exc:
                errors.append(f"record verification: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "execution.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20,
                               fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
                report["video"] = {"path": str(output / "execution.mp4"), "fps": 20,
                                   "padding_frames_before": 10, "padding_frames_after": 20}
            except Exception as exc:
                errors.append(f"video: {type(exc).__name__}: {exc}")
        if errors:
            report["status"] = "failed"


def self_test() -> dict[str, Any]:
    requests = {"left": np.r_[np.zeros(7), 1.], "right": np.r_[np.ones(7), 1.]}
    for mode, phase, expected in (("positive", "release", False), ("retain-grip", "carry_lower", False),
                                   ("retain-grip", "release", True), ("retain-grip", "retreat", False)):
        actual, inject = applied_action(requests, mode=mode, phase=phase, active_key="right")
        assert isinstance(inject, bool) and inject == expected and requests["right"][-1] == 1
        json.dumps({"fault_injected": inject}, allow_nan=False)
        assert actual["right"][-1] == (-1 if expected else 1)
        assert np.array_equal(actual["left"], requests["left"])
    actual, _ = applied_action(requests, mode="retain-grip", phase="release", active_key="right")
    tick = {"phase": "release", "fault_injected": True, "requested_actions": requests,
            "actions": actual, "is_grasping": True, "right_qpos": [0.] * 7 + [.024, .024]}
    result = {"status": "release_timeout", "trace": []}
    assert retain_grip_valid(result, [tick], ["left", "right"])
    assert not retain_grip_valid({**result, "error": "checker failed"}, [tick], ["left", "right"])
    assert not retain_grip_valid({**result, "status": "aborted"}, [tick], ["left", "right"])
    assert not retain_grip_valid(result, [tick, {"phase": "retreat"}], ["left", "right"])
    pose = [0, 0, .1, 1, 0, 0, 0]
    row = {"phase": "carry_lower", "right_tcp_pose": pose, "banana_pose": pose}
    assert held_pose_envelope([row], np.eye(4))["status"] == "passed"
    assert held_pose_envelope([{**row, "banana_pose": [.01, 0, .1, 1, 0, 0, 0]}], np.eye(4))["status"] == "failed"
    joint_rows = [{"control_step": step, "sim_timestep_s": .01, "right_qpos": [0.] * 9}
                  for step in range(3) for _ in range(5)]
    assert joint_hold_substep_audit(joint_rows, [1, 2])["status"] == "passed"
    bad = [{**row, "right_qpos": list(row["right_qpos"])} for row in joint_rows]
    bad[-1]["right_qpos"][0] = .01
    assert joint_hold_substep_audit(bad, [1, 2])["status"] == "failed"
    slow = [{**row, "right_qpos": [i * .0004] + [0.] * 8} for i, row in enumerate(joint_rows)]
    assert joint_hold_substep_audit(slow, [1, 2])["status"] == "failed"
    assert (ROOT / "infrastructure").is_dir()
    return {"status": "passed", "checks": ["release-only explicit fault", "immutable requests and passive left action",
                                             "control rejects unrelated errors and continuation", "held pose envelope",
                                             "100Hz joint hold envelope catches motion and slow drift"]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("positive", "retain-grip"), default="positive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--release-yaw-deg", type=float, default=90.)
    parser.add_argument("--release-offset-x", type=float, default=.03)
    parser.add_argument("--release-offset-y", type=float, default=0.)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not np.isfinite([args.release_yaw_deg, args.release_offset_x, args.release_offset_y]).all():
        parser.error("Release waypoint parameters must be finite")
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
        parser.error("Select one available GPU UUID for rendering")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(
        f"banana-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    names = ["trajectory_schedule.py", "trajectory_execution.py", "combined_collision.py", "scene_collision.py",
             "grasp_collision.py", "grasp_execution.py", "validate_grasp.py", "held_collision.py",
             "lift_execution.py", "validate_lift.py", "place_execution.py", "validate_place.py",
             "single_arm_execution.py", "table_support_collision.py", "validate_banana.py"]
    sources = {name: (HERE / name).read_bytes() for name in names}
    snapshot = output / "source"
    snapshot.mkdir()
    for name, contents in sources.items():
        (snapshot / name).write_bytes(contents)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "task": TASK, "seed": args.seed, "mode": args.mode, "gpu_uuid": gpu,
              "scope": "right banana pick, carry, release, 8cm retreat, 20-tick settled placement; left holds",
              "active_robot": "right", "target_actor": "banana_out", "passive_robot": "left",
              "waypoint_configuration": {"world_yaw_change_deg": args.release_yaw_deg,
                  "release_offset_m": [args.release_offset_x, args.release_offset_y, .12]},
              "candidate_rationale": "Rotate the long banana toward basket Y; retain basket fruit. Official unrotated release projection exceeds the short X edge in CPU mesh analysis; actual planning/collision and placement remain required.",
              "max_episode_steps": 400, "official_default_max_episode_steps": 200,
              "physical_attachment_created": False, "continuous_collision_detection": False,
              "source_snapshot": str(snapshot),
              "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}}
    save_report(output / "report.json", report)
    started = time.monotonic()
    try:
        native_run(args, output, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
