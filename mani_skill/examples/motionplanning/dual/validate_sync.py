"""Validate paired physical apple/banana manipulation on one control timeline.

Development-only 400-step budget; the task's official 200-step default remains.
All environment actions submit physical left/right together. Success-only native
termination is retained in H5/ticks and continued for physical release/settling;
failure and truncation still stop. No physical pose/state setter or weld is used.
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

from .validate_grasp import TASK, grasp_poses, json_value, recording_summary, save_report
from .validate_banana import banana_grasp_poses
from .validate_lift import pose_matrix

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
ACTOR_NAMES = ("apple_out", "banana_out")


# Compatibility exports: implementations live in shared runtime modules.
from .physical_observation import (applied_action, continue_success_only, held_substep_audit)
from .task_targets import (physical_region, collision_vertex_region)





def retain_right_valid(result: dict[str, Any], ticks: list[dict[str, Any]], keys: list[str]) -> bool:
    if (result.get("status") != "release_timeout" or any(result.get(k) for k in ("error", "error_type", "traceback"))
            or any(row.get("error") for row in result.get("trace", []))):
        return False
    injected = [r for r in ticks if r["fault_injected"]]
    if len(injected) != 30 or any(r["phase"] in ("retreat", "verify_placement") for r in ticks):
        return False
    for row in injected:
        expected = np.concatenate([row["requested_actions"][k] for k in keys]).astype(np.float32)
        actual = np.concatenate([row["actions"][k] for k in keys]).astype(np.float32)
        if (row["phase"] != "release" or expected[15] != 1 or actual[15] != -1
                or not np.array_equal(np.flatnonzero(actual != expected), [15])
                or not row["objects"][1]["is_grasping"]):
            return False
    last = injected[-1]
    return bool(max(last["right_qpos"][-2:]) < .038 and min(last["left_qpos"][-2:]) >= .038
                and not last["objects"][0]["is_grasping"])








def hold_joint_substep_audit(rows: list[dict[str, Any]], control_steps: list[int]) -> dict[str, Any]:
    wanted = sorted(set(control_steps))
    if not wanted or wanted != list(range(wanted[0], wanted[-1] + 1)):
        raise ValueError("Need a contiguous hover window")
    selected = [i for i, row in enumerate(rows) if row["control_step"] in wanted]
    if not selected or selected[0] == 0:
        raise ValueError("Need preceding physical anchor and hover samples")
    window = rows[selected[0] - 1:selected[-1] + 1]
    dt = np.asarray([r["sim_timestep_s"] for r in window])
    if not np.allclose(dt, .01, rtol=0, atol=1e-10) or any(
            sum(r["control_step"] == step for r in window[1:]) != 5 for step in wanted):
        raise ValueError("Need five100Hz observations per control tick")
    values = []
    for key in ("left_qpos", "right_qpos"):
        q = np.asarray([r[key][:7] for r in window], dtype=float)
        if q.shape != (len(window), 7) or not np.isfinite(q).all():
            raise ValueError("Need finite arm joint observations")
        values.append({"robot": key.split("_")[0], "max_position_difference_speed_rad_s": float(np.max(np.abs(np.diff(q, axis=0) / dt[1:, None]))),
                       "max_hold_joint_drift_rad": float(np.max(np.abs(q - q[:1])))})
    return {"status": "passed" if all(v["max_position_difference_speed_rad_s"] <= .05 and v["max_hold_joint_drift_rad"] <= .003 for v in values) else "failed",
            "physical_hold_substeps": len(selected), "control_steps": wanted, "robots": values,
            "speed_limit_rad_s": .05, "joint_drift_limit_rad": .003}


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

    env, raw, original_hook = None, None, None
    frames, ticks, substeps, keys = [], [], [], []
    report["fault_events"] = []
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos", sim_backend="cpu",
                        render_backend="gpu", render_mode="rgb_array", max_episode_steps=400)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
                            source_type="motionplanning", source_desc=f"synchronized apple and banana; mode={args.mode}")
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

        raw._after_simulation_step = record_substep
        captured.frame()
        initial_hold = {k: np.r_[as_numpy(a.robot.get_qpos())[0, :7], 1].astype(np.float32) for k, a in zip(keys, agents)}
        for _ in range(10):
            _, _, done, truncated, _ = captured.step(initial_hold)
            if done or bool(as_numpy(truncated).reshape(-1)[0]): raise RuntimeError("Environment ended during preparation")
        report["preparation_steps"] = 10
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=a.robot.pose, multi_robot_id=i, debug=False,
                                               vis=False, visualize_target_grasp_pose=False, print_env_info=False)
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
            before = int(as_numpy(raw.elapsed_steps)[0])
            schedule = plan_dual_pose(*solvers, poses, gripper_commands=(-1, -1) if closed else (1, 1))
            if int(as_numpy(raw.elapsed_steps)[0]) != before: raise RuntimeError("Planning advanced environment")
            np.savez(output / f"{name}-plan.npz", left_time=schedule.trajectories[0].times_s,
                     right_time=schedule.trajectories[1].times_s, left_position=schedule.trajectories[0].positions_rad,
                     right_position=schedule.trajectories[1].positions_rad, target_times=schedule.target_times_s,
                     left_commands=schedule.left_commands, right_commands=schedule.right_commands)
            report.setdefault("planning", []).append({"phase": name, "environment_steps": 0, "scheduled_steps": schedule.num_steps,
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
        lift_poses = tuple(sapien.Pose(as_numpy(a.tcp.pose.p)[0] + [0, 0, .12], as_numpy(a.tcp.pose.q)[0]) for a in agents)
        begin("lift_and_hold")
        schedule = plan("lift_and_hold", lift_poses, closed=True)
        captured.observe_substeps = True
        lifted = execute_dual_held_move(captured, schedule, checker, targets, min_height_gain_m=(.10, .10), hold_steps=20)
        if not finish("lift_and_hold", lifted): return
        hover_steps = [int(lifted["initial_elapsed_steps"][0]) + row["step"] for row in lifted["trace"] if row["phase"] == "hover"]
        report["physics_hold_joint_audit"] = hold_joint_substep_audit(substeps, hover_steps)
        if report["physics_hold_joint_audit"]["status"] != "passed":
            report.update(status="failed", reason="physics_hold_joint_stability"); return
        baskets = (raw.basket_apple, raw.basket_banana)
        release_xyz = [as_numpy(b.pose.p)[0] + offset for b, offset in zip(baskets, ([.03, .03, .12], [.03, 0, .12]))]
        original_q = [as_numpy(a.tcp.pose.q)[0].copy() for a in agents]
        carry_q = [original_q[0], (sapien.Pose(q=[2**-.5, 0, 0, 2**-.5]) * sapien.Pose(q=original_q[1])).q]
        clearance_z = [float(xyz[2] + .08) for xyz in release_xyz]
        high_current = [np.r_[as_numpy(a.tcp.pose.p)[0, :2], z] for a, z in zip(agents, clearance_z)]
        report["selected_release_tcp_xyz_m"] = [p.tolist() for p in release_xyz]
        waypoints = [("carry_raise", tuple(sapien.Pose(x, q) for x, q in zip(high_current, original_q))),
                     ("carry_orient", (None, sapien.Pose(high_current[1], carry_q[1]))),
                     ("carry_above_baskets", tuple(sapien.Pose([xyz[0], xyz[1], z], q) for xyz, z, q in zip(release_xyz, clearance_z, carry_q))),
                     ("carry_lower", tuple(sapien.Pose(xyz, q) for xyz, q in zip(release_xyz, carry_q)))]
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
        retreat_poses = tuple(sapien.Pose(as_numpy(a.tcp.pose.p)[0] + [0, 0, .08], as_numpy(a.tcp.pose.q)[0]) for a in agents)
        begin("retreat")
        schedule = plan("retreat", retreat_poses, closed=False)
        if not finish("retreat", execute_dual_reach(captured, schedule, checker)): return

        local_meshes = {}
        for actor in (*targets, *baskets):
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            if mesh is None:
                raise ValueError(f"No complete native collision mesh for {actor.name}")
            local_meshes[id(actor)] = np.asarray(mesh.vertices, dtype=np.float64).copy()

        def predicate(index: int) -> Any:
            def measure(actor: Any) -> dict[str, Any]:
                return collision_vertex_region(local_meshes[id(actor)], as_numpy(actor.pose.raw_pose)[0],
                    local_meshes[id(baskets[index])], as_numpy(baskets[index].pose.raw_pose)[0])
            return measure
        begin("verify_placement")
        result = verify_dual_placement(captured, checker, targets, region_predicates=(predicate(0), predicate(1)))
        if finish("verify_placement", result):
            evaluation = json_value(raw.evaluate())
            report["status"] = "passed" if bool(np.asarray(evaluation["success"]).all()) and not report["fault_events"] else "failed"
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
        if errors: report["status"] = "failed"


def self_test() -> dict[str, Any]:
    assert continue_success_only(True, False, {"success": True}) == (False, True)
    assert continue_success_only(True, False, {"success": True, "fail": True}) == (True, False)
    assert continue_success_only(True, True, {"success": True}) == (True, False)
    assert continue_success_only(True, False, {}) == (True, False)
    request = {"left": np.r_[np.zeros(7), 1], "right": np.r_[np.ones(7), 1]}
    actual, inject = applied_action(request, mode="retain-right", phase="release", keys=["left", "right"])
    assert inject and actual["right"][-1] == -1 and request["right"][-1] == 1
    tick = {"phase": "release", "fault_injected": True, "requested_actions": request, "actions": actual,
            "left_qpos": [0.] * 7 + [.04, .04], "right_qpos": [0.] * 7 + [.02, .02],
            "objects": [{"is_grasping": False}, {"is_grasping": True}]}
    assert retain_right_valid({"status": "release_timeout"}, [tick] * 30, ["left", "right"])
    assert not retain_right_valid({"status": "release_timeout", "error": "fault"}, [tick] * 30, ["left", "right"])
    assert not retain_right_valid({"status": "aborted"}, [tick] * 30, ["left", "right"])
    bounds = [[-.2, -.2, 0], [.2, .2, .1]]
    assert physical_region([0, 0, .05], bounds, [[-.1, -.1, .02], [.1, .1, .08]])["inside"]
    assert not physical_region([0, 0, .05], bounds, [[-.1, -.1, .02], [.3, .1, .08]])["inside"]
    # A rotated triangular prism fits while its empty local-box corners do not.
    # Keep the old local-box centre so only the bounds representation changes.
    from itertools import product
    prism = np.array([[x, y, z] for x, y in ((0., 0.), (.1, 0.), (0., .1)) for z in (0., .02)])
    basket_mesh = np.array(list(product((-.08, .08), (-.01, .09), (0., .1))))
    pose = np.array([0., 0., .02, np.cos(np.pi/8), 0., 0., np.sin(np.pi/8)])
    identity = np.array([0., 0., 0., 1., 0., 0., 0.])
    prism_before, basket_before, pose_before = prism.copy(), basket_mesh.copy(), pose.copy()
    tight = collision_vertex_region(prism, pose, basket_mesh, identity)
    assert tight["inside"]
    matrix = pose_matrix(pose)
    corners = np.array(list(product((0., .1), (0., .1), (0., .02))))
    coarse = corners @ matrix[:3, :3].T + matrix[:3, 3]
    assert not physical_region(tight["object_geom_center_m"],
        tight["basket_aabb_m"], [coarse.min(axis=0), coarse.max(axis=0)])["inside"]
    np.testing.assert_allclose(tight["object_geom_center_m"], [.0, np.sqrt(.005), .03], atol=1e-14)
    assert not np.allclose(tight["object_geom_center_m"], np.mean(tight["object_aabb_m"], axis=0))
    # Above-rim geometry remains diagnostic; the centre/floor/XY gates still apply.
    assert physical_region([0, 0, .05], bounds, [[-.1, -.1, .02], [.1, .1, .3]])["inside"]
    for moved in (pose + [.2, 0, 0, 0, 0, 0, 0], pose + [0, 0, -.03, 0, 0, 0, 0],
                  pose + [0, 0, .1, 0, 0, 0, 0]):
        assert not collision_vertex_region(prism, moved, basket_mesh, identity)["inside"]
    assert not collision_vertex_region(prism, pose, basket_mesh, identity + [.2, 0, 0, 0, 0, 0, 0])["inside"]
    assert all(np.array_equal(a, b) for a, b in ((prism, prism_before), (basket_mesh, basket_before), (pose, pose_before)))
    rejected = 0
    for bad_vertices, bad_pose in ((np.empty((0, 3)), pose), (np.zeros((3, 2)), pose),
                                   ([[0., 0., np.nan]], pose), (prism, np.zeros(7)),
                                   (prism, np.r_[np.nan, pose[1:]])):
        try:
            collision_vertex_region(bad_vertices, bad_pose, basket_mesh, identity)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Malformed collision geometry was accepted")
    assert rejected == 5
    rows = [{"control_step": i // 5, "sim_timestep_s": .01, "left_qpos": [0.] * 9, "right_qpos": [0.] * 9} for i in range(15)]
    assert hold_joint_substep_audit(rows, [1, 2])["status"] == "passed"
    rows[-1]["left_qpos"][0] = .01
    assert hold_joint_substep_audit(rows, [1, 2])["status"] == "failed"
    pose = [0, 0, .1, 1, 0, 0, 0]
    held = {"phase": "carry_lower", "tcp_poses": [pose, pose], "object_poses": [pose, pose]}
    assert held_substep_audit([held], [np.eye(4), np.eye(4)])["status"] == "passed"
    held["object_poses"] = [pose, [.01, 0, .1, 1, 0, 0, 0]]
    assert held_substep_audit([held], [np.eye(4), np.eye(4)])["status"] == "failed"
    return {"status": "passed", "checks": ["continue native success only, preserve failure and truncation", "right release fault only",
        "release control rejects errors", "full XY and height bounds",
        "actual collision vertices eliminate rotated empty corners and preserve local-box centre",
        "actual outside XY floor centre-height and moved basket still reject; rim remains diagnostic",
        "invalid collision meshes and poses reject without input mutation",
        "both100Hz joint envelopes", "both held object envelopes"]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("positive", "retain-right"), default="positive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu): parser.error("Select one available GPU UUID for rendering")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(f"sync-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    names = ["trajectory_schedule.py", "trajectory_execution.py", "combined_collision.py", "scene_collision.py", "grasp_collision.py",
             "grasp_execution.py", "validate_grasp.py", "held_collision.py", "lift_execution.py", "validate_lift.py", "place_execution.py",
             "validate_place.py", "single_arm_execution.py", "table_support_collision.py", "validate_banana.py",
             "dual_payload_collision.py", "dual_arm_execution.py", "validate_sync.py"]
    sources = {n: (HERE / n).read_bytes() for n in names}
    snapshot = output / "source"; snapshot.mkdir()
    for name, contents in sources.items(): (snapshot / name).write_bytes(contents)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "task": TASK, "seed": args.seed,
        "mode": args.mode, "gpu_uuid": gpu, "scope": "synchronized left apple and right banana on one physical control timeline",
        "synchronization": "phase barriers; simultaneous start; each arm holds final joint and gripper targets until both ready",
        "max_episode_steps": 400, "official_default_max_episode_steps": 200,
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
