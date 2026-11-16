"""Validate a physical left-apple lift and hover; right arm holds throughout.

The attachment exists only in collision prediction. No physical weld, actor
pose change, or reset makes the apple follow the hand. Release-control mode
explicitly overrides the left gripper action after a measured 10 cm rise.
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

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
RELEASE_ABORT_REASONS = frozenset(("grasp_lost", "position_slip", "rotation_slip", "object_dropped"))


def applied_action(action: dict[str, np.ndarray], *, mode: str, lift_active: bool,
                   height_gain_m: float, active_key: str) -> tuple[dict[str, np.ndarray], bool]:
    """Copy requests; only the named negative control can override the gripper."""
    if mode not in ("positive", "release-control") or not np.isfinite(height_gain_m):
        raise ValueError("Expected a known mode and finite measured height")
    actual = {key: np.asarray(value, dtype=np.float32).copy() for key, value in action.items()}
    if active_key not in actual or actual[active_key].shape != (8,):
        raise ValueError("Expected an eight-value active-arm action")
    inject = mode == "release-control" and lift_active and height_gain_m >= .10
    if inject:
        actual[active_key][-1] = 1
    return actual, inject


def release_control_valid(result: dict[str, Any], ticks: list[dict[str, Any]],
                          keys: list[str]) -> bool:
    """Accept an applied release followed by measured grasp failure, not errors."""
    if (result.get("status") != "aborted" or result.get("reason") not in RELEASE_ABORT_REASONS
            or any(result.get(key) for key in ("error", "error_type", "traceback"))
            or any(row.get("error") for row in result.get("trace", []))):
        return False
    released = [tick for tick in ticks if tick.get("fault_injected")]
    if not released or released[0].get("height_gain_before_step_m", -1) < .10:
        return False
    for tick in released:
        requested, actual = tick["requested_actions"], tick["actions"]
        if (requested[keys[0]][-1] != -1 or actual[keys[0]][-1] != 1
                or not np.allclose(requested[keys[0]][:7], actual[keys[0]][:7], rtol=0, atol=2e-7)
                or any(not np.allclose(requested[key], actual[key], rtol=0, atol=2e-7) for key in keys[1:])):
            return False
    return True


def pose_matrix(pose: Any) -> np.ndarray:
    from transforms3d.quaternions import quat2mat
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all() or not np.isclose(np.linalg.norm(values[3:]), 1, atol=1e-5):
        raise ValueError("Expected one finite xyz/wxyz pose")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = quat2mat(values[3:]), values[:3]
    return matrix


def substep_summary(rows: list[dict[str, Any]], reference: np.ndarray) -> dict[str, Any]:
    """Audit real physics-substep pose envelopes, separately from velocity rates."""
    if not rows:
        raise ValueError("Physical substep observations are required")
    position, rotation, angular_rates = [], [], []
    peak_z, max_drop = -np.inf, 0.
    for row in rows:
        obj, tcp = pose_matrix(row["apple_pose"]), pose_matrix(row["left_tcp_pose"])
        relative = np.linalg.inv(tcp) @ obj
        position.append(float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])))
        cosine = (np.trace(reference[:3, :3].T @ relative[:3, :3]) - 1) / 2
        rotation.append(float(np.arccos(np.clip(cosine, -1, 1))))
        rate = np.asarray(row["apple_angular_velocity_rad_s"], dtype=float)
        if rate.shape != (3,) or not np.isfinite(rate).all():
            raise ValueError("Expected a finite measured substep angular velocity")
        angular_rates.append(float(np.linalg.norm(rate)))
        peak_z = max(peak_z, obj[2, 3])
        max_drop = max(max_drop, peak_z - obj[2, 3])
    return {"status": "passed" if max(position) <= .005 and max(rotation) <= .15 and max_drop <= .01 else "failed",
            "physical_substeps": len(rows), "max_position_slip_m": max(position),
            "max_rotation_slip_rad": max(rotation), "max_height_drop_from_peak_m": float(max_drop),
            "max_instant_angular_speed_rad_s": max(angular_rates),
            "position_limit_m": .005, "rotation_limit_rad": .15, "drop_limit_m": .01,
            "scope": "sampled physics-substep attachment pose envelope, not continuous collision detection"}


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import sapien
    import torch
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy
    from .grasp_execution import plan_left_reach, execute_reaching, close_and_confirm
    from .grasp_collision import LiveGraspCollisionChecker
    from .held_collision import AttachedObjectCollisionChecker
    from .lift_execution import plan_left_lift, execute_lift

    env, raw, frames, ticks = None, None, [], []
    substeps: list[dict[str, Any]] = []
    original_after_simulation_step = None
    keys: list[str] = []
    fault_events: list[dict[str, Any]] = []
    report["fault_events"] = fault_events
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos",
                        sim_backend="cpu", render_backend="gpu", render_mode="rgb_array", max_episode_steps=200)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False,
                            record_env_state=True, source_type="motionplanning",
                            source_desc=f"development physical left lift and hover; mode={args.mode}")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        keys = list(raw.agent.agents_dict)
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), stages=[])
        initial_right = as_numpy(raw.right_agent.robot.get_qpos())[0, :7].copy()

        class Capture:
            unwrapped = raw
            phase = "preparation"
            lift_active = False
            lift_initial_z = 0.0
            release_latched = False
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
                    raise ValueError("Every tick must submit both arms")
                gain = float(as_numpy(raw.apple_out.pose.p)[0, 2]) - self.lift_initial_z
                actual, inject = applied_action(action, mode=args.mode, lift_active=self.lift_active,
                                                height_gain_m=gain, active_key=keys[0])
                if inject:
                    self.release_latched = True
                if self.release_latched:
                    actual[keys[0]][-1] = 1
                    fault_events.append({"next_environment_step": int(as_numpy(raw.elapsed_steps)[0]) + 1,
                                         "type": "override_left_gripper_open", "height_gain_before_step_m": gain})
                self.pending_control_step = int(as_numpy(raw.elapsed_steps)[0]) + 1
                result = env.step(actual)
                left = raw.left_agent
                forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, raw.apple_out))[0]
                          for link in (left.finger1_link, left.finger2_link)]
                ticks.append({"phase": self.phase, "environment_step": int(as_numpy(raw.elapsed_steps)[0]),
                              "requested_actions": json_value(action), "actions": json_value(actual),
                              "fault_injected": self.release_latched,
                              "height_gain_before_step_m": gain if self.lift_active else None,
                              "left_qpos": as_numpy(left.robot.get_qpos())[0].tolist(),
                              "right_qpos": as_numpy(raw.right_agent.robot.get_qpos())[0].tolist(),
                              "left_tcp_pose": as_numpy(left.tcp.pose.raw_pose)[0].tolist(),
                              "apple_pose": as_numpy(raw.apple_out.pose.raw_pose)[0].tolist(),
                              "apple_linear_velocity_m_s": as_numpy(raw.apple_out.linear_velocity)[0].tolist(),
                              "is_grasping": bool(as_numpy(left.is_grasping(raw.apple_out))[0]),
                              "finger_contact_force_norms_N": [float(np.linalg.norm(force)) for force in forces],
                              "right_initial_error_rad": float(np.max(np.abs(
                                  as_numpy(raw.right_agent.robot.get_qpos())[0, :7] - initial_right)))})
                self.frame()
                return result

        captured = Capture()
        original_after_simulation_step = raw._after_simulation_step

        def record_substep() -> None:
            original_after_simulation_step()
            if captured.lift_active:
                substeps.append({"substep": len(substeps) + 1,
                                 "control_step": captured.pending_control_step,
                                 "sim_timestep_s": float(raw.sim_timestep),
                                 "apple_pose": as_numpy(raw.apple_out.pose.raw_pose)[0].tolist(),
                                 "left_tcp_pose": as_numpy(raw.left_agent.tcp.pose.raw_pose)[0].tolist(),
                                 "apple_linear_velocity_m_s": as_numpy(raw.apple_out.linear_velocity)[0].tolist(),
                                 "apple_angular_velocity_rad_s": as_numpy(raw.apple_out.angular_velocity)[0].tolist()})

        raw._after_simulation_step = record_substep
        captured.frame()
        hold = {key: np.r_[as_numpy(agent.robot.get_qpos())[0, :7], 1].astype(np.float32)
                for key, agent in raw.agent.agents_dict.items()}
        for _ in range(10):
            captured.step(hold)
        report["preparation_steps"] = 10
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=agent.robot.pose, multi_robot_id=i,
                                               debug=False, vis=False, visualize_target_grasp_pose=False,
                                               print_env_info=False)
                   for i, agent in enumerate(raw.agent.agents)]
        live = LiveGraspCollisionChecker.from_solvers(raw, *solvers, target_actor=raw.apple_out,
                                                     active_robot="left")
        report["scene_coverage"] = live.coverage
        for name, pose, contact in zip(("pregrasp", "descend"), grasp_poses(raw), (False, True)):
            report["stage"] = captured.phase = name
            before = int(as_numpy(raw.elapsed_steps)[0])
            schedule = plan_left_reach(*solvers, pose)
            assert int(as_numpy(raw.elapsed_steps)[0]) == before
            save_report(output / "report.json", report)
            result = execute_reaching(captured, schedule, live, allow_target_contact=contact)
            report["stages"].append({"name": name, "planning_environment_steps": 0, **result})
            save_report(output / "report.json", report)
            if result["status"] != "passed":
                report.update(status="failed", reason=f"{name}:{result['status']}")
                return
        report["stage"] = captured.phase = "close_and_confirm"
        result = close_and_confirm(captured, live, active_key=keys[0], target_actor=raw.apple_out)
        report["stages"].append({"name": "close_and_confirm", **result})
        save_report(output / "report.json", report)
        if result["status"] != "passed":
            report.update(status="failed", reason=f"close_and_confirm:{result['status']}")
            return
        report["stage"] = "attach_collision_prediction"
        attached = AttachedObjectCollisionChecker.from_live(live, raw.apple_out, active_robot="left")
        report["attachment_initial_coverage"] = json_value(attached.coverage)
        report["attachment_coverage"] = attached.coverage
        report["attachment_T_tcp_object"] = np.asarray(attached.attachment_T_tcp_object).tolist()
        captured.lift_initial_z = float(as_numpy(raw.apple_out.pose.p)[0, 2])
        report["lift_initial_object_z_m"] = captured.lift_initial_z
        current_tcp = raw.left_agent.tcp.pose
        target = sapien.Pose(as_numpy(current_tcp.p)[0] + np.array([0., 0., .12]), as_numpy(current_tcp.q)[0])
        solvers[0].gripper_state = -1
        solvers[1].gripper_state = 1
        before = int(as_numpy(raw.elapsed_steps)[0])
        schedule = plan_left_lift(*solvers, target)
        assert int(as_numpy(raw.elapsed_steps)[0]) == before
        np.savez(output / "lift-plan.npz", left_time=schedule.trajectories[0].times_s,
                 left_position=schedule.trajectories[0].positions_rad, target_times=schedule.target_times_s,
                 left_commands=schedule.left_commands, right_commands=schedule.right_commands)
        report["lift_planning"] = {"environment_steps": 0, "requested_tcp_rise_m": .12,
                                   "scheduled_steps": schedule.num_steps}
        report["stage"] = captured.phase = "lift_and_hold"
        captured.lift_active = True
        save_report(output / "report.json", report)
        result = execute_lift(captured, schedule, attached, raw.apple_out, lift_height_m=.12, hold_steps=20)
        report["attachment_coverage"] = attached.coverage
        report["stages"].append({"name": "lift_and_hold", **result})
        report["physical_substep_audit"] = substep_summary(substeps, np.asarray(attached.attachment_T_tcp_object)) if substeps else None
        report["fault_events"] = fault_events
        report["observed_lift_status"] = result["status"]
        if args.mode == "positive":
            report["status"] = "passed" if (result["status"] == "passed" and not fault_events
                                               and report["physical_substep_audit"]["status"] == "passed") else "failed"
        else:
            report["expected_outcome"] = "abort after explicit gripper-open injection; no continuation"
            report["release_control_verified"] = release_control_valid(result, ticks, keys)
            report["status"] = "passed" if report["release_control_verified"] else "failed"
        report["official_task_evaluation"] = json_value(raw.evaluate())
        report["right_max_error_from_initial_rad"] = max(t["right_initial_error_rad"] for t in ticks)
        if report["right_max_error_from_initial_rad"] > .03:
            report.update(status="failed", reason="right_global_hold_tolerance")
        report["stage"] = "finished"
    finally:
        if raw is not None:
            report["total_environment_steps"] = int(as_numpy(raw.elapsed_steps)[0])
            report["torch_cuda_initialized"] = torch.cuda.is_initialized()
            if original_after_simulation_step is not None:
                raw._after_simulation_step = original_after_simulation_step
        errors = report.setdefault("artifact_errors", [])
        if substeps:
            try:
                with (output / "physics-substeps.jsonl").open("w", encoding="utf-8") as handle:
                    for row in substeps:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
            except Exception as exc:
                errors.append(f"physics substeps: {type(exc).__name__}: {exc}")
        if ticks:
            try:
                with (output / "ticks.jsonl").open("w", encoding="utf-8") as handle:
                    for tick in ticks:
                        handle.write(json.dumps(tick, allow_nan=False) + "\n")
            except Exception as exc:
                errors.append(f"ticks: {type(exc).__name__}: {exc}")
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
                expected = np.asarray([np.concatenate([tick["actions"][key] for key in keys])
                                       for tick in ticks]) if ticks else None
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
    requests = {"left": np.r_[np.zeros(7), -1.], "right": np.r_[np.ones(7), 1.]}
    for mode, active, height, expected in (("positive", True, .15, False),
                                           ("release-control", False, .15, False),
                                           ("release-control", True, .09, False),
                                           ("release-control", True, .10, True)):
        actual, injected = applied_action(requests, mode=mode, lift_active=active,
                                          height_gain_m=height, active_key="left")
        assert injected == expected and set(actual) == set(requests)
        assert actual["left"][-1] == (1 if expected else -1)
        assert np.array_equal(actual["right"], requests["right"])
        assert requests["left"][-1] == -1
    released, _ = applied_action(requests, mode="release-control", lift_active=True,
                                height_gain_m=.11, active_key="left")
    tick = {"fault_injected": True, "height_gain_before_step_m": .11,
            "requested_actions": requests, "actions": released}
    result = {"status": "aborted", "reason": "grasp_lost", "trace": []}
    assert release_control_valid(result, [tick], ["left", "right"])
    assert not release_control_valid({**result, "reason": "step_or_measurement_failed"}, [tick], ["left", "right"])
    assert not release_control_valid({**result, "error": "checker failed"}, [tick], ["left", "right"])
    assert not release_control_valid(result, [], ["left", "right"])
    assert not release_control_valid(result, [{**tick, "height_gain_before_step_m": .09}], ["left", "right"])
    pose = [0, 0, .1, 1, 0, 0, 0]
    observation = {"apple_pose": pose, "left_tcp_pose": pose, "apple_angular_velocity_rad_s": [0, 0, 2.]}
    assert substep_summary([observation], np.eye(4))["status"] == "passed"
    bad = {**observation, "apple_pose": [.01, 0, .1, 1, 0, 0, 0]}
    assert substep_summary([bad], np.eye(4))["status"] == "failed"
    assert (ROOT / "infrastructure").is_dir()
    return {"status": "passed", "checks": ["fault injection is explicit and mode/height gated",
                                             "requested actions not mutated", "right arm preserved",
                                             "release control rejects unrelated errors and unapplied faults", "artifact root"]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("positive", "release-control"), default="positive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
        parser.error("Select one available GPU UUID for rendering")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(
        f"lift-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    names = ["trajectory_schedule.py", "trajectory_execution.py", "combined_collision.py", "scene_collision.py",
             "grasp_collision.py", "grasp_execution.py", "validate_grasp.py", "held_collision.py",
             "lift_execution.py", "validate_lift.py"]
    sources = {name: (HERE / name).read_bytes() for name in names}
    snapshot = output / "source"
    snapshot.mkdir()
    for name, contents in sources.items():
        (snapshot / name).write_bytes(contents)
    report: dict[str, Any] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                              "task": TASK, "seed": args.seed, "mode": args.mode, "gpu_uuid": gpu,
                              "scope": "physical left apple lift by 12cm and 20-tick hover; right holds",
                              "physical_attachment_created": False, "continuous_collision_detection": False,
                              "source_snapshot": str(snapshot),
                              "source_sha256": {name: hashlib.sha256(contents).hexdigest()
                                                for name, contents in sources.items()}}
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
