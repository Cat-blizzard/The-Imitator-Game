"""Validate left-arm apple contact grasp with paired actions and recorded states.

This stage ends on the table: it does not lift, attach, or transport the apple.
CPU physics/MPLib use one explicitly selected GPU only for SAPIEN rendering.
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

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
TASK = "TwoRobotPickAppleBananaToBaskets-v1"


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def grasp_poses(raw: Any) -> tuple[Any, Any]:
    """Reuse the task solution's OBB grasp and 8 cm pregrasp offset."""
    import sapien
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
    from .trajectory_execution import as_numpy
    closing = as_numpy(raw.left_agent.tcp.pose.to_transformation_matrix())[0, :3, 1]
    info = compute_grasp_info_by_obb(get_actor_obb(raw.apple_out), approaching=np.array([0, 0, -1]),
                                   target_closing=closing, depth=.025)
    grasp = raw.left_agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
    return grasp * sapien.Pose([0, 0, -.08]), grasp


def recording_summary(path: Path, expected_steps: int,
                      expected_actions: np.ndarray | None = None) -> dict[str, Any]:
    import h5py
    with h5py.File(path, "r") as h5:
        if expected_steps == 0:
            return {"path": str(path), "recorded_steps": 0}
        group = h5["traj_0"]
        lengths: dict[str, int] = {}
        group.visititems(lambda name, obj: lengths.update({name: len(obj)})
                         if isinstance(obj, h5py.Dataset) else None)
        actions = group["actions"][()]
        states = {k: n for k, n in lengths.items() if k.startswith("env_states/")}
        if actions.shape != (expected_steps, 16) or not np.isfinite(actions).all() or not states or any(
                n != expected_steps + 1 for n in states.values()):
            raise RuntimeError(f"Recording lengths disagree with actual steps: {lengths}")
        if expected_actions is not None and (expected_actions.shape != actions.shape or
                not np.allclose(actions, expected_actions, rtol=0, atol=1e-7)):
            raise RuntimeError("Recorded arm action order/values differ from submitted actions")
        return {"path": str(path), "recorded_steps": expected_steps,
                "actions_shape": list(actions.shape), "submitted_actions_matched": expected_actions is not None,
                "state_lengths": states,
                "state_playback_tested": False}


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import torch
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy
    from .grasp_execution import plan_left_reach, execute_reaching, close_and_confirm
    from .grasp_collision import LiveGraspCollisionChecker

    env, raw, frames, ticks = None, None, [], []
    action_keys: list[str] = []
    captured = None
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos",
                        sim_backend="cpu", render_backend="gpu", render_mode="rgb_array", max_episode_steps=200)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False,
                            record_env_state=True, source_type="motionplanning",
                            source_desc="development paired executor; contact grasp stage, no lift")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        action_keys = list(raw.agent.agents_dict)
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep))
        initial_right = as_numpy(raw.right_agent.robot.get_qpos())[0, :7].copy()

        class Capture:
            unwrapped = raw
            phase = "settle"

            def frame(self) -> None:
                frame = as_numpy(env.render())
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError("Expected HWC uint8 RGB render")
                frames.append(frame.copy())

            def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
                if set(action) != set(raw.agent.agents_dict):
                    raise ValueError("Every tick must contain both arms")
                result = env.step(action)
                left = raw.left_agent
                forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, raw.apple_out))[0]
                          for link in (left.finger1_link, left.finger2_link)]
                ticks.append({"phase": self.phase, "environment_step": int(as_numpy(raw.elapsed_steps)[0]),
                              "actions": json_value(action),
                              "left_qpos": as_numpy(left.robot.get_qpos())[0].tolist(),
                              "right_qpos": as_numpy(raw.right_agent.robot.get_qpos())[0].tolist(),
                              "left_tcp_pose": as_numpy(left.tcp.pose.raw_pose)[0].tolist(),
                              "apple_pose": as_numpy(raw.apple_out.pose.raw_pose)[0].tolist(),
                              "is_grasping": bool(as_numpy(left.is_grasping(raw.apple_out))[0]),
                              "finger_contact_forces_N": [f.tolist() for f in forces],
                              "finger_contact_force_norms_N": [float(np.linalg.norm(f)) for f in forces],
                              "right_initial_error_rad": float(np.max(np.abs(
                                  as_numpy(raw.right_agent.robot.get_qpos())[0, :7] - initial_right)))})
                self.frame()
                return result

        captured = Capture()
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
        checker = LiveGraspCollisionChecker.from_solvers(raw, *solvers, target_actor=raw.apple_out,
                                                        active_robot="left")
        report["scene_coverage"] = checker.coverage
        report["stages"] = []
        if args.mode == "positive":
            pregrasp, grasp = grasp_poses(raw)
            report["poses"] = {"pregrasp_wxyz": np.r_[pregrasp.p, pregrasp.q].tolist(),
                               "grasp_wxyz": np.r_[grasp.p, grasp.q].tolist()}
            for name, pose, allow_contact in (("pregrasp", pregrasp, False), ("descend", grasp, True)):
                report["stage"] = captured.phase = name
                checker.set_target_contact_allowed(allow_contact)
                before = int(as_numpy(raw.elapsed_steps)[0])
                schedule = plan_left_reach(*solvers, pose)
                assert int(as_numpy(raw.elapsed_steps)[0]) == before, "Planning advanced the environment"
                np.savez(output / f"{name}-plan.npz", left_time=schedule.trajectories[0].times_s,
                         left_position=schedule.trajectories[0].positions_rad,
                         target_times=schedule.target_times_s, left_commands=schedule.left_commands,
                         right_commands=schedule.right_commands)
                save_report(output / "report.json", report)
                result = execute_reaching(captured, schedule, checker, allow_target_contact=allow_contact)
                report["stages"].append({"name": name, "planning_environment_steps": 0, **result})
                save_report(output / "report.json", report)
                if result["status"] != "passed":
                    report.update(status="failed", reason=f"{name}:{result['status']}")
                    return
        report["stage"] = captured.phase = "close_and_confirm"
        checker.set_target_contact_allowed(True)
        save_report(output / "report.json", report)
        result = close_and_confirm(captured, checker, active_key=list(raw.agent.agents_dict)[0],
                                   target_actor=raw.apple_out, max_steps=40, stable_steps=5)
        report["stages"].append({"name": "close_and_confirm", **result})
        if args.mode == "positive":
            report["status"] = "passed" if result["status"] == "passed" else "failed"
        else:
            report["status"] = "passed" if result["status"] == "grasp_timeout" else "failed"
            report["expected_outcome"] = "grasp_timeout; no later stage may run"
        report["observed_grasp_status"] = result["status"]
        report["official_task_evaluation"] = json_value(raw.evaluate())
        report["right_max_error_from_initial_rad"] = max(t["right_initial_error_rad"] for t in ticks)
        if report["right_max_error_from_initial_rad"] > .03:
            report.update(status="failed", reason="right_arm_global_hold_tolerance")
        report["allowed_mount_contacts"] = [list(pair) for pair in sorted(checker.allowed_mount_contacts)]
        report["allowed_contact_observations"] = checker.allowed_contact_observations
        report["stage"] = "finished"
    finally:
        if raw is not None:
            report["total_environment_steps"] = int(as_numpy(raw.elapsed_steps)[0])
            report["torch_cuda_initialized"] = torch.cuda.is_initialized()
        artifact_errors = report.setdefault("artifact_errors", [])
        if ticks:
            try:
                with (output / "ticks.jsonl").open("w", encoding="utf-8") as handle:
                    for tick in ticks:
                        handle.write(json.dumps(tick, allow_nan=False) + "\n")
            except Exception as exc:
                artifact_errors.append(f"tick trace: {type(exc).__name__}: {exc}")
        if env is not None:
            try:
                if getattr(env, "_trajectory_buffer", None) is not None:
                    env.flush_trajectory()
            except Exception as exc:
                artifact_errors.append(f"record flush: {type(exc).__name__}: {exc}")
            finally:
                try:
                    env.close()
                except Exception as exc:
                    artifact_errors.append(f"environment close: {type(exc).__name__}: {exc}")
            try:
                expected = np.asarray([np.concatenate([tick["actions"][key] for key in action_keys])
                                       for tick in ticks]) if ticks else None
                report["recording"] = recording_summary(output / "trajectory.h5", len(ticks), expected)
                report["recording"]["action_key_order"] = action_keys
            except Exception as exc:
                artifact_errors.append(f"record verification: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "execution.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20,
                               fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
                report["video"] = {"path": str(output / "execution.mp4"), "playback_fps": 20,
                                   "padding_frames_before": 10, "padding_frames_after": 20}
            except Exception as exc:
                artifact_errors.append(f"video: {type(exc).__name__}: {exc}")
        if artifact_errors:
            report["status"] = "failed"


def self_test() -> dict[str, Any]:
    import tempfile
    import h5py
    value = json_value({"array": np.array([1., 2.]), "flag": np.bool_(True)})
    assert value == {"array": [1., 2.], "flag": True}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("traj_0")
            group.create_dataset("actions", data=np.arange(32).reshape(2, 16))
            group.create_dataset("env_states/actors/apple", data=np.zeros((3, 13)))
        actions = np.arange(32).reshape(2, 16)
        assert recording_summary(path, 2, actions)["recorded_steps"] == 2
        try:
            recording_summary(path, 2, np.roll(actions, 8, axis=1))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Wrong arm action order accepted")
        try:
            recording_summary(path, 3)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Invalid record length accepted")
    assert (ROOT / "infrastructure").is_dir()
    return {"status": "passed", "checks": ["JSON conversion", "recorded action/state lengths", "artifact root"]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("positive", "empty-grasp"), default="positive")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
        parser.error("Select one available GPU UUID for the SAPIEN renderer")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(
        f"grasp-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    names = ["trajectory_schedule.py", "trajectory_execution.py", "combined_collision.py",
             "scene_collision.py", "grasp_collision.py", "grasp_execution.py", "validate_grasp.py"]
    report: dict[str, Any] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                              "task": TASK, "seed": args.seed, "mode": args.mode, "gpu_uuid": gpu,
                              "scope": "left apple contact grasp on table; right arm holds; no lift or attachment",
                              "grasp_predicate": {"min_force_N": .5, "max_angle_degrees": 85,
                                                   "consecutive_stable_ticks": 5},
                              "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                                for name in names}}
    save_report(output / "report.json", report)
    started = time.monotonic()
    try:
        native_run(args, output, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if hasattr(exc, "coverage"):
            report["scene_coverage"] = exc.coverage
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
