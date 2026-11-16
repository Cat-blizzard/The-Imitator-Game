"""Bounded native validation of option A; --execute opts into a short reaching phase.

Uses CPU physics/MPLib and one explicitly selected GPU for SAPIEN rendering.
Writes every run to a new directory, including rejected paths and source hashes.
"""
from __future__ import annotations

import argparse
import faulthandler
from datetime import datetime, timezone
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


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def target_offsets() -> tuple[np.ndarray, np.ndarray]:
    return np.array([0., -1., 1.]) * .03 / np.sqrt(2), np.array([0., 1., 1.]) * .08 / np.sqrt(2)


class SceneGuard:
    """Reject stale snapshot geometry, with explicit numerical pose tolerances."""
    def __init__(self, env: Any, native: Any, coverage: dict[str, Any]):
        from .trajectory_execution import as_numpy
        self.env, self.native, self.coverage = env, native, coverage
        self.scope = "sampled_dual_arm_self_mutual_and_exported_scene_with_plane_halfspace_guard"
        self.entities = self._entities()
        self.actors = {name: entity.pose.to_transformation_matrix().copy()
                       for name, entity in self.entities.items()}
        self.robots = [(as_numpy(agent.robot.pose.raw_pose).copy(),
                        as_numpy(agent.robot.get_qpos())[0, 7:].copy()) for agent in env.agent.agents]
        # These two immobile bases are bolted to the tabletop in this task.
        # Keep every moving link/table/object collision enabled.
        self.allowed_mount_contacts = {(robot, "panda_link0", "scene::table::shape_0")
                                       for robot in ("left", "right")}
        self.allowed_mount_observations = 0

    def assert_scene_unchanged(self) -> None:
        entities = self._entities()
        if set(self.actors) != set(entities):
            raise RuntimeError("Scene actor set changed after collision export")
        for name, initial in self.actors.items():
            current = entities[name].pose.to_transformation_matrix()
            if not np.allclose(current, initial, rtol=0, atol=1e-4):
                raise RuntimeError(f"Scene geometry moved after collision export: {name}")

    def _entities(self) -> dict[str, Any]:
        from .scene_collision import scene_entities
        return scene_entities(self.env)

    def assert_robot_geometry_unchanged(self) -> None:
        from .trajectory_execution import as_numpy
        for agent, (base, fingers) in zip(self.env.agent.agents, self.robots):
            if not np.allclose(as_numpy(agent.robot.pose.raw_pose), base, rtol=0, atol=1e-6):
                raise RuntimeError("Robot base moved after collision model creation")
            if not np.allclose(as_numpy(agent.robot.get_qpos())[0, 7:], fingers, rtol=0, atol=1e-4):
                raise RuntimeError("Finger geometry changed after collision model creation")

    def check(self, left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
        from .scene_collision import check_plane_halfspaces
        result = []
        for collision in self.native.check(left, right):
            key = (collision["object_name1"], collision["link_name1"], collision["object_name2"])
            if key in self.allowed_mount_contacts:
                self.allowed_mount_observations += 1
            else:
                result.append(collision)
        result.extend(check_plane_halfspaces(self.native.models, self.coverage["planes"]))
        return result


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import sapien
    import torch
    import mani_skill.envs
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy, plan_pair, execute_schedule
    from .combined_collision import NativeDualCollisionChecker, validate_schedule
    from .scene_collision import export_scene_collision

    env = None
    try:
        env = gym.make("TwoRobotPickRemoteControl-v1", obs_mode="none", reward_mode="none",
                       control_mode="pd_joint_pos", sim_backend="cpu", render_backend="gpu",
                       render_mode="rgb_array", max_episode_steps=200)
        env.reset(seed=args.seed)
        raw = env.unwrapped
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), task="TwoRobotPickRemoteControl-v1")
        if args.execute:
            # Let passive objects settle before freezing scene geometry.
            hold = {key: np.r_[as_numpy(agent.robot.get_qpos())[0, :7], 1].astype(np.float32)
                    for key, agent in raw.agent.agents_dict.items()}
            for _ in range(10): env.step(hold)
            report["preparation_steps"] = 10
        solvers = [PandaArmMotionPlanningSolver(env, base_pose=agent.robot.pose, multi_robot_id=i,
                                               debug=False, vis=False, visualize_target_grasp_pose=False,
                                               print_env_info=False)
                   for i, agent in enumerate(raw.agent.agents)]
        poses = [sapien.Pose(as_numpy(agent.tcp.pose.p)[0] + offset, as_numpy(agent.tcp.pose.q)[0])
                 for agent, offset in zip(raw.agent.agents, target_offsets())]
        steps_before = as_numpy(raw.elapsed_steps).copy()
        schedule = plan_pair(*solvers, *poses)
        assert np.array_equal(steps_before, as_numpy(raw.elapsed_steps))
        report["planning"] = {"environment_steps": 0,
                              "left_duration_s": schedule.trajectories[0].duration_s,
                              "right_duration_s": schedule.trajectories[1].duration_s,
                              "common_horizon_s": float(schedule.target_times_s[-1]),
                              "scheduled_control_steps": schedule.num_steps}
        np.savez(output / "trajectory.npz", left_time=schedule.trajectories[0].times_s,
                 left_position=schedule.trajectories[0].positions_rad,
                 right_time=schedule.trajectories[1].times_s,
                 right_position=schedule.trajectories[1].positions_rad,
                 target_times=schedule.target_times_s, left_commands=schedule.left_commands,
                 right_commands=schedule.right_commands)
        starts = [t.positions_rad[0] for t in schedule.trajectories]
        bare = NativeDualCollisionChecker.from_solvers(*solvers)
        report["stage"] = "checking_dual_arm_path"
        save_report(output / "report.json", report)
        report["dual_arm_validation"] = validate_schedule(schedule, *starts, bare)
        # A real negative control: coincident robot bases must cause inter-arm collision.
        original_base = bare.models[1].get_base_pose().copy()
        bare.models[1].set_base_pose(bare.models[0].get_base_pose())
        negative = bare.check(*starts)
        bare.models[1].set_base_pose(original_base)
        assert any({row["object_name1"], row["object_name2"]} == {"left", "right"} for row in negative)
        report["native_collision_negative_control"] = {"detected": True, "collisions": negative}
        objects, names, coverage = export_scene_collision(raw)
        report["scene_coverage"] = coverage
        # Native Panda-convex vs FCL Plane segfaults in this pinned MPLib 0.1.1.
        # Ground is instead fully guarded by the conservative halfspace test.
        plane_names = {plane["name"] for plane in coverage["planes"]}
        pairs = [(obj, name) for obj, name in zip(objects, names) if name not in plane_names]
        objects, names = [p[0] for p in pairs], [p[1] for p in pairs]
        coverage["plane_check_backend"] = "conservative_bounding_sphere_halfspace"
        coverage["native_world_object_count"] = len(objects)
        report["stage"] = "constructing_full_collision_world"
        save_report(output / "report.json", report)
        native = NativeDualCollisionChecker.from_solvers(*solvers, normal_objects=objects,
                                                        normal_object_names=names)
        checker = SceneGuard(raw, native, coverage)
        report["allowed_contacts"] = {"pairs": [list(pair) for pair in sorted(checker.allowed_mount_contacts)],
                                      "reason": "fixed Panda bases mounted to tabletop; base poses are guarded"}
        report["stage"] = "checking_full_scene_path"
        save_report(output / "report.json", report)
        report["scene_validation"] = validate_schedule(schedule, *starts, checker)
        save_report(output / "report.json", report)
        report["status"] = "passed" if report["scene_validation"]["status"] == "passed" else "rejected"
        if args.execute and report["status"] == "passed":
            def capture_frame() -> np.ndarray:
                frame = as_numpy(env.render())
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError(f"Expected HWC RGB uint8 render, got {frame.shape}/{frame.dtype}")
                return frame.copy()
            frames = [capture_frame()]
            class Capture:
                unwrapped = raw
                def step(self, action: Any) -> tuple[Any, ...]:
                    result = env.step(action)
                    frames.append(capture_frame())
                    return result
            execution = execute_schedule(Capture(), schedule, checker)
            report["execution"] = execution
            report["stage"] = "execution_finished"
            save_report(output / "report.json", report)
            report["status"] = "passed" if execution["status"] == "completed" else execution["status"]
            import imageio.v2 as imageio
            # Display still frames before/after the motion without extra physics ticks.
            imageio.mimsave(output / "execution.mp4", [frames[0]] * 20 + frames + [frames[-1]] * 20,
                           fps=round(raw.control_freq), macro_block_size=1)
            report["video"] = {"path": str(output / "execution.mp4"), "padding_frames_per_side": 20,
                               "playback_fps": raw.control_freq}
        report["torch_cuda_initialized"] = torch.cuda.is_initialized()
        report["total_environment_steps"] = as_numpy(raw.elapsed_steps).tolist()
    finally:
        if env is not None:
            report["total_environment_steps"] = as_numpy(env.unwrapped.elapsed_steps).tolist()
            report["torch_cuda_initialized"] = torch.cuda.is_initialized()
            env.close()


def self_test() -> dict[str, Any]:
    left, right = target_offsets()
    assert np.isclose(np.linalg.norm(left), .03) and np.isclose(np.linalg.norm(right), .08)
    assert left[1] < 0 < right[1] and left[2] > 0 and right[2] > 0
    assert (ROOT / "infrastructure").is_dir()
    return {"status": "passed", "checks": ["fixed 3cm/8cm outward/upward targets", "artifact root"]}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu):
        parser.error("Select one GPU UUID for the SAPIEN renderer")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime("native-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    files = [HERE / name for name in ["trajectory_schedule.py", "trajectory_execution.py",
                                     "combined_collision.py", "scene_collision.py", "validate_execution.py"]]
    report: dict[str, Any] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                              "seed": args.seed, "gpu_uuid": gpu, "execute_requested": args.execute,
                              "preparation_steps": 0, "training_started": False,
                              "continuous_collision_detection": False, "grasping_validated": False,
                              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    started = time.monotonic()
    try:
        native_run(args, output, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if hasattr(exc, "coverage"): report["scene_coverage"] = exc.coverage
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
