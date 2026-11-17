"""Exercise real robot controls in Empty-v1 and save measured capabilities.

This is a loading/control example, not a manipulation benchmark. Arm commands are
absolute radians in the named controller order. Gripper commands follow each
agent's controller range; measurements use physical finger-link separation.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401: registers Empty-v1 and its agents
from mani_skill.utils import sapien_utils


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def action_for_pose(agent: Any, arm_target: np.ndarray, gripper: float) -> np.ndarray:
    """Pack (N,) named arm targets and one gripper target into native action."""
    parts: dict[str, torch.Tensor] = {}
    for name, controller in agent.controller.controllers.items():
        width = controller.single_action_space.shape[0]
        if name == "arm":
            value = arm_target
        elif "gripper" in name and width:
            value = np.full(width, gripper, dtype=np.float32)
        elif width == 0:
            value = np.empty(0, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported control group {name!r}; specify its hold targets")
        if value.shape != (width,):
            raise ValueError(f"{name}: targets {value.shape}, action width {width}")
        parts[name] = torch.as_tensor(value, dtype=torch.float32, device=agent.device)
    return as_array(agent.controller.from_action_dict(parts)).astype(np.float32)


def finger_distance(agent: Any) -> float:
    p1 = as_array(agent.finger1_link.pose.p)[0]
    p2 = as_array(agent.finger2_link.pose.p)[0]
    return float(np.linalg.norm(p1 - p2))


def rgb_frame(observation: dict[str, Any]) -> np.ndarray:
    frame = as_array(observation["sensor_data"]["base_camera"]["rgb"])[0]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {frame.shape}")
    return frame.astype(np.uint8)


def smoke_robot(robot_uid: str, output_root: Path, seed: int, steps: int) -> dict[str, Any]:
    """Run one CPU-physics robot, retaining a separate result for every attempt."""
    directory = output_root / f"{robot_uid}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"robot_uid": robot_uid, "seed": seed, "output_dir": str(directory), "passed": False}
    env = None
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    qpositions: list[np.ndarray] = []
    start_time = time.monotonic()
    try:
        env = gym.make(
            "Empty-v1", robot_uids=robot_uid, num_envs=1,
            obs_mode="rgb", reward_mode="none", control_mode="pd_joint_pos",
            sim_backend="physx_cpu", render_backend="sapien_cuda",
            sim_config=dict(sim_freq=100, control_freq=20),
            sensor_configs=dict(width=256, height=256, shader_pack="default", fov=1.0,
                                pose=sapien_utils.look_at([1.0, -1.0, 1.0], [0.0, 0.0, 0.3])),
        )
        base = env.unwrapped
        agent = base.agent
        robot = agent.robot
        joint_names = [joint.name for joint in robot.get_active_joints()]
        arm_indices = [joint_names.index(name) for name in agent.arm_joint_names]
        result.update(load=True, active_joint_names=joint_names, arm_joint_names=agent.arm_joint_names,
                      action_dim=int(np.prod(env.action_space.shape)), qpos_dim=len(joint_names),
                      control_freq=base.control_freq, sim_freq=base.sim_freq,
                      tcp_link=agent.tcp.name)
        keyframe = agent.keyframes["rest"]

        def reset_to_rest() -> np.ndarray:
            env.reset(seed=seed)
            agent.reset(torch.as_tensor(keyframe.qpos, dtype=torch.float32, device=base.device))
            robot.set_pose(keyframe.pose)
            agent.controller.reset()
            return as_array(robot.get_qpos())[0].copy()

        q_initial = reset_to_rest()
        first_reset_pose = as_array(agent.tcp.pose.raw_pose).copy()
        q_second = reset_to_rest()
        reset_error = float(np.max(np.abs(q_initial - q_second)))
        reset_tcp_error = float(np.max(np.abs(first_reset_pose - as_array(agent.tcp.pose.raw_pose))))
        result["reset"] = {"keyframe": "rest", "max_qpos_error": reset_error,
                           "max_tcp_pose_error": reset_tcp_error, "passed": reset_error < 1e-6 and reset_tcp_error < 1e-6}

        grip_name, grip_controller = next(
            (name, controller) for name, controller in agent.controller.controllers.items()
            if "gripper" in name and controller.single_action_space.shape[0]
        )
        grip_space = grip_controller.single_action_space
        open_action = float(getattr(agent, "gripper_open_action", grip_space.low[0]))
        close_action = float(getattr(agent, "gripper_closed_action", grip_space.high[0]))
        arm_target = q_initial[arm_indices].copy()
        result["gripper_controller"] = {"name": grip_name, "open_action": open_action,
                                        "close_action": close_action, "normalized": bool(grip_controller.config.normalize_action)}

        def advance(target: np.ndarray, grip: float, count: int) -> None:
            action = action_for_pose(agent, target, grip)
            for _ in range(count):
                observation, _, _, _, _ = env.step(action)
                frames.append(rgb_frame(observation))
                actions.append(action.copy())
                qpositions.append(as_array(robot.get_qpos())[0].copy())

        advance(arm_target, open_action, steps)
        imageio.imwrite(directory / "initial.png", frames[-1])
        initial_arm = as_array(robot.get_qpos())[0, arm_indices].copy()
        initial_tcp = as_array(agent.tcp.pose.p)[0].copy()
        limits = as_array(robot.get_qlimits())[0, arm_indices]
        offset = np.full(len(arm_indices), 0.06, dtype=np.float32)
        offset[0] = 0.12
        offset[1::2] *= -1
        outside = ((arm_target + offset) < limits[:, 0] + 0.01) | ((arm_target + offset) > limits[:, 1] - 0.01)
        offset[outside] *= -1
        arm_target = np.clip(arm_target + offset, limits[:, 0] + 0.01, limits[:, 1] - 0.01)
        advance(arm_target, open_action, steps)
        imageio.imwrite(directory / "arm_moved.png", frames[-1])
        actual_arm = as_array(robot.get_qpos())[0, arm_indices].copy()
        displacement = actual_arm - initial_arm
        target_error = float(np.max(np.abs(actual_arm - arm_target)))
        tcp_displacement = float(np.linalg.norm(as_array(agent.tcp.pose.p)[0] - initial_tcp))
        result["arm_motion"] = {"command_rad": arm_target.tolist(), "actual_rad": actual_arm.tolist(),
                                "displacement_rad": displacement.tolist(), "max_target_error_rad": target_error,
                                "passed": bool(np.all(np.abs(displacement) > 0.005) and target_error < 0.03)}
        result["tcp"] = {"displacement_m": tcp_displacement, "passed": tcp_displacement > 0.001}

        # Repeat both directions, measuring link geometry rather than commanded values.
        advance(arm_target, close_action, steps)
        closed_distance = finger_distance(agent)
        imageio.imwrite(directory / "gripper_closed.png", frames[-1])
        advance(arm_target, open_action, steps)
        open_distance = finger_distance(agent)
        imageio.imwrite(directory / "gripper_open.png", frames[-1])
        result["gripper"] = {"measurement": "finger link origin separation", "closed_distance_m": closed_distance,
                             "open_distance_m": open_distance, "change_m": open_distance - closed_distance,
                             "passed": open_distance - closed_distance > 0.001}
        before_hold = as_array(robot.get_qpos())[0].copy()
        advance(arm_target, open_action, 20)
        drift = float(np.max(np.abs(as_array(robot.get_qpos())[0] - before_hold)))
        velocity = float(np.max(np.abs(as_array(robot.get_qvel())[0])))
        result["hold"] = {"steps": 20, "max_qpos_drift": drift, "max_qvel": velocity,
                          "passed": drift < 0.01 and velocity < 0.1}
        if not result["hold"]["passed"]:
            result["hold"]["qvel"] = as_array(robot.get_qvel())[0].tolist()
            contacts = []
            for contact in base.scene.get_contacts():
                impulse = sum((np.asarray(point.impulse) for point in contact.points), np.zeros(3))
                if np.linalg.norm(impulse) > 1e-6:
                    contacts.append({"links": [body.entity.name for body in contact.bodies], "impulse": impulse.tolist()})
            result["hold"]["contacts"] = contacts
        rgb_min = min(int(frame.min()) for frame in frames)
        rgb_max = max(int(frame.max()) for frame in frames)
        changed_pixels = float(np.mean(np.abs(frames[steps - 1].astype(float) - frames[2 * steps - 1].astype(float))))
        result["rgb"] = {"shape": list(frames[0].shape), "range": [rgb_min, rgb_max],
                         "mean_motion_pixel_change": changed_pixels, "passed": rgb_max > rgb_min and changed_pixels > 0}
        # Reset after actual motion, not just twice in the initial unchanged state.
        after_motion_reset = reset_to_rest()
        result["reset"]["after_motion_max_qpos_error"] = float(np.max(np.abs(after_motion_reset - q_initial)))
        result["reset"]["after_motion_max_tcp_pose_error"] = float(np.max(np.abs(first_reset_pose - as_array(agent.tcp.pose.raw_pose))))
        result["reset"]["passed"] = result["reset"]["passed"] and result["reset"]["after_motion_max_qpos_error"] < 1e-6 and result["reset"]["after_motion_max_tcp_pose_error"] < 1e-6
        result["passed"] = all(result[name]["passed"] for name in ("reset", "arm_motion", "tcp", "gripper", "hold", "rgb"))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        (directory / "error.txt").write_text(traceback.format_exc())
    finally:
        if frames:
            imageio.mimwrite(directory / "control.mp4", frames, fps=20)
            np.savez_compressed(directory / "controls.npz", actions=np.asarray(actions), qpos=np.asarray(qpositions))
        if env is not None:
            env.close()
        result["elapsed_seconds"] = time.monotonic() - start_time
        (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-uids", nargs="+", default=["piper", "arx_x5", "ur5_wsg", "xarm6_robotiq"])
    parser.add_argument("--output-dir", type=Path, required=True, help="Parent directory; every attempt gets a new child")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60, help="Control steps for each motion/open/close stage")
    options = parser.parse_args(args)
    if options.steps <= 0:
        parser.error("--steps must be positive")
    results = [smoke_robot(uid, options.output_dir, options.seed, options.steps) for uid in options.robot_uids]
    print(json.dumps(results, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
