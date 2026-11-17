"""Replay standard sync H5 actions from one recorded initial state.

Only the initial state is restored. Every subsequent state comes from physics;
no planning or extra settling actions are inserted during replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.trajectory.utils import index_dict
from .dual_arm_execution import (_Phase, ARM_SPEED, OBJECT_SPEED, OBJECT_ANGULAR_SPEED,
    WINDOW_POSITION, WINDOW_ROTATION, ENDPOINT)
from .measurements import _rotation_distance
from .observation import ObservedEnvironment
from .physical_observation import measured_source_departure
from .pill_support_collision import PillDualPayloadCollisionChecker
from .pill_task import PillTask, TASK_ID
from .runtime_policy import RuntimePolicy
from .solve import phase_sequence
from .trajectory_execution import as_numpy

ACTION_ORDER = ("panda_wristcam-0", "panda_wristcam-1")


def load_episode(path: str | Path, episode_id: int = 0) -> tuple:
    """Read T paired actions and S0, requiring explicit sync phase boundaries."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    episode = next((row for row in metadata["episodes"] if row["episode_id"] == episode_id), None)
    if episode is None or "dual_sync" not in episode:
        raise ValueError("Selected episode needs standard synchronous episode metadata")
    sync = episode["dual_sync"]
    if metadata["env_info"]["env_id"] != TASK_ID or sync.get("schema") != "dual-panda-sync-v1":
        raise ValueError("This action replay supports the synchronous pill task only")
    if tuple(metadata.get("motion_planning", {}).get("action_order", ())) != ACTION_ORDER:
        raise ValueError("Missing or incompatible physical left/right action order")
    policy = RuntimePolicy.from_dict(sync["policy"])
    if policy.max_episode_steps != 200 or policy.last_completion_step != 199:
        raise ValueError("The official 200-step horizon must be preserved")
    expected = phase_sequence(policy)
    if not policy.require_final_stability:
        expected.append("observe_task_outcome")
    stages = sync["stages"]
    if [row.get("name") for row in stages] != expected or any(row.get("status") != "passed" for row in stages):
        raise ValueError("Replay requires complete successful stage boundaries; partial records are unsupported")
    phases = [("preparation", 10)] + [(row["name"], row["environment_steps"]) for row in stages]
    if any(type(count) is not int or count < 0 or (count == 0 and name != "observe_task_outcome")
           for name, count in phases):
        raise ValueError("Replay phase action counts must be positive, except a zero-step outcome observation")
    with h5py.File(path, "r") as h5:
        group = h5[f"traj_{episode_id}"]
        actions = np.asarray(group["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
            raise ValueError("Recorded actions must be finite T by 16 absolute joint/gripper commands")
        if len(actions) != sum(count for _, count in phases) or len(actions) >= policy.max_episode_steps:
            raise ValueError("Action count differs from phase metadata or official episode horizon")
        initial_state = index_dict(group["env_states"], 0)
    return metadata, episode, policy, phases, actions, initial_state


def paired_action(action: Any) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise ValueError("Expected one finite 16-component paired action")
    return {key: action[8*i:8*(i+1)].copy() for i, key in enumerate(ACTION_ORDER)}


def placement_window(row: dict, opened: bool, regions: list[bool], count: int,
                     window: list[np.ndarray] | None) -> tuple[int, list | None]:
    """Use the executor's release, velocity and cumulative drift criteria."""
    valid = opened and all(regions) and row["max_arm_velocity_rad_s"] <= ARM_SPEED and all(
        not obj["is_grasping"] and obj["control_frame_linear_speed_m_s"] <= OBJECT_SPEED
        and obj["control_frame_angular_speed_rad_s"] <= OBJECT_ANGULAR_SPEED for obj in row["objects"])
    if not valid:
        return 0, None
    poses = [np.asarray(obj["object_T_world"]) for obj in row["objects"]]
    if window is None:
        window = [pose.copy() for pose in poses]
    if any(np.linalg.norm(pose[:3, 3]-reference[:3, 3]) > WINDOW_POSITION
           or _rotation_distance(reference[:3, :3], pose[:3, :3]) > WINDOW_ROTATION
           for pose, reference in zip(poses, window)):
        return 1, [pose.copy() for pose in poses]
    return count+1, window


def replay(path: str | Path, episode_id: int = 0) -> dict[str, Any]:
    metadata, episode, policy, phases, actions, initial_state = load_episode(path, episode_id)
    kwargs = dict(metadata["env_info"]["env_kwargs"])
    kwargs.update(sim_backend="physx_cpu", num_envs=1, obs_mode="none")
    env = gym.make(TASK_ID, **kwargs)
    raw, capture, checker = env.unwrapped, None, None
    result = {"schema": "dual-panda-action-replay-v1", "episode_id": episode_id,
        "seed": episode["episode_seed"], "recorded_actions": len(actions), "executed_actions": 0,
        "execution": {"status": "running"}, "quality": {"status": "not_evaluated"}}
    release_count = stable_count = 0
    retreat_gain = None
    try:
        reset = dict(episode["reset_kwargs"])
        reset["seed"] = episode["episode_seed"]
        env.reset(**reset)
        raw.set_state_dict(initial_state)
        if (raw.num_envs != 1 or not np.isclose(raw.control_timestep, .05)
                or not np.isclose(raw.sim_timestep, .01)
                or any(agent.control_mode != "pd_joint_pos" for agent in raw.agent.agents)):
            raise ValueError("Replay requires original 20 Hz absolute-joint control and 100 Hz CPU physics")
        task = PillTask(raw)
        capture = ObservedEnvironment(env, task, lambda: checker, policy)
        capture.install()
        predicates = task.region_predicates()
        for name, count in phases:
            capture.phase = name
            held = name in ("lift_and_hold", "carry_lower")
            if name == "pregrasp":
                solvers = [PandaArmMotionPlanningSolver(env, base_pose=agent.robot.pose, multi_robot_id=i,
                    debug=False, vis=False, visualize_target_grasp_pose=False, print_env_info=False)
                    for i, agent in enumerate(raw.agent.agents)]
                checker = PillDualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=task.targets)
                checker.set_measured_attachment_prediction(True)
            if name == "lift_and_hold":
                checker.attach()
                capture.references = np.asarray(checker.attachments).copy()
            if name == "release":
                checker.detach()
            phase = None
            if checker is not None:
                checker.set_target_contacts_allowed((name in ("descend", "close_and_confirm", "release") or held,)*2)
                phase = _Phase(capture, checker, name, targets=task.targets, held=held, policy=policy)
                if not phase.preflight():
                    raise RuntimeError(f"{name}: {phase.report['status']}")
            if name == "retreat":
                retreat_start = np.asarray([as_numpy(agent.tcp.pose.p)[0] for agent in raw.agent.agents])
            window, grasp_counts = None, np.zeros(2, dtype=int)
            previous_fingers = ([np.asarray(phase.report["initial_measurements"][arm.key]["finger_qpos_m"])
                for arm in phase.arms] if phase is not None else None)
            row = phase.report["initial_observation"] if phase is not None else None
            for _ in range(count):
                command = paired_action(actions[result["executed_actions"]])
                before = int(as_numpy(raw.elapsed_steps).item())
                row = phase.tick(command, name) if phase is not None else capture.step(command)
                result["executed_actions"] += int(as_numpy(raw.elapsed_steps).item())-before
                if phase is None:
                    continue
                phase.report["trace"].clear()
                if row is None or phase.report["status"] != "running" or (held and not phase.monitor(row)):
                    raise RuntimeError(f"{name}: {phase.report.get('reason', phase.report['status'])}")
                opened = all(phase.opened(arm, row) for arm in phase.arms)
                if name == "close_and_confirm":
                    if row["tracking_error_rad"] > ENDPOINT:
                        raise RuntimeError("Grasp arm hold error exceeded executor tolerance")
                    for i, arm in enumerate(phase.arms):
                        fingers = np.asarray(row["measurements"][arm.key]["finger_qpos_m"])
                        valid = row["objects"][i]["is_grasping"] and np.max(np.abs(fingers-previous_fingers[i])) <= .0002
                        grasp_counts[i] = grasp_counts[i]+1 if valid else 0
                        previous_fingers[i] = fingers
                if name == "release":
                    valid = opened and row["tracking_error_rad"] <= ENDPOINT and all(not obj["is_grasping"] for obj in row["objects"])
                    release_count = release_count+1 if valid else 0
                if name == "verify_placement":
                    evaluation = raw.evaluate()
                    regions = [bool(as_numpy(evaluation[key]).item()) and bool(predicate(actor)["inside"])
                        for key, predicate, actor in zip(task.official_predicate_keys, predicates, task.targets)]
                    stable_count, window = placement_window(row, opened and row["tracking_error_rad"] <= ENDPOINT,
                        regions, stable_count, window)
            if name == "close_and_confirm" and min(grasp_counts) < policy.grasp_stable_steps:
                raise RuntimeError("Recorded grasp window did not reproduce")
            if name == "lift_and_hold" and (not all(measured_source_departure(checker)) or not all(capture.physical_released)
                    or any(obj["object_height_gain_m"] < policy.min_lift_m for obj in row["objects"])):
                raise RuntimeError("Physical source departure did not reproduce")
            if name == "retreat":
                retreat_gain = np.asarray([as_numpy(agent.tcp.pose.p)[0] for agent in raw.agent.agents])-retreat_start
        result["execution"]["status"] = "passed"
        final_regions = [predicate(actor) for predicate, actor in zip(predicates, task.targets)]
        final_opened = all(phase.opened(arm, row) for arm in phase.arms)
        final_released = final_opened and all(not obj["is_grasping"] for obj in row["objects"])
        released = release_count >= policy.release_stable_steps and final_released
        result["placement"] = {"inside": all(region["inside"] for region in final_regions), "regions": final_regions}
        quality = ((not policy.require_retreat or retreat_gain is not None and np.all(retreat_gain[:, 2] >= policy.minimum_retreat_m))
            and (not policy.require_final_stability or stable_count >= policy.final_stable_steps))
        result["release"] = {"confirmed": bool(released), "stable_steps": release_count, "final_fingers_open": final_opened, "final_not_grasping": all(not obj["is_grasping"] for obj in row["objects"])}
        result["quality"] = {"status": ("passed" if quality else "failed") if policy.require_retreat or policy.require_final_stability else "not_requested",
            "retreat_evaluated": policy.require_retreat, "final_stability_evaluated": policy.require_final_stability,
            "final_stable_steps": stable_count,
            "retreat_displacement_m": None if retreat_gain is None else retreat_gain.tolist()}
    except Exception as exc:
        result["execution"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        if capture is not None:
            capture.restore()
            result["physics"] = capture.summary()
        success = bool(as_numpy(raw.evaluate()["success"]).item())
        steps = int(as_numpy(raw.elapsed_steps).item())
        result["official"] = {"success": success, "elapsed_steps": steps, "within_horizon": steps < policy.max_episode_steps,
            "truncated": bool(as_numpy(capture.last_step[3]).item()) if capture is not None and capture.last_step else False}
        env.close()
    result["passed"] = bool(result["execution"]["status"] == "passed" and result["official"]["success"]
        and result["official"]["within_horizon"] and not result["official"]["truncated"]
        and result.get("release", {}).get("confirmed", False) and result["quality"]["status"] in ("passed", "not_requested")
        and result.get("placement", {}).get("inside", False)
        and all(value <= policy.position_slip_m for value in result["physics"]["max_held_position_slip_m"])
        and all(value <= policy.rotation_slip_rad for value in result["physics"]["max_held_rotation_slip_rad"]))
    return result


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-path", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional new JSON report; existing files are preserved.")
    args = parser.parse_args(args)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    result = replay(args.traj_path, args.episode_id)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized+"\n")
    print(serialized)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
