"""Replay standard sync H5 actions from one recorded initial state.

The seeded reset is retained when it exactly matches recorded S0; otherwise S0
is restored once. Subsequent states come from physics, without extra actions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from mani_skill.agents.robots.embodiment_profiles import get_robot_profile
from mani_skill.trajectory.utils import index_dict
from .dual_arm_execution import (_Phase, ARM_SPEED, OBJECT_SPEED, OBJECT_ANGULAR_SPEED,
    WINDOW_POSITION, WINDOW_ROTATION, ENDPOINT)
from .measurements import _rotation_distance, finger_positions_m, finger_displacement_m
from .observation import ObservedEnvironment
from .physical_observation import measured_source_departure
from .pill_support_collision import PillDualPayloadCollisionChecker
from .pill_task import PillTask, TASK_ID
from .runtime_policy import RuntimePolicy
from .solve import SYNC_TASK_IDS, phase_sequence, pill_planners, task_components
from .trajectory_execution import as_numpy, initialize_replay_state

ACTION_ORDER = ("panda_wristcam-0", "panda_wristcam-1")


def action_layout(metadata: dict) -> tuple[tuple[str, str], tuple[int, int]]:
    """Read native controller widths; old Panda archives retain their contract."""
    motion = metadata.get("motion_planning", {})
    robots = tuple(motion.get("robot_uids", ("panda_wristcam", "panda_wristcam")))
    widths = {uid: get_robot_profile(uid).action_size
              for uid in ("panda_wristcam", "piper", "ur5_wsg", "arx_x5", "xarm6_robotiq")}
    if len(robots) != 2 or robots[0] != robots[1] or robots[0] not in widths:
        raise ValueError("Action replay requires a supported homogeneous robot pair")
    order = tuple(motion.get("action_order", ()))
    expected = tuple(f"{uid}-{i}" for i, uid in enumerate(robots))
    sizes = tuple(motion.get("action_sizes", (widths[robots[0]],)*2))
    if order != expected or sizes != (widths[robots[0]],)*2:
        raise ValueError("Missing or incompatible physical left/right action order or widths")
    return order, sizes


def release_separation_enabled(metadata: dict[str, Any], episode: dict[str, Any]) -> bool:
    """Replay the archive's retreat semantics, limited to the Panda pill route.

    New archives declare execution features explicitly. Before that declaration
    existed, the cuRobo branch also used separation on its MPLib route. Its
    physics summary records that fact; older cuRobo archives additionally
    identify the backend in collection metadata or the episode report.
    Legacy Panda archives identify their pair through the native action order.
    """
    sync = episode["dual_sync"]
    motion = metadata.get("motion_planning", {})
    features = sync.get("execution_features", {})
    if "release_separation" in features:
        enabled = features["release_separation"]
        if type(enabled) is not bool:
            raise ValueError("release_separation execution feature must be boolean")
    else:
        enabled = (isinstance(sync.get("physics", {}).get("release_separation"), dict)
                   or motion.get("planner_backend", sync.get("backend")) == "curobo")
    if not enabled or metadata["env_info"]["env_id"] != TASK_ID:
        return False
    order, _ = action_layout(metadata)
    if order != ACTION_ORDER:
        return False
    panda_pair = ("panda_wristcam",) * 2
    declarations = (motion, metadata["env_info"].get("env_kwargs", {}), sync)
    return all("robot_uids" not in source or tuple(source["robot_uids"]) == panda_pair
               for source in declarations)


def load_episode(path: str | Path, episode_id: int = 0) -> tuple:
    """Read T paired actions and S0, requiring explicit sync phase boundaries."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    episode = next((row for row in metadata["episodes"] if row["episode_id"] == episode_id), None)
    if episode is None or "dual_sync" not in episode:
        raise ValueError("Selected episode needs standard synchronous episode metadata")
    sync = episode["dual_sync"]
    if metadata["env_info"]["env_id"] not in SYNC_TASK_IDS or sync.get("schema") not in ("dual-panda-sync-v1", "dual-arm-sync-v2"):
        raise ValueError("This action replay supports declared synchronous tasks only")
    _, sizes = action_layout(metadata)
    policy = RuntimePolicy.from_dict(sync["policy"])
    if policy.max_episode_steps != 200 or policy.last_completion_step != 199:
        raise ValueError("The official 200-step horizon must be preserved")
    expected = phase_sequence(policy)
    if not policy.require_final_stability:
        expected.append("observe_task_outcome")
    stages = sync["stages"]
    if [row.get("name") for row in stages] != expected or any(row.get("status") != "passed" for row in stages):
        raise ValueError("Replay requires complete successful stage boundaries; partial records are unsupported")
    preparation_steps = 10
    if "preparation" in sync:
        preparation = sync["preparation"]
        if (not isinstance(preparation, dict)
                or type(preparation.get("environment_steps")) is not int
                or not 1 <= preparation["environment_steps"] <= 10):
            raise ValueError("Explicit preparation metadata requires environment_steps integer in [1,10]")
        preparation_steps = preparation["environment_steps"]
    phases = [("preparation", preparation_steps)] + [(row["name"], row["environment_steps"]) for row in stages]
    if any(type(count) is not int or count < 0 or (count == 0 and name != "observe_task_outcome")
           for name, count in phases):
        raise ValueError("Replay phase action counts must be positive, except a zero-step outcome observation")
    with h5py.File(path, "r") as h5:
        group = h5[f"traj_{episode_id}"]
        actions = np.asarray(group["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != sum(sizes) or not np.isfinite(actions).all():
            raise ValueError("Recorded actions must be finite T by native paired controller width")
        if len(actions) != sum(count for _, count in phases) or len(actions) >= policy.max_episode_steps:
            raise ValueError("Action count differs from phase metadata or official episode horizon")
        initial_state = index_dict(group["env_states"], 0)
    return metadata, episode, policy, phases, actions, initial_state


def paired_action(action: Any, order=ACTION_ORDER, sizes=(8, 8)) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (sum(sizes),) or not np.isfinite(action).all():
        raise ValueError("Expected one finite native paired action")
    offsets = np.r_[0, np.cumsum(sizes)]
    return {key: action[offsets[i]:offsets[i+1]].copy() for i, key in enumerate(order)}


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


def failure_details(name: str, report: dict[str, Any]) -> dict[str, Any]:
    """Keep existing refusal evidence without another collision/state query."""
    details = {"phase": name, **{key: report[key] for key in
        ("status", "reason", "error", "initial_collisions") if key in report}}
    if report.get("trace"):
        details["last_event"] = {key: report["trace"][-1][key] for key in
            ("error", "collisions", "attachment_geometry", "elapsed_step_delta_on_error")
            if key in report["trace"][-1]}
    return details


def replay(path: str | Path, episode_id: int = 0) -> dict[str, Any]:
    metadata, episode, policy, phases, actions, initial_state = load_episode(path, episode_id)
    order, sizes = action_layout(metadata)
    release_separation = release_separation_enabled(metadata, episode)
    task_id = metadata["env_info"]["env_id"]
    task_factory, checker_factory = task_components(task_id)
    kwargs = dict(metadata["env_info"]["env_kwargs"])
    kwargs.update(sim_backend="physx_cpu", num_envs=1, obs_mode="none")
    env = gym.make(task_id, **kwargs)
    raw, capture, checker = env.unwrapped, None, None
    result = {"schema": ("dual-panda-action-replay-v1" if raw.robot_uids[0] == "panda_wristcam"
                         else "dual-arm-action-replay-v2"),
        "task_id": task_id, "robot_uids": list(raw.robot_uids), "episode_id": episode_id,
        "seed": episode["episode_seed"], "recorded_actions": len(actions), "executed_actions": 0,
        "execution": {"status": "running"}, "quality": {"status": "not_evaluated"}}
    release_count = stable_count = 0
    retreat_gain = None
    try:
        reset = dict(episode["reset_kwargs"])
        reset["seed"] = episode["episode_seed"]
        env.reset(**reset)
        result["initialization"] = initialize_replay_state(raw, initial_state)
        if (raw.num_envs != 1 or not np.isclose(raw.control_timestep, .05)
                or not np.isclose(raw.sim_timestep, .01)
                or any(agent.control_mode != "pd_joint_pos" for agent in raw.agent.agents)):
            raise ValueError("Replay requires original 20 Hz absolute-joint control and 100 Hz CPU physics")
        task = task_factory(raw)
        capture = ObservedEnvironment(env, task, lambda: checker, policy)
        capture.install()
        predicates = task.region_predicates()
        for name, count in phases:
            capture.phase = name
            held = name in ("lift_and_hold", "carry_lower")
            if name == "pregrasp":
                solvers = pill_planners(env)
                checker = checker_factory.from_solvers(raw, *solvers, targets=task.targets)
                checker.set_measured_attachment_prediction(True)
            if name == "lift_and_hold":
                checker.attach()
                capture.references = np.asarray(checker.attachments).copy()
            if name == "release":
                checker.detach()
            phase = None
            if checker is not None:
                if name == "retreat" and release_separation:
                    capture.begin_release_separation(checker.begin_release_separation())
                checker.set_target_contacts_allowed((name in ("descend", "close_and_confirm", "release") or held,)*2)
                phase = _Phase(capture, checker, name, targets=task.targets, held=held, policy=policy)
                if not phase.preflight():
                    result["failure"] = failure_details(name, phase.report)
                    raise RuntimeError(f"{name}: {phase.report['status']}")
            if name == "retreat":
                retreat_start = np.asarray([as_numpy(agent.tcp.pose.p)[0] for agent in raw.agent.agents])
            window, grasp_counts = None, np.zeros(2, dtype=int)
            previous_fingers = ([finger_positions_m(arm, phase.report["initial_measurements"][arm.key])
                for arm in phase.arms] if phase is not None else None)
            row = phase.report["initial_observation"] if phase is not None else None
            for _ in range(count):
                command = paired_action(actions[result["executed_actions"]], order, sizes)
                before = int(as_numpy(raw.elapsed_steps).item())
                row = phase.tick(command, name, native_actions=True) if phase is not None else capture.step(command)
                result["executed_actions"] += int(as_numpy(raw.elapsed_steps).item())-before
                if phase is None:
                    continue
                if row is None or phase.report["status"] != "running" or (held and not phase.monitor(row)):
                    result["failure"] = failure_details(name, phase.report)
                    raise RuntimeError(f"{name}: {phase.report.get('reason', phase.report['status'])}")
                phase.report["trace"].clear()
                opened = all(phase.opened(arm, row) for arm in phase.arms)
                if name == "close_and_confirm":
                    if row["tracking_error_rad"] > ENDPOINT:
                        raise RuntimeError("Grasp arm hold error exceeded executor tolerance")
                    for i, arm in enumerate(phase.arms):
                        fingers = finger_positions_m(arm, row["measurements"][arm.key])
                        valid = row["objects"][i]["is_grasping"] and finger_displacement_m(fingers, previous_fingers[i]) <= .0002
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
                if release_separation:
                    if checker.release_separation.finish():
                        raise RuntimeError("Retreat ended before geometry separation completed")
                    if not capture.release_separation_complete:
                        raise RuntimeError("Retreat ended before physical contact separation completed")
                    capture.end_release_separation()
                    checker.end_release_separation()
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
        if checker is not None and release_separation:
            checker.end_release_separation()
        if capture is not None:
            if release_separation:
                capture.end_release_separation()
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
