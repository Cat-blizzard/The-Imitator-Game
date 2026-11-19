"""Replay a recorded shared pick/place episode using only its native actions.

Restore S0 once. Planning models provide measured collision checks, never new
plans or corrective actions. State differences are telemetry, not task success.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import h5py
import numpy as np

from mani_skill.trajectory.utils import index_dict
from .arm_planner import ArmMotionPlanner
from .dual_arm_execution import (ARM_SPEED, OBJECT_SPEED, OBJECT_ANGULAR_SPEED,
                                 WINDOW_POSITION, WINDOW_ROTATION)
from .dual_payload_collision import DualPayloadCollisionChecker
from .measurements import _rotation_distance, _transform, gripper_is_open, gripper_open_settings
from .runtime_policy import RuntimePolicy
from .trajectory_execution import as_numpy, bindings, initialize_replay_state

PHASES = ("preparation", "pregrasp", "descend", "grasp", "lift", "carry", "lower",
          "release", "retreat", "settle")


def _flat(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, (dict, h5py.Group)):
        result = {}
        for name, child in value.items():
            result.update(_flat(child, f"{prefix}/{name}" if prefix else name))
        return result
    return {prefix: value}


def validate_action_order(order: Any, width: int) -> None:
    """Validate physical controller identity before splitting a 14/16-D row."""
    if not isinstance(order, list) or len(order) != 2:
        raise ValueError("Exactly two recorded native controller descriptions are required")
    keys = []
    for item in order:
        if not isinstance(item, dict):
            raise ValueError("Each action description must be a mapping")
        key, size = item.get("key"), item.get("size")
        if not isinstance(key, str) or not key or type(size) is not int:
            raise ValueError("Native action keys and integer sizes are required")
        keys.append(key)
        arm, state = item.get("arm_joint_names"), item.get("state_joint_names")
        if any(not isinstance(names, list) or not names or
               any(not isinstance(name, str) or not name for name in names) or
               len(names) != len(set(names)) for names in (arm, state)):
            raise ValueError("Recorded arm/state joint names must be unique nonempty lists")
        if size != len(arm)+1 or not set(arm).issubset(state):
            raise ValueError("Native action width must cover named arm joints and one gripper")
    if len(set(keys)) != 2 or sum(item["size"] for item in order) != width:
        raise ValueError("Native action keys or total width differ from the recorded actions")


def validate_live_action_order(order: list[dict], arms: Any) -> None:
    if [item["key"] for item in order] != [arm.key for arm in arms]:
        raise ValueError("Recorded action keys/order differ from the live robot pair")
    for item, arm in zip(order, arms):
        if item["arm_joint_names"] != list(arm.names):
            raise ValueError(f"Recorded arm joint order differs for {arm.key}")
        if item["state_joint_names"] != list(arm.all_names):
            raise ValueError(f"Recorded state joint order differs for {arm.key}")
        if tuple(arm.agent.controller.single_action_space.shape) != (item["size"],):
            raise ValueError(f"Recorded native action size differs for {arm.key}")


def load_archive(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.with_suffix(".json").read_text())
    original = metadata["embodiment_sync"]
    if metadata["env_info"]["env_id"] != "DualArmPickPlace-v1" or original["task"] != "DualArmPickPlace-v1":
        raise ValueError("This replay entry requires the shared dual-arm pick/place task")
    policy = RuntimePolicy.from_dict(original["policy"])
    with h5py.File(path, "r") as source:
        episode = metadata["episodes"]
        if len(episode) != 1:
            raise ValueError("Replay one recorded episode at a time")
        group = source[f'traj_{episode[0]["episode_id"]}']
        actions = np.asarray(group["actions"], dtype=np.float32)
        if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("Recorded actions must be a nonempty finite T by native-width matrix")
        validate_action_order(original["action_order"], actions.shape[1])
        for name in ("obs", "env_states"):
            leaves = _flat(group[name])
            if not leaves or any(not isinstance(leaf, h5py.Dataset) or leaf.ndim < 1 or
                                 leaf.shape[0] != len(actions)+1 for leaf in leaves.values()):
                raise ValueError(f"{name} must contain exactly T+1 frames for T actions")
        for item in original["action_order"]:
            qpos = group[f'obs/agent/{item["key"]}/qpos']
            if qpos.shape != (len(actions)+1, len(item["state_joint_names"])):
                raise ValueError(f'Recorded qpos width differs for {item["key"]}')
        initial = index_dict(group["env_states"], 0)
        states = {name: np.asarray(leaf) for name, leaf in _flat(group["env_states"]).items()}
        if any(not np.isfinite(value).all() for value in states.values()):
            raise ValueError("Recorded environment states must be finite")
    events = original["events"]
    if len(events) != len(actions):
        raise ValueError("Action/event count mismatch")
    previous_phase = -1
    for step, event in enumerate(events, 1):
        if event.get("step") != step or event.get("phase") not in PHASES:
            raise ValueError("Recorded events require ordered control ticks and known phases")
        phase = PHASES.index(event["phase"])
        if phase < previous_phase:
            raise ValueError("Recorded phase order goes backwards")
        previous_phase = phase
    frequency = original["control_freq"]
    if type(frequency) not in (int, float) or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("A positive recorded control frequency is required")
    return {"metadata": metadata, "original": original, "policy": policy,
            "actions": actions, "initial": initial, "states": states}


def _state_error(raw: Any, source: dict[str, np.ndarray], frame: int) -> float:
    current = _flat(raw.get_state_dict())
    if set(current) != set(source):
        raise ValueError("Current environment state names differ from the source archive")
    largest = 0.
    for name, value in current.items():
        actual = as_numpy(value)
        expected = source[name][frame]
        if actual.shape == (1, *expected.shape):
            actual = actual[0]
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError(f"Current environment state shape differs for {name}")
        largest = max(largest, float(np.max(np.abs(actual-expected))))
    return largest


def held_observations(tcp_poses: list[np.ndarray], object_poses: list[np.ndarray],
                      references: tuple[np.ndarray, np.ndarray], grasping: list[bool],
                      previous_loss: list[int], policy: RuntimePolicy) -> list[dict[str, Any]]:
    """Apply the generator's existing measured slip and two-tick loss rules."""
    rows = []
    for tcp, obj, reference, held, loss in zip(tcp_poses, object_poses, references,
                                              grasping, previous_loss):
        relation = np.linalg.inv(_transform(tcp)) @ _transform(obj)
        slip = float(np.linalg.norm(relation[:3, 3]-reference[:3, 3]))
        rotation = float(_rotation_distance(reference[:3, :3], relation[:3, :3]))
        loss = 0 if held else loss+1
        reasons = []
        if slip > policy.position_slip_m:
            reasons.append("position_slip")
        if rotation > policy.rotation_slip_rad:
            reasons.append("rotation_slip")
        if loss >= 2:
            reasons.append("grasp_lost")
        rows.append({"position_slip_m": slip, "rotation_slip_rad": rotation,
                     "consecutive_grasp_loss_ticks": loss, "violations": reasons})
    return rows


def replay(path: Path, output: Path, *, save_video: bool = False) -> dict[str, Any]:
    """Run the archive's actions from matched reset or one S0 restoration."""
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "multi-embodiment-action-replay-v2", "passed": False,
              "executed_actions": 0, "restored_states": 0, "failures": [],
              "events": [], "collision_events": [], "max_state_error": 0.,
              "state_error_is_acceptance_gate": False, "replanning_calls": 0,
              "monitoring": "sampled_control_ticks", "continuous_collision_detection": False}
    started, env, writer, phase = time.monotonic(), None, None, None

    def fail(reason: str, **details: Any) -> None:
        result["failures"].append({"reason": reason, "phase": phase,
                                   "action_index": result["executed_actions"], **details})

    try:
        archive = load_archive(path)
        # Imported here to keep the CLI module free to delegate to this file.
        from .pick_place import ObservedEpisode, native_action, _facts, scalar
        metadata, original, policy = (archive[name] for name in ("metadata", "original", "policy"))
        actions = archive["actions"]
        result.update(recorded_actions=len(actions), policy=policy.to_dict(),
                      source_scope=original.get("scope"),
                      source_execution_status=original.get("execution", {}).get("status"))
        kwargs = dict(metadata["env_info"]["env_kwargs"])
        kwargs.update(obs_mode="rgb", sim_backend="physx_cpu", num_envs=1)
        env = gym.make("DualArmPickPlace-v1", **kwargs)
        raw = env.unwrapped
        env.reset(seed=metadata["episodes"][0]["episode_seed"])
        arms = bindings(raw, tuple(raw.agent.agents_dict))
        validate_live_action_order(original["action_order"], arms)
        if not np.isclose(1/raw.control_timestep, original["control_freq"], rtol=0, atol=1e-8):
            raise ValueError("Replay control frequency differs from the recorded frequency")
        _state_error(raw, archive["states"], 0)  # Check names/shapes before writing any state.
        result["initialization"] = initialize_replay_state(raw, archive["initial"])
        result["restored_states"] = result["initialization"]["state_write_count"]
        result["max_state_error"] = _state_error(raw, archive["states"], 0)
        observed = ObservedEpisode(env)
        result["events"] = observed.rows
        checker, attachment_references, grasp_loss = None, None, [0, 0]
        previous_pose = [as_numpy(obj.pose.to_transformation_matrix())[0].copy() for obj in raw.targets]
        initial_height = [pose[2, 3] for pose in previous_pose]
        lifted, max_lift = [False, False], [0., 0.]
        grasp_count, grasp_max = 0, 0
        release_count, release_max = 0, 0
        release_tcp, retreat_distance = None, [0., 0.]
        stable_count, stable_max, stability_reference = 0, 0, None
        requirements = {"min_lift_m": max(.08, policy.min_lift_m),
                        "minimum_retreat_m": max(.08, policy.minimum_retreat_m),
                        "final_stable_steps": max(20, policy.final_stable_steps),
                        "gripper_open": {arm.key: gripper_open_settings(arm) for arm in arms}}
        result["requirements"] = requirements
        if save_video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(output / "replay.mp4", fps=original["control_freq"])

        def frame() -> None:
            if writer is not None:
                image = as_numpy(env.render())
                writer.append_data(image[0] if image.ndim == 4 else image)

        frame()
        for action, source in zip(actions, original["events"]):
            previous_phase, phase = phase, source["phase"]
            if phase != previous_phase:
                if phase == "pregrasp":
                    solvers = [ArmMotionPlanner(observed, arm.agent) for arm in arms]
                    checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=raw.targets)
                    checker.set_measured_attachment_prediction(True)
                if phase not in ("preparation", "pregrasp") and checker is None:
                    raise ValueError("A replay motion phase appeared without pregrasp collision initialization")
                if phase == "lift":
                    if grasp_max < policy.grasp_stable_steps:
                        raise RuntimeError("Both real grasps were not confirmed before payload attachment")
                    checker.attach()
                    attachment_references = tuple(reference.copy() for reference in checker.attachments)
                if phase == "release":
                    if not checker.attached:
                        raise RuntimeError("Release appeared without preceding carried payloads")
                    checker.detach()
                if phase == "retreat":
                    release_tcp = [as_numpy(arm.agent.tcp.pose.p)[0].copy() for arm in arms]
                if checker is not None:
                    checker.set_target_contacts_allowed((phase in ("descend", "grasp", "release"),)*2)
            observed.phase = phase
            before = int(scalar(raw.elapsed_steps))
            observed.step(native_action(action, original["action_order"]))
            result["executed_actions"] += 1
            row = observed.rows[-1]
            if int(scalar(raw.elapsed_steps)) != before+1:
                raise RuntimeError("Replay must execute exactly one physical tick per recorded action")
            poses = [as_numpy(obj.pose.to_transformation_matrix())[0].copy() for obj in raw.targets]
            linear = [float(np.linalg.norm(p[:3, 3]-old[:3, 3])/raw.control_timestep)
                      for p, old in zip(poses, previous_pose)]
            angular = [float(_rotation_distance(old[:3, :3], p[:3, :3])/raw.control_timestep)
                       for p, old in zip(poses, previous_pose)]
            previous_pose = poses
            row.update(object_linear_speed_m_s=linear, object_angular_speed_rad_s=angular)
            held_failed = False
            if phase in ("lift", "carry", "lower"):
                if attachment_references is None:
                    raise RuntimeError("Held motion requires the original measured grasp references")
                row["held_objects"] = held_observations(
                    [_transform(arm.agent.tcp.pose.to_transformation_matrix()) for arm in arms],
                    poses, attachment_references, row["grasping"], grasp_loss, policy)
                grasp_loss = [item["consecutive_grasp_loss_ticks"] for item in row["held_objects"]]
                for index, item in enumerate(row["held_objects"]):
                    for reason in item["violations"]:
                        fail(reason, object_index=index, measurement=item)
                        held_failed = True
            grasp_count = grasp_count+1 if phase == "grasp" and all(row["grasping"]) else 0
            grasp_max = max(grasp_max, grasp_count)
            for index, (pose, held) in enumerate(zip(poses, row["grasping"])):
                if phase == "lift" and held:
                    gain = float(pose[2, 3]-initial_height[index])
                    max_lift[index] = max(max_lift[index], gain)
                    lifted[index] |= gain >= requirements["min_lift_m"]
            opened = [gripper_is_open(arm) for arm in arms]
            row["fingers_open"] = opened
            release_count = release_count+1 if phase in ("release", "retreat", "settle") and not any(row["grasping"]) and all(opened) else 0
            release_max = max(release_max, release_count)
            if release_tcp is not None:
                retreat_distance = [float(np.linalg.norm(as_numpy(arm.agent.tcp.pose.p)[0]-start))
                                    for arm, start in zip(arms, release_tcp)]
            row["retreat_distance_m"] = list(retreat_distance)
            arm_speed = max(float(np.max(np.abs(as_numpy(arm.agent.robot.get_qvel())[0, arm.indices]))) for arm in arms)
            valid = (phase == "settle" and all(row["in_regions"]) and not any(row["grasping"])
                     and all(opened) and arm_speed <= ARM_SPEED
                     and max(linear) <= OBJECT_SPEED and max(angular) <= OBJECT_ANGULAR_SPEED)
            if not valid:
                stable_count, stability_reference = 0, None
            elif stability_reference is None:
                stable_count, stability_reference = 1, [pose.copy() for pose in poses]
            else:
                drift = any(np.linalg.norm(p[:3, 3]-ref[:3, 3]) > WINDOW_POSITION or
                            _rotation_distance(ref[:3, :3], p[:3, :3]) > WINDOW_ROTATION
                            for p, ref in zip(poses, stability_reference))
                stable_count = 1 if drift else stable_count+1
                if drift:
                    stability_reference = [pose.copy() for pose in poses]
            stable_max = max(stable_max, stable_count)
            row.update(final_stable_steps=stable_count, max_arm_speed_rad_s=arm_speed)
            if checker is not None:
                checker.refresh()
                collisions = checker.measured_check() if checker.attached else checker.check(*(arm.arm_qpos() for arm in arms))
                row["collisions"] = collisions
                if collisions:
                    result["collision_events"].append({"step": row["step"], "phase": phase,
                                                        "collisions": collisions})
                    fail("measured_collision", collisions=collisions)
                    break
            result["max_state_error"] = max(result["max_state_error"],
                _state_error(raw, archive["states"], result["executed_actions"]))
            frame()
            if held_failed:
                break
            if row["truncated"]:
                fail("environment_timeout")
                break
        result.update(_facts(raw))
        result.update(grasp_confirmed=grasp_max >= policy.grasp_stable_steps,
            grasp_stable_steps_observed=grasp_max, lifted=lifted,
            max_lift_m=max_lift, release_stable_steps_observed=release_max,
            retreat_distance_m=retreat_distance, final_stable_steps_observed=stable_count,
            max_final_stable_steps_observed=stable_max)
        conditions = {
            "incomplete_action_replay": result["executed_actions"] == len(actions),
            "task_not_successful": result["task_success"],
            "grasp_not_confirmed": result["grasp_confirmed"],
            "insufficient_actual_lift": all(lifted),
            "objects_not_released": all(result["released"]) and all(opened) and release_max >= policy.release_stable_steps,
            "insufficient_retreat": release_tcp is not None and min(retreat_distance) >= requirements["minimum_retreat_m"],
            "insufficient_final_stability": stable_count >= requirements["final_stable_steps"],
            "episode_budget_exceeded": result["steps"] <= policy.last_completion_step,
        }
        for reason, passed in conditions.items():
            if not passed:
                fail(reason)
        result["passed"] = not result["failures"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        fail("replay_error", error=result["error"])
        if env is not None:
            try:
                from .pick_place import _facts
                result.update(_facts(env.unwrapped))
            except Exception:
                pass
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        result["wall_seconds"] = time.monotonic()-started
        (output / "replay.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    for count in (6, 7):
        order = [{"key": f"arm-{i}", "size": count+1,
                  "arm_joint_names": [f"j{x}" for x in range(count)],
                  "state_joint_names": [f"j{x}" for x in range(count+2)]} for i in range(2)]
        validate_action_order(order, 2*(count+1))
    print("Native 14/16-D replay action contracts passed")
