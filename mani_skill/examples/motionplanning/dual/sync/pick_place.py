"""Shared Panda/Piper pick-place example, standard recording and action replay.

Generate: python -m mani_skill.examples.motionplanning.dual.sync.pick_place \
    --robot piper --seed 401 --output-dir outputs/piper-401
Replay:  python -m mani_skill.examples.motionplanning.dual.sync.pick_place \
    --replay outputs/piper-401/trajectory.h5 --output-dir outputs/piper-replay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.dual_pick_place import PROFILES
from mani_skill.utils.wrappers.record import RecordEpisode
from .arm_planner import ArmMotionPlanner
from .dual_arm_execution import (plan_dual_pose, execute_dual_reach,
    close_dual_and_confirm, execute_dual_held_move, release_dual_and_confirm,
    verify_dual_placement)
from .dual_payload_collision import DualPayloadCollisionChecker
from .runtime_policy import RuntimePolicy
from .trajectory_execution import as_numpy, bindings, native_arm_action


TASK_ID = "DualArmPickPlace-v1"
POLICY = RuntimePolicy(policy_id="dual-pick-place-v1", max_episode_steps=400,
    last_completion_step=399, min_lift_m=.08, require_retreat=True,
    require_final_stability=True)


def scalar(value: Any) -> Any:
    array = as_numpy(value)
    return array.item() if array.size == 1 else array.tolist()


def native_action(row: Any, order: list[dict]) -> dict[str, np.ndarray]:
    """Split a standard flat H5 action using recorded native controller sizes."""
    row = np.asarray(row, dtype=np.float32)
    if row.shape != (sum(item["size"] for item in order),) or not np.isfinite(row).all():
        raise ValueError("Action shape differs from its recorded robot controllers")
    result, offset = {}, 0
    for item in order:
        key, size = item["key"], item["size"]
        if key in result or size < 1:
            raise ValueError("Action keys must be unique and controller sizes positive")
        result[key] = row[offset:offset+size].copy()
        offset += size
    return result


class ObservedEpisode:
    """Record task events around actual paired actions, without changing them."""

    def __init__(self, env: Any):
        self.env, self.unwrapped = env, env.unwrapped
        self.phase = "preparation"
        self.rows = []

    def step(self, action: Any):
        before = int(scalar(self.unwrapped.elapsed_steps))
        output = self.env.step(action)
        raw = self.unwrapped
        if int(scalar(raw.elapsed_steps)) != before+1:
            raise RuntimeError("A paired command must advance exactly one control tick")
        self.rows.append({"step": before+1, "phase": self.phase,
            "grasping": [bool(scalar(a.is_grasping(o))) for a, o
                         in zip(raw.agent.agents, raw.targets)],
            "object_positions_m": [as_numpy(o.pose.p)[0].tolist() for o in raw.targets],
            "in_regions": [bool(scalar(raw.region_contains(i))) for i in range(2)],
            "success": bool(scalar(output[-1]["success"])),
            "terminated": bool(scalar(output[-3])),
            "truncated": bool(scalar(output[-2]))})
        # Match the existing sync observer: a successful task may become true
        # during release. Keep the recorder's original success/terminal facts,
        # while allowing the caller to finish release, retreat and settling.
        obs, reward, terminated, truncated, info = output
        if (bool(scalar(terminated)) and bool(scalar(info["success"]))
                and not bool(scalar(info.get("fail", False))) and not bool(scalar(truncated))):
            return obs, reward, False, truncated, info
        return output


def _facts(raw: Any) -> dict:
    evaluation = raw.evaluate()
    return {"task_success": bool(scalar(evaluation["success"])),
            "in_regions": [bool(scalar(raw.region_contains(i))) for i in range(2)],
            "released": [not bool(scalar(a.is_grasping(o))) for a, o
                         in zip(raw.agent.agents, raw.targets)],
            "steps": int(scalar(raw.elapsed_steps)),
            "task_horizon": 400}


def solve(env: Any, *, reach_only: bool = False) -> dict:
    """Use one task and the existing shared executor for both robot profiles."""
    raw, started = env.unwrapped, time.monotonic()
    capture = ObservedEpisode(env)
    operation = "execution"
    arms = bindings(raw, tuple(raw.agent.agents_dict))
    report = {"schema": "multi-embodiment-sync-v1", "task": TASK_ID,
        "robot": raw.robot_uids[0], "control_freq": 1/raw.control_timestep,
        "action_order": [{"key": arm.key, "size": len(arm.names)+1,
            "arm_joint_names": list(arm.names),
            **({"gripper_coordinate_unit": "rad",
                "gripper_open_action": float(arm.agent.gripper_open_action),
                "gripper_closed_action": float(arm.agent.gripper_closed_action)}
               if getattr(arm.agent, "gripper_coordinate_unit", "m") == "rad" else {}),
            "state_joint_names": [j.name for j in arm.agent.robot.get_active_joints()]}
            for arm in arms], "policy": POLICY.to_dict(), "stages": [],
        "planning": [], "execution": {"status": "running"}}

    def finish(name: str, result: dict):
        report["stages"].append({"name": name,
            **{key: result[key] for key in ("status", "reason", "environment_steps",
                "settling_steps", "stable_steps_observed", "wall_seconds") if key in result},
            **({"first_collision": result["validation"].get("first_collision")}
                if "validation" in result else {}),
            **({"last_event": result["trace"][-1]} if result.get("trace")
                and result["status"] != "passed" else {})})
        print(f'{name}: {result["status"]} at step {scalar(raw.elapsed_steps)}', flush=True)
        if result["status"] != "passed":
            report["execution"].update(phase=capture.phase,
                reason=result.get("reason"),
                operation="planning_or_geometry_rejection" if result["status"] == "rejected" else "execution")
            raise RuntimeError(f'{name}: {result.get("reason", result["status"])}')

    try:
        for _ in range(8):
            capture.step({arm.key: native_arm_action(arm, np.r_[arm.arm_qpos(), 1.])
                          for arm in arms})
        report["stages"].append({"name": "preparation", "status": "passed", "environment_steps": 8})
        solvers = [ArmMotionPlanner(capture, a.agent) for a in arms]
        checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=raw.targets)
        checker.set_measured_attachment_prediction(True)

        def move(name: str, poses: tuple, *, held: bool = False, contacts: bool = False):
            nonlocal operation
            capture.phase = name
            operation = "planning"
            tick, call = int(scalar(raw.elapsed_steps)), time.monotonic()
            try:
                schedule = plan_dual_pose(*solvers, poses,
                    gripper_commands=(-1., -1.) if held else (1., 1.), policy=POLICY)
            finally:
                report["planning"].append({"name": name, "seconds": time.monotonic()-call,
                    "endpoints": [getattr(s, "last_endpoint_audit", {}) for s in solvers],
                    "methods": [getattr(s, "last_plan_method", {}) for s in solvers]})
            if int(scalar(raw.elapsed_steps)) != tick:
                raise RuntimeError("Planning changed physical state")
            report["planning"][-1]["source_durations_s"] = [t.duration_s for t in schedule.trajectories]
            report["planning"][-1]["scheduled_steps"] = schedule.num_steps
            remaining = POLICY.last_completion_step-tick-schedule.num_steps
            if remaining < 0:
                operation = "budget"
                raise RuntimeError(f"{name}: motion exceeds the task's remaining control ticks")
            settling = min(POLICY.max_settle_steps, remaining)
            operation = "execution"
            if held:
                result = execute_dual_held_move(capture, schedule, checker, raw.targets,
                    min_height_gain_m=(.08, .08) if name == "lift" else (None, None),
                    settle_max_steps=settling, policy=POLICY)
            else:
                result = execute_dual_reach(capture, schedule, checker,
                    allow_target_contacts=(contacts, contacts), settle_max_steps=settling,
                    policy=POLICY)
            finish(name, result)

        p = raw.profile
        quaternion = p.grasp_quaternion
        positions = [as_numpy(o.pose.p)[0]+[0, 0, p.grasp_height_offset] for o in raw.targets]
        move("pregrasp", tuple(sapien.Pose(xyz+[0, 0, p.approach_height], quaternion)
                               for xyz in positions))
        if reach_only:
            report["execution"]["status"] = "passed"
            report["scope"] = "synchronous_reach_only"
            return report
        move("descend", tuple(sapien.Pose(xyz, quaternion) for xyz in positions), contacts=True)
        capture.phase = "grasp"
        finish("grasp", close_dual_and_confirm(capture, checker, raw.targets, policy=POLICY))
        checker.attach()
        move("lift", tuple(sapien.Pose(as_numpy(a.tcp.pose.p)[0]+[0, 0, p.carry_height],
                                      as_numpy(a.tcp.pose.q)[0]) for a in raw.agent.agents), held=True)
        carry = tuple(sapien.Pose([*xy, p.carry_height+self_h], quaternion)
                      for xy, self_h in zip(raw.destination_xy, [raw.cube_half_size]*2))
        move("carry", carry, held=True)
        # Preserve measured grasp geometry when calculating the low release TCP.
        release_poses = []
        for agent, obj, xy in zip(raw.agent.agents, raw.targets, raw.destination_xy):
            delta = as_numpy(agent.tcp.pose.p)[0]-as_numpy(obj.pose.p)[0]
            release_poses.append(sapien.Pose(np.r_[xy, raw.cube_half_size+.004]+delta,
                                            as_numpy(agent.tcp.pose.q)[0]))
        move("lower", tuple(release_poses), held=True)
        checker.detach()
        capture.phase = "release"
        finish("release", release_dual_and_confirm(capture, checker, raw.targets, policy=POLICY))
        move("retreat", tuple(sapien.Pose(as_numpy(a.tcp.pose.p)[0]+[0, 0, .10],
                                         as_numpy(a.tcp.pose.q)[0]) for a in raw.agent.agents))
        capture.phase = "settle"
        finish("settle", verify_dual_placement(capture, checker, raw.targets,
            official_predicate_keys=("left_in_region", "right_in_region"),
            region_predicates=tuple((lambda _, i=i: {"inside": bool(scalar(raw.region_contains(i)))})
                                    for i in range(2)), policy=POLICY))
        report["execution"]["status"] = "passed"
        report["scope"] = "grasp_lift_carry_release_retreat_and_settle"
    except Exception as exc:
        report["execution"] = {"phase": capture.phase, "operation": operation,
            **report["execution"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        report.update(_facts(raw))
        report["events"] = capture.rows
        report["wall_seconds"] = time.monotonic()-started
    return report


def replay(path: Path, output: Path) -> dict:
    """Delegate to the same native-action replay for either robot profile."""
    from .pick_place_replay import replay as replay_actions
    return replay_actions(path, output)


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(TASK_ID,), default=TASK_ID,
                        help="Compatible shared task; other TIG tasks keep their existing entrypoints")
    parser.add_argument("--robot", choices=tuple(PROFILES), default="piper")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reach-only", action="store_true")
    options = parser.parse_args(args)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    if options.replay:
        result = replay(options.replay, options.output_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return
    env = None
    result = {"execution": {"status": "not_started"}, "recording": {"status": "not_started"}}
    try:
        env = gym.make(options.task, robot_uids=(options.robot, options.robot),
            obs_mode="rgb", control_mode="pd_joint_pos", sim_backend="physx_cpu",
            render_mode="rgb_array", num_envs=1, shader_dir="default")
        env = RecordEpisode(env, str(options.output_dir), trajectory_name="trajectory",
            video_fps=20, save_on_reset=False, source_type="motionplanning",
            source_desc="Shared named-arm MPLib planner and synchronous executor")
        env.reset(seed=options.seed)
        result = solve(env, reach_only=options.reach_only)
        result["recording"] = {"status": "running"}
        env._json_data["embodiment_sync"] = result
        env.flush_trajectory()
        env.flush_video()
        result["recording"] = {"status": "passed"}
    except Exception as exc:
        detail = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if result["execution"]["status"] == "not_started":
            result["execution"] = detail
        else:
            # Preserve the environment's task result if recording failed.
            result["recording"] = detail
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["recording"] = {"status": "failed",
                    "error": f"{type(exc).__name__}: {exc}"}
    metadata_path = options.output_dir/"trajectory.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata["embodiment_sync"] = result
        metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    (options.output_dir/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("events", "planning", "stages")}, indent=2))
    if result["execution"]["status"] != "passed" or result["recording"]["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
