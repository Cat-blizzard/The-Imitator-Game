"""Negative test: overlap private robot models and require mutual collision.

Only the collision-model bases move. The SAPIEN robots, their joints and the
environment step counter must remain unchanged. Example:
    python -m mani_skill.examples.embodiments.check_sync_collision --robot piper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from mani_skill.envs.tasks.tabletop.dual_pick_place import PROFILES
from mani_skill.examples.motionplanning.dual.sync.arm_planner import ArmMotionPlanner
from mani_skill.examples.motionplanning.dual.sync.combined_collision import NativeDualCollisionChecker
from mani_skill.examples.motionplanning.dual.sync.trajectory_execution import bindings, snapshot


def overlap_private_models(checker: Any, left_qpos: Any, right_qpos: Any) -> dict:
    """Restore both model bases even if querying raises or the test fails."""
    bases = tuple(np.array(model.get_base_pose(), dtype=float, copy=True)
                  for model in checker.models)
    if len(bases) != 2:
        raise ValueError("Exactly two independent full robot collision models are required")
    original = checker.check(left_qpos, right_qpos)
    try:
        for model in checker.models:
            model.set_base_pose(bases[0].copy())
        collisions = checker.check(left_qpos, right_qpos)
        mutual = [value for value in collisions
                  if {value.get("object_name1"), value.get("object_name2")} == {"left", "right"}]
        if not mutual:
            raise AssertionError("Overlapping private robots did not produce a mutual collision")
    finally:
        for model, base in zip(checker.models, bases):
            model.set_base_pose(base)
    restored = checker.check(left_qpos, right_qpos)
    if any(not np.array_equal(model.get_base_pose(), base)
           for model, base in zip(checker.models, bases)) or restored != original:
        raise AssertionError("Private collision-world bases or collision results were not restored")
    return {"passed": True, "overlap_mutual_collisions": mutual,
            "original_collision_count": len(original), "private_model_bases_restored": True,
            "normal_scene_objects_included": False}


def run(robot: str, seed: int) -> dict:
    env = gym.make("DualArmPickPlace-v1", robot_uids=(robot, robot),
                   obs_mode="none", control_mode="pd_joint_pos",
                   sim_backend="physx_cpu", num_envs=1)
    try:
        env.reset(seed=seed)
        raw = env.unwrapped
        arms = bindings(raw, tuple(raw.agent.agents_dict))
        before = snapshot(raw, arms)
        solvers = tuple(ArmMotionPlanner(env, arm.agent) for arm in arms)
        checker = NativeDualCollisionChecker.from_solvers(*solvers)
        try:
            result = overlap_private_models(checker, *(arm.arm_qpos() for arm in arms))
        finally:
            if not all(np.array_equal(a, b) for a, b in zip(before, snapshot(raw, arms))):
                raise AssertionError("A private collision query changed the physical environment")
        return {"schema": "native-dual-collision-negative-v1", "robot": robot, "seed": seed,
                "environment_steps": 0, "physical_state_unchanged": True, **result}
    finally:
        env.close()


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=tuple(PROFILES), default="piper")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(args)
    result = run(options.robot, options.seed)
    rendered = json.dumps(result, indent=2) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with options.output.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
