"""Observe a released dual task without adding a stability or velocity gate.

The same paired-action, joint-order, actual collision, truncation and partial
failure contracts as ``dual_arm_execution._Phase`` remain active. Target/TCP
poses are world transforms; arm commands are seven absolute joint radians and
an open-gripper scalar. Region callbacks retain their task-defined geometry.
An already satisfied task is observed at zero steps, independently of any
optional retreat or final quality window. No state or attachment is assigned.
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Callable

import numpy as np

from .dual_arm_execution import _Phase, _bool, _pair, _regions, _runtime_policy
from .grasp_execution import _actions_valid, _budget, _error
from .runtime_io import json_value
from .runtime_policy import LEGACY_STRICT, RuntimePolicy
from .trajectory_execution import as_numpy


def observe_dual_task_outcome(env: Any, checker: Any, targets: tuple[Any, Any], *,
                              official_predicate_keys: tuple[str, str] = ("apple_in_basket", "banana_in_basket"),
                              region_predicates: tuple[Callable[[Any], Any], Callable[[Any], Any]],
                              max_steps: int = 0,
                              policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    """Return on the first actual released/placed official success, or a failure.

    ``max_steps`` is the caller's remaining control-step budget and may be zero.
    The caller keeps zero-step results outside action stages/ticks. The returned
    final observation is a separate copy, including the measured control step;
    a later measurement failure never relabels an older observation as current.
    Native termination/truncation and actual collision rejection take priority
    over a task-success observation at the same tick.
    """
    policy = _runtime_policy(policy)
    targets = _pair(targets, "targets")
    keys = _pair(official_predicate_keys, "official_predicate_keys")
    predicates = _pair(region_predicates, "region_predicates")
    if any(target is None for target in targets) or any(not callable(predicate) for predicate in predicates):
        raise ValueError("Both actual targets and physical region callbacks are required")
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("Explicit official predicate keys are required")
    if getattr(checker, "attached", False):
        raise ValueError("Task outcome observation requires detached collision predictions")
    budget = _budget(max_steps, "max_steps", zero=True)
    phase = _Phase(env, checker, "observe_task_outcome", targets=targets, policy=policy)
    phase.report.update(max_steps=budget, official_predicate_keys=list(keys),
                        quality_window_evaluated=False, velocity_gate_evaluated=False,
                        task_outcome_observation_evaluated=False,
                        final_observation=None)

    def action(row: dict[str, Any]) -> dict[str, np.ndarray]:
        return {arm.key: np.r_[row["measurements"][arm.key]["qpos_rad"], 1.]
                for arm in phase.arms}

    def observe(row: dict[str, Any]) -> bool:
        regions = _regions(env, targets, keys, predicates)
        evaluation = env.unwrapped.evaluate()
        if "success" not in evaluation:
            raise ValueError("Missing official task success predicate")
        success = _bool(evaluation["success"])
        row.update(fingers_open=[phase.opened(arm, row) for arm in phase.arms],
                   official_success=success, official_evaluation=json_value(evaluation),
                   observed_elapsed_steps=as_numpy(env.unwrapped.elapsed_steps).tolist())
        for obj, region in zip(row["objects"], regions):
            obj.update(region)
        valid = bool(success and all(row["fingers_open"]) and all(
            obj["official_region"] and obj["physical_region"]["inside"] and not obj["is_grasping"]
            for obj in row["objects"]))
        row["task_outcome_satisfied"] = valid
        phase.report.update(task_outcome_observation_evaluated=True,
                            final_observation=copy.deepcopy(row))
        return valid

    checker.set_target_contacts_allowed((False, False))
    try:
        row = phase.report["initial_observation"]
        _actions_valid(env, phase.arms, action(row))
        if not phase.preflight():
            return phase.finish()
        try:
            satisfied = observe(row)
        except Exception as exc:
            _error(phase.report, "task_outcome_measurement_failed", exc)
            return phase.finish()
        if satisfied:
            phase.report["status"] = "passed"
            return phase.finish()
        for _ in range(budget):
            row = phase.tick(action(row), "observe_task_outcome")
            if row is None:
                break
            try:
                satisfied = observe(row)
            except Exception as exc:
                row["task_outcome_observation_error"] = f"{type(exc).__name__}: {exc}"
                if phase.report["status"] == "running":
                    _error(phase.report, "task_outcome_measurement_failed", exc)
                break
            if phase.report["status"] != "running":
                break
            if satisfied:
                phase.report["status"] = "passed"
                break
        if phase.report["status"] == "running":
            phase.report.update(status="task_outcome_timeout", reason="official_released_placements_not_observed")
        return phase.finish()
    finally:
        checker.set_target_contacts_allowed((False, False))


def self_test() -> dict[str, Any]:
    """Exercise the real _Phase controller/observation contract using CPU spies."""
    from types import SimpleNamespace as NS
    from gymnasium import spaces

    names = tuple(f"joint{i}" for i in range(7))

    class Pose:
        def __init__(self): self.matrix = np.eye(4)
        def to_transformation_matrix(self): return self.matrix[None].copy()

    class Actor:
        def __init__(self): self.pose = Pose()
        def get_linear_velocity(self): return np.array([[4., 0., 0.]])
        def get_angular_velocity(self): return np.array([[0., 0., 8.]])

    class Robot:
        def __init__(self): self.q = np.array([[0.] * 7 + [.04, .04]]); self.v = np.full((1, 9), 3.)
        def get_qpos(self): return self.q.copy()
        def get_qvel(self): return self.v.copy()
        def get_qlimits(self): return np.array([[[-3., 3.]] * 7 + [[0., .04]] * 2])
        def get_active_joints(self):
            return [NS(get_name=lambda name=name: name) for name in (*names, "f1", "f2")]

    class Env:
        def __init__(self, ready_after=0, mode="normal"):
            self.unwrapped = self; self.num_envs = 1; self.control_timestep = .05
            self.elapsed_steps = np.array([0]); self.actions = []; self.mode = mode
            self.ready_after = ready_after; self.targets = (Actor(), Actor()); self.grasp = [False, False]
            self.agent = NS(agents=[], agents_dict={})
            for i, key in enumerate(("left", "right")):
                robot = Robot()
                controller = NS(joints=robot.get_active_joints()[:7],
                    config=NS(normalize_action=False, use_delta=False, joint_names=names))
                agent = NS(robot=robot, control_mode="pd_joint_pos",
                    controller=NS(controllers={"arm": controller}), tcp=NS(pose=Pose()))
                agent.is_grasping = lambda actor, i=i: np.array([self.grasp[i]])
                self.agent.agents.append(agent); self.agent.agents_dict[key] = agent
            self.left_agent, self.right_agent = self.agent.agents
            box = spaces.Box(np.array([-3.] * 7 + [-1.], np.float32), np.array([3.] * 7 + [1.], np.float32))
            self.agent.single_action_space = spaces.Dict({"left": box, "right": box})
            if mode == "retained_grasp":
                self.grasp[1] = True; self.right_agent.robot.q[0, -2:] = .024

        def step(self, action):
            assert list(action) == ["left", "right"]
            self.actions.append({k: np.asarray(v).copy() for k, v in action.items()})
            self.elapsed_steps += 1
            if self.mode == "partial_error": raise RuntimeError("actual-shaped step failed after advancement")
            for i, key in enumerate(("left", "right")):
                assert action[key][-1] == 1
                robot = self.agent.agents[i].robot
                robot.q[0, :7] = action[key][:7]
                # Measured drift proves the next command holds CURRENT q.
                robot.q[0, 0] += .01
                robot.q[0, -2:] = .024 if self.mode == "retained_grasp" and i == 1 else .04
            if self.mode == "measurement_error": self.left_agent.robot.v[0, 0] = np.nan
            return None, 0., np.array([self.mode == "terminated"]), np.array([self.mode == "truncated"]), {}

        def evaluate(self):
            return {"success": np.array([len(self.actions) >= self.ready_after]),
                    "left_inside": np.array([True]), "right_inside": np.array([self.mode != "official_region_false"])}

    class Checker:
        def __init__(self, env, mode="normal"):
            self.env = env; self.targets = env.targets; self.attached = False
            self.allowed = (True, True); self.mode = mode; self.calls = 0; self.refreshes = 0
        def set_target_contacts_allowed(self, value): self.allowed = value
        def refresh(self): self.refreshes += 1
        def check(self, left, right):
            self.calls += 1; assert self.allowed == (False, False)
            if self.mode == "initial_collision" or self.mode == "later_collision" and self.env.actions:
                return [{"actual_spy_collision": True}]
            if self.mode == "query_error": raise RuntimeError("actual-shaped checker query failed")
            return []

    def run(env, *, budget=4, checker=None, region=True):
        checker = Checker(env) if checker is None else checker
        def predicate(target):
            if env.mode == "region_error" and env.actions: raise RuntimeError("actual-shaped region measurement failed")
            return {"inside": region, "source": "measured fixture region"}
        result = observe_dual_task_outcome(env, checker, env.targets,
            official_predicate_keys=("left_inside", "right_inside"),
            region_predicates=(predicate, predicate), max_steps=budget)
        assert checker.allowed == (False, False)
        assert result["quality_window_evaluated"] is False and result["velocity_gate_evaluated"] is False
        return result

    checks = []
    env = Env(); checker = Checker(env); result = run(env, budget=0, checker=checker)
    assert result["status"] == "passed" and result["environment_steps"] == 0 and not result["trace"] and not env.actions
    assert checker.calls == checker.refreshes == 1
    assert result["final_observation"] is not result["initial_observation"]
    assert result["final_observation"]["official_success"] and result["final_observation"]["observed_elapsed_steps"] == [0]
    result["initial_observation"]["objects"][0]["is_grasping"] = True
    assert not result["final_observation"]["objects"][0]["is_grasping"]
    checks.append("zero-step success has actual preflight and independent final observation")
    env = Env(3); result = run(env)
    assert result["status"] == "passed" and result["environment_steps"] == len(env.actions) == 3
    assert len(result["trace"]) == 3 and result["final_observation"]["observed_elapsed_steps"] == [3]
    assert all(np.isclose(action["left"][0], .01 * i) for i, action in enumerate(env.actions))
    assert result["trace"][-1]["max_arm_velocity_rad_s"] == 3.
    assert result["trace"][-1]["objects"][0]["object_speed_m_s"] == 4.
    checks.append("first valid sample ends the phase; current-q paired open holds; no velocity/window gate")
    env = Env(1); result = run(env, budget=0)
    assert result["status"] == "task_outcome_timeout" and not env.actions
    checks.append("zero remaining budget cannot create a successful observation")
    for mode, region in (("retained_grasp", True), ("official_region_false", True), ("normal", False)):
        env = Env(mode=mode); result = run(env, budget=2, region=region)
        assert result["status"] == "task_outcome_timeout" and len(env.actions) == 2
    checks.append("actual retained grasp, official per-target region or physical region false cannot pass")
    env = Env(); result = run(env, checker=Checker(env, "initial_collision"))
    assert result["status"] == "rejected" and result["reason"] == "initial_collision" and not env.actions
    env = Env(); result = run(env, checker=Checker(env, "query_error"))
    assert result["status"] == "rejected" and result["reason"] == "collision_preflight_failed" and not env.actions
    env = Env(1); result = run(env, checker=Checker(env, "later_collision"))
    assert result["status"] == "aborted" and result["reason"] == "measured_collision" and len(env.actions) == 1
    assert result["final_observation"]["task_outcome_satisfied"]
    checks.append("actual initial/later collision and checker errors retain executor rejection priority")
    for mode in ("terminated", "truncated"):
        env = Env(1, mode); result = run(env)
        assert result["status"] == mode and result["reason"] == "environment_done" and len(env.actions) == 1
        assert result["final_observation"]["official_success"]
    checks.append("native truncation/termination cannot be overwritten by same-tick success")
    env = Env(1, "region_error"); result = run(env)
    assert result["status"] == "aborted" and result["reason"] == "task_outcome_measurement_failed"
    assert result["environment_steps"] == 1 and "task_outcome_observation_error" in result["trace"][-1]
    assert result["final_observation"]["observed_elapsed_steps"] == [0]
    env = Env(1, "measurement_error"); result = run(env)
    assert result["status"] == "aborted" and result["reason"] == "step_or_measurement_failed"
    assert result["environment_steps"] == 1 and result["trace"][-1]["elapsed_step_delta_on_error"] == [1]
    env = Env(1, "partial_error"); result = run(env)
    assert result["status"] == "aborted" and result["reason"] == "step_or_measurement_failed"
    assert result["environment_steps"] == 0 and env.elapsed_steps.tolist() == [1]
    assert result["trace"][-1]["elapsed_step_delta_on_error"] == [1]
    checks.append("region/state/partial-step measurement errors preserve prior observation and actual error trace")
    for budget in (-1, True, 1.5):
        env = Env()
        try: run(env, budget=budget)
        except ValueError: pass
        else: raise AssertionError("Invalid step budget accepted")
        assert not env.actions
    env = Env(); checker = Checker(env); checker.attached = True
    try: run(env, checker=checker)
    except ValueError: pass
    else: raise AssertionError("Attached checker accepted")
    assert not env.actions
    checks.append("invalid budgets and attached prediction reject before any action")
    return {"status": "passed", "check_count": len(checks), "checks": checks,
            "physical_environment_steps": 0, "native_environment_created": False, "gpu_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
