"""Observe actual task completion when no optional settling window is requested."""
from __future__ import annotations
import copy
from typing import Any, Callable
import numpy as np
from .dual_arm_execution import _Phase, _bool, _pair, _regions, _runtime_policy
from .measurements import _actions_valid, _budget, _error
from .runtime_policy import DEFAULT_POLICY, RuntimePolicy
from .trajectory_execution import as_numpy

def observe_dual_task_outcome(env: Any, checker: Any, targets: tuple[Any, Any], *,
                              official_predicate_keys: tuple[str, str] = ("apple_in_basket", "banana_in_basket"),
                              region_predicates: tuple[Callable[[Any], Any], Callable[[Any], Any]],
                              max_steps: int = 0,
                              policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
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
                   official_success=success, official_evaluation={key: as_numpy(value).tolist() for key, value in evaluation.items()},
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
