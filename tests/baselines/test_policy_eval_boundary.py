"""The native vectorized policy evaluator does not enforce expert data gates."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_drop_regrasp_and_final_success_are_not_vetoed_by_physical_facts(monkeypatch):
    """Optional native facts must not change actions, rewards, done or success.

    This is a scripted evaluator-boundary test, not a physical recovery rollout.
    The original evaluator has no event sampler, so compare native info with and
    without optional grasp/region facts instead of adding an observer to it.
    """
    # The official script imports train_act_single as a sibling module. Match
    # that CLI import context without changing the production evaluation path.
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "examples/baselines/act"))
    from examples.baselines.act import eval_act_single as evaluation

    script = [(True, False, False), (False, False, False), (True, False, False),
              (True, True, True), (False, True, True)]

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = 0

        def get_action(self, rgb, state):
            self.queries += 1
            return (torch.arange(64, dtype=torch.float32).reshape(1, 4, 16) / 100.
                    + (self.queries - 1) / 10.)

    class Environment:
        num_envs = 1

        def __init__(self, report_physical_facts):
            self.report_physical_facts = report_physical_facts
            self.steps = 0
            self.actions = []
            self.native_results = []
            self.physical_info = []

        def observation(self):
            return {"rgb": np.full((1, 3, 8, 8), self.steps / 10., dtype=np.float32),
                    "state": np.full((1, 18), self.steps / 10., dtype=np.float32)}

        def reset(self):
            self.steps = 0
            return self.observation(), {}

        def step(self, action):
            self.actions.append(copy.deepcopy(action))
            grasped, in_region, success = script[self.steps]
            self.steps += 1
            terminated, truncated = success, self.steps == len(script)
            reward = float(sum(value.sum() for value in action.values())) + self.steps
            self.native_results.append((reward, terminated, truncated, success))
            info = {"success": np.array([success])}
            if self.report_physical_facts:
                info.update(is_pill_a_grasped=np.array([grasped]),
                            is_pill_b_grasped=np.array([grasped]),
                            pill_a_in_region=np.array([in_region]),
                            pill_b_in_region=np.array([in_region]))
                self.physical_info.append(info)
            if truncated:
                info["final_info"] = {"episode": {
                    "success_once": torch.tensor([True]), "success_at_end": torch.tensor([success]),
                    "episode_len": torch.tensor([self.steps])}}
            return (self.observation(), np.array([reward]), np.array([terminated]),
                    np.array([truncated]), info)

    normalizer = SimpleNamespace(normalize_rgb=lambda value: value,
                                 normalize_state=lambda value: value,
                                 denormalize_action=lambda value: value)
    outputs = []
    for report in (True, False):
        env, policy = Environment(report), Policy()
        result = evaluation.evaluate_episode(
            1, policy, env, normalizer, SimpleNamespace(sim_backend="physx_cpu"),
            torch.device("cpu"), SimpleNamespace(pred_horizon=4))
        assert policy.queries == 2
        assert len(env.actions) == len(script)
        np.testing.assert_array_equal(result["success_once"], [True])
        np.testing.assert_array_equal(result["success_at_end"], [True])
        np.testing.assert_array_equal(result["episode_len"], [len(script)])
        outputs.append((env, result))

    observed, unobserved = (row[0] for row in outputs)
    assert [bool(row["is_pill_a_grasped"][0]) for row in observed.physical_info] == [
        True, False, True, True, False]
    assert observed.native_results == unobserved.native_results
    for actual, expected in zip(observed.actions, unobserved.actions):
        for key in ("panda_wristcam-0", "panda_wristcam-1"):
            np.testing.assert_array_equal(actual[key], expected[key])
    for key in outputs[0][1]:
        np.testing.assert_array_equal(outputs[0][1][key], outputs[1][1][key])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
