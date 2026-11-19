"""Native policy evaluation records physical facts without expert data gates."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces


def test_drop_regrasp_and_final_success_are_not_vetoed_by_physical_facts():
    """Optional facts must not change actions, rewards, done or official success.

    Run the real public-observation adapter and seeded native evaluator twice,
    with and without grasp/region facts. The fixture checks the evaluation
    boundary; it does not simulate physical grasp recovery.
    """
    from examples.baselines.act import eval_act_single as evaluation

    script = [(True, False, False), (False, False, False), (True, False, False),
              (True, True, True), (False, True, True)]

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = 0

        def get_action(self, rgb, state):
            assert rgb.shape == (1, 1, 3, 8, 8)
            assert state.shape == (1, 1, 18)
            self.queries += 1
            return (torch.arange(64, dtype=torch.float32).reshape(1, 4, 16) / 100.
                    + (self.queries - 1) / 10.)

    class Environment:
        def __init__(self, report_physical_facts):
            self.report_physical_facts = report_physical_facts
            self.steps = 0
            self.actions, self.native_results, self.physical_info = [], [], []
            self.returned_info, self.original_info, self.reset_seeds = [], [], []
            keys = ("panda_wristcam-0", "panda_wristcam-1")
            self.unwrapped = SimpleNamespace(control_freq=20,
                action_space=spaces.Dict({key: spaces.Box(-1., 1., shape=(8,), dtype=np.float32)
                                          for key in keys}),
                agent=SimpleNamespace(agents=[None, None], agents_dict=dict.fromkeys(keys)))

        def observation(self):
            return {"sensor_data": {"zed2i": {"rgb": np.full((1, 8, 8, 3), self.steps, np.uint8)}},
                    "agent": {key: {"qpos": torch.full((1, 9), self.steps / 10.)}
                              for key in self.unwrapped.agent.agents_dict}}

        def reset(self, seed):
            self.reset_seeds.append(seed)
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
            self.returned_info.append(info)
            self.original_info.append(copy.deepcopy(info))
            return (self.observation(), np.array([reward]), np.array([terminated]),
                    np.array([truncated]), info)

    normalizer = SimpleNamespace(resolve_idx=lambda env_id: 0,
                                 normalize_state=lambda value, idx: value,
                                 denormalize_action=lambda value, idx: value)
    args = SimpleNamespace(fps=20, execute_steps=4, max_episode_steps=7, native_contract={})
    train = SimpleNamespace(cameras=["zed2i"], state_type="qpos", single_arm=False,
                            action_dim=16, image_size=[8, 8], obs_horizon=1, pred_horizon=4)
    outputs = []
    for report in (True, False):
        env, policy = Environment(report), Policy()
        result = evaluation.evaluate_episode(policy, env, normalizer, args, train,
                                             401, "fixture", torch.device("cpu"))
        assert policy.queries == 2
        assert env.reset_seeds == [401]
        assert len(env.actions) == result["steps"] == len(script)
        assert result["official_success_once"] and result["official_success_at_end"]
        assert result["first_success_step"] == 4
        assert result["terminated"] and result["truncated"]
        assert result["failure_stage"] is None and not result["timeout"]
        # Event sampling must leave every native env.step info object intact.
        for actual, original in zip(env.returned_info, env.original_info):
            assert actual.keys() == original.keys()
            for key in original:
                np.testing.assert_array_equal(actual[key], original[key])
        outputs.append((env, result))

    observed, unobserved = (row[0] for row in outputs)
    assert [bool(row["is_pill_a_grasped"][0]) for row in observed.physical_info] == [
        True, False, True, True, False]
    assert observed.native_results == unobserved.native_results
    for actual, expected in zip(observed.actions, unobserved.actions):
        for key in ("panda_wristcam-0", "panda_wristcam-1"):
            assert actual[key].shape == (8,)
            np.testing.assert_array_equal(actual[key], expected[key])
    assert outputs[0][1]["physical"]["grasp_placement"]["arms"]["left"][
        "grasp_to_ungrasped_outside_region_steps"] == [2]
    for key in ("steps", "official_success_once", "official_success_at_end", "first_success_step",
                "terminated", "truncated", "timeout", "action_clipped_steps"):
        assert outputs[0][1][key] == outputs[1][1][key]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
