"""The Imitator evaluation loop (imitator.eval.rollout) runs any policy that implements imitator.eval.policy.

Runs without simulator, dataset or lerobot: a stub vector env with the dual-arm Panda action layout, a stub processor
and examples/policy_template's random policy. Checks the call protocol (prepare_for_eval once per episode, an action
chunk every pred_horizon steps, clear_cache after every batch) and the episode bookkeeping.
"""
import enum
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gymnasium = pytest.importorskip("gymnasium")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)


@pytest.fixture()
def rollout(monkeypatch):
    try:
        import lerobot  # noqa: F401
    except ImportError:  # the loop only needs InputMode from the dataset package: stub it
        class InputMode(enum.Enum):
            VIDEO_ONLY = "video_only"
            LANGUAGE_ONLY = "language_only"
            VIDEO_AND_LANGUAGE = "video_and_language"
        stub = types.ModuleType("imitator.dataset.lerobot_paired_dataset")
        stub.InputMode = InputMode
        monkeypatch.setitem(sys.modules, "imitator.dataset.lerobot_paired_dataset", stub)
    pytest.importorskip("mani_skill")
    from imitator.eval import rollout as R
    return R


class StubEnvs:
    """2 envs, dual-arm Panda action dict (8 + 8), episodes of T steps; success_once = 1 for env 0"""

    def __init__(self, n=2, T=20):
        self.num_envs, self.T, self.t = n, T, 0
        self.single_observation_space = gymnasium.spaces.Dict({"state": gymnasium.spaces.Box(-1, 1, (18,))})
        self.action_space = {"panda_wristcam-0": gymnasium.spaces.Box(-1, 1, (n, 8)),
                             "panda_wristcam-1": gymnasium.spaces.Box(-1, 1, (n, 8))}
        self.actions = []

    def _obs(self):
        return {"state": torch.zeros(self.num_envs, 18), "rgb": torch.zeros(self.num_envs, 8, 8, 3)}

    def reset(self):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        assert set(action) == {"panda_wristcam-0", "panda_wristcam-1"}
        assert action["panda_wristcam-0"].shape == (self.num_envs, 8)
        self.actions.append(action)
        self.t += 1
        trunc = torch.full((self.num_envs,), self.t >= self.T)
        info = {"success": torch.tensor([True, False])}
        if trunc.any():
            info["final_info"] = {"episode": {"success_once": torch.tensor([1.0, 0.0]),
                                              "success_at_end": torch.tensor([1.0, 0.0])}}
        return self._obs(), torch.zeros(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), trunc, info


class StubProcessor:
    def get_task_description(self, env_id):
        return f"task description of {env_id}"


def test_rollout_calls_policy_protocol(rollout):
    sys.path.insert(0, os.path.join(REPO, "examples"))
    from policy_template.policy import RandomPolicy

    calls = {"prepare": 0, "act": 0, "clear": 0}

    class Spy(RandomPolicy):
        def prepare_for_eval(self, human_video, robot_obs, human_desc):
            calls["prepare"] += 1
            assert human_video is None and human_desc == ["task description of L0_Dummy-v1"] * 2
            super().prepare_for_eval(human_video, robot_obs, human_desc)

        def get_action(self, obs):
            calls["act"] += 1
            return super().get_action(obs)

        def clear_cache(self):
            calls["clear"] += 1

    envs = StubEnvs(n=2, T=20)
    pol = Spy("cpu", pred_horizon=8)
    kw = dict(env_id="L0_Dummy-v1", delta_control=True, pred_horizon=8, temporal_agg=False, max_timesteps=20,
              device="cpu", sim_backend="physx_cpu")
    m = rollout.evaluate_with_task_encoder(4, pol, envs, kw, StubProcessor(), rollout.InputMode.LANGUAGE_ONLY,
                                           progress_bar=False)
    assert calls["prepare"] == 2 and calls["clear"] == 2           # 4 episodes = 2 batches of 2 envs
    assert calls["act"] == 2 * 3                                    # steps 0, 8, 16 of each 20-step episode
    assert len(envs.actions) == 40
    np.testing.assert_allclose(m["success_once"], [[1.0, 0.0], [1.0, 0.0]])


def test_level_helpers(rollout):
    assert rollout.extract_level("L3_TwoRobotStirSpoon-v1") == "L3"
    assert rollout.extract_base_env_name("L3_TwoRobotStirSpoon-v1") == "TwoRobotStirSpoonL3-v1"
    assert rollout.extract_base_env_name("L1_TwoRobotStirSpoon-v1") == "TwoRobotStirSpoon-v1"
    assert rollout.extract_level("TwoRobotStirSpoon-v1") == "L0"
