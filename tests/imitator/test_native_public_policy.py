"""Public native policy contracts and a recoverable physical-event rollout."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.baselines.act.native_video_policy import NativeVideoPolicy
from imitator.eval import native


def test_public_policy_demo_cache_and_single_environment():
    class Agent:
        def eval(self):
            pass

        def set_demonstration(self, video, demo_id):
            self.value = video.mean()
            return {"human_demo_id": demo_id}

        def get_action(self, rgb, state):
            return torch.zeros(1, 2, 14) + self.value

        def clear_cache(self):
            self.value = None

    a, b = torch.zeros(2, 3, 8, 8), torch.ones(2, 3, 8, 8)
    p = NativeVideoPolicy(Agent(), {"pill": {"human_video": a, "human_demo_id": "clip-a"}})
    obs = {"states": torch.zeros(1, 1, 16), "view_1": torch.zeros(1, 1, 3, 8, 8)}
    p.prepare_for_eval(a, obs, None); aa = p.get_action(obs)
    assert p.conditioning == {"human_demo_id": "clip-a"}
    p.prepare_for_eval(b, obs, None); bb = p.get_action(obs)
    p.prepare_for_eval(a, obs, None)
    assert torch.equal(p.get_action(obs), aa) and not torch.equal(aa, bb)
    with pytest.raises(ValueError, match="exactly one"):
        p.prepare_for_eval(a.unsqueeze(0).repeat(2, 1, 1, 1, 1), obs, None)
    p.clear_cache(); assert p.conditioning is None


def recorded_contract():
    return {"env_id": "pill", "robot_uids": ["piper", "piper"], "control_mode": "pd_joint_pos",
            "control_freq": 20, "sim_freq": 100, "model_fingerprints": ["a", "a"],
            "mounts": [[0, 0, 0], [0, 1, 0]],
            "action_order": [{"key": "left", "size": 7}, {"key": "right", "size": 7}]}


def test_same_task_datasets_keep_distinct_recording_paths():
    first = {"dataset_idx": 0, "repo_id": "group-a/pill", "env_id": "pill", "native": recorded_contract()}
    second = dict(first, dataset_idx=1, repo_id="group-b/pill")
    assert native.record_identity(first) != native.record_identity(second)
    assert native.record_identity(first) == native.record_identity(deepcopy(first))
    assert "/" not in native.record_identity(first)


@pytest.mark.parametrize("field", ["robot_uids", "control_freq", "sim_freq", "mounts", "action_order",
                                  "model_fingerprints", "control_mode"])
def test_physical_controller_contract_drift_rejected(field):
    contract = recorded_contract()
    changed = deepcopy(contract); changed[field] = "different"
    with pytest.raises(ValueError, match="differs"):
        native.validate_native_contract(contract, changed)


def test_legacy_physics_is_not_silently_claimed_equivalent():
    contract = recorded_contract(); del contract["model_fingerprints"]
    with pytest.raises(ValueError, match="legacy weights"):
        native.validate_native_contract(contract, recorded_contract())


def test_drop_event_does_not_end_policy_and_recovery_counts(monkeypatch):
    """An env may report transient grasp loss and succeed later in its budget."""
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.action_space = {key: SimpleNamespace(low=np.full(7, -1), high=np.full(7, 1))
                                 for key in ("left", "right")}
            self.actions = []

        def reset(self, seed):
            self.index = 0
            return {"public": True}, {}

        def step(self, action):
            self.actions.append(action)
            self.index += 1
            return {"public": True}, 0, False, self.index == 4, {"success": self.index == 4,
                                                              "is_pill_a_grasped": self.index != 2}

    class Policy:
        def eval(self):
            pass

        def prepare_for_eval(self, human_video, robot_obs, human_desc):
            assert set(robot_obs) == {"states", "view_1"}

        def get_action(self, obs):
            assert set(obs) == {"states", "view_1"}
            return torch.full((1, 2, 14), .25)

        def clear_cache(self):
            pass

    monkeypatch.setattr(native, "live_observation", lambda *a, **kw: {
        "rgb": torch.zeros(3, 8, 8), "state": torch.zeros(16)})
    normalizer = SimpleNamespace(denormalize_action=lambda action, idx: action)
    config = {"native_contract": recorded_contract(), "pred_horizon": 2, "obs_horizon": 1,
              "image_size": [8, 8], "state_dim": 16, "action_dim": 14, "cameras": ["zed2i"]}
    env = Env()
    row = native.run_episode(Policy(), env, normalizer, dataset_idx=0, config=config,
                             seed=1, execute_steps=2, max_steps=4,
                             demonstration={"human_video": torch.zeros(2, 3, 8, 8),
                                            "human_demo_id": "clip", "metadata": {}}, device=torch.device("cpu"))
    assert row["success_at_end"] and row["control_steps"] == 4 and row["first_success_step"] == 4
    assert not row["events"][1]["left_grasped"]
    assert len(env.actions) == 4
    assert all(np.array_equal(a["left"], np.full(7, .25, np.float32)) for a in env.actions)


def test_normal_public_cli_default_is_unchanged():
    from imitator.eval.run import parse_args

    args = parse_args(["--policy", "module:factory", "--eval-config", "eval.txt", "--output-dir", "out"])
    assert not args.native and args.max_episode_steps == 500
    args = parse_args(["--native", "--policy", "module:factory", "--output-dir", "out"])
    assert args.native and args.max_episode_steps is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
