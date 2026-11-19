"""Minimal policy showing the evaluation interface (random actions in the normalised [-1, 1] action space).

Run it end to end (needs the IG-10K human/sim data under --human-root/--sim-root):

    python -m imitator.eval.run --policy examples.policy_template.policy:make_policy \
        --eval-config imitator/eval/lists/debug_eval.txt --output-dir results/random --num-episodes 2

Replace :meth:`RandomPolicy.get_action` with your model; use :meth:`prepare_for_eval` to encode the human video
and/or the task description once per episode.
"""
import torch

from imitator.eval.policy import DEFAULT_MODEL_CONFIG, ImitatorPolicy


class RandomPolicy(ImitatorPolicy):
    def __init__(self, device, action_dim: int = 16, pred_horizon: int = 16, seed: int = 0):
        self.device = device
        self.action_dim = action_dim
        self.model_config = dict(DEFAULT_MODEL_CONFIG, name="random", pred_horizon=pred_horizon)
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.batch = 1

    def prepare_for_eval(self, human_video, robot_obs, human_desc):
        states = robot_obs.get("states")
        self.batch = int(states.shape[0]) if states is not None else 1

    def get_action(self, obs):
        h = self.model_config["pred_horizon"]
        a = torch.rand((self.batch, h, self.action_dim), generator=self.g) * 2 - 1
        return a.to(self.device)


def make_policy(checkpoint=None, device="cpu", **kwargs):
    p = RandomPolicy(device, **kwargs)
    return p, p.model_config
