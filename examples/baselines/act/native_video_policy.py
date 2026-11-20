"""Native video ACT adapter for ``imitator.eval.run --policy module:factory``.

The public interface consumes normalized public robot states/RGB and a human
clip, and returns normalized native chunks (1, Q, A). The evaluator owns the
recorded normalizer, controller split and environment. This first adapter is
explicitly single-environment; it never observes simulator task truth.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from imitator.eval.policy import DEFAULT_MODEL_CONFIG, ImitatorPolicy


def _video_hash(video: torch.Tensor) -> str:
    value = video.detach().float().cpu().contiguous()
    return hashlib.sha256(str(tuple(value.shape)).encode() + value.numpy().tobytes()).hexdigest()


class NativeVideoPolicy(ImitatorPolicy):
    """Adapt the existing agent without changing its model or weights."""

    def __init__(self, agent: Any, demonstrations: dict[str, dict[str, Any]]) -> None:
        self.agent = agent
        self.identities = {_video_hash(demo["human_video"]): demo["human_demo_id"]
                           for demo in demonstrations.values()}
        self.conditioning: dict[str, Any] | None = None

    def prepare_for_eval(self, human_video: torch.Tensor | None,
                         robot_obs: dict[str, Any], human_desc: list[str] | None) -> None:
        if human_video is None:
            raise ValueError("Native video ACT requires a human RGB demonstration")
        if human_video.ndim == 4:
            human_video = human_video.unsqueeze(0)
        if human_video.ndim != 5 or human_video.shape[0] != 1:
            raise ValueError("Native video ACT currently supports exactly one environment")
        digest = _video_hash(human_video[0])
        identity = self.identities.get(digest, "tensor-sha256:" + digest)
        self.conditioning = self.agent.set_demonstration(human_video, identity)

    def get_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        states, rgb = obs["states"], obs["view_1"]
        if states.shape[0] != 1 or rgb.shape[0] != 1:
            raise ValueError("Native video ACT currently supports exactly one environment")
        return self.agent.get_action(rgb, states)

    def clear_cache(self) -> None:
        self.agent.clear_cache()
        self.conditioning = None

    def eval(self) -> NativeVideoPolicy:
        self.agent.eval()
        return self

    def train(self, mode: bool = True) -> NativeVideoPolicy:
        self.agent.train(mode)
        return self


def load_native_checkpoint(checkpoint: str, device: torch.device | str = "cpu", *,
                           hf_cache_dir: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Strict reload only; physical compatibility is checked by the evaluator."""
    from examples.baselines.act.native_video_model import NativeVideoACTAgent
    from examples.baselines.act.train_act_imitator import TrainingArgs

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("policy_type") != "native_video_act":
        raise ValueError("Expected a native_video_act checkpoint")
    model_args = TrainingArgs(**payload["model_args"])
    if hf_cache_dir is not None:
        model_args.hf_cache_dir = hf_cache_dir
    mode = payload.get("conditioning_mode", "video")
    native_args = payload["args"]
    for key in ("state_dim", "action_dim", "state_type", "pred_horizon", "obs_horizon",
                "single_arm", "include_depth"):
        if key in native_args and native_args[key] != getattr(model_args, key):
            raise ValueError(f"Checkpoint native/model {key} settings differ")
    for key in ("cameras", "image_size"):
        if list(native_args[key]) != list(getattr(model_args, key)):
            raise ValueError(f"Checkpoint native/model {key} settings differ")
    if payload.get("training_args", {}).get("conditioning_mode", mode) != mode:
        raise ValueError("Checkpoint conditioning settings differ")
    agent = NativeVideoACTAgent(model_args, torch.device(device), conditioning_mode=mode)
    agent.load_state_dict(payload["agent_state_dict"], strict=True)
    agent.eval()
    return agent, payload


def make_policy(checkpoint: str | None = None, device: torch.device | str = "cpu", *,
                human_manifest: str | None = None, human_role: str = "evaluation",
                human_demo_id: str | None = None, hf_cache_dir: str | None = None,
                video_backend: str | None = None) -> tuple[NativeVideoPolicy, dict[str, Any]]:
    """Build the public policy and its saved native evaluation configuration."""
    from examples.baselines.act.native_video_data import load_human_demo

    if checkpoint is None:
        raise ValueError("Native video ACT requires --checkpoint")
    agent, payload = load_native_checkpoint(checkpoint, device, hf_cache_dir=hf_cache_dir)
    native_args = payload["args"]
    contract = payload["data_contract"]
    records = contract.get("datasets")
    if not records:
        native = contract.get("native") or {}
        if not native.get("env_id"):
            raise ValueError("Checkpoint must identify its recorded native task")
        records = [{"dataset_idx": 0, "env_id": native["env_id"],
                    "repo_id": payload["dataset_stats"][0]["repo_id"], "native": native}]
    manifest = human_manifest or payload.get("human_manifest")
    if not manifest:
        raise ValueError("No saved human manifest; supply human_manifest in --policy-kwargs")
    backend = video_backend or native_args.get("video_backend", "torchcodec")
    demonstrations = {}
    for record in records:
        task = record["env_id"]
        selection = {} if human_demo_id is not None else {"task_id": task}
        demonstrations[task] = load_human_demo(
            manifest, role=human_role, demo_id=human_demo_id,
            image_size=tuple(native_args["image_size"]), video_backend=backend, **selection)
    policy = NativeVideoPolicy(agent, demonstrations)
    config = dict(DEFAULT_MODEL_CONFIG,
                  name="native_video_act", cameras=native_args["cameras"],
                  include_depth=native_args["include_depth"],
                  image_size=tuple(native_args["image_size"]), state_type=native_args["state_type"],
                  single_arm=native_args["single_arm"], pred_horizon=native_args["pred_horizon"],
                  obs_horizon=native_args["obs_horizon"],
                  native_evaluation={"checkpoint": str(Path(checkpoint).resolve()),
                                     "records": records, "dataset_stats": payload["dataset_stats"],
                                     "normalization_method": native_args.get("normalization_method", "bounds_q99"),
                                     "normalization_min_range": native_args.get("normalization_min_range", 0.0),
                                     "state_dim": native_args["state_dim"],
                                     "action_dim": native_args["action_dim"],
                                     "demonstrations": demonstrations})
    policy.model_config = config
    return policy, config


def _sanity_check() -> None:
    """Check the real public adapter with a tiny deterministic tensor policy."""
    class TensorAgent:
        def set_demonstration(self, video: torch.Tensor, identity: str) -> dict[str, Any]:
            self.z = video.mean()
            return {"human_demo_id": identity}

        def get_action(self, rgb: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 2, 14) + self.z

        def clear_cache(self) -> None:
            self.z = None

        def eval(self) -> None:
            pass

        def train(self, mode: bool) -> None:
            pass

    a, b = torch.zeros(2, 3, 8, 8), torch.ones(2, 3, 8, 8)
    p = NativeVideoPolicy(TensorAgent(), {"a": {"human_video": a, "human_demo_id": "clip-a"}})
    obs = {"view_1": torch.zeros(1, 1, 3, 8, 8), "states": torch.zeros(1, 1, 16)}
    p.prepare_for_eval(a, obs, None); first = p.get_action(obs)
    assert p.conditioning["human_demo_id"] == "clip-a"
    p.prepare_for_eval(b, obs, None); second = p.get_action(obs)
    p.prepare_for_eval(a, obs, None)
    assert first.shape == second.shape == (1, 2, 14)
    assert torch.equal(first, p.get_action(obs)) and not torch.equal(first, second)
    p.clear_cache(); assert p.conditioning is None
    print({"public_policy_action_shape": list(first.shape), "A_B_A_cache": True})


if __name__ == "__main__":
    _sanity_check()
