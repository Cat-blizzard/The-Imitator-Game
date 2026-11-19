"""The interface a policy implements to be evaluated on The Imitator Game.

The rollout loop (:func:`imitator.eval.rollout.evaluate_with_task_encoder`) calls exactly these methods, so any model
(diffusion policy, VLA, in-context imitator, scripted controller...) can be evaluated without changes to the benchmark.

Shapes (B = number of parallel envs):
- ``prepare_for_eval(human_video, robot_obs, human_desc)`` is called once at the start of every episode.
  ``human_video``: tensor of the human demonstration (``None`` in ``language_only`` mode);
  ``robot_obs``: ``{"states": (B, ...), "view_1": (B, ...)}`` first robot observation;
  ``human_desc``: list of B task descriptions (``None`` in ``video_only`` mode).
- ``get_action(obs)`` returns an action chunk ``(B, H, A)`` (or ``(H, A)`` for B = 1); A = 16 for the dual-arm
  Panda setup (8 per arm: 7 joints + gripper), normalised unless the control mode is a delta mode.
- ``clear_cache()`` is called after every batch of episodes.
- ``eval()`` / ``train()`` toggle inference mode (no-ops for non-torch policies).

A policy factory used by ``python -m imitator.eval.run --policy module:factory`` returns ``(policy, model_config)``;
see :data:`DEFAULT_MODEL_CONFIG` for the keys the evaluator reads.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "name": "policy",            # recorded in the results
    "cameras": ["zed2i"],        # human-video cameras the policy consumes
    "include_depth": False,      # rgbd observations instead of rgb
    "num_video_frames": 10,      # frames sampled from the human video
    "image_size": (224, 224),
    "state_type": "qpos",        # qpos | eepos | mixpos (selects the normalisation statistics)
    "single_arm": False,
    "pred_horizon": 16,          # action-chunk length H
    "obs_horizon": 1,
}


class ImitatorPolicy:
    """Base class; subclass and implement :meth:`get_action` (and :meth:`prepare_for_eval` to use the task)."""

    model_config: Dict[str, Any] = DEFAULT_MODEL_CONFIG

    def prepare_for_eval(self, human_video: Optional[torch.Tensor], robot_obs: Dict[str, Any],
                         human_desc: Optional[list]) -> None:
        pass

    def get_action(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def clear_cache(self) -> None:
        pass

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self


def resolve_factory(spec: str):
    """``"package.module:callable"`` -> the callable."""
    import importlib

    if ":" not in spec:
        raise ValueError(f"--policy must look like 'package.module:factory', got {spec!r}")
    mod, attr = spec.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


def build_policy(spec: str, checkpoint: Optional[str], device: torch.device, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    out = resolve_factory(spec)(checkpoint=checkpoint, device=device, **kwargs)
    policy, cfg = out if isinstance(out, tuple) else (out, getattr(out, "model_config", {}))
    merged = dict(DEFAULT_MODEL_CONFIG)
    merged.update(cfg or {})
    return policy, merged
