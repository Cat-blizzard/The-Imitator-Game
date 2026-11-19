"""Evaluate human-video-conditioned ACT with a recorded native robot contract.

Human RGB frames condition the policy; public robot RGB/qpos observations drive
the shared native rollout. This entry never calls a planner or restores states.
Use ``--human-demo-id`` to explicitly select a replacement demonstration for a
conditioning diagnostic. The selected video identity and latent are reported.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import torch

from examples.baselines.act import eval_act_single
from examples.baselines.act.act.single_task_eval_utils import extract_base_env_name
from examples.baselines.act.train_act_single import TrainingArgs as NativeTrainingArgs


def load_checkpoint(ckpt_path: str, args_override: Any, device: torch.device):
    """Strictly reload the video policy and its saved native observation settings."""
    from examples.baselines.act.native_video_model import NativeVideoACTAgent
    from examples.baselines.act.train_act_imitator import TrainingArgs as VideoTrainingArgs

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") != "native_video_act":
        raise ValueError("Expected a native_video_act checkpoint")
    native_args = NativeTrainingArgs(**checkpoint["args"])
    model_args = VideoTrainingArgs(**checkpoint["model_args"])
    for key in ("state_dim", "action_dim", "state_type", "pred_horizon", "obs_horizon",
                "single_arm", "include_depth"):
        if getattr(native_args, key) != getattr(model_args, key):
            raise ValueError(f"Checkpoint native/model {key} settings differ")
    for key in ("cameras", "image_size"):
        if list(getattr(native_args, key)) != list(getattr(model_args, key)):
            raise ValueError(f"Checkpoint native/model {key} settings differ")
    for key in ("state_type", "pred_horizon", "obs_horizon"):
        value = getattr(args_override, key, None)
        if value is not None and value != getattr(native_args, key):
            raise ValueError(f"Evaluation {key} must match the checkpoint")
    agent = NativeVideoACTAgent(model_args, device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    agent.eval()
    return agent, native_args


def _task_id(env_id: str) -> str:
    # TIG registers L3 as a separate environment name for the same task mapping.
    return re.sub(r"L3(?=-v\d+$)", "", extract_base_env_name(env_id))


class HumanVideoConditioning:
    """Load a selected video once and set its effective condition each episode."""

    def __init__(self, manifest: str | Path | None = None, *,
                 role: str = "evaluation", demo_id: str | None = None):
        self.manifest_override = manifest
        self.role = role
        self.demo_id = demo_id
        self.demo: dict[str, Any] | None = None
        self.task_id: str | None = None

    def __call__(self, agent: Any, checkpoint: dict[str, Any], config: dict[str, Any],
                 args: Any, seed: int, env_id: str, device: torch.device) -> dict[str, Any]:
        from examples.baselines.act.native_video_data import load_human_demo

        if self.demo is None:
            source = self.manifest_override or checkpoint.get("human_manifest")
            if not source:
                raise ValueError("No saved human manifest; supply --human-manifest")
            model_args = checkpoint["model_args"]
            self.demo = load_human_demo(
                source, role=self.role, demo_id=self.demo_id,
                image_size=tuple(model_args["image_size"]),
                video_backend=checkpoint["args"].get(
                    "video_backend", model_args.get("video_backend", "torchcodec")),
            )
            self.task_id = self.demo["metadata"]["task_id"]
        task_matches = _task_id(self.task_id) == _task_id(env_id)
        if not task_matches and self.demo_id is None:
            raise ValueError("Human demonstration task differs from the evaluation task; "
                             "an intentional replacement requires --human-demo-id")
        # Content/sampling identity comes from the loader, rather than repo ID.
        # Setting the condition does not receive the environment or alter rollout.
        effective_id = self.demo["human_demo_id"]
        conditioning = agent.set_demonstration(self.demo["human_video"], effective_id)
        return {
            "policy_type": "native_video_act", "human_demo_id": effective_id,
            "human_demonstration": {**self.demo["metadata"],
                                    "selection_role": self.role,
                                    "explicit_demo_override": self.demo_id is not None,
                                    "task_matches_environment": task_matches},
            "conditioning": conditioning,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--human-manifest", help="Override the checkpoint's saved human-video manifest")
    parser.add_argument("--human-demo-id", help="Select a demonstration, including explicit replacement diagnostics")
    parser.add_argument("--human-role", default="evaluation", help="Manifest role; defaults to evaluation")
    selected, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    prepare_episode = HumanVideoConditioning(selected.human_manifest,
                                             role=selected.human_role,
                                             demo_id=selected.human_demo_id)
    return eval_act_single.main(remaining, checkpoint_loader=load_checkpoint,
                                prepare_episode=prepare_episode,
                                report_prefix="act_native_video")


if __name__ == "__main__":
    main()
