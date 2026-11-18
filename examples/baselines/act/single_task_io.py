"""Observation, action and loss conventions shared by single-task ACT entries."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import torch


def prepare_rgb(rgb: Any, image_size: Sequence[int] = (224, 224)) -> torch.Tensor:
    """One RGB frame: uint8 HWC or float CHW -> float32 CHW in [0, 1]."""
    value = torch.as_tensor(rgb)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[-1] == 3:
        value = value.permute(2, 0, 1)
    expected = (3, *image_size)
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected RGB with three channels, got {tuple(value.shape)}")
    if tuple(value.shape) != expected:
        # Match the simulation dataloader's OpenCV bilinear resize before /255.
        pixels = value.permute(1, 2, 0).cpu().numpy()
        pixels = cv2.resize(pixels, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)
        value = torch.from_numpy(pixels).permute(2, 0, 1)
    if value.dtype == torch.uint8:
        value = value.float() / 255.0
    elif value.is_floating_point():
        value = value.float()
    else:
        raise ValueError("RGB must be uint8 pixels or floating-point [0,1]")
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("RGB must be finite and in [0,1]")
    return value.contiguous()


def live_observation(raw: Any, observation: Mapping[str, Any], camera: str,
                     normalizer: Any, dataset_idx: int = 0,
                     image_size: Sequence[int] = (224, 224),
                     action_order: Sequence[Mapping[str, Any]] | None = None) -> dict[str, torch.Tensor]:
    """Selected camera and native qpos in recorded controller/joint order."""
    agents = raw.agent.agents
    if len(agents) != 2:
        raise ValueError("Single-task dual-arm evaluation requires two agents")
    if hasattr(raw, "left_agent") and (agents[0] is not raw.left_agent or agents[1] is not raw.right_agent):
        raise ValueError("Agent order must be physical left then physical right")
    keys = list(raw.agent.agents_dict)
    # Proprioception is part of the public RGB observation. Do not feed hidden
    # simulator state or object poses to the policy's state encoder.
    positions = [torch.as_tensor(observation["agent"][key]["qpos"]).reshape(-1) for key in keys]
    if action_order is not None:
        if keys != [item["key"] for item in action_order]:
            raise ValueError("Live robot controller keys differ from the training recording")
        for agent, item, position in zip(agents, action_order, positions):
            names = [joint.name for joint in agent.robot.get_active_joints()]
            if names != item["state_joint_names"] or len(position) != len(names):
                raise ValueError("Live native qpos order differs from training")
            arm_controller = agent.controller.controllers.get("arm") if hasattr(agent, "controller") else None
            if arm_controller is not None and list(arm_controller.config.joint_names) != item.get("arm_joint_names"):
                raise ValueError("Live native arm action joint order differs from training")
    state = torch.cat(positions).float()
    return {
        "rgb": prepare_rgb(observation["sensor_data"][camera]["rgb"], image_size),
        "state": normalizer.normalize_state(state, dataset_idx),
    }


def split_action(action: Any, keys: Sequence[str],
                 sizes: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Split the native flattened row into recorded per-controller action widths."""
    value = torch.as_tensor(action).detach().cpu().numpy().astype(np.float32)
    sizes = list(sizes) if sizes is not None else [8, 8]
    if (len(keys) != 2 or len(set(keys)) != 2 or len(sizes) != 2
            or any(type(size) is not int or size < 1 for size in sizes)
            or value.shape != (sum(sizes),) or not np.isfinite(value).all()):
        raise ValueError("Expected two ordered robot keys and one finite native action row")
    boundaries = np.cumsum([0]+sizes)
    return {key: value[boundaries[side]:boundaries[side+1]].copy() for side, key in enumerate(keys)}


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """Average over real action elements, excluding the repeated terminal tail."""
    if prediction.shape != target.shape or prediction.ndim != 3 or is_pad.shape != prediction.shape[:2]:
        raise ValueError("Expected prediction/target (B,H,D) and padding mask (B,H)")
    valid = ~is_pad.bool()
    if not valid.any():
        raise ValueError("An ACT batch must contain at least one real action")
    return ((prediction - target).abs() * valid.unsqueeze(-1)).sum() / (valid.sum() * prediction.shape[-1])


if __name__ == "__main__":
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    assert prepare_rgb(rgb).shape == (3, 224, 224)
    prediction = torch.zeros(1, 2, 16, requires_grad=True)
    loss = masked_l1(prediction, torch.ones_like(prediction), torch.tensor([[False, True]]))
    loss.backward()
    assert loss.item() == 1 and torch.count_nonzero(prediction.grad[:, 1:]) == 0
    assert list(split_action(np.arange(16), ["left", "right"])["right"]) == list(range(8, 16))
    print("single_task_io: RGB, left/right actions and padding gradients passed")
