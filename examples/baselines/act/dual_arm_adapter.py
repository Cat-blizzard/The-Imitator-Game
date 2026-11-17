"""Shared observation/action contract for the first dual-Panda ACT bridge.

RGB is one zed2i image, uint8 HWC -> float32 CHW in [0,1]. No augmentation,
extra privileged fields, or per-camera concatenation is used. State is the
physical left Panda's nine qpos followed by the right Panda's nine qpos.
Actions retain the recorder's left/right blocks (seven joint targets + one
gripper command each); physical two-finger positions are never action labels.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

CAMERA_KEY = "observation.images.zed2i"
STATE_KEY = "observation.qpos_gripper_states"
ACTION_KEY = "action.qpos_gripper_actions"
FPS, HORIZON, STATE_DIM, ACTION_DIM = 20, 16, 18, 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def prepare_rgb(rgb: Any) -> torch.Tensor:
    """Return float32 (3,224,224), accepting uint8 HWC or loader float CHW."""
    value = torch.as_tensor(rgb)
    if value.shape == (1, 224, 224, 3) or value.shape == (1, 3, 224, 224):
        value = value[0]
    if value.shape == (224, 224, 3):
        value = value.permute(2, 0, 1)
    if tuple(value.shape) != (3, 224, 224):
        raise ValueError(f"Expected one zed2i 224 RGB image, got {tuple(value.shape)}")
    if value.dtype == torch.uint8:
        value = value.float().div(255)
    elif not value.is_floating_point():
        raise ValueError("RGB must be uint8 pixels or floating-point [0,1]")
    else:
        value = value.float()
    if not bool(torch.isfinite(value).all()) or value.min() < 0 or value.max() > 1:
        raise ValueError("RGB must be finite and in [0,1]")
    return value.contiguous()


def prepare_observation(rgb: Any, qpos: Any, normalizer: Any) -> dict[str, torch.Tensor]:
    """Same preprocessing for LeRobot examples and live sensor observations."""
    state = torch.as_tensor(qpos, dtype=torch.float32).reshape(-1)
    if tuple(state.shape) != (STATE_DIM,) or not bool(torch.isfinite(state).all()):
        raise ValueError("Expected exactly 18 finite physical qpos values")
    return {"rgb": prepare_rgb(rgb),
            "state": normalizer.normalize_state(state, 0, method="bounds_q99")}


def live_observation(raw: Any, observation: Mapping[str, Any], normalizer: Any) -> dict[str, torch.Tensor]:
    agents = raw.agent.agents
    if len(agents) != 2 or (hasattr(raw, "left_agent") and
            (agents[0] is not raw.left_agent or agents[1] is not raw.right_agent)):
        raise ValueError("Live robot sequence must be the physical left/right Panda pair")
    positions = [numpy_array(agent.robot.get_qpos()).reshape(-1) for agent in agents]
    if any(value.shape != (9,) for value in positions):
        raise ValueError("Each Panda must contribute seven joint and two finger positions")
    return prepare_observation(observation["sensor_data"]["zed2i"]["rgb"],
                               np.concatenate(positions), normalizer)


def split_action(action: Any, keys: list[str]) -> dict[str, np.ndarray]:
    """Keep recorder 8+8 order, allowing continuous policy gripper commands.

    Expert replay's split helper requires binary grippers; policy predictions
    must not be silently thresholded to satisfy that different contract.
    """
    value = np.asarray(numpy_array(action), dtype=np.float32)
    if len(keys) != 2 or len(set(keys)) != 2 or value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError("Expected two ordered robot keys and one finite 16-action")
    return {key: value[8 * side:8 * (side + 1)].copy() for side, key in enumerate(keys)}


def train_statistics(states: torch.Tensor, actions: torch.Tensor) -> dict[str, dict[str, list[float]]]:
    """Compute only on the selected train rows; the T+1 terminal is excluded."""
    result: dict[str, dict[str, list[float]]] = {}
    for key, value, dim in ((STATE_KEY, states, STATE_DIM), (ACTION_KEY, actions, ACTION_DIM)):
        value = value.float()
        if value.ndim != 2 or value.shape[1] != dim or value.shape[0] < 1 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid training statistics input for {key}")
        result[key] = {"min": value.min(0).values.tolist(), "max": value.max(0).values.tolist(),
                       "mean": value.mean(0).tolist(), "std": value.std(0, unbiased=False).tolist(),
                       "q01": torch.quantile(value, .01, dim=0).tolist(),
                       "q99": torch.quantile(value, .99, dim=0).tolist()}
    return result


def make_normalizer(stats: dict[str, Any], repo_id: str) -> Any:
    from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer
    normalizer = ActionNormalizer()
    normalizer.add_dataset_stats(0, repo_id, stats, STATE_KEY, ACTION_KEY, single_arm=False)
    return normalizer


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over real action elements only, never repeated tail labels."""
    if prediction.shape != target.shape or prediction.ndim != 3 or is_pad.shape != prediction.shape[:2]:
        raise ValueError("Expected prediction/target (B,H,D) and padding mask (B,H)")
    valid = ~is_pad.bool()
    if not valid.any():
        raise ValueError("An ACT batch must contain at least one real action")
    return ((prediction - target).abs() * valid.unsqueeze(-1)).sum() / (valid.sum() * prediction.shape[-1])


def self_test() -> None:
    states = torch.zeros(3, 18)
    actions = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    normalizer = make_normalizer(train_statistics(states, actions), "synthetic-only")
    rgb = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    a = prepare_observation(rgb, states[0], normalizer)
    b = prepare_observation(prepare_rgb(rgb), states[0], normalizer)
    assert torch.equal(a["rgb"], b["rgb"]) and torch.equal(a["state"], b["state"])
    split = split_action(actions[0], ["left", "right"])
    assert np.array_equal(split["left"], actions[0, :8].numpy())
    assert np.array_equal(split["right"], actions[0, 8:].numpy())
    prediction = torch.zeros(1, 16, 16, requires_grad=True)
    target = torch.ones_like(prediction)
    is_pad = torch.ones(1, 16, dtype=torch.bool); is_pad[:, 0] = False
    loss = masked_l1(prediction, target, is_pad); loss.backward()
    assert loss.item() == 1 and torch.count_nonzero(prediction.grad[:, 1:]) == 0
    assert torch.count_nonzero(prediction.grad[:, 0]) == 16
    print({"status": "passed", "rgb": list(a["rgb"].shape), "state": list(a["state"].shape),
           "action_blocks": [list(v.shape) for v in split.values()], "tail_mask_gradient": "passed"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.parse_args()
    self_test()
