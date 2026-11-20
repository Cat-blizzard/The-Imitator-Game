"""Numeric training rows and checkpoint normalization for native robot policies.

These helpers consume public robot state/action features. They never turn a
simulator's full state (which can contain object truth) into proprioception.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .normalizer import ActionNormalizer


def select_train_rows(dataset: Any, keys: Sequence[str]) -> dict[str, torch.Tensor]:
    """Read numeric columns from an already episode-selected training dataset.

    Each returned feature has shape (N, D). No videos or dataset-wide aggregate
    statistics are read. The caller chooses the training episode split first.
    """
    result = {}
    for key in keys:
        values = torch.stack([torch.as_tensor(value) for value in dataset.hf_dataset[key]]).float()
        if values.ndim != 2 or not len(values) or not torch.isfinite(values).all():
            raise ValueError(f"Invalid selected training rows for {key}")
        result[key] = values
    return result


def selected_row_statistics(dataset: Any, keys: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    """Compute per-coordinate statistics using only the chosen training rows."""
    result = {}
    for key, values in select_train_rows(dataset, keys).items():
        result[key] = {
            "min": values.min(0).values.tolist(), "max": values.max(0).values.tolist(),
            "mean": values.mean(0).tolist(), "std": values.std(0, unbiased=False).tolist(),
            "q01": torch.quantile(values, .01, dim=0).tolist(),
            "q99": torch.quantile(values, .99, dim=0).tolist(),
        }
    return result


def _base_env_name(env_id: str) -> str:
    """Match the benchmark's L0-L3 environment naming convention."""
    if env_id.startswith(("L0_", "L1_", "L2_", "L3_")):
        level, base = env_id.split("_", 1)
        if level == "L3":
            if "-v" in base:
                name, version = base.rsplit("-v", 1)
                return f"{name}L3-v{version}"
            return f"{base}L3"
        return base
    return env_id


class NativeSingleTaskNormalizer:
    """Restore the exact selected-training statistics saved in a checkpoint."""

    def __init__(self, dataset_stats: list[dict[str, Any]], method: str = "bounds_q99",
                 min_range: float = 0.0):
        if not dataset_stats:
            raise ValueError("Checkpoint has no saved training statistics")
        self.normalizer = ActionNormalizer(min_range=min_range)
        self.method = method
        self.repo_to_idx = {}
        for record in dataset_stats:
            idx = int(record["dataset_idx"])
            self.normalizer.add_dataset_stats(
                idx, record["repo_id"], record["stats"], record["state_key"],
                record["action_key"], record.get("single_arm", False),
            )
            self.repo_to_idx[record["repo_id"]] = idx

    def resolve_idx(self, env_id: str) -> int:
        if env_id in self.repo_to_idx:
            return self.repo_to_idx[env_id]
        matching = [idx for repo, idx in self.repo_to_idx.items()
                    if _base_env_name(repo) == _base_env_name(env_id)]
        if len(matching) == 1:
            return matching[0]
        raise ValueError(f"No unambiguous checkpoint normalizer for {env_id}")

    def normalize_state(self, state: torch.Tensor, dataset_idx: int = 0) -> torch.Tensor:
        return self.normalizer.normalize_state(state, dataset_idx, self.method)

    def denormalize_action(self, action: torch.Tensor, dataset_idx: int = 0) -> torch.Tensor:
        return self.normalizer.denormalize_action(action, dataset_idx, self.method)


if __name__ == "__main__":
    from types import SimpleNamespace

    selected = SimpleNamespace(hf_dataset={"qpos": [[0., 1.], [2., 3.]],
                                          "action": [[1.], [3.]]})
    stats = selected_row_statistics(selected, ["qpos", "action"])
    saved = [{"dataset_idx": 0, "repo_id": "L0_DualCubePlace-v1", "stats": stats,
              "state_key": "qpos", "action_key": "action"}]
    normalizer = NativeSingleTaskNormalizer(saved, method="bounds")
    assert normalizer.resolve_idx("DualCubePlace-v1") == 0
    state = normalizer.normalize_state(torch.tensor([[1., 2.]]))
    action = normalizer.denormalize_action(torch.tensor([[0.]]))
    torch.testing.assert_close(state, torch.zeros(1, 2))
    torch.testing.assert_close(action, torch.tensor([[2.]]))
    print("native dataset sanity:", tuple(state.shape), tuple(action.shape))
