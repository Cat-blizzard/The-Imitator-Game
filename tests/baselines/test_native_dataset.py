"""Native training uses selected episodes, saved normalizers and episode-local history."""
from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch
from imitator.dataset import lerobot_sim_dataset as sim
from imitator.dataset.native import NativeSingleTaskNormalizer as SingleTaskNormalizer
from examples.baselines.act.train_act_single import ACTSingleTaskAgent, TrainingArgs

STATE = "observation.qpos_gripper_states"
ACTION = "action.qpos_gripper_actions"

class ColumnTable:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def __getitem__(self, key):
        return [row[key] for row in self.rows] if isinstance(key, str) else self.rows[key]


class FakeDataset:
    def __init__(self, repo_id, root, episodes=None, delta_timestamps=None, **kwargs):
        self.repo_id, self.root, self.episodes = repo_id, root, episodes
        self.meta = SimpleNamespace(features={STATE: {"shape": [18], "dtype": "float32"}, ACTION: {"shape": [16], "dtype": "float32"},
                                             "observation.images.zed2i": {"dtype": "video"}},
                                    info={"splits": {"train": [0], "validation": [1]}},
                                    stats={STATE: {"min": [-999] * 18}})
        self.features = self.meta.features
        self.horizon = len((delta_timestamps or {ACTION: [0]})[ACTION])
        values = [(0, 1.), (0, 3.), (1, 100.)]
        self.hf_dataset = ColumnTable([
            {"episode_index": torch.tensor(ep), STATE: torch.full((18,), value),
             ACTION: torch.full((16,), value)}
            for ep, value in values if episodes is None or ep in episodes
        ])

    def __len__(self):
        return len(self.hf_dataset.rows)

    def __getitem__(self, index):
        row = self.hf_dataset[index]
        result = dict(row, task="pill", **{"observation.images.zed2i": torch.full((3, 224, 224), 128 / 255)})
        result[ACTION] = row[ACTION].repeat(self.horizon, 1)
        result[f"{ACTION}_is_pad"] = torch.arange(self.horizon) + index >= len(self)
        return result


@pytest.mark.parametrize("min_range", [0.0, 4.0])
def test_selected_train_statistics_validation_reuse_and_tail_padding(tmp_path, monkeypatch, min_range):
    monkeypatch.setattr(sim, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(sim, "LeRobotDatasetMetadata", lambda **kw: FakeDataset(**kw).meta)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps([{"repo_id": "pill", "root": "pill"}]))
    config = sim.LeRobotSimDataConfig(root=str(tmp_path), dataset_file=str(config_file),
                                     cameras=["zed2i"], horizon=3, enable_augmentation=False,
                                     fps=20, obs_horizon=2, normalization_min_range=min_range,
                                     normalization_stats_source="train_rows")
    train = sim.LeRobotSimDataset(config)
    assert train.normalizer.dataset_stats[0][STATE]["max"] == [3.] * 18
    assert train[1]["action_is_pad"].tolist() == [False, True, True]
    # The first observation of a new episode repeats its own initial state.
    both_config = tmp_path / "both.json"
    both_config.write_text(json.dumps([{"repo_id": "pill", "root": "pill", "train": [0, 1]}]))
    both = sim.LeRobotSimDataset(replace(config, dataset_file=str(both_config)))
    assert torch.equal(both[2]["states"][0], both[2]["states"][1])
    validation = sim.LeRobotSimDataset(replace(config, split="validation"))
    assert validation.normalizer.dataset_stats[0][STATE]["max"] == [3.] * 18
    saved = [{"dataset_idx": 0, **train.normalizer.dataset_info[0],
              "stats": train.normalizer.dataset_stats[0]}]
    heldout = sim.LeRobotSimDataset(replace(config, split="validation", normalization_stats=saved))
    normalizer = SingleTaskNormalizer(saved, min_range=min_range)
    assert train.normalizer.min_range == min_range
    assert torch.equal(heldout[0]["states"][-1], normalizer.normalize_state(torch.full((18,), 100.)))
    assert heldout[0]["view_1"].shape == (2, 3, 224, 224)



def test_metadata_normalization_remains_the_legacy_default(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(sim, "LeRobotDatasetMetadata", lambda **kw: FakeDataset(**kw).meta)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps([{"repo_id": "pill", "root": "pill"}]))
    cfg = sim.LeRobotSimDataConfig(root=str(tmp_path), dataset_file=str(config_file),
                                   cameras=["zed2i"], enable_augmentation=False)
    ds = sim.LeRobotSimDataset(cfg)
    assert ds.normalizer.dataset_stats[0][STATE]["min"] == [-999] * 18


@pytest.mark.parametrize("state_dim,action_dim", [(18,16), (16,14), (24,14)])
def test_native_padding_does_not_change_posterior_or_loss(state_dim, action_dim):
    torch.manual_seed(7)
    args = TrainingArgs(cuda=False, obs_encoder_type="simple_cnn", image_size=[32,32],
                        state_dim=state_dim, action_dim=action_dim, pred_horizon=4,
                        hidden_dim=32, enc_layers=1, dec_layers=1, dim_feedforward=64, dropout=0.)
    agent = ACTSingleTaskAgent(args, torch.device("cpu")).eval()
    batch = {"view_1": torch.rand(1,1,3,32,32), "states": torch.rand(1,1,state_dim),
             "actions": torch.rand(1,4,action_dim),
             "action_is_pad": torch.tensor([[False,False,True,True]])}
    torch.manual_seed(19)
    first = agent.compute_loss(batch)
    changed = dict(batch, actions=batch["actions"].clone())
    changed["actions"][:,2:] += 1000
    torch.manual_seed(19)
    second = agent.compute_loss(changed)
    torch.testing.assert_close(first, second, rtol=0, atol=1e-6)
    first.backward()
    assert agent.obs_proj.weight.grad is not None
    assert torch.isfinite(agent.obs_proj.weight.grad).all()
    assert agent.get_action(batch["view_1"], batch["states"]).shape == (1,4,action_dim)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
