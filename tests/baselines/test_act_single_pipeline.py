"""CPU checks for the actual single-task ACT data and model boundary."""
from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from examples.baselines.act.single_task_io import masked_l1, prepare_rgb, split_action
from examples.baselines.act.train_act_single import ACTSingleTaskAgent, TrainingArgs
from examples.baselines.act.eval_act_single import SingleTaskNormalizer, evaluation_seeds
from examples.baselines.lerobot_dataset import lerobot_sim_dataset as sim

STATE = "observation.qpos_gripper_states"
ACTION = "action.qpos_gripper_actions"


def test_rgb_and_action_order_and_padding_gradient():
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    assert torch.equal(prepare_rgb(image), prepare_rgb(torch.from_numpy(image).permute(2, 0, 1).float() / 255))
    import cv2
    large = np.arange(256*256*3, dtype=np.uint8).reshape(256, 256, 3)
    assert torch.equal(prepare_rgb(large), prepare_rgb(cv2.resize(large, (224, 224))))
    blocks = split_action(torch.arange(16), ["left", "right"])
    np.testing.assert_array_equal(blocks["left"], np.arange(8))
    np.testing.assert_array_equal(blocks["right"], np.arange(8, 16))
    predicted = torch.zeros(1, 3, 16, requires_grad=True)
    loss = masked_l1(predicted, torch.ones_like(predicted), torch.tensor([[False, True, True]]))
    loss.backward()
    assert loss.item() == 1
    assert torch.count_nonzero(predicted.grad[:, 1:]) == 0
    with pytest.raises(ValueError, match="real action"):
        masked_l1(predicted, torch.ones_like(predicted), torch.ones(1, 3, dtype=torch.bool))


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


def test_selected_train_statistics_validation_reuse_and_tail_padding(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(sim, "LeRobotDatasetMetadata", lambda **kw: FakeDataset(**kw).meta)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps([{"repo_id": "pill", "root": "pill"}]))
    config = sim.LeRobotSimDataConfig(root=str(tmp_path), dataset_file=str(config_file),
                                     cameras=["zed2i"], horizon=3, enable_augmentation=False,
                                     fps=20, obs_horizon=2)
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
    normalizer = SingleTaskNormalizer(saved)
    assert torch.equal(heldout[0]["states"][-1], normalizer.normalize_state(torch.full((18,), 100.)))
    assert heldout[0]["view_1"].shape == (2, 3, 224, 224)


def test_explicit_seed_selection_never_filters_by_expert_result(tmp_path):
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps({"seed_splits": {"train": [1], "development": [2], "evaluation": [30, 31]}}))
    args = SimpleNamespace(seeds=None, seeds_file=str(path), seed_split="evaluation", seed=0, num_episodes=20)
    assert evaluation_seeds(args) == [30, 31]
    args.seeds_file, args.seeds = None, [7, 7]
    with pytest.raises(ValueError, match="unique"):
        evaluation_seeds(args)


@pytest.mark.parametrize("state_dim,action_dim", [(18, 16), (16, 14)])
def test_full_default_model_wiring_backward_and_strict_reload(state_dim, action_dim):
    torch.manual_seed(7)
    args = TrainingArgs(dropout=0.0, state_dim=state_dim, action_dim=action_dim)
    agent = ACTSingleTaskAgent(args, torch.device("cpu"))
    batch = {"view_1": torch.rand(1, 1, 3, 224, 224), "states": torch.rand(1, 1, state_dim),
             "actions": torch.rand(1, 16, action_dim), "action_is_pad": torch.tensor([[False] * 5 + [True] * 11])}
    # Fixed RNG makes posterior samples comparable; padded labels cannot change its posterior.
    agent.eval()
    torch.manual_seed(19)
    first = agent.compute_loss(batch)
    changed = dict(batch, actions=batch["actions"].clone())
    changed["actions"][:, 5:] += 1000
    torch.manual_seed(19)
    second = agent.compute_loss(changed)
    assert torch.allclose(first, second, atol=1e-6)
    first.backward()
    assert agent.obs_proj.weight.grad is not None
    assert torch.isfinite(agent.obs_proj.weight.grad).all()
    predicted = agent.get_action(batch["view_1"], batch["states"])
    assert predicted.shape == (1, 16, action_dim)
    cloned = ACTSingleTaskAgent(args, torch.device("cpu"))
    cloned.load_state_dict(agent.state_dict(), strict=True)
    cloned.eval()
    assert torch.equal(predicted, cloned.get_action(batch["view_1"], batch["states"]))


def test_piper_native_action_blocks_and_live_qpos_order():
    from examples.baselines.act.single_task_io import live_observation
    blocks = split_action(torch.arange(14), ['piper-0', 'piper-1'], [7, 7])
    np.testing.assert_array_equal(blocks['piper-1'], np.arange(7, 14))
    names = [f'joint{i}' for i in range(1, 9)]
    def robot(offset):
        return SimpleNamespace(get_qpos=lambda: torch.arange(8)+offset,
                               get_active_joints=lambda: [SimpleNamespace(name=name) for name in names])
    agents = [SimpleNamespace(robot=robot(0)), SimpleNamespace(robot=robot(10))]
    raw = SimpleNamespace(agent=SimpleNamespace(agents=agents, agents_dict={'piper-0': agents[0], 'piper-1': agents[1]}))
    identity = SimpleNamespace(normalize_state=lambda value, _: value)
    order = [{'key': f'piper-{side}', 'state_joint_names': names} for side in range(2)]
    sample = live_observation(raw, {'sensor_data': {'base_camera': {'rgb': np.zeros((1, 224, 224, 3), np.uint8)}}},
                              'base_camera', identity, action_order=order)
    assert sample['state'].tolist() == list(range(8))+list(range(10, 18))
    with pytest.raises(ValueError, match='controller keys'):
        live_observation(raw, {}, 'base_camera', identity, action_order=list(reversed(order)))
